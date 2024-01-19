import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # self.norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_head=8, window=7):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim//2)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_conv = nn.Conv2d(dim//2, dim//2, 7, padding=3, groups=dim//2)
        self.e_fore = nn.Linear(dim//2, dim//2)
        self.e_back = nn.Linear(dim//2, dim//2)
        
        

        self.proj = nn.Linear(dim//2*3, dim)
        self.proj_e = nn.Linear(dim//2*3, dim//2)
        if window != 0:
            self.short_cut_linear = nn.Linear(dim//2*3, dim//2)#####
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7,7))
            self.kv = nn.Linear(dim, dim)
            # self.m = nn.Parameter(torch.zeros(1, window, window, dim // 2), requires_grad=True)
            self.proj = nn.Linear(dim * 2, dim)
            self.proj_e = nn.Linear(dim*2, dim//2)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim//2, eps=1e-6, data_format="channels_last")

    def forward(self, x,x_e):
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)
        if self.window != 0:
            short_cut = torch.cat([x,x_e],dim=3)##########
            short_cut = short_cut.permute(0,3,1,2)#############

        q = self.q(x)   
        cutted_x = self.q_cut(x)     
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)
            
        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)

        if self.window != 0:
            b = x.permute(0, 2, 3, 1)
            kv = self.kv(b)
            kv = kv.reshape(B, H*W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            ####
            short_cut = self.pool(short_cut).permute(0,2,3,1)
            short_cut = self.short_cut_linear(short_cut)#(B,7,7,3DIM//2)->(B,7,7,DIM//2)
            ####
            short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
            # m = self.m.reshape(1, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
            # print(m.shape,short_cut.shape)
            m = short_cut
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1) 
            attn = attn.softmax(dim=-1)
            attn = (attn @ v).reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2).permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.window, self.window)
            attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        x_e = self.e_back(self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        cutted_x = cutted_x * x_e
        x = q * a

        if self.window != 0:
            x = torch.cat([x, attn,cutted_x], dim=3)
        else:
            x = torch.cat([x,cutted_x], dim=3)
        
        x_e = self.proj_e(x)
        x = self.proj(x)

        return x,x_e



class Block(nn.Module):
    def __init__(self, index, dim, num_head, window=7, mlp_ratio=4., drop_path=0.,block_index=0, last_block_index=50):
        super().__init__()
        
        self.index = index
        layer_scale_init_value = 1e-6 
        if block_index>last_block_index:
                window=0 
        self.attn = Attention(dim, num_head, window=window)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = MLP(dim, mlp_ratio)
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_1_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
        self.layer_scale_2_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
        # self.mlp_e1 = MLP(dim//2, mlp_ratio)
        self.mlp_e2 = MLP(dim//2, mlp_ratio)

    def forward(self, x, x_e):
        res_x,res_e=x,x_e
        x,x_e=self.attn(x,x_e)

        x_e = res_e + self.drop_path(self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e)
        x = res_x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x )

        x_e = x_e + self.drop_path(self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))
        return x,x_e

class DFormer_model(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], windows=[7, 7, 7, 7],
                 mlp_ratios=[4, 4, 4, 4],last_block=[50,50,50,50], num_heads=[2, 4, 10, 16], layer_scale_init_value=1e-6, head_init_scale=1., 
                 drop_path_rate=0., drop_rate=0., **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
                nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
        )

        self.downsample_layers_e = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem_e = nn.Sequential(
                nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 4),
                nn.GELU(),
                nn.Conv2d(dims[0] // 4, dims[0]//2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]//2),
        )

        self.downsample_layers.append(stem)
        self.downsample_layers_e.append(stem_e)

        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                    nn.BatchNorm2d(dims[i]//2),
                    nn.Conv2d(dims[i]//2, dims[i+1]//2, kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur+j, dim=dims[i], window=windows[i], drop_path=dp_rates[cur + j], num_head=num_heads[i], mlp_ratio=mlp_ratios[i],block_index=depths[i]-j,last_block_index=last_block[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.pred = nn.Linear(dims[-1]//2*3, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x,x_e = x[:,:3,:,:],x[:,3,:,:]
        x_e = x_e.unsqueeze(1)
        assert len(x_e.shape)==4
        for i in range(4):
            # print(x.shape)
            # print(x.shape,i,'before downsample_layers')
            # try:
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)
            # except:
            #     print(i)
            #     print(self.downsample_layers[i])
            #     print(x.shape)
            #     print(x_e.shape)
            x = x.permute(0, 2, 3, 1)
            x_e = x_e.permute(0, 2, 3, 1)
            # print(x.shape,i,'before stages')
            for blk in self.stages[i]:
                x,x_e = blk(x,x_e)
            x = x.permute(0, 3, 1, 2)
            x_e = x_e.permute(0, 3, 1, 2)
        # x = self.norm(x)
        x = torch.cat([x,x_e],dim=1) 
        return x.mean([-2, -1]) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pred(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def load_model_weights(model, arch, kwargs):
    checkpoint = torch.load(model_urls[arch], map_location="cpu")
    strict = True
    if "num_classes" in kwargs and kwargs["num_classes"] != 1000:
        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return model

@register_model
def DFormer_Tiny(pretrained=False, **kwargs):
    model = DFormer_model(dims=[32, 64, 128, 256], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, 'dformer', kwargs)
    return model

@register_model
def DFormer_Small(pretrained=False, **kwargs):
    model = DFormer_model(dims=[64, 128, 256, 512], mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, 'dformer', kwargs)
    return model

@register_model
def DFormer_Base(pretrained=False, **kwargs):
    model = DFormer_model(dims=[64, 128, 256, 512], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, 'dformer', kwargs)
    return model

@register_model
def DFormer_Large(pretrained=False, **kwargs):
    model = DFormer_model(dims=[96, 192, 288, 576], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, 'dformer', kwargs)
    return model