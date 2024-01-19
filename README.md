# <p align=center>`RGBD-pretraining in DFormer`</p>

> **Authors:**
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Xuying Zhang](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=zh-CN),
> [Li Liu](https://scholar.google.com/citations?hl=zh-CN&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN)


This repository provides the RGBD pretraining code of '[ICLR 2024] DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation'.
Our implementation is modified from the [timm](https://github.com/huggingface/pytorch-image-models) repository.
If there are any questions, please let me know via [raising issues](https://github.com/VCIP-RGBD/DFormer/issues) or e-mail (bowenyin@mail.nankai.edu.cn).

## 1. 🚀 Get Start

**1.1. Install**


```
XXXXXXXXXXXXXXXXXXX
```

If the above pipeline not work, You can also install following the [timm](https://github.com/huggingface/pytorch-image-models).


**1.2. Prepare Datasets**

First, you need to prepare the ImageNet-1k dataset.
We share the depth maps for the ImageNet-1K in the following links:


| [Baidu Netdisk]() | [OneDrive]() |
|  ----  | ----  |


If the share links have any questions, please let me know (bowenyin@mail.nankai.edu.cn). 








## 2. 🚀 Train.

```
bash train.sh
```

After training, the checkpoints will be saved in the path `outputs/XXX', where the XXX is depends on the training config.

>Then, the pretrained checkpoint is endowed with the capacity to encode the RGBD represetions and can be applied to various RGBD tasks. 



> We invite all to contribute in making this project and RGBD representation learning more acessible and useful. If you have any questions or suggestions about our work, feel free to contact me via e-mail (bowenyin@mail.nankai.edu.cn) or raise an issue. 


## Reference
```
@article{yin2023dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhongyu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  journal={arXiv preprint arXiv:2309.09668},
  year={2023}
}
```


### Acknowledgment

Our implementation is mainly based on [timm](https://github.com/huggingface/pytorch-image-models). Thanks for their authors.

### License

Code in this repo is for non-commercial use only.