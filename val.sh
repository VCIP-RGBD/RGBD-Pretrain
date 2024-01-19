CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python validate.py \
        /root/siton-gpfs-pubdata/imagenet/ \
        --model RGB_VIT_baseline_Base \
        --checkpoint /root/siton-gpfs-archive/bowenyin/pretrain/RGB/SegNextV2-main/outputs/20231219-204959-vit_base_patch16_224-224/model_best.pth.tar