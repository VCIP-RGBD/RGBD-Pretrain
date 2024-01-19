MODEL=B02_vit_base
#B01_vit_base # van_{tiny, small, base, large}
# Dpeth_VIT_baseline_Base

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=31929 bash distributed_train.sh 8 \
		/root/siton-gpfs-pubdata/imagenet/ \
	  	--model $MODEL \
		-b 128 \
		--epochs 300 \
		--opt adamw \
		-j 16 \
		--input-size 4 224 224 \
		--warmup-lr 1e-6 \
		--warmup-epochs 5 \
		--weight-decay 0.05 \
		--amp \
		--drop-path 0.1 \
		--lr 0.001 \
		--aa rand-m9-mstd0.5-inc1 \
		--remode pixel \
		--reprob 0.25 \
		--cutmix 1.0 \
		--mixup 0.8 \
		# --resume /root/siton-gpfs-archive/bowenyin/pretrain/RGBD/version02_gai_rgbd_cat/outputs/20231227-170758-Dpeth_VIT_baseline_Base-224/last.pth.tar
		# --model-ema \
		# --resume /root/siton-gpfs-archive/bowenyin/pretrain/RGBD/version02_gai_rgbd_cat/outputs/20231227-170758-Dpeth_VIT_baseline_Base-224/last.pth.tar