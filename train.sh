MODEL=DFormer_Small # drop-path 0.1, 0.1, 0.15, 0.2 for T,S,B,L

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
		--mixup 0.8 