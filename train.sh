MODEL=DFormer_Small # drop-path 0.1, 0.1, 0.15, 0.2 for T,S,B,L

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=31929 bash distributed_train.sh 8 \
		datasets/ImageNet/ \
	  	--model $MODEL \
		-b 128 \
		--epochs 300 \
		--opt adamw \
		-j 16 \
		--input-size 4 224 224 \
		--warmup-lr 1e-6 \
		--warmup-epochs 5 \
		--weight-decay 0.05 \
		--grad-accum-steps 1 \
		--amp \
		--clip-grad 1.0 \
		--drop-path 0.45 \
		--lr 0.001 \
		--aa rand-m9-mstd0.5-inc1 \
		--remode pixel \
		--reprob 0.25 \
		--cutmix 1.0 \
		--mixup 0.8 \
  		# --resume outputs/XXXXX/last.pth.tar




