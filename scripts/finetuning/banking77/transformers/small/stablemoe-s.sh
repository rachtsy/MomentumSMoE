mkdir -p checkpoints/banking77/transformers-s/stablemoe

args="
--data data/finetuning/banking77 \
--data_name banking77 \
--base_arch transformer \
--architecture sgsgsgsgsgsg \
--gate_name stablemoe \
--nlayers 6 \
--hid-sz 264 \
--inner-hid-sz 264 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.00001 \
--lr-warmup 0 \
--niter 50 \
--batch-sz 16 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/banking77/transformers-s/stablemoe/stablemoe.pt \
--pretrained_weight checkpoints/enwik8/transformers-s/stablemoe_old/stablemoe.pt \
"

echo "Training ..."
python finetune_train.py $args

