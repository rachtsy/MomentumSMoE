mkdir -p checkpoints/wikitext-103/transformers-s/stablemoe

args="
--data /home/gtruong/Project/ICML2/data/wikitext-103 \
--base_arch transformer \
--architecture sgsgsg \
--gate_name stablemoe \
--nlayers 3 \
--hid-sz 128 \
--inner-hid-sz 128 \
--nheads 8 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 96 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/wikitext-103/transformers-s/stablemoe/stablemoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8