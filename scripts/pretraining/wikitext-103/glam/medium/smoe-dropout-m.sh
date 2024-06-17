mkdir -p checkpoints/wikitext-103/glam-m/smoe_dropout

args="
--data /home/gtruong/Project/ICML2/data/wikitext-103 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsf \
--gate_name smoe \
--nlayers 4 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.00007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--smoe_dropout \
--checkpoint checkpoints/wikitext-103/glam-m/smoe_dropout/smoe-dropout.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8