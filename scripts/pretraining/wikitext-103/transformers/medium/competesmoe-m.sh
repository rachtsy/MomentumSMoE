mkdir -p checkpoints/wikitext-103/transformers-m/competesmoe

args="
--data /home/gtruong/Project/ICML2/data/wikitext-103 \
--base_arch transformer \
--architecture sgsgsgsgsgsg \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 1024 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 50 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--optimal_policy \
--act_experts linear \
--opt_blance \
--opt_loss mse \
--freq_type fix \
--freq 1.0 \
--alpha 5.0 \
--checkpoint checkpoints/wikitext-103/transformers-m/competesmoe/competesmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8