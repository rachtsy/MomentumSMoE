mkdir -p checkpoints/enwik8/transformers-m/competesmoe

args="
--data /home/gtruong/Project/ICML2/data/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsg \
--gate_name smoe \
--nlayers 8 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--optimal_policy \
--act_experts shuffle \
--opt_loss mse \
--freq_type fix \
--freq 0.03 \
--alpha 5.0 \
--checkpoint checkpoints/enwik8/transformers-m/competesmoe/competesmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8