mkdir -p checkpoints/enwik8/glam-m/competesmoe

args="
--data /home/gtruong/Project/ICML2/data/enwik8 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsf \
--gate_name smoe \
--nlayers 12 \
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
--act_experts linear \
--combine_gate \
--opt_loss mse \
--freq_type fix \
--freq 1.0 \
--alpha 5.0 \
--checkpoint checkpoints/enwik8/glam-m/competesmoe/competesmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8