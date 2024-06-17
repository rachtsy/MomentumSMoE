mkdir -p checkpoints/text8/glam-l/competesmoe

args="
--data /home/gtruong/Project/ICML2/data/text8 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsf \
--gate_name smoe \
--nlayers 18 \
--hid-sz 528 \
--inner-hid-sz 528 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 6000 \
--niter 120 \
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
--checkpoint checkpoints/text8/glam-l/competesmoe/competesmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8