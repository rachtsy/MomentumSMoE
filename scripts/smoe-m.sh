mkdir -p /root/smoe/symmetric/

args="
--data /root/wikitext-103/ \
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
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /root/smoe/symmetric/smoe.pt \
--wandb-flag \
--job-name sym_smoe_resume \
--project-name neurips_momentumSMoE \
--resume 
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env /root/repos/MomentumSMoE/train.py $args

# echo "Evaluation ..."
# CUDA_VISIBLE_DEVICES='3' python -m torch.distributed.launch --master_port 10012 --nproc_per_node=1 --use_env /root/repos/MomentumSMoE/train_eigen.py $args --resume --full-eval-mode