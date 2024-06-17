for i in 0 1 2 3 4 5 
  do
    echo "Eval ..."
    mkdir -p /root/checkpoints/smoe/baseline/
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
    --niter 25 \
    --batch-sz 24 \
    --batch-split 2 \
    --nbatches 1000 \
    --distributed \
    --mu 0.2 \
    --gamma 1.25 \
    --layer-n $i \
    --checkpoint /root/checkpoints/smoe/baseline/smoe.pt \
    --full-eval-mode \
    --resume \
    "
    CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10233 --nproc_per_node=4 --use_env /root/repos/moe_opt/moe_expert_count/train.py $args
done