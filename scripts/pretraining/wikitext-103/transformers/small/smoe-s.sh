for i in {9..9}
  do
    echo "Training ..."
    eps=$(perl -e "print $i / 10")
    if [ $eps == 0.9 ]
    then 
    eps=$(perl -e "print $eps + 0.09")
    fi
    mkdir -p /root/checkpoints/moe/wikitext-103/transformers-s/smoev5_0${i}_01
    args="
    --data /root/data/wikitext-103/ \
    --base_arch transformer \
    --architecture smsmsm \
    --gate_name smoe \
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
    --niter 40 \
    --batch-sz 96 \
    --batch-split 2 \
    --nbatches 1000 \
    --distributed \
    --checkpoint /root/checkpoints/moe/wikitext-103/transformers-s/smoev5_0${i}_01/smoe.pt \
    --gamma 0.1 \
    --mu $eps \
    --wandb \
    --project-name moe_momentum \
    --job-name momv5_0_${i}_0_1 \
    "
    CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10010 --nproc_per_node=4 --use_env /root/repos/moe_opt/moe/train.py $args
done

# echo "Evaluation ..."
# python train.py $args --full-eval-mode --batch-sz 8