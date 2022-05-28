gpu=0
model=punet
extra_tag=punet_baseline


python -u train.py \
    --model ${model} \
    --batch_size 32 \
    --log_dir logs/${extra_tag} \
    --gpu ${gpu} \
    >> logs/${extra_tag}/nohup.log 2>&1 &
