conda=/home/hsj/anaconda3
conda_env=py36
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

root=/bigdata/hsj/2021

mc_dir=/home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk

# path to this project
exp_root=/bigdata/hsj/2021/one-shot/WadaIN-VC
main_script=$exp_root/main.py

# path to your mels of training
train_data_dir=/home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk
scp_path=/home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk/feats.scp
speaker_path=/home/hsj/2021/data/vctk-80p-all-norm/80_train_speaker_used.json

# path to save model and log
model_save_dir=${exp_root}/model/
log_dir=${exp_root}/logs/

# training for 10w iters
num_iters=100000

# model modules' parameters
generator=GeneratorPlain
discriminator=PatchDiscriminator1
res_block=Style2ResidualBlock1DBeta

CUDA_VISIBLE_DEVICES=0 $PYTHON $main_script  \
                    --train_data_dir $train_data_dir \
                    --scp_path $scp_path \
                    --speaker_path $speaker_path \
                    --model_save_dir $model_save_dir \
                    --model_save_step 1000 \
                    --log_step 10 \
                    --log_dir $log_dir \
                    --num_workers 0 \
                    --batch_size 8 \
                    --min_length 256 \
                    --d_lr 1e-4 \
                    --g_lr 2e-4 \
                    --num_iters $num_iters \
                    --drop_id_step 100000 \
                    --lambda_id 2 \
                    --lambda_rec 4 \
                    --lambda_adv 1 \
                    --lambda_spid 2 \
                    --num_heads 8 \
                    --kernel 3 \
                    --num_res_block 9 \
                    --g_hidden_size 256 \
                    --generator $generator \
                    --discriminator $discriminator \
                    --res_block $res_block \
                    --spenc SPEncoder \
                    --use_sp_enc \
                    --use_ema \





