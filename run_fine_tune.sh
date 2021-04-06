conda=/home/hsj/anaconda3
conda_env=py36
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python


root=/home/hsj/2021

# path to this project
exp_root=/bigdata/hsj/2021/one-shot/WadaIN-VC
main_script=$exp_root/main.py

# path to save model and log
model_save_dir=${exp_root}/model
log_dir=${exp_root}/logs/ft-logs/

# path to your mels of training
train_data_dir=/home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk
scp_path=/home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk/feats.scp
speaker_path=/home/hsj/2021/data/vctk-80p-all-norm/80_train_speaker_used.json

# path to your mels of fine tuning
mc_dir=$root/data/0222_All_data_for_fine_tune_train/larger_than_128frame_norm/train80p_fine-tune10p

ft_data_dir=$mc_dir/fine-tune-5M_5F/fine_tune_10p_1norm \
ft_scp=$mc_dir/fine-tune-5M_5F/fine_tune_10p_1norm/feats.scp \
ft_speaker=$mc_dir/fine-tune-5M_5F/fine_tune_10p_1norm/10p_fine-tune_speaker_used.json \


# model modules' parameters
generator=GeneratorPlain
discriminator=PatchDiscriminator1
res_block=Style2ResidualBlock1DBeta
spenc=SPEncoder

# resume from 10w iters, and fine tune for 5k iters
resume_iters=100000
num_iters=105000

# number of speakers for fine tuning
# and number of each unseen speaker used for one-shot, we use 1 utterance per unseen persom
num_ft_speakers=10
num_few_shot=1


CUDA_VISIBLE_DEVICES=0 $PYTHON $main_script  \
                    --model_save_dir $model_save_dir \
                    --model_save_step 1000 \
                    --log_step 10 \
                    --log_dir $log_dir \
                    --train_data_dir $train_data_dir \
                    --scp_path $scp_path \
                    --speaker_path $speaker_path \
                    --num_workers 0 \
                    --batch_size 8 \
                    --min_length 256 \
                    --d_lr 1e-4 \
                    --g_lr 2e-4 \
                    --drop_id_step 100000 \
                    --lambda_id 2 \
                    --lambda_rec 4 \
                    --lambda_adv 1 \
                    --lambda_spid 5 \
                    --kernel 3 \
                    --num_res_block 9 \
                    --g_hidden_size 256 \
                    --generator $generator \
                    --discriminator $discriminator \
                    --res_block $res_block \
                    --spenc $spenc \
                    --use_sp_enc \
                    --use_ema \
                    --num_ft_speakers $num_ft_speakers \
                    --ft_data_dir $ft_data_dir \
                    --ft_scp $ft_scp \
                    --ft_speaker $ft_speaker \
                    --num_few_shot $num_few_shot \
                    --resume_iters $resume_iters \
                    --num_iters $num_iters \






