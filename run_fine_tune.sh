conda=/home/hsj/anaconda3
conda_env=py36
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

root=/home/hsj/2021

mc_dir=$root/data/0222_All_data_for_fine_tune_train/larger_than_128frame_norm/train80p_fine-tune10p


exp_root=$root/few_shot/Fine_tune/task74
main_script=$exp_root/main.py

train_count=all-norm
fine_tune_count=1norm

CUDA_VISIBLE_DEVICES=1 $PYTHON $main_script  \
                    --model_save_dir ${exp_root}/fine-tune-model/ \
                    --sample_step 10000 \
                    --model_save_step 1000 \
                    --log_step 10 \
                    --log_dir ${exp_root}/logs-fine-tune/ \
                    --train_data_dir $mc_dir/train/${train_count} \
                    --scp_path $mc_dir/train/${train_count}/feats.scp \
                    --speaker_path $mc_dir/train/${train_count}/80_train_speaker_used.json \
                    --sample_dir ${exp_root}/samples/ \
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
                    --generator GeneratorPlain \
                    --discriminator PatchDiscriminator1 \
                    --res_block Style2ResidualBlock1DBeta \
                    --spenc SPEncoder \
                    --use_sp_enc \
                    --use_ema \
                    --num_ft_speakers 10 \
                    --ft_data_dir $mc_dir/fine_tune/fine_tune_10p_1norm \
                    --ft_scp $mc_dir/fine_tune/fine_tune_10p_1norm/feats.scp \
                    --ft_speaker $mc_dir/fine_tune/fine_tune_10p_1norm/10p_fine-tune_speaker_used.json \
                    --num_few_shot 1 \
                    --resume_iters 100000\
                    --num_iters 105000 \
                    
#                    --test_data_dir $mc_dir/test/test_10_mel \
#                    --kconv \
#                    --use_r1reg \





