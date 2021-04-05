conda=/home/hsj/anaconda3
conda_env=py36
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

root=/bigdata/hsj/2021

#/home/hsj/2021/data/0222_All_data_for_fine_tune_train
mc_dir=/home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk



exp_root=$root/one-shot/github开源-5
main_script=$exp_root/main.py


CUDA_VISIBLE_DEVICES=1 $PYTHON $main_script  \
                    --train_data_dir /home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk \
                    --scp_path /home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk/feats.scp \
                    --speaker_path /home/hsj/2021/data/vctk-80p-all-norm/80_train_speaker_used.json \
                    --model_save_dir ${exp_root}/model/ \
                    --sample_step 10000 \
                    --model_save_step 1000 \
                    --log_step 10 \
                    --log_dir ${exp_root}/logs/ \
                    --sample_dir ${exp_root}/samples/ \
                    --num_workers 0 \
                    --batch_size 8 \
                    --min_length 256 \
                    --d_lr 1e-4 \
                    --g_lr 2e-4 \
                    --num_iters 100000 \
                    --drop_id_step 100000 \
                    --lambda_id 2 \
                    --lambda_rec 4 \
                    --lambda_adv 1 \
                    --lambda_spid 2 \
                    --num_heads 8 \
                    --kernel 3 \
                    --num_res_block 9 \
                    --g_hidden_size 256 \
                    --generator GeneratorPlain \
                    --discriminator PatchDiscriminator1 \
                    --res_block Style2ResidualBlock1DBeta \
                    --spenc SPEncoder \
                    --use_sp_enc \
                    --use_ema \





