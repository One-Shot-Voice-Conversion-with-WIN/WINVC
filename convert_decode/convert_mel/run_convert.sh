conda=/home/hsj/anaconda3
conda_env=py36
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

root=/home/hsj/2021

num_spks=30
model_name=wadain_stargan_vc



#/home/hsj/2021/data/0222_All_data_for_fine_tune_train
mc_dir=$root/data/0222_All_data_for_fine_tune_train


exp_root=$root/few_shot/Fine_tune/task22
main_script=$exp_root/convert_vctk.py


iters=150000
generator_model=GeneratorFlat
resblock=Style2ResidualBlock1DBeta
speaker_encoder=SPEncoder

scp_path=$exp_root/feats.scp
speaker_path=$mc_dir/train/speaker_used.json
ft_speakers_path=$mc_dir/fine_tune/fine_tune_speaker_used.json
dump_dir=$mc_dir/test/test_10_mel
ft_dump_dir=$mc_dir/fine_tune/1norm
model_save_dir=$exp_root/model

#src=p226
#trg=p

CUDA_VISIBLE_DEVICES=-1 $PYTHON $main_script  \
                    --dump_dir $dump_dir \
                    --model_save_dir $model_save_dir \
                    --resume_iters $iters \
                    --convert_dir $exp_root/converted_mel_$iters \
                    --num_speakers $num_spks \
                    --generator $generator_model \
                    --res_block $resblock \
                    --speaker_encoder $speaker_encoder \
                    --sample_rate 16000 \
                    --speaker_path $speaker_path \
                    --num_converted_wavs 4 \
                    --scp_path $scp_path \
                    --g_hidden_size 256 \
                    --num_heads 8 \
                    --kernel 3 \
                    --num_res_block 9 \
                    --use_sp_enc \
                    --use_ema \
#                    --src_spk p226 \
#                    --trg_spk p227 \
#                    --ft_dump_dir $ft_dump_dir \
#                    --num_ft_speakers 10 \
#                    --ft_speakers_path $ft_speakers_path \
#                    --src_spk $src \
#                    --trg_spk $trg \

cat $scp_path
converted_wav_dir=$exp_root/convertes_wav_$iters
bash $root/few_shot/convert_decode/pgan_decode/vctk_pgan.sh $scp_path $converted_wav_dir





