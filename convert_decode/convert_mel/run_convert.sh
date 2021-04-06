conda=/home/hsj/anaconda3
conda_env=py36
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

root=/home/hsj/2021

# number of speakers used for training
num_spks=80
# number of speakers used for fine tuning
num_ft_speakers=10


# your path to this project
exp_root=/bigdata/hsj/2021/one-shot/WadaIN-VC
main_script=$exp_root/convert_vctk.py

# model modules' parameters
generator_model=GeneratorPlain
resblock=Style2ResidualBlock1DBeta
speaker_encoder=SPEncoder

# fine-tune's output scp
scp_path=$exp_root/fine-tune-feats-${iters}.scp

# mels for training, will not used in fine tuning
speaker_path=/home/hsj/2021/data/vctk-80p-all-norm/80_train_speaker_used.json
dump_dir=/home/hsj/2021/data/vctk-80p-all-norm/train-all-80-vctk

# your path to models
model_save_dir=$exp_root/model

# your path to mels data
mc_dir=$root/data/train80p_fine-tune10p
ft_dump_dir=$mc_dir/fine-tune-5M_5F/fine_tune_test_10p_20norm
ft_speaker_path=$mc_dir/fine-tune-5M_5F/fine_tune_test_10p_20norm/10p_fine-tune_speaker_used.json

# We fine tune only 5,000 iters to perform best one-shot voice conversion
iters=105000



for src in p312  p313  p314  p334  p345  p347  p316  p317  p318  p326
do
    for trg in p318 p312  p313  p314  p334  p345  p347  p316  p317 p326
    do
        if [ $src != $trg ]; then
            CUDA_VISIBLE_DEVICES=-1 $PYTHON $main_script  \
                          --dump_dir $dump_dir \
                          --model_save_dir $model_save_dir \
                          --resume_iters $iters \
                          --convert_dir $exp_root/one-shot_convert/one-shot_wav/convertes_mel_$iters \
                          --num_speakers $num_spks \
                          --generator $generator_model \
                          --res_block $resblock \
                          --speaker_encoder $speaker_encoder \
                          --sample_rate 16000 \
                          --speaker_path $speaker_path \
                          --num_converted_wavs 20 \
                          --scp_path $scp_path \
                          --g_hidden_size 256 \
                          --num_heads 8 \
                          --kernel 3 \
                          --num_res_block 9 \
                          --use_sp_enc \
                          --use_ema \
                          --ft_dump_dir $ft_dump_dir \
                          --num_ft_speakers $num_ft_speakers \
                          --ft_speaker_path $ft_speaker_path \
                          --src_spk $src \
                          --trg_spk $trg \

        fi
    done
done




cat $scp_path
CUDA_VISIBLE_DEVICES=0

# your path to contain one-shot voice conversion utterances
converted_wav_dir=$exp_root/one-shot_convert/one-shot_wav/convertes_wav_$iters

bash /bigdata/hsj/2021/convert_decode/pgan_decode/vctk_pgan.sh $scp_path $converted_wav_dir $CUDA_VISIBLE_DEVICES=1





