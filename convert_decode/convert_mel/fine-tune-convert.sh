conda=/home/hsj/anaconda3
conda_env=py36
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

#root=/bigdata/hsj/2021
root=/home/hsj/2021

num_spks=80
model_name=wadain_stargan_vc



#/home/hsj/2021/data/0222_All_data_for_fine_tune_train
mc_dir=/home/hsj/2021/data/0222_All_data_for_fine_tune_train/larger_than_128frame_norm/train80p_fine-tune10p


#exp_root=/bigdata/hsj/2021/new_fine_tune-73_74/half_lr_new_fine-tune73
exp_root=/bigdata/hsj/2021/new_fine_tune-73_74/task141
main_script=$exp_root/new_convert_vctk.py

train_count=5norm

iters=102000



generator_model=GeneratorPlain
resblock=Style2ResidualBlock1DBeta
#Style2ResidualBlock1DBeta
#DynamicWadain
speaker_encoder=SPEncoder


# fine-tune的转换
scp_path=$exp_root/fine-tune-feats-${iters}.scp



# 这个是 训练80人 的测试集，ft时不会用
speaker_path=$mc_dir/test/test_20norm_80p/80_train_speaker_used.json
dump_dir=$mc_dir/test/test_20norm_80p


# 训练的model
#model_save_dir=$exp_root/model
# fine-tune的model
model_save_dir=$exp_root/fine-tune-model


num_ft_speakers=10
ft_dump_dir=$mc_dir/fine-tune-5M_5F/fine_tune_test_10p_20norm
ft_speaker_path=$mc_dir/fine-tune-5M_5F/fine_tune_test_10p_20norm/10p_fine-tune_speaker_used.json

# 这个是，10人来自训练80人，10人来自ft，全部都是测试集
#num_ft_speakers=20
#ft_dump_dir=$mc_dir/fine-tune-5M_5F/ft_test_20p_20norm_10train_10ft
#ft_speaker_path=$mc_dir/fine-tune-5M_5F/ft_test_20p_20norm_10train_10ft/20p_fine-tune_speaker_used.json



#src=p226
#trg=p

for src in p312  p313  p314  p334  p345  p347  p316  p317  p318  p326  
do
    for trg in p312  p313  p314  p334  p345  p347  p316  p317  p318  p326
    do
        if [ $src != $trg ]; then
            CUDA_VISIBLE_DEVICES=-1 $PYTHON $main_script  \
                          --dump_dir $dump_dir \
                          --model_save_dir $model_save_dir \
                          --resume_iters $iters \
                          --convert_dir $exp_root/For_acc_test/fine-tune-converted_mel_$iters \
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




# 重新生成没有 cpysys 文件的 scp

get_without_cpsys_scp_PATH=/bigdata/hsj/2021/convert_decode/convert_mel/get_without_cpsys_scp.py

$PYTHON $get_without_cpsys_scp_PATH  \
                           --convert_dir $exp_root/For_acc_test/fine-tune-converted_mel_$iters \
                           --scp_path $scp_path \



cat $scp_path
CUDA_VISIBLE_DEVICES=0
out_path=/bigdata/hsj/2021/all_wav_all_task/task141_$iters
converted_wav_dir=$out_path/convertes_wav_$iters
bash /bigdata/hsj/2021/convert_decode/pgan_decode/vctk_pgan.sh $scp_path $converted_wav_dir $CUDA_VISIBLE_DEVICES=1





