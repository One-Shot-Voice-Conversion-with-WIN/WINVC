import glob
import os
from os.path import join, basename, dirname, split, exists
import argparse

convert_dir = '/home/hsj/2021/data/0222_All_data_for_fine_tune_train/larger_than_128frame_norm/train80p_fine-tune10p/fine-tune-5M_5F/ft_test_20p_20norm_10train_10ft'
scp_path = '/home/hsj/2021/data/0222_All_data_for_fine_tune_train/larger_than_128frame_norm/train80p_fine-tune10p/fine-tune-5M_5F/ft_test_20p_20norm_10train_10ft/feats.scp'


# ********************************************

parser = argparse.ArgumentParser(description="Decode dumped features with trained Parallel WaveGAN Generator ")
parser.add_argument("--convert_dir", "--convert", default=None, type=str)
parser.add_argument("--scp_path", default=None, type=str)

args = parser.parse_args()


# ********************************************


npy_files = sorted(glob.glob(os.path.join(args.convert_dir,'*.npy')))
print(f"found {len(npy_files)} npy files")
if os.path.exists(args.scp_path):
    os.remove(args.scp_path)
f = open(args.scp_path, 'w')
for npy_f in npy_files:
    filename = basename(npy_f).split('.')[-2]
    if(filename[0] is not 'p'):
        pass
    else:
        print(f"{filename}")
        f.write(filename + ' ' + npy_f + '\n')