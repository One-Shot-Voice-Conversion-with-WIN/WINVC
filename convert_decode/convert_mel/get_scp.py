import glob
import os
from os.path import join, basename, dirname, split, exists
import argparse

convert_dir = ''
scp_path = ''



# ********************************************

parser = argparse.ArgumentParser(description="Decode dumped features with trained Parallel WaveGAN Generator ")
parser.add_argument("--convert_dir", "--convert", default=convert_dir, type=str)
parser.add_argument("--scp_path", default=scp_path, type=str)

args = parser.parse_args()


# ********************************************


npy_files = sorted(glob.glob(os.path.join(args.convert_dir,'*.npy')))
print(f"found {len(npy_files)} npy files")
if os.path.exists(args.scp_path):
    os.remove(args.scp_path)
f = open(args.scp_path, 'w')
for npy_f in npy_files:
    filename = basename(npy_f).split('.')[-2]
    print(f"{filename}")
    f.write(filename + ' ' + npy_f + '\n')
