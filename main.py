import os
import argparse
from solver import Solver
from data_loader_vctk import get_loader, get_ft_loader
from torch.backends import cudnn
import json
import numpy as np
import random
import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.speaker_path):
        raise Exception(f"speaker list {config.speaker_path} does not exist")
    with open(config.speaker_path) as f:
        speakers = json.load(f)
    print(f"load speakers {speakers}", flush=True)

    if config.num_ft_speakers is not None:
        if not os.path.exists(config.ft_speakers):
            raise Exception(f"ft speakers {config.ft_speakers} json file doe snot exist!")
        with open(config.ft_speakers) as f_:
            ft_speakers = json.load(f_)
        print(f"load ft speakers {ft_speakers}", flush=True)

    config.speakers = speakers
    config.num_speakers = len(speakers)

    if config.num_ft_speakers is None:
        train_loader = get_loader(speakers, config.train_data_dir, config.min_length, config.batch_size, 'train',
                                  num_workers=config.num_workers, use_sp_enc=config.use_sp_enc,
                                  scp_path=config.scp_path, stat_path=config.stat_path)
    else:
        train_loader = get_ft_loader(speakers, ft_speakers, config.train_data_dir, config.ft_data_dir,
                                     config.min_length, config.batch_size, 'train', config.num_workers,
                                     config.use_sp_enc, config.scp_path, config.stat_path, config.ft_scp)

    solver = Solver(train_loader, config)

    if config.mode == 'train':
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cls', type=float, default=10, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=4, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_adv', type=float, default=1, help='weight for adversarial training')
    parser.add_argument('--lambda_id', type=float, default=2, help='weight for id mapping loss')
    parser.add_argument('--lambda_spid', type=float, default=5, help='weight for id mapping loss')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')

    # modules
    parser.add_argument('--discriminator', type=str, default='PatchDiscriminator1')
    parser.add_argument('--use_sp_enc', default=False, action='store_true', help='use speaker encoder or not')
    parser.add_argument('--spenc', type=str, default='SPEncoder')
    parser.add_argument('--generator', type=str, default='GeneratorPlain')
    parser.add_argument('--g_hidden_size', type=int, default=256, help='bottleneck hidden size for generator')
    parser.add_argument('--res_block', type=str, default='Style2ResidualBlock1DBeta')
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--min_length', type=int, default=256)
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
    parser.add_argument('--drop_id_step', type=int, default=100000, help='steps drop id mapping loss')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--device', type=int, default=0, help='choosing cuda device')
    parser.add_argument('--use_ema', default=False, action='store_true', help='use ema for training')
    parser.add_argument('--use_r1reg', default=False, action='store_true',
                        help='use r1 discriminator regularization term')

    # dynamic wadain module configs
    parser.add_argument('--kernel', default=None, type=int, help='kernel size for dynamic wadain')
    parser.add_argument('--num_heads', default=None, type=int, help='num_heads for dynamic wadain')
    parser.add_argument('--kconv', default=False, action='store_true', help='use kconv for dynamic wadain')
    parser.add_argument('--num_res_blocks', default=False, type=int, help='num of res blocks for G')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')
    parser.add_argument('--test_src_spk', type=str, default='SSB0033')
    parser.add_argument('--test_trg_spk', type=str, default='SSB0073')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    # fine tune
    parser.add_argument('--num_ft_speakers', type=int, default=None, help="number of speakers for fine tuning")
    parser.add_argument('--ft_scp', type=str, default=None, help="scp files path for fine tuning")
    parser.add_argument('--ft_speakers', type=str, default=None, help="json file for fine tuning speakers")
    parser.add_argument('--num_few_shot', type=int, default=None, help="number of fine tuning samples for each speaker")
    parser.add_argument('--ft_data_dir', type=str, default=None, help='dump dir for fine tunine')
    # Directories.
    parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='./data/mc/test')
    parser.add_argument('--wav_dir', type=str, default="./data/VCTK-Corpus/wav16")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--speaker_path', type=str)
    parser.add_argument('--scp_path', type=str, help="scp path")
    parser.add_argument('--stat_path', type=str, default=None, help="mean variance statics")
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    set_seed(1234)
    main(config)
