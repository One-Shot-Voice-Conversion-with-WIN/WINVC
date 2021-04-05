import argparse

from model import GeneratorPlain
from model import SPEncoder
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split, exists
import time
import datetime
import librosa
import glob
import json

from data_loader_vctk import TestDataset
from concurrent.futures import ProcessPoolExecutor
import subprocess
from tqdm import tqdm
from functools import partial
import pyloudnorm


def load_mel(melfile):
    tmp_mel = np.load(melfile)
    return tmp_mel


def process_test_loader(test_loader, G, sp_enc, device, sampling_rate, num_mels, frame_period, config):
    print("line 34: batch_size=config.num_converted_wavs = ", config.num_converted_wavs)
    test_melfiles = test_loader.get_batch_test_data(batch_size=config.num_converted_wavs)
    if config.use_sp_enc:
        test_mels = [(load_mel(melfile), load_mel(trgfile)) for melfile, trgfile in test_melfiles]
    else:
        test_mels = [load_mel(melfile) for melfile in test_melfiles]
    if config.use_loudnorm:
        loud_meter = pyloudnorm.Meter(sampling_rate)
    else:
        loud_meter = None
    with torch.no_grad():
        for idx, data in enumerate(test_mels):
            if isinstance(data, tuple):
                mel, trg_mel = data
                wav_name = basename(test_melfiles[idx][0])
            else:
                mel = data
                wav_name = basename(test_melfiles[idx])

            coded_sp_norm_tensor = torch.FloatTensor(np.array([mel.T])).unsqueeze_(0).to(device)
            if config.use_sp_enc:
                trg_sp_tensor = torch.FloatTensor(np.array([trg_mel.T])).unsqueeze_(0).to(device)

                conds = sp_enc(trg_sp_tensor, torch.LongTensor([test_loader.spk_idx_trg]).to(device))
            else:
                conds = torch.FloatTensor(test_loader.spk_c_trg).to(device)
            print("Before being fed into G: ", coded_sp_norm_tensor.size(), flush=True)

            coded_sp_converted_norm = G(coded_sp_norm_tensor, conds, conds).data.cpu().numpy()

            coded_sp_converted = np.squeeze(coded_sp_converted_norm).T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

            print("After being fed into G: ", coded_sp_converted.shape, flush=True)

            try:
                np.save(join(config.convert_dir, wav_name.split('-')[0] + '-vcto-{}'.format(
                    test_loader.trg_spk) + '.npy'), coded_sp_converted)
            except:
                print(f"converted voice contains nan! skip", flush=True)
                pass
            np.save(join(config.convert_dir, 'cpsyn-' + wav_name), mel)


def _convert(test_loader, G, sp_enc, device, sampling_rate, num_mels, frame_period, config):
    process_test_loader(test_loader, G, sp_enc, device, sampling_rate, num_mels, frame_period, config)


def convert(config):
    # load speakers
    with open(config.speaker_path) as f:
        speakers = json.load(f)
    if config.num_ft_speakers is not None:
        with open(config.ft_speaker_path) as f:
            ft_speakers = json.load(f)
    else:
        ft_speakers = None
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mels, frame_period = config.sample_rate, 80, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.use_sp_enc:
        spk_emb_dim = 128
    else:
        spk_emb_dim = len(speakers)
    G = eval(config.generator)(num_speakers=config.num_speakers, res_block_name=config.res_block,
                               kernel=config.kernel,
                               use_kconv=config.kconv,
                               num_res_blocks=config.num_res_blocks,
                               num_heads=config.num_heads,
                               spk_emb_dim=spk_emb_dim,
                               hidden_size=config.g_hidden_size).to(device)
    if config.use_sp_enc:
        sp_enc = eval(config.speaker_encoder)(num_speakers=config.num_speakers,
                                              num_ft_speakers=config.num_ft_speakers).to(device)

        if config.num_ft_speakers:
            sp_enc.init_ft_params()

    # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...', flush=True)
    if config.use_ema:

        G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt.ema')
        sp_enc_path = join(config.model_save_dir, f"{config.resume_iters}-sp.ckpt.ema")
    else:
        G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
        sp_enc_path = join(config.model_save_dir, f"{config.resume_iters}-sp.ckpt")
    if config.num_ft_speakers is not None:
        G_path += '.ft'
        sp_enc_path += '.ft'
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    G.eval()
    if config.use_sp_enc:
        sp_enc.load_state_dict(torch.load(sp_enc_path, map_location=lambda storage, loc: storage))
        sp_enc.eval()
    else:
        sp_enc = None

    if config.src_spk is not None and config.trg_spk is not None:

        test_loader = TestDataset(speakers_using=speakers, ft_speakers=ft_speakers, data_dir=config.dump_dir,
                                  ft_data_dir=config.ft_dump_dir, src_spk=config.src_spk, trg_spk=config.trg_spk,
                                  use_sp_enc=config.use_sp_enc, model_stat_path=config.model_stat_path
                                  , pgan_stat_path=config.pgan_stat_path)
        process_test_loader(test_loader, G, sp_enc, device, sampling_rate, num_mels, frame_period, config)
    else:
        for src in speakers:
            for trg in speakers:
                if src != trg:
                    test_loader = TestDataset(speakers_using=speakers, ft_speakers=ft_speakers,
                                              data_dir=config.dump_dir, src_spk=src, trg_spk=trg,
                                              use_sp_enc=config.use_sp_enc, model_stat_path=config.model_stat_path,
                                              pgan_stat_path=config.pgan_stat_path)
                    _convert(test_loader, G, sp_enc, device, sampling_rate, num_mels, frame_period, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate')
    parser.add_argument('--num_converted_wavs', type=int, default=None,
                        help='number of wavs to convert, if not defined, will convert all')
    parser.add_argument('--resume_iters', type=int, default=None, help='step to resume for testing.')
    parser.add_argument('--src_spk', type=str, default=None, help='target speaker.')
    parser.add_argument('--trg_spk', type=str, default=None, help='target speaker.')
    parser.add_argument('--generator', type=str, default='Gen', required=True)
    parser.add_argument('--g_hidden_size', type=int, default=256, help='bottleneck hidden size for generator')
    parser.add_argument('--use_sp_enc', default=False, action='store_true')
    parser.add_argument('--speaker_encoder', type=str, required=True)
    parser.add_argument('--res_block', type=str, required=True)
    parser.add_argument('--use_ema', default=False, action='store_true')
    parser.add_argument('--use_loudnorm', default=False, action='store_true')
    parser.add_argument('--dump_dir', type=str, required=True)
    parser.add_argument('--ft_dump_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--convert_dir', type=str, default='./converted')
    parser.add_argument('--speaker_path', type=str, required=True)
    parser.add_argument('--ft_speaker_path', type=str, default=None, help="fine tune speaker list")
    parser.add_argument('--num_ft_speakers', type=int, default=None, help="number of fine tune speakers")
    parser.add_argument('--kernel', default=None, type=int, help='kernel size for dynamic wadain')
    parser.add_argument('--num_heads', default=None, type=int, help='num_heads for dynamic wadain')
    parser.add_argument('--kconv', default=False, action='store_true', help='use kconv for dynamic wadain')
    parser.add_argument('--num_res_blocks', default=False, type=int, help='num of res blocks for G')

    parser.add_argument('--model_stat_path', type=str, help="path to stats for WadaIN-VC model training")
    parser.add_argument('--pgan_stat_path', type=str, help="path to stats for pgan vocoder")
    parser.add_argument('--cpsyn', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=None, help='multi-process')

    parser.add_argument('--scp_path', type=str, help='scp file for pgan vocoder')
    config = parser.parse_args()

    print(config, flush=True)
    if config.resume_iters is None:
        raise RuntimeError("Please specify the step number for resuming.")
    convert(config)

    npy_files = sorted(glob.glob(config.convert_dir + '/*.npy'))
    print(f"found {len(npy_files)} npy files")
    if os.path.exists(config.scp_path):
        os.remove(config.scp_path)
    f = open(config.scp_path, 'a')
    for npy_f in npy_files:
        filename = basename(npy_f).split('.')[-2]
        print(f"{filename}")
        f.write(filename + ' ' + npy_f + '\n')
