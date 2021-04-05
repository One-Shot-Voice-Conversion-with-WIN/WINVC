from torch.utils import data
from sklearn.preprocessing import StandardScaler
import torch
import glob
import os
from os.path import join, basename, dirname, split
import numpy as np


# convert int to one-hot
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# for fine tune
class FTDataset(data.Dataset):
    """Fine Tune Dataset for Mel-Spectrogram features and speaker labels."""

    def __init__(self, speakers_using, ft_speakers, data_dir, ft_data_dir, min_length=256, use_sp_enc=False,
                 scp_path=None, stat_path=None, ft_scp_path=None):
        self.speakers = speakers_using + ft_speakers
        self.ft_speakers = ft_speakers

        self.min_length = min_length
        self.mc_files = []
        self.spk2files = {}
        self.ft_spk2files = {}

        # load stat for training model
        if stat_path is not None:
            if not os.path.exists(stat_path):
                raise Exception(f"stat path {stat_path} not exist!")
            stat = np.load(stat_path)
            print(f"load stat from {stat_path} with {stat[0].shape} shape")
            self.scaler = StandardScaler()
            self.scaler.mean_ = stat[0]
            self.scaler.scale_ = stat[1]
        else:
            self.scaler = None
        # use speaker encoder or not
        # use speaker encoder means sample one source feature and one target feature
        # otherwise only sample one source feature, then use one-hot as target speaker input
        self.use_sp_enc = use_sp_enc
        # parse scp file

        # parse ft_scp file
        if ft_scp_path is not None:
            ft_scp_f = open(ft_scp_path, encoding='utf-8')
            for line in ft_scp_f:
                spk = line.split(' ')[0].split('_')[0]
                if spk not in self.ft_speakers:
                    raise Exception(f"spk {spk} not in ft_speakers {self.ft_speakers}")
                if spk not in self.ft_spk2files:
                    self.ft_spk2files[spk] = []
                filename = line.split(' ')[0] + '.npy'
                path = join(ft_data_dir, spk, filename)
                if not os.path.exists(path):
                    raise Exception(f"fine tune path {path} does not exists")
                self.ft_spk2files[spk].append(path)
        else:
            raise Exception(f"we need a scp file for fine tuning")

        self.prefix_length = len(self.speakers[0])
        self.spk2idx = dict(
            zip(self.speakers, range(len(self.speakers))))

        # add ft mc files into src mc files
        for spk in self.ft_spk2files:
            _files = self.ft_spk2files[spk]
            _spkfiles = [(spk, f) for f in _files]
            self.mc_files.extend(_spkfiles)
        self.num_files = len(self.mc_files)

        for spk, path in self.mc_files:
            if spk not in self.spk2files:
                self.spk2files[spk] = []
            self.spk2files[spk].append(path)
        print("\t Number of training samples: ", self.num_files)

    def rm_too_short_utt(self, mc_files, min_length):
        """hsj:0225-pad to 256 frames"""
        new_mc_files = []
        print("Hello World! :min_length = ", min_length)

        for spk, mc_file in mc_files:
            new_mc_files.append(spk, mc_file)

        return new_mc_files

    def sample_seg(self, feat):
        assert feat.shape[0] - self.min_length >= 0
        s = np.random.randint(0, feat.shape[0] - self.min_length + 1)
        if self.scaler is not None:

            return self.scaler.transform(feat[s:s + self.min_length, :])
        else:
            return feat[s:s + self.min_length, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        src_spk, src_filename = self.mc_files[index]

        if src_spk not in self.speakers:
            raise Exception(f"speaker {src_spk} not in self.speakers {self.speakers}")
        src_spk_idx = self.spk2idx[src_spk]

        src_mc = np.load(src_filename)
        # 0225-黄圣杰
        if src_mc.shape[0] < self.min_length:
            src_mc = np.pad(src_mc, ((0, 256 - src_mc.shape[0] + 1), (0, 0)), 'constant')
        src_mc = self.sample_seg(src_mc)
        src_mc = np.transpose(src_mc, (1, 0))
        # to one-hot
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))

        if self.use_sp_enc:
            # sample target speaker, source speaker is excluded
            ft_speakers = self.ft_speakers[:]
            if src_spk in self.ft_speakers:
                ft_speakers.remove(src_spk)

            trg_spk_sample = np.random.randint(0, len(ft_speakers))
            trg_spk = ft_speakers[trg_spk_sample]
            # trg_spk_idx = len(self.speakers) + trg_spk_sample
            trg_spk_idx = self.spk2idx[trg_spk]
            trg_spk_cat = np.squeeze(
                to_categorical([trg_spk_idx], num_classes=len(self.speakers) + len(self.ft_speakers)))
            # sample one target speaker feature file, will be the input to the speaker encoder
            trg_spk_files = self.ft_spk2files[trg_spk]
            trg_file_sample = np.random.randint(0, len(trg_spk_files))
            trg_filename = trg_spk_files[trg_file_sample]

            trg_mc = np.load(trg_filename)
            # segment length also min_length
            # 0225-黄圣杰
            if trg_mc.shape[0] < self.min_length:
                trg_mc = np.pad(trg_mc, ((0, 256 - trg_mc.shape[0] + 1), (0, 0)), 'constant')
            trg_mc = self.sample_seg(trg_mc)
            trg_mc = np.transpose(trg_mc, (1, 0))
        else:
            raise Exception("we need speaker encoder for fine tuning")

        return torch.FloatTensor(src_mc), torch.LongTensor([src_spk_idx]).squeeze_(), torch.FloatTensor(
            src_spk_cat), torch.FloatTensor(trg_mc), torch.LongTensor([trg_spk_idx]).squeeze_(), torch.FloatTensor(
            trg_spk_cat)


# for train
class MyDataset(data.Dataset):
    """Dataset for mel features and speaker labels."""

    def __init__(self, speakers_using, data_dir, min_length=256, use_sp_enc=False, scp_path=None, stat_path=None):
        self.speakers = speakers_using
        self.min_length = min_length
        self.mc_files = []
        self.spk2files = {}

        # load stat for training model
        if stat_path is not None:
            if not os.path.exists(stat_path):
                raise Exception(f"stat path {stat_path} not exist!")
            stat = np.load(stat_path)
            print(f"load stat from {stat_path} with {stat[0].shape} shape")
            self.scaler = StandardScaler()
            self.scaler.mean_ = stat[0]
            self.scaler.scale_ = stat[1]
        else:
            self.scaler = None
        # use speaker encoder or not
        # use speaker encoder means sample one source feature and one target feature
        # otherwise only sample one source feature, then use one-hot as target speaker input
        self.use_sp_enc = use_sp_enc

        if scp_path is not None:
            scp_f = open(scp_path, encoding="utf-8")
            files = []
            for line in scp_f:

                spk = line.split(' ')[0].split('_')[0]
                #                print("229:spk= ",spk)
                if spk not in speakers_using:
                    raise Exception(f"spk {spk} is not in speaker using list {speakers_using}")
                filename = line.split(' ')[0] + '.npy'
                path = join(data_dir, spk, filename)
                if not os.path.exists(path):
                    raise Exception(f"file {path} does not exist!")

                # 贾宇康-借鉴——>数据全部读进内存，加速训练
                file_data = np.load(path)
                if file_data.shape[0] <= min_length:
                    pass
                else:
                    assert file_data.shape[0] >= min_length, 'some less length in scp files'
                    self.mc_files.append((spk, file_data))


        else:
            raise Exception(f"we need a scp file here!")
        self.prefix_length = len(self.speakers[0])
        self.spk2idx = dict(
            zip(self.speakers, range(len(self.speakers))))

        self.num_files = len(self.mc_files)

        for spk, path in self.mc_files:
            if spk not in self.spk2files:
                self.spk2files[spk] = []
            self.spk2files[spk].append(path)
        print("\t Load dataset finished ")
        print("\t Number of training samples: ", self.num_files)

    def rm_too_short_utt(self, mc_files, min_length):
        """hsj:0225-pad to 256 frames"""
        new_mc_files = []
        print("Hello World! :min_length = ", min_length)

        for spk, mc_file in mc_files:
            mc = np.load(mc_file)  # mc_file 是地址
            if mc.shape[0] <= min_length:
                mc = np.pad(mc, ((0, 256 - mc.shape[0] + 1), (0, 0)), 'constant')
            if mc.shape[0] > min_length:
                new_mc_files.append((spk, mc_file))

        return new_mc_files

    def sample_seg(self, feat):
        assert feat.shape[0] - self.min_length >= 0
        s = np.random.randint(0, feat.shape[0] - self.min_length + 1)
        if self.scaler is not None:

            return self.scaler.transform(feat[s:s + self.min_length, :])
        else:
            return feat[s:s + self.min_length, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        src_spk, src_filename = self.mc_files[index]

        if src_spk not in self.speakers:
            raise Exception(f"speaker {src_spk} not in self.speakers {self.speakers}")
        src_spk_idx = self.spk2idx[src_spk]

        # jyk-借鉴
        src_mc = src_filename
        # 黄圣杰-0225
        if src_mc.shape[0] < self.min_length:
            src_mc = np.pad(src_mc, ((0, 256 - src_mc.shape[0] + 1), (0, 0)), 'constant')
        src_mc = self.sample_seg(src_mc)
        src_mc = np.transpose(src_mc, (1, 0))
        # to one-hot
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))

        if self.use_sp_enc:
            # sample target speaker, source speaker is excluded
            speakers = self.speakers[:]
            speakers.remove(src_spk)

            trg_spk_sample = np.random.randint(0, len(speakers))
            trg_spk = speakers[trg_spk_sample]
            trg_spk_idx = self.speakers.index(trg_spk)
            trg_spk_cat = np.squeeze(to_categorical([trg_spk_idx], num_classes=len(self.speakers)))
            # sample one target speaker feature file, will be the input to the speaker encoder
            trg_spk_files = self.spk2files[trg_spk]
            trg_file_sample = np.random.randint(0, len(trg_spk_files))
            trg_filename = trg_spk_files[trg_file_sample]

            # jyk-借鉴
            trg_mc = trg_filename

            # 黄圣杰-0225
            if trg_mc.shape[0] < self.min_length:
                trg_mc = np.pad(trg_mc, ((0, 256 - trg_mc.shape[0] + 1), (0, 0)), 'constant')
            # segment length also min_length
            trg_mc = self.sample_seg(trg_mc)
            trg_mc = np.transpose(trg_mc, (1, 0))

            return torch.FloatTensor(src_mc), torch.LongTensor([src_spk_idx]).squeeze_(), torch.FloatTensor(
                src_spk_cat), torch.FloatTensor(trg_mc), torch.LongTensor([trg_spk_idx]).squeeze_(), torch.FloatTensor(
                trg_spk_cat)
        else:
            return torch.FloatTensor(src_mc), torch.LongTensor([src_spk_idx]).squeeze_(), torch.FloatTensor(src_spk_cat)


# abandoned
class TestDataset(object):
    def __init__(self, speakers_using, data_dir, ft_data_dir=None, src_spk='SSB0005', trg_spk='SSB0073',
                 use_sp_enc=False, model_stat_path=None, pgan_stat_path=None, ft_speakers=None):
        self.speakers = speakers_using
        if ft_speakers is not None:
            self.speakers += ft_speakers

        self.spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.prefix_length = len(self.speakers[0])
        self.use_sp_enc = use_sp_enc
        self.src_spk = src_spk
        self.trg_spk = trg_spk
        # self.mc_files = sorted(glob.glob(join(data_dir, '{}*.npy'.format(self.src_spk))))

        mc_files = []
        mc_files.extend(sorted(glob.glob(join(data_dir, src_spk, '*.npy'))))
        if ft_data_dir is not None:
            mc_files.extend(sorted(glob.glob(join(ft_data_dir, src_spk, '*.npy'))))
        self.mc_files = mc_files
        if len(self.mc_files) == 0:
            raise Exception(f"found no mc files in path {data_dir}")

        if use_sp_enc:
            if ft_speakers is None:
                self.trg_mc_files = sorted(glob.glob(join(data_dir, trg_spk, '*.npy')))
            else:
                self.trg_mc_files = sorted(glob.glob(join(ft_data_dir, trg_spk, '*.npy')))

        '''原始说话人的 mel 特征——>不需要目标人的，只需要source说话人的80-mel'''
        self.src_mel_dir = f'{data_dir}/{src_spk}'

        '''这部分是 说话人one-hot 标签，需要保留'''
        self.spk_idx_src, self.spk_idx_trg = self.spk2idx[src_spk], self.spk2idx[trg_spk]
        spk_cat_src = to_categorical([self.spk_idx_src], num_classes=len(self.speakers))
        spk_cat_trg = to_categorical([self.spk_idx_trg], num_classes=len(self.speakers))
        self.spk_c_org = spk_cat_src
        self.spk_c_trg = spk_cat_trg

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mc_file = self.mc_files[i]  # ./data/dump/train_nodev/norm/SSB0005/SSB00050084-feats.npy
            if self.use_sp_enc:
                trg_file = self.trg_mc_files[0]
                batch_data.append((mc_file, trg_file))  # 到这里为止，返回的是一个mel特征的存储地址 list[]
            else:
                batch_data.append(mc_file)
        return batch_data


# for train
def get_loader(speakers_using, data_dir, min_length, batch_size=32, mode='train', num_workers=1, use_sp_enc=False,
               scp_path=None, stat_path=None):
    dataset = MyDataset(speakers_using, data_dir, min_length=min_length, use_sp_enc=use_sp_enc, scp_path=scp_path,
                        stat_path=stat_path)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


# for fine tune
def get_ft_loader(speakers_using, ft_speakers, data_dir, ft_data_dir, min_length, batch_size=32, mode='train',
                  num_workers=1, use_sp_enc=False, scp_path=None, stat_path=None, ft_scp_path=None):
    dataset = FTDataset(speakers_using, ft_speakers, data_dir, ft_data_dir, min_length=min_length,
                        use_sp_enc=use_sp_enc, scp_path=scp_path, stat_path=stat_path, ft_scp_path=ft_scp_path)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader
