from model import GeneratorPlain
from model import PatchDiscriminator1
from model import SPEncoder
import torch
import torch.nn.functional as F
import os
from os.path import join, basename, exists
import time
import datetime
from data_loader_vctk import to_categorical
import numpy as np
from tqdm import tqdm
import numpy as np
import copy


class Solver(object):
    """Solver for training and testing WadaIN-VC."""

    def __init__(self, train_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        # self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # submodules
        self.D_name = config.discriminator
        self.use_sp_enc = config.use_sp_enc
        self.SPE_name = config.spenc
        self.G_name = config.generator
        self.res_block_name = config.res_block
        self.num_ft_speakers = config.num_ft_speakers
        # Model configurations.

        self.g_hidden_size = config.g_hidden_size
        self.num_speakers = config.num_speakers
        if self.use_sp_enc:
            self.spk_emb_dim = 128
        else:
            self.spk_emb_dim = self.num_speakers

        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.lambda_spid = config.lambda_spid
        self.lambda_adv = config.lambda_adv
        self.lambda_cls = config.lambda_cls
        self.drop_id_step = config.drop_id_step

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.use_ema = config.use_ema
        self.use_r1reg = config.use_r1reg
        # dynamic wadain configs
        self.kernel = config.kernel
        self.use_kconv = config.kconv
        self.num_heads = config.num_heads
        self.num_res_blocks = config.num_res_blocks

        # Test configurations.
        self.test_iters = config.test_iters
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.generator = eval(self.G_name)(num_speakers=self.num_speakers, res_block_name=self.res_block_name,
                                           kernel=self.kernel,
                                           num_heads=self.num_heads,
                                           num_res_blocks=self.num_res_blocks,
                                           use_kconv=self.use_kconv,
                                           spk_emb_dim=self.spk_emb_dim,
                                           hidden_size=self.g_hidden_size)
        self.discriminator = eval(self.D_name)(num_speakers=self.num_speakers, num_ft_speakers=self.num_ft_speakers)
        if self.use_sp_enc:
            self.sp_enc = eval(self.SPE_name)(num_speakers=self.num_speakers, num_ft_speakers=self.num_ft_speakers)
        # restore model
        if self.resume_iters:
            print("resuming step %d ..." % self.resume_iters, flush=True)
            self.restore_model(self.resume_iters)
        # initialize fine tune parameters

        if self.num_ft_speakers is not None:
            self.discriminator.init_ft_params()
            if self.use_sp_enc:
                self.sp_enc.init_ft_params()
        # [0921 new feature]: add ema model ckpt for evaluation
        if self.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            if self.use_sp_enc:
                self.sp_enc_ema = copy.deepcopy(self.sp_enc)
        # when fine tune, only update parameters in the ft_layers
        if self.num_ft_speakers is not None:
            if not self.use_sp_enc:
                raise Exception("fine tune must use speaker encoder")
            g_params = []
            for layer in self.sp_enc.ft_layers:
                g_params += list(layer.parameters())
            d_params = []
            for layer in self.discriminator.ft_layers:
                d_params += list(layer.parameters())
        else:
            g_params = list(self.generator.parameters())
            if self.use_sp_enc:
                g_params += list(self.sp_enc.parameters())
            d_params = list(self.discriminator.parameters())

        self.g_optimizer = torch.optim.Adam(g_params, self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(d_params, self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.generator, 'Generator')
        self.print_network(self.discriminator, 'Discriminator')
        if self.use_sp_enc:
            self.print_network(self.sp_enc, 'SpeakerEncoder')

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        if self.use_ema:
            self.generator_ema.to(self.device)
        if self.use_sp_enc:
            self.sp_enc.to(self.device)
            if self.use_ema:
                self.sp_enc_ema.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model, flush=True)
        print(name, flush=True)
        print("The number of parameters: {}".format(num_params), flush=True)

    def moving_average(self, model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters), flush=True)
        g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        sp_path = os.path.join(self.model_save_dir, '{}-sp.ckpt'.format(resume_iters))

        # [0919 new feature]: save and restore optimizer
        g_opt_path = os.path.join(self.model_save_dir, '{}-g_opt.ckpt'.format(resume_iters))
        d_opt_path = os.path.join(self.model_save_dir, '{}-d_opt.ckpt'.format(resume_iters))

        self.generator.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))
        if self.use_sp_enc:
            self.sp_enc.load_state_dict(torch.load(sp_path, map_location=lambda storage, loc: storage))
        # we do not restore optimizers when start fine tune training
        if self.num_ft_speakers is None:
            if exists(g_opt_path):
                self.g_optimizer.load_state_dict(torch.load(g_opt_path, map_location=lambda storage, loc: storage))
            if exists(d_opt_path):
                self.d_optimizer.load_state_dict(torch.load(d_opt_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradientgradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_mel(self, melfile):
        tmp_mel = np.load(melfile)
        return tmp_mel

    def train(self):
        """Train WadaIN-VC."""
        # Set data loader.
        train_loader = self.train_loader
        data_iter = iter(train_loader)

        cpsyn_flag = [True, False][0]

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            # print("resuming step %d ..."% self.resume_iters, flush=True)
            start_iters = self.resume_iters
            # self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...', flush=True)
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            if self.use_sp_enc:
                try:
                    mc_src, spk_label_org, spk_c_org, mc_trg, spk_label_trg, spk_c_trg = next(data_iter)
                except:
                    data_iter = iter(train_loader)
                    mc_src, spk_label_org, spk_c_org, mc_trg, spk_label_trg, spk_c_trg = next(data_iter)

                mc_src.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d
                mc_trg.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d
            else:
                try:
                    mc_src, spk_label_org, spk_c_org = next(data_iter)
                except:
                    data_iter = iter(train_loader)
                    mc_src, spk_label_org, spk_c_org = next(data_iter)
                spk_label_trg, spk_c_trg = self.sample_spk_c(mc_src.size(0))
                mc_src.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

            mc_src = mc_src.to(self.device)  # Input mc.
            if self.use_sp_enc:
                mc_trg = mc_trg.to(self.device)  # Input mc.
            spk_label_org = spk_label_org.to(self.device)  # Original spk labels.
            spk_c_org = spk_c_org.to(self.device)  # Original spk one-hot.
            spk_label_trg = spk_label_trg.to(self.device)  # Target spk labels.
            spk_c_trg = spk_c_trg.to(self.device)  # Target spk one-hot.

            # =================================================================================== #
            #                             2. Train the Discriminator                              #
            # =================================================================================== #
            pretrain_step = -1
            if i > pretrain_step:

                # org and trg speaker cond
                if self.use_sp_enc:
                    spk_c_trg = self.sp_enc(mc_trg, spk_label_trg)

                    spk_c_org = self.sp_enc(mc_src, spk_label_org)

                # Compute loss with real mc feats.
                if self.use_r1reg:
                    mc_src = mc_src.requires_grad_()
                d_out_src = self.discriminator(mc_src, spk_label_trg, spk_label_org)
                d_loss_real = torch.mean((1.0 - d_out_src) ** 2)

                def r1_reg(d_out, x_in):
                    batch = x_in.size(0)
                    grad_dout = torch.autograd.grad(
                        outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True
                    )[0]
                    grad_dout2 = grad_dout.pow(2)
                    assert (grad_dout2.size() == x_in.size())
                    reg = 0.5 * grad_dout2.view(batch, -1).sum(1).mean(0)
                    return reg

                if self.use_r1reg:
                    d_reg = r1_reg(d_out_src, mc_src)

                # Compute loss with face mc feats.
                mc_fake = self.generator(mc_src, spk_c_org, spk_c_trg)
                d_out_fake = self.discriminator(mc_fake.detach(), spk_label_org, spk_label_trg)
                d_loss_fake = torch.mean(d_out_fake ** 2)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake
                if self.use_r1reg:
                    d_loss += d_reg
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss'] = d_loss.item()
                if self.use_r1reg:
                    loss['D/loss_reg'] = d_reg.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            if (i + 1) % self.n_critic == 0:

                # org and trg speaker cond
                if self.use_sp_enc:
                    spk_c_trg = self.sp_enc(mc_trg, spk_label_trg)
                    spk_c_org = self.sp_enc(mc_src, spk_label_org)

                # Original-to-target domain.
                mc_fake = self.generator(mc_src, spk_c_org, spk_c_trg)
                g_out_src = self.discriminator(mc_fake, spk_label_org, spk_label_trg)
                g_loss_fake = torch.mean((1.0 - g_out_src) ** 2)

                # Target-to-original domain. Cycle-consistent.
                mc_reconst = self.generator(mc_fake, spk_c_trg, spk_c_org)
                g_loss_rec = torch.mean(torch.abs(mc_src - mc_reconst))

                # Original-to-original, Id mapping loss. Mapping
                mc_fake_id = self.generator(mc_src, spk_c_org, spk_c_org)
                g_loss_id = torch.mean(torch.abs(mc_src - mc_fake_id))

                if self.use_sp_enc:
                    # style encoder reconstruction loss

                    mc_fake_style_c = self.sp_enc(mc_fake, spk_label_trg)

                    g_loss_spid = -(torch.mean(torch.cosine_similarity(mc_fake_style_c, spk_c_trg, dim=1)))

                    # MAE-version  spid-loss(speaker embedding cycle loss)
                    # g_loss_spid = torch.mean(torch.abs(mc_fake_style_c - spk_c_trg))


                if i > self.drop_id_step:
                    self.lambda_id = 0
                # Backward and optimize.
                g_loss = self.lambda_adv * g_loss_fake \
                         + self.lambda_rec * g_loss_rec \
                         + self.lambda_id * g_loss_id
                if self.use_sp_enc:
                    g_loss += self.lambda_spid * g_loss_spid

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_id'] = g_loss_id.item()
                if self.use_sp_enc:
                    loss['G/loss_spid'] = g_loss_spid.item()
            if self.use_ema:
                self.moving_average(self.generator, self.generator_ema)
                if self.use_sp_enc:
                    self.moving_average(self.sp_enc, self.sp_enc_ema)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log, flush=True)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                if self.num_ft_speakers is None:
                    g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                    g_path_ema = os.path.join(self.model_save_dir, '{}-G.ckpt.ema'.format(i + 1))
                    d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                    if self.use_sp_enc:
                        sp_path = os.path.join(self.model_save_dir, '{}-sp.ckpt'.format(i + 1))
                        sp_path_ema = os.path.join(self.model_save_dir, '{}-sp.ckpt.ema'.format(i + 1))

                    # save and restore optimizer
                    g_opt_path = os.path.join(self.model_save_dir, '{}-g_opt.ckpt'.format(i + 1))
                    d_opt_path = os.path.join(self.model_save_dir, '{}-d_opt.ckpt'.format(i + 1))
                else:
                    g_path = os.path.join(self.model_save_dir, '{}-G.ckpt.ft'.format(i + 1))
                    g_path_ema = os.path.join(self.model_save_dir, '{}-G.ckpt.ema.ft'.format(i + 1))
                    d_path = os.path.join(self.model_save_dir, '{}-D.ckpt.ft'.format(i + 1))
                    if self.use_sp_enc:
                        sp_path = os.path.join(self.model_save_dir, '{}-sp.ckpt.ft'.format(i + 1))
                        sp_path_ema = os.path.join(self.model_save_dir, '{}-sp.ckpt.ema.ft'.format(i + 1))

                    # save and restore optimizer
                    g_opt_path = os.path.join(self.model_save_dir, '{}-g_opt.ckpt.ft'.format(i + 1))
                    d_opt_path = os.path.join(self.model_save_dir, '{}-d_opt.ckpt.ft'.format(i + 1))

                torch.save(self.generator.state_dict(), g_path)
                if self.use_ema:
                    torch.save(self.generator_ema.state_dict(), g_path_ema)
                torch.save(self.discriminator.state_dict(), d_path)
                if self.use_sp_enc:
                    torch.save(self.sp_enc.state_dict(), sp_path)
                    if self.use_ema:
                        torch.save(self.sp_enc_ema.state_dict(), sp_path_ema)
                torch.save(self.g_optimizer.state_dict(), g_opt_path)
                torch.save(self.d_optimizer.state_dict(), d_opt_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir), flush=True)

