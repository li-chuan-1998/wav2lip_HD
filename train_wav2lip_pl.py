from models import SyncNet_color as SyncNet, Wav2Lip
# from utils.dataloader import Dataset
from utils.helper import load_lip_sync_expert
from torch.utils import data as data_utils

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl

import torch.nn as nn
import torch, argparse, os

from hparams import hparams, get_image_list
from os.path import dirname, join, basename, isfile
import cv2, numpy as np, random
import utils.audio as audio
from glob import glob

syncnet_T = 5
syncnet_mel_step_size = 16
class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y

class Wav2LipLightning(pl.LightningModule):

    def __init__(self, syncnet):
        super(Wav2LipLightning, self).__init__()
        # self.save_hyperparameters(ignore=['syncnet'])
        self.syncnet = syncnet
        self.model = Wav2Lip()
        self.logloss = nn.BCEWithLogitsLoss() # nn.BCELoss()
        self.recon_loss = nn.L1Loss()
        self.syncnet_T = 5
        self.val_sync_loss = []

    def forward(self, indiv_mels, x):
        return self.model(indiv_mels, x)

    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = self.logloss(d.unsqueeze(1), y)
        return loss

    def get_sync_loss(self, mel, g):
        g = g[:, :, :, g.size(3)//2:]
        g = torch.cat([g[:, :, i] for i in range(self.syncnet_T)], dim=1).cuda()
        # B, 3 * T, H//2, W
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().cuda()
        return self.cosine_loss(a, v, y)

    def training_step(self, batch, batch_idx):
        x, indiv_mels, mel, gt = batch
        
        g = self(indiv_mels, x)
        
        if hparams.syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(mel, g)
        else:
            sync_loss = 0.

        l1loss = self.recon_loss(g, gt)
        loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss

        # Logging
        self.log('train_l1_loss', l1loss, prog_bar=True)
        self.log('train_sync_loss', sync_loss, prog_bar=True)
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=hparams.initial_learning_rate)

    # Optional: define validation_step for validation
    def validation_step(self, batch, batch_idx):
        x, indiv_mels, mel, gt = batch
        g = self(indiv_mels, x)

        if hparams.syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(mel, g)
        else:
            sync_loss = 0.

        l1loss = self.recon_loss(g, gt)
        
        # Logging
        self.log('val_l1_loss', l1loss, prog_bar=True)
        self.log('val_sync_loss', sync_loss, prog_bar=True)
        self.val_sync_loss.append(sync_loss)

    def on_validation_epoch_end(self):
        # Calculate the mean val_sync_loss
        mean_val_sync_loss = torch.tensor(self.val_sync_loss).mean()
        self.log('mean_val_sync_loss', mean_val_sync_loss, prog_bar=True)

        # Update hparams.syncnet_wt if mean_val_sync_loss is lower than 0.75
        if mean_val_sync_loss < 0.75:
            hparams.syncnet_wt = 0.01
            
        self.val_sync_loss.clear()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="./ckp_w2l/", type=str)
    parser.add_argument('--syncnet_ckp_path', help='Load the pre-trained Expert discriminator', default="./ckp_ls/lip_sync.pth", type=str)
    parser.add_argument("--batch_size", type=int, default=16,  help="the batch size")
    parser.add_argument("--num_devices", type=int, default=1,  help="the number of gpus to use")
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
    args = parser.parse_args()
    
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    
    # init Lip-Sync Expert (setting all params to non-trainable)
    syncnet = SyncNet()
    syncnet = load_lip_sync_expert(syncnet, args.syncnet_ckp_path)

    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_data_loader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=16)

    model = Wav2LipLightning(syncnet)


    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, 
                                          filename="w2l-{epoch:05d}-{train_loss:.5f}", 
                                          save_top_k=10, monitor="train_loss")
    trainer = pl.Trainer(max_epochs=hparams.nepochs,default_root_dir=args.checkpoint_dir, callbacks=[checkpoint_callback],  
                         accelerator="gpu", devices=args.num_devices) 
    # default_root_dir=args.checkpoint_dir, callbacks=[checkpoint_callback], precision=16
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
