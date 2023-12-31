from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
import utils.audio as audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
from utils.train_utils import recon_loss, get_sync_loss, eval_model, save_checkpoint, load_checkpoint
from utils.helper import maintain_num_checkpoints

global_step = 0
global_epoch = 0
lowest_sync_loss = 1.0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

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

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch, lowest_sync_loss

    scaler = torch.cuda.amp.GradScaler()
    while global_epoch < nepochs:
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {global_epoch}", ncols=100)
        model.train()
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            # if global_step == 2: break
            optimizer.zero_grad()

            # Move data to CUDA device
            x, mel, indiv_mels, gt  = x.to(device), mel.to(device), indiv_mels.to(device), gt.to(device)

            with torch.cuda.amp.autocast():
                g = model(indiv_mels, x)

                if hparams.syncnet_wt > 0.:
                    with torch.cuda.amp.autocast(enabled=False):
                        sync_loss = get_sync_loss(mel.float(), g.float(), syncnet)
                else:
                    sync_loss = 0.

                l1loss = recon_loss(g, gt)
                loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            prog_bar.set_description(f'Epoch {global_epoch} - L1: {running_l1_loss/(step + 1):.5f}, Sync Loss: {running_sync_loss/(step + 1):.5f}')
            prog_bar.refresh()

        torch.cuda.empty_cache()
        global_epoch += 1
        if global_epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

        with torch.no_grad():
            average_sync_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
            torch.cuda.empty_cache()
            if average_sync_loss < lowest_sync_loss:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, average_sync_loss)
                lowest_sync_loss = average_sync_loss

            if average_sync_loss < .75:
                hparams.set_hparam('syncnet_wt', 0.01)

        maintain_num_checkpoints(checkpoint_dir, args.max_no_ckpts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')
    parser.add_argument('-ds',"--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
    parser.add_argument('-ckpt_dir','--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('-syncnet_ckpt','--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator',required=True, type=str)
    parser.add_argument('-bs',"--batch_size", type=int, default=64,  help="the batch size")
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
    parser.add_argument('-nw','--num_workers', type=int, default=8, help="numer of workers for dataloader")
    parser.add_argument('-mnc','--max_no_ckpts', type=int, default=8, help="Save a max number of ckpts in the dir")
    parser.add_argument('-sf','--save_freq', type=int, default=5, help="Save per x epoch")
    args = parser.parse_args()

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')
    train_data_loader = data_utils.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_data_loader = data_utils.DataLoader( test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=args.checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)