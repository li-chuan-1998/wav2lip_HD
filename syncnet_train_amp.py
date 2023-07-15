import os, random, cv2, argparse
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from glob import glob
from torch.utils.tensorboard import SummaryWriter

from models import SyncNet_color as SyncNet
from hparams import hparams, get_image_list
from utils.helper import cosine_loss, save_checkpoint, load_checkpoint, maintain_num_checkpoints, eval_model
import audio

import torch
from torch import optim
from torch.utils import data as data_utils
import numpy as np


global_step = 0
global_epoch = 0
global_lowest_eval_loss = 0.4
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
device = torch.device("cuda" if use_cuda else "cpu")


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

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


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

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, nepochs=None):

    global global_step, global_epoch, global_lowest_eval_loss
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(args.log_dir)

    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {global_epoch}", ncols=100)
        model.train()
        for step, (x, mel, y) in prog_bar:
            optimizer.zero_grad()

            # Transform data to CUDA device
            x, mel, y = x.to(device), mel.to(device), y.to(device)
            
            with torch.cuda.amp.autocast():
                a, v = model(mel, x)
                with torch.cuda.amp.autocast(enabled=False):
                    loss = cosine_loss(a, v, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            running_loss += loss.item()

            writer.add_scalar('Loss/train', running_loss / (step + 1), global_step)
            prog_bar.set_description(f'Epoch {global_epoch} - Loss: {running_loss / (step + 1):.4f}')
            prog_bar.refresh()

        torch.cuda.empty_cache()
        global_epoch += 1

        # Save the model every x epochs
        if global_epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
        
        # Evaluate after every epoch
        with torch.no_grad():
            eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
            writer.add_scalar('Loss/eval', eval_loss, global_step)
            torch.cuda.empty_cache()
            maintain_num_checkpoints(checkpoint_dir, args.num_ckpts_save)

            # Check if this is the lowest evaluation loss
            if eval_loss < global_lowest_eval_loss:
                print(f"Epoch {global_epoch} Lowest Loss (eval): {eval_loss}")
                global_lowest_eval_loss = eval_loss
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, lowest_eval=global_lowest_eval_loss)
                maintain_num_checkpoints(checkpoint_dir, 1, is_lowest_eval=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--log_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
    parser.add_argument('--nepochs', type=int, default=200000000000000000, help="stop whenever eval loss > train loss for ~10 epochs")
    parser.add_argument('-bs','--batch_size', type=int, default=256, help="the batch size")
    parser.add_argument('-sf','--save_freq', type=int, default=20, help="Save per x epoch")
    parser.add_argument('-lr','--learning_rate', type=int, default=1e-4, help="learning rate for SyncNet")
    parser.add_argument('-ncs','--num_ckpts_save', type=int, default=5, help="Save a max number of ckpts in the dir")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir): 
        os.mkdir(args.checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    syncnet_batch_size = args.batch_size
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=syncnet_batch_size, shuffle=True,
        num_workers=8)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=syncnet_batch_size,
        num_workers=8)

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False, use_cuda=use_cuda)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=args.checkpoint_dir, nepochs=args.nepochs)
