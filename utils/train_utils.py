from os.path import join
import torch
import torch.nn as nn
from hparams import hparams

def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

syncnet_T = 5
recon_loss = nn.L1Loss()
def get_sync_loss(mel, g, syncnet, device="cuda"):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def eval_model(test_data_loader, device, model):
    sync_losses, recon_losses = [], []
    model.eval()
    for x, indiv_mels, mel, gt in test_data_loader:
        x, mel, indiv_mels, gt  = x.to(device), mel.to(device), indiv_mels.to(device), gt.to(device)
        g = model(indiv_mels, x)

        sync_loss = get_sync_loss(mel, g)
        l1loss = recon_loss(g, gt)

        sync_losses.append(sync_loss.item())
        recon_losses.append(l1loss.item())

    averaged_sync_loss = sum(sync_losses) / len(sync_losses)
    averaged_recon_loss = sum(recon_losses) / len(recon_losses)

    print('Evaluation - L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))
    return averaged_sync_loss

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, sync_loss=None):
    ckpt_name = f"checkpoint_step{global_step:09d}" + (f"_sync-loss:{sync_loss:.5f}.pth" if sync_loss else ".pth")
    checkpoint_path = join(checkpoint_dir, ckpt_name)
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "lowest_sync_loss": sync_loss,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path, use_cuda=True):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step, global_epoch, lowest_sync_loss

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]
        lowest_sync_loss = checkpoint["lowest_sync_loss"]

    return model