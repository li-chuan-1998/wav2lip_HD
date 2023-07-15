import cv2, os, torch, torch.nn as nn, numpy as np
from os.path import join
from glob import glob

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

def _load(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, use_cuda=True):
    global global_step, global_epoch, global_lowest_eval_loss

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_lowest_eval_loss = checkpoint["global_lowest_eval_loss"]

    return model

def load_lip_sync_expert(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    for p in model.parameters():
        p.requires_grad = False
    return model

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

# def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, lowest_eval=None, save_optim=True):
#     pth_name = f"checkpoint_step{step:09d}_lowest_{lowest_eval:.4f}.pth" if lowest_eval else "checkpoint_step{:09d}.pth".format(step)
#     checkpoint_path = join(checkpoint_dir, pth_name)
#     optimizer_state = optimizer.state_dict() if save_optim else None
#     torch.save({
#         "state_dict": model.state_dict(),
#         "optimizer": optimizer_state,
#         "global_step": step,
#         "global_epoch": epoch,
#         "global_lowest_eval_loss": lowest_eval
#     }, checkpoint_path)
#     print("Saved checkpoint:", checkpoint_path)

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, lowest_eval=None, save_optim=True):
    if lowest_eval:
        pth_name = f"checkpoint_step{step:09d}_lowestEval_{lowest_eval:.4f}.pth"
    else:
        pth_name = f"checkpoint_step{step:09d}.pth"

    checkpoint_path = join(checkpoint_dir, pth_name)
    optimizer_state = optimizer.state_dict() if save_optim else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_lowest_eval_loss": lowest_eval
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def maintain_num_checkpoints(checkpoint_dir, max_num_ckpts, is_lowest_eval=False):
    pattern = f"{checkpoint_dir}/*_{'lowestEval' if is_lowest_eval else 'ckpt'}*.pth"
    files = glob.glob(pattern)
    if len(files) > max_num_ckpts:
        files.sort(key=os.path.getmtime)
        for file in files[:-max_num_ckpts]: # removing oldest files
            os.remove(file)

def eval_model(test_data_loader, device, model):
    losses = []
    model.eval()
    for step, (x, mel, y) in enumerate(test_data_loader):

        x, mel, y = x.to(device), mel.to(device), y.to(device)
        a, v = model(mel, x)
        loss = cosine_loss(a, v, y)

        losses.append(loss.item())

    return sum(losses) / len(losses)