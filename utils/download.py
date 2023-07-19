import gdown
import os

id = "1Wdbg7jr10iJfXAkcZNS_mDM463SIrurg"
output_dir = "./ckp_ls/"
os.makedirs(output_dir, exist_ok=True)
gdown.download(id=id, output=output_dir, quiet=False)