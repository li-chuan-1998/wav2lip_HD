import gdown
import os

id = "1dM2ZurB8bKBYepsQdMjKxmqiyQYlmPJh"
output = "./checkpoints/ckp_ls.pth"
os.makedirs(output, exist_ok=True)
gdown.download(id=id, output=output, quiet=False)