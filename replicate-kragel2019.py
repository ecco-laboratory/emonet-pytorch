# %%
# imports up top, babies

import pandas as pd
import torch
import torchvision
from models import EmoNet
from myutils import Cowen2017Dataset, emonet_output_classes
from tqdm import tqdm

# %%
# Set paths to data

# Currently, this assumes that the videos are saved in subfolders by emotion category
video_path = '/home/data/eccolab/CowenKeltner/Videos_by_Category'
metadata_path = '.'

# %%
# Fire up the full EmoNet
emonet_torch = EmoNet()
# If you already have the state dict downloaded, use load_state_dict_from_path()
emonet_torch.load_state_dict_from_web()

# Turn off backward gradients for all runs of the model
# Right now we just be inferencing
for param in emonet_torch.parameters():
    param.requires_grad = False
# %%
# read those motherfuckin videos in

ck_torchdata_test = Cowen2017Dataset(
    root=video_path,
    annPath=metadata_path,
    censor=False,
    train=False,
    transform=torchvision.transforms.Resize((227, 227))
)

# Set batch_size here to 1 so it's just one video at a time
# BUT! Each video effectively acts as a batch of frames, as long as time is in the first dim
ck_torchloader_test = torch.utils.data.DataLoader(ck_torchdata_test, batch_size=1)
# %%
# Let's get predicting (full EmoNet)
emonet_torch.eval()

preds_all = {}

# For full EmoNet, only need to do it on the test data 
# because it's been trained on the training data (duh)
for vid, lab in tqdm(ck_torchloader_test):
    vid = vid.squeeze()
    pred = emonet_torch(vid)
    pred = pred.numpy()
    pred = pd.DataFrame(pred, columns=emonet_output_classes)
    pred.index.name = 'frame_num'
    preds_all[lab['id'][0]] = pred

# A pandas dataframe of each 20-class framewise prediction for your perusal!
preds_df = pd.concat(preds_all, names=['video_id', 'frame_num'])
# %%
