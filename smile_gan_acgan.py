import torch
import numpy as np
import skimage.io
import skimage
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import random
import pickle
import os
from os import listdir
import pandas as pd
from tqdm import tqdm
import argparse
import gan_models
from acgantrue_models import Generator as g2

manualSeed = 246
random.seed(246)
torch.manual_seed(246)
ngpu=1
parser = argparse.ArgumentParser(description='DLCV hw3 pb3')

parser.add_argument('--resume1', type=str, default='model')
parser.add_argument('--resume2', type=str, default='model')
parser.add_argument('--save_dir', type=str, default='./acgan_images')
args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# use for random generation
fixed_class = np.hstack(( np.ones(10),np.zeros(10)))
fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).type(torch.FloatTensor)
fixed_noise = torch.randn(10, 100, 1, 1)
fixed_noise = torch.cat((fixed_noise,fixed_noise))
fixed_input = Variable(torch.cat((fixed_noise, fixed_class),1))

gen1 = gan_models.Generator(ngpu).to(device)
gen1.load_state_dict(torch.load(args.resume1))

fixed_noise_gan = torch.randn(32, 101, 1, 1, device=device)
fake = gen1(fixed_noise_gan).detach()

gen2 = g2()
gen2.load_state_dict(torch.load(args.resume2))

path = args.save_dir

torchvision.utils.save_image(fake.cpu().data, os.path.join(path,'fig1_2.jpg'))


fixed_img_output_acgan = gen2(fixed_input)
torchvision.utils.save_image(fixed_img_output_acgan.cpu().data, os.path.join(path,'fig2_2.jpg'),nrow=10)

