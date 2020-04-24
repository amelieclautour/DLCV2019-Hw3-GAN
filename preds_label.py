import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dann_model import DANN_Neural_Network
import numpy as np
from tqdm import tqdm
from skimage import io, transform
import cv2
import pandas as pd
import argparse
from os import listdir

class dataclass(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.join(root_dir)

        self.transform = transform
        self.file = sorted(listdir(self.root_dir))

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,self.file[index])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.file)

parser = argparse.ArgumentParser(description='DLCV hw3 pb3pb4')
parser.add_argument('--data_test_dir', type=str, default='./hw3_data/digits/mnistm/test', help="root path to test data directory")
parser.add_argument('--target', type=str, default='svhn', help="target to test")
parser.add_argument('--save_dir', type=str, default='./predictions/test_pred.csv')
parser.add_argument('--resume', type=str, default='model')

args = parser.parse_args()

dataroot = args.data_test_dir
predict_file = args.save_dir

cuda = True
cudnn.benchmark = True
lr = 0.001
batch_size = 128
image_size = 28
num_epoch = 100


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

test_dataset =  dataclass(root_dir=dataroot,transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))




# Testing accuracy

dataset_name = args.target
model_root = os.path.join('.','dann_NN_models/mnist_to_svhn')
cuda = True
cudnn.benchmark = True
cst_test = 0
dataset = test_dataset
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
)

""" test training """

NN_test = torch.load(args.resume)
NN_test = NN_test.eval()
NN_test = NN_test.to(device)
len_dataloader = len(dataloader)
target_data_iter = iter(dataloader)
i = 0
n_total = 0
n_correct = 0
predict = []
while i < len_dataloader:
    # test model using target data
    target_data = target_data_iter.next()
    target_image = target_data
    target_image = target_image.to(device)
    batch_size = len(target_image)

    input_target_image = torch.FloatTensor(batch_size, 3, image_size, image_size)
    input_target_image = input_target_image.to(device)
    target_domain = torch.ones(batch_size)
    target_domain = target_domain.long()
    target_domain = target_domain.to(device)
    input_img = torch.FloatTensor(batch_size, 3, image_size, image_size).to(device)
    class_label = torch.LongTensor(batch_size).to(device)
    input_img.resize_as_(target_image).copy_(target_image)
    class_output, _ ,_= NN_test(input=input_img, cst=cst_test)
    pred = class_output.data.max(1, keepdim=True)[1]
    predict += pred.squeeze().cpu().tolist()
    n_total += batch_size

    i += 1

img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
csv={'label':predict,'image_name':img_name}
df = pd.DataFrame(csv,columns=['image_name','label'])
df.to_csv(predict_file,index=0)
