import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from dann_model import DANN_Neural_Network
from dann_data import MNIST, SVHN
import numpy as np
from tqdm import tqdm
import os
from skimage import io, transform
import cv2
import pandas as pd

cuda = True
cudnn.benchmark = True
lr = 0.001
batch_size = 128
image_size = 28
num_epoch = 100

manual_seed = 975
random.seed(manual_seed)
torch.manual_seed(manual_seed)

print("Random Seed: ", manual_seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

test_dataset =  SVHN(csv_file="./hw3_data/digits/svhn/test.csv", root_dir="./hw3_data/digits/svhn/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
print ('data loaded')

# Testing accuracy

dataset_name ='SVHN'
model_root = os.path.join('.','dann_NN_models/svhn_to_mnist')
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

# NN_test = torch.load(os.path.join(
#     model_root, 'lowerbound_model' + str(epoch) + '.pth'
# ))
NN_test = torch.load(os.path.join(
        model_root, 'upperbound_model180-0' + '.pth'
    ))
# NN_test = torch.load(os.path.join(
#     model_root, 'upper_model' + str(epoch) + '.pth'
# ))
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
    target_image, target_label = target_data
    target_image = target_image.to(device)
    target_label = target_label.to(device)
    batch_size = len(target_image)

    input_target_image = torch.FloatTensor(batch_size, 3, image_size, image_size)
    input_target_image = input_target_image.to(device)
    target_domain = torch.ones(batch_size)
    target_domain = target_domain.long()
    target_domain = target_domain.to(device)
    input_img = torch.FloatTensor(batch_size, 3, image_size, image_size).to(device)
    class_label = torch.LongTensor(batch_size).to(device)
    input_img.resize_as_(target_image).copy_(target_image)
    class_label.resize_as_(target_label.long()).copy_(target_label.long())
    class_output, _ ,_= NN_test(input=input_img, cst=cst_test)
    pred = class_output.data.max(1, keepdim=True)[1]
    n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
    predict += pred.squeeze().cpu().tolist()
    n_total += batch_size

    i += 1

accu = n_correct.data.numpy() * 1.0 / n_total
print (' accuracy of the %s dataset: %f' % ( dataset_name, accu))
img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
csv={'label':predict,'image_name':img_name}
df = pd.DataFrame(csv,columns=['image_name','label'])
#df.to_csv("./predictions/M2S_upperbound_predict.csv",index=0)
df.to_csv("./predictions/M2S_da28_predict.csv",index=0)
# df.to_csv("./predictions/M2S_upperbound_predict.csv",index=0)