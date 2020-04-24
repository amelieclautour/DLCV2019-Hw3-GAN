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
from acgantrue_models import Generator, Discriminator

# DATA : importing data previously saved as np data
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

training_images_face = np.load("all_img.npy")
print('data load')
train_attr = pd.read_csv("./hw3_data/face/train.csv")

# Picking smiling as the attribute
smiling_attr = np.hstack(np.repeat(np.array(train_attr["Smiling"]),1))

face_images = torch.from_numpy(training_images_face).type(torch.FloatTensor)
face_class = torch.from_numpy(smiling_attr).type(torch.FloatTensor).view(-1,1,1,1)

manualSeed = 527
random.seed(527)
torch.manual_seed(527)
print("Random Seed: ", manualSeed)
# use for random generation
fixed_class = np.hstack(( np.ones(10),np.zeros(10)))
fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).type(torch.FloatTensor)
fixed_noise = torch.randn(10, 100, 1, 1)
fixed_noise = torch.cat((fixed_noise,fixed_noise))
fixed_input = Variable(torch.cat((fixed_noise, fixed_class),1)).to(device)

# loss functions
criterion_image = nn.BCELoss()
criterion_class = nn.BCELoss()

# training
latent_size = 100
batch_size = 64
gen = Generator()
dis = Discriminator()
gen.to(device)
dis.to(device)

# setup optimizer
beta_1 = 0.5
optimizerG = optim.Adam(gen.parameters(), lr=0.0002, betas=(beta_1,0.999))
optimizerD = optim.Adam(dis.parameters(), lr=0.0002, betas=(beta_1,0.999))


dis_loss_list = []
gen_loss_list = []
D_fake_acc_list = []
D_real_acc_list = []

D_fake_class_list = []
D_real_class_list = []

print ('start training')
for epoch in tqdm(range(200)):
    print("Epoch:", epoch+1)
    epoch_dis_loss = 0.0
    epoch_gen_loss = 0.0
    D_fake_acc = 0.0
    D_real_acc = 0.0
    D_fake_class = 0.0
    D_real_class = 0.0
    total_length = len(training_images_face)
    # shuffle
    perm_index = torch.randperm(total_length)
    training_images_face_sfl = face_images[perm_index]
    train_class_sfl = face_class[perm_index]

    if (epoch+1) == 15:
        optimizerG.param_groups[0]['lr'] /= 2
        optimizerD.param_groups[0]['lr'] /= 2
        print("learning rate change!")

    for index in tqdm(range(0,total_length ,batch_size)):
        for _ in range(1):
            # zero the parameter gradients
            dis.zero_grad()
            input_X = training_images_face_sfl[index:index+batch_size]
            intput_class = train_class_sfl[index:index+batch_size]

            # train with all real batch
            real_image = Variable(input_X.to(device))
            real_class = Variable(intput_class.to(device))
            real_label = Variable(torch.ones((batch_size))).to(device)
            dis_ouput, aux_output = dis(real_image)
            D_real_dis_loss = criterion_image(dis_ouput, real_label.view(batch_size,1))
            D_real_aux_loss = criterion_class(aux_output, real_class.view(batch_size,1))
            D_real_acc += np.mean(((dis_ouput > 0.5).cpu().data.numpy() == real_label.cpu().data.numpy()))
            D_real_loss = (D_real_dis_loss + D_real_aux_loss)/2
            print(D_real_aux_loss.data)
            D_real_class += D_real_aux_loss.data

            # train with all fake batch
            noise = torch.randn(batch_size, 100, 1, 1)
            fake_class = torch.from_numpy(np.random.randint(2,size=batch_size)).view(batch_size,1,1,1)
            intput_vector =Variable(torch.cat((noise,fake_class.type(torch.FloatTensor)),1)).to(device)

            fake_label = Variable(torch.zeros((batch_size))).to(device)
            fake_class = Variable(fake_class.type(torch.FloatTensor)).to(device)

            fake_image = gen(intput_vector)
            dis_output, aux_output = dis(fake_image.detach())
            D_fake_dis_loss = criterion_image(dis_output, fake_label.view(batch_size,1))
            D_fake_aux_loss = criterion_class(aux_output, fake_class.view(batch_size,1))
            D_fake_loss = (D_fake_dis_loss + D_fake_aux_loss)/2
            D_fake_acc += np.mean(((dis_output > 0.5).cpu().data.numpy() == fake_label.cpu().data.numpy()))
            D_fake_class += D_fake_aux_loss.data
            # update D
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            epoch_dis_loss+=(D_train_loss.data)

            optimizerD.step()

        # train generator
        for _ in range(2):
            gen.zero_grad()
            # generate fake image
            noise = torch.randn(batch_size, 100, 1, 1)
            fake_class = torch.from_numpy(np.random.randint(2,size=batch_size)).view(batch_size,1,1,1)

            intput_vector = Variable(torch.cat((noise,fake_class.type(torch.FloatTensor)),1)).to(device)

            fake_class = Variable(fake_class.type(torch.FloatTensor)).to(device)
            fake_label_for_gen = Variable(torch.ones((batch_size))).to(device)

            fake_image = gen(intput_vector)
            dis_output, aux_output = dis(fake_image)
            gen_dis_loss = criterion_image(dis_output, fake_label_for_gen.view(batch_size,1))
            gen_aux_loss = criterion_class(aux_output, fake_class.view(batch_size,1))
            gen_train_loss = gen_dis_loss + gen_aux_loss
            gen_train_loss.backward()
            optimizerG.step()
        epoch_gen_loss += (gen_train_loss.data)
        print(index)
    print("training D Loss:",epoch_dis_loss/(total_length))
    print("training gen Loss:", epoch_gen_loss/(total_length))
    dis_loss_list.append(epoch_dis_loss/(total_length))
    gen_loss_list.append(epoch_gen_loss/(total_length))

    print("D_real_dis_acc:", D_real_acc/(total_length/batch_size))
    print("D_fake_dis_acc:", D_fake_acc/(total_length/batch_size))
    print("D_real_aux_loss:", D_real_class/(total_length/batch_size))
    print("D_fake_aux_loss:", D_fake_class/(total_length/batch_size))
    D_real_acc_list.append(D_real_acc/(total_length/batch_size))
    D_fake_acc_list.append(D_fake_acc/(total_length/batch_size))
    D_real_class_list.append(D_real_class/(total_length/batch_size))
    D_fake_class_list.append(D_fake_class/(total_length/batch_size))
    # evaluation
    gen.eval()
    fixed_img_output = gen(fixed_input)
    gen.train()
    torchvision.utils.save_image(fixed_img_output.cpu().data, './acgan_image/fakesmilingdouble_'+str(epoch+1)+'.jpg',nrow=10)
    torch.save(gen.state_dict(), "./acgan_model/gen/ACgen2_model"+str(epoch+1)+".pth")
torch.save(gen.state_dict(), "./acgan_model/gen/ACgen2_model_last.pth")