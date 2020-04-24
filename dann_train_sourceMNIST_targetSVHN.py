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




# Constants

source_dataset_name = 'MNIST'
target_dataset_name = 'SVHN'

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

# load data

sourceTrainDataset = MNIST(csv_file="./hw3_data/digits/mnistm/train.csv", root_dir="./hw3_data/digits/mnistm/train",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader_source = torch.utils.data.DataLoader(dataset=sourceTrainDataset, batch_size=batch_size, shuffle=True)

targetTrainDataset = SVHN(csv_file="./hw3_data/digits/svhn/train.csv", root_dir="./hw3_data/digits/svhn/train",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader_target = torch.utils.data.DataLoader(dataset=targetTrainDataset, batch_size=batch_size, shuffle=True)

test_dataset =  SVHN(csv_file="./hw3_data/digits/svhn/test.csv", root_dir="./hw3_data/digits/svhn/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
print ('data loaded')
model_root = os.path.join('.','dann_NN_models/mnist_to_svhn')

NeuralNetwork = torch.load(os.path.join(
        model_root, 'domainadaptation4_model1-0' + '.pth'
    ))
NeuralNetwork = NeuralNetwork.to(device)

# Optimizer and loss

optimizer = optim.Adam(NeuralNetwork.parameters(), lr=lr)

criterion_class = torch.nn.NLLLoss()
criterion_domain = torch.nn.NLLLoss()
criterion_class = criterion_class.to(device)
criterion_domain = criterion_domain.to(device)

for p in NeuralNetwork.parameters():
    p.requires_grad = True

# training
print('Start training...')

for epoch in range(num_epoch):

    NeuralNetwork =NeuralNetwork.train()
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    source_data_iter = iter(dataloader_source)
    target_data_iter = iter(dataloader_target)

    iters = 0
    while iters < len_dataloader:

        p = float(iters + epoch * len_dataloader) / num_epoch / len_dataloader
        cst = 2. / (1. + np.exp(-10 * p)) - 1

        # training with source data (optionnal)
        source_data = source_data_iter.next()
        source_image, source_target = source_data
        source_image = source_image.to(device)
        source_target = source_target.to(device)
        NeuralNetwork.zero_grad()
        batch_size = len(source_target)

        inputarget_image = torch.FloatTensor(batch_size, 3, image_size, image_size)
        inputarget_image = inputarget_image.to(device)
        classource_target = torch.LongTensor(batch_size)
        classource_target = classource_target.to(device)
        target_domain = torch.zeros(batch_size)
        target_domain = target_domain.long()
        target_domain = target_domain.to(device)
        inputarget_image.resize_as_(source_image).copy_(source_image)
        classource_target.resize_as_(source_target.long()).copy_(source_target.long())

        class_output, domain_output,_ = NeuralNetwork(input=inputarget_image, cst=cst)
        errorNN_source_target = criterion_class(class_output, classource_target.squeeze())
        errorNN_source_domain = criterion_domain(domain_output, target_domain)
        # training with target data (optionnal)
        target_data = target_data_iter.next()
        target_image, _ = target_data
        target_image = target_image.to(device)

        batch_size = len(target_image)

        input_target_image = torch.FloatTensor(batch_size, 3, image_size, image_size)
        input_target_image = input_target_image.to(device)
        target_domain = torch.ones(batch_size)
        target_domain = target_domain.long()
        target_domain = target_domain.to(device)
        input_target_image.resize_as_(target_image).copy_(target_image)
        _, domain_output,_ = NeuralNetwork(input=input_target_image, cst=cst)
        errorNN_target_domain = criterion_domain(domain_output, target_domain)
        err = errorNN_source_domain + errorNN_source_target  + errorNN_target_domain
        err.backward()
        optimizer.step()
        iters += 1

        print ('epoch: %d, [iter: %d / all %d], errorNN_source_target: %f, errorNN_s_domain: %f , errorNN_t_domain: %f' \
            % (epoch, iters, len_dataloader, errorNN_source_target.cpu().data.numpy(),
                errorNN_source_domain.cpu().data.numpy(), errorNN_target_domain.cpu().data.numpy()))
        if (iters % 10 == 0) or (iters==1) or (iters == 28) or ((epoch == num_epoch-1) and (iters == len(dataloader)-1)):
            torch.save(NeuralNetwork, './dann_NN_models/mnist_to_svhn/domainadaptation4_model'+str(iters)+'-'+str(epoch)+'.pth')
     #torch.save(NeuralNetwork, './dann_NN_models/mnist_to_svhn/lowerbound_model'+str(epoch)+'.pth')
    torch.save(NeuralNetwork, './dann_NN_models/mnist_to_svhn/domainadaptation_model'+str(epoch)+'.pth')
     # torch.save(NeuralNetwork, './dann_NN_models/mnist_to_svhn/upperbound_model'+str(epoch)+'.pth')

    # Testing accuracy

    dataset_name ='SVHN'
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

    # NN_test = torch.load(os.path.join(
    #     model_root, 'lowerbound_model' + str(epoch) + '.pth'
    # ))
    NN_test = torch.load(os.path.join(
         model_root, 'domainadaptation_model' + str(epoch) + '.pth'
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

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
    csv={'label':predict,'image_name':img_name}
    df = pd.DataFrame(csv,columns=['image_name','label'])
    #df.to_csv("./predictions/M2S_lowerbound_predict.csv",index=0)
    df.to_csv("./predictions/M2S_domainadaptation_predict.csv",index=0)
    # df.to_csv("./predictions/M2S_upperbound_predict.csv",index=0)


print ('done')