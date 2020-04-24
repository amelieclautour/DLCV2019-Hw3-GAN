import torch
import torch.cuda as tcuda
import torchvision.utils as tutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import math
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import itertools
from tqdm import tqdm
import os
from skimage import io, transform
import cv2
from dann_data import MNIST, SVHN
from uda_models import UDA_NeuralNetwork_Classifier, UDA_NeuralNetwork , Discriminator
batchSize = 256
learningRate = 0.0002
dSteps = 1
numIterations = 500
weightDecay = 2.5e-5
betas = (0.5, 0.999)
numberHiddenUnitsD = 500
grey_weights = [0.2989, 0.5870, 0.1140]



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# DATA

dataset_source_train = MNIST(csv_file="./hw3_data/digits/mnistm/train.csv", root_dir="./hw3_data/digits/mnistm/train",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataset_source_test =  MNIST(csv_file="./hw3_data/digits/mnistm/test.csv", root_dir="./hw3_data/digits/mnistm/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

loader_source_train = torch.utils.data.DataLoader(dataset=dataset_source_train, batch_size=batchSize, shuffle=True)
loader_source_test = torch.utils.data.DataLoader(dataset=dataset_source_test, batch_size=batchSize, shuffle=False)

dataset_target_train = SVHN(csv_file="./hw3_data/digits/svhn/train.csv", root_dir="./hw3_data/digits/svhn/train",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataset_target_test = SVHN(csv_file="./hw3_data/digits/svhn/test.csv", root_dir="./hw3_data/digits/svhn/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
loader_target_train = torch.utils.data.DataLoader(dataset=dataset_target_train, batch_size=batchSize, shuffle=True)
loader_target_test = torch.utils.data.DataLoader(dataset=dataset_target_test, batch_size=batchSize, shuffle=False)


# Model UDA

NeuralNetwork = UDA_NeuralNetwork()
NeuralNetwork.eval()
NeuralNetworkTarget = UDA_NeuralNetwork()

NeuralNetworkTarget.train()

classifier = UDA_NeuralNetwork_Classifier()


NeuralNetwork.to(device)
NeuralNetworkTarget.to(device)
classifier.to(device)

correct = 0
total = 0

for param in NeuralNetwork.parameters():
    param.requires_grad = False

D = Discriminator()
D.train()
NeuralNetworkTarget.train()
classifier.train()

OptimizerD = optim.Adam(D.parameters(),  lr=learningRate, betas = betas, eps=1e-09, weight_decay= weightDecay)
OptimizerTarget = optim.Adam(NeuralNetworkTarget.parameters(),  lr=learningRate, betas = betas, eps=1e-09, weight_decay= weightDecay)
criterion = torch.nn.CrossEntropyLoss()

# Creating Labels for D:
labels_source = torch.zeros(batchSize, 1).long().squeeze()
labels_target = torch.ones(batchSize, 1).long().squeeze()


D.to(device)
NeuralNetworkTarget.to(device)
NeuralNetwork.to(device)
labels_target = labels_target.to(device)
labels_source = labels_source.to(device)
criterion.to(device)

i = 0
maxTargetAcc = 60
numValidation = 500 # 500
num_epoch= int(math.ceil(float(numIterations) / float(min(len(loader_source_train), len(loader_target_train)))))
for currentEpoch in tqdm(range(num_epoch)):
    targetError = 0
    DError = 0
    for it, ((images_source, label_source), (targetImages, target_label)) in enumerate(tqdm(zip(loader_source_train, loader_target_train))):

        if images_source.size(0) != targetImages.size(0):
            continue

        images_source = images_source.to(device)
        targetImages = targetImages.to(device)

        if images_source.size(1) == 3:
            images_source = grey_weights[0] * images_source[:,0,:,:] + grey_weights[1] * images_source[:,1,:,:] + grey_weights[2] * images_source[:,2,:,:]
            images_source.unsqueeze_(1)

        if targetImages.size(1) == 3:
            targetImages = grey_weights[0] * targetImages[:,0,:,:] + grey_weights[1] * targetImages[:,1,:,:] + grey_weights[2] * targetImages[:,2,:,:]
            targetImages.unsqueeze_(1)

        # Training D:
        D.zero_grad()

        sourceFeaturesForD = NeuralNetwork(Variable(images_source))
        targetFeaturesForD = NeuralNetworkTarget(Variable(targetImages))

        pred_images_sourceForD = D(sourceFeaturesForD.detach())
        pred_TargetImagesForD = D(targetFeaturesForD.detach())
        pred_D = torch.cat((pred_images_sourceForD, pred_TargetImagesForD), 0)
        labelsForD = torch.cat((labels_source, labels_target), 0)

        DError = criterion(pred_D, Variable(labelsForD))
        DError.backward()

        OptimizerD.step()

        D.zero_grad()

        # Training Target:
        NeuralNetworkTarget.zero_grad()

        targetFeatures = NeuralNetworkTarget(Variable(targetImages))
        pred_TargetImages = D(targetFeatures)

        labels_targetT = Variable(1 - labels_target)

        TargetTargetError = criterion(pred_TargetImages, labels_targetT)
        TargetTargetError.backward()

        if (i > 5):
            OptimizerTarget.step()

        NeuralNetworkTarget.zero_grad()

        targetError = TargetTargetError
        i = i + 1
        if (i-1) % 1 == 0:
            print('Train Itr: {} \t D Loss: {:.6f} \t Target Loss: {:.6f} \n '.format(
            i, DError.data, targetError.data))
            torch.save(NeuralNetworkTarget, './UDA_NN_models/mnist_to_svhn/domainadaptation_model'+str(i)+'-'+str(currentEpoch)+'.pth')
        if (i-1) % 100 == 0:
            print('Train Itr: {} \t D Loss: {:.6f} \t Target Loss: {:.6f} \n '.format(
            i, DError.data, targetError.data))

            # if (i - 1) % 100 == 0:
            #
            #     correctT = 0
            #     totalT = 0
            #     correctD = 0
            #     totalD = 0
            #     j = 0
            #     for images, test_labels in loader_target_test:
            #         images, test_labels= images.to(device), test_labels.to(device)
            #
            #         test_labels = test_labels.long()
            #         test_labels[torch.eq(test_labels, 10)] = 0
            #         if images.size(1) == 3:
            #             images = grey_weights[0] * images[:, 0, :, :] + grey_weights[1] * images[:, 1, :, :] + \
            #                      grey_weights[2] * images[:, 2, :, :]
            #             images.unsqueeze_(1)
            #
            #         images = Variable(images)
            #         outputs = classifier(NeuralNetworkTarget(images))
            #         _, predicted = torch.max(outputs.data, 1)
            #
            #         totalT += test_labels.size(0)
            #
            #         correctT += (predicted == test_labels.squeeze(1)).sum()
            #         _, predictedD = torch.max(outputs.data, 1)
            #         totalD += predictedD.size(0)
            #         labelsT = torch.ones(predictedD.size()).long()
            #         labelsT = labelsT.to(device)
            #
            #         correctD += (predictedD == labelsT).sum()
            #         j += 1
            #         if j > numValidation:
            #             break;
            #
            #     currentAcc = 100 * correctT / totalT
            #
            #     if currentAcc > maxTargetAcc:
            #         torch.save(NeuralNetworkTarget, './UDA_NN_models/mnist_to_svhn/domainadaptation_bestmodel'+str(i)+'-'+str(currentEpoch)+'.pth')
            #         maxTargetAcc = currentAcc
            #
            #     print('\n\nAccuracy of target on target test images: %d %%' % (100 * correctT / totalT))
            #
            #     j = 0
            #     for images, test_labels in loader_source_test:
            #         images, test_labels = images.to(device), test_labels.to(device)
            #
            #         test_labels = test_labels.long()
            #         test_labels[torch.eq(test_labels, 10)] = 0
            #
            #         if images.size(1) == 3:
            #             images = grey_weights[0] * images[:, 0, :, :] + grey_weights[1] * images[:, 1, :, :] + \
            #                      grey_weights[2] * images[:, 2, :, :]
            #             images.unsqueeze_(1)
            #
            #         test_labels.squeeze_()
            #         images = Variable(images)
            #         outputsDFromSource = D(NeuralNetwork(images))
            #
            #         _, predictedD = torch.max(outputsDFromSource.data, 1)
            #         totalD += predictedD.size(0)
            #         labelsT = torch.zeros(predictedD.size()).long()
            #         labelsT = labelsT.to(device)
            #
            #         correctD += (predictedD == labelsT).sum()
            #         j += 1
            #         if j > numValidation:
            #             break;
            #
            #     print('Accuracy of D on validation images: %d %%' % (100 * correctD / totalD))

# Save the Trained Model
torch.save(NeuralNetworkTarget, './UDA_NN_models/mnist_to_svhn/uda_domainadaptation_model_MNISTtoSVHN'+str(i)+'-'+str(currentEpoch)+'.pth')
# print('Max target accuracy achieved is %d %%' %maxTargetAcc)
NeuralNetworkTarget.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
# correct = 0
# total = 0
# for images, labels in loader_source_test:
#     images, labels = images.to(device), labels.to(device)
#
#     labels = labels.long()
#     labels[torch.eq(labels, 10)] = 0
#
#     if images.size(1) == 3:
#         images = grey_weights[0] * images[:, 0, :, :] + grey_weights[1] * images[:, 1, :, :] + \
#                        grey_weights[2] * images[:, 2, :, :]
#         images.unsqueeze_(1)
#
#     labels.squeeze_()
#     images = Variable(images)
#     outputs = classifier(NeuralNetworkTarget(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Test Accuracy of the model on the source test images: %d %%' % (100 * correct / total))
#
# correct = 0
# total = 0
# for images, labels in loader_target_test:
#     if tcuda.is_available():
#         images, labels = images.to(device), labels.to(device)
#
#     labels = labels.long()
#     labels[torch.eq(labels, 10)] = 0
#
#     if images.size(1) == 3:
#         images= grey_weights[0] * images[:, 0, :, :] + grey_weights[1] * images[:, 1, :, :] + \
#                        grey_weights[2] * images[:, 2, :, :]
#         images.unsqueeze_(1)
#
#     images = Variable(images)
#     outputs = classifier(NeuralNetworkTarget(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Test Accuracy of the model on the target test images: %d %%' % (100 * correct / total))
#
# correct = 0
# total = 0
# for images, labels in loader_target_train:
#     images, labels = images.to(device), labels.to(device)
#
#     labels = labels.long()
#     labels[torch.eq(labels, 10)] = 0
#
#     if images.size(1) == 3:
#         images= grey_weights[0] * images[:, 0, :, :] + grey_weights[1] * images[:, 1, :, :] + \
#                        grey_weights[2] * images[:, 2, :, :]
#         images.unsqueeze_(1)
#
#     images = Variable(images)
#     outputs = classifier(NeuralNetworkTarget(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Test Accuracy of the model on the target train images: %d %%' % (100 * correct / total))