import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.manifold import TSNE
from dann_data import MNIST, SVHN
from dann_model import DANN_Neural_Network
import os

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mnistm_dataset = MNIST(root_dir="./hw3_data/digits/mnistm/test", csv_file="./hw3_data/digits/mnistm/test.csv", transform=T.ToTensor())
    svhn_dataset = SVHN(root_dir="./hw3_data/digits/svhn/test", csv_file="./hw3_data/digits/svhn/test.csv", transform=T.ToTensor())

    model_root = os.path.join('.','dann_NN_models/mnist_to_svhn')

    source_dataset = mnistm_dataset
    target_dataset = svhn_dataset
    source = "mnistm"
    target = "svhn"


    batch_size = 64
    source_loader= DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
    target_loader= DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    model = DANN_Neural_Network()
    model.to(device)
    model= torch.load(os.path.join(
        model_root, 'domainadaptation2_model28-0' + '.pth'
    ))
    model.eval()

    tsne = TSNE(n_components=2, init="pca")

    #X = np.array([]).reshape(0, 800)
    #Y_class = np.array([], dtype=np.int16).reshape(0,1)
    Y_class =[]
    #Y_domain = np.array([], dtype=np.int16)
    X=[]
    Y_domain=[]

    with torch.no_grad():
        steps = len(source_loader)
        for i, data in enumerate(source_loader):
            inputs, classes = data
            inputs= inputs.to(device)

            outputs = model.attr(inputs,kwargs=0).contiguous().view(inputs.size(0), -1).cpu().numpy()
            classes = classes.numpy()

            X.append(output)
            # Y_class = np.concatenate((Y_class, classes))
            # Y_domain = np.concatenate((Y_domain, np.array([0 for _ in range(inputs.size(0))], dtype=np.int16)))
            Y_class.append(classes)
            Y_domain = np.array([0 for _ in range(inputs.size(0))], dtype=np.int16)
            print("Source stpes: [{}/{}]".format(i, steps))

        print(X.shape)
        print(Y_class.shape)
        print(Y_domain.shape)

        steps = len(target_loader)
        for i, data in enumerate(target_loader):
            inputs, classes = data
            inputs= inputs.to(device)

            outputs = model.attr(inputs,kwargs=0).contiguous().view(inputs.size(0), -1).cpu().numpy()
            classes = classes.numpy()
            X.append(output)
            Y_class.append(classes)
            Y_domain = np.array([1 for _ in range(inputs.size(0))], dtype=np.int16)
            # X = np.vstack((X, outputs))
            # Y_class = np.concatenate((Y_class, classes))
            # Y_domain = np.concatenate((Y_domain, np.array([1 for _ in range(inputs.size(0))], dtype=np.int16)))

            print("Target stpes: [{}/{}]".format(i, steps))

        print(X.shape)
        print(Y_class.shape)
        print(Y_domain.shape)

 # X_tsne = tsne.fit_transform(X)
 # print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
 #
 # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
 # X_norm = (X_tsne - x_min) / (x_max - x_min)
    # Y_class = np.transpose(Y_class)
    # print(Y_class[0])
    # print(Y_class[1])
    # print('o')
    # class_color = []
    # domain_color = []
    color = np.array( ['r', 'g', 'b', 'k', 'gold', 'm', 'c', 'orange', 'cyan', 'pink'])
    # for i in range (len(Y_class)) :
    #     label = Y_class[i]
    #     class_color.append(color[label])
    #     domain_color.append(color[label])
    class_color = [color[label] for label in Y_class]
    domain_color = [color[label] for label in Y_domain]
    print("o")

    plt.figure(1, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=class_color, s=1)
    plt.savefig("./dann{}_{}_class.png".format(source, target))
    plt.close("all")


    plt.figure(2, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=domain_color, s=1)
    plt.savefig("./im/dann{}_{}_domain.png".format(source, target))
    plt.close("all")

if __name__ == "__main__":
    main(sys.argv)