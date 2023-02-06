
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import itertools
#from layers import 
from attr_vq import semiVAE
from tqdm import tqdm

import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image

trans = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def denormalize(data):
    return data*0.5 + 0.5

class CelebA(Dataset):
    def __init__(self, root, image_id, label, index_list, mode='train', transform=None):
        self.root = root

        self.image_id = image_id
        self.label = label
        self.mode = mode
        self.index_list = index_list
        self.label = label
        self.y = []




    def __len__(self):

        
        return len(self.index_list) 
    
    def __getitem__(self, index):
        i = self.index_list[index]
        img = Image.open(self.root + self.image_id[i])
        y = self.label[i]

        if y == -1:
            y = 0
        else:
            y = 1

        return trans(img), torch.tensor(y)

def train(data_loader, net):
    net.train()
    loss_G = []
    loss_D = []
    loss_kld = []
    
    idx = 0

    for img, y in tqdm(data_loader):
      
        img, y = img.to(device), y.to(device) 
        bs = img.shape[0]
        idx += 1
                
        results = net.optimize_parameters(img, y)
        
        loss_G.append(results["loss"].item())
        loss_D.append((results["gaussian"].item()))
        if results["kld"].item() != 0:
            loss_kld.append((results["kld"].item()))
   
    return np.mean(loss_G), np.mean(loss_D), np.mean(loss_kld)



def evaluate(data_loader, net, FID = False, classifier = None):
    net.eval()
    loss_G = []
    loss_D = []
    loss_kld = []
    sample = []
    sample_gt = []

    gt_x = []
    gt_y = []
    fake_x = []
    fake_y = []
    fid_y = 0
    fid_x = 0
    print("evaluating ...")
    recon_sample = []
    total = 0
    total_correct = 0
    total_loss = []
    with torch.no_grad():
        for img, y in tqdm(data_loader):
            
            img, y = img.to(device), y.to(device) 

            bs = img.shape[0]
            recon, fake, correct, vae_loss = net(img, y)

            total += bs
            total_correct += correct.item()

            total_loss.append(vae_loss.item())
            for i, l in enumerate(y):
                if l.item() == 0:

                    fake_y.append(fake[i].cpu().numpy())

                else:

                    fake_x.append(fake[i].cpu().numpy())
                if np.random.uniform()>0.7:

                    sample.append((recon[i].cpu().numpy(), fake[i].cpu().numpy()))
                    sample_gt.append(img[i].cpu().numpy())



    return  sample, sample_gt, total_correct/total, np.mean(total_loss)



def visulization(sample, sample_gt, checkpoint, epoch):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(50,100))
    for i in range(12):
        fake_x = sample[i][0]
        fake_y = sample[i][1]

        gt = sample_gt[i]
       
        fake_x = np.transpose(fake_x, (1,2,0))
        fake_y = np.transpose(fake_y, (1,2,0))

        gt = np.transpose(gt, (1,2,0))

        
        fake_x = denormalize(fake_x)
        fake_y = denormalize(fake_y)

        gt = denormalize(gt)

        
        plt.subplot(6, 6, i*3+1)
        plt.imshow(fake_x)
        plt.subplot(6, 6, i*3+2)
        plt.imshow(fake_y)
        plt.subplot(6, 6, 3+3*i)
        plt.imshow(gt)

    plt.savefig(checkpoint + "results_"+str(epoch))
    plt.close()
import sys
import os
 
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


def main(arg): 
    checkpoint = arg.checkpoint_dir # "./results/GAN_baseline/"
    start_epoch = 0
    checkpoint_weight =None# checkpoint + str(start_epoch)
    batch_size = arg.bs
    latent_size = arg.latent_dim
    n_attr = arg.attr_dim
    total_epoch = arg.num_epoch
    attr = arg.attr


    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(checkpoint + 'training.txt')


    nRowsRead = 202599 # specify 'None' if want to read whole file
    # list_attr_celeba.csv has 202599 rows in reality, but we are only loading/previewing the first 1000 rows
    df1 = pd.read_csv('../GAN/data/list_attr_celeba.csv', delimiter=',', nrows = nRowsRead)
    df1.dataframeName = 'list_attr_celeba.csv'
    nRow, nCol = df1.shape
    #print(f'There are {nRow} rows and {nCol} columns')

    path = "../GAN/data/align_5p/"

    image_id = df1["image_id"]
    label = df1[attr]

    print("saving weight and results in", checkpoint)

    #balance the dataset
    train_list = list(np.arange(162771))
    valid_list = list(np.arange(162771, 182637))
    import copy
    reference = copy.copy(train_list)
    count = 0
    count_1 = 0
    for i in  reference:
        if label[i] == -1:
            
            if np.random.uniform()>0.2:
                train_list.remove(i)
            else:
                count += 1
        else:
            count_1 += 1
    print("class 1:", count)
    print("class 2:", count_1)


    count = 0
    count_1 = 0
    reference = copy.copy(valid_list)
    for i in  reference:
        if label[i] == -1:
            
            if np.random.uniform()>0.2:
                valid_list.remove(i)
            else:
                count += 1
        else:
            count_1 += 1

    print("class 1:", count)
    print("class 2:", count_1)
    image_id_train = image_id[train_list]
    label_train = label[train_list]
    image_id_valid = image_id[valid_list]
    label_valid = label[valid_list]



    FID = True
    print("Loading dataset ...")
    train_dataset = CelebA(path, image_id_train, label_train, train_list)
    test_dataset = CelebA(path, image_id_valid, label_valid, valid_list, mode = 'valid')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, drop_last = True)
    test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers = 4)
    sample = []


    print("Building model ...")
    if checkpoint_weight is not None:
        net = semiVAE(latent_size, n_attr ).to(device)
        net.Q.load_state_dict(torch.load(checkpoint_weight+"_Q.pt"))
        net.G.load_state_dict(torch.load(checkpoint_weight+"_G.pt"))
        print("Loading model from", checkpoint_weight)
    else:
        print("initializing model....")


        net = semiVAE(latent_size, n_attr ).to(device)



    best_loss = 1000
    for i in range(start_epoch+1, total_epoch+1):
       
        if i > 70:
            for p in zip(net.optimizer_G.param_groups):
                p[0]['lr'] = 0.00001
    

        loss, gaussian, kld = train(train_dataloader, net)
        print("epoch {}, loss {}, gaussian {}, kld {}".format(i, loss,gaussian,  kld))
        
        
        if i%2 == 0:
            sample, sample_gt, acc, valid_loss= evaluate(test_dataloader, net, FID)   
            print("epoch {},  acc {}, valid_loss {}".format(i, acc, valid_loss ))

            visulization(sample, sample_gt, results_path, i)
            torch.save(net.Q.state_dict(), checkpoint + str(i)+"_Q.pt")
            torch.save(net.G.state_dict(), checkpoint + str(i)+"_G.pt")
            best_loss = valid_loss


        if len(sample) > 0:
            sample.clear()
            sample_gt.clear()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument( '--root', help='path of config file')
    parser.add_argument( '--bs',type = int, default= 32)
    parser.add_argument( '--lr',type = float, default= 1e-4)
    parser.add_argument( '--latent_dim',type = int, default= 100)
    parser.add_argument( '--attr_dim',type = int, default= 2)
    parser.add_argument( '--num_epoch',type = int, default= 200)

    parser.add_argument( '--checkpoint_dir',type = str, default=  "./results/attr_glasses/")
    parser.add_argument( '--attr',type = str, default=  "Eyeglasses")




    args = parser.parse_args()


    main(args)

