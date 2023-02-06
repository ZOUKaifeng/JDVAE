import cv2
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
from VAE import semiVAE
from tqdm import tqdm
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image
from util import calculate_FID
trans = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
import argparse
import copy

def denormalize(data):
    return data*0.5 + 0.5

class CelebA(Dataset):
    def __init__(self, root, image_id, label, index_list, mode='train', transform=None):
        self.root = root

        self.image_id = image_id
        self.label = label
        self.mode = mode

        self.x = []
        self.y = []

        for i in index_list:
            if label[i] == -1:
                self.x.append(self.image_id[i])
            if label[i] == 1:
                self.y.append(self.image_id[i])



    def __len__(self):

        
        return len(self.y) 
    
    def __getitem__(self, index):

        i = index
     
        # img_x = Image.fromarray(self.x[i], 'RGB')
        # img_y = Image.fromarray(self.y[index], 'RGB')

        img_x = Image.open(self.root + self.x[i])
        img_y = Image.open(self.root + self.y[index])
        # x_  = Image.fromarray(img_x, 'RGB')
        # y_  = Image.fromarray(img_y, 'RGB')
        
        return trans(img_x), trans(img_y)



def compute_acc(classifier, data, y):
    y_pred = classifier(data)
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]

    return acc


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)



def training_Dz(Dz,net,  trainLoader, validLoader):
    optimizer = torch.optim.Adam(Dz.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss() 
    training_loss = []
    Dz = Dz.to(device)
    for _ in range(10):
        for img_0, img_1 in tqdm(trainLoader):
            img_0, img_1 = img_0.to(device), img_1.to(device) 
            bs = img_0.shape[0]

            x_label = torch.zeros((bs,)).to(device).long()
            y_label = torch.ones((bs,)).to(device).long()
            optimizer.zero_grad()
            with torch.no_grad():
                mean_x, _, logit_x = net.Q(img_0, x_label, False)
                mean_y, _, logit_y = net.Q(img_1, y_label, False)

            pred_y = Dz(mean_y.detach())

            pred_x = Dz(mean_x.detach())
            



            loss = 0.5 * criterion(pred_x, x_label.float()) + 0.5 * criterion(pred_y, y_label.float())
            loss.backward()
            training_loss.append(loss.item())
            optimizer.step()

        print("training_loss:", np.mean(training_loss))
        correct = 0
        total = 0
        best_acc = 0
        for img_0, img_1 in tqdm(validLoader):
            img_0, img_1 = img_0.to(device), img_1.to(device) 
            bs = img_0.shape[0]
            total += 2*bs
            x_label = torch.zeros((bs,)).to(device).long()
            y_label = torch.ones((bs,)).to(device).long()
            with torch.no_grad():
                mean_x, _, logit_x = net.Q(img_0, x_label, False)
                mean_y, _, logit_y = net.Q(img_1, y_label, False)

                pred_y = Dz(mean_y.detach())
                pred_x = Dz(mean_x.detach())

            correct += (pred_y>0.5).sum().item() + (pred_x<0.5).sum().item()
        print("valid acc:", correct/total)
        if best_acc < correct/total:
            best_acc = correct/total
            best_Dz = copy.deepcopy(Dz)

    return best_Dz


def visulization(sample, sample_gt, checkpoint, epoch):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(60,10))
    lis = [161, 192, 196, 242, 253, 310, 342, 353, 362, 377, 378,392,394,398,410]
    for i in range(200):
        fake_x, recon_x, fake_y, recon_y = sample[i]
        x, y = sample_gt[i]
        fake_x = np.transpose(fake_x, (1,2,0))
        recon_x = np.transpose(recon_x, (1,2,0))
        fake_y = np.transpose(fake_y, (1,2,0))
        recon_y = np.transpose(recon_y, (1,2,0))
        x =  np.transpose(x, (1,2,0))
        y =  np.transpose(y, (1,2,0))
        
        fake_x = denormalize(fake_x)
        recon_x = denormalize(recon_x)
        fake_y = denormalize(fake_y)
        recon_y = denormalize(recon_y)
        x = denormalize(x)
        y = denormalize(y)
        
      #  np.savez(checkpoint + str(i), x = x, y = y, fake_x = fake_x, fake_y = fake_y)
        plt.imshow(x)
        plt.axis('off')
        plt.subplot(1, 6, 1)
        plt.imshow(y)
        plt.axis('off')
        plt.subplot(1, 6, 2)
       
        plt.imshow(fake_y)
        plt.axis('off')
        plt.subplot(1, 6, 3)
        plt.imshow(recon_x)
        plt.axis('off')
        plt.subplot(1, 6, 4)

        plt.subplot(1, 6, 5)
        plt.imshow(fake_x)
        plt.axis('off')
        plt.subplot(1, 6, 6)
        plt.imshow(recon_y)
        plt.axis('off')
        plt.savefig(checkpoint + "swap_"+str(i))



def vis_inter(sample, x, y, checkpoint, epoch):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(50,100))
    bs = x.shape[0]
    lis = [161, 192, 196, 242, 253, 310, 342, 353, 362, 377, 378,392,394,398,410]
    for i in lis:

        fake_x, recon_x, fake_y, recon_y = sample[i]
        x_gt = x[i]
        y_gt = y[i]
        x_gt =  np.transpose(x_gt, (1,2,0))
        y_gt =  np.transpose(y_gt, (1,2,0))
        x_gt = denormalize(x_gt)
        y_gt = denormalize(y_gt)

        plt.subplot(bs, 10, i*10+1)
        plt.imshow(x_gt)

        for t in range(8):
            temp = sample[t]
            inter_img = temp[i] 

            inter_img = np.transpose(inter_img, (1,2,0))
            inter_img = denormalize(inter_img)

            plt.subplot(bs, 10, i*10+t+1+1)
            plt.imshow(inter_img)


        plt.subplot(bs, 10, i*10+10)
        plt.imshow(y_gt)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace = 0)
    plt.savefig(checkpoint + "interpolation_"+str(epoch))


def interpolation(attr, net, mean_x, y_label):
    one = attr/7
    y_attr = torch.zeros_like(attr).cuda()
    sample = []
    for i in range(8):
        
        fake_y = net.G(mean_x, y_attr, y_label,mean_x)
        y_attr = y_attr + one
        sample.append(fake_y.cpu().numpy())
    return sample

def main(args):
    checkpoint_weight = args.dir + args.epoch #"./results/batch_norm/" 
    batch_size = 16


    n = 0
    nRowsRead = 202599 # specify 'None' if want to read whole file
    # list_attr_celeba.csv has 202599 rows in reality, but we are only loading/previewing the first 1000 rows
    df1 = pd.read_csv('./data/list_attr_celeba.csv', delimiter=',', nrows = nRowsRead)
    df1.dataframeName = 'list_attr_celeba.csv'
    nRow, nCol = df1.shape
    print(f'There are {nRow} rows and {nCol} columns')

    path = "./data/align_5p/"

    image_id = df1["image_id"]
    label = df1["Eyeglasses"]

    test_list = list(np.arange(182637, 202599))
    image_id_test = image_id[test_list]
    label_test = label[test_list]
    test_dataset = CelebA(path, image_id_test, label_test, test_list, mode = 'test')
    test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers = 4)


    classifier = torch.load("../GAN/resnet50.pt").to(device)
    acc = 0


    latent_size = 100
    n_attr = 2
    net = semiVAE(latent_size, n_attr).to(device)

    net.Q.load_state_dict(torch.load(checkpoint_weight+"_Q.pt"))
    net.G.load_state_dict(torch.load(checkpoint_weight+"_G.pt"))

    Dz = nn.Sequential(nn.Linear(100, 1),
                                nn.Sigmoid()).cuda()

    train_list = list(np.arange(162771))
    valid_list = list(np.arange(162771, 182637))
    image_id_train = image_id[train_list]
    label_train = label[train_list]
    image_id_valid = image_id[valid_list]
    label_valid = label[valid_list]

    train_dataset = CelebA(path, image_id_train, label_train, train_list)
    valid_dataset = CelebA(path, image_id_valid, label_valid, valid_list, mode = 'valid')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4, drop_last = True)
    valid_dataloader = DataLoader(valid_dataset,  batch_size=batch_size, shuffle=False, num_workers = 4)

  #  Dz = training_Dz(Dz,net,  train_dataloader, valid_dataloader)

    sample = []
    sample_gt = []
    perceptual_score = []
    perceptual_fake = []

    with torch.no_grad():
        total  = 0
        correct = 0

        cls_correct = 0

        test_sample_x = []
        test_sample_y = []
        t = 0
        for img_x, img_y in tqdm(test_dataloader):
            img_x, img_y = img_x.to(device), img_y.to(device) 
            bs = img_x.shape[0]
           # net.set_input(img_0,img_1)
            total += 2*bs
            x_label = torch.zeros((bs,)).to(device).long()
            y_label = torch.ones((bs,)).to(device).long()

            mean_x, _, logit_x = net.Q(img_x, x_label, False)
            mean_y, _, logit_y = net.Q(img_y, y_label, False)


            pred_x = (torch.sigmoid(logit_x)>0.5).int().squeeze()
            cls_correct += (pred_x == x_label).sum()
            pred_y = (torch.sigmoid(logit_y)>0.5).int().squeeze()
            cls_correct += (pred_y == y_label).sum()

            recon_x = net.G(mean_x, x_label, mean_x)
            recon_y = net.G(mean_y, y_label, mean_y)

            fake_x = net.G(mean_y, x_label, mean_y)
            fake_y = net.G(mean_x, y_label, mean_x)

            pred_x = Dz(mean_x.detach())
            correct += ((pred_x>0.5).squeeze().float() == x_label.float()).sum().item()

            pred_y = Dz(mean_y.detach())
            correct += ((pred_y>0.5).squeeze().float() == y_label.float()).sum().item()


            acc_x = compute_acc(classifier,fake_x,x_label)
            acc_y = compute_acc(classifier,fake_y,y_label)

            d = 0.5* loss_fn_alex(recon_x, img_x) +  0.5*loss_fn_alex(recon_y, img_y)
            perceptual_score.append(d.mean().cpu().numpy())

            fake_d =  0.5* loss_fn_alex(fake_y, img_x) +  0.5*loss_fn_alex(fake_x, img_y)
            perceptual_fake.append(fake_d.mean().cpu().numpy())


            acc += ((acc_x + acc_y)/2).item()

            for i in range(bs):
                test_sample_x.append(fake_x[i].cpu().numpy())
                test_sample_y.append(fake_y[i].cpu().numpy())
                
                sample.append((fake_x[i].cpu().numpy(), recon_x[i].cpu().numpy(), fake_y[i].cpu().numpy(), recon_y[i].cpu().numpy()))
                sample_gt.append((img_x[i].cpu().numpy(),img_y[i].cpu().numpy()))



    print("classifier acc:",acc/len(test_dataloader))
    print("latent acc:",correct/total)
    print("perceptual_score", np.mean(perceptual_score))
    print("perceptual_fake", np.mean(perceptual_fake))

    print("clasification accuracy:", cls_correct/total)

    acc = acc/len(test_dataloader)

    train_set = np.load("../GAN/training_set.npz", allow_pickle = True)
    train_sample_x = train_set["train_sample_x"]
    train_sample_y = train_set["train_sample_y"]

    visulization(sample, sample_gt, args.dir, 10)
    fid_x = calculate_FID(train_sample_x, test_sample_x, device)
    fid_y = calculate_FID(train_sample_y, test_sample_y, device)

    print("acc", acc, "fid_x", fid_x, "fid_y", fid_y, 'mean fid', (fid_x+fid_y)/2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-w', '--epoch', type = str, default= "320", help='number of subject')
    parser.add_argument('-d', '--dir', type = str, default= "./results/token_128/", help='number of subject')

    args = parser.parse_args()

    main(args)

