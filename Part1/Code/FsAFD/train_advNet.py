import gc
from operator import ne
from GPUtil import showUtilization as gpu_usage
import matplotlib.pyplot as plt
import statistics
import time
import argparse
import datetime
import os
import numpy as np
import torch
import sys
import csv
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import transforms
import torchvision.datasets as dataset
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from model import Generator, Discriminator 
from loss import ContrastiveLoss, ssim, AdversarialLoss
from data_5 import SiameseNetworkDataset, CsvDataset
from torch.optim import lr_scheduler
from utils import AverageMeter

gpu_usage()

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, help='initial learning rate')  # default=0.00001
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--beta1', default=0.5, type=float, help='hyperparam for Adam optimizers')

    args = parser.parse_args()
    
    country_name = 'alb'

    train_csv_dir = './data/training_20_samples/training_advNet_v3_' + country_name + '.csv'

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    ngpu = torch.cuda.device_count()

    # Create the Discriminator
    netD = Discriminator().to(device)
    print('Discriminator Model created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netD = nn.DataParallel(netD.to(device))
    # print(netD)


    # Create the Generator
    netG = Generator().to(device)
    print('Generator Model created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = nn.DataParallel(netG.to(device))
    # print(netG)
   

    print('model and cuda mixing done')

    # Training parameters
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    D_lr_scheduler = lr_scheduler.StepLR(optimizerD, step_size=20, gamma=0.1)
    G_lr_scheduler = lr_scheduler.StepLR(optimizerG, step_size=20, gamma=0.1)

    best_loss = 100.0
    best_acc = 0.0
    batch_size = args.bs

    # Load data
    transform = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform=transform, should_invert=False)
    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(),
                              batch_size=batch_size, pin_memory=True)
    dataset_sizes = len(train_loader.dataset)
    print(dataset_sizes)
    print("Total number of batches in train loader are :", len(train_loader))


    # Loss
    criterion_contrastive = ContrastiveLoss()
    l1_criterion = nn.L1Loss()
    loss_cross = torch.nn.CrossEntropyLoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
    adversarial_loss = AdversarialLoss()
    adversarial_loss = adversarial_loss.to(device)

    writer = SummaryWriter('./runs/real_siamese_net_running')


    
    real_imag_label = 1
    fake_imag_label = 0
    d_label_real_img = torch.cuda.LongTensor([1]*batch_size)
    d_label_fake_img = torch.cuda.LongTensor([0]*batch_size)

    # Start training...
    for epoch in range(args.epochs):
        # torch.cuda.empty_cache()

        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        # model.train()
        G_losses = []
        D_losses = []
        get_corrects = 0.0

        for i, data in enumerate(train_loader):

            # netD.train()
            # netG.train() 
            gen_loss = 0
            dis_loss = 0
            # with torch.enable_grad():   

           # Prepare sample and target    
            img1, img2, label_1, label_2 = data
            img1, img2, label_1, label_2 = img1.to(device), img2.to(device), label_1.to(device), label_2.to(device) 
            label_pair = label_1[:] == label_2[:]  # label of pairs: 1 if the two images in the pair are
            # of the same class, 0 if the images belong to two different classes
            label_pair = label_pair.long()
            
            latent_1 = netG(img1)
            latent_2 = netG(img2)


            ############################
            # Calculate the contrastive loss
            ############################
            # contrastive_loss_densNet = criterion_contrastive(latent_1, latent_2, label_pair)
          
            ############################
            # (1) Update D network: maximize log(D(x)))     # discriminator adversarial loss
            ###########################
            # Calculate loss on all-real batch
            # netD.zero_grad()
            optimizerD.zero_grad()
            real_vid_feat, y1o_softmax  = netD(latent_1)
            # h = torch.max(y1o_softmax, 1)[1]
            dis_real_loss = loss_cross(real_vid_feat, d_label_real_img)
            dis_real_loss.backward(retain_graph=True)


            # Calculate loss on all-fake batch
            fake_vid_feat, y2o_softmax = netD(latent_2.detach()) 
            # h2 = torch.max(y2o_softmax, 1)[1]       
            dis_fake_loss = loss_cross(fake_vid_feat, d_label_fake_img)
            dis_fake_loss.backward(retain_graph=True) 

            D_losses.append(dis_fake_loss.item())

            # Update D
            optimizerD.step()
                        
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            # generator adversarial loss
            ###########################
            
            # netG.zero_grad()
            optimizerG.zero_grad()
           
            gen_fake_feat, y2o_softmax_fake = netD(latent_2)
            gen_fake_loss = loss_cross(gen_fake_feat, d_label_real_img)
          
            gen_loss =  gen_fake_loss
            # gen_loss =  (gen_fake_loss + loss_image_total_2)/2
            G_losses.append(gen_loss.item())
            # gen_loss_total = gen_loss +  loss_image_total_1 + loss_image_total_2
            # loss_total = contrastive_loss_densNet + loss_image_total_1 + loss_image_total_2            
            gen_loss.backward()
            # Update G
            optimizerG.step()
            
            get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_1, torch.max(y2o_softmax, 1)[1] == label_2))
            print('get_corrects', get_corrects)

            # print('loss_total---',  dis_loss, gen_loss)

        variable_acc = get_corrects.item() / dataset_sizes

        # print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, losses.avg, variable_acc))
        print('Epoch: [{:.4f}] \t The dis_loss of this epoch is: {:.4f} \t The gen_loss of this epoch is: {:.4f}\t The accuracy of this epoch is: {:.4f} '.format(epoch, statistics.mean(D_losses), statistics.mean(G_losses), variable_acc))

        

        if not os.path.isdir('./models_v1' ):
            os.makedirs('./models_v1')
        if variable_acc > best_acc:
            print("Here the training accuracy got reduced, hence printing")
            print('Current best epoch accuracy is {:.4f}'.format(variable_acc), 'previous best was {}'.format(best_acc))
            best_acc = variable_acc

            torch.save({
                'netD_state_dict': netD.state_dict(),
                'netG_state_dict': netG.state_dict()},
                './models_v1/data20' + country_name  + '.ckpt')
            

        # save the losses avg in .csv file
        if not os.path.isdir('./losses_v1'):
            os.makedirs('./losses_v1')
        with open('./losses_v1/loss_avg_data20_' + country_name + '.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, statistics.mean(D_losses), statistics.mean(G_losses), variable_acc])
        
       
        D_lr_scheduler.step()
        G_lr_scheduler.step()



if __name__ == '__main__':
    main()