import random
import PIL.ImageOps
import matplotlib
import statistics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
from model import Generator, Discriminator 
from loss import ContrastiveLoss, ssim, AdversarialLoss
import argparse
from data_5 import SiameseNetworkDataset, CsvDataset
import tensorflow as tf
from loss import ContrastiveLoss, ssim


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    args = parser.parse_args()
    batch_size = args.bs

    country_name = 'rus'

    test_csv_dir = './data/testing/testing_advNet_v3_' + country_name + '.csv'

    # Load data
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
    testing_dataset = CsvDataset(csv_file=test_csv_dir, transform1=transform1, transform2=transform2, should_invert=False)
    test_loader = DataLoader(testing_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(), batch_size=batch_size, pin_memory=True)
    print('test_loader', len(test_loader))

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    #################################
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    path = './checkpoint/siamese_net/module_3_advNet_v3_' + country_name + '.ckpt' 
    checkpoint = torch.load(path, map_location=device)
    dis_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['netD_state_dict'].items()}
    netD.load_state_dict(dis_loaded_state)
    # netD.load_state_dict(checkpoint['netD_state_dict'])
    # netD.load_state_dict(torch.load('./checkpoint/siamese_net/module_3_advNet_v2_dis.ckpt', map_location=device))
    gen_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['netG_state_dict'].items()}
    # netG.load_state_dict(checkpoint['netG_state_dict'])
    netG.load_state_dict(gen_loaded_state)
    netD.eval()
    netG.eval()
        
    # Loss
    criterion_contrastive = ContrastiveLoss()
    adversarial_loss = AdversarialLoss()
    l1_criterion = nn.L1Loss()
    loss_cross = torch.nn.CrossEntropyLoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)


    ################
    similar_images = []
    dissimilar_images = []
    get_corrects = 0.0
    accuracy_list = []
    
    testing_data_length = [15,30,45,60,75,90,105,120]   
    for testing_length in testing_data_length:
        accuracy_init_list = []
        for epoch in range(3):
            get_corrects = 0.0
            for i, data in enumerate(test_loader):
                # print(i)
                if i == testing_length - 1:
                    break
                # Prepare sample and target
                img1, img1_half, img2, img2_half, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
                img1, img1_half, img2, img2_half, label_image_1, label_image_2 = img1.to(device), img1_half.to(device), img2.to(
                    device), img2_half.to(device), label_image_1.to(device), label_image_2.to(device)

                latent_1, latent_2, decoder_op_1, decoder_op_2 = netG(img1, img2)
                real_vid_feat, y1o_softmax  = netD(latent_1)
                fake_vid_feat, y2o_softmax = netD(latent_2)   
                
                # get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_image_1, torch.max(y2o_softmax, 1)[1] == label_image_2))
            
                get_corrects += torch.max(y1o_softmax, 1)[1] == label_image_1
                get_corrects += torch.max(y2o_softmax, 1)[1] == label_image_2
                #print(get_corrects)
            variable_acc = get_corrects.item() / (testing_length * 2)
            accuracy_init_list.append(variable_acc)
        # print('Accuracy@' + str(testing_length) + ': ', str(variable_acc))
        print('Accuracy@' + str(testing_length) + ': ', str(statistics.mean(accuracy_init_list)))
        accuracy_list.append(statistics.mean(accuracy_init_list))
    print('accuracy_list', accuracy_list)
if __name__ == '__main__':
    main()
