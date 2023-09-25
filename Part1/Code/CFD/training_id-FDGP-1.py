import argparse
import os
from GPUtil import showUtilization as gpu_usage
gpu_usage()
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import SiameseNetwork
from data_in import CsvDataset
from loss import ContrastiveLoss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from utils import AverageMeter
import csv

p_size = '32'

def _save_best_model(model, best_loss, epoch, country_name):
    # Save Model
    state = {
        'state_dict': model.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }
    if not os.path.isdir('./models_P' + p_size):
        os.makedirs('./models_P'+ p_size)
    torch.save(state, './models_P'+ p_size +'/data20_' + country_name + '.ckpt')

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=2, type=int, help='batch size')
    args = parser.parse_args()

    # train_set_dir for a country
    country_name = 'svk'
    print('-' *20, country_name)

    train_csv_dir = './data/training_P'+ p_size + '_20_samples/training_' + country_name + '.csv'

    train_me_where = "from_beginning"  # "from_middle"

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    model = SiameseNetwork().to(device)

    print('Model created.')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model.to(device))

    print('model and cuda mixing done')

    # Loss
    criterion_contrastive = ContrastiveLoss()
    loss_crossEntopy = torch.nn.CrossEntropyLoss()

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    batch_size = args.bs
    best_acc = 0.0


    # Load data
    transform = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform=transform, should_invert=False)
    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(),
                              batch_size=batch_size, pin_memory=True)
    dataset_sizes = len(train_loader.dataset)
    print(dataset_sizes)
    print("Total number of batches in train loader are :", len(train_loader))

    if train_me_where == "from_middle":
        checkpoint = torch.load('./models/' + country_name + '_net.ckpt')
        model.load_state_dict(checkpoint['state_dict'])

    # writer = SummaryWriter('./runs/real_siamese_net_running')

    print('-' * 10)

    for epoch in range(args.epochs):
        losses = AverageMeter()
        # Switch to train mode
        model.train()
        get_corrects = 0.0

        for i, data in enumerate(train_loader, 0):
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()

            # Prepare sample and target    
            img1, img2, label_1, label_2 = data
            img1, img2, label_1, label_2 = img1.to(device), img2.to(device), label_1.to(device), label_2.to(device) 
            label_pair = label_1[:] == label_2[:]  # label of pairs: 1 if the two images in the pair are
            # of the same class, 0 if the images belong to two different classes
            label_pair = label_pair.long()
            
            latent_1, class_op_1, y1o_softmax = model(img1)
            latent_2, class_op_2, y2o_softmax = model(img2)

            loss_cross_entropy_1 = loss_crossEntopy(class_op_1, label_1)
            loss_cross_entropy_2 = loss_crossEntopy(class_op_2, label_2)

            loss_enc = criterion_contrastive(latent_1, latent_2, label_pair)
            
            loss_total = loss_enc + loss_cross_entropy_1 + loss_cross_entropy_2

            get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_1, torch.max(y2o_softmax, 1)[1] == label_2))

            
            loss_total.backward()
            optimizer.step()

            # Update step
            total_batch_loss = loss_total
            losses.update(total_batch_loss.data.item(), img1.size(0))

            torch.cuda.empty_cache()
        
        variable_acc = get_corrects.item() / dataset_sizes

        # Log progress; print after every epochs into the console
        print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, losses.avg, variable_acc))

        if variable_acc > best_acc:
            print("Here the training accuracy got increased, hence printing")
            print('Current best epoch accuracy is {:.4f}'.format(variable_acc), 'previous best was {}'.format(best_acc))
            best_acc = variable_acc
            _save_best_model(model, best_acc, epoch, country_name)
            
        # save the losses avg in .csv file
        if not os.path.isdir('./losses_P'+ p_size):
            os.makedirs('./losses_P'+ p_size)
        with open('./losses_P'+ p_size + '/loss_avg_P'+ p_size + '_data20_' + country_name + '.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, losses.avg, variable_acc])
        
        my_lr_scheduler.step()


if __name__ == '__main__':
    main()
