import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
from torchvision import transforms
import torchvision.datasets as dataset
from data_6 import SiameseNetworkDataset

training = True

country_list = ['alb','aze','esp','est','fin','grc','iva','rus','srb','svk']

for country_name in country_list:

    if training:
        train_dir = './data/training_20_samples_advNet/' + country_name
        train_csv_dir = './data/training_20_samples_advNet/advNet_' + country_name + '.csv'
        training_dataset = dataset.ImageFolder(root=train_dir)
        print('hi', len(training_dataset))
        training_pairs = SiameseNetworkDataset(dataset_folder=training_dataset, file_name=train_csv_dir)
        print(len(training_pairs))

    else:
        test_dir = './data/training_20_samples_advNet/' + country_name
        train_csv_dir = './data/training_20_samples_advNet/' + country_name + '.csv'
        training_dataset = dataset.ImageFolder(root=test_dir)
        print('hi', len(training_dataset))
        training_pairs = SiameseNetworkDataset(dataset_folder=training_dataset, file_name=train_csv_dir)
        print(len(training_pairs))