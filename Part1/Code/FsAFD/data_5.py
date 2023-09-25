import PIL.ImageOps
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import csv


class SiameseNetworkDataset(Dataset):

    def __init__(self, dataset_folder, file_name=''):
        self.images = dataset_folder
        self.file_name = file_name
        self.images_size = len(self.images)
        print('images_size', self.images_size)
        size_fake_images = 0
        size_true_images = 0
        for r in range(self.images_size):
            if self.images.imgs[r][1] == 0:
                size_fake_images += 1
            else:
                size_true_images += 1
        print(size_fake_images, size_true_images)

        list_pairs = []
        # 1 set  280 * 280 for real image
        for idx1 in range(size_fake_images, self.images_size, 1):  # loop for all the 280 real images
            img0_tuple = self.images.imgs[idx1]
            for idx2 in range(size_fake_images, self.images_size, 1):  # loop for all the 280 fake images
                if idx1 != idx2:  # for the first time only
                    img1_tuple = self.images.imgs[idx2]
                    list_pairs.append([img0_tuple[0], img1_tuple[0], img0_tuple[1], img1_tuple[1]])
        print('end real-real')

        # 2 set  280 * 1680 for real-fake image
        for idx1 in range(size_fake_images, self.images_size, 1):  # loop for all the 280 real images
            img0_tuple = self.images.imgs[idx1]
            for idx2 in range(0, size_fake_images, 1):  # loop for all the 1680 fake images
                img1_tuple = self.images.imgs[idx2]
                list_pairs.append([img0_tuple[0], img1_tuple[0], img0_tuple[1], img1_tuple[1]])
        print('end real-fake')

        # 3 set for fake-fake image
        for idx1 in range(0, size_fake_images, 1):  # loop for all the 280 real images
            img0_tuple = self.images.imgs[idx1]
            for idx2 in range(0, size_fake_images, 1):  # loop for all the 2680 fake images
                if idx1 != idx2:  # for the first time only
                    img1_tuple = self.images.imgs[idx2]
                    list_pairs.append([img0_tuple[0], img1_tuple[0], img0_tuple[1], img1_tuple[1]])

        print('end fake-fake')
        print('len(list_pairs', len(list_pairs))
        with open(self.file_name, "w", newline='') as file:
            write = csv.writer(file)
            write.writerows(list_pairs)

        self.pairs = list_pairs

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


class CsvDataset(Dataset):
    def __init__(self, csv_file, transform1=None, transform2=None, should_invert=True):
        with open(csv_file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            pairs_dataset = list(csv_reader)
        self.paths = pairs_dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.should_invert = should_invert

    def __getitem__(self, idx):
        img_name1, img_name2, label_0, label_1 = self.paths[idx]
        label_0 = torch.tensor(int(label_0))
        label_1 = torch.tensor(int(label_1))
        img_1 = Image.open(img_name1)
        img_1 = img_1.convert("L")

        if self.transform2 is not None:
            img_1_half = self.transform2(img_1)

        if self.should_invert:
            img_1 = PIL.ImageOps.invert(img_1)
        if self.transform1 is not None:
            img_1 = self.transform1(img_1)

        img_2 = Image.open(img_name2)
        img_2 = img_2.convert("L")

        if self.transform2 is not None:
            img_2_half = self.transform2(img_2)

        if self.should_invert:
            img_2 = PIL.ImageOps.invert(img_2)
        if self.transform1 is not None:
            img_2 = self.transform1(img_2)
        

        return img_1, img_1_half, img_2, img_2_half, label_0, label_1

    def __len__(self):
        return len(self.paths)
