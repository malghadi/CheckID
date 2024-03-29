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
        # # 1 set  280 * 280 for real image
        # for idx1 in range(size_fake_images, self.images_size, 1):  # loop for all the 280 real images
        #     img0_tuple = self.images.imgs[idx1]
        #     for idx2 in range(size_fake_images, self.images_size, 1):  # loop for all the 280 fake images
        #         if idx1 != idx2:  # for the first time only
        #             img1_tuple = self.images.imgs[idx2]
        #             list_pairs.append([img0_tuple[0], img1_tuple[0], img0_tuple[1], img1_tuple[1]])
        # print('end real-real')

        # 2 set  280 * 1680 for real-fake image
        for idx1 in range(size_fake_images, self.images_size, 1):  # loop for all the 280 real images
            img0_tuple = self.images.imgs[idx1]
            for idx2 in range(0, size_fake_images, 1):  # loop for all the 1680 fake images
                img1_tuple = self.images.imgs[idx2]
                list_pairs.append([img0_tuple[0], img1_tuple[0], img0_tuple[1], img1_tuple[1]])
        print('end real-fake')

        # # 3 set for fake-fake image
        # for idx1 in range(0, size_fake_images, 1):  # loop for all the 280 real images
        #     img0_tuple = self.images.imgs[idx1]
        #     for idx2 in range(0, size_fake_images, 1):  # loop for all the 2680 fake images
        #         if idx1 != idx2:  # for the first time only
        #             img1_tuple = self.images.imgs[idx2]
        #             list_pairs.append([img0_tuple[0], img1_tuple[0], img0_tuple[1], img1_tuple[1]])

        # print('end fake-fake')
        print('len(list_pairs', len(list_pairs))
        with open(self.file_name, "w", newline='') as file:
            write = csv.writer(file)
            write.writerows(list_pairs)

        self.pairs = list_pairs

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)