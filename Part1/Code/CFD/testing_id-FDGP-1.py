import csv
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import SiameseNetwork, SiameseNetworkDataset

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_use_cuda else "cpu")
print("Let's use", torch.cuda.device_count(), "GPUs!")


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    args = parser.parse_args()
    batch_size = args.bs

    # test_set_dir for a country
    country = 'alb'
    test_set_dir = './data/testing_set/' + country + '/'

    # Load data
    folder_dataset = datasets.ImageFolder(root=test_set_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((300, 300)),
                                                                          transforms.ToTensor()]), should_invert=False)
    test_loader = DataLoader(siamese_dataset, shuffle=True, num_workers=1, batch_size=batch_size)

    # Lunch model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load('./models/' + country + '_net.ckpt', map_location=device), strict=False)
    model.eval()

    print('-' * 10)
    cnt_similar_pairs = 0
    cnt_dissimilar_pairs = 0
    similar_pairs = []
    dissimilar_pairs = []
    test_score = []
    test_lbl = []
    threshold = 1.0

    for i, data in enumerate(test_loader, 0):
        img0, img1, label = data
        test_lbl.append(int(label.item()))
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        output1, output2 = model(Variable(img0), Variable(img1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        sim = '{:.4f}'.format(euclidean_distance.item())
        test_score.append(float(sim))
        if label == 0:
            similar_pairs.append(float(sim))
            if float(sim) < threshold:
                cnt_similar_pairs += 1
        else:
            dissimilar_pairs.append(float(sim))
            if float(sim) < threshold:
                cnt_dissimilar_pairs += 1

    with open(country + ".csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(similar_pairs)
        writer.writerow(dissimilar_pairs)

    fpr_list = []
    tpr_list = []
    print(test_lbl, test_score)
    fpr, tpr, _ = roc_curve(test_lbl, test_score)
    for item in range(len(fpr)):
        fpr_list.append(fpr[item])
    for item in range(len(tpr)):
        tpr_list.append(tpr[item])
    file = open('ROC_' + country + '.csv', 'a')
    np.savetxt(file, fpr_list, fmt='%1.4f', delimiter=',', newline='\n')
    file.write("\n\n")
    np.savetxt(file, tpr_list, fmt='%1.4f', delimiter=',', newline='\n')

    print('-' * 10)

    similar_dis = plt.scatter([*range(0, len(similar_pairs), 1)], similar_pairs, marker='x', label="similar_pair")
    dissimilar_dis = plt.scatter([*range(0, len(dissimilar_pairs), 1)], dissimilar_pairs, marker='.',
                                 label="dissimilar_pair")
    plt.yticks(np.arange(0.0, max(similar_pairs) + 0.5, 0.5))
    plt.ylim(-0.1, max(dissimilar_pairs) + 0.5)
    plt.xlabel('pair no.')
    plt.ylabel('distance')
    plt.legend((similar_dis, dissimilar_dis), ('similar_pair', 'dissimilar_pair'))
    plt.title("Distances distribution for " + country)
    plt.savefig('distances_distribution_for_' + country + '.png')
    # plt.show()


if __name__ == '__main__':
    main()
