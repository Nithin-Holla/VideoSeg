import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.transform import resize

from CamVidDataset import CamVidDataset
from SiameseNetwork import SiameseNetwork


if __name__ == '__main__':

    data_dir = 'F:\Projects\Honours\Data\CamVid'
    img_dir = data_dir + '\Raw'
    label_dir = data_dir + '\Labels'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = CamVidDataset(data_dir=data_dir, img_dir=img_dir, label_dir=label_dir)
    dataloader = DataLoader(dataset=data, batch_size=1, shuffle=True)
    model = SiameseNetwork().to(device)

    with torch.no_grad():
        sample = iter(dataloader).next()
        query_label = sample['query_label'].numpy()[0]
        # plt.figure(1)
        # plt.imshow(query_label)

        q_feat, s_feat = model(sample['query_img'].to(device), sample['search_img'].to(device))
        print(q_feat.shape, s_feat.shape)

        for obj in data.label_colors:
            mask = (query_label == data.label_colors[obj]).all(-1)
            sub_mask = resize(mask, q_feat.shape[2:])
            sub_mask = torch.ByteTensor(sub_mask).expand_as(q_feat)
            pass
            # plt.figure(obj)
            # plt.imshow(sub_mask, cmap='gray')
            # plt.waitforbuttonpress()
            # plt.close()

    # plt.figure(2)
    # plt.imshow(transforms.ToPILImage()(sample['search_label'].squeeze(0)))
    plt.show()