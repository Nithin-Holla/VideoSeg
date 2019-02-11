import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.transform import resize
import torch.nn.functional as F

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
            fg_collection = q_feat[sub_mask].view(-1, q_feat.shape[1])
            bg_collection = q_feat[sub_mask == 0].view(-1, q_feat.shape[1])
            s_feat_normalized = F.normalize(s_feat.view(-1, q_feat.shape[1]))
            if fg_collection.nelement() != 0:
                fg_collection = F.normalize(fg_collection, dim=1, p=2)
                fg_similarity = torch.matmul(s_feat_normalized, torch.t(fg_collection))
            else:
                fg_similarity = torch.tensor(0)
            if bg_collection.nelement() != 0:
                bg_collection = F.normalize(bg_collection, dim=1, p=2)
                bg_similarity = torch.matmul(s_feat_normalized, torch.t(bg_collection))
            else:
                bg_similarity = torch.tensor(0)
            print(obj)
            print("FG ", fg_similarity.shape)
            print("BG ", bg_similarity.shape)
            # plt.figure(obj)
            # plt.imshow(sub_mask, cmap='gray')
            # plt.waitforbuttonpress()
            # plt.close()

    # plt.figure(2)
    # plt.imshow(transforms.ToPILImage()(sample['search_label'].squeeze(0)))
    plt.show()