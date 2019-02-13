import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from CamVidDataset import CamVidDataset
from videomatch.VideoMatchNetwork import VideoMatchNetwork

if __name__ == '__main__':

    data_dir = 'F:\Projects\Honours\Data\CamVid'
    img_dir = data_dir + '\Raw'
    label_dir = data_dir + '\Labels'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = CamVidDataset(data_dir=data_dir, img_dir=img_dir, label_dir=label_dir)
    dataloader = DataLoader(dataset=data, batch_size=1, shuffle=True)
    vm_net = VideoMatchNetwork(K=20).to(device)

    with torch.no_grad():
        sample = iter(dataloader).next()
        query_label = sample['query_label'].numpy()[0]
        label_pred = vm_net(sample, data, query_label)
        output = np.array(list(data.label_colors.values()))[label_pred.squeeze(0)]
        plt.figure()
        plt.imshow(output)
        plt.show()
