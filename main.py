import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.transform import resize
import torch
import numpy as np
import warnings

from CamVidDataset import CamVidDataset
from videomatch.VideoMatchNetwork import VideoMatchNetwork

if __name__ == '__main__':

    data_dir = 'F:\Projects\Honours\Data\CamVid'
    img_dir = data_dir + '\Raw'
    label_dir = data_dir + '\Labels'
    output_path = 'C:\\Users\\Pavilion\\Desktop'
    pretrained_model_dir = 'F:\Projects\Honours\deeplabv2_resnet101_msc-vocaug-20000.pth'
    # pretrained_model_dir = 'F:\Projects\Honours\deeplabv2_resnet101_msc-cocostuff164k-100000.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = CamVidDataset(data_dir=data_dir, img_dir=img_dir, label_dir=label_dir)
    # dataloader = DataLoader(dataset=data, batch_size=1, shuffle=True)
    dataloader = DataLoader(dataset=data, batch_size=1, shuffle=False)
    vm_net = VideoMatchNetwork(n_classes=32, K=1, c1=0.9, c2=0.4, dc=20, device='cuda', pretrained_model_dir=pretrained_model_dir).to(device)

    vm_net.eval()

    with torch.no_grad():
        sample = iter(dataloader).next()

        # save_path = 'F:\Projects\Honours\Data\Results'
        # i = 1
        # for frame_pair, video_first_label in data.fetch_frame('0001TP'):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
        #         label_pred = vm_net(frame_pair, data, video_first_label)
        #     output = np.array(list(data.label_colors.values()))[label_pred.squeeze(0).cpu().numpy()]
        #     plt.figure(1)
        #     plt.imshow(output)
        #     # plt.savefig(save_path + "\\" + str(i) + ".png")
        #     i += 1
        #     print(i)
        #     plt.pause(3)
        # exit(0)

        i = 1
        for sample in iter(dataloader):
            if i == 5:
                break
            i += 1
        query_label = sample['query_label'].numpy()[0]
        with warnings.catch_warnings():
            label_pred = vm_net(sample, data, query_label)
        output = np.array(list(data.label_colors.values()))[label_pred.squeeze(0).cpu().numpy()]
        plt.figure()
        plt.imshow(output)
        # plt.imshow(transforms.ToPILImage()(sample['query_img'].squeeze(0)))
        # plt.figure()
        # plt.imshow(transforms.ToPILImage()(sample['search_img'].squeeze(0)))
        plt.figure()
        plt.imshow(query_label)
        # plt.figure()
        # plt.imshow(transforms.ToPILImage()(label_pred.detach().cpu()))
        # plt.imshow(label_pred.cpu().numpy(), cmap='gray')
        # plt.savefig(output_path + '\pred.png', bbox_inches='tight')
        plt.figure()
        plt.imshow(resize(sample['search_label'].numpy()[0], (224, 224)))
        # plt.savefig(output_path + '\expected.png', bbox_inches='tight')
        plt.show()
