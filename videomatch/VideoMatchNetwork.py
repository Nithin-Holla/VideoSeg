import torch
from torch import nn
import torch.nn.functional as F
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

from videomatch.MatchingLayer import MatchingLayer
from videomatch.SiameseNetwork import SiameseNetwork


class VideoMatchNetwork(nn.Module):

    def __init__(self, pretrained_model_dir, K=20, device='cuda'):
        super(VideoMatchNetwork, self).__init__()
        self.device = torch.device(device)
        self.sm_net = SiameseNetwork(pretrained_model_dir, device=self.device)
        self.match_layer = MatchingLayer(K=K, device=self.device)

    def forward(self, sample, data, query_label):
        q_feat, s_feat = self.sm_net(sample['query_img'].to(self.device), sample['search_img'].to(self.device))
        # label_prob = torch.zeros(1, len(data.label_colors), 224, 224).cuda()#.to(device)
        label_prob = torch.zeros(1, 2, 224, 224).to(self.device)

        for i, (obj, color) in enumerate(data.label_colors.items()):
            if obj != 'Tree':
                continue
            fg_score, bg_score = self.match_layer(query_label, color, q_feat, s_feat)
            fg_score_upsampled = F.interpolate(fg_score.unsqueeze(0).unsqueeze(0), size=data.shape, mode='bilinear')
            bg_score_upsampled = F.interpolate(bg_score.unsqueeze(0).unsqueeze(0), size=data.shape, mode='bilinear')
            # score_concat = torch.cat((bg_score.view(-1, 1), fg_score.view(-1, 1)), dim=1)
            score_concat = torch.cat((bg_score_upsampled.view(-1, 1), fg_score_upsampled.view(-1, 1)), dim=1)
            # label_prob[0, i, :, :] = F.softmax(score_concat, dim=1)[:, 0, :, :]
            label_prob = F.softmax(score_concat, dim=1)
            # print(obj)
            # print("FG ", label_prob.shape)
            # print("BG ", bg_score.shape)
            # plt.figure(obj)
            # plt.imshow(fg_score, cmap='gray')
            # plt.waitforbuttonpress()
            # plt.close()
            obj_color = color

        label_pred = torch.argmax(label_prob, dim=1)

        search_label = sample['search_label'].numpy()[0]
        search_mask = (search_label == obj_color).all(-1)
        search_sub_mask = resize(search_mask, (224, 224))

        plt.figure()
        plt.imshow(search_sub_mask, cmap='gray')

        i = 5
        j = 15
        search_pixel_feat = s_feat[0, :, i, j].view(1, -1)
        search_pixel_feat = F.normalize(search_pixel_feat, dim=1, p=2)
        frame_feat = q_feat.permute(0, 2, 3, 1).contiguous().view(-1, q_feat.shape[1])
        frame_feat = F.normalize(frame_feat, dim=1, p=2)
        heat = torch.mm(search_pixel_feat, torch.t(frame_feat)).view(29, 29).cpu()
        plt.figure()
        sns.heatmap(heat)

        return label_pred.view(224, 224)