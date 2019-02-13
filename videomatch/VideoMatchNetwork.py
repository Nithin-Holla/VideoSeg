import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from videomatch.MatchingLayer import MatchingLayer
from videomatch.SiameseNetwork import SiameseNetwork


class VideoMatchNetwork(nn.Module):

    def __init__(self, K=20):
        super(VideoMatchNetwork, self).__init__()
        self.sm_net = SiameseNetwork()#.to(device)
        self.match_layer = MatchingLayer(K=K)#.to(device)

    def forward(self, sample, data, query_label):
        # q_feat, s_feat = self.sm_net(sample['query_img'].to(device), sample['search_img'].to(device))
        q_feat, s_feat = self.sm_net(sample['query_img'].cuda(), sample['search_img'].cuda())
        label_prob = torch.zeros(1, len(data.label_colors), 224, 224).cuda()#.to(device)

        for i, (obj, color) in enumerate(data.label_colors.items()):
            fg_score, bg_score = self.match_layer(query_label, color, q_feat, s_feat)
            fg_score_upsampled = F.interpolate(fg_score.unsqueeze(0).unsqueeze(0), size=(224, 224))
            bg_score_upsampled = F.interpolate(bg_score.unsqueeze(0).unsqueeze(0), size=(224, 224))
            score_concat = torch.cat((fg_score_upsampled, bg_score_upsampled), dim=1)
            label_prob[0, i, :, :] = F.softmax(score_concat, dim=1)[:, 0, :, :]
            # print(obj)
            # print("FG ", label_prob.shape)
            # print("BG ", bg_score.shape)
            # plt.figure(obj)
            # plt.imshow(fg_score, cmap='gray')
            # plt.waitforbuttonpress()
            # plt.close()

        label_pred = torch.argmax(label_prob, dim=1)
        return label_pred