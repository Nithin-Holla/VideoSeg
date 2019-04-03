import torch
from torch import nn
import torch.nn.functional as F
from skimage.morphology import binary_dilation, square
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

from videomatch.MatchingLayer import MatchingLayer
from videomatch.SiameseNetwork import SiameseNetwork


class VideoMatchNetwork(nn.Module):

    def __init__(self, pretrained_model_dir, n_classes, K=20, c1=0.95, c2=0.4, dc=50, device='cuda'):
        super(VideoMatchNetwork, self).__init__()
        self.device = torch.device(device)
        self.sm_net = SiameseNetwork(pretrained_model_dir, device=self.device)
        self.match_layer = MatchingLayer(n_classes=n_classes, K=K, device=self.device)
        self.c1 = c1
        self.c2 = c2
        self.dc = dc
        self.prev_pred = {}
        for i in range(n_classes):
            self.prev_pred[i] = torch.ones((224, 224)).cuda()

    def extrude(self, input_mask):
        coord = input_mask.nonzero()
        minr = max(0, torch.min(coord[:, 0]).item() - self.dc)
        maxr = min(224, torch.max(coord[:, 0]).item() + self.dc)
        minc = max(0, torch.min(coord[:, 1]).item() - self.dc)
        maxc = min(224, torch.max(coord[:, 1]).item() + self.dc)
        extruded_mask = torch.zeros(input_mask.shape).cuda()
        extruded_mask[minr:maxr, minc:maxc] = 1
        # for (r, c) in coord:
        #     r_seq = range(max(0, r - self.dc), min(224, r + self.dc))
        #     c_seq = range(max(0, c - self.dc), min(224, c + self.dc))
        #     extruded_mask[r_seq, c] = 1
        #     extruded_mask[r, c_seq] = 1
        return extruded_mask

    def forward(self, sample, data, query_label):
        q_feat, s_feat = self.sm_net(sample['query_img'].to(self.device), sample['search_img'].to(self.device))
        label_prob = torch.zeros((224 * 224, len(data.label_colors))).to(self.device)

        for i, (obj, color) in enumerate(data.label_colors.items()):
            if obj != 'Truck_Bus':
                continue
            fg_score, bg_score = self.match_layer(query_label, color, q_feat, s_feat, object_index=i)
            fg_score_upsampled = F.interpolate(fg_score.unsqueeze(0).unsqueeze(0), size=data.shape, mode='bilinear',
                                               align_corners=False)
            bg_score_upsampled = F.interpolate(bg_score.unsqueeze(0).unsqueeze(0), size=data.shape, mode='bilinear',
                                               align_corners=False)

            score_concat_sub = torch.cat((bg_score.view(-1, 1), fg_score.view(-1, 1)), dim=1)
            fg_map = F.softmax(score_concat_sub, dim=1)[:, 1]
            fg_update_mask = (fg_map > self.c1).view(fg_score.shape)
            if torch.sum(fg_update_mask.view(-1, 1)) != 0:
                print("Added %d new pixels to fg" % torch.sum(fg_update_mask.view(-1, 1)))
            new_fg_feat = s_feat.squeeze(0).permute(1, 2, 0)[fg_update_mask, :]
            self.match_layer.fg_model[i] = torch.cat((self.match_layer.fg_model[i], new_fg_feat), dim=0)

            score_concat = torch.cat((bg_score_upsampled.view(-1, 1), fg_score_upsampled.view(-1, 1)), dim=1)
            softmax_fg_score = F.softmax(score_concat, dim=1)[:, 1]

            mask_extrusion = self.extrude(self.prev_pred[i])
            bg_update_mask = (mask_extrusion == 0) * (softmax_fg_score.view(224, 224) > 0.5)
            bg_update_mask_sub = torch.ByteTensor(resize(bg_update_mask.cpu().numpy(), (29, 29))).cuda()
            if torch.sum(bg_update_mask_sub.view(-1, 1)) != 0:
                print("Added %d new pixels to bg" % torch.sum(bg_update_mask_sub.view(-1, 1)))
            new_bg_feat = s_feat.squeeze(0).permute(1, 2, 0)[bg_update_mask_sub, :]
            self.match_layer.bg_model[i] = torch.cat((self.match_layer.bg_model[i], new_bg_feat), dim=0)

            softmax_fg_score *= torch.cuda.FloatTensor(mask_extrusion.view(-1, 1)).squeeze(1).cuda()

            label_prob[:, i] = softmax_fg_score
            self.prev_pred[i] = torch.round(softmax_fg_score).view(224, 224).cuda()
            # label_prob = F.softmax(score_concat, dim=1)
            # print(obj)
            # print("FG ", label_prob.shape)
            # print("BG ", bg_score.shape)
            # plt.figure(obj)
            # plt.imshow(fg_score, cmap='gray')
            # plt.waitforbuttonpress()
            # plt.close()
            # obj_color = color

        label_pred = torch.argmax(label_prob, dim=1)
        bg_pixels = (label_prob < self.c2).all(dim=1)
        label_pred[bg_pixels] = 30  # Assign as void class

        # search_label = sample['search_label'].numpy()[0]
        # search_mask = (search_label == obj_color).all(-1)
        # search_sub_mask = resize(search_mask, (224, 224))
        #
        # plt.figure()
        # plt.imshow(search_sub_mask, cmap='gray')
        #
        # i = 5
        # j = 15
        # search_pixel_feat = s_feat[0, :, i, j].view(1, -1)
        # search_pixel_feat = F.normalize(search_pixel_feat, dim=1, p=2)
        # frame_feat = q_feat.permute(0, 2, 3, 1).contiguous().view(-1, q_feat.shape[1])
        # frame_feat = F.normalize(frame_feat, dim=1, p=2)
        # heat = torch.mm(search_pixel_feat, torch.t(frame_feat)).view(29, 29).cpu()
        # plt.figure()
        # sns.heatmap(heat)

        return label_pred.view(224, 224)
