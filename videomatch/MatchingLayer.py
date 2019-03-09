import torch
from torch import nn
from skimage.transform import resize
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MatchingLayer(nn.Module):

    def __init__(self, K=20, device='cuda'):
        super(MatchingLayer, self).__init__()
        self.K = K
        self.device = torch.device(device)

    def forward(self, query_label, color, q_feat, s_feat):
        mask = (query_label == color).all(-1)
        # plt.figure()
        # plt.imshow(mask, cmap='gray')
        sub_mask = resize(mask, q_feat.shape[2:])
        # plt.figure()
        # plt.imshow(sub_mask, cmap='gray')
        # output_path = 'C:\\Users\\Pavilion\\Desktop'
        # plt.savefig(output_path + '\submask.png', bbox_inches='tight')
        # sub_mask = torch.ByteTensor(sub_mask).expand_as(q_feat).cuda()
        # fg_collection = q_feat[sub_mask].view(-1, q_feat.shape[1])
        # bg_collection = q_feat[sub_mask == 0].view(-1, q_feat.shape[1])
        sub_mask = torch.tensor(sub_mask, dtype=torch.uint8).to(self.device)
        fg_collection = q_feat.squeeze(0).permute(1, 2, 0)[sub_mask, :]
        bg_collection = q_feat.squeeze(0).permute(1, 2, 0)[sub_mask == 0, :]
        s2 = s_feat.permute(0, 2, 3, 1).contiguous().view(-1, q_feat.shape[1])
        s_feat_normalized = F.normalize(s2, dim=1, p=2)

        if fg_collection.nelement() != 0:
            fg_collection = F.normalize(fg_collection, dim=1, p=2)
            fg_similarity = torch.mm(s_feat_normalized, torch.t(fg_collection))
            fg_similarity = torch.sort(fg_similarity, dim=1, descending=True)[0][:, 0:self.K]
            fg_score = torch.mean(fg_similarity, dim=1).view(s_feat.shape[2:])
        else:
            fg_score = torch.zeros(s_feat.shape[2:]).to(self.device)
        if bg_collection.nelement() != 0:
            bg_collection = F.normalize(bg_collection, dim=1, p=2)
            bg_similarity = torch.mm(s_feat_normalized, torch.t(bg_collection))
            bg_similarity = torch.sort(bg_similarity, dim=1, descending=True)[0][:, 0:self.K]
            bg_score = torch.mean(bg_similarity, dim=1).view(s_feat.shape[2:])
        else:
            bg_score = torch.zeros(s_feat.shape[2:]).to(self.device)
        return fg_score, bg_score