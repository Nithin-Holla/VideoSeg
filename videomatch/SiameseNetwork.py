import torch
from torch import nn

from deeplab.deeplabv2 import DeepLabV2
from deeplab.msc import MSC


class SiameseNetwork(nn.Module):

    def __init__(self, pretrained_model_dir, device='cuda'):
        super(SiameseNetwork, self).__init__()
        # pretrained_model = models.resnet101(pretrained=True)
        # self.cnn = nn.Sequential(*list(pretrained_model.children())[:4])
        # self.cnn = nn.Sequential(*list(pretrained_model.children())[:3])
        base = DeepLabV2(
            n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        )
        pretrained_model = MSC(base=base, scales=[0.5, 0.75])
        pretrained_model.load_state_dict(torch.load(pretrained_model_dir))
        self.cnn = nn.Sequential(*list(pretrained_model.base.children())[:])

        self.device = torch.device(device)

    def forward(self, query_img, search_img):
        q_feat = self.cnn(query_img)
        s_feat = self.cnn(search_img)
        return q_feat, s_feat
