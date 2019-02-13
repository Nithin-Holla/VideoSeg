from torch import nn
from torchvision import models


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        pretrained_model = models.resnet101(pretrained=True)
        self.cnn = nn.Sequential(*list(pretrained_model.children())[:4])

    def forward(self, query_img, search_img):
        q_feat = self.cnn(query_img)
        s_feat = self.cnn(search_img)
        return q_feat, s_feat
