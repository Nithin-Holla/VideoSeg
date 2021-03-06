from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import pandas as pd
from collections import defaultdict


class CamVidDataset(Dataset):

    def __init__(self, data_dir, img_dir, label_dir, shape=(224, 224)):
        super(CamVidDataset, self).__init__()
        self.data_dir = img_dir
        self.image_paths = []
        self.label_paths = []
        self.video_frames = defaultdict(list)
        self.video_frame_label = defaultdict(str)
        self.shape = shape
        self.transform = transforms.Compose([transforms.Resize(shape),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        for file_name in os.listdir(img_dir):
            if file_name.endswith('.png'):
                self.image_paths.append(os.path.join(img_dir, file_name))
                self.label_paths.append(os.path.join(label_dir, '_L.'.join(file_name.rsplit('.'))))
                video_name = file_name.split('.')[0].split('_')[0]
                self.video_frames[video_name].append(os.path.join(img_dir, file_name))
                if self.video_frame_label[video_name] == "":
                    self.video_frame_label[video_name] = os.path.join(label_dir, '_L.'.join(file_name.rsplit('.')))

        label_colors_df = pd.read_table(os.path.join(data_dir, 'label_colors.txt'), sep='\s+',
                                        names=['R', 'G', 'B', 'Class'])
        label_colors_df.index = label_colors_df['Class']
        label_colors_df = label_colors_df.drop('Class', axis=1)
        self.label_colors = label_colors_df.T.to_dict(orient='list')

    def __getitem__(self, index):
        query_img_path = self.image_paths[index]
        print(query_img_path)
        query_label_path = self.label_paths[index]
        query_img_name = query_img_path.split(os.sep)[-1].split('.')[0].split('_')[0]
        query_img = Image.open(query_img_path)
        query_label = Image.open(query_label_path)
        while True:
            rand_index = index + 1
            # if rand_index != index and 0 <= rand_index < self.__len__():
            if 0 <= rand_index < self.__len__():
                rand_img_path = self.image_paths[rand_index]
                rand_img_name = rand_img_path.split(os.sep)[-1].split('.')[0].split('_')[0]
                # if rand_img_name == query_img_name:
                if True:
                    rand_label_path = self.label_paths[rand_index]
                    break
        search_img = Image.open(rand_img_path)
        search_label = Image.open(rand_label_path)
        data = {'query_img': self.transform(query_img), 'query_label': np.asarray(query_label),
                'search_img': self.transform(search_img), 'search_label': np.asarray(search_label)}
        return data

    def __len__(self):
        return len(self.image_paths)

    def fetch_frame(self, video_name):
        first_frame = Image.open(self.video_frames[video_name][0])
        first_frame_label = Image.open(self.video_frame_label[video_name])
        for frame_path in self.video_frames[video_name]:
            print(frame_path)
            frame = Image.open(frame_path)
            frame_pair = {'query_img': self.transform(first_frame).unsqueeze(0), 'search_img': self.transform(frame).unsqueeze(0)}
            yield frame_pair, np.asarray(first_frame_label)
