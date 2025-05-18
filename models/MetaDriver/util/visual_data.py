import os
import os.path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

from .get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):
    assert split in [0, 1, 2, 3]

    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2,
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Checking data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')

        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])

        label_class = [int(line_split[2])]

        item = (image_name, label_name, label_class)
        if not os.path.exists(image_name):
            raise FileNotFoundError(f"Error: '{image_name}' is not existing!!!")

        if label_class[0] in sub_list:
            image_label_list.append(item)
            sub_class_file_list[label_class[0]].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list


class SemData(Dataset):
    def __init__(self, split=0, shot=1, data_root=None, base_data_root=None, data_list=None, data_type=None,
                 test_base=False, data_set=None,
                 use_split_coco=False, \
                 transform=None, transform_tri=None, mode='train', ann_type='mask', \
                 ft_transform=None, ft_aug_size=None, \
                 ms_transform=None):

        assert mode in ['train', 'val', 'demo', 'finetune', 'test']
        assert data_type in ['metadada', 'metapsad']
        if mode == 'finetune':
            assert ft_transform is not None
            assert ft_aug_size is not None

        # if data_set == 'pascal':
        #     self.num_classes = 20
        # elif data_set == 'coco':
        #     self.num_classes = 80

        self.mode = mode
        self.split = split
        self.shot = shot
        self.data_root = data_root
        self.base_data_root = base_data_root
        self.ann_type = ann_type

        if data_type == 'metadada':
            self.class_list = list(range(1, 21))
            if self.split == 3:
                self.sub_list = list(range(14, 53))
                self.sub_val_list = list(range(1, 14))
            elif self.split == 2:
                self.sub_list = list(range(1, 14)) + list(range(27, 53))
                self.sub_val_list = list(range(14, 27))
            elif self.split == 1:
                self.sub_list = list(range(1, 27)) + list(range(40, 53))
                self.sub_val_list = list(range(27, 40))
            elif self.split == 0:
                self.sub_list = list(range(1, 40))
                self.sub_val_list = list(range(40, 53))

        if data_type == 'metapsad':
            self.class_list = list(range(2, 6))
            if self.split == 3:
                self.sub_list = list(range(2, 5))
                self.sub_val_list = list(range(5, 6))
            elif self.split == 2:
                self.sub_list = list(range(2, 4)) + list(range(5, 6))
                self.sub_val_list = list(range(4, 5))
            elif self.split == 1:
                self.sub_list = list(range(2, 3)) + list(range(4, 6))
                self.sub_val_list = list(range(3, 4))
            elif self.split == 0:
                self.sub_list = list(range(3, 6))
                self.sub_val_list = list(range(2, 3))

        if test_base:
            self.sub_val_list = self.sub_list

        if mode == 'train':
            print(mode, 'sub_list: ', self.sub_list)
        else:
            print(mode, 'sub_val_list: ', self.sub_val_list)

        #
        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list, True)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune' or self.mode == 'test':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list,
                                                                    False)
            
            self.visual_list = []
            if data_type == 'metadada':
                path = '/data/workspace/zcy/metaDriver_Right/lists/visual_lists/visual_path_pairs_dada.txt'
            else:
                path = '/data/workspace/zcy/metaDriver_Right/lists/visual_lists/visual_path_pairs_psad.txt'
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        # query_path = parts[0].strip("'")
                        support_path = parts[1].strip("'")
                        self.visual_list.append(support_path)

            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)

        self.transform = transform
        self.transform_tri = transform_tri
        self.ft_transform = ft_transform
        self.ft_aug_size = ft_aug_size
        self.ms_transform_list = ms_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # [1] load query image
        query_image_path, query_label_path, query_label_class = self.data_list[index]
        query_image = cv2.imread(query_image_path, cv2.IMREAD_COLOR)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_image = np.float32(query_image)
        query_image = cv2.resize(query_image, dsize=(473, 473), interpolation=cv2.INTER_LINEAR)

        query_label = cv2.imread(query_label_path, cv2.IMREAD_GRAYSCALE)
        query_label = np.float32(query_label)
        query_label = cv2.resize(query_label, dsize=(473, 473), interpolation=cv2.INTER_NEAREST)
        if query_image.shape[:2] != query_label.shape[:2]:
            raise (RuntimeError(
                "Query Image & label shape mismatch: " + query_image_path + " " + query_label_path + "\n"))

        # [2] load support images
        current_cls = query_label_class[0]
        support_image_path_list = []
        support_label_path_list = []
        selection_path_list = []
        # ----> [2-1] get support images path

        if self.mode == 'test':
            for k in range(self.shot):
                selection_idx = random.randint(1, len(self.sub_class_file_list[current_cls])) - 1
                selection_path, _, _ = self.sub_class_file_list[current_cls][selection_idx]
                support_image_path = self.data_root + '/' + self.visual_list[index]
                support_label_path = self.data_root + '/' + self.visual_list[index].replace('images', 'maps')
                support_image_path_list.append(support_image_path)
                support_label_path_list.append(support_label_path)

        else:
            for k in range(self.shot):
                selection_idx = random.randint(1, len(self.sub_class_file_list[current_cls])) - 1
                selection_path, _, _ = self.sub_class_file_list[current_cls][selection_idx]
                support_image_path = query_image_path
                support_label_path = query_label_path
                while (support_image_path == query_image_path or selection_path in selection_path_list):
                    selection_idx = random.randint(1, len(self.sub_class_file_list[current_cls])) - 1
                    support_image_path, support_label_path, _ = self.sub_class_file_list[current_cls][selection_idx]
                    selection_path = support_image_path

                selection_path_list.append(selection_path)
                support_image_path_list.append(support_image_path)
                support_label_path_list.append(support_label_path)
        


        # ----> [2-2] get support images numpy
        support_image_list = []
        support_label_list = []
        for k in range(self.shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]

            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_image = cv2.resize(support_image, dsize=(473, 473), interpolation=cv2.INTER_LINEAR)

            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            support_label = np.float32(support_label)
            support_label = cv2.resize(support_label, dsize=(473, 473), interpolation=cv2.INTER_NEAREST)

            if support_image.shape[:2] != support_label.shape[:2]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        raw_query_label = query_label.copy()

        # if self.transform is not None:
        #     q_img, q_lab = self.transform(query_image, query_label)
        #     for k in range(self.shot):
        #         support_image_list[k], support_label_list[k] = self.transform(support_image_list[k],
        #                                                                       support_label_list[k])

        # ----> [2-3] support images numpy to tensor
        q_img = torch.from_numpy(query_image).float() / 255.0
        q_lab = torch.from_numpy(query_label).float() / 255.0
        q_img = q_img.permute(2, 0, 1)

        for k in range(self.shot):
            support_image_list[k] = torch.from_numpy(support_image_list[k]) / 255.0
            support_label_list[k] = torch.from_numpy(support_label_list[k]) / 255.0
            support_image_list[k] = support_image_list[k].permute(2, 0, 1)
        s_imgs = support_image_list
        s_labs = support_label_list

        s_img = s_imgs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_img = torch.cat([s_imgs[i].unsqueeze(0), s_img], 0)

        s_lab = s_labs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_lab = torch.cat([s_labs[i].unsqueeze(0), s_lab], 0)

        if q_lab.max() > 1.0:
            q_lab = (q_lab - q_lab.min()) / (q_lab.max() - q_lab.min())
        if s_lab.max() > 1.0:
            s_lab = (s_lab - s_lab.min()) / (s_lab.max() - s_lab.min())

        if torch.isnan(q_lab).any() or torch.isinf(q_lab).any():
            raise ValueError("q_lab: NaN or Inf found in input tensor.")
        if torch.isnan(s_lab).any() or torch.isinf(s_lab).any():
            raise ValueError("s_lab: NaN or Inf found in input tensor.")

        q_img = q_img.to(dtype=torch.float32)
        q_lab = q_lab.to(dtype=torch.float32)
        s_img = s_img.to(dtype=torch.float32)
        s_lab = s_lab.to(dtype=torch.float32)  #
        # subcls_list = subcls_list[0].to(dtype=torch.long)

        if self.mode == 'train':
            return q_img, q_lab, q_lab, s_img, s_lab, current_cls
        elif self.mode == 'val':
            return q_img, q_lab, q_lab, s_img, s_lab, current_cls, raw_query_label
        elif self.mode == 'test':
            return q_img, q_lab, q_lab, s_img, s_lab, query_image_path, support_image_path_list