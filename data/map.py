r""" historic map semantic segmentation dataset """
import os
import pickle
import random   
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

_original_dict = {
    0: np.array([0, 0, 0]), #"frame"
    1: np.array([0, 0, 255]), #"water"
    2: np.array([255, 0, 255]), #"blocks"
    3: np.array([0, 255, 255]), #"non-built"
    4: np.array([255, 255, 255]) #"road_network"
}

# Swap keys and values, converting NumPy arrays to tuples
CLASSES = {tuple(v): k for k, v in _original_dict.items()}

POSITIONS = {
    0: "tl",
    1: "tr",
    2: "bl",
    3: "br"
}


class DatasetMAPSEG(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split == 'eval' else 'trn'
        self.fold = 'paris' if fold == 0 else 'world'

        print('Training on maps from %s' % self.fold)

        self.nfolds = 2
        self.nclass = 5
        self.class_ids = [0, 1, 2, 3, 4]
        self.benchmark = 'maps'
        self.shot = shot
        assert self.shot <= 3
        self.base_path = os.path.join(datapath, f'{self.benchmark}/maps_other')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)*4 if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_img_metadata(self):
        def read_metadata(split, fold_id):
            data = os.listdir(os.path.join(self.base_path, fold_id, split, 'images'))
            fold_n_metadata = [d.split('/')[-1].split('.')[0] for d in data]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            img_metadata = read_metadata('train', self.fold)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata('eval', self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata


    @staticmethod
    def convert_to_binary_mask(query_mask, drawn_class):
        # Initialize a binary mask with zeros (same height and width as query_mask)
        query_mask_binary = np.zeros(query_mask.shape[:2], dtype=np.uint8)

        # Set pixels matching the drawn_class to 1
        query_mask_binary[np.all(query_mask == drawn_class, axis=-1)] = 1

        return query_mask_binary


    def load_frame(self):
        query_name = random.sample(self.img_metadata, 1)[0]

        base_path = os.path.join(self.base_path, self.fold)
        if self.split == 'val':
            base_path = os.path.join(base_path, 'eval')
        else:
            base_path = os.path.join(base_path, 'train')

        query_img = Image.open(os.path.join(base_path, 'images', query_name + '.png')).convert('RGB')
        query_mask = np.asarray(Image.open(os.path.join(base_path, 'labels', query_name + '.png')).convert('RGB')).copy()
        # Define the size of the slices
        width, height = query_img.size

        # Define the slices for each region (top-left, top-right, bottom-left, bottom-right)
        sub_images_pairs = [
            (query_img.crop((0, 0, width // 2, height // 2)), query_mask[0:height // 2, 0:width // 2]),  # Top-left
            (query_img.crop((width // 2, 0, width, height // 2)), query_mask[0:height // 2, width // 2:]),  # Top-right
            (query_img.crop((0, height // 2, width // 2, height)), query_mask[height // 2:, 0:width // 2]),  # Bottom-left
            (query_img.crop((width // 2, height // 2, width, height)), query_mask[height // 2:, width // 2:]),  # Bottom-right
        ]

        # Select a random quadrant of the image as query
        query_id = np.random.randint(0, 4)
        query_img = sub_images_pairs[query_id][0]
        query_mask = sub_images_pairs[query_id][1]

        # Select a random present class for query
        classes_in_query = list(np.unique(query_mask.reshape(query_mask.shape[0]*query_mask.shape[1], query_mask.shape[2]), axis=0))
        drawn_class = random.sample(classes_in_query, 1)[0]
        class_sample = CLASSES[tuple(drawn_class)]

        # Convert query mask to binary
        query_mask = DatasetMAPSEG.convert_to_binary_mask(query_mask, drawn_class)

        org_qry_imsize = query_img.size

        # selects self.shot many quadrants as support
        support_ids = random.sample([i for i in range(4) if i != query_id], self.shot)
        support_names = [POSITIONS[support_id] for support_id in support_ids]

        support_imgs = []
        support_masks = []
        for id in support_ids:
            support_imgs.append(sub_images_pairs[id][0])
            support_mask = DatasetMAPSEG.convert_to_binary_mask(sub_images_pairs[id][1], drawn_class)
            support_masks.append(torch.tensor(support_mask, dtype=torch.long))

        query_mask = torch.tensor(query_mask, dtype=torch.long)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize