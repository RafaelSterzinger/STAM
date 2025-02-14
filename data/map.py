r""" historic map semantic segmentation dataset """
import os
import pickle
import random   
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import albumentations as A

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

        print('Using maps from %s' % self.fold)

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
        if self.split == 'trn':
            self.augmentations_full = A.Compose([
                 A.ColorJitter(),
            ])
            self.augmentations_patch = A.Compose([
                 A.RandomRotate90(),
                 A.HorizontalFlip(),
            ])
        else:
            self.augmentations_full = A.Compose([A.NoOp()])
            self.augmentations_patch = A.Compose([A.NoOp()])
        if self.split == 'trn':
            self.class_weigth = self.calc_pixel_density()
    
    def calc_pixel_density(self):
        if os.path.exists(f'data/class_pixel_counts_{self.fold}.npy'):
            return np.load(f'data/class_pixel_counts_{self.fold}.npy')
        from collections import defaultdict

        class_pixel_counts = [0] * len(CLASSES)

        for img in self.img_metadata:
            base_path = os.path.join(self.base_path, self.fold)
            base_path = os.path.join(base_path, 'train')
            mask = np.array(Image.open(os.path.join(base_path, 'labels', img + '.png')).convert('RGB'))
            unique, counts = np.unique(mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2]), axis=0, return_counts=True)
            unique = [CLASSES[tuple(cls)] for cls in list(unique)]
            for cls, count in zip(unique, counts):
                class_pixel_counts[cls] += count
        
        class_pixel_counts = np.array(class_pixel_counts)
        class_pixel_counts = class_pixel_counts / class_pixel_counts.sum()
        np.save(f'data/class_pixel_counts_{self.fold}.npy', class_pixel_counts)
        return class_pixel_counts

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else len(self.img_metadata)*4

    def __getitem__(self, idx):
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(idx)

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


    def load_frame(self, idx):
        if self.split == 'val':
            sub_idx = idx-idx//4
            idx = idx//4

        query_name = self.img_metadata[idx]

        base_path = os.path.join(self.base_path, self.fold)
        if self.split == 'val':
            base_path = os.path.join(base_path, 'eval')
        else:
            base_path = os.path.join(base_path, 'train')

        query_img = np.array(Image.open(os.path.join(base_path, 'images', query_name + '.png')).convert('RGB'))
        query_mask = np.array(Image.open(os.path.join(base_path, 'labels', query_name + '.png')).convert('RGB'))
        transform =  self.augmentations_full(image=query_img, mask=query_mask)
        query_img, query_mask = transform['image'], transform['mask']

        # Define the size of the slices
        width, height = query_img.shape[0:2]

        # Define the slices for each region (top-left, top-right, bottom-left, bottom-right)
        sub_images_pairs = [
            (query_img[0:height // 2, 0:width // 2], query_mask[0:height // 2, 0:width // 2]),  # Top-left
            (query_img[0:height // 2, width // 2:], query_mask[0:height // 2, width // 2:]),  # Top-right
            (query_img[height // 2:, 0:width // 2], query_mask[height // 2:, 0:width // 2]),  # Bottom-left
            (query_img[height // 2:, width // 2:], query_mask[height // 2:, width // 2:]),  # Bottom-right
        ]

        # Select a random quadrant of the image as query
        query_id = np.random.randint(0, 4) if self.split == 'trn' else sub_idx
        query_img, query_mask = sub_images_pairs[query_id]

        transform =  self.augmentations_patch(image=query_img, mask=query_mask)
        query_img, query_mask = transform['image'], transform['mask']
        query_img = Image.fromarray(query_img)


        # Select a random present class for query
        classes_in_query = list(np.unique(query_mask.reshape(query_mask.shape[0]*query_mask.shape[1], query_mask.shape[2]), axis=0))

        if self.split == 'val':
            drawn_class = random.sample(classes_in_query, 1)[0]
        else:
            present_weights = [1/self.class_weigth[CLASSES[tuple(cls)]] for cls in classes_in_query]
            total_weight = sum(present_weights)
            sample_probs = [(w / total_weight) for w in present_weights]
            drawn_class = random.choices(classes_in_query, k=1, weights=sample_probs)[0]
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
            sub_img, sub_mask = sub_images_pairs[id]
            transform =  self.augmentations_patch(image=sub_img, mask=sub_mask)
            sub_img, sub_mask = transform['image'], transform['mask']
            sub_img = Image.fromarray(sub_img)
            
            support_imgs.append(sub_img)
            support_mask = DatasetMAPSEG.convert_to_binary_mask(sub_mask, drawn_class)
            support_masks.append(torch.tensor(support_mask, dtype=torch.long))

        query_mask = torch.tensor(query_mask, dtype=torch.long)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize