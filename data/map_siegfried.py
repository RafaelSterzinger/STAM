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

class DatasetMAPSEG_SIEGFRIED(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split == 'val' else 'trn'
        self.fold = 'railway' if fold == 0 else 'vineyard'

        print('Using maps siegfried from %s' % self.fold)

        self.nfolds = 2
        self.nclass = 1
        self.class_ids = [0]
        self.benchmark = 'maps_siegfried'
        self.shot = shot
        self.base_path = os.path.join(datapath, f'maps/maps_siegfried')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.img_metadata = self.build_img_metadata('test')
        self.img_sup_metadata = self.build_img_metadata('train')

        if self.split == 'trn':
            self.augmentations = A.Compose([
                 A.ColorJitter(),
                 A.RandomRotate90(),
                 A.HorizontalFlip(),
            ])
        else:
            self.augmentations = A.Compose([A.NoOp()])

    def __len__(self):
        return len(self.img_metadata)

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

    def build_img_metadata(self, split):
        def read_metadata(split, fold_id):
            data = os.listdir(os.path.join(self.base_path, 'dataset_' + fold_id, split))
            fold_n_metadata = [d.split('/')[-1].split('.')[0] for d in data]
            return fold_n_metadata

        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata


    def convert_to_binary_mask(query_mask):
        # Initialize a binary mask with zeros (same height and width as query_mask)
        query_mask_binary = np.zeros(query_mask.shape[:2], dtype=np.uint8)

        # Set pixels matching the drawn_class to 1
        query_mask_binary[np.all(query_mask == [255,255,255], axis=-1)] = 1

        return query_mask_binary


    def load_frame(self, idx):
        query_name = self.img_metadata[idx]

        query_img, query_mask = self.get_img_and_mask(query_name, 'test')

        org_qry_imsize = query_img.size

        support_ids = random.sample(self.img_sup_metadata, self.shot)

        support_imgs = []
        support_masks = []
        for id in support_ids:
            sub_img, sub_mask = self.get_img_and_mask(id, 'train')
            support_imgs.append(sub_img)
            support_masks.append(torch.tensor(sub_mask, dtype=torch.long))

        query_mask = torch.tensor(query_mask, dtype=torch.long)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_ids, 0, org_qry_imsize

    def get_img_and_mask(self, query_name, split):
        base_path = os.path.join(self.base_path, 'dataset_' + self.fold)
        query_img = np.array(Image.open(os.path.join(base_path, split, query_name + '.tif')).convert('RGB'))
        query_mask = np.array(Image.open(os.path.join(base_path, 'annotation', split, query_name + '.tif')).convert('RGB'))
        transform =  self.augmentations(image=query_img, mask=query_mask)
        query_img, query_mask = transform['image'], transform['mask']
        query_mask= DatasetMAPSEG_SIEGFRIED.convert_to_binary_mask(query_mask)
        query_img = Image.fromarray(query_img)
        return query_img,query_mask