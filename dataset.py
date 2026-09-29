import os
from glob import glob
from random import choices

import cv2
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset

# CRC class to index mapping
CLASS2IDX = { 'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8 }

BREAKHIS = ['tubular_adenoma', 'papillary_carcinoma', 'phyllodes_tumor', 'mucinous_carcinoma', 
            'fibroadenoma', 'ductal_carcinoma', 'lobular_carcinoma', 'adenosis']

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None, 
                 multi_level=True, num_sample_per_class=None, one_level=False,
                 pad_mode=cv2.BORDER_CONSTANT):
        """
        Args:
            paths (list): list of paths to image folders
            size (int): size to which images will be resized
            random_crop (bool): whether to apply random cropping
            labels (dict): dictionary of labels for each image
            num_sample_per_class (int): number of samples to use per class
            multi_level (bool): whether to use resolution-dependent preprocessing
            one_level (bool): whether to use 20x magnification only or multi-magnification preprocessing
            pad_mode (int): padding mode for images
        """
        self.size = size
        self.random_crop = random_crop
        self.multi_level = multi_level

        self.labels = dict() if labels is None else labels

        self.labels["file_path_"] = []
        self.labels["class_label"] = [] # for conditioning 
        self.labels["human_label"] = [] # for image logging
        self._length = 0
        for path in paths:
            if 'vgh' in path or ('gpuhome' in path and 'data' in path):
                files = [os.path.join(path, subpath, fname) for subpath in os.listdir(path) 
                         for fname in os.listdir(os.path.join(path, subpath))]
                self.labels["file_path_"].extend(files)
                self._length += len(files)
            elif 'CRC' in path:
                tissue_type = os.listdir(path)
                human_labels, class_labels, train_files = [], [], []
                for tt in tissue_type:
                    filenames = os.listdir(os.path.join(path, tt))
                    if num_sample_per_class is not None and num_sample_per_class < len(filenames):
                        filenames = filenames[:num_sample_per_class]
                    train_files.extend([os.path.join(path, tt, fn) for fn in filenames])
                    human_labels.extend([tt] * len(filenames))
                    class_labels.extend([CLASS2IDX[tt]] * len(filenames))
                self.labels["file_path_"].extend(train_files) 
                self.labels["class_label"].extend(class_labels)
                self.labels["human_label"].extend(human_labels)
                self._length += len(train_files)
            elif 'BreakHis' in path:
                human_labels, class_labels, train_files = [], [], []
                for tt in BREAKHIS:
                    filenames = os.listdir(os.path.join(path, tt))
                    if num_sample_per_class is not None and num_sample_per_class < len(filenames):
                        filenames = filenames[:num_sample_per_class]
                    train_files.extend([os.path.join(path, tt, fn) for fn in filenames])
                    human_labels.extend([tt] * len(filenames))
                    class_labels.extend([BREAKHIS.index(tt)] * len(filenames))
                self.labels["file_path_"].extend(train_files)
                self.labels["class_label"].extend(class_labels)
                self.labels["human_label"].extend(human_labels)
                self._length += len(train_files)
            else: # for TCGA, PAIP, CPTAC
                filenames = glob(f'{path}/*/*/*.jpg')
                self.labels["file_path_"].extend(filenames)
                self._length += len(filenames)

        # Remove empty label lists to avoid issues during training
        if len(self.labels["class_label"]) == 0:
            del self.labels["class_label"]
        if len(self.labels["human_label"]) == 0:
            del self.labels["human_label"]

        if not multi_level: # Do not apply resolution-dependent preprocessing 
            if self.size is not None and self.size > 0: 
                self.rescaler = A.LongestMaxSize(max_size = self.size)
                if not self.random_crop:
                    self.cropper = A.CenterCrop(height=self.size, width=self.size, 
                                                pad_if_needed=True, pad_mode=pad_mode)
                    # This requires albumentations>=1.4.21
                else:
                    self.cropper = A.RandomCrop(height=self.size,width=self.size)
                self.preprocessor = A.Compose([self.rescaler, self.cropper])
            else:
                self.preprocessor = lambda **kwargs: kwargs
        else: # Use resolution-dependent preprocessing 
            self.cropper = A.RandomCrop(height=self.size, width=self.size)
            self.cropper_mid = A.RandomCrop(height=self.size * 2, width=self.size * 2)
            self.rescaler = A.SmallestMaxSize(max_size=self.size)
            if not one_level: # use multi-magnification preprocessing for training
                self.preprocessors = {
                    2048: [[self.cropper, A.Compose([self.cropper_mid, self.rescaler]), self.rescaler], [.6, .3, .1]], # 40x, 20x, 10x
                    1024: [[self.cropper, self.rescaler], [.6, .4]]# 20x, 10x
                }
            else: # only use 20x magnification for training
                self.preprocessors = {
                    2048: [[self.cropper, A.Compose([self.cropper_mid, self.rescaler]), self.rescaler], [0., 1., 0.]], # 40x, 20x, 10x
                    1024: [[self.cropper, self.rescaler], [1., 0.]]# 20x, 10x
                }


    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.multi_level:
            preprocessor = choices(*self.preprocessors[image.shape[0]])[0]
        else:
            preprocessor = self.preprocessor
        image = preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class HistoBase(Dataset):
    """Base class for histopathology datasets. 
    Subclasses should implement the __init__, __len__, and __getitem__ methods.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class HistoTrain(HistoBase):
    """Histo-7M for training"""
    def __init__(self, size, use_full_dataset=False, num_sample_per_class=None):
        super().__init__()
        if use_full_dataset:
            paths = ['/path_to_paip_patches', '/path_to_cptac_patches', 
                   '/path_to_tcga_patches1', '/path_to_tcga_patches2']
        else:
            paths = ['/path_to_paip_patches1'] # debugging with a smaller dataset
        self.data = ImagePaths(paths=paths, size=size, multi_level=True, random_crop=False, 
                               num_sample_per_class=num_sample_per_class)

class HistoTrain20x(HistoBase):
    """Histo-7M for training, only using 20x magnification"""
    def __init__(self, size, use_full_dataset=False, num_sample_per_class=None):
        super().__init__()
        if use_full_dataset:
            paths = ['/path_to_paip_patches', '/path_to_cptac_patches', 
                   '/path_to_tcga_patches1', '/path_to_tcga_patches2']
        else:
            paths = ['/path_to_paip_patches']
        self.data = ImagePaths(paths=paths, size=size, multi_level=True, random_crop=False,
                               one_level=True, num_sample_per_class=num_sample_per_class)

class HistoTrainV1(HistoBase):
    """Dataset used in HistoDiffusion v1, which consists of TCGA-BRCA, VGH and PanNuke"""
    def __init__(self, size, use_full_dataset=False, num_sample_per_class=None):
        super().__init__()
        if use_full_dataset:
            paths = ['/path_to_TCGA_BRCA', '/path_to_vgh_patches', '/path_to_PanNuke_post']
        else:
            paths = ['/path_to_paip_patches']
        self.data = ImagePaths(paths=paths, size=size, multi_level=False, random_crop=False, 
                               num_sample_per_class=num_sample_per_class)

class ProstateDataset(HistoBase):
    """Prostate dataset for virtual staining"""
    def __init__(self, size, anno_file, root_dir):
        super().__init__()
        with open(anno_file, 'r') as f:
            id_list = [line.strip() for line in f.readlines()]
        self.stained_paths, self.unstained_paths = [], []
        for id in id_list:
            self.stained_paths += sorted(glob(f'{root_dir}/stained/sample_{id}_stained/*.jpg'))
            self.unstained_paths += sorted(glob(f'{root_dir}/unstained/sample_{id}_unstained/*.jpg'))
        assert len(self.stained_paths) == len(self.unstained_paths), "Number of stained and unstained images do not match."
        self.preprocessor = A.SmallestMaxSize(max_size=size)

    def __len__(self):
        return len(self.stained_paths)

    def preprocess_image(self, image_path, scale=2.):
        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = image / 127.5 - 1.0 if scale == 2. else image / 255.0
        image = np.transpose(image.astype(np.float32), (2, 0, 1))
        return image

    def __getitem__(self, i):
        example = dict()
        example["stained"] = self.preprocess_image(self.stained_paths[i])
        example["unstained"] = self.preprocess_image(self.unstained_paths[i], scale=1.)
        folder, fname = self.stained_paths[i].split('/')[-2:]
        example["folder"] = folder
        example["fname"] = fname
        return example


class CRCDataset(HistoBase):
    """CRC dataset for tissue type classification"""
    def __init__(self, size, mode='train', num_sample_per_class=None):
        super().__init__()
        paths = [f'/path_to_CRC_Data/{mode}']
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, multi_level=False,
                               num_sample_per_class=num_sample_per_class)
        

class BreakHisDataset(HistoBase):
    """BreakHis dataset for tissue classification"""
    def __init__(self, size, mode='train', num_sample_per_class=None, pad_mode='constant'):
        pad_modes = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT,
        }
        pad_mode = pad_modes[pad_mode]
        super().__init__()
        if mode == 'train':
            paths = ['/path_to_BreakHis/train']
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, multi_level=False,
                               num_sample_per_class=num_sample_per_class, pad_mode=pad_mode)

class MoNuSACDataset(Dataset):
    """MoNuSAC dataset for nuclei segmentation"""
    def __init__(self, mode='train'):
        super().__init__()
        images = np.load(f'/path_to_MoNuSAC/{mode}_images.npy').astype(np.float32)
        images = images/127.5 - 1.0
        self.images = np.transpose(images, (0, 3, 1, 2))
        del images
        labels = np.load(f'/path_to_MoNuSAC/{mode}_masks.npy').astype(np.float32)
        masks_inst = labels[..., 0] / 255
        masks_segm = labels[..., 1] / 3  
        self.labels = np.transpose(np.stack([masks_inst, masks_segm], axis=-1), (0, 3, 1, 2))
        del labels
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, i):
        example = dict()
        example['image'] = self.images[i]
        example['mask'] = self.labels[i]
        return example
        

class CoNSePDataset(Dataset):
    """CoNSeP dataset for nucleus segmentation"""
    def __init__(self, mode='train'):
        super().__init__()
        images = np.load(f'/path_to_CoNSeP/{mode}_images.npy').astype(np.float32)
        images = images/127.5 - 1.0
        self.images = np.transpose(images, (0, 3, 1, 2))
        del images
        labels = np.load(f'/path_to_CoNSeP/{mode}_masks.npy').astype(np.float32)
        masks_inst = labels[..., 0] / 2283 # maximum instance number is 2283
        masks_segm = labels[..., 1] / 4 # total 4 classes excluding background
        self.labels = np.transpose(np.stack([masks_inst, masks_segm], axis=-1), (0, 3, 1, 2))
        del labels
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, i):
        example = dict()
        example['image'] = self.images[i]
        example['mask'] = self.labels[i]
        return example
        
    
if __name__ == '__main__':
    dataset = HistoTrain(512)
    print(len(dataset))
    