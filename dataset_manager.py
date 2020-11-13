from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from imageio import imread
import scipy.io as io
import h5py
from representations import *
import data_transforms as transforms
import random

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    TAG_FLOAT = 202021.25
    TAG_CHAR = 'PIEH'
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth


class GeoDataset(Dataset):
    def __init__(self, img_list, root_dir='', img_size=480, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_occ=False,
                 use_normals=True,
                 input_type='image'):

        if root_dir == '':
            self.root_dir = os.getcwd()
        else:
            self.root_dir = root_dir
        self.transforms = transforms

        self.img_list = img_list
        self.img_size = img_size
        self.use_boundary = use_boundary
        self.use_depth = use_depth
        self.use_occ=use_occ,
        self.use_normals = use_normals
        self.input_type = input_type

    def __len__(self):
        return len(self.img_list)

    def format_data(self, image=None,
                    mask_valid=None,
                    depth=None,
                    normals=None,
                    boundary=None):
        # augment data and format it to tensor type

        data = [image, mask_valid,
                depth if self.use_depth else None,
                normals if self.use_normals else None,
                boundary if self.use_boundary else None]

        crop_corner = [0,0]
        if self.transforms is not None:
            ratio = 1
            crop_size = None
            angle = 0
            gamma_ratio = 1
            normalize = False

            if 'SCALE' in self.transforms.keys():
                ratio = random.uniform(1.0 / self.transforms['SCALE'], 1.0 * self.transforms['SCALE'])
            if 'HORIZONTALFLIP' in self.transforms.keys():
                flip = random.random() < 0.5
            if 'CROP' in self.transforms.keys():
                crop_size = self.transforms['CROP']
                # x1, y1, tw, th = transforms.get_random_bbox(data, crop_size, crop_size)
            if 'ROTATE' in self.transforms.keys():
                angle = random.uniform(0, self.transforms['ROTATE'] * 2) - self.transforms['ROTATE']
            if 'GAMMA' in self.transforms.keys():
                gamma_ratio = random.uniform(1 / self.transforms['GAMMA'], self.transforms['GAMMA'])
            if 'NORMALIZE' in self.transforms.keys():
                normalize = True

            for mode in data:
                if mode is not None:
                    if ratio != 1:
                        mode.scale(ratio)
            if crop_size is not None:
                data, crop_corner = transforms.get_random_crop(data, crop_size, crop_size)
            
            for m, mode in enumerate(data):
                if mode is not None:
                    if flip:
                        mode.fliplr()
                    if angle != 0:
                        mode.rotate(angle, cval=0)
                    if m == 0:
                        if gamma_ratio != 1:
                            data[0].gamma(gamma_ratio)
                    mode.to_tensor()
                    if m == 0:
                        mode.normalize(mean=self.transforms['NORMALIZE']['mean'],
                                       std=self.transforms['NORMALIZE']['std'])
                    mode.data = mode.data.float()
    
        return [m.data for m in data if m is not None] + [crop_corner, ratio]


    def __getitem__(self, idx):
        # overwrite this function when creating a new dataset
        image = self.img_list[idx]
        mask_valid = np.ones(np.array(image).shape[:2])
        depth = None
        normals = None
        boundary = None

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)
        return sample


class PBRSDataset(GeoDataset):
    def __init__(self, img_list, root_dir='', img_size=480, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_normals=True,
                 input_type='image'):
        super(PBRSDataset, self).__init__(img_list, root_dir=root_dir, img_size=img_size,
                                          transforms=transforms,
                                          use_boundary=use_boundary,
                                          use_depth=use_depth,
                                          use_normals=use_normals,
                                          input_type=input_type)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, 'img', img_name)
        image = Image.open(img_path)

        normals = None
        boundary = None
        depth = None

        mask_valid = imread(os.path.join(self.root_dir, 'normals', img_name.replace('_mlt.png', '_valid.png')))
        mask_valid = mask_valid / 255
        mask_valid = Mask(data=mask_valid.copy())
        image = InputImage(data=image)

        if self.use_depth:
            data = imread(os.path.join(self.root_dir, 'depth', img_name.replace('_mlt.png', '_depth.png')))
            data = data.astype('float32') / 65535.0
            depth = Depth(data=data.copy())

        if self.use_normals:
            data = imread(os.path.join(self.root_dir, 'normals', img_name.replace('_mlt.png', '_norm_camera.png')))
            normals_tmp = data.astype('float32')
            normals = Normals(data=normals_tmp.copy())
            normals.data[..., 0] = ((255 - normals_tmp[..., 0]) - 127.5) / 127.5
            normals.data[..., 1] = (normals_tmp[..., 2] - 127.5) / 127.5
            normals.data[..., 2] = -2.0 * ((255.0 - normals_tmp[..., 1]) / 255.0) + 1

        if self.use_boundary:
            data = imread(
                os.path.join(self.root_dir, 'boundaries', img_name.replace('_mlt.png', '_instance_boundary.png')))
            data = data / 255
            boundary = Contours(data=data.copy())

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)

        return sample


class NYUDataset(GeoDataset):
    def __init__(self, dataset_path, split_type='train', root_dir='', img_size=480, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_occ=False,
                 use_normals=True,
                 input_type='image'):
        super(NYUDataset, self).__init__(img_list=None, root_dir=root_dir, img_size=img_size,
                                         transforms=transforms,
                                         use_boundary=use_boundary,
                                         use_depth=use_depth,
                                         use_occ=use_occ,
                                         use_normals=use_normals,
                                         input_type=input_type)

        self.dataset_path = os.path.join(root_dir, dataset_path)
        used_split = io.loadmat(os.path.join(root_dir, 'nyuv2_splits.mat'))
        self.idx_list = [idx[0] - 1 for idx in used_split[split_type + 'Ndxs']]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # Get image from NYUv2 mat file
        # Crop border by 6 pixels
        dataset = h5py.File(self.dataset_path, 'r', libver='latest', swmr=True)
        image = dataset['images'][self.idx_list[idx]]
        image_new = image.swapaxes(0, 2)
        image_old = image

        normals = None
        boundary = None
        depth = None

        crop_ROI = [6, 6, 473, 630]
        image_new = image_new[crop_ROI[0]:crop_ROI[2], crop_ROI[1]:crop_ROI[3], :]

        mask_valid = np.ones(shape=image_new.shape[:2])
        mask_valid = Mask(data=mask_valid.copy())

        image_new = Image.fromarray(image_new)
        image = InputImage(data=image_new.copy())

        if self.use_depth:
            data = dataset['depths'][self.idx_list[idx]].swapaxes(0, 1).astype('float32') * 1000 / 65535
            data = data[crop_ROI[0]:crop_ROI[2], crop_ROI[1]:crop_ROI[3]]
            depth = Depth(data=data.copy())

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)
        return sample


class ReplicaDataset(GeoDataset):
    def __init__(self, img_list, root_dir='', img_size=512, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_occ=False,
                 use_normals=True,
                 input_type='image'):
        super(ReplicaDataset, self).__init__(img_list, root_dir=root_dir, img_size=img_size,
                                          transforms=transforms,
                                          use_boundary=use_boundary,
                                          use_occ=use_occ,
                                          use_depth=use_depth,
                                          use_normals=use_normals,
                                          input_type=input_type)
        
        self.img_list = [os.path.join('replica/image_left', x.split(' ')[0]) for x in img_list]
        self.depth_list = [os.path.join('replica/depth_left', x.split(' ')[1]) for x in img_list]
        self.img_list.sort()
        self.depth_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        normals = None
        boundary = None
        depth = None

        mask_valid = np.ones(shape=(512,512))
        mask_valid = Mask(data=mask_valid.copy())
        image = InputImage(data=image)

        if self.use_depth or self.use_occ:
            depth_name = self.depth_list[idx]
            depth_path = os.path.join(self.root_dir, depth_name)
            data = depth_read(depth_path) / 65.535 
            depth = Depth(data=data.copy())

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)

        return sample
