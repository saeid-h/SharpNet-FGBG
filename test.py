import argparse, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import Bottleneck as ResBlock
from sharpnet_model import SharpNet
from loss import *
from PIL import Image
try:
    from imageio import imread, imsave
except:
    from scipy.misc import imread, imsave
from data_transforms import ToTensor, Compose, Normalize
from skimage import feature
from scipy import ndimage
import time
from tqdm import tqdm

import os
import scipy.io as io
import h5py
import cv2

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """    
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def depth_write(filename, depth):
    """ Write depth to file. """
    height,width = depth.shape[:2]
    f = open(filename,'wb')
    np.array(TAG_FLOAT).astype(np.float32).tofile(f)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    depth.astype(np.float32).tofile(f)
    f.close()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_pred_from_input(image_pil, args):
    normals = None
    boundary = None
    depth = None
    raw = None

    image_np = np.array(image_pil)
    
    if len(image_np.shape) == 2 or image_np.shape[-1] == 1:
        print("Input image has only 1 channel, please use an RGB or RGBA image")
        sys.exit(0)

    if len(image_np.shape) == 4 or image_np.shape[-1] == 4:
        # RGBA image to be converted to RGB
        image_pil = image_pil.convert('RGBA')
        image = Image.new("RGB", (image_np.shape[1], image_np.shape[0]), (255, 255, 255))
        image.paste(image_pil.copy(), mask=image_pil.split()[3])
    else:
        image = image_pil

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = []
    t.extend([ToTensor(), normalize])
    transf = Compose(t)

    data = [image, None]
    image = transf(*data)

    image = torch.autograd.Variable(image).unsqueeze(0)
    image = image.to(device)

    if args.boundary:
        if args.depth and args.normals:
            depth_pred, normals_pred, boundary_pred = model(image)
        elif args.depth and not args.normals:
            depth_pred, boundary_pred = model(image)
        elif args.normals and not args.depth:
            normals_pred, boundary_pred = model(image)
        else:
            boundary_pred = model(image)
    else:
        if args.depth:
            depth_pred = model(image)
        if args.depth and args.normals:
            depth_pred, normals_pred = model(image)
        if args.normals and not args.depth:
            normals_pred = model(image)

    if args.normals:
        normals_pred = normals_pred.data.cpu().numpy()[0, ...]
        normals_pred = normals_pred.swapaxes(0, 1).swapaxes(1, 2)
        normals_pred[..., 0] = 0.5 * (normals_pred[..., 0] + 1)
        normals_pred[..., 1] = 0.5 * (normals_pred[..., 1] + 1)
        normals_pred[..., 2] = -0.5 * np.clip(normals_pred[..., 2], -1, 0) + 0.5

        normals_pred[..., 0] = normals_pred[..., 0] * 255
        normals_pred[..., 1] = normals_pred[..., 1] * 255
        normals_pred[..., 2] = normals_pred[..., 2] * 255

        normals = normals_pred.astype('uint8')

    if args.depth:
        raw = depth_pred.data.cpu().numpy()[0, 0, ...] * 65535 / 1000
        depth_pred = raw
        m = np.min(depth_pred)
        M = np.max(depth_pred)
        depth_pred = (depth_pred - m) / (M - m)
        depth = Image.fromarray(np.uint8(plt.cm.jet(depth_pred) * 255))
        depth = np.array(depth)[:, :, :3]

    if args.boundary:
        boundary_pred = boundary_pred.data.cpu().numpy()[0, 0, ...]
        boundary_pred = np.clip(boundary_pred, 0, 10)
        boundary = (boundary_pred * 255).astype('uint8')

    return {'rgb':image_np, 'depth':depth, 'normals':normals, 'boundary':boundary, 'raw': raw}


def save_preds(outpath, preds, image_path, args):
    image_name = '_'.join(image_path.split('.')[0].split('/')[-2:])+'.png'

    imsave(os.path.join(outpath, 'rgb' ,image_name), preds['rgb'])

    if args.depth: imsave(os.path.join(outpath, 'depth' ,image_name), preds['depth'])
    if args.normals: imsave(os.path.join(outpath, 'normals' ,image_name), preds['normals'])
    if args.boundary: imsave(os.path.join(outpath, 'boundary' ,image_name), preds['boundary'])

    if args.depth: imsave(os.path.join(outpath, 'cmap' ,image_name), (np.min(preds['raw']) / preds['raw'] * 255).astype(np.uint8)) 
    if args.depth: depth_write(os.path.join(outpath, 'raw' ,image_name.replace('.png','.dpt')), preds['raw'])

    if not preds['gt'] is None and args.depth:
        gt_depth = preds['gt'][192:192+128,192:192+160]
        pred_depth = preds['raw'][192:192+128,192:192+160]
        ref_depth = np.percentile(gt_depth, 50)
        
        gt_mask = np.zeros_like(gt_depth)
        gt_mask[ref_depth>gt_depth] = 255
        imsave(os.path.join(outpath, 'occ_mask_gt' ,image_name), gt_mask.astype(np.uint8))
        
        init_mask = np.zeros_like(pred_depth)
        init_mask[ref_depth>pred_depth] = 255
        imsave(os.path.join(outpath, 'occ_mask_init',image_name), init_mask.astype(np.uint8))

        final_mask = sigmoid(ref_depth-pred_depth) * 255
        imsave(os.path.join(outpath, 'occ_mask_final' ,image_name), final_mask.astype(np.uint8))


parser = argparse.ArgumentParser(description="Test a model on an image")
parser.add_argument('--model', '-m', dest='model', help="checkpoint.pth which contains the model")
parser.add_argument('--model_name', dest='model_name', help="model's name.", default='noname')
parser.add_argument('--dataset', dest='dataset', help="dataset name")
parser.add_argument('--data_path',          type=str,   help='path to the data', required=False)
parser.add_argument('--gt_path',            type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--filenames_file',     type=str,   help='path to the filenames text file', required=False)
parser.add_argument('--save_path', dest='save_path', type=str)
parser.add_argument('--cuda', dest='cuda_device', default='', help="To activate inference on GPU, set to GPU_ID")
parser.add_argument('--nocuda', action='store_true')
parser.add_argument('--crop', dest='eigen_crop', action='store_true', help='Flag to evaluate on center crops defined by Eigen')
parser.add_argument('--edges', action='store_true', help='Flag to evaluate on occlusion boundaries')
parser.add_argument('--low', dest='low_threshold', type=float, default=0.03, help='Low threshold of Canny edge detector')
parser.add_argument('--high', dest='high_threshold', type=float, default=0.05, help='High threshold of Canny edge detector')
parser.add_argument('--normals', action='store_true', help='Activate to predict normals')
parser.add_argument('--depth', action='store_true', help='Activate to predict depth')
parser.add_argument('--boundary', action='store_true', help='Activate to predict occluding contours')
parser.add_argument('--bias', action='store_true')

if sys.argv.__len__() == 2:
    arg_list = list()
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    for line in lines:
        arg_list += line.strip().split()
    args = parser.parse_args(arg_list)
else:
    args = parser.parse_args()

if not args.nocuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    device = torch.device("cuda" if args.cuda_device != '' else "cpu")
    # print("Running on " + torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print("Running on CPU")

with open(args.filenames_file, 'r') as f:
    lines = f.readlines()
image_list = [os.path.join(args.data_path, line.strip().split()[0]) for line in lines]
image_list.sort()
if args.gt_path:
    gt_list = [os.path.join(args.gt_path, line.strip().split()[1]) for line in lines]
    gt_list.sort()


model = SharpNet(ResBlock, [3, 4, 6, 3], [2, 2, 2, 2, 2],
                 use_normals=True if args.normals else False,
                 use_depth=True if args.depth else False,
                 use_boundary=True if args.boundary else False,
                 bias_decoder=args.bias)

torch.set_grad_enabled(False)

model_dict = model.state_dict()

# Load model
trained_model_path = args.model
trained_model_dict = torch.load(trained_model_path, map_location=lambda storage, loc: storage)

# load image resnet encoder and mask_encoder and normals_decoder (not depth_decoder or normal resnet)
model_weights = {k: v for k, v in trained_model_dict.items() if k in model_dict}

model.load_state_dict(model_weights)
model.eval()
model.to(device)

mean_RGB = np.array([0.485, 0.456, 0.406])
mean_BGR = np.array([mean_RGB[2], mean_RGB[1], mean_RGB[0]])

if args.save_path is not None:
    outpath = os.path.join(args.save_path, args.model_name)
    os.system ('mkdir -p '+os.path.join(outpath, 'rgb'))
    if args.depth: os.system ('mkdir -p '+os.path.join(outpath, 'raw'))
    if args.depth: os.system ('mkdir -p '+os.path.join(outpath, 'cmap')) 
    if args.depth: os.system ('mkdir -p '+os.path.join(outpath, 'depth'))
    if args.gt_path: os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_init'))
    if args.gt_path: os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_final'))
    if args.gt_path: os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_gt'))
    if args.normals: os.system ('mkdir -p '+os.path.join(outpath, 'normals'))
    if args.boundary: os.system ('mkdir -p '+os.path.join(outpath, 'boundary'))


for indx in tqdm(range(len(image_list))):
    image_path = image_list[indx]
    image_pil = Image.open(image_path)
    
    gt = depth_read(gt_list[indx]) if args.gt_path else None

    preds = get_pred_from_input(image_pil, args)
    preds.update({'gt': gt})

    if args.save_path is not None:
        save_preds(outpath, preds, image_path, args)

