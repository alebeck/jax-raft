"""
Script for validating our model against the official PyTorch RAFT implmentation.
This is largely a copy of RAFT's original Sintel validation logic (train set) from
https://github.com/princeton-vl/RAFT.
"""

from time import time
import random
import os
import os.path as osp
from glob import glob
import sys
from functools import partial

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from jax import jit
from tqdm import trange


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def read_gen(file_name, pil=False):
    ext = osp.splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    else:
        raise NotImplementedError


class FlowDataset(torch.utils.data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        assert not self.sparse
        assert aug_params is None

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = read_gen(self.image_list[index][0])
            img2 = read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        flow = read_gen(self.flow_list[index])

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


def validate_sintel_jax(model, params, data_root, iters=32):
    """ Peform validation using the Sintel (train) split """

    apply_fn = jit(partial(model.apply, params, train=False, num_flow_updates=iters))

    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = MpiSintel(split='training', root=data_root, dstype=dstype)
        epe_list = []
        time_list = []

        for val_id in trange(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None] / 255. * 2. - 1.  # normalize to [-1, 1]
            image2 = image2[None] / 255. * 2. - 1.

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            image1 = image1.numpy().transpose(0, 2, 3, 1)
            image2 = image2.numpy().transpose(0, 2, 3, 1)

            tic = time()
            flow_pr = apply_fn(image1, image2)[-1].block_until_ready()
            if val_id > 0:
                time_list.append(time() - tic)

            flow_pr = torch.from_numpy(np.array(flow_pr)).permute(0, 3, 1, 2)
            flow = padder.unpad(flow_pr[0])

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        time_ = np.mean(time_list)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, time: %f" % (dstype, epe, px1, px3, px5, time_))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_torch(model, data_root, iters=32):
    """ Peform validation using the Sintel (train) split """

    model.eval().cuda()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = MpiSintel(split='training', root=data_root, dstype=dstype)
        epe_list = []
        time_list = []

        for val_id in trange(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda() / 255. * 2. - 1.  # normalize to [-1, 1]
            image2 = image2[None].cuda() / 255. * 2. - 1.

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            tic = time()
            flow_pr = model(image1, image2, num_flow_updates=iters)[-1]
            time_list.append(time() - tic)

            flow = padder.unpad(flow_pr).cpu()

            epe = torch.sum((flow - flow_gt[None]) ** 2, dim=1).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        time_ = np.mean(time_list)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, time: %f" % (dstype, epe, px1, px3, px5, time_))
        results[dstype] = np.mean(epe_list)

    return results


if __name__ == '__main__':
    print('JAX evaluation:\n')
    from jax_raft import raft_large, raft_small

    for model_size in (raft_large, raft_small):
        model, variables = model_size(pretrained=True)
        validate_sintel_jax(model, variables, data_root=sys.argv[1])

    print('PyTorch evaluation:\n')
    from torchvision.models.optical_flow import raft_large, raft_small

    for model_size in (raft_large, raft_small):
        model = model_size(pretrained=True)
        validate_sintel_torch(model, data_root=sys.argv[1])
