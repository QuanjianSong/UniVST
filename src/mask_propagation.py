import os

import queue
import argparse
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image

import torch
from torch.nn import functional as F



@torch.no_grad()
def video_mask_propogation(args):
    # save path
    name = args.mask_path.split('/')[-1].split('.')[0]
    output_path = os.path.join(args.output_path, args.backbone, name)
    os.makedirs(output_path, exist_ok=True)
    #
    color_palette = []
    with open("src/palette.txt", 'r', encoding='utf-8') as file:
        for line in file:
            color_palette.append([int(i) for i in line.strip().split()])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)
    # save first mask
    first_seg = Image.open(args.mask_path)
    imageio.imwrite(os.path.join(output_path, "00000.png"), np.asarray(first_seg).astype(np.uint8))
    ori_w, ori_h = first_seg.size
    _, _h, _w = read_feature(args.feature_path, 0, return_h_w=True)
    first_seg = np.array(first_seg.resize((_w, _h), 0))
    first_seg = torch.from_numpy(first_seg).float()
    first_seg = to_one_hot(first_seg.unsqueeze(0))
    # -------------------------------------------------------------------------------------------
    # the queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)
    # extract first frame feature
    feat_first = read_feature(args.feature_path, 0).T #  dim x h*w
    # for loop
    for cnt in tqdm(range(1, args.num_frames)):
        # we use the first segmentation and the n previous ones
        feat_src_list = [feat_first] + [pair[0] for pair in list(que.queue)]
        segs_src_list = [first_seg.squeeze(0).flatten(1)] + [pair[1] for pair in list(que.queue)]
        # -----------------------------------------------------------------------------------
        # concat feat_src, segs_src
        feat_src = torch.cat(feat_src_list, dim=-1)
        segs_src = torch.cat(segs_src_list, dim=-1)
        C, _ = segs_src.shape
        # extract feat_tgt of current cnt frame
        feat_tgt, _h, _w = read_feature(args.feature_path, cnt, return_h_w=True)
        # mask propogation
        final_mask, feat_tgt, segs_tar = mask_propogation(feat_src, feat_tgt, segs_src, args)
        # pop out oldest frame if neccessary
        if que.qsize() == args.n_last_frames:
            que.get()
        # push current results into queue
        que.put([feat_tgt, segs_tar])
        # upsampling & argmax
        final_mask = final_mask.reshape(1, C, _h, _w)
        final_mask = F.interpolate(final_mask, size=(ori_h, ori_w), mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        final_mask = norm_mask(final_mask)
        _, final_mask = torch.max(final_mask, dim=0)
        
        # save the results
        final_mask = np.array(final_mask.cpu(), dtype=np.uint8)
        final_mask[final_mask != 0] = 255
        final_mask = Image.fromarray(final_mask)
        final_mask.save(os.path.join(output_path, f"%05d.png" % (cnt)))


def mask_propogation(feat_src, feat_tar, segs, args,):
    feat_tar_ori = feat_tar.T
    #
    feat_src = F.normalize(feat_src, dim=0, p=2)
    feat_tar = F.normalize(feat_tar, dim=1, p=2).squeeze(0)
    # *********************************************************************************
    aff = torch.exp(feat_tar @ feat_src / args.temperature).transpose(1, 0)
    tk_val_min = torch.topk(aff, args.topk, dim=0).values.min(dim=0).values
    aff[aff < tk_val_min] = 0
    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    # get mask
    segs_tar = torch.mm(segs, aff)


    # ************************** sampling of return_feat_tar **************************
    fore_index = torch.where(segs_tar[0, :] != 0)[0]
    fore_nums = len(fore_index)
    back_index = torch.where(segs_tar[0, :] == 0)[0]
    back_nums = len(back_index)
    #
    random_indices = torch.randperm(len(fore_index))[: int(len(fore_index) * fore_nums / (fore_nums + back_nums) * args.sample_ratio)]
    fore_index_sample = fore_index[random_indices]
    random_indices = torch.randperm(len(back_index))[: int(len(back_index) * back_nums / (fore_nums + back_nums) * args.sample_ratio)]
    back_index_sample = back_index[random_indices]
    # concat
    all_index = torch.cat([fore_index_sample, back_index_sample])

    return segs_tar, feat_tar_ori[:, all_index], segs_tar[:, all_index]


def read_feature(path, frame_index, return_h_w=False):
    """Extract one frame feature everytime."""
    data = torch.load(path, weights_only=True).to("cuda").float()
    data = data[frame_index]
    _h, _w, _ = data.shape
    data = data.view(_h * _w, -1).contiguous()

    if return_h_w:
            return data, _h, _w
    return data


def norm_mask(mask):
    # norm to 0~1
    c, _, _ = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _, h, w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0).cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=0.2, type=float, help='The temperature for softmax.')
    parser.add_argument("--n_last_frames", type=int, default=9, help="The numbers of anchor frames.")
    parser.add_argument("--topk", type=int, default=15, help="The hyper-parameters of KNN top k.")
    parser.add_argument("--sample_ratio", type=float, default=0.3, help="The sample ratio of mask propagation.")
    parser.add_argument("--num_frames", type=int, default=16, help="The total nums of mask.")
    #
    parser.add_argument("--mask_path", type=str, default='examples/masks/mallard-fly.png', help="The path of first frame.")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--feature_path", type=str, default=None, help="The path of inversion feature map.")
    parser.add_argument("--output_path", type=str, default=None, help="The path of output.")
    args = parser.parse_args()
    # -------------------------------------------------------------------------------------------------------------
    video_mask_propogation(args)
