import os
import copy
import glob
import queue
import argparse
import numpy as np
from tqdm import tqdm

import gc
import cv2
import torch
from torch.nn import functional as F
from PIL import Image


@torch.no_grad()
def video_mask_propogation(args):
    color_palette = []
    with open("video_diffusion/palette.txt", 'r', encoding='utf-8') as file:
        for line in file:
            color_palette.append([int(i) for i in line.strip().split()])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)
    index2factor = {0:32, 1:16, 2:8, 3:8}
    scale_factor = index2factor[args.up_ft_index]
    # save path
    name = args.mask_path.split('/')[-1].split('.')[0]
    output_dir = os.path.join(args.output_dir, 'mask', name)
    os.makedirs(output_dir, exist_ok=True)
    # read first mask
    first_seg, seg_ori = read_seg(args.mask_path, scale_factor, [args.H, args.W])
    # -------------------------------------------------------------------------------------------
    # The queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)
    # first frame
    ori_h, ori_w = args.H, args.W
    # extract first frame feature 301
    frame1_feat = read_feature(args.feature_path, 0).T #  dim x h*w
    # saving first segmentation
    out_path = os.path.join(output_dir, "00000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    for cnt in tqdm(range(1, 16)):
        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg.squeeze(0).flatten(1).cuda()] + [pair[1].cuda() for pair in list(que.queue)]
        frame_tar_avg, seg_sample, feat_tar = label_propagation(args, used_frame_feats, used_segs, index=cnt)
        # pop out oldest frame if neccessary
        if que.qsize() == args.n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(seg_sample)
        que.put([feat_tar, seg])
        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)
        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
        frame_nm = f"%05d.png" % (cnt * 1)
        frame_tar_seg[frame_tar_seg != 0] = 255
        imwrite_indexed(os.path.join(output_dir, frame_nm), frame_tar_seg, color_palette)

def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

def label_propagation(args, list_frame_feats, list_segs, index=None):
    # ----------------------------------------------------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------------------------
    ## we only need to extract feature of the target frame
    feat_tar, h, w = read_feature(args.feature_path, index, return_h_w=True)
    # ----------------------------------------------------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------------------------
    # load mask
    segs = torch.cat(list_segs, dim=-1) # C x nmb_context*h*w
    C, _ = segs.shape
    nmb_context = len(list_segs)
    return_feat_tar = feat_tar.T # dim x h*w
    ncontext = len(list_frame_feats)

    feat_tar = F.normalize(feat_tar, dim=1, p=2).squeeze(0)
    feat_sources = torch.cat(list_frame_feats, dim=-1) # nmb_context x dim x h*w
    feat_sources = F.normalize(feat_sources, dim=0, p=2)

    aff = torch.exp(torch.mm(feat_tar, feat_sources) / args.temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)
    # nmb_context*h*w (source: keys) x h*w (tar: queries)
    aff = aff.transpose(1, 0)
    tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    # ----------------------------------------------------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------------------------
    # get mask
    seg_tar = torch.mm(segs, aff)
    # down sample for points of return_feat_tar
    fore_index = torch.where(seg_tar[0, :] != 0)[0]
    fore_nums = len(fore_index)
    back_index = torch.where(seg_tar[0, :] == 0)[0]
    back_nums = len(back_index)
    # generate random index
    random_indices = torch.randperm(len(fore_index))[: int(len(fore_index) * fore_nums / (fore_nums + back_nums) * args.sample_ratio)]
    # choice sub data from all data
    fore_index_sample = fore_index[random_indices]
    # ------------------------------------------------------------------------------------
    # generate random index
    random_indices = torch.randperm(len(back_index))[: int(len(back_index) * back_nums / (fore_nums + back_nums) * args.sample_ratio)]
    # choice sub data from all data
    back_index_sample = back_index[random_indices]
    # concat
    all_index = torch.cat([fore_index_sample, back_index_sample])
    # get sub data
    seg_sample = seg_tar[:, all_index]
    seg_tar = seg_tar.reshape(1, C, h, w)
    return_feat_tar = return_feat_tar[:, all_index]

    return seg_tar, seg_sample, return_feat_tar

def read_feature(path, frame_index, return_h_w=False):
    """Extract one frame feature everytime."""
    data = torch.load(path)[0].to("cuda").float()
    data = data.permute(1, 0, 2, 3)[frame_index]

    dim, h, w = data.shape
    data = torch.permute(data, (1, 2, 0)) # h,w,c
    data = data.view(h * w, dim) # hw,c

    if return_h_w:
            return data, h, w
    return data

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)

def read_frame_list(video_dir, flag='*.jpg'):
    frame_list = [img for img in glob.glob(os.path.join(video_dir, flag))]
    frame_list = sorted(frame_list)
    return frame_list

def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 32) * 32)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 32) * 32)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w

def read_seg(mask_path, scale_factor, scale_size=[480]):
    seg = Image.open(mask_path)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 32) * 32)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 32) * 32)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // scale_factor, _th // scale_factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), np.asarray(seg)

def color_normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--up_ft_index', default=2, type=int, help='Which upsampling block to extract the ft map.')
    parser.add_argument('--temperature', default=0.2, type=float, help='The temperature for softmax.')
    parser.add_argument("--n_last_frames", type=int, default=9, help="The numbers of anchor frames.")
    parser.add_argument("--topk", type=int, default=15, help="The hyper-parameters of KNN top k.")
    parser.add_argument("--sample_ratio", type=float, default=0.3, help="The sample ratio of mask propagation.")
    parser.add_argument("--H", type=int, default=512, help="The height of mask.")
    parser.add_argument("--W", type=int, default=512, help="The weight of mask.")
    parser.add_argument("--feature_path", type=str, default='output/features/libby/inversion_feature_301.pt', help="The path of ddim feature.")
    parser.add_argument("--mask_path", type=str, default='example/mask/libby.png', help="The path of first frame.")
    parser.add_argument("--output_dir", type=str, default='output', help="The path of output.")
    args = parser.parse_args()
    # -------------------------------------------------------------------------------------------
    video_mask_propogation(args)
