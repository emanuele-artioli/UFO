"""Standalone test script for UFO model (non-package usage).

For package usage, import from ufo:
    from ufo import segment_frames
    masks = segment_frames(frames, device='cuda:0')
"""
import os
from PIL import Image
import torch
from torchvision import transforms
from model_video import build_model
import numpy as np
import argparse


def run_test(gpu_id, model_path, datapath, save_root_path, group_size, img_size, img_dir_name):
    """Run UFO segmentation on image directories and save mask results."""
    device = torch.device(gpu_id)
    net = build_model(device).to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=gpu_id))
    net.eval()
    net = net.module.to(device)

    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_transform_gray = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.449], std=[0.226])
    ])
    
    with torch.no_grad():
        for p in range(len(datapath)):
            img_path = os.path.join(datapath[p], img_dir_name)
            
            if not os.path.exists(img_path):
                continue
                
            all_class = os.listdir(img_path)
            
            image_list, save_list = list(), list()
            for s in range(len(all_class)):
                class_path = os.path.join(datapath[p], img_dir_name, all_class[s])
                image_path = sorted(os.listdir(class_path))
                
                idx = []
                block_size = (len(image_path) + group_size - 1) // group_size
                for ii in range(block_size):
                    cur = ii
                    while cur < len(image_path):
                        idx.append(cur)
                        cur += block_size
                
                new_image_path = []
                for ii in range(len(image_path)):
                    new_image_path.append(image_path[idx[ii]])
                image_path = new_image_path
                
                image_list.append(list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path)))
                save_list.append(list(map(lambda x: os.path.join(save_root_path[p], all_class[s], x[:-4]+'.png'), image_path)))
            
            for i in range(len(image_list)):
                cur_class_all_image = image_list[i]
                
                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(len(cur_class_all_image), img_size, img_size)
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                
                if divided != 0:
                    for k in range(divided):
                        group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                        group_rgb = group_rgb.to(device)
                        _, pred_mask = net(group_rgb)
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
                        
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size - rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    group_rgb = group_rgb.to(device)
                    _, pred_mask = net(group_rgb)
                    cur_class_mask[(divided * group_size):] = pred_mask[:rested]

                class_save_path = os.path.join(save_root_path[p], all_class[i])
                if not os.path.exists(class_save_path):
                    os.makedirs(class_save_path)

                for j in range(len(cur_class_all_image)):
                    exact_save_path = save_list[i][j]
                    result = cur_class_mask[j, :, :].cpu().numpy()
                    result = Image.fromarray((result * 255).astype(np.uint8))
                    w, h = Image.open(image_list[i][j]).size
                    result = result.resize((w, h), Image.BILINEAR)
                    result.convert('L').save(exact_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./weights/video_best.pth', help="restore checkpoint")
    parser.add_argument('--data_path', default='./datasets/elvis/', help="dataset for evaluation")
    parser.add_argument('--output_dir', default='./VSOD_results/wo_optical_flow/elvis/', help='directory for result')
    parser.add_argument('--task', default='VSOD', help='task')
    parser.add_argument('--gpu_id', default='cuda:0', help='id of gpu')
    args = parser.parse_args()
    
    gpu_id = args.gpu_id
    device = torch.device(gpu_id)
    model_path = args.model
    val_datapath = [args.data_path]
    save_root_path = [args.output_dir]
    
    run_test(gpu_id, model_path, val_datapath, save_root_path, 5, 224, 'image')