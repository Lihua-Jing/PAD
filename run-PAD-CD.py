import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import glob
import os
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

import math
from PIL import Image

from heatmap_MI import img_heatmap_mi
from heatmap_CD import img_heatmap_cd
from fuse_filter import fuse_heatmap, heatmap_filter

iou_thre = 0.5
ratio_mi = 0.5 # ratio_cd = 1-ratio_mi
kernel_pram = 80
thresh_pram = 80 # percentile, from small to big
input_path = "/home/dell/jlh/ultralytics/ultralytics/datasets/inria/images/inria_P5/"
# image_path = "/home/dell/jlh/ultralytics/ultralytics/datasets/inria/images/inria_P3/"
# final_map_path = "/home/dell/jlh/my_patch_defense/code/inria_P3_final_map"
save_path = "defended_inria_P5-CD/"

def get_mask(image, mask_generator):
    
    masks = mask_generator.generate(image.astype(np.uint8))
    return masks

if __name__ == "__main__":
    device = "cuda:0"
    # sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
    sam = sam_model_registry["vit_l"](checkpoint="segment-anything/models/sam_vit_l_0b3195.pth")
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(save_path)
    folder = os.path.exists(save_path)

    if not folder:
        os.makedirs(save_path)

    with torch.no_grad():
        data_dir = input_path
        data_files = os.listdir(data_dir)
        for data_file in data_files:
            print(data_file)
            name = data_file.split(".")[0]
            impath = data_dir + data_file
            
            ori_img = Image.open(impath).convert('RGB')
            ori_width, ori_height = ori_img.size
            print("ori_height , ori_width", ori_height, ori_width)

            mi_img, cd_img, fuse_img = fuse_heatmap(impath, ori_height, ori_width)

            threshold = np.percentile(cd_img, thresh_pram)
            h_t, h_t_o, h_t_o_c, h_t_o_c_o = heatmap_filter(cd_img, threshold, ori_height, ori_width)

            gray = np.where(h_t_o_c_o >0,1,0)

            rgb_color = cv2.imread(impath)

            image = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB)
            
            #just for Dpatch
            #image = cv2.resize(image,(416,416))

            h = image.shape[0]
            w = image.shape[1]
            mask = get_mask(image, mask_generator)
            print(len(mask))

            result_mask = np.zeros((h,w))
            for k in range(len(mask)):

                mask_k = mask[k].get('segmentation')
                n = mask_k&gray
                u = mask_k #|gray
                iou = np.sum(n)/(np.sum(u))
                print("iou",iou)

                n_1 = mask_k&result_mask.astype(np.uint8)
                u_1 = mask_k
                iou1 =  np.sum(n_1)/(np.sum(u_1))
                print("iou1",iou1)

                if(iou>iou_thre and iou1<0.1):
                    mask_k_save = np.expand_dims(mask_k,axis=2)
                    mask_k_save = np.tile(mask_k_save,3)
                    rgb_color = rgb_color*(~mask_k_save) 
                    result_mask = result_mask.astype(np.uint8) | mask_k
                    '''mask_k_save = np.expand_dims(mask_k,axis=2)
                    mask_k_save = np.tile(mask_k_save,3)
                    mask_gray = np.expand_dims(mask_k*128,axis=2)
                    mask_gray = np.tile(mask_gray,3)
                    rgb_color = rgb_color*(~mask_k_save) + mask_gray
                    result_mask = result_mask.astype(np.uint8) | mask_k'''
                    '''result_mask = result_mask.astype(np.uint8) | mask_k
                    rgb_color = cv2.inpaint(rgb_color, mask_k.astype(np.uint8), 3, cv2.INPAINT_NS)'''

            cv2.imwrite(save_path+name+".png",rgb_color)
            #cv2.imwrite("./mask_hxx_0.05_gray/"+name+".png",cv2.inpaint(rgb_color, cv2.blur(result_mask.astype(np.uint8),(5,5)), 3, cv2.INPAINT_NS))
