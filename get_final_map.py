import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os

from heatmap_MI import img_heatmap_mi
from heatmap_CD import img_heatmap_cd
from fuse_filter import fuse_heatmap, heatmap_filter

ratio_mi = 0.5 # ratio_cd = 1-ratio_mi
kernel_pram = 80
thresh_pram = 80 # percentile, from small to big
input_path = "/home/dell/jlh/ultralytics/ultralytics/datasets/inria/images/inria_P6/"
# input_path = "/home/dell/jlh/patch_attack/APRICOT/"
savefig_path = "P6_final_map0428/"


if __name__ == "__main__":
    # 读图
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
        # cv2.imshow("fuse_img", fuse_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if not os.path.exists(savefig_path):
            os.makedirs(savefig_path)

        threshold = np.percentile(fuse_img, thresh_pram)
        h_t, h_t_o, h_t_o_c, h_t_o_c_o = heatmap_filter(fuse_img, threshold, ori_height, ori_width)

        # cv2.imshow("h_t_o_c_o", h_t_o_c_o)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(savefig_path+name+"_htoco.png", h_t_o_c_o)

