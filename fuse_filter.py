import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os
import time

# from heatmap_MI import img_heatmap_mi
from heatmap_MI import img_heatmap_mi
from heatmap_CD import img_heatmap_cd

ratio_mi = 0.5 # ratio_cd = 1-ratio_mi
kernel_pram = 80
thresh_pram = 80 # percentile, from small to big
input_path = "/home/dell/jlh/my_patch_defense/code/000/0125/"
# input_path = "/home/dell/jlh/patch_attack/physical_video/20231106/hunhe/"
# savefig_path = "/home/dell/jlh/my_patch_defense/code/0fuse_"+str(ratio_mi)+'_'+str(thresh_pram)+"/APRICOT/"
savefig_path = "/home/dell/jlh/my_patch_defense/code/000/0125-out/"

def fuse_heatmap(impath, ori_height, ori_width):
    time_start = time.time()
    h_mi = img_heatmap_mi(impath)
    print('h_mi.shape', h_mi.shape)
    time_mi_end = time.time()
    # print('--------------mi cost %f s' %(time_mi_end-time_start))

    h_cd, qt = img_heatmap_cd(impath)
    h_cd = np.mean(h_cd, axis=0)
    print('h_cd.shape', h_cd.shape)
    time_cd_end = time.time()
    # print('--------------cd cost %f s' %(time_cd_end-time_mi_end))

    h_mi = cv2.resize(h_mi, (ori_width, ori_height))
    print('h_mi resize to ori size')
    # plt.imshow(h_mi)
    # plt.title('h_mi_resize')
    # plt.show()
    h_cd = cv2.resize(h_cd, (ori_width, ori_height))
    print('h_cd resize to ori size')
    # plt.imshow(h_cd)
    # plt.title('h_cd_resize')
    # plt.show()

    h_mi_max = np.max(h_mi)
    h_mi_min = np.min(h_mi)
    print('h_mi_max:', h_mi_max)
    print('h_mi_min:', h_mi_min)
    h_mi = [int((h_mi[i][j]-h_mi_min)*255/(h_mi_max-h_mi_min)) for i in range(len(h_mi)) for j in range(len(h_mi[0]))]

    h_cd_max = np.max(h_cd)
    h_cd_min = np.min(h_cd)
    print('h_cd_max:', h_cd_max)
    print('h_cd_min:', h_cd_min)
    h_cd = [int((h_cd[i][j]-h_cd_min)*255/(h_cd_max-h_cd_min)) for i in range(len(h_cd)) for j in range(len(h_cd[0]))]

    h_fuse = [int(h_mi[i]*ratio_mi + h_cd[i]*(1-ratio_mi)) for i in range(len(h_mi))]
    print('len(h_fuse)', len(h_fuse))

    time_fuse_end = time.time()
    # print('--------------fuse cost %f s' %(time_fuse_end-time_cd_end))


    h_fuse_flatNumpyArray = np.array(h_fuse,dtype=np.uint8)
    h_fuse_grayImage = h_fuse_flatNumpyArray.reshape(ori_height, ori_width)

    h_mi_flatNumpyArray = np.array(h_mi,dtype=np.uint8)
    h_mi_grayImage = h_mi_flatNumpyArray.reshape(ori_height, ori_width)

    h_cd_flatNumpyArray = np.array(h_cd,dtype=np.uint8)
    h_cd_grayImage = h_cd_flatNumpyArray.reshape(ori_height, ori_width)

    return h_mi_grayImage, h_cd_grayImage, h_fuse_grayImage

def heatmap_filter(heatmap, threshold, height, width):
    # thresh
    thresh,h_t = cv2.threshold(heatmap, threshold, maxval=255, type=cv2.THRESH_TOZERO)
    # cv2.imshow('thresh',img)
    # cv2.waitKey(0) #0为任意键位终止
    # cv2.destroyAllWindows()
    # cv2.imwrite(savefig_path+name+"_t.png", img)

    # compute base kernel size
    base_kernel_size = int(min(height, width)/kernel_pram)
    print(base_kernel_size)

    # MORPH_OPEN
    kernel=np.ones((base_kernel_size*2,base_kernel_size*2),np.uint8)
    # kernel=np.ones((base_kernel_size,base_kernel_size),np.uint8)
    h_t_o=cv2.morphologyEx(h_t, cv2.MORPH_OPEN,kernel, iterations=1)
    # cv2.imwrite(savefig_path+name+"_t_open.png", crosion)

    # MORPH_CLOSE
    kernel=np.ones((base_kernel_size,base_kernel_size),np.uint8)
    # kernel=np.ones((base_kernel_size*2,base_kernel_size*2),np.uint8)
    h_t_o_c=cv2.morphologyEx(h_t_o,cv2.MORPH_CLOSE,kernel, iterations=2)
    # cv2.imwrite(savefig_path+name+"_t_open_close.png", crosion2)

    # MORPH_OPEN
    kernel=np.ones((base_kernel_size*3,base_kernel_size*3),np.uint8)
    h_t_o_c_o=cv2.morphologyEx(h_t_o_c,cv2.MORPH_OPEN,kernel, iterations=2)
    # cv2.imwrite(savefig_path+name+"_t_open_close_open.png", crosion3)

    return h_t, h_t_o, h_t_o_c, h_t_o_c_o


if __name__ == "__main__":
    # 读图
    data_dir = input_path
    #data_dir = "proper_patched"
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

        # plt.figure()
        # plt.subplot(241)
        # plt.imshow(ori_img)
        # plt.title('ori_img')
        # plt.subplot(242)
        # plt.imshow(mi_img, cmap=plt.cm.jet)
        # plt.title('mi_heatmap')
        # plt.subplot(243)
        # plt.imshow(cd_img, cmap=plt.cm.jet)
        # plt.title('cd_heatmap')
        # plt.subplot(244)
        # plt.imshow(fuse_img, cmap=plt.cm.jet)
        # plt.title('fuse_heatmap')
        # plt.subplot(245)
        # plt.imshow(h_t)
        # plt.title('h_t')
        # plt.subplot(246)
        # plt.imshow(h_t_o)
        # plt.title('h_t_o')
        # plt.subplot(247)
        # plt.imshow(h_t_o_c)
        # plt.title('h_t_o_c')
        # plt.subplot(248)
        # plt.imshow(h_t_o_c_o)
        # plt.title('h_t_o_c_o')
        # # plt.show()
        # plt.savefig(savefig_path+name+".png")

        plt.imshow(ori_img)
        plt.title('ori_img')
        plt.savefig(savefig_path+name+"ori_img.png")
        plt.imshow(mi_img, cmap=plt.cm.jet)
        # plt.imshow(mi_img)
        plt.title('mi_heatmap')
        plt.savefig(savefig_path+name+"mi_heatmap.png")
        plt.imshow(cd_img, cmap=plt.cm.jet)
        plt.title('cd_heatmap')
        plt.savefig(savefig_path+name+"cd_heatmap.png")
        plt.imshow(fuse_img, cmap=plt.cm.jet)
        plt.title('fuse_heatmap')
        plt.savefig(savefig_path+name+"fuse_heatmap.png")
        plt.imshow(h_t, cmap=plt.cm.jet)
        plt.title('h_t')
        plt.savefig(savefig_path+name+"h_t.png")
        plt.imshow(h_t_o, cmap=plt.cm.jet)
        plt.title('h_t_o')
        plt.savefig(savefig_path+name+"h_t_o.png")
        plt.imshow(h_t_o_c, cmap=plt.cm.jet)
        plt.title('h_t_o_c')
        plt.savefig(savefig_path+name+"h_t_o_c.png")
        plt.imshow(h_t_o_c_o, cmap=plt.cm.jet)
        plt.title('h_t_o_c_o')
        plt.savefig(savefig_path+name+"h_t_o_c_o.png")

