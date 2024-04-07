import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os

from sklearn.metrics.cluster import  mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score


def get_heatmap_mi(img_gr,size_x,size_y,winsize,strd,i):
    heatmap_mi = np.zeros_like(img_gr, dtype=np.float32)
    stride_x = int(strd)
    stride_y = int(strd)
    win_size_x = int(winsize)
    win_size_y = int(winsize)
    ents = []
    img_gr = cv2.copyMakeBorder(img_gr, int(win_size_y/2), int(win_size_y/2), int(win_size_x/2), int(win_size_x/2), cv2.BORDER_REFLECT)
    for x in range(0, size_x - win_size_x,stride_x):
        for y in range(0, size_y - win_size_y,stride_y):
            window_cur = img_gr[x:x+win_size_x,y:y+win_size_y].flatten()
            #===============================mutual_info======================jlh
            mi_sum = 0
            neighbor_count = 0
            if (x-win_size_x) >= 0: #left
                window_cur_left = img_gr[x-win_size_x:x,y:y+win_size_y].flatten()
                mi_left = mutual_info_score(window_cur, window_cur_left)
                # mi_left = normalized_mutual_info_score(window_cur, window_cur_left)
                # mi_left = adjusted_mutual_info_score(window_cur, window_cur_left)
                neighbor_count += 1
                mi_sum += mi_left
            if (y-win_size_y) >= 0: #up
                window_cur_up = img_gr[x:x+win_size_x,y-win_size_y:y].flatten()
                mi_up = mutual_info_score(window_cur, window_cur_up)
                # mi_up = normalized_mutual_info_score(window_cur, window_cur_up)
                # mi_up = adjusted_mutual_info_score(window_cur, window_cur_up)
                neighbor_count += 1
                mi_sum += mi_up
            if (x + win_size_x*2) < size_x: #right
                window_cur_right = img_gr[x+win_size_x:x+win_size_x*2,y:y+win_size_y].flatten()
                mi_right = mutual_info_score(window_cur, window_cur_right)
                # mi_right = normalized_mutual_info_score(window_cur, window_cur_right)
                # mi_right = adjusted_mutual_info_score(window_cur, window_cur_right)
                neighbor_count += 1
                mi_sum += mi_right
            if (y + win_size_y*2) < size_y: #down
                window_cur_down = img_gr[x:x+win_size_x,y+win_size_y:y+win_size_y*2].flatten()
                mi_down = mutual_info_score(window_cur, window_cur_down)
                # mi_down = normalized_mutual_info_score(window_cur, window_cur_down)
                # mi_down = adjusted_mutual_info_score(window_cur, window_cur_down)
                neighbor_count += 1
                mi_sum += mi_down

            win_mi = mi_sum / neighbor_count
            #===================================================================

            heatmap_mi[x:x+win_size_x,y:y+win_size_y] = win_mi
            ents.append(win_mi)
            # print(win_mi)

    x_exc = heatmap_mi.shape[0] - size_x
    y_exc = heatmap_mi.shape[1] - size_y
    heatmap_mi = heatmap_mi[round(x_exc / 2):size_x+round(x_exc / 2)-1,round(y_exc / 2):size_y+round(y_exc / 2)-1]
    print(heatmap_mi.shape)

    return heatmap_mi

# input mi heatmap
def img_heatmap_mi(impath):
    colorIm = cv2.imread(impath)
    greyIm=cv2.cvtColor(colorIm,cv2.COLOR_BGR2GRAY)
    greyIm=np.array(greyIm)
    
    S=greyIm.shape
    E=np.array(greyIm)

    size_x = S[0]
    size_y = S[1]
        
    sx = np.ceil(size_x/100) + np.mod(np.ceil(size_x/100),2)
    sy = np.ceil(size_y/100) + np.mod(np.ceil(size_y/100),2)
        
    s1 = max(sx,sy)
    s2 = max(s1,8)
        
    ws = [s2,s2*1.5+np.mod(s2*1.5,2),s2*2]
    strd = []
    for a in ws:
        strd.append(a/2)
    #strd = ws/2
        
    area = size_x * size_y
    E = get_heatmap_mi(greyIm,size_x,size_y,ws[0],strd[0],0)
    # E = get_heatmap_mi(greyIm,size_x,size_y,40,strd[0],0)

    return E


if __name__ == "__main__":
    # 读图
    data_dir = "/home/dell/jlh/my_patch_defense/code/000/0125/"
    #data_dir = "proper_patched"
    data_files = os.listdir(data_dir)
    for data_file in data_files:
        print(data_file)
        name = data_file.split(".")[0]
        impath = data_dir + data_file

        E = img_heatmap_mi(impath)
        print(E.shape)

        E_max = np.max(E)
        E_min = np.min(E)
        print('max:', E_max)
        print('min:', E_min)
        E = (E-E_min)*25/(E_max-E_min)  # expand pixel to [0, 255]
        E = E.astype(int) # float-->int

        plt.subplot(1,1,1)
        plt.imshow(E, cmap=plt.cm.jet)
        plt.xlabel('mi')
        #plt.colorbar()
        plt.savefig("/home/dell/jlh/my_patch_defense/code/000/0125/"+name+"mi.png")
        # plt.show()


















