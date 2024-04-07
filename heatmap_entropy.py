import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os

from sklearn.metrics.cluster import  mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score

p_mean = [5.32,6.06,6.47]
p_std = [0.32,0.34,0.35]

c_mean = [2.75,3.19,3.47]
c_std = [1.16,1.13,1.07]
entropy_values = {'patch_mean':p_mean,'patch_stdev':p_std,
                        'clean_mean':c_mean,'clean_stdev':c_std}

# 熵
def entropy(signal):
    lensig=signal.size
    symset=list(set(signal))
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]#每个值的概率
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def get_entr_heatmap(img_gr,size_x,size_y,winsize,strd,entropy_values,i):
    entr_heatmap = np.zeros_like(img_gr, dtype=np.float32)
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
            # mi_sum = 0
            # neighbor_count = 0
            # if (x-win_size_x) >= 0: #left
            #     window_cur_left = img_gr[x-win_size_x:x,y:y+win_size_y].flatten()
            #     mi_left = mutual_info_score(window_cur, window_cur_left)
            #     # mi_left = normalized_mutual_info_score(window_cur, window_cur_left)
            #     # mi_left = adjusted_mutual_info_score(window_cur, window_cur_left)
            #     neighbor_count += 1
            #     mi_sum += mi_left
            # if (y-win_size_y) >= 0: #up
            #     window_cur_up = img_gr[x:x+win_size_x,y-win_size_y:y].flatten()
            #     mi_up = mutual_info_score(window_cur, window_cur_up)
            #     # mi_up = normalized_mutual_info_score(window_cur, window_cur_up)
            #     # mi_up = adjusted_mutual_info_score(window_cur, window_cur_up)
            #     neighbor_count += 1
            #     mi_sum += mi_up
            # if (x + win_size_x*2) < size_x: #right
            #     window_cur_right = img_gr[x+win_size_x:x+win_size_x*2,y:y+win_size_y].flatten()
            #     mi_right = mutual_info_score(window_cur, window_cur_right)
            #     # mi_right = normalized_mutual_info_score(window_cur, window_cur_right)
            #     # mi_right = adjusted_mutual_info_score(window_cur, window_cur_right)
            #     neighbor_count += 1
            #     mi_sum += mi_right
            # if (y + win_size_y*2) < size_y: #down
            #     window_cur_down = img_gr[x:x+win_size_x,y+win_size_y:y+win_size_y*2].flatten()
            #     mi_down = mutual_info_score(window_cur, window_cur_down)
            #     # mi_down = normalized_mutual_info_score(window_cur, window_cur_down)
            #     # mi_down = adjusted_mutual_info_score(window_cur, window_cur_down)
            #     neighbor_count += 1
            #     mi_sum += mi_down

            # win_entr = mi_sum / neighbor_count
            #===================================================================
            win_entr = entropy(window_cur)
            entr_heatmap[x:x+win_size_x,y:y+win_size_y] = win_entr
            ents.append(win_entr)
            print(win_entr)
    x_exc = entr_heatmap.shape[0] - size_x
    y_exc = entr_heatmap.shape[1] - size_y
    entr_heatmap = entr_heatmap[round(x_exc / 2):size_x+round(x_exc / 2)-1,round(y_exc / 2):size_y+round(y_exc / 2)-1]
    print(entr_heatmap.shape)
    # emax = np.max(entr_heatmap)
    # emin=np.min(entr_heatmap)
    # ediff = emax - emin
    # patch_mean = entropy_values['patch_mean'][i]
    # patch_std = entropy_values['patch_stdev'][i]
    # clean_mean = entropy_values['clean_mean'][i]
    # clean_stdev = entropy_values['clean_stdev'][i]
    # img_mean = np.mean(ents)
    # img_spread = (img_mean - clean_mean) / clean_stdev
    # dev_mult = ((patch_mean - clean_mean) / patch_mean) * 2
    # thr_base = (patch_mean + 1.5 * patch_std) - (dev_mult * patch_std)
    # entr_thr = thr_base + (img_spread * patch_std)
    # e_htmp = entr_heatmap - entr_thr
    # #print(e_htmp.shape)
    # #
    # mask = np.where(e_htmp<0,0,1)
    # e_htmp = e_htmp*mask
    # #print(e_htmp.shape)
    # #print(e_htmp)
    # #e_htmp = ~e_htmp
    return entr_heatmap



# 读图
data_dir = "/home/dell/jlh/my_patch_defense/code/000/test/"
#data_dir = "proper_patched"
data_files = os.listdir(data_dir)
for data_file in data_files:
            print(data_file)
            name = data_file.split(".")[0]
            # name = name[:len(name)-2]
            print(name)
            colorIm = cv2.imread(data_dir+data_file)
            #colorIm = cv2.cvtColor (colorIm, cv2.COLOR_BGR2RGB)
            #colorIm=Image.open('20201210_3.bmp')
            colorIm=np.array(colorIm)

            # 灰度
            greyIm=cv2.cvtColor(colorIm,cv2.COLOR_BGR2GRAY)
            greyIm=np.array(greyIm)
            # N=16
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
            E = get_entr_heatmap(greyIm,size_x,size_y,ws[0],strd[0],entropy_values,0)
            #plt.subplot(1,3,1)
            '''plt.imshow(colorIm)
            plt.subplot(1,3,2)
            plt.imshow(greyIm, cmap=plt.cm.gray)'''

            E_max = np.max(E)
            E_min = np.min(E)
            print('max:', E_max)
            print('min:', E_min)
            E = (E-E_min)*25/(E_max-E_min)  # expand pixel to [0, 255]
            E = E.astype(int) # float-->int

            plt.subplot(1,1,1)
            plt.imshow(E, cmap=plt.cm.jet)
            # plt.colorbar(label='color bar')
            plt.xlabel('entropy')
            #plt.colorbar()
            plt.savefig("/home/dell/jlh/my_patch_defense/code/000/output/"+name+"_entropy.png")
            #plt.show()


















