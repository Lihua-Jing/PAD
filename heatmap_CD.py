import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

savefig_path = "/home/dell/jlh/my_patch_defense/code/0double_jpeg/000/"

def recompress_diff(imorig, checkDisplacements):
    minQ = 51
    maxQ = 100
    stepQ = 1

    if checkDisplacements == 1:
        maxDisp = 7
    else:
        maxDisp = 0

    mins = []
    Output = []

    smoothing_b = 17
    Offset = (smoothing_b - 1) // 2

    # print(imorig.shape)
    height, width, _ = imorig.shape
    print("height , width", height, width)

    dispImages = []

    for ii in range(minQ, maxQ + 1, stepQ):
        cv2.imwrite('tmpResave.jpg', imorig, [int(cv2.IMWRITE_JPEG_QUALITY), ii])
        tmpResave = cv2.imread('tmpResave.jpg').astype(float)
        Deltas = []
        overallDelta = []

        for dispx in range(maxDisp + 1):
            for dispy in range(maxDisp + 1):
                DisplacementIndex = dispx * 8 + dispy + 1
                tmpResave_disp = tmpResave[dispx:, dispy:, :]
                imorig_disp = imorig[:height-dispx, :width-dispy, :].astype(float)
                # print('imorig_disp.shape', imorig_disp.shape)
                # print('tmpResave_disp.shape', tmpResave_disp.shape)
                Comparison = np.square(imorig_disp - tmpResave_disp)

                h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2
                Comparison = cv2.filter2D(Comparison, -1, h)

                Comparison = Comparison[Offset:-Offset, Offset:-Offset, :]
                Deltas.append(np.mean(Comparison, axis=2))
                overallDelta.append(np.mean(Deltas[DisplacementIndex - 1]))

        minOverallDelta, minInd = min(overallDelta), np.argmin(overallDelta)
        mins.append(minInd)
        Output.append(minOverallDelta)
        delta = Deltas[minInd]
        delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))

        dispImages.append(cv2.resize(delta.astype(np.float32), (delta.shape[1] // 4, delta.shape[0] // 4), interpolation=cv2.INTER_LINEAR))

    OutputY = Output
    OutputX = list(range(minQ, maxQ + 1, stepQ))
    xmax, imax, xmin, imin = cv2.minMaxLoc(np.array(OutputY))
    imin = sorted(imin)
    Qualities = [i * stepQ + minQ - 1 for i in imin]

    return OutputX, OutputY, dispImages, imin, Qualities, mins

def clean_up_image(filename):
    im = cv2.imread(filename)

    if len(im.shape) > 3:
        im = im[:, :, :, 0, 0, 0, 0]

    dots = filename.rfind('.')
    extension = filename[dots:]
    
    if extension.lower() == '.gif' and im.shape[2] < 3:
        im_gif, gif_map = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im_gif = im_gif[:, :, 0]
        im = np.uint8(cv2.cvtColor(im_gif, cv2.COLOR_GRAY2RGB) * 255)

    if im.shape[2] < 3:
        im[:, :, 1] = im[:, :, 0]
        im[:, :, 2] = im[:, :, 0]

    if im.shape[2] > 3:
        im = im[:, :, 0:3]

    if im.dtype == np.uint16:
        im = np.uint8(np.floor(im / 256))

    im_out = im

    return im_out

def img_heatmap_cd(impath):
    im = clean_up_image(impath)
    checkDisplacements = 0
    # smoothFactor = 1
    OutputX, OutputY, dispImages, imin, Qualities, Mins = recompress_diff(im, checkDisplacements)
    OutputMap = dispImages

    return OutputMap, OutputX

if __name__ == "__main__":
    data_dir = "/home/dell/jlh/ultralytics/ultralytics/datasets/inria/images/000/"
    data_files = os.listdir(data_dir)
    for data_file in data_files:
        print(data_file)
        name = data_file.split(".")[0]
        impath = data_dir + data_file

        OutputMap, OutputX = img_heatmap_cd(impath)
        print(len(OutputMap))

        # for ii in range(len(OutputMap)):
        #     print(OutputMap[ii].shape)
        #     print(OutputX[ii])
        #     plt.imshow(OutputMap[ii])
        #     plt.title(OutputX[ii])
        #     plt.savefig(savefig_path+name+str(OutputX[ii])+".png")

        average_OutputMap = np.mean(OutputMap, axis=0)
        # plt.imshow(average_OutputMap)
        # plt.title('average')
        # plt.savefig(savefig_path+name+"_average.png")

        OutputMap_max = np.max(average_OutputMap)
        OutputMap_min = np.min(average_OutputMap)
        print('max:', OutputMap_max)
        print('min:', OutputMap_min)

        out_height = len(average_OutputMap)
        out_width = len(average_OutputMap[0])
        print("out_height , out_width", out_height, out_width)

        average_OutputMap = [int((average_OutputMap[i][j]-OutputMap_min)*255/(OutputMap_max-OutputMap_min)) for i in range(out_height) for j in range(out_width)]
        # print(OutputMap)

        # translate into numpy array
        flatNumpyArray = np.array(average_OutputMap,dtype=np.uint8)
        # Convert the array to make a grayscale image(灰度图像)
        grayImage = flatNumpyArray.reshape(out_height, out_width)
        # show gray image
        print(grayImage)

        #resize to original size
        img = cv2.imread(impath)
        ori_height, ori_width, _ = img.shape
        print("ori_height , ori_width", ori_height, ori_width)
        grayImage = cv2.resize(grayImage, (ori_height, ori_width))

        cv2.imshow("GrayImage", grayImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

