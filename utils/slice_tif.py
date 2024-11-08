import os
import time

import numpy as np
import skimage.io
import pandas as pd
from PIL import Image
from PIL import ImageEnhance
Image.MAX_IMAGE_PIXELS = None


#裁剪路径下的tif文件为png图片，并保存到指定文件夹下，如果输出文件夹不存在，则创建，如果存在，则输出提示信息
#tif_path为输入路径，out_path为输出目录，out_name为输出文件夹


def slice_tif_(im_path, out_dir, out_name,
              sliceHeight=640, sliceWidth=640,
              overlap=0.1, out_ext='.png',pad = 0,
              overwrite=True,
              skip_highly_overlapped_tiles=False,enhance_contrast=True):

    #计时开始
    t0 = time.time()

    #读取文件
    image = skimage.io.imread(im_path)
    print("输入图片尺寸:", image.shape)

    win_h, win_w = image.shape[:2]
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    #通过带重叠率的滑窗获得裁剪目标，其本质是窗口内值的复制
    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):

            n_ims += 1
            if (n_ims % 100) == 0:
                print(n_ims)

            # 当遇到后边界时，不希望裁剪出太小的部分，因此通过后边界向内裁的方式
            if y0+sliceHeight > image.shape[0]:
                # 处于边界时，如果向内裁的部分比希望得到的尺寸的0.6倍还多，考虑放弃，这里选择的是false，即不管重叠多少，都会裁剪
                if skip_highly_overlapped_tiles:
                    if (y0+sliceHeight - image.shape[0]) > (0.6*sliceHeight):
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image.shape[1]:
                if skip_highly_overlapped_tiles:
                    if (x0+sliceWidth - image.shape[1]) > (0.6*sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            # 这里获得了要裁剪的部分在整幅图像上的位置
            xmin, xmax, ymin, ymax = x, x+sliceWidth, y, y+sliceHeight
            window_c = image[ymin:ymax, xmin:xmax]



            #window_c= window_c.astype(np.uint8)

            #拼接出输出路径来,out_dir是传入的完整路径，精确到图片文件夹，out_name是tif图片的前缀
            out_final = os.path.join(out_dir, out_name + '__' + str(y) + '_' + str(x) + '_'
                                     + str(sliceHeight) + '_' + str(sliceWidth)
                                     + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                                     + out_ext)


            #输出图片
            if not os.path.exists(out_final):
                skimage.io.imsave(out_final, window_c, check_contrast=False)
            elif overwrite:
                 skimage.io.imsave(out_final, window_c, check_contrast=False)
            else:
                print("文件已存在，overwrite设置为False，不进行覆盖")

    print("图片裁剪数量:", n_ims,
          "裁剪图片高为:{}，宽为:{}".format(sliceHeight, sliceWidth))
    print("裁剪用时:", im_path, time.time()-t0, '秒')
    return







def slice_to_png(tif_path, out_path,
                     Height = 640, Width = 640,
                     overlap = 0.1, out_ext = '.png',
                     overwrite = False,in_ext='tif',
                     skip_highly_overlapped_tiles=False):

    #计时
    t0 = time.time()

    #获取输入路径中的全部tif文件
    im_list = [z for z in os.listdir(tif_path) if z.endswith(in_ext)]

    #拼完整输出路径出来
    out_folder = out_path
    #不存在则创建，存在则给出提示信息
    if not os.path.exists(out_folder):
        print("将创建输出文件夹,路径为:", out_folder)
        os.makedirs(out_folder)

    #裁剪前的一些准备
    for i, im_name in enumerate(im_list):
        im_path = os.path.join(tif_path, im_name)
        im_tmp = skimage.io.imread(im_path)
        h, w = im_tmp.shape[:2]
        print("第{}张图片:{};".format(i+1, im_name), "高，宽:", h, w)

        out_name = im_name.split('.')[0]
        slice_tif_(im_path=im_path, out_dir=out_folder, out_name=out_name,
                  sliceHeight=Height, sliceWidth=Width,
                  overlap=overlap, out_ext=out_ext,
                  overwrite=overwrite,
                  skip_highly_overlapped_tiles=skip_highly_overlapped_tiles)
    out_list = []
    for f in sorted([z for z in os.listdir(out_folder) if z.endswith(out_ext)]):
        out_list.append(os.path.join(out_folder, f))
        df_tmp = pd.DataFrame({'image': out_list})



    print("进程结束，所用时间为:", time.time()-t0, "秒")


# if __name__ == "__main__":
#     slice_to_png(tif_path='/home/kingargroo/arieal/big', out_path='/home/kingargroo/corn/ninjin_crop', out_space='img')