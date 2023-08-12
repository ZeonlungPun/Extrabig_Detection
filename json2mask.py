import os
import json
import base64

import cv2
import imgviz
import PIL.Image
import os.path as osp

from tqdm import tqdm
from labelme import utils
from threading import Thread


def ConvertOne(labelme_dir, json_file, save_dir, label_name_to_value):
    out_dir = os.path.join(save_dir, json_file.replace(".json", ""))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    json_path = osp.join(labelme_dir, json_file)
    with open(json_path, "r") as jf:
        data = json.load(jf)
        imageData = data.get("imageData")

        # labelme 的图像数据编码以及返回处理格式
        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)

        for name, value in label_name_to_value.items():
            label_names[value] = name

        # label_names={'_background_','line','border'}
        lbl_viz = imgviz.label2rgb(
            lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
        )

        PIL.Image.fromarray(img).save(osp.join(out_dir, "img.jpg"))
        # 保存标签图片
        utils.lblsave(osp.join(out_dir, "label.png"), lbl)
        # 保存带标签的可视化图像
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.jpg"))

        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in label_names:
                f.write(lbl_name + "\n")


def main():
    labelme_dir = r'/home/kingargroo/corn/beauty/file'  # json文件存放文件夹
    save_dir = r'/home/kingargroo/corn/beauty/result'  # 结果生成文件夹
    # 类别标签
    class_names = {
        '_background_': 0,
        "beauty": 1,
        "book": 2,
        "bottle": 3,

    }
    # 列出labelme勾画标签后文件夹中保存的所有文件名
    file_list = os.listdir(labelme_dir)
    # 找到勾画保存的所有json标签
    json_list = []
    [json_list.append(x) for x in file_list if x.endswith(".json")]

    for json_file in tqdm(json_list):
        # 单线程
        # ConvertOne(labelme_dir, json_file, save_dir, class_names)

        # 多线程
        Thread(target=ConvertOne, args=(labelme_dir, json_file, save_dir, class_names)).start()
        print(f"生成结果保存地址：{save_dir}")


if __name__ == "__main__":
    #main()

    mask=cv2.imread('/home/kingargroo/corn/beauty/result/1/label.png',cv2.COLOR_BGR2GRAY)
    cv2.imwrite('mask_gray.jpg', mask)
    #print(mask)

