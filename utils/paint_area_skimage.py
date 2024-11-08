from PIL import Image, ImageDraw,  ImageFont
import numpy as np
from skimage import io
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def paint_plot(img_path, plot_2_area, plot_4_area, row_list, col_list, out_path, normal_4 = (255, 0, 0), normal_2 = (255, 255, 0)):
    row_list_reversed = row_list[::-1]
    image = io.imread(img_path)
    pil_image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(pil_image)

    font = ImageFont.truetype("arial.ttf", 40)

    if 'pass' in plot_4_area.columns:
        plot_4_area.rename(columns={'pass': 'pass_'}, inplace=True)

    if 'pass' in plot_2_area.columns:
        plot_2_area.rename(columns={'pass': 'pass_'}, inplace=True)

    for dt in plot_4_area.itertuples(index=False):
        range = int(dt.Range)
        pass_ = int(dt.pass_)
        column_begin = int(dt.column_begin)
        count = int(dt.final_count)

        up = row_list[range-1]
        bottom = row_list[range]
        left = col_list[column_begin-2]
        right = col_list[column_begin+2]

        #绘图
        rect_coords = (4*left, 4*up, 4*right, 4*bottom)
        print(rect_coords)

        if count > 37:
            draw.rectangle(rect_coords, outline=normal_4, width=5)
        else:
            draw.rectangle(rect_coords, outline=(0, 0, 0), width=5)
        #写字
        text = f"{range}-{pass_}\nnum:{count}"

        position = (4*left, 4*bottom)
        color = (0, 0, 255)
        draw.text(position, text, fill=color, font=font)


    for dt in plot_2_area.itertuples(index=False):

        range = int(dt.Range)
        pass_ = int(dt.pass_)
        column_begin = int(dt.column_begin)
        count = int(dt.final_count)

        up = row_list[range - 1]
        bottom = row_list[range]
        left = col_list[column_begin - 1]
        right = col_list[column_begin + 1]

        rect_coords = (4 * left, 4 * up, 4 * right, 4 * bottom)
        print(rect_coords)
        if count > 37:
            draw.rectangle(rect_coords, outline=normal_2, width=5)
        else:
            draw.rectangle(rect_coords, outline=(0, 0, 0), width=7)

        # 写字
        text = f"{range}-{pass_}\nnum:{count}"

        position = (4 * left, 4 * bottom)
        color = (0, 0, 255)
        draw.text(position, text, fill=color, font=font)

    output_path = out_path
    pil_image.save(output_path)
    print('绘制结束')



if __name__ == '__main__':

    # data = {
    #     'range': [1, 1, 1],
    #     'pass': [1, 3, 5],
    #     'column_num_begin': [1, 5, 9]
    # }
    # df_4 = pd.DataFrame(data)
    # df_2 = pd.DataFrame(data)
    #
    # row_list = [0, 1000]
    # col_list = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]

    out_path = r"D:\Pycharm_Projects\tian_utils\maize_count\7788.png"

    df_4 = pd.read_csv(r"D:\Pycharm_Projects\tian_utils\images\linghaiqu\4.csv")
    df_2 = pd.read_csv(r"D:\Pycharm_Projects\tian_utils\images\linghaiqu\2.csv")


    row_list = [1, 330, 665, 1005, 1341, 1675, 2003, 2338, 2673, 3003, 3347, 3672, 4003, 4340, 4671, 5004, 5340, 5671, 6002, 6335, 6667, 7003, 7339, 7666, 8004, 8340, 8669, 9002, 9335, 9666, 10005, 10333, 10667, 10999, 11333, 11667,
 12003, 12333, 12667, 12998, 13335, 13668, 13999, 14339, 14666, 14999, 15333, 15669, 16004, 16334, 16667, 17001, 17335, 17669, 18001, 18333, 18667, 19002, 19334, 19673, 19999]
    row_list_reversed = row_list[::-1]

    col_list = [1, 28, 65, 95, 132, 168, 198, 242, 272, 302, 336, 368, 412, 439, 477, 508, 539, 572, 605, 639, 673, 706, 738, 771, 812, 847, 879, 908, 945, 973, 1013, 1050, 1075, 1112, 1146, 1178, 1213, 1245, 1280, 1313, 1345, 1386, 1419,
 1450, 1489, 1512, 1548, 1584, 1615, 1654, 1685, 1717, 1754, 1787, 1823, 1852, 1885, 1921, 1960, 1987, 2036, 2063, 2088, 2121, 2155, 2202, 2226, 2262, 2299, 2340, 2363, 2390, 2442]


    img_path = r"D:\Pycharm_Projects\tian_utils\images\linghaiqu\linghaizhengqu15-1.png"

    paint_plot(img_path=img_path, plot_2_area=df_2, plot_4_area=df_4, row_list=row_list_reversed, col_list=col_list, out_path=out_path)
    print('hhh')
