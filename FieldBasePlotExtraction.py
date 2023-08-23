import cv2,time
import pandas as pd
import numpy as np

class FieldBasePlotExtraction(object):
    def __init__(self,raw_img):
        """
        :param raw_img: operate in the raw image
        """
        self.raw_img=raw_img
        self.h, self.w = self.raw_img.shape[0:2]
        # visualize in copy image
        self.copy_img = self.raw_img.copy()

    def PreposeImg(self):
        # convert the raw image to gray image
        self.raw_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        # binary the image
        ret, self.raw_img = cv2.threshold(self.raw_img, 100, 255, 0)
        # detect the edges
        self.edges = cv2.Canny(self.raw_img, 100, 200, apertureSize=3)

    def DetectVerticalLines(self,length_threshold=10,line_gap_thre=15,line_gap2_thre=15):

        # detect the vertical line with Hough detection algorithm
        lines = cv2.HoughLinesP(self.edges, 10, np.pi / 180, 200, np.array([]), minLineLength=1, maxLineGap=1800)

        #container for lines
        self.vertical_lines = []
        # add first vertical line
        self.vertical_lines.append([2, 0, 2, self.h])
        # store the center x of vertical line to reduce the redundant lines
        gap_lin_x = np.array([2]).reshape((-1, 1))

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # calculate slope
            k = (y2 - y1) / (x2 - x1 + 0.001)
            # convert to arch
            angle = np.arctan(k) * 180 / np.pi

            length = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

            # #get vertical line 90°/-90°/270°/-270°
            if (angle >= 89 and angle <= 91) or (angle >= -91 and angle <= -89) or (
                    angle >= -271 and angle <= -269) or (angle >= 269 and angle <= 271):

                if length > length_threshold:
                    x_center = (x1 + x2) / 2

                    # only keep one center line in each seperated region
                    gap = abs(x_center - gap_lin_x)

                    gap_judge = (gap < line_gap_thre)
                    if np.sum(gap_judge) > 0:
                        continue
                    else:
                        # keep the seperated line
                        gap_lin_x = np.concatenate([gap_lin_x, np.array(x_center).reshape((1, -1))], axis=0)
                        self.vertical_lines.append([x1 + 10, y1, x2 + 10, self.h])


        # detect the  vertical line again in case no detection at first
        for w_ in range(int(min(gap_lin_x[1::]))):
            region = raw_img[0:self.h, w_]
            total = np.sum(region)

            if total < 100:
                gap = abs(w_ - gap_lin_x)
                gap_judge = (gap < line_gap2_thre)
                if np.sum(gap_judge) > 0:
                    continue
                else:
                    gap_lin_x = np.concatenate([gap_lin_x, np.array(w_).reshape((-1, 1))], axis=0)
                    self.vertical_lines.append([w_ + 10, 0, w_ + 10, self.h])

    def DetectHorizontalLines(self,line_gap_thre=100):
        # horizontal seperation detecting horizontal lines
        self.horizontal_lines = []
        #add top and bottom  horizontal lines
        self.horizontal_lines.append([1, 5, self.w, 5, self.w])
        self.horizontal_lines.append([1, self.h - 5, self.w, self.h - 5, self.w])
        gap_lin_y = np.array([self.h - 5]).reshape((-1, 1))
        for h_ in range(self.h):
            region = self.raw_img[h_, 0:self.w]
            # count the pixel num
            total = np.sum(region)
            print(total)
            if total < 100:
                gap = abs(h_ - gap_lin_y)
                gap_judge = (gap < line_gap_thre)
                if np.sum(gap_judge) > 0:
                    continue
                else:
                    gap_lin_y = np.concatenate([gap_lin_y, np.array(h_).reshape((-1, 1))], axis=0)
                    self.horizontal_lines.append([0, h_ + 5, self.w, h_ + 5, self.w])


    def Visualize(self,save_path):
        # visualize the line to check
        vertical_lines, horizontal_lines = np.array(self.vertical_lines), np.array(self.horizontal_lines)
        vertical_lines = vertical_lines[vertical_lines[:, 0].argsort()]
        horizontal_lines = horizontal_lines[horizontal_lines[:, 1].argsort()]

        self.plot = np.array([0, 0, 0, 0]).reshape((1, -1))

        for i in range(vertical_lines.shape[0]-1):
            v1 = vertical_lines[i]
            x1, _, _, _ = v1
            v2 = vertical_lines[i + 1]
            x2, _, _, _ = v2

            for j in range(horizontal_lines.shape[0]):
                h1 = horizontal_lines[j]
                _, y1, _, _, _ = h1
                if j != horizontal_lines.shape[0] - 1:
                    h2 = horizontal_lines[j + 1]
                    _, y2, _, _, _ = h2
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w = x2 - x1
                    h = y2 - y1

                    cv2.rectangle(self.copy_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
                    plot_ = np.array([x1, y1, x2, y2]).reshape((1, -1))
                    self.plot = np.concatenate([self.plot, plot_], axis=0)
        self.plot = self.plot[1::]

        cv2.imwrite(save_path, self.copy_img)
    
    def Count(self,save_path):
        # count the plant number in each plot
        plot_number = np.zeros((self.plot.shape[0], 5))
        for area_index, area in enumerate(self.plot):
            x1, y1, x2, y2 = area
            w = x2 - x1
            h = y2 - y1
            roi = self.raw_img[y1:y1 + h, x1:x1 + w]
            plant_num = cv2.countNonZero(roi)
            plot_number[area_index, 4] = plant_num
            plot_number[area_index, 0:4] = np.array([x1, y1, x2, y2]).reshape((1, -1))

        df = pd.DataFrame(plot_number)
        df.columns = ['x1', 'y1', 'x2', 'y2', 'count']
        df.to_csv(save_path)



if __name__ == '__main__':
    raw_img = cv2.imread('/home/kingargroo/corn/out_results/M26/wenshang_result.jpg')
    extract_class=FieldBasePlotExtraction(raw_img=raw_img)
    extract_class.PreposeImg()
    extract_class.DetectVerticalLines(line_gap2_thre=10,line_gap_thre=20)
    extract_class.DetectHorizontalLines()
    extract_class.Visualize(save_path='/home/kingargroo/corn/out_results/M26/result3.jpg')
    extract_class.Count(save_path='/home/kingargroo/corn/out_results/M26/result1.csv')

        








