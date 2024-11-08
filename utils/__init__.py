#standardlize the modules and import the key functions

from FlipImage import FlipImage
from slice_tif import slice_to_png
from yolo_predict import predict_with_yolov8
from post_process import execute
from transform_streamline import ImageTransform
from SegementPlotsAndFirstCount import SegmentMain,Check_Consecutive
from StandardYieldOutput import GetStandardOutput
from ExtractFourLinesNum import GetFourLinesNum
from draw_dot_img import DrawAuxiliaryImages
