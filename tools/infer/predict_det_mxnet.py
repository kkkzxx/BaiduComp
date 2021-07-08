import mxnet as mx
from mxnet import gluon
import cv2
import numpy as np
import time
import sys
import math
import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from ppocr.utils.logging import get_logger
from tools.infer.project_utils import PostProcess
import tools.infer.utility as utility
from ppocr.utils.utility import get_image_file_list, check_and_read_gif

logger = get_logger()

class TextDetector(object):
    def __init__(self):
        self.mxnet_detect_model_path = '/home/kkkzxx/Experiment/PaddleOCR-release-2.1/inference/mxnet_model/resnet50-db-symbol.json'
        self.mxnet_detect_param_path = '/home/kkkzxx/Experiment/PaddleOCR-release-2.1/inference/mxnet_model/resnet50-db-0000.params'
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.ctx = mx.gpu(0)
        self.detector = gluon.SymbolBlock.imports(
            self.mxnet_detect_model_path, ['data'], self.mxnet_detect_param_path, ctx=self.ctx)

        self.detect_short_side = 736
        self.max_side = 1024
        self.post_process = PostProcess(thresh=0.3, box_thresh=0.3)

    def resize_detect_img(self, img, scale, min_divisor=32, max_scale=1440):
        height, width = img.shape[:2]
        if height < width:
            new_height = scale
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
            if new_width > max_scale:
                new_width = max_scale
                new_height = int(
                    math.ceil(
                        new_width /
                        width *
                        height /
                        32) *
                    32)
        else:
            new_width = scale
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
            if new_height > max_scale:
                new_height = max_scale
                new_width = int(
                    math.ceil(
                        new_height /
                        height *
                        width /
                        32) *
                    32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def __call__(self, img):
        start_time=time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origh_h, origin_w = img.shape[:2]
        img = self.resize_detect_img(img, self.detect_short_side)
        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self.mean, std=self.std)
        img = img.expand_dims(0).as_in_context(self.ctx)
        outputs = self.detector(img)
        pred = outputs[0].asnumpy()[0, 0]
        boxes, scores = self.post_process.boxes_from_bitmap(
            pred, origin_w, origh_h)
        boxes = boxes.astype('float32')
        elapse=time.time()-start_time
        return boxes,elapse

if __name__ == "__main__":
    image_file_list = get_image_file_list('/home/kkkzxx/Experiment/PaddleOCR-release-2.1/test_images/projects/0806')
    text_detector = TextDetector()
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        dt_boxes, elapse = text_detector(img)
        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))
        src_im = utility.draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        logger.info("The visualized image saved in {}".format(img_path))
    if count > 1:
        logger.info("Avg Time: {}".format(total_time / (count - 1)))