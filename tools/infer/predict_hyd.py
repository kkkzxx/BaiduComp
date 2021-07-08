# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

# os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
from PIL import Image
import datetime
from collections import Counter
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt

from project_utils import load_check_dict,order_points_clockwise,small_rotate_degree,get_lines
from project_utils import split_connect_box,revise_texts

logger = get_logger()

class TextSystem(object):
    def __init__(self):
        args=utility.parse_args()
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.split_boxes = True
        self.revise = True
        self.visulization=True
        self.drop_score = args.drop_score
        self.font_path=args.vis_font_path
        if self.revise:
            self.check_texts = load_check_dict('./doc/laboratory_report_check.txt')
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
            rotate90_first=1
        else:
            rotate90_first=0
        return dst_img,rotate90_first

    def sorted_boxes(self, dt_boxes, img_height):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        hh = img_height // 70 // 10 * 10
        if hh == 0:
            hh = 10
        dt_boxes = sorted(dt_boxes, key=lambda x: (x[0][1] // hh, x[0][0] // hh))
        _boxes = list(dt_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def predict_angle(self,img):
        start_angle_time=time.time()
        dt_boxes, elapse = self.text_detector(img)
        dt_boxes=order_points_clockwise(dt_boxes)
        if dt_boxes is None:
            return None, None

        def max_side_diff(box):
            width = box[1][0] - box[0][0]
            height = box[3][1] - box[0][1]
            return abs(height - width)

        dt_boxes = dt_boxes.tolist()
        dt_boxes.sort(key=lambda x: max_side_diff(x), reverse=True)
        # if len(dt_boxes)>60:
        #     dt_boxes=dt_boxes[:60]
        dt_boxes = np.array(dt_boxes, dtype=np.float32)
        img_crop_list = []
        # dt_boxes = sorted_boxes(dt_boxes)
        rotate90_first_all = []
        small_degrees = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])

            small_degree = small_rotate_degree(dt_boxes[bno])
            small_degrees.append(small_degree)

            img_crop, rotate90_first = self.get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
            rotate90_first_all.append(rotate90_first)
        img_crop_list, angle_list, elapse = self.text_classifier(
            img_crop_list)
        rotate180_all = [int(angle[0] == '180') for angle in angle_list]
        angle_list_final = [rotate180_all[i]*2 + rotate90_first_all[i] for i in range(len(img_crop_list))]
        angle_img = Counter(angle_list_final).most_common()[0][0]

        degree = np.array(small_degrees).mean()
        degree = float(degree)

        logger.info("angle of img: {}, elapse : {}".format(angle_img, time.time()-start_angle_time))
        return angle_img,degree

    def __call__(self, image_file):
        logger.info('=' * 50)
        start_time=time.time()
        img = cv2.imread(image_file)

        angle_of_img = 0
        if self.use_angle_cls:
            angle_of_img, small_degree = self.predict_angle(img)
            if angle_of_img != 0 and angle_of_img is not None:
                img = np.rot90(img, angle_of_img)
            if abs(small_degree) > 1:
                im = Image.fromarray(img)
                im = im.rotate(small_degree)
                img = np.array(im)

        height, width, _ = img.shape
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None

        if self.split_boxes:
            lines = get_lines(img)
            lines_v, lines_h = lines[0], lines[1]
            if len(lines_v) != 0:
                dt_boxes = split_connect_box(dt_boxes, lines_v)
        else:
            lines_h = []

        dt_boxes = np.array(dt_boxes, dtype=np.float32)

        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes, img.shape[0])
        dt_boxes = np.array(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop, _ = self.get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))

        texts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_texts,filter_rec_scores = [], [], []
        for box, text,score in zip(dt_boxes,texts,scores):
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_texts.append(text)
                filter_rec_scores.append(score)

        if self.revise:
            filter_rec_texts,filter_boxes,filter_rec_scores = revise_texts(filter_boxes,
                                                                           filter_rec_texts,
                                                                           filter_rec_scores,
                                                                           self.check_texts)

        if self.visulization:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw_img = draw_ocr_box_txt(
                image,
                filter_boxes,
                filter_rec_texts,
                filter_rec_scores,
                drop_score=self.drop_score,
                font_path=self.font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            logger.info("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))
        end_time=time.time()
        logger.info("Predict time of %s: %.3fs" % (image_file, end_time-start_time))


if __name__=='__main__':
    image_file_list = get_image_file_list('/home/kkkzxx/Experiment/PaddleOCR-release-2.1/test_images/projects/1012/0.jpg')
    image_file_list.sort()
    text_sys=TextSystem()
    for img_path in image_file_list:
        text_sys(img_path)