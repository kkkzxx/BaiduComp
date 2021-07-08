import argparse
import os, sys
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import math
import re
import difflib
from shapely.geometry import Polygon
import pyclipper

class PostProcess(object):
    def __init__(self, thresh=0.3, box_thresh=0.7,
                 max_candidates=300, **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        # self.min_size = 3.9
        self.min_size = 1.0
        self.scale_ratio = 0.4
        # self.min_area = 40
        self.min_area = 10

    def polygons_from_bitmap(self, pred, dest_width, dest_height):
        '''
        pred: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        if pred.shape[0] == 1:
            pred = pred[0]
        bitmap = pred > self.thresh
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        if pred.shape[0] == 1:
            pred = pred[0]

        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        bitmap = pred > self.thresh
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        boxes = []
        scores = []

        for index in range(num_contours):

            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            area = cv2.contourArea(box)
            if area < self.min_area:
                continue
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            # boxes[index, :, :] = box.astype(np.int16)
            # scores[index] = score
            boxes.append(box.astype(np.int16))
            scores.append(score)
        boxes = np.array(boxes, dtype=np.int16)
        scores = np.array(scores)
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.8):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def order_points_clockwise(pts):
    num=len(pts)
    rect = np.zeros((num,4, 2), dtype="float32")
    for i,pt in enumerate(pts):
        s = pt.sum(axis=1)
        rect[i,0] = pt[np.argmin(s)]
        rect[i,2] = pt[np.argmax(s)]
        diff = np.diff(pt, axis=1)
        rect[i,1] = pt[np.argmin(diff)]
        rect[i,3] = pt[np.argmax(diff)]
    return rect

def get_lines_mask(binary_image, which='v'):
    assert which in ('v', 'h')
    # Create the images that will use to extract the horizontal or vertical lines
    img = np.copy(binary_image)
    h, w = img.shape[:2]
    # Specify size on horizontal axis
    if which == 'v':
        kernel_tuple = (1, h // 30)
    else:
        kernel_tuple = (w // 30, 1)
    # Create structure element for extracting vertical lines through morphology operations
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_tuple)
    # Apply morphology operations
    img = cv2.erode(img, structure)
    img = cv2.dilate(img, structure)
    # Inverse vertical image
    img = cv2.bitwise_not(img)
    # extract edges
    edges = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    # dilate(edges)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel)
    # src.copyTo(smooth)
    smooth = np.copy(img)
    # blur smooth img
    smooth = cv2.blur(smooth, (2, 2))
    # smooth.copyTo(src, edges)
    (rows_or_cols, cols) = np.where(edges != 0)
    img[rows_or_cols, cols] = smooth[rows_or_cols, cols]
    img = cv2.bitwise_not(img)
    thresh, mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    mask = mask.astype(bool)
    return mask


def line_angle(line):
    if len(line.shape) == 1:
        gradient = (line[3] - line[1]) / (line[2] - line[0])
    else:
        gradient = (line[:, 3] - line[:, 1]) / (line[:, 2] - line[:, 0] + 0.00001)
    line_angle = np.arctan(gradient) / np.pi * 180
    return line_angle


def get_lines(img):
    output_lines = []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    h_edges = get_lines_mask(bw, 'h')
    h_edges = h_edges * 255
    h_edges = h_edges.astype(np.uint8)
    h_lines = cv2.HoughLinesP(h_edges, 1, np.pi / 360, int(h / 10), minLineLength=int(h / 10), maxLineGap=int(h / 50))

    v_edges = get_lines_mask(bw, 'v')
    v_edges = v_edges * 255
    v_edges = v_edges.astype(np.uint8)
    v_lines = cv2.HoughLinesP(v_edges, 1, np.pi / 360, int(w / 10), minLineLength=int(w / 10), maxLineGap=int(w / 50))

    if h_lines is not None:
        lines = h_lines[:, 0, :].tolist()
        sort_lines = np.ones_like(lines)
        for i, line in enumerate(lines):
            if line[0] <= line[2]:
                sort_lines[i, :] = line
            else:
                sort_lines[i, :] = np.array([line[2], line[3], line[0], line[1]])
        h_lines = sort_lines

        h_index = np.array(list(range(len(h_lines))))
        h_clusters = []
        while h_index.size > 0:
            sub_cluster = []
            i = h_index[0]
            sub_cluster.append(i)
            y_dist = abs(h_lines[i, 1] - h_lines[h_index[1:], 1])
            dist_idx = np.where(abs(y_dist) < int(h / 100))[0]
            angle_diff = abs(line_angle(h_lines[i]) - line_angle(h_lines[h_index[1:]]))
            angle_idx = np.where(angle_diff < 3)[0]
            merge_idx = np.intersect1d(dist_idx, angle_idx)
            # leave_idx = np.setdiff1d(h_index,merge_idx+1)
            # leave_idx=np.setdiff1d(leave_idx,h_index[0])
            merge_idx = (merge_idx + 1).tolist()
            sub_cluster.extend(h_index[merge_idx])
            h_clusters.append(sub_cluster)
            leave_idx = np.setdiff1d(h_index, np.array(sub_cluster))
            h_index = leave_idx
        h_lines_cluster = []
        for cl in h_clusters:
            line_cluster = []
            cl_line = h_lines[cl]
            min_x = np.argsort(cl_line[:, 0])
            max_x = np.argsort(cl_line[:, 2])
            start = cl_line[min_x[0], :2]
            end = cl_line[max_x[-1], 2:]
            line_cluster.extend(list(start))
            line_cluster.extend(list(end))
            h_lines_cluster.append(line_cluster)
    else:
        h_lines_cluster = []

    if v_lines is not None:
        lines = v_lines[:, 0, :].tolist()
        sort_lines = np.ones_like(lines)
        for i, line in enumerate(lines):
            if line[1] <= line[3]:
                sort_lines[i, :] = line
            else:
                sort_lines[i, :] = np.array([line[2], line[3], line[0], line[1]])
        v_lines = sort_lines
        v_index = np.array(list(range(len(v_lines))))
        v_clusters = []
        while v_index.size > 0:
            sub_cluster = []
            i = v_index[0]
            sub_cluster.append(i)
            w_dist = abs(v_lines[i, 0] - v_lines[v_index[1:], 0])
            idx = np.where(abs(w_dist) < int(w / 80))[0]
            leave_idx = np.where(abs(w_dist) >= int(w / 80))[0]
            idx = (idx + 1).tolist()
            sub_cluster.extend(v_index[idx])
            v_clusters.append(sub_cluster)
            leave_idx = (leave_idx + 1).tolist()
            v_index = v_index[leave_idx]
        v_lines_cluster = []
        for cl in v_clusters:
            line_cluster = []
            cl_line = v_lines[cl]
            min_x = np.argsort(cl_line[:, 1])
            max_x = np.argsort(cl_line[:, 3])
            start = cl_line[min_x[0], :2]
            end = cl_line[max_x[-1], 2:]
            line_cluster.extend(list(start))
            line_cluster.extend(list(end))
            v_lines_cluster.append(line_cluster)
    else:
        v_lines_cluster = []

    new_h_line_cluster = []
    if len(h_lines_cluster) > 0:
        for line in h_lines_cluster:
            if h / 20 < line[1] < (h - h / 20):
                new_h_line_cluster.append(line)
    new_v_line_cluster = []
    if len(v_lines_cluster) > 0:
        for line in v_lines_cluster:
            if w / 20 < line[0] < (w - w / 20):
                new_v_line_cluster.append(line)

    # for line in new_h_line_cluster:
    #     x1, y1, x2, y2 = line
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # for line in new_v_line_cluster:
    #     x1, y1, x2, y2 = line
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    # cv2.imshow('out', img)
    # cv2.waitKey(0)

    output_lines.append(new_v_line_cluster)
    output_lines.append(new_h_line_cluster)

    return output_lines


def split_connect_box(dt_boxes, lines):
    new_boxes = []

    vertical_line_x = [a[0] for a in lines]
    vertical_lines_top = [a[1] for a in lines]
    vertical_lines_down = [a[3] for a in lines]
    for dt_box in dt_boxes:
        for x, top, down in zip(vertical_line_x, vertical_lines_top, vertical_lines_down):
            # if dt_box[0][0]<x<dt_box[1][0] and dt_box[0][1]>top and dt_box[3][1]<down:
            if dt_box[0][0] < x < dt_box[1][0] and dt_box[2][1] > top:
                top_y = dt_box[0][1] + (x - dt_box[0][0]) / (dt_box[1][0] - dt_box[0, 0]) * (
                            dt_box[1][1] - dt_box[0][1])
                down_y = dt_box[3][1] + (x - dt_box[3][0]) / (dt_box[2][0] - dt_box[3, 0]) * (
                            dt_box[2][1] - dt_box[3][1])
                top_y = int(top_y)
                down_y = int(down_y)
                if abs(x - dt_box[0][0]) > 1:
                    left_box = np.array([dt_box[0], [x - 0.01, top_y], [x - 0.01, down_y], dt_box[3]])
                    right_box = np.array([[x + 0.01, top_y], dt_box[1], dt_box[2], [x + 0.01, down_y]])
                    new_boxes.append(left_box)
                    new_boxes.append(right_box)
                    break
                else:
                    continue
        else:
            new_boxes.append(dt_box)
    return np.array(new_boxes)


def load_check_dict(check_path):
    with open(check_path) as f:
        file = f.readlines()
    file = [content.strip() for content in file]
    return file


def is_all_chinese(text):
    for t in text:
        if u'\u4e00' <= t <= u'\u9fff':
            continue
        else:
            return False
    return True


def is_all_not_chinese(text):
    no_chinese = True
    for t in text:
        if u'\u4e00' <= t <= u'\u9fff':
            return False
    return no_chinese


def revise_texts(boxes, texts, scores, check_texts):
    new_texts = []
    new_boxes = []
    new_scores = []
    for index, (box, text, score) in enumerate(zip(boxes, texts, scores)):
        text_raw = text
        have_percent = False
        if text != '' and text[-1] == '儿':
            text = text.replace('儿', '/L')
        if '%' in text and list(text).index('%') > int(len(text) * 0.5):
            have_percent = True
        if 'mo1' in text:
            text = text.replace('mo1', 'mol')
        if "细" in text and "脑" in text:
            text = text.replace("脑", "胞")
        if '期' in text and '口' in text:
            text = text.replace('口', '日')
        if "肌" in text:
            if "配" in text:
                text = text.replace("配", "酐")
            if '酉' in text:
                text = text.replace('酉', '酐')
        if "诊" in text and "内" in text and not '科' in text:
            text = text.replace("内", "门")
        if "细胞" in text and "已" in text:
            text = text.replace("已", "巴")
        if text.startswith('个'):
            text_list = list(text)
            text_list[0] = '↑'
            text = ''.join(text_list)
        if text.endswith('个'):
            text_list = list(text)
            text_list[-1] = '↑'
            text = ''.join(text_list)
        if text == '个':
            text = text.replace('个', '↑')
        if text == '业':
            text = text.replace('业', '↓')
        if text == '平':
            text = text.replace('业', '↑')
        if text == '山':
            text = text.replace('山', '↓')
        # if '↑' in text:
        #     text=text.replace('↑','')
        # if '↓' in text:
        #     text = text.replace('↓', '')
        if '←' in text:
            text = text.replace('←', '')
        if '→' in text:
            text = text.replace('→', '')
        if "A" in text:
            text_list = list(text)
            A_index = text_list.index("A")
            if A_index > 1:
                if text_list[A_index - 1] == '0' and text_list[A_index - 2] == '1':
                    text = text.replace("A", '^')
        if 'l' in text:
            if len(text) > 1:
                if text[-2] == 'l':
                    text_list = list(text)
                    if text_list[-1] == ']' or text_list[-1] == '[':
                        text_list[-1] = 'l'
                        text = ''.join(text_list)
        try:
            text_list = list(text)
            idx = text_list.index('/')
            if text_list[idx + 1] == '1' or text_list[idx + 1] == 'I':
                if idx == (len(text_list) - 2) and not u'\u4e00' <= text[idx - 2] <= u'\u9fff':
                    text_list[idx + 1] = 'L'
            text = ''.join(text_list)
        except:
            pass

        if ':' in text or '：' in text:
            if not '2019' in text and not '2020' in text:
                if not u'\u4e00' <= text[0] <= u'\u9fff':
                    text = text.strip()
                    text = text.replace('：', ':')
                    texts = text.split(':')
                    if len(texts[0]) != 0 and len(texts[-1]) != 0:
                        text = text.replace(':', '.')

        res = re.findall('\d{1,10}个', text)
        if res:
            text = text.replace('个', '↑')

        # 判断是否存在x10的幂
        res = re.findall("x?10[\S]{0,3}/L", text)
        if res and 'mg' not in text and '^' not in text and 'E' not in text:
            res_raw = res[0]
            not_need = re.findall("[^a-z^0-9^A-Z^/]", res_raw)
            if not_need:
                res = res_raw.replace(not_need[0], '')
            else:
                res = res_raw
            res_new = list(res)
            if res_new[0] == 'x':
                if len(res_new) == 5:
                    res_new.insert(3, '#')
                res_new.insert(3, '^')
            else:
                if len(res_new) == 4:
                    res_new.insert(2, '#')
                res_new.insert(2, '^')
            res_new = ''.join(res_new)
            text = text.replace(res_raw, res_new)

        res = re.findall(r'[^x^×^X^ ]10[\S][\d|#]{0,3}/L', text)
        if res and '*' not in text:
            sapce_text = list(res[0])
            sapce_text.insert(1, 'x')
            sapce_text = ''.join(sapce_text)
            text = text.replace(res[0], sapce_text)

        # 2021.3.1
        if 'mo' in text or 'mg' in text:
            text = text.replace('mo1', 'mol')
            res = re.findall("^[\S]{0,20}L\s*[\d]{1,8}", text)
        else:
            res = re.findall("x?10[\S]{0,3}/?L\s*\S{1,8}", text)
        if res:
            text_list = list(text)
            L_index = text_list.index('L')
            unity = text_list[:L_index + 1]
            unity = ''.join(unity)
            value = text[L_index + 1:]
            value = ''.join(value)

            unity_len_ratio = (L_index + 1) / len(text)
            up_line_width = box[1][0] - box[0][0]
            down_line_width = box[2][0] - box[3][0]
            up_line_height = box[1][1] - box[0][1]
            down_line_height = box[2][1] - box[3][1]

            x1 = box[0][0] + int(up_line_width * unity_len_ratio)
            y1 = box[0][1] + int(up_line_height * unity_len_ratio)
            x2 = box[3][0] + int(down_line_width * unity_len_ratio)
            y2 = box[3][1] + int(down_line_height * unity_len_ratio)

            box1 = np.zeros_like(box)
            box2 = np.zeros_like(box)
            box1[0] = box[0]
            box1[1] = np.array([x1, y1])
            box1[2] = np.array([x2, y2])
            box1[3] = box[3]
            box2[0] = np.array([x1, y1])
            box2[1] = box[1]
            box2[2] = box[2]
            box2[3] = np.array([x2, y2])

            similarity = 0.79
            if is_all_not_chinese(unity) and 'E' not in unity:
                for check in check_texts[945:]:
                    text_check_similarity = difflib.SequenceMatcher(None, unity, check).ratio()
                    if text_check_similarity > similarity:
                        unity = check
                        similarity = text_check_similarity

            new_texts.append(unity)
            new_texts.append(value)
            new_boxes.append(box1)
            new_boxes.append(box2)
            new_scores.append(score)
            new_scores.append(score)
            continue

        element_num = 0
        for ele in '钾钠氯钙镁':
            if ele in text:
                element_num += 1
        if element_num >= 2 and (box[2][1] - box[1][1]) > 1.5 * (box[1][0] - box[0][0]):
            text_len = len(text)
            box_height = box[2][1] - box[1][1]
            box_height_sub = int(box_height / text_len)
            left_top_y = box[0][1]
            right_top_y = box[1][1]
            for ele_idx in range(text_len):
                ele_box = np.zeros_like(box)
                ele_box[0] = np.array([box[0][0], left_top_y + ele_idx * box_height_sub])
                ele_box[1] = np.array([box[1][0], right_top_y + ele_idx * box_height_sub])
                ele_box[2] = np.array([box[1][0], right_top_y + (ele_idx + 1) * box_height_sub])
                ele_box[3] = np.array([box[0][0], left_top_y + (ele_idx + 1) * box_height_sub])
                new_texts.append(text[ele_idx])
                new_boxes.append(ele_box)
                new_scores.append(score)
            continue

        if have_percent:
            text = text.replace('%', '百分比')
        similarity = 0.79
        new_text = text
        if is_all_chinese(text):
            for check in check_texts:
                text_check_similarity = difflib.SequenceMatcher(None, text, check).ratio()
                if text_check_similarity > similarity:
                    new_text = check
                    similarity = text_check_similarity
        if is_all_not_chinese(text) and 'E' not in text:
            for check in check_texts[945:]:
                text_check_similarity = difflib.SequenceMatcher(None, text, check).ratio()
                if text_check_similarity > similarity:
                    new_text = check
                    similarity = text_check_similarity
        if have_percent:
            new_text = new_text.replace('百分比', '%')
        # if text_raw!=new_text:
        #     print('raw:',text_raw)
        #     print('new:',new_text)
        #     print('-'*50)
        new_texts.append(new_text)
        new_boxes.append(box)
        new_scores.append(score)
    return new_texts, np.array(new_boxes), new_scores


def small_rotate_degree(dt_boxes):
    point1 = dt_boxes[0]
    point2 = dt_boxes[1]
    point3 = dt_boxes[2]
    point4 = dt_boxes[3]
    if (point3[1] - point2[1]) < (point2[0] - point1[0]):  # 横长竖短
        y = point2[1] - point1[1]
        l = math.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)
    else:
        y = point1[0] - point4[0]
        l = math.sqrt((point4[1] - point1[1]) ** 2 + (point4[0] - point1[0]) ** 2)
    sin_theta = y / l
    # 需要逆时针旋转的角度
    theta = math.asin(sin_theta) * 180 / (math.pi)
    return theta


if __name__ == '__main__':
    img_path = '/home/kkkzxx/Projects/PaddleOCR-develop/test_paper/pdf_images/2.png'
    img = cv2.imread(img_path)
    get_lines(img)
