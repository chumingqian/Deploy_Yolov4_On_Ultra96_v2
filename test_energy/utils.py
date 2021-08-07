#  Copyright (C) 2020 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from ctypes import *
from typing import List
from pynq_dpu import DpuOverlay

import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys
import threading
import random


from pynq_dpu import DpuOverlay
import json
  
def sigmoid(x):
    return 1/(1+np.exp(-x))

class YoloV3_tf:
    def __init__(self, bit_name = "dpu.bit", model_name = "yolov3_tf_ultra96v2-b1600.xmodel"):
        """
            get input tensor and get four output tensor
        """
        #self.test_mAP = False
        self.test_mAP = True 
        #self.overlay = DpuOverlay("dpu.bit")
        self.overlay = DpuOverlay(bit_name)
        #self.overlay.load_model("yolov3_tf_ultra96v2-b1600.xmodel")
        self.overlay.load_model(model_name)
        self.runner = self.overlay.runner
        params = dict()
        params['num_classes'] = 20
        params['nms_thresh'] = 0.45
        params['anchor_cnt'] = 3
        params['biases'] = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 193, 373, 326]

        if self.test_mAP :
            params['conf_thresh'] = 0.005
        else :
            params['conf_thresh'] = 0.3
        self.params = params

        self.input_tensors = self.runner.get_input_tensors()
        self.input_dims = self.input_tensors[0].dims
        #print("input dims:", self.input_dims) 
        self.input_h = self.input_tensors[0].dims[1] 
        #print("input h:", self.input_h) 
        self.input_w = self.input_tensors[0].dims[2] 
        #print("input w:", self.input_w) 
        self.output_tensors = self.runner.get_output_tensors()


    def nms(self, boxes, box_confidences, conf_threshold, nms_threshold=0.5):
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]
        areas = width * height
        ordered = box_confidences.argsort()[::-1]
        keep = list()
        while ordered.size > 0:
            i = ordered[0]
            if box_confidences[i] < conf_threshold :
                ordered = ordered[1:]
                continue

            keep.append(i)
            xx1 = np.maximum(x_coord[i] - width[i] / 2, x_coord[ordered[1:]] -  width[ordered[1:]] / 2)
            yy1 = np.maximum(y_coord[i] - height[i] / 2, y_coord[ordered[1:]] - height[ordered[1:]] / 2)
            xx2 = np.minimum(x_coord[i] + width[i] / 2, x_coord[ordered[1:]] + width[ordered[1:]] / 2)
            yy2 = np.minimum(y_coord[i] + height[i] / 2 , y_coord[ordered[1:]] + height[ordered[1:]] / 2)
            width1 = np.maximum(0.0, xx2 - xx1)
            height1 = np.maximum(0.0, yy2 - yy1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)
            iou = intersection / union
            indexes = np.where((iou <= nms_threshold))[0]
            ordered = ordered[indexes + 1]
        keep = np.array(keep).astype(int)
        #print("keep:", keep)
        bboxes_keep = boxes[keep]
        #print("bboxes keep:", bboxes_keep.shape)
        conf_keep = box_confidences[keep]
        conf_keep = conf_keep.reshape(-1, 1) 
        #print("conf keep:", conf_keep.shape)
        bboxes_single_class = np.hstack((bboxes_keep, conf_keep))

        return bboxes_single_class 

    def transform_bbox(self, bboxes):
        bboxes_result = []
        for i in range(len(bboxes)):
            x = bboxes[i][0] - bboxes[i][2] / 2.0
            y = bboxes[i][1] - bboxes[i][3] / 2.0
            w = bboxes[i][2]
            h = bboxes[i][3]
            score = bboxes[i][4]
            label = int(bboxes[i][5])
            bboxes_result.append([x, y, w, h, label, score])
        return bboxes_result

    def get_bbox(self, bboxes, w, h):
        bboxes_result = []
        for i in range(len(bboxes)):
            x_min = bboxes[i][0] * w + 1
            y_min = bboxes[i][1] * h + 1
            x_max = (bboxes[i][0] + bboxes[i][2]) * w + 1
            y_max = (bboxes[i][1] + bboxes[i][3]) * h + 1
            if x_min < 0:
                x_min = 1
            if y_min < 0:
                y_min = 1
            if x_max > w: 
                x_max = w
            if y_max > h: 
                x_max = h
            label = int(bboxes[i][4])
            confidence = bboxes[i][5]
            bboxes_result.append([x_min, y_min, x_max, y_max, label, confidence])
        return bboxes_result
    
    """
    correct region boxes for yolov3_adas_pruned_0_9
    boxes: 2D array of boxes layout [[x_coord, y_coord, width, height],...]
    h, w: height and width of source image
    neth, netw: height and width of yolov3_adas_pruned_0_9 input
    num_classes: num of classes
    return: corrected boxes
    """
    def correct_region_boxes(self, boxes, h, w, neth, netw, num_classes = 3,relative = 0):
        new_w=0
        new_h=0
        if netw/w < neth/h:
            new_w = int(netw)
            new_h = int((h*netw) / w)
        else:
            new_h = int(neth)
            new_w = int((w * neth)/h)
        #print("w:", w, "h:", h)
        #print("netw:", netw, "neth:", neth)
        #print("new_w:", new_w, "new_h:", new_h)
        boxes[:,0] =  (boxes[:,0] - (netw - new_w)/2./netw) / (new_w/netw)
        boxes[:,1] =  (boxes[:,1] - (neth - new_h)/2./neth) / (new_h/neth)
        boxes[:,2] =  boxes[:,2] * netw/new_w
        boxes[:,3] =  boxes[:,3] * neth/new_h
        return boxes
    
    def letter_box(self, image, input_w, input_h):
        #print("image shape:", image.shape)
        scale = min(input_w / image.shape[1], input_h / image.shape[0])
        #print("scale:", scale)
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_w, new_h)) 
        rot_mat = np.zeros([2, 3])
        rot_mat[0][0] = 1
        rot_mat[0][2] = (input_w - new_w) / 2 
        rot_mat[1][1] = 1
        rot_mat[1][2] = (input_h - new_h) / 2
        new_image = cv2.warpAffine(image, rot_mat, (input_w, input_h), borderValue = (128, 128, 128))
        return new_image

    def preprocess(self, image):
        inputTensors = self.runner.get_input_tensors()
        input_w = inputTensors[0].dims[1]
        input_h = inputTensors[0].dims[2]
        #print("input tensor w, h:", input_w, input_h)
        if (self.test_mAP):
            image = self.letter_box(image, input_w, input_h)
        else:
            image = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            #image = np.transpose(np.array(image, dtype=np.float32, order='C'), (0, 1, 2))

        image = np.array(image, dtype=np.float32, order='C')
        image /= 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        return image

    def detect(self, output_buffer, height, width, anchor_index, input_h, input_w):
        #print('output_buffer shape:', output_buffer.shape)
        #print('output_buffer type:', type(output_buffer))
        #print('anchor index:', anchor_index)
        #print("input_h, input_w:", input_h, input_w) 
        conf_desigmoid = -math.log(1.0 / self.params['conf_thresh'] - 1.0)
        conf_box = 5 + self.params['num_classes'] # 25
        anchor_cnt = self.params['anchor_cnt'] # 3
        biases = self.params['biases']
        bboxes = []
        for h in range(height): 
            for w in range(width): 
                for c in range(anchor_cnt): 
                    idx = ((h * width + w) * anchor_cnt + c) * conf_box;
                    obj_score = output_buffer[h][w][c * conf_box + 4]
                    if obj_score < conf_desigmoid:
                        continue

                    box = []
                    box.append((w + sigmoid(output_buffer[h][w][c * conf_box + 0])) / width)
                    box.append((h + sigmoid(output_buffer[h][w][c * conf_box + 1])) / height)
                    bias_index = 2 * c + 2 * anchor_cnt * anchor_index
                    box.append(math.exp(output_buffer[h][w][c*conf_box + 2]) * biases[bias_index] / input_w)
                    box.append(math.exp(output_buffer[h][w][c*conf_box + 3]) * biases[bias_index + 1] / input_h)
                    #box.append(-1)
                    sigmoid_score = sigmoid(obj_score)
                    box.append(sigmoid_score)
                    for n in range(self.params['num_classes']):
                        box.append(sigmoid_score * sigmoid(output_buffer[h][w][c * conf_box + 5 + n]))
                    #print("h:", h, "w:", w, "c:", c, "anchor_index:", anchor_index, "bbox:", box)

                    bboxes.append(box)
 
        return bboxes

    def postprocess(self, outputData, batch_index, ori_shape): 
        boxes = []
        #input_w = self.runner.get_input_tensors()[0].dims[1]
        input_w = self.input_w
        #input_h = self.runner.get_input_tensors()[0].dims[2]
        input_h = self.input_h
        #print("ori_shape:", ori_shape)
        for i in range(len(self.runner.get_output_tensors())):
            height,width,channel = self.runner.get_output_tensors()[i].dims[1:]
            #print("h, w, c:", height, width, channel)
            size = channel * width * height
            #self.detect(boxes, results[i][0], height, width, i, input_w, input_h)
            anchor_index = len(self.runner.get_output_tensors()) - 1 - i
            boxes.extend(self.detect(outputData[i][batch_index], height, width, anchor_index, input_h, input_w))
        np_box = np.array(boxes)
        #np.savetxt("np_box.txt", np_box)
        if self.test_mAP:
            np_box = self.correct_region_boxes(np_box, ori_shape[0], ori_shape[1], input_h, input_w, self.params['num_classes'])
        result_bboxes = []
        for i in range(self.params['num_classes']):
            bboxes = self.nms(np_box[:, :4], np_box[:, 5 + i], self.params['conf_thresh'], self.params['nms_thresh'])
            labels = np.ones(bboxes.shape[0]) * i
            labels = labels.reshape(-1,1)
            result_bboxes.extend(np.hstack((bboxes, labels)))
            
        #nmsed_bboxes = self.nms(res, res[:, 5], self.params['nms_thresh'])
        result_bboxes = self.transform_bbox(result_bboxes)

        return result_bboxes 

    def run_test(self, images): 
        if self.test_mAP :
            self.params['conf_thresh'] = 0.005
        else :
            self.params['conf_thresh'] = 0.3
        inputTensors = self.runner.get_input_tensors()
        outputTensors = self.runner.get_output_tensors()
        shapeIn = tuple(inputTensors[0].dims)
        #print("images len:", len(images))
        #print("inputTensors len:", len(inputTensors))
        #for i in range(len(inputTensors)):
            #print("inputTensors[", i, "] dims:", inputTensors[i].dims)
            #print("inputTensors[", i, "] name:", inputTensors[i].name)

        inputData = [np.empty(shapeIn, dtype=np.float32, order="C")]
        outputData = []
        #print("outputTensors len:", len(outputTensors))
        for i in range(len(outputTensors)):
            shapeOut = tuple(outputTensors[i].dims)
            outputSize = int(outputTensors[i].get_data_size() / shapeIn[0])
            #print("outputTensors[", i, "] dims:", outputTensors[i].dims)
            #print("outputTensors[", i, "] name:", outputTensors[i].name)
            outputData.append(np.empty(shapeOut, dtype=np.float32, order="C"))

        picture_num = len(images)    
        batch_size = shapeIn[0]

        result = list() 
    
        group = 0
        if picture_num % batch_size: 
            group = picture_num // batch_size + 1
        else: 
            group = picture_num //batch_size 

        count = 0
        ori_size = []
        while count < group:
            print("process batch:", count)
            batch_name = list()
            batch_run_size = min(batch_size, picture_num - count * batch_size)
            # preprocess
            for i in range(batch_run_size):
                img = images[count * batch_run_size + i]
                ori_size.append(img.shape)
                img = self.preprocess(img)
                #print("image after preprocess:", img.shape)
                imageRun = inputData[0]
                imageRun[i, ...] = img

            # run dpu
            job_id = self.runner.execute_async(inputData, outputData)
            self.runner.wait(job_id)
            
            #print("after run dpu:")
            # postprocess
            for i in range(batch_run_size):
                boxes = self.postprocess(outputData, i, ori_size[count * batch_run_size + i])
                result.append(boxes)

            count = count + 1

        return result
    
class Processor:
    def __init__(self, bit_name = "dpu.bit", model_name = "yolov3_tf_ultra96v2-b1600.xmodel"):
        #self.processor = AdasDetection(bit_name, model_name) 
        self.processor = YoloV3_tf(bit_name, model_name) 
 
    def run(self, images):
        cv2.setUseOptimized(True)
        
        list_image = images
        result = self.processor.run_test(list_image) 
        return result 

if __name__ == "__main__": 
    detector = YoloV3_tf("dpu.bit", "yolov3_tf_ultra96v2-b1600.xmodel")
    #print("detector:", type(detector))
    images_list = ['../images/000001.jpg', '../images/000002.jpg'] 
    #images_list = ["./sample_yolov3.jpg"] 
    images = []
    for i in range(len(images_list)):
        image = cv2.imread(images_list[i])
        images.append(image)

    #detector.test_mAP = False 
    results = detector.run_test(images)
    #print("result len:", len(results))
    f = open("result.json", 'w')
    datas = []
    for i in range(len(results)):
        #print("result[", i, "] size:", len(results[i]))
        #print("images[", i, "] shape:", images[i].shape)
        image_id = int(images_list[i].split('/')[-1].split('.')[0])
        if detector.test_mAP:
            for j in range(len(results[i])):
                label = int(results[i][j][4])
                score = results[i][j][5]
                x_min = results[i][j][0] * images[i].shape[1] + 1
                y_min = results[i][j][1] * images[i].shape[0] + 1
                x_max = (results[i][j][0] + results[i][j][2]) * images[i].shape[1] + 1
                y_max = (results[i][j][1] + results[i][j][3]) * images[i].shape[0] + 1
                width = results[i][j][2] * images[i].shape[1]
                height = results[i][j][3] * images[i].shape[0]
                #print("result[", i, "][", j, "] :", results[i][j])
                start = (int(x_min), int(y_min))
                end = (int(x_max), int(y_max))
                print("result[", i, "][", j, "] label:", label, ", score:", score, ", bbox:", start, end)
                data = {"image_id" : image_id, 
                         "category_id" : label, 
                         "bbox": [x_min, y_min, width, height],
                         "score": score,
                        }
                datas.append(data)
        else:
            for j in range(len(results[i])):
                label = results[i][j][4]
                score = results[i][j][5]
                x_min = results[i][j][0] * images[i].shape[1] + 1
                y_min = results[i][j][1] * images[i].shape[0] + 1
                x_max = x_min + results[i][j][2] * images[i].shape[1]
                y_max = y_min + results[i][j][3] * images[i].shape[0]
                #print("result[", i, "][", j, "] :", results[i][j])
                start = (int(x_min), int(y_min))
                end = (int(x_max), int(y_max))
                print("result[", i, "][", j, "] label:", label, ", score:", score, ", bbox:", start, end)
                #print("result[", i, "][", j, "] :", start, end)
                cv2.rectangle(images[i], start, end, (0, 255, 0), 1)
                cv2.imwrite("./result_" + str(i) + ".jpg", images[i]) 

    json.dump(datas, f)
    f.close()
    #AdasDetection()
