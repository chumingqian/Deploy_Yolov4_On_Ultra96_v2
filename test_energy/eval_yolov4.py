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

import os
import time
import sys

sys.path.append(os.path.abspath("../common"))

import pynq
import lpcv_eval

team_name = 'CY_yolov4_team'
team = lpcv_eval.Team(team_name, batch_size = 10)


# import utils
import utils_yolov4

import cv2

interval_time = 0
total_time = 0
total_energy = 0

team.reset_batch_count()
rails = pynq.get_rails()


# Create detector with bit file name and model name
# you can replace the default bit file by your own bit
detector = utils_yolov4.Processor(bit_name = "dpu.bit", model_name = "dpu1.3_v4_voc.xmodel")


start = time.time() 
# 5V power rail is used as an example for full sytem power modify the code below
# refer to ultra96_pmbus.ipynb notebook under 
# /home/xilinx/jupyter_notebooks/common/ultra96_pmbus.ipynb for details on power rails
# recorder = pynq.DataRecorder(rails["5V"].power) 
recorder = pynq.DataRecorder(rails["3V3"].power)
all_results = list()
all_images = []
with recorder.record(0.05): 
    while True:
        # get a batch of images
        image_paths = team.get_next_batch()
        if image_paths is None:
            break
            
        batch_images = list()
        # run processor and save output  
        for image_path in image_paths:
            print("image_path:", image_path)
            bgr_img = cv2.imread(str(image_path))    
            batch_images.append(bgr_img)
            all_images.append([image_path, bgr_img])
        #batch_result = detector.run_test(batch_images)   
        batch_result = detector.run(batch_images)  
        print("batch result size:", len(batch_result))
        for i in range(len(batch_result)):
            all_results.append(batch_result[i])
    #time.sleep(10)

# timer stop after batch processing is complete
end = time.time()
t = end - start

total_time = t

#print('All processing time: {} seconds.'.format(total_time))
print("all results size:", len(all_results))

# Energy measurements    
# energy = recorder.frame["5V_power"].mean() * t    
energy = recorder.frame["3V3_power"].mean() * t 

total_energy = energy
print("Total time:", total_time, "seconds")
print("Total energy:", total_energy, "J")



# Format results and save
save_results = []
for i in range(len(all_results)):
    for j in range(len(all_results[i])):
        label = all_results[i][j][4]
        score = all_results[i][j][5]
        x = all_results[i][j][0] * all_images[i][1].shape[1] + 1
        y = all_results[i][j][1] * all_images[i][1].shape[0] + 1
        width = all_results[i][j][2] * all_images[i][1].shape[1]
        height = all_results[i][j][3] * all_images[i][1].shape[0]
        #print("result[", i, "][", j, "] name:", all_images[i][0], "label:", label, ", score:", score, ", bbox:", x, y, width, height)
        image_id = str(all_images[i][0]).split('/')[-1].split('.')[0]
        save_results.append([image_id, x, y, width, height, label, score])
    
# team.save_results_xml(save_results, total_time, total_energy)
# save results to json file for test accuracy
team.save_results_json(save_results)
print("save result done!")