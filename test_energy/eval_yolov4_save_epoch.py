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
import time
import xml.dom.minidom
import pathlib
import json
import gc
import tqdm

sys.path.append(os.path.abspath("../common"))

import pynq
import lpcv_eval

team_name = 'LIVES'
team = lpcv_eval.Team(team_name, batch_size = 400)


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
detector = utils_yolov4.Processor(bit_name = "dpu.bit", model_name = "dpu1.3_coco55M_v4.xmodel")

RESULT_DIR = pathlib.Path('/home/xilinx/jupyter_notebooks/lpcv_2021/epoch_json')

if not os.path.exists(RESULT_DIR):  
      os.mkdir(RESULT_DIR) 

def save_epoch_results_json(save_dir, epoch, result_rectangle):      
        json_file = open(save_dir / str(str(epoch) + "epoch_time"+ ".json"), "w") 
        datas = []
        for i in range(len(result_rectangle)):  
            image_id = int(result_rectangle[i][0])
            label = int(result_rectangle[i][5])
            x = result_rectangle[i][1]
            y = result_rectangle[i][2]
            width = result_rectangle[i][3]
            height = result_rectangle[i][4]
            score = result_rectangle[i][6]
            data = {"image_id" : image_id, 
                    "category_id" : label, 
                    "bbox": [x, y, width, height],
                    "score": score,
                    }

            datas.append(data)

        json.dump(datas, json_file)
        json_file.close()






start = time.time() 
# 5V power rail is used as an example for full sytem power modify the code below
# refer to ultra96_pmbus.ipynb notebook under 
# /home/xilinx/jupyter_notebooks/common/ultra96_pmbus.ipynb for details on power rails
# recorder = pynq.DataRecorder(rails["5V"].power) 
recorder = pynq.DataRecorder(rails["5V"].power)


epoch_results = list()
epoch_images = []

max_epoch    = 21000
current_epoch = 0
total_num = 0

with recorder.record(0.05): 
    while True:
        gc.collect()
        # get a batch of images
        image_paths = team.get_next_batch()
        if image_paths is None:
            break
        
        if current_epoch >= max_epoch:
            break
            
        current_epoch = current_epoch + 1
        
        batch_images = list()
        batch_result = []
        
        epoch_results = list()
        epoch_images = []
        
        
        # run processor and save output  
        for image_path in image_paths:
            print("image_path:", image_path)
            bgr_img = cv2.imread(str(image_path))    
            batch_images.append(bgr_img)
            epoch_images.append([image_path,bgr_img])
              
        batch_result = detector.run(batch_images)
       
        #print("batch result ", batch_result)
        # print("batch result size:", len(batch_result))
        
        for i in range(len(batch_result)):
            epoch_results.append(batch_result[i])
            
        cur_num = len(batch_result)
        total_num = total_num + cur_num
        
                
        batch_result = None
        image_paths  =  None
        del batch_result
        del image_paths
        
        gc.collect()
        # Format  batch results and save
        save_epoch_results = []
        for i in range(len(epoch_results)):
            for j in range(len(epoch_results[i])):
                label = epoch_results[i][j][4]
                score = epoch_results[i][j][5]
                x = epoch_results[i][j][0] * epoch_images[i][1].shape[1] + 1
                y = epoch_results[i][j][1] * epoch_images[i][1].shape[0] + 1
                width = epoch_results[i][j][2] * epoch_images[i][1].shape[1]
                height = epoch_results[i][j][3] * epoch_images[i][1].shape[0]
               
                image_id = str(epoch_images[i][0]).split('/')[-1].split('.')[0]
                save_epoch_results.append([image_id, x, y, width, height, label, score])
    
        # team.save_results_xml(save_results, total_time, total_energy)
        # save results to json file for test accuracy
        save_epoch_results_json(RESULT_DIR,current_epoch, save_epoch_results)
        print("save current epoch result, and this is the  epoch :", current_epoch)
        
        batch_images = None  
        
        epoch_results = None
        epoch_images  = None 
        save_epoch_results = None 
        
        
        del   batch_images 
        
        del   epoch_results 
        del   epoch_images 
        del   save_epoch_results
        gc.collect()
 




 #timer stop after batch processing is complete
end = time.time()
t = end - start
total_time = t

#print('All processing time: {} seconds.'.format(total_time))
print("The whole number of  images:", total_num )

# Energy measurements    
# energy = recorder.frame["5V_power"].mean() * t    
energy = recorder.frame["5V_power"].mean() * t 

total_energy = energy
print("Total time:", total_time, "seconds")
print("Total energy:", total_energy, "J")



def merge_json(path_results, path_merges):
    """
    :param path_results:
    :param path_merges:
    :return:
    """
    merges_file = os.path.join(path_merges, "LIVES.json")
    with open(merges_file, "w", encoding="utf-8") as f0:
        for file in os.listdir(path_results):
            with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                for line in tqdm.tqdm(f1):
                    line_dict = json.loads(line)
                    js = json.dumps(line_dict, ensure_ascii=False)
                    f0.write(js + '\n')
                f1.close()
        f0.close()



path_results, path_merges = "./epoch_json", "./"
if not os.path.exists(path_merges):  
        os.mkdir(path_merges)
merge_json(path_results, path_merges)
print(" merge  all results done!")

'''
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
        # print("batch result size:", len(batch_result))
        for i in range(len(batch_result)):
            all_results.append(batch_result[i])
    #time.sleep(10)





# Format all  results and save
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


'''