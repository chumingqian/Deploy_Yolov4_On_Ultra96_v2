
# Deploy_Yolov4_On_Ultra96_v2
Model deployment,  Yolov4,  Xilinx,  Ultra96_v2.


<div align="center">
📖 Github
&emsp;&emsp; | &emsp;&emsp;
<a href="https://github.com/chumingqian/Deploy_Yolov4_On_Ultra96_v2/">📚 Docsify</a>
</div> 
<br>

<div align="center">
简体中文
&emsp;&emsp; | &emsp;&emsp;
<a href="https://github.com/chumingqian/Deploy_Yolov4_On_Ultra96_v2/blob/main/README_eng.md">English</a>
</div> 
<br>

This repository has contain the following part：
------------
* Part1: Modify the network  -- yolov4.cfg. 
* Part2: Use Vitis-ai tool to quantify and compile the yolov4 network. 
* Part3: Deploy the yolov4 to the edge device(ultra_96_v2)上, write the notebook.ipynb to call the pynq-dpu and inference the network. 
    
 Note: Before we deploy the YOLOV4, some friends may want pruning the YOLOV4 network, in this repository we use this [channel pruning](https://github.com/chumingqian/Model_Compression_For_YOLOV4) method to prune the network and deployed the pruned yolov4 network successfully.
 We train the pruned both on the  VOC dataset (contains 20 classes) &&  COCO dataset (contains 80 classes).  Here are the weights both for original network and pruned network：https://pan.baidu.com/s/1lL1tPSOKJc4V4eF_SqVoHw , Extract code: rvrg.
 
 And the yolov4-voc-leaky.cfg which suit the dpu are locate at 07-yolov4-tutorial/dk_model/ . 
 
 Notice that, we need modify the yolov4.cfg firstly so that network can call the dpu module, then we can prune the modified yolov4.cfg.
  

Part1:  Modify yolov4.cfg network.
------------ 
   
   1.0 Due to the current pynq-DPU 1.2 version doesn't support MISH activation, and the maximum kernel size of the maxpooling  is only support to 8*8 ; In this repository, we made two modifications to enable compliance with  the Zynq Ultrascale + DPU, and also the modified the network can be  quantized and compiled by the  Vitis Ai tool. 
   
               m1  The MISH acitvation are  swapped to leakrelu.      
               m2  The SPP Moudle maxpool sizes have been changed from 5 ,9,13 to 5,5,7.
         


Part2: On Host machine(ubuntu18.04) use the Vitis -ai 1.3.2(Xilinx) tool to quantize and compile the network.
------------

   2.0 Install Docker on Ubuntu18.04, if Docker not installed on your machine (https://docs.docker.com/engine/install/ubuntu/ ). Follow the Post installation steps  for Linux ( https://docs.docker.com/engine/install/linux-postinstall/ ) to ensure that your Linux user is in the group Docker. Or you can  reference the  https://www.xilinx.com/html_docs/vitis_ai/1_3/installation.html to install the vitis ai.
   
   2.1 Clone the Vitis ai repository to obtain the examples, reference code, and scripts.
   
	          git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI  
            
                  cd Vitis-AI

   2.2 Run the CPU image from the Docker Hub: (Pelese prepare the above 32G Memory, if you want build the CPU image locally.)
   
	          ./docker pull xilinx/vitis-ai-cpu:latest  
		  
	          Notice that default startup version is the latest, you can add the number if you want start the specified version.
       
		  	         Vitis AI v1.4	./docker_run.sh xilinx/vitis-ai-cpu:1.4.916
				 Vitis AI v1.3	./docker_run.sh xilinx/vitis-ai-cpu:1.3.411
				 Vitis AI v1.3.1
				 Vitis AI v1.3.2
				 Vitis AI v1.2	./docker_run.sh xilinx/vitis-ai-cpu:1.2.82
             
         
   2.3  After we startup the vitis ai, we can see that currently  it support the following deep learning frames:Pytorch、Tensorflow、Tensorflow 2 and Caffe .
       
   In this repository the yolov4 network was trained by the DarkNet, the network file of the modle  is .cfg format.
   So we both convert the network file's format and weights's format. Here provide the two ways which are convert the Darknet to  Tensorflow and Caffe.
       
   Then we quantize and compile the converted model, here  is the official  user  guide (https://china.xilinx.com/products/design-tools/vitis/vitis-ai.html), ug1414-vitis-ai.pdf.     
        		

   2.4 Darknet Convert to Tensorflow(conda activate Tensorflow) (For pynq-dpu1.2, generate the dpu_model.elf )
		
		STEP1: convert the model's .cfg file and  model's weights:		
		python ../keras-YOLOv3-model-set/tools/model_converter/convert.py --yolo4_reorder ../dk_model/yolov4-voc-leaky.cfg ../dk_model/leakcy-v4.weights ../keras_model/v4_voc_leaky.h5
		python ../keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py --input_model ../keras_model/v4_voc_leaky.h5 --output_model=../tf_model/v4_tf_model.pb
	
	   The name of input nodes and output nodes will be different according to the models, use the vai_q_tensorflow  quantizer to check and estimate these nodes: 
     
		$ vai_q_tensorflow inspect --input_frozen_graph=../tf_model/v4_tf_model.pb

		 Otherwise we can visualize the grapth to obtain the name of input nodes and output nodes. 
     TensorBoard and  Netron both support this operation. Here use the Netron：
		$ pip install netron
		$  netron ../tf_model/v4_tf_model.pb

		STEP2: Quantizatin：
		vai_q_tensorflow quantize --input_frozen_graph ../tf_model/v4_tf_model.pb --input_fn yolov4_graph_input_keras_fn.calib_input   --output_dir ../chu_v4_quantized --input_nodes image_input --output_nodes conv2d_93/BiasAdd,conv2d_101/BiasAdd,conv2d_109/BiasAdd --input_shapes ?,416,416,3 --calib_iter 30

		STEP3:COMPLIE 		
		For pynq-dpu1.2 use the following commond, it will generate the .elf  modle.
		dnnc-dpuv2 --save_kernel --parser tensorflow --frozen_pb ../chu_v4_quantized/deploy_model.pb --dcf dpuPynq_ultra96v2.dcf  --cpu_arch arm64 --output_dir ../chu_v4_compiled --net_name tf_model_v4_416

				
   2.5 Darnet  convert to caffe ( conda activate caffe ) (for pynq-dpu1.3, generate the dpu_model.xmodel )	    
		
		STEP1: MODEL CONVERT  TO CAFFE
		python /opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/convert.py ../dk_model/yolov4-voc-leaky.cfg ../dk_model/leakcy-v4.weights  ../dpu1.3.2_caffe_model/v4_leacky.prototxt ../dpu1.3.2_caffe_model/v4_leacky.caffemodel

		STEP2:  MDOEL  Quantization
      *1.Before quantizing the model, we will need to make a minor modifcation to .prototxt file to point to the calibaration images.  Make a new copy of the prototxt file and make the following edits:
            name: "Darkent2Caffe"
            #input: "data"
            #input_dim: 1
            #input_dim: 3
            #input_dim: 416
            #input_dim: 416

            ####Change input data layer to VOC validation images #####
            layer {
              name: "data"
              type: "ImageData"
              top: "data"
              top: "label"
              include {
                phase: TRAIN
              }
              transform_param {
                mirror: false
                yolo_height:416  #change height according to Darknet model
                yolo_width:416   #change width according to Darknet model
              }
              image_data_param {
                source: "voc/calib.txt"  #list of calibration imaages     
                root_folder: "images/" #path to calibartion images

                batch_size: 1
                shuffle: false
              }
            }
            #####No changes to the below layers##### 
    		*2. Notice that the calibration images in file.txt, the .txt file needs to be a two column format to realize the quantization.(For the quantize calibration, the images without labels are enough, but to realize the quantization we need a two column format .txt file, one column is the image_id, the other column just set as the zero)
		
    		*3. Notice that the path of the calibration images should under the Doker environment, meanwhile the  workspace can be regard as computer  and the vitis ai  regard as  a home:	
		
		vai_q_caffe quantize -model ../dpu1.3.2_caffe_model/v4_leacky_quanti.prototxt  -keep_fixed_neuron -calib_iter 3 -weights ../dpu1.3.2_caffe_model/v4_leacky.caffemodel -sigmoided_layers layer133-conv,layer144-conv,layer155-conv -output_dir ../dpu1.3.2_caffe_model/ -method 1 

		STEP3:  MODEL  COMPILE 
		vai_c_caffe --prototxt ../dpu1.3.2_caffe_model/original_model_quanti/deploy.prototxt --caffemodel ../dpu1.3.2_caffe_model/original_model_quanti/deploy.caffemodel --arch ./u96pynq_v2.json --output_dir ../dpu1.3.2_caffe_model/ --net_name dpu1-3-2_v4_voc --options "{'mode':'normal','save_kernel':''}";
    
		 notice that :  After generate the  .xmodle, when we inference the network use the pynq-dpu 1.3 and call the dpu.xmodel on  ultra_96_v2, if it appears the  error "the target footprint xxx  not match XXX  "，  we can use the u96pynq_v2.json instead of u96pynq.json, reference is https://forums.xilinx.com/t5/AI-and-Vitis-AI/vitis-ai-1-3-with-ultra96/td-p/1189251 .
		 
		



Part3: On the edge device (ultra_96_v2), using the pynq-dpu1.2 to test the inference speed both of the   unpruned network and  pruned network,  and using the pynq-dpu1.3 to test the energy consumption both of the unpruned yolov4 and pruned yolov4 network.
------------
       3.1 Prepare a SD card (32G)to flash the image of PYNQ2.6, image file can be obtain at (https://github.com/Xilinx/PYNQ/releases or http://www.pynq.io/board.html) 
       3.2  Load the SD card, startup the ultra_96_v2 board, we can use the MobaXterm to connect the pc, input the 192.168.3.1 on the local browser,  install DPU-PYNQ (https://github.com/Xilinx/DPU-PYNQ), if the download speed  is too slow, we can downlad the file to local PC, then  copy to the board.
       3.3  Programming the  notebook.ipynb for  inference the network, the following is main step to inference  a neuro network, (the evaluation.ipynb to test the energy power is under ./test_energy file)

			* load the model (generate by the vitis ai,“dpu_model.xmodel”  or "dup_model.elf" )
			  	overlay.load_model(“dpu_model.xmodel” )
			* define the dpu  object
			   	dpu = overlay.runner
			* create the input and output buffer   
				output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
				input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
			* to make a prediction:
				job_id = dpu.execute_async(input_data, output_data)
				dpu.wait(job_id)
			* save the prediction result into the output_data file.



Part4: demo.video https://www.bilibili.com/video/BV1AU4y1n7w6/.
------------
When the input size of the images are: 416 *416, From the following repective to compare the unpruned yolov4 and pruned yolov4 network.1. the size of the network. 2. the inference speed of the network. 3. the energy comsumption of the network.
 
      1. the size of the unpruned yolov4 and pruned yolov4 network.
           
      2  On ultra96_v2,  for the pynq-dpu1.2 load the modle.elf,  run the notebook.ipynb.
         2.1 pruned yolov4: the speed of inferencing single image was  250 ms.
         2.2 unpruned yolov4: the speed of inferencing single image was 330 ms. 
         
      3  On ultra96_v2,  for the pynq-dpu1.3 load the modle.xmodle,  run the eval.ipynb.
         3.1 pruned yolov4 , consume 39 J when inference 10 images, consume 1872J when inference 500 images. 
         3.2 unpruned yolov4 , consume 2347 J when inference 500 images.
    
                           
#####   The result  as shown in the figure.

<div align="center">
<img src="./images_in_readme/fig1.png" width = "700" height = "360" />
</div>	


Acknowledgements: 
====
Thank you for the Summer School co-organized by XILINX & NICU.

This Summer School is memorable. We had experienced the Nanjing epidemic and Shanghai typhoon-'Fireworks'.

Finally we arrived on the land of XILINX _ 2021 SUMMER SCHOOL. 

======  
  
<div align="center">
<img src=https://img-blog.csdnimg.cn/20200822014538211.png />
</div>

