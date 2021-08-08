
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

本仓库主要包含以下部分的内容：
------------
* Part1:  调整 yolov4.cfg 网络文件. 
* Part2: 使用vitis -ai 工具对网络进行量化和编译. 
* Part3: 将网络部署到边缘端(ultra_96_v2)上,编写notebook.ipynb 文件，调用pynq-dpu 推理运行网络.  


    
        Note:在部署YOLOV4 网络之前，需要对YOLOV4网络进行剪枝的同学，可以参考[本仓库所使用的剪枝方法](https://github.com/chumingqian/Model_Compression_For_YOLOV4) . 注意到如果要对网络进行剪枝，需要先对yolov4.cfg 修改成dpu支持的网络，再对修改后的网络进行剪枝。 
====
  

Part1:  调整yolov4.cfg 网络.
==== 

            受限于当前pynq-dpu1.2 并不支持MISH激活函数，且dpu 支持的最大池化的kernel size为8, 故修改yolov4网络的结构, 使修改后的yolov4.cfg 网络能够使用配合使用Xilinx的 vitis-ai 工具进行量化和编译，并在pynq-dpu 上运行.本仓库中对原始网络的 yolov4.cfg 文件做了如下修改.
               1  将MISH激活函数替换成leaky.     
               2  将SPP Moudle maxpool 由5 ，9，13 替换成 5，5，7; 之后对修改后的网络进行重新微调训练。
         




Part2: 在主机端(ubuntu18.04)上使用Xilinx 的vitis -ai  工具完成对剪枝网络的量化和编译部署。
====
         2.1 主机端安装 vitis ai 工具,  推荐使用 docker 环境安装，若在本地安装请准备32G 以上的内存用于安装时的编译。
         
         
         2.2 在主机端完成vitis ai配置后， 可对模型进行量化和编译，具体参考 vitis ai 中的技术文档(https://china.xilinx.com/products/design-tools/vitis/vitis-ai.html)，其中有中文版c_ug1414-vitis-ai.pdf.  vitis ai 当前支持的深度学习框架有Pytorch、Tensorflow、Tensorflow 2 和 Caffe，  由于笔者实现的是Darknet 版本的yolov4, 网络文件为.cfg格式， 故先要对网络文件，以及权重文件的格式进行转换， 此处介绍两种转换方式由darknet 分别转换成 caffe  和 Tensorflow,  之后对caffe 和 Tensorflow 模型进行量化和编译。      
        
	
	
	   2.3 darnet  convert to caffe

		STEP1: MODEL CONVERT  TO CAFFE
		python /opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/convert.py yolov4.cfg yolov4.weights VOC/yolov4.prototxt VOC/yolov4.caffemodel

		python /opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/convert.py ../dk_model/yolov4-voc-leaky.cfg ../dk_model/leakcy-v4.weights  ../dpu1.3.2_caffe_model/v4_leacky.prototxt ../dpu1.3.2_caffe_model/v4_leacky.caffemodel

		python /opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/convert.py ../dk_model/regular_prune_0.319_remove_bn.cfg ../dk_model/yolov4-prune-0.319.weights  ../dpu1.3.2_caffe_model/v4_pruned.prototxt ../dpu1.3.2_caffe_model/v4_pruned.caffemodel



		STEP2:  MDOEL  QUANTI
		1.在量化之前，对原始的.prototxt网络拷贝一个副本，在副本中加入校准图片的路径， 使用该副本网络进行量化；
		2.并且注意到校准图片的.txt 文档中，实现量化时需要含两列的列表文件，这与tensorflow 的校准文件的txt文档不一样。(对于量化校准，不含标签的校准数据即可足够。但实现需要含 2 列的图像列表文件。只需将第 2 列设为随机值或 0 即可)
		3.注意到校准图片的路径应该是docker 环境下的路径， 即路径应该是  workspace 是vitis-ai 为工作空间的，  此时的vitis-ai 可以理解成主机上的home;
		vai_q_caffe quantize -model VOC/yolov4_quant.prototxt -calib_iter 100 -weights VOC/yolov4.caffemodel -sigmoided_layers layer133-conv,layer144-conv,layer155-conv -output_dir yolov4_quantized/ -method 1

		vai_q_caffe quantize -model ../dpu1.3.2_caffe_model/v4_leacky_quanti.prototxt  -keep_fixed_neuron -calib_iter 3 -weights ../dpu1.3.2_caffe_model/v4_leacky.caffemodel -sigmoided_layers layer133-conv,layer144-conv,layer155-conv -output_dir ../dpu1.3.2_caffe_model/ -method 1 

		vai_q_caffe quantize -model ../dpu1.3.2_caffe_model/v4_pruned_quanti.prototxt  -keep_fixed_neuron -calib_iter 3 -weights ../dpu1.3.2_caffe_model/v4_pruned.caffemodel -sigmoided_layers layer133-conv,layer144-conv,layer155-conv -output_dir ../dpu1.3.2_caffe_model/ -method 1 



		STEP3:  MODEL  COMPILE 
		vai_c_caffe --prototxt ../dpu1.3.2_caffe_model/original_model_quanti/deploy.prototxt --caffemodel ../dpu1.3.2_caffe_model/original_model_quanti/deploy.caffemodel --arch ./u96pynq.json --output_dir ../dpu1.3.2_caffe_model/ --net_name dpu1.3_v4_voc --options "{'mode':'normal','save_kernel':''}";

		vai_c_caffe --prototxt ../dpu1.3.2_caffe_model/pruned_model_quanti/deploy.prototxt  --caffemodel  ../dpu1.3.2_caffe_model/pruned_model_quanti/deploy.caffemodel --arch ./u96pynq.json --output_dir ../dpu1.3.2_caffe_model/ --net_name dpu1.3_pruned_v4_voc --options "{'mode':'normal','save_kernel':''}";

		vai_c_caffe --prototxt ../dpu1.3.2_caffe_model/original_model_quanti/deploy.prototxt --caffemodel ../dpu1.3.2_caffe_model/original_model_quanti/deploy.caffemodel --arch ./u96pynq_v2.json --output_dir ../dpu1.3.2_caffe_model/ --net_name dpu1-3-2_v4_voc --options "{'mode':'normal','save_kernel':''}";

		vai_c_caffe --prototxt ../dpu1.3.2_caffe_model/pruned_model_quanti/deploy.prototxt  --caffemodel  ../dpu1.3.2_caffe_model/pruned_model_quanti/deploy.caffemodel --arch ./u96pynq_v2.json --output_dir ../dpu1.3.2_caffe_model/ --net_name dpu1.3_pruned_v4_voc --options "{'mode':'normal','save_kernel':''}"



		./docker_run.sh  xilinx/vitis-ai-cpu
		Way2: 

		https://tvm.apache.org/docs/deploy/vitis_ai.html


		STEP1: 网络模型，权重格式转换步骤：
		Convert to tensorflow:


		python ../keras-YOLOv3-model-set/tools/model_converter/convert.py --yolo4_reorder ../dk_model/yolov4-voc-leaky.cfg ../dk_model/leakcy-v4.weights ../keras_model/v4_voc_leaky.h5
		python ../keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py --input_model ../keras_model/v4_voc_leaky.h5 --output_model=../tf_model/v4_tf_model.pb

		python ../keras-YOLOv3-model-set/tools/model_converter/convert.py --yolo4_reorder ../dk_model/regular_prune_0.319_remove_bn.cfg ../dk_model/yolov4-prune-0.319.weights ../keras_model/pruned_v4_leaky.h5
		python ../keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py --input_model ../keras_model/pruned_v4_leaky.h5 --output_model=../tf_model/pruned_v4_tf_model.pb

		python ../keras-YOLOv3-model-set/tools/model_converter/convert.py --yolo4_reorder ../dk_model/regular_prune_0.319_remove_bn.cfg ../dk_model/yolov4-prune-0.319.weights ../keras_model/slim_v4_leaky.h5
		python ../keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py --input_model ../keras_model/slim_v4_leaky.h5 --output_model=../tf_model/slim_v4_tf_model.pb

		输入节点和输出节点名称因模型而异， 但您可使用 vai_q_tensorflow 量化器来检查和估算这些节点。请参阅以下代码片段示例：
		$ vai_q_tensorflow inspect --input_frozen_graph=../tf_model/v4_tf_model.pb

		种获取图的输入和输出名称的方法是将图可视化。 TensorBoard 和 Netron 均可执行此操作。请参阅以下示例， 其中使用的是 Netron：
		$ pip install netron
		$  netron ../tf_model/v4_tf_model.pb

		STEP2:量化步骤：

		vai_q_tensorflow quantize --input_frozen_graph ../tf_model/v4_tf_model.pb --input_fn yolov4_graph_input_keras_fn.calib_input   --output_dir ../chu_v4_quantized --input_nodes image_input --output_nodes conv2d_93/BiasAdd,conv2d_101/BiasAdd,conv2d_109/BiasAdd --input_shapes ?,416,416,3 --calib_iter 30


		vai_q_tensorflow quantize --input_frozen_graph ../tf_model/slim_v4_tf_model.pb --input_fn yolov4_graph_input_keras_fn.calib_input   --output_dir ../pruned_quantized --input_nodes image_input --output_nodes conv2d_93/BiasAdd,conv2d_101/BiasAdd,conv2d_109/BiasAdd --input_shapes ?,416,416,3 --calib_iter 3


		STEP3:COMPLIE 编译步骤

		vitisai 版本1.3.598
		pynq-dpu1.2 使用以下这个，编译生成的.elf 文件用于 Pynq-dpu1.2 版本：
		dnnc-dpuv2 --save_kernel --parser tensorflow --frozen_pb tf_quant/deploy_model.pb --dcf dpuPynq_ultra96v2.dcf  --cpu_arch arm64 --output_dir dnnccompiled --net_name tf416yolov4

		dnnc-dpuv2 --save_kernel --parser tensorflow --frozen_pb ../chu_v4_quantized/deploy_model.pb --dcf dpuPynq_ultra96v2.dcf  --cpu_arch arm64 --output_dir ../chu_v4_compiled --net_name tf_model_v4_416

		dnnc-dpuv2 --save_kernel --parser tensorflow --frozen_pb ../pruned_quantized/deploy_model.pb --dcf dpuPynq_ultra96v2.dcf  --cpu_arch arm64 --output_dir ../pruned_compiled --net_name slim_model_v4_416



Part3: 在边缘端(ultra_96_v2),  使用pynq-dpu1.2 分别测试剪枝前后yolov4网络的推理速度， 使用pynq-dpu1.3 分别测试剪枝前后yolov4网络消耗的能量。
====




demo.video https://www.bilibili.com/video/BV1AU4y1n7w6/ ，展示了当image input size 416 *416，从：1.网络的体积，2.网络的推理速度 3.网络消耗的能量，这三个方面来对比剪枝前后的网络的性能:
----
 
      1  对比剪枝前后网络模型的体积大小.     
      2  在ultra96_v2, pynq-dpu1.2,的环境下载入生成的.elf 文件，运行对应的.ipynb文件.
         2.1 测试剪枝网络模型的推理速度 250 ms.
         2.2 测试未剪枝网络模型的推理速度 330 ms. 
         
      3  在ultra96_v2, pynq-dpu1.3,的环境下载入生成的.xmodel 文件，运行对应的.ipynb文件.
         3.1 测试剪枝网络模型推理10 张images 所消耗的功耗，约为39J.  随后测试推理500 images，所消耗的功耗，约为1872J .
         3.2 测试未剪枝网络模型推理500 images，所消耗的功耗，约为2347J .
    
                           
          实验结果如图1所示。
######  ![fig1](https://user-images.githubusercontent.com/46816091/128596310-88837fbf-3fec-47f4-a19e-ae7da825b611.png)
