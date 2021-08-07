
##  本仓库主要包含以下部分的内容：


###     
###  如果在部署YOLOV4 网络之前，需要对YOLOV4网络进行剪枝，可以参考 https://github.com/chumingqian/Model_Compression_For_YOLOV4 。注意到如果要将剪枝的网络到ultra_96v2上，需要先对yolov4.cfg 进行如下修改，之后将修改后的网进行剪枝。 
   

###  Part1:  修改 yolov4.cfg 网络文件。
            修改yolov4网络的结构, 使其yolov4.cfg 网络中的函数能够使用配合使用Xilinx的 vitis-ai 工具进行量化和编译， 并在pynq-dpu 运行。
            受限于当前pynq-dpu1.2 并不支持MISH激活函数，且dpu 支持的最大池化的kernel size为8。 本仓库中对原始网络的.cfg 文件做了如下修改。   
             1 将MISH激活函数替换为ｌｅａｋｙ；
          2  　将ｓｐｐ　ｍｏｄｕｌｅ　.
            2.1 测试剪枝网络模型的推理速度 250 ms.
         2.2 测试未剪枝网络模型的推理速度 330 ms. 

###  Part2: 在主机端(ubuntu18.04)上使用Xilinx 的vitis -ai1.3 工具完成对剪枝网络的量化和编译部署。


###  Part3: 在边缘端(ultra_96_v2),  使用pynq-dpu1.2 分别测试剪枝前后yolov4网络的推理速度， 使用pynq-dpu1.3 分别测试剪枝前后yolov4网络消耗的能量。




### 实验结果从：1.网络的体积，2.网络的推理速度 3.网络消耗的能量 这三个方面来对比剪枝前后的网络的性能。 

 demo.video https://www.bilibili.com/video/BV1AU4y1n7w6/ ：展示了以下功能 image input size 416 *416:
 
      1  对比剪枝前后网络模型的体积大小.     
      2  在ultra96_v2, pynq-dpu1.2,的环境下载入生成的.elf 文件，运行对应的.ipynb文件.
         2.1 测试剪枝网络模型的推理速度 250 ms.
         2.2 测试未剪枝网络模型的推理速度 330 ms. 
         
      3  在ultra96_v2, pynq-dpu1.3,的环境下载入生成的.xmodel 文件，运行对应的.ipynb文件.
         3.1 测试剪枝网络模型推理10 张images 所消耗的功耗，约为39J.  随后测试推理500 images，所消耗的功耗，约为1872J .
         3.2 测试未剪枝网络模型推理500 images，所消耗的功耗，约为2347J .
    
                           
          实验结果如图1所示。
####  ![fig1](https://user-images.githubusercontent.com/46816091/128596310-88837fbf-3fec-47f4-a19e-ae7da825b611.png)
