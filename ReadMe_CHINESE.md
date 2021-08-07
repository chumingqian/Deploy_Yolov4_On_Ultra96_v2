
##  本仓库主要包含以下部分的内容：

Part1: 对yolov4 网络在结构上微改，使修改后的网络能够配合使用dpu, 并对修改后的网络进行剪枝，减小网络的体积。

Part2: 在主机端(ubuntu18.04)上使用Xilinx 的vitis -ai1.3 工具完成对剪枝网络的量化和编译部署。


Part3: 在边缘端(ultra_96_v2),  使用pynq-dpu1.2 分别测试剪枝前后yolov4网络的推理速度， 使用pynq-dpu1.3 分别测试剪枝前后yolov4网络消耗的能量。


实验结果从：1.网络的体积，2.网络的推理速度 3.网络消耗的能量 这三个方面来对比剪枝前后的网络的性能。 实验结果如图1所示。





#### !!!![fig1](https://user-images.githubusercontent.com/46816091/128596310-88837fbf-3fec-47f4-a19e-ae7da825b611.png)
