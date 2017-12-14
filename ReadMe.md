c++(visual studio) implementation of Efficient Convolution Operator(ECO)tracker.

version:1.0

环境：caffe+vs2015+opencv3.x

Publication：

Details about the tracker can be found in the CVPR 2017 paper:

Martin Danelljan, Goutam Bhat, Fahad Khan, Michael Felsberg.
ECO: Efficient Convolution Operators for Tracking.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Please cite the above publication if you use the code or compare with the ECO tracker in your work. Bibtex entry:

@InProceedings{DanelljanCVPR2017,
Title = {ECO: Efficient Convolution Operators for Tracking},
Author = {Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael},
Booktitle = {CVPR},
Year = {2017}
}

Contact
Email: flying_tan@163.com   ucasws@gmail.com

installation:

1. caffe-windows
   参照网上caffe-windows安装
2. opencv
   推荐opencv3.x(opencv2.x 也可以使用，只需将其中部分函数替换即可)
3. vs2015/vs2013 build and run 
   将caffe-windows目录等在vs项目属性表中配置完成即可build



代码相关描述：

ECO.h, ECO.cpp: tracker参数,特征参数的初始化;生成标签、窗函数等--------

eco_sample_update.h, eco_sample_update.cpp: 更新样本的相关函数(计算距离、更新样本空间模型等)

feature_extractor.h, feature_extractor.cpp: 特征提取的相关函数(采样、cnn、hog特征的提取)

feature_operator.h, feature_operator.cpp: 运算符重载、映射函数

fftTools.h, fftTools.cpp: 复数矩阵的运算

optimize_scores.h, optimize_scores.cpp: 傅里叶域计算

training.h, training.cpp: 特征的训练(更新)函数

相关详细说明及描述后续更新

注意事项：
1. 如果编译显示：编译器内部错误，一个解决方法为： 将属性表中常规的全程需优化选项设置为无全程序优化；c/c++子属性优化中的全程序优化选项改为否
2. 目前使用caffe-windows CPU版本，GPU将尽快更新