# mobilenetV2-arcfaceloss-keras-tflite
 该仓库归纳了用mobilenet加arcfaceloss训练模型的keras框架，并提供将模型转为八位tflite的脚本。该仓库包括：
* 针对人脸识别场景优化后的mobilenetV2主干网络(keras实现)。
* ArcfaceLoss(Keras实现)
* 基于keras的训练框架与评估框架
* 训练保存权重(h5文件)转8位量化tflite文件脚本，以及对应的测试脚本

## 0. 准备环境
**训数据**

训练数据与测试数据引用了前人的成果，数据集采用ms1m-refine-v2(MS1M-ArcFace, [链接](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo))，该数据集为arcfaceloss原作者清洗后的人脸数据，包含85742个不同人的人脸数据，原作者使用的mxnet数据存储格式。本仓库使用tf后端的keras，可以采用这个[脚本](https://github.com/auroua/InsightFace_TF/blob/master/data/mx2tfrecords.py)，将其转换为tfrecode的格式。
也可以使用我转换好的数据：[baidu](https://pan.baidu.com/s/17wUppPURFISmsJHtfUd2HA)(提取码：5nwq)。 注意，其中train.tfrecords为转换后的数据集，train.rec与train.idx为mxnet格式数据集。其他*.bin为测试数据集。

**依赖安装**
```
pip install -r requirements.txt
```
注意，这里的keras使用tensorflow-gpu自带版本，不再另外安装。tensorflow-gpu版本为1.13.1，经过测试1.14,1.15版本也没问题，但是目前安装系统只支持1.13.1版本保存的模型，所以就先用1.13.1了

## 1. 训练脚本
```
python train.py
```
注意，
* 训练脚本权重(H5文件)与模型结构(model.json)分开保存
* 训练脚本会产生三个文件，分别为: checkpoint(存储每个epoch权重)，output(输出tensorboard文件)，log(训练日志)
* 每个epoch结束会读取valid_data文件夹下测试数据集(*.bin)文件，统计人脸识别的ACC、AUC、VR@FAR(10e-3)等信息

## 2. H5权重转PB文件以及PB文件测试
train.py脚本保存权重为MobilenetV2主干网络加上ArcFaceloss层，其中包含一个512*85742的全连接层，所以权重较大(近200M)，我们实际要的仅仅是其中的主干网输出部分。先后执行以下脚本：
```
# 注意修改21行的weight_file为checkpoint中保存的权重文件
# 修改22行json_file为项目目录下保存的模型结构文件
# 脚本会生成一个按时间创建的临时文件夹(tmp_YYYY-MM-DD_hhmmss),其中包含一个frozen_model.pb文件
python h5_2_pb.py
```

```
# 执行optimize_pb.sh脚本，去除上一步中生成pb文件中与输入输出无关的变量,这里使用tensrflow自带脚本
# 修改脚本中--input为上一步生成的frozen_model.pb
# 脚本会在项目目录下生成一个优化后的pb文件，该文件已经可以部署到实际应用环境中
./optimize_pb.sh
```

```
# 在数据集上测试最后生成的pb模型
# 注意修改该脚本16行的PB_DIR变量
# 脚本将输出pb模型在各个数据集上实际AUC、ACC、VAR@FAR等信息
python pb_test.py
```

## 3. PB文件转8位量化tflite文件、以及tflite文件测试
```
# 执行脚本，将pb文件转为8位量化的tflite文件，这里tflite量化仅仅是权重量化，底层实际运算依旧是浮点运算
# 修改该脚本第7行 pb_dir为上一步最后生成的pb文件路径
# 脚本将在项目目录下生成tflite文件
python pb_2_tflite.py
```
```
# tflite文件PC端测试
# 注意修改该脚本第16行RF_DIR变量为要测试的tflite文件
# 脚本将输出tflite模型在各个数据集上实际AUC、ACC、VAR@FAR等信息
python tflite_test.py
```

## 4. 模型基准测试
### MobilenetV2
PB模型  
![MobilenetV2_PB_Result](https://github.com/Linzmin1927/mobilenetV2-arcfaceloss-keras-tflite/images/pb_result.png "mobi_pb_result")
TFLITE模型  
![MobilenetV2_TFLITE_Result](https://github.com/Linzmin1927/mobilenetV2-arcfaceloss-keras-tflite/images/tflite_result.png "tflite_pb_result")
## 项目文件说明
.
├── valid_dataset/  验证数据集  
├── model/  一些已经训练好的模型  
├── dataset.py      tfrecoder接口  
├── evaluate_data.py  每个epoch结束读取valid_dataset数据统计FAR@VAR  
├── face_loss_k.py  ArcfaceLoss实现  
├── my_Mobilenet.py 重构的MobilenetV2主干网络，原始版本为keras自带结构，本项目中主要修改：网络主干加粗，所有ReLu改为PReLu  
├── train.py     训练脚本  
├── h5_2_pb.py   
├── optimize_pb.sh  
├── pb_test.py  
├── pb_2_tflite.py  
├── tflite_test.py  
├── requirements.txt  




## 参考
https://github.com/auroua/InsightFace_TF  
https://github.com/sirius-ai/MobileFaceNet_TF  
https://github.com/deepinsight/insightface/wiki/Dataset-Zoo  
https://arxiv.org/pdf/1804.07573  


