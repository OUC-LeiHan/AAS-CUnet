
代码说明
########################################################################################
model：包括网络的核心代码，CUnet_model_parts.py描述网络子模块，CUnet.py描述网络总体搭建过程。
checkpoint: 保存网络训练的中间模型，后续进行模型测试。
log：存储网络训练日志以及训练过程中的误差曲线等。
utils：包括数据集制作、读取、日志生成、曲线绘制等功能性代码。
          其中Datasets.py描述数据集制作以及数据读取。
                 My_loss.py描述损失函数构造
                 Tools.py 描述评分指标计算方式
                 log_txt.py 描述日志生成
                 plot_fun.py描述误差曲线绘制

train.py: 项目入口代码，实现模型的训练，需要注意此项目采用AAS文章试验过程中的数据处理方式以及数据路径作为示例，
              对于不同数据对象需根据项目实际情况自行处理。
evaluate.py: 实现模型验证。
########################################################################################






Description
##############################################################################################
model：Including the core code of the network, CUnet_ model_ parts.py description network sub module, CUnet.py describes the 
                 overall network construction process.

checkpoint: Save the intermediate model of network.

log：Save the network training log and the error curve in the training process.

utils：Including functional codes such as  Datasets.py、My_loss.py、Tools.py、log_txt.py、 plot_fun.py.

          Datasets.py：Describe the process of making and reading dataset.

          My_loss.py: Construct loss function.

          Tools.py: Describe the calculation method of scoring index.

           log_txt.py: Record the running process of the code.

           plot_fun.py: plot error curve.

train.py: The entry code of the project is used to realize the training of the model. It should be noted that the project adopts the data
               processing method and data path in the test process of AAS article. Different data objects need to be processed 
                according to the actual situation of the project.

evaluate.py: Verify the model effect.
###########################################################################################
