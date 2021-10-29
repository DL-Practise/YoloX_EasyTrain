# YoloX_EasyTrain
这是一个YoloX的训练插件，通过可视化的方式管理工程，训练模型。方便，高效！适合yolox新人朋友以及重度用户，让你点点鼠标就能训练自己的YoloX模型。
**更详尽的使用文档请移步公众号： DL工程实践，后台回复：yolox**  

# 测试环境
操作系统：Ubuntu16.04  
pytorch：1.8.0  
cuda：11.0  
python：3.6  
这个环境我是测试过OK的，能保证可以正常使用。

# 第一步：安装YoloX
按照YoloX官网的教程应该是比较顺利的，感谢YoloX官网的耐心细致的教程。

# 第二步：下载工程源码
从github上面下载YoloX EasyTrain的代码。通过git clone或者直接界面下载都可以。

# 第三步：安装YoloX EasyTrain的依赖
cd /path/to/YoloX_EasyTrain/  
pip install -r requirements.txt  

# 第四步：下载预训练模型（可选）
因为我们一般都是在自己的数据集上迁移学习，使用官方的预训练模型会快很多，因此推荐下载官方的预训练模型。从yolox的官方github上面下载。
下载完之后，将模型放到YoloX EasyTrain的weights文件夹下面。
YOLOX-s预训练模型：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth  
YOLOX-m预训练模型：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth  
YOLOX-l预训练模型：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth  
YOLOX-x预训练模型：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth  
YOLOX-nano预训练模型：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth  
YOLOX-tiny预训练模型：https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth  

# 第五步：启动插件
在命令行使用：  python  main_widget.py 启动YoloX EasyTrain 插件。然后就享受你的快乐时光吧。

# 备注
**更详尽的使用文档请移步公众号： DL工程实践，后台回复：yolox**  
**交流QQ群：552703875**  
**演示视频，B站：https://www.bilibili.com/video/BV1gq4y1Z7tE?spm_id_from=333.999.0.0**


