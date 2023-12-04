> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/Little_Carter/article/details/133610076)

**目录**

[一、前言](#t0)

[二、开发环境（前提条件）](#t1)

[三、环境搭建教程](#t2)

[3.1、创建虚拟环境](#t3)

[3.2、选择虚拟环境并安装所需要的包](#t4)

[3.3、运行代码步骤](#t5)

[3.3.1、克隆 git 储存库](#t6)

[3.3.2、转到克隆库的文件夹下](#t7)

[3.3.3、安装依赖项](#t8)

[3.3.4、转到检测目录下](#t9)

[3.3.5、用于 yolov8 物体检测 + 跟踪 + 车辆计数](#t10)

[四、效果图](#t11)

一、前言
----

欢迎阅读本篇博客！今天我们深入探索 YOLOv8+deepsort 视觉跟踪算法。结合 YOLOv8 的目标检测和 deepsort 的特征跟踪，该算法在复杂环境下确保了目标的准确与稳定跟踪。在计算机视觉中，这种跟踪技术在安全监控、无人驾驶等领域有着广泛应用。**本文重点探讨基于此算法的车辆检测、跟踪及计数。**演示效果如下：

二、[开发环境](https://so.csdn.net/so/search?q=%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83&spm=1001.2101.3001.7020)（前提条件）
------------------------------------------------------------------------------------------------------------

1、Anaconda3 环境

2、pycharm 代码编辑器

3、虚拟环境 python 3.8

**（安装教程：[Anaconda3+pycharm 安装教程](https://blog.csdn.net/Little_Carter/article/details/131031595?spm=1001.2014.3001.5501 "Anaconda3+pycharm安装教程")）（强烈推荐**√**）**

**（代码安装资源：[YOLOv8-Deepsort 免费源码](https://download.csdn.net/download/Little_Carter/88398917 "YOLOv8-Deepsort 免费源码")）（强烈推荐**√**）**

**因为看到很多开源的资源都是要么付费，要么需要 vip 才能下载，实在看不下去了！！！所以我决定代码直接免费，这么良心的博主不给个点赞 + 关注 + 收藏嘛 (๑′ᴗ‵๑)**

三、[环境搭建](https://so.csdn.net/so/search?q=%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA&spm=1001.2101.3001.7020)教程
--------------------------------------------------------------------------------------------------------

### 3.1、[创建虚拟环境](https://so.csdn.net/so/search?q=%E5%88%9B%E5%BB%BA%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83&spm=1001.2101.3001.7020)

首先打开 anaconda prompt，输入 conda env list 查看环境列表，如果没有安装虚拟环境，会显示只有一个 base。

![](https://img-blog.csdnimg.cn/170e5c9606d8474694213af9cbfcf01f.png)

然后输入指令：

```
conda create -n YOLOv8-Deepsort python=3.8

```

接着输入 y，等待安装完毕，即可创建好虚拟环境。

（注意：YOLOv8-Deepsort 是我自己命名的环境名称，可随意命名。）

等待安装好后再次输入：

```
conda env list

```

查看环境列表，此时环境中就会多出自己创建的虚拟环境了。

![](https://img-blog.csdnimg.cn/90761058bb0e40c8a400b5c5ed1c7c64.png)

### 3.2、选择虚拟环境并安装所需要的包

输入 conda activate YOLOv8-Deepsort 进入你的虚拟环境

```
conda activate YOLOv8-Deepsort

```

如果前面的括号里由原来的 base 变成了你的虚拟环境名称，那么恭喜你，环境选择成功了哟！ 

![](https://img-blog.csdnimg.cn/cf6141b3f2fd4517a5ff160ba7a739b4.png)

### 3.3、运行代码步骤

#### 3.3.1、克隆 git 储存库

```
git clone https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking.git

```

**也可以点击这个资源免费下载：[YOLOv8-Deepsort 免费源码](https://download.csdn.net/download/Little_Carter/88398917 "YOLOv8-Deepsort 免费源码")（强烈推荐）**

#### 3.3.2、转到克隆库的文件夹下

```
cd YOLOv8-DeepSORT-Object-Tracking

```

#### 3.3.3、安装依赖项

```
pip install -e ".[dev]"

```

#### 3.3.4、转到检测目录下

```
cd ultralytics/yolo/v8/detect

```

#### 3.3.5、用于 yolov8 物体检测 + 跟踪 + 车辆计数

```
python predict.py model=yolov8l.pt source="test3.mp4" show=True

```

**四、效果图**
---------

**运行完以上那些命令就可以像视频那样的效果啦 (～￣▽￣)～**

![](https://img-blog.csdnimg.cn/2f7fb14b90cd4ccda7ee0f8f74803a20.png)

**五、可能出现的 Error（如果出现报错请看这个）**
-----------------------------

![](https://img-blog.csdnimg.cn/7b427b8409944b0a834c01521a766e3a.png)

```
AttributeError: module 'numpy' has no attribute 'float'
 
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

**出现以上的报错呢，**可以这样改，我使用的 numpy 版本是 1.24，但是从代码中所用的代码是依赖于旧版本的 Numpy。您可以将你的 Numpy 版本降级到 1.23.5。

```
pip install numpy==1.23.5

```

**然后再运行上面那个用于 yolov8 物体检测 + 跟踪 + 车辆计数的运行命令即可。**