## 项目介绍：
集装箱运输在国际物流中扮演着至关重要的角色。集装箱的标准化使得货物的装载、运输和卸载变得更加高效，本工作借助图像识别技术自动检测和分析集装箱装箱情况，以降低传统的人工检查方法耗时耗力问题，提高装箱效率和安全性。具体功能包含：
- XXXXX
- XXXXX
- XXXXX

## 文件目录结构
```bash
Container-Packing-Detection-System
|---ceshitu  # 照片
|
|---container_number   # 货柜号竖排截字转横排模型
|
|---detect	# 破损检测以及，货柜号位置检测模型
|
|---detectors    # 集成模型，有货物标签识别器、拆托识别器、货柜号识别器、异物识别器、托盘角识别器、封条号识别器、绑带识别器
|     |---XXX    # 货物标签识别器
|     |---XXX    # 拆托识别器
|     |---XXX    # 货柜号识别器
|     |---XXX    # 异物识别器
|     |     |- XXX   # 图片
|     |---XXX    # 托盘角识别器
|     |     |- XXX   # XXX
|     |     |   |- XXX	# XXX
|     |     |   |- XXX	# XXX
|     |---XXX    # 封条号识别器
|     |---XXX    # 绑带识别器
|---weights    # 权重文件
|     |---XXX    # 货物标签
|     |---XXX    # 封条
|     |---XXX    # 异物
|     |---XXX    # 手写体
|     |---XXX    # 托角
|---detector   # 集成探测器
|
|---test	# 测试脚本
|
|---app	# 调用探测器的文件
```


## 问题修复：
1. 问题：单个箱体被识别为多个箱体。
> 原因：没有设置置信度，导致对于置信度低的也认为是箱体。
> 解决办法：修改货物标签探测器的置信度为0.65

2. 报错情况：TypeError: Object of type float32 is not JSON serializable
> 原因：数据中存在的float32数据为numpy格式的数值，python的内置类型float可以写入json，然而numpy类型的float不能写入json。                                                                
> 解决办法：转换np类型为python类型float
