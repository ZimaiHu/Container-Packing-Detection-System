## 文件说明：

```markdown
1. ceshitu（照片）
2. container_number(货柜号竖排截字转横排模型)
3. detect(破损检测以及，货柜号位置检测模型)
4. detectors（集成模型，有货物标签识别器、拆托识别器、货柜号识别器、异物识别器、托盘角识别器、封条号识别器、绑带识别器）
5. weights(权重文件：货物标签、封条、异物、手写体、托角)
6. detector.py(集成探测器)
7. app.py(调用探测器的文件)
```

## 问题修复：

1. 单个箱体没识别为多个
```markdown
> 解决办法：修改货物标签探测器的置信度为0.65
```

2. 报错情况：TypeError: Object of type float32 is not JSON serializable
```markdown
> 原因：数据中存在的float32数据为numpy格式的数值，python的内置类型float可以写入json，然而numpy类型的float不能写入json。                                                                
> 解决办法：转换np类型为python类型float
```
