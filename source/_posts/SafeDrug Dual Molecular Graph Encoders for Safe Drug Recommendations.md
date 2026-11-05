# SafeDrug: Dual Molecular Graph Encoders for Safe Drug Recommendations

# 论文总结与启发

![image-20261022105054758](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261022105054758.png)

![image-20261022105128593](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261022105128593.png)

![image-20261022105145903](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261022105145903.png)

# 论文内容

## 摘要

safedrug 使用了NPNN模型和局部二分类模型充分编码了药物分子的连接性和功能性

### 主要问题

1.药物编码不合适：每个药物被视为独立个体，忽略了药物之间重要的联系

2.隐藏的不可控的DDI模型：使得最后的药物推荐的准确率不可控

### safedrug所做的改动

1.定义了两个分子编码器捕获全局和局部药物分子规律

全局层面使用的是MPNN编码器逐层捕获全部的药物分子信息。局部层面采用（ bipartite encoder）二分类编码器，将药物分子拆分成更小的功能团，与其与更小的功能组联系

2.DDI可以控制的损失函数

受pid控制器的启发，组合监督损失和无监督控制器相组合，设计了可变的DDI控制器，若DDI率超过阈值DDI抑制信号会被强调

```
PID控制器（比例-积分-微分控制器）是工业控制领域应用最广泛的反馈控制机制，已有近70年历史。它的核心思想是：通过持续测量“实际输出”与“期望目标”之间的误差，动态调整控制量，使系统输出稳定在目标值附近。

它由三个核心部分组成：

比例项（P）：根据当前误差的大小进行调节。误差越大，调节力度越强。

积分项（I）：累积历史误差，用于消除长期存在的稳态偏差。

微分项（D）：预测误差的变化趋势，提前进行调节，避免系统“矫枉过正”。

PID控制器被借鉴到深度学习领域，用于优化网络训练过程。例如，CVPR 2018的一篇论文就利用PID思想来优化随机梯度下降，有效减少了训练中的震荡现象。SafeDrug正是受此启发，将其用于动态平衡推荐准确性与用药安全性。

```

```
VGG、ResNet、DenseNet等主流架构大多使用SGD-Momentum进行优化，它通过同时考虑历史梯度和当前梯度来更新权重。

然而，SGD-Momentum存在一个固有缺陷——过冲问题（Overshoot Problem）。在优化过程中，参数更新量可能会越过最优点，导致损失函数在最小值附近来回震荡，而不是平稳收敛。这种震荡阻碍了网络的快速收敛，造成了不必要的训练时间消耗。

论文的关键洞察在于：SGD-Momentum的过冲问题，本质上是控制系统中典型的“超调”现象。而控制领域早已有成熟的解决方案——PID控制器。
 核心方法：将优化过程视为控制系统
论文的核心思路是将深度网络的训练过程类比为一个控制系统：

```

![image-20261015204828746](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261015204828746.png)

- **SGD是P控制器**：其更新仅依赖当前梯度，对应PID中的比例项。
- **SGD-Momentum是PI控制器**：其更新依赖当前梯度（P项）和历史梯度累积（I项），但**完全没有微分项（D项）**。

**缺少D项正是过冲问题的根源**。在控制系统中，微分项的作用是**预测误差的变化趋势，提前施加“阻尼”**，防止系统冲过头。SGD-Momentum因为缺少这一项，无法感知“即将过冲”的趋势，因而在接近最优点时会冲过头，产生震荡。

![image-20261015205026501](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261015205026501.png)

![image-20261015205112848](C:\Users\zzy15\AppData\Roaming\Typora\typora-user-images\image-20261015205112848.png)

3.全方面评估

## 模型的定义

### 问题定义

![image-20261020095848589](C:\Users\zzy15\AppData\Roaming\Typora\typora-user-images\image-20261020095848589.png)



![image-20261020105421638](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020105421638.png)

![image-20261020105330338](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020105330338.png)

### 多值损失函数

![image-20261020110729489](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020110729489.png)

![image-20261020110821927](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020110821927.png)

![image-20261020110831954](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020110831954.png)

### Jaccard是什么准确度

![image-20261020141852623](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020141852623.png)

![image-20261020141959810](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020141959810.png)

![image-20261020142005502](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020142005502.png)

### 显著性测试

![image-20261020143342486](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020143342486.png)

# 代码debug

## 数据的输入

1.数据清洗--med_process

```
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={"NDC": "category"})   #创建pd对象，读取csv文件
    med_pd.drop
    #删除不需要的列,axis=1表示列，0表示行 inplace=True：原地修改。直接在原 DataFrame 上删除，而不是返回一个新的副本     
        columns=[],
        axis=1,                                   
        inplace=True,
    )
    med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    #删除ndc为0的行，ndc为药物编号，ndc为0，表示医生没有录入药物，为无效数据
    
    med_pd.fillna(method="pad", inplace=True)
    ###缺失值填充，如果当前行的值为NAN就用它上一行的数据来填补，pad值向前填充，PRESCRIPTIONS.csv中医生开了一组医嘱（同一个	药，但不同剂量或频次），系统在导出 CSV 时，可能会拆成好几行
							第 1 行：NDC_01, 阿司匹林, 剂量100mg
							第 2 行：(空), (空), 剂量50mg
							第 3 行：(空), (空), 剂量25mg
	###
    
    med_pd.dropna(inplace=True)
    #删除缺失值的行或列对于某些数据第一行开始就是空的，，fillna无法填充，所以就整行整列都删掉
    
    med_pd.drop_duplicates(inplace=True)
    #删除完全相同的的重复行
    
    med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
    #把 ICUSTAY_ID 这一列强制转换为 64 位整数。因为原始数据里可能因为有空值或浮点数，被 Pandas 自动识别成了 float64（如 301.0），转成整数后方便后续排序和展示。
    
    
    
    med_pd["STARTDATE"] = pd.to_datetime(
        med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S"
    )
    #把字符串格式的时间转换成 Pandas 的 Timestamp 对象，这是时间序列处理的基础
    
    med_pd.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], inplace=True
    )
    #指定排序顺序
    
    med_pd = med_pd.reset_index(drop=True)
	#经过上面的排序，index已经打乱，删除原来的索引chong'xin'pai'xv
    med_pd = med_pd.drop(columns=["ICUSTAY_ID"])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd	
```



```

# ATC3-to-drugname，将atc3转换为药物名字

def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[["ATC3", "DRUG"]].values:    #
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)

    return atc3toDrugDict


```















![image-20261020144026549](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020144026549.png)

![](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020144426165.png)

![image-20261020145616451](C:\Users\zzy15\AppData\Roaming\Typora\typora-user-images\image-20261020145616451.png)

![image-20261020144503722](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20261020144503722.png)

