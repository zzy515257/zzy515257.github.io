# 一.代码笔记

## 1.argparse类

`argparse.ArgumentParser` 严格来说不是“函数”，而是 Python 标准库 `argparse` 里的一个**类**。
你写 `argparse.ArgumentParser(...)` 时，是在**创建这个类的实例**，也就是创建一个“命令行参数解析器”对象。

它的作用是：
让你在运行 Python 脚本时，可以通过命令行传入参数，例如：

```
python train.py --config config.json --experiment msra_bert
```

![image-20260921002929127](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921002929127.png)

![image-20260921002946047](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921002946047.png)

![image-20260921002957928](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921002957928.png)

完整例子

```
# demo.py
import argparse


def main():
    parser = argparse.ArgumentParser(description="参数解析示例")

    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--experiment", type=str, default=None, help="实验名")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--use_swanlab", action="store_true", help="是否启用 SwanLab")

    args = parser.parse_args()

    print("config =", args.config)
    print("experiment =", args.experiment)
    print("epochs =", args.epochs)
    print("lr =", args.lr)
    print("use_swanlab =", args.use_swanlab)


if __name__ == "__main__":
    main()
```

![image-20260921003901702](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921003901702.png)

![image-20260921003914148](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921003914148.png)

![image-20260921003928495](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921003928495.png)

![image-20260921003944565](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921003944565.png)

# 二.逻辑分析

## 1.config文件

## 2.dataoader文件

### 1.怎么输入数据的，模型对输入的要求是是什么，dataloader是怎么处理数据的

![image-20260921005748349](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921005748349.png)

![image-20260921005815996](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921005815996.png)

![image-20260921005831321](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921005831321.png)

![image-20260921005933635](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921005933635.png)

![image-20260921005946705](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921005946705.png)

![image-20260921005959793](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921005959793.png)

### 2.如何给token打标签

![image-20260921101412728](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101412728.png)

为什么要这么做

![image-20260921101517262](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101517262.png)

![image-20260921101545384](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101545384.png)

![image-20260921101605407](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101605407.png)

![image-20260921101629263](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101629263.png)

![image-20260921101648239](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101648239.png)

![image-20260921101715389](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101715389.png)

![image-20260921101729043](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921101729043.png)

### 3.NERDataset和DataLoader作用

![image-20260921111356965](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111356965.png)

![image-20260921111417625](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111417625.png)

![image-20260921111434917](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111434917.png)

-----------------------------------------dataloader-----------------------------------

![image-20260921111514790](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111514790.png)

![image-20260921111525598](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111525598.png)

![image-20260921111537533](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111537533.png)

![image-20260921111607352](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111607352.png)

![image-20260921111618197](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111618197.png)

![image-20260921111650184](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260921111650184.png)

### 3.train文件

![image-20260922090915971](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922090915971.png)

```

            for pred, label in zip(pred_ids, label_ids):
            #将预测值的id与标签id分别传给pred, label
                p_list, t_list = [], []
                for p, l in zip(pred, label):
                    if l == -100:
                        continue
                    p_list.append(id2label[int(p)])
                    t_list.append(id2label[int(l)])
                preds.append(p_list)
                trues.append(t_list)

```

![image-20260922091735462](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922091735462.png)

![image-20260922091808888](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922091808888.png)

## 三.代码解释

### 1.zip迭代器

![image-20260922091249414](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922091249414.png)

### 2.classification_report函数

![image-20260922091925029](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922091925029.png)

![image-20260922092019413](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922092019413.png)

![image-20260922092333818](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922092333818.png)

### 3.函数定义的类型注解（type hints）和 默认参数字段

![image-20260922093240845](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922093240845.png)

## 四.理论学习

#### 1.adaw算法

AdamW 优化器是为了解决**原始 Adam 优化器中权重衰减（Weight Decay）实现方式存在缺陷**而提出的。它将权重衰减与梯度更新**解耦**，使得 Adam 在引入权重衰减时也能获得与 SGD 相当的泛化性能。

**为什么要引入 AdamW？**

核心原因在于：对于 Adam 这类自适应梯度算法，L2 正则化与权重衰减并不等价。

在标准的随机梯度下降（SGD）中，L2 正则化与权重衰减是等效的。但在 Adam 中，由于自适应学习率的存在，二者出现了分歧。

**AdamW 解决了什么问题？**

原始 Adam 中权重衰减的“耦合”问题

在原始 Adam 中，L2 正则化是通过在损失函数中加入惩罚项 `(λ/2) * ||θ||²` 来实现的。这个惩罚项的梯度 `λ * θ` 会混入任务梯度 `g_t` 中，一起进入一阶矩（动量）和二阶矩（方差）的估计。

这会导致一个严重问题：自适应学习率机制会“错误地”缩放这个正则化项。具体来说，梯度 `g_t` 会被 `1 / (√v̂_t + ε)` 按坐标自适应地缩放，而包含了正则化项的 `g_t` 也同样被缩放。这意味着，对于梯度较大的参数，其权重衰减的力度会被自适应地减小，导致“让权重变小”这一目标变得不再干净和均匀。

**AdamW 的解决方案：解耦权重衰减**

AdamW 的核心思想是将权重衰减从梯度计算中彻底分离出来。它不再把 L2 惩罚的梯度混入 `g_t`，而是在每一步更新时，先独立地对参数进行一次按比例的衰减，然后再用纯任务梯度进行 Adam 的自适应更新。

这样做的好处是：

1. 权重衰减的力度变得均匀且可预测：所有参数都按相同的比例 `λ` 衰减，不再受各自梯度大小的影响。
2. 显著提升了泛化能力：实验表明，使用解耦权重衰减的 AdamW 在训练损失和泛化误差上都优于使用 L2 正则化的原始 Adam。

**AdamW 的计算过程**

AdamW 的更新步骤可以概括为以下几个步骤（以 PyTorch 文档为准）：

![image-20260922104928999](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922104928999.png)

![image-20260922104958259](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922104958259.png)

![image-20260922105017395](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105017395.png)

![image-20260922105025855](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105025855.png)

-----------------------------进一步解释

![image-20260922105512176](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105512176.png)

![image-20260922105526866](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105526866.png)

![image-20260922105541206](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105541206.png)

![image-20260922105552683](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105552683.png)

![image-20260922105607027](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105607027.png)

![image-20260922105624909](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105624909.png)

![image-20260922105637300](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105637300.png)

![image-20260922105647532](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105647532.png)

![image-20260922105656478](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260922105656478.png)