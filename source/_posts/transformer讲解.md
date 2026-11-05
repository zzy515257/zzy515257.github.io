# transformer讲解

## 学习内容

1.前置知识

```
线性代数：矩阵乘法、转置、softmax、广播。

PyTorch：Tensor、nn.Linear、nn.Embedding、CrossEntropyLoss、DataLoader。

NLP 基础：tokenization、词表、padding、OOV、词性标注。

数学基础：交叉熵、LayerNorm、注意力公式。
```

## 一.多头注意力机制

### 1.注意力机制原理

![](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918141430260.png)

注意力机制通过定义注意力函数，利用可学习的线性投影（Wq,Wk,Wv*W**q*,*W**k*,*W**v*）生成查询、键和值，计算 Query 和 Key 的相似度得分并归一化，从而对 Value 进行加权求和。

在训练过程中，通过反向传播最小化整体损失函数（Loss），模型会自动调整这些投影参数。最终，**与当前查询（Query）语义高度相关的键（Key）**会获得更高的注意力权重，从而让模型动态聚焦于最相关的信息。它关注的是‘相关性’，而不是绝对的‘临近性’。”

### 2.例子解释

![image-20260918140903198](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918140903198.png)

![image-20260918140921136](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918140921136.png)

![image-20260918140942692](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918140942692.png)

![image-20260918141000382](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918141000382.png)

![image-20260918141022947](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918141022947.png)

![image-20260918141051987](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918141051987.png)

![image-20260918141117847](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918141117847.png)

![image-20260918154206515](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918154206515.png)

![image-20260918154234290](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918154234290.png)

![image-20260918154716434](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918154716434.png)

## 二.位置编码

原因：多注意力机制无法得知序列的位置信息，通过在输入中添加位置编码，注入绝对或相对的位置信息

![image-20260918163802556](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918163805756.png)

![image-20260918162217734](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918162217734.png)

![image-20260918163616917](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918163616917.png)

![image-20260918163706995](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918163706995.png)

![image-20260918164057684](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918164057684.png)

![image-20260918164128291](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918164128291.png)

![image-20260918164204432](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918164204432.png)

## 三.transformer架构

![image-20260917112449964](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260917112449964.png)

transformer没有任何的卷积层或者循环神经网络，完全基于注意力机制，它由编码器和解码器组成，transformer由多个相同的层叠加而成，每个层都有两个子层，一个是多注意力机制汇聚，一个是基于位置的前馈神经网络，解码器还会有编码器——解码器注意力子层

### 编码器与解码器的输入与输出

![image-20260918170247663](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918170247663.png)

![image-20260918170216777](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918170216777.png)

![image-20260918170444660](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918170444660.png)

### 1.工作过程

![image-20260919104832243](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919104832243.png)

![image-20260919105032956](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919105032956.png)

![image-20260919105334561](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919105334561.png)

![image-20260919105352189](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919105352189.png)

为什么编码器要叠多层：单层只能看到浅表关系，多层才能“消化”和“提炼”出深层次的语境与逻辑。

#### 参数tgt_seq_len

tgt_seq_len：目标句子长度，训练阶段为目标句子长度，预测阶段动态调整

![image-20260918174018810](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918174018810.png)

![image-20260918174116266](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260918174116266.png)

### 2.编码器的两个子层

#### 多头注意力机制

输入参数：（key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)

输出形状（batch_size,seq_len,num_hiddens)，使用的是缩放点积注意力

多头注意力机制输出的是什么，有什么含义：

![image-20260919121002261](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919121002261.png)

![image-20260919121539890](C:\Users\zzy15\AppData\Roaming\Typora\typora-user-images\image-20260919121539890.png)

![image-20260919121725053](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919121725053.png)

![image-20260919121753419](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919121753419.png)

![image-20260919121440053](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919121440053.png)

问题

```
是不是有几个头就对应有几个键值对和查询，但是键值对和查询都是由输入乘权重矩阵的来的，是不是就代表有几组权重矩阵，但是权重矩阵又是学习参数，怎么保证权重矩阵实在不同语义对句子进行的理解呢
```

![image-20260919122235925](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919122235925.png)

![image-20260919122339686](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919122339686.png)

![image-20260919122417041](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919122417041.png)

#### 残差连接

残差连接是 Transformer 能够堆叠到几十层（甚至上百层）而不会崩溃的根本原因。

```
LayerNorm(x + Sublayer(x))
这就是“Add & Norm”（相加与归一化） 层
```

1. 编码器（Encoder）中

- 自注意力子层（Self-Attention）外包了一层残差连接。
- 前馈网络子层（FFN）外包了一层残差连接。

2. 解码器（Decoder）中

- 掩蔽自注意力子层（Masked Self-Attention）外包了一层。
- 交叉注意力子层（Cross-Attention）外包了一层。
- 前馈网络子层（FFN）外包了一层。
- ![image-20260920123301431](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920123301431.png)

![image-20260920123458488](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920123458488.png)

![image-20260920123937367](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920123937367.png)

![image-20260920123953124](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920123953124.png)

![image-20260920124013181](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920124013181.png)

![image-20260920124046618](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920124046618.png)

![](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920124359743.png)

![image-20260920124707354](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920124707354.png)

问题：那梯度至少为1会不会模型已经收敛了，但梯度仍不为0，还在学习，新参数不停在变，导致模型没法收敛

![image-20260920131705301](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920131705301.png)

![image-20260920132456831](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920132456831.png)

如果输出层求导为零，根据链式求导法则，模型也能收敛

#### 归一化流程

![image-20260920134017481](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920134017481.png)

#### ![image-20260920134810906](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920134810906.png)

![image-20260920134833861](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920134833861.png)

### 3.解码器中的三个子层

#### 基于位置的前馈网络

1.构成：它极其简单，就是一个两层全连接网络（MLP），中间夹着一个激活函数

```
self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
self.relu = nn.ReLU()
self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
输入维度：ffn_num_input（通常等于 Transformer 的隐藏层维度 ，比如 512）。

隐藏层维度：ffn_num_hiddens（通常是dmodel的 4 倍，比如 2048）。

输出维度：ffn_num_outputs（通常重新变回dmodel，512，以便与残差连接相加）。

激活函数：ReLU（原版论文用的是 ReLU，后来也有用 GELU 的，如 BERT 和 GPT）。
```

![image-20260919120254851](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919120254851.png)

![image-20260919120643416](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919120643416.png)



#### 解码器自注意力机制--掩蔽多注意力

特点：查询，键值都来自上一编码器输出，在训练阶段通过掩蔽多头注意力，确保每个位置只能考虑该位置之前的所有位置，保留了自回归属性

![image-20260920140741372](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920140741372.png)

![image-20260920140928815](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920140928815.png)

![image-20260920140945464](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920140945464.png)

![image-20260920140959332](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920140959332.png)

![image-20260920141526141](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920141526141.png)

掩蔽多注意力是只有第一个解码器层有吗，后续堆叠的解码器层会有吗

![image-20260920141855074](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920141855074.png)

![image-20260920141912008](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920141912008.png)

![image-20260920141931088](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920141931088.png)

#### 编码器-解码器注意力（交叉注意力）

特点：在编码器-解码器注意力中，查询来自前一个解码器层的输出，而键和值来自整个编码器的输出

**Query（查询）**：代表解码器当前的生成状态（“我现在想生成下一个词，我需要什么信息？”）。

**Key（键）和 Value（值）**：代表编码器对源句子的完整理解（“源句子里有什么信息可以提供给解码器？”）。

![image-20260919103528871](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919103528871.png)

![image-20260919103747558](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919103747558.png)

![image-20260919103802870](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919103802870.png)

![image-20260919104239576](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919104239576.png)

![image-20260919104340186](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919104340186.png)

![image-20260919104403270](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919104403270.png)

![image-20260919104518304](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260919104518304.png)



## 四.Transformer各层输入维度

![image-20260920142657423](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920142657423.png)



![image-20260920142718782](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920143008234.png)

![image-20260920142735723](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920142735723.png)

![image-20260920142754193](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920142754193.png)

![image-20260920142808054](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920142808054.png)

![image-20260920142827426](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920142827426.png)

## 五.杂记

### 1.预测输出只会根据batch内的计算输出

![image-20260920144436699](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920144436699.png)

![image-20260920144534274](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920144534274.png)

### 2.关于batch之间是否共享参数的讨论

那我有1000字，每个batch500字，那么我输入1000字还是500字，对1001个字的预测是一样的对吗

![image-20260920144914847](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920144914847.png)

![image-20260920144927474](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920144927474.png)

![image-20260920144950109](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920144950109.png)

![image-20260920145019438](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145019438.png)

![image-20260920145038680](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145038680.png)

### 3.多batch训练的意义

训练的时候为什么要把样本切片，既然batch之间权重不共享，那我训练一个batch不就行了，预测的时候用训练好的参数，那为什么还要训练这么多次

![image-20260920145341873](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145341873.png)

![image-20260920145350817](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145350817.png)

![image-20260920145502308](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145502308.png)

![image-20260920145515022](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145515022.png)

![image-20260920145528563](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145528563.png)

### 4.多batch训练哪些参数

![image-20260920145643151](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145643151.png)

![image-20260920145655773](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145655773.png)

![image-20260920145713520](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145713520.png)

### 5.transformer训练过程

![image-20260920145941116](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145941116.png)

![image-20260920145952505](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920145952505.png)

![image-20260920150004500](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920150004500.png)

![image-20260920150012438](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920150012438.png)

### 6.训练过程中的教师机制

![image-20260920150211790](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920150211790.png)

![image-20260920150220472](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920150220472.png)

![image-20260920150232224](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920150232224.png)

![image-20260920150239779](https://cdn.jsdelivr.net/gh/zzy515257/bolgimage/img/image-20260920150239779.png)

![image-20260920150248354](C:\Users\zzy15\AppData\Roaming\Typora\typora-user-images\image-20260920150248354.png)