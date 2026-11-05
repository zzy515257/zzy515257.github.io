# 一.python函数解释

## 1.enumerate() 

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中

```
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

## 2.map（）函数

```
map(label2id) 是什么意思？
map() 是 Pandas（pd）中非常常用的一种数据映射方法。它会遍历 train_df['label'] 这一列里的每一个数值，拿着这个数值去 label2id 这个“字典”里查找，把查到的结果放回新的一列中。

就像查字典：给你一个英文单词（原始 Label），你查英汉字典（label2id），拿到中文释义（新的 Label_id）。
```

## 3.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

```
1. BertTokenizer
这是什么：它是 transformers 库中专门为 BERT 模型设计的文本预处理工具。

它的任务：它不懂中文语义，只懂“切分”和“编号”。它负责把你看得懂的中文句子（如 "我想买车"）切分成 BERT 模型能识别的 Token（词块），并转换成对应的数字索引（input_ids）。

注意：中文 BERT 是按字切分的（偶尔包含少量英文词汇），所以它会把 "我想买车" 切分成 ['我', '想', '买', '车']。

2. .from_pretrained(...)
这是什么：这是 transformers 库提供的一个统一加载接口。它非常智能，既可以接受官方的模型名称（如 'bert-base-chinese'），也可以接受你电脑里的本地文件夹路径。

它的任务：它会去读取文件夹里的 vocab.txt（词表文件）、tokenizer_config.json（配置）等文件，根据这些文件来构建分词器的内部逻辑。

3. MODEL_PATH
这是什么：代码开头定义的一个变量，指向你本地存放模型的文件夹路径（比如 G:\...\bert-base-chinese）。

为什么不用 'bert-base-chinese'：如果你写字符串 'bert-base-chinese'，程序会默认去 Hugging Face 官网下载。鉴于你之前遇到了网络超时错误，这里直接指定你已下载好的本地路径，彻底避免了联网请求。

4. local_files_only=True（核心关键参数）
这是什么：这是一个强制离线开关。

它的任务：告诉程序：“只准在 MODEL_PATH 这个文件夹里找文件，如果找不到任何文件（比如 vocab.txt 缺失），就直接报错，绝对不许尝试联网去下载。”

为什么加它：加了这个参数后，程序运行非常稳定。即使你的网络环境突然波动，或者 Hugging Face 官网崩溃，也完全不影响你运行代码。如果没有这个参数，程序一旦发现本地文件不完整，就会偷偷去联网下载，导致你再次遇到 Connection error。


```

## 4.DataFrame.iterrows()

```
iterrows() 方法用于逐行迭代 DataFrame，每次迭代返回一个 (index, Series) 元组，其中 index 是行标签，Series 包含行的数据。
```

## 5.f"{row['title']} {row['content']}"

```
f"{row['title']} {row['content']}"：

这是一个 f-string（格式化字符串）。

它从当前行取出 title 列和 content 列的内容，中间加一个空格将它们连接起来。

为什么要这样？因为标题往往是正文的核心摘要，把标题放在正文前面，用空格隔开，能让 BERT 模型优先注意到标题的关键信息。
```

## 6.TensorDataset

```
这是什么：TensorDataset 是 PyTorch 中 torch.utils.data 模块提供的一个包装类。

它的功能：它只做一件非常简单的事——按位置一一对应地存储张量。

核心机制：当你用索引 [0] 去访问它时，它会把你传入的所有张量中，索引为 0 的那一行数据打包成一个元组（Tuple） 返回。
```

## 7.compile 函数

**compile 函数用于编译正则表达式，生成一个 Pattern 对象**，它的一般使用形式如下：

```
re.compile(pattern[, flag])
match 方法 search 方法 findall 方法 finditer 方法 split 方法 sub 方法 subn 方法
```

# 二.知识积累

## 把文字标题数字化了（tokenizer）

用户问的是哪一步把文本标题数字化的。他们想具体了解从文本到数字的转换发生在哪里。

数字化发生在 **`tokenizer(...)` 这一步**，也就是 `encode_texts` 函数里的第 3 行到第 8 行。

具体来说，就是下面这段代码：

python

```
return tokenizer(
    texts,                 # 上一步拼接好的中文文本列表
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors='pt'
)
```



**在这一行代码内部**，发生了从「中文文字」到「数字 ID」的转换。

------

一、转换的完整过程（以一条文本为例）

假设原始拼接后的文本是：

text

```
"暴雨天气开车 需要这波操作"
```



当它被传入 `tokenizer` 时，内部依次执行了 5 个步骤：

第 1 步：分词（Tokenize）

把整句切成一个个 Token（中文 BERT 按字切分）：

text

```
['暴', '雨', '天', '气', '开', '车', '需', '要', '这', '波', '操', '作']
```



第 2 步：添加特殊标记

BERT 需要在句首加 `[CLS]`，句尾加 `[SEP]`：

text

```
['[CLS]', '暴', '雨', '天', '气', '开', '车', '需', '要', '这', '波', '操', '作', '[SEP]']
```



第 3 步：查词表，转成数字 ID（**这就是数字化的核心**）

分词器内部维护了一个**词表文件 `vocab.txt`**，里面记录了每个汉字对应的整数编号。查表后得到：

text

```
[101, 6887, 5107, 5955, 6295, 6575, 6846, 4490, 7552, 6903, 4263, 6677, 5411, 102]
```



其中：

- `101` 是 `[CLS]` 的固定编号
- `102` 是 `[SEP]` 的固定编号
- 其他数字都是对应汉字的编号

**至此，中文文字已经变成了数字 ID 序列（`input_ids`）。**

第 4 步：补齐长度（Padding）

如果上面只有 14 个数字，而 `MAX_LEN = 128`，就在末尾补 114 个 `0`（`[PAD]` 的编号），凑够 128 个数字。

第 5 步：生成 attention_mask

同时生成一个与之配套的掩码：

text

```
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ... 0]
```



前 14 位是 1（真实文字），后 114 位是 0（填充符）。

最终返回的字典里，`input_ids` 就是数字化的结果。

------

二、关键：`vocab.txt` 词表文件

数字化的依据来自你下载的模型文件夹里的 **`vocab.txt`**：

text

```
G:\...\bert-base-chinese\vocab.txt
```



这个文件的内容大致长这样（简化版）：

text

```
[PAD]   0
[unused1] 1
...
[CLS]  101
[SEP]  102
[UNK]  100
暴     6887
雨     5107
天     5955
...
```



分词器读入这个文件后，就建立了一张**「汉字 → 整数」的映射表**。当你传入中文句子时，它逐个字符去查表，拿到对应的数字。

------

三、数字化代码对应关系图

text

```
encode_texts(df)
    │
    ├── texts = [...]                 # ① 拼接原始中文文本
    │
    └── tokenizer(texts, ...)         # ② 这里发生数字化！
            │
            ├── 分词（按字切分）
            ├── 加 [CLS] / [SEP]
            ├── 查 vocab.txt 词表   ← 💡 核心：汉字变成整数
            ├── 补齐到 128 长度
            └── 返回 input_ids (3000, 128)   ← 💡 数字化的结果
```



------

四、直观对比

| 阶段                               | 数据长什么样                                                 |
| :--------------------------------- | :----------------------------------------------------------- |
| **数字化前**（Python 字符串）      | `"暴雨天气开车 需要这波操作"`                                |
| **数字化后**（张量，截取前 14 位） | `[101, 6887, 5107, 5955, 6295, 6575, 6846, 4490, 7552, 6903, 4263, 6677, 5411, 102]` |

------

五、一句话总结

**数字化的那一刻，就是调用 `tokenizer(...)` 的瞬间。** 分词器通过查 `vocab.txt` 词表，把每一个汉字替换成对应的整数编号，输出 `input_ids` 张量。你之前拼接文本（`f"{row['title']} {row['content']}"`）那一步只是**准备原料**，真正的**数字化加工**全部发生在 `tokenizer(...)` 内部。









## 将训练的权重保存下来--torch.save()

用户想要保存训练好的模型权重。这是 PyTorch 中的标准实践。让我为他们的 BERT 文本分类项目提供全面的指导。

保存训练好的模型权重是深度学习项目的标准操作，方便后续推理、部署或继续训练。对于你现在的 BERT 文本分类项目，有三种常用方式，各有适用场景。下面逐一介绍，并给出整合到你现有代码的最佳方案。

------

一、三种保存方式对比

| 方式                                      | 保存内容        | 文件大小       | 适用场景               |
| :---------------------------------------- | :-------------- | :------------- | :--------------------- |
| **1. 只保存 state_dict**                  | 仅权重参数      | 小（约 400MB） | **推荐**，标准做法     |
| **2. 保存整个模型对象**                   | 模型结构 + 权重 | 大             | 快速恢复，但依赖原代码 |
| **3. 用 transformers 的 save_pretrained** | 权重 + 配置文件 | 中等           | 兼容 Hugging Face 生态 |

**最推荐方式 1**：通用、可移植、只保存参数，加载时需要有相同的模型代码。

------

二、方式一：保存 `state_dict`（推荐）

在训练结束后，调用 `torch.save()` 保存模型的参数字典。

修改 `run_experiment` 函数

python

```
def run_experiment(lr, batch_size, dropout, run_name=None, save_path=None):
    train_loader, dev_loader, test_loader = build_loaders(batch_size)
    model = BertClassifier(num_classes, dropout=dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    swanlab.init(
        project=SWAN_PROJECT,
        experiment_name=run_name or f"lr{lr}_bs{batch_size}_do{dropout}",
        config={...},
        mode=SWAN_MODE
    )

    # 训练循环
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)
        swanlab.log({...})
        print(f"Epoch {epoch}: ...")

    # 测试集评估
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    swanlab.log({"test_acc": test_acc})
    print(f"Test Acc: {test_acc:.4f}")

    # ===== 保存权重 =====
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to: {save_path}")

    swanlab.finish()
    return test_acc, model
```



在主程序中调用

python

```
if __name__ == "__main__":
    DO_HYPERPARAM_SEARCH = cfg['training']['do_hyperparam_search']

    if not DO_HYPERPARAM_SEARCH:
        # 指定保存路径（建议放在模型目录下）
        SAVE_PATH = os.path.join(DATA_DIR, "bert_classifier_weights.pt")

        test_acc, model = run_experiment(
            lr=DEFAULT_LR,
            batch_size=DEFAULT_BS,
            dropout=DEFAULT_DO,
            run_name="default_from_json",
            save_path=SAVE_PATH
        )
        print(f"Test Accuracy: {test_acc:.4f}")
```



运行后，你会在 `DATA_DIR` 下看到 `bert_classifier_weights.pt` 文件（约 400MB）。

------

三、加载权重进行推理

保存后，如果想用这个权重做预测，需要**先重建模型结构**，再加载参数：

python

```
def load_model_for_inference(weight_path, num_classes, dropout=0.2):
    """加载训练好的模型用于推理"""
    # 1. 创建空模型（结构必须与训练时完全一致）
    model = BertClassifier(num_classes, dropout=dropout).to(DEVICE)
    
    # 2. 加载保存的权重
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    
    # 3. 切换到评估模式（关闭 Dropout）
    model.eval()
    return model


# 使用示例
trained_model = load_model_for_inference(
    weight_path="G:/.../bert_classifier_weights.pt",
    num_classes=17,
    dropout=0.2
)

# 用训练好的模型预测新文本
def predict(text, model, tokenizer):
    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        input_ids = enc['input_ids'].to(DEVICE)
        mask = enc['attention_mask'].to(DEVICE)
        logits = model(input_ids, mask)
        pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

# 测试
text = "想买一辆七座SUV，预算30万"
result = predict(text, trained_model, tokenizer)
print(f"预测类别：{result}")
```



------

四、方式二：保存完整模型（不推荐，但简单）

python

```
# 保存
torch.save(model, "full_model.pt")

# 加载（无需重建结构）
model = torch.load("full_model.pt")
```



**缺点**：文件更大，且加载时必须确保模型类的定义代码完全一致，跨项目移植困难。

------

五、方式三：保存为 Hugging Face 格式（进阶）

如果希望模型能被 `transformers` 库的 `from_pretrained` 直接加载：

python

```
# 保存
SAVE_DIR = os.path.join(DATA_DIR, "bert_classifier_saved")
os.makedirs(SAVE_DIR, exist_ok=True)

# 保存 BERT 主干
model.bert.save_pretrained(SAVE_DIR)
# 保存分类头
torch.save(model.classifier.state_dict(), os.path.join(SAVE_DIR, "classifier.pt"))
# 保存分词器
tokenizer.save_pretrained(SAVE_DIR)
```



这种方式适合发布或团队共享模型。

------

六、保存最佳权重（进阶技巧）

如果你希望**在每个 epoch 后自动保存验证集准确率最高的那个模型**，可以这样改造：

python

```
def run_experiment_with_best_save(lr, batch_size, dropout, save_path):
    train_loader, dev_loader, test_loader = build_loaders(batch_size)
    model = BertClassifier(num_classes, dropout=dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)

        # 若当前轮验证准确率更高，则保存
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Epoch {epoch}: New best dev_acc={dev_acc:.4f}, saved.")

    # 测试集使用最终模型（也可以重新加载 best 权重再测）
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Acc (last epoch): {test_acc:.4f}")

    # 推荐：加载最佳权重后再测一次
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Acc (best model): {test_acc:.4f}")

    return test_acc, model
```



这种方式能避免过拟合带来的性能损失，**强烈推荐**。

------

七、注意事项

1. **权重文件很大**：BERT-base 约 400MB，如果保存多份实验权重会占用大量磁盘空间，建议用完及时清理或只保留最佳权重。
2. **加载时模型结构必须一致**：`state_dict` 保存的只是参数，加载时 `BertClassifier` 的层数、隐藏维度、类别数必须与训练时完全相同。
3. **`map_location=DEVICE`**：在 CPU 上加载 GPU 训练的权重时需要这个参数，否则会报错。
4. **推理时务必 `model.eval()`**：关闭 Dropout 层，避免预测结果随机。
5. **使用 `torch.no_grad()`**：推理时不需要计算梯度，可以节省大量显存并加快速度。

------

八、推荐做法总结

| 场景             | 推荐方式                                          |
| :--------------- | :------------------------------------------------ |
| **日常保存**     | `torch.save(model.state_dict(), path)`            |
| **加载推理**     | 先构建模型 → `load_state_dict()` → `model.eval()` |
| **保存最佳模型** | 每轮比较 `dev_acc`，只保留最优权重                |
| **跨团队共享**   | `save_pretrained()` + 分类头单独保存              |

按方式一整合到你的项目后，训练结束会自动保存权重。之后无论做演示、部署还是继续微调，都可以直接加载，无需重新训练。如果需要我把整合后的完整 `main.py` 写出来，可以告诉我。
