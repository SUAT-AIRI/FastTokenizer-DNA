## 引言

传统的Tokenizer方法如BPE（Byte-Pair Encoding）和Unigram LM主要通过频率统计来从底层单元（字符或子词）构建子词单元。在新的长序列建模需求（如DNA、蛋白质）以及多模态任务中，存在一种新的构想：

> 是否可以在整个语料中，**优先识别最长且高频的子序列（如n-gram或DNA motif）作为token**，然后再处理低频、短小或无序的剩余子序列？

本报告将调研该问题的已有相关工作、理论可行性、与现有方法的差异，以及是否存在开源实现。

---

## 一、BPE与Unigram Tokenizer简介

### BPE (Byte-Pair Encoding)

* **思想**：从字符级开始，统计所有相邻token对的频率，选择频率最高的一对合并，构建新的token。
* **优点**：自下而上构建常用子词，词表效率高。
* **缺点**：局部贪心，长词或不可切割短语可能被误分。

参考文献：Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016) \[\[1]]

### Unigram Language Model

* **思想**：初始化一个大型词表，通过EM算法选择保留高概率的子词单位。
* **优点**：可随机切分，有词表概率。
* **缺点**：收敛过程较慢。

参考文献：Kudo, Taku. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" (2018) \[\[2]]

---

## 二、“自长而短”的Tokenizer：理论基础与可行性

### 主要思路

1. **全语料扫描**：提取最长、频率高的字符串（如频率 > 某阈值，长度 > n）。
2. **优先切分**：将其视为atomic token，作为词表的起点。
3. **残差处理**：剩余字符串再做常规BPE或Unigram处理。

### 与现有方法的区别

| 方法         | 切分方向 | 切分单位起点 | 策略              |
| ---------- | ---- | ------ | --------------- |
| BPE        | 自下而上 | 字符     | 合并频率最高的字符对      |
| Unigram LM | 自上而下 | 大词表    | 剪枝低概率token      |
| 本策略（提议）    | 自长而短 | 高频长字符串 | 优先锁定长高频串，保留结构语义 |

### 优势

* 结构感知：保留domain-specific结构（如DNA motif、成语、常用表达）
* token压缩比高：长token减少总token数
* 与任务更契合（如编码区域识别）

### 风险或难点

* 不可泛化的长串：罕见句法结构可能不再出现
* token查找困难：需全语料构建前缀树或hash-map

---

## 三、相关工作与算法实现

### 1. **SENTENCEPIECE** 中的UnigramTokenizer

虽然不是完全自长而短，但Unigram LM有一定自上而下精神：保留高概率token、丢弃低效token。

开源地址：[https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

### 2. **YaMLM**: Yet Another Multi-granularity LM (ACL 2023)

* 提出多粒度token训练机制，将长词和短词统一建模。
* 训练时使用多级token粒度预测loss进行联合训练。

论文链接：[https://aclanthology.org/2023.acl-long.420/](https://aclanthology.org/2023.acl-long.420/)

### 3. **Morfessor**: 形态学启发的模型

* 虽不完全自长而短，但使用最大熵/MDL原则提取词干和构词规则。
* 对自然语言、DNA序列均有适配版本。

开源地址：[https://github.com/aalto-speech/morfessor](https://github.com/aalto-speech/morfessor)

### 4. **Prefix Trie-Based Compression**

* 以前缀树表示所有高频子串，进行最大匹配分词。
* 结合LZ78和RE-Pair压缩技术。

已用于 DNA compression: see \[CDNA tool] \[\[3]]

---

## 四、哈夫曼编码是否可用于Tokenizer？

### 理论分析

* Huffman编码是为**最小化平均编码长度**而设计的，与BPE有相似目标（压缩 token 个数）。
* 然而 Huffman 编码的“叶节点是符号”假设不适用于子串分词，因为：

  * 子串可能重叠
  * Huffman构建的是唯一前缀码树，不适合表达多个语义模糊的token

### 举例

若“ABCDEF”中“ABC”、“CDE”都高频，则两者不能共存于Huffman树中（编码冲突）。

因此：**Huffman适用于编码阶段，不适合token切分阶段。**

---

## 五、关于“Closed Optimal”与频率优先的比较

### 频率优先策略

* 贪心合并高频子串（BPE、RE-Pair）
* 局部最优，但无法保证全局最优token划分

### Closed Optimal Strategy

* 定义最优的切分覆盖，使得整体loss最小
* 需要动态规划/EM等全局算法
* 如：Unigram LM采用EM估计token概率，近似达到最优编码效率

### 总结比较

| 策略类型   | 优势            | 缺点              |
| ------ | ------------- | --------------- |
| 高频优先策略 | 简单高效，易于部署     | 容易局部最优，结构不稳     |
| 全局最优策略 | 编码质量高，支持多切分路径 | 训练复杂、需估计token分布 |

---

## 六、结论与未来方向

* 自长而短的Tokenizer方法尚未作为主流技术被系统化提出，但其思想**在多个工作中已有体现**。
* 对于结构化序列（DNA、蛋白质、代码、拼音）或语义固定表达（中文成语、金融术语）而言，**优先识别长token可显著降低token数量，提高模型效率**。
* 构建这类Tokenizer，可使用**最大匹配算法 + Trie 树**，或参考Morfessor等非监督形态学分析方法。
* 若用于 LLM，可进一步采用 soft tokenization 接入 transformer 输入层，或设计混合token embedding机制。

---

## 参考文献

\[1] Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Neural machine translation of rare words with subword units." ACL 2016.

\[2] Kudo, Taku. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." ACL 2018.

\[3] Matsumoto, T., Sadakane, K., & Imai, H. (2000). "Biological sequence compression algorithms." Genome Informatics.

\[4] Virpioja et al. "Morfessor 2.0: Python implementation and extensions for Morfessor Baseline." Aalto University.

---
