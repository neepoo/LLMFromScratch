# 假设我们有四个输入TOKEN，ID分别是2,3,5,1
import torch

input_ids = [2, 3, 5, 1]
# 假设词汇表只有6个词(BPE有50257)，嵌入向量的维度是3(GPT-3是12288)
vocab_size = 6  # 词汇表大小
output_dim = 3  # 嵌入维度
# 初始化权重矩阵
torch.manual_seed(123)  # 为了复现结果
"""
嵌入层是一个可训练的权重矩阵，其大小为 vocab_size × output_dim。
vocab_size：词汇表中 token 的总数。
output_dim：每个 token 的嵌入向量维度。
这个矩阵的每一行表示词汇表中一个 token 的嵌入向量。

矩阵的第 0 行是 token ID 为 0 的嵌入向量，第 1 行是 token ID 为 1 的嵌入向量，

权重矩阵的初始权重是随机生成的，在训练过程中会被更新，以更好地表示数据的语义特征。通过设置随机种子可以确保结果的可复现性。
"""
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))

