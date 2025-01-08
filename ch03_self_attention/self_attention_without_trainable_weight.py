# context vectors
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.64],  # starts (x^3)
     [0.22, 0.58, 0.33],  # with (x^4)
     [0.77, 0.25, 0.10],  # one (x^5)
     [0.05, 0.80, 0.55]]  # step (x^6)
)

# 第一步骤是计算中间值w，-> attention scores（computing the  attention scores  between the query x(2) and all other input elements as a dot product.）
query = inputs[1]  # 第二个输入token作为query
shape0 = inputs.shape[0]
attn_scores_2 = torch.empty(shape0)
for i, x_i in enumerate(inputs):
    print(f"{x_i} dot {query} = {torch.dot(x_i, query)}")
    # We determine these scores by computing the dot
    # product of the query, x(2), with every other input token
    # 点积算出来是一个数
    # 点积值越高，两个元素之间的相似度和注意力得分越高。
    # 几何意义
    #   点积等于两个向量的模长乘以它们之间夹角的余弦值
    # a⋅b =∥a∥∥b∥cosθ
    # 点积为负，表示两个向量方向相反。
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# 归一化，目的是保证注意力得分的总和为1
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# 通过softmax函数，将注意力得分转换为注意力权重
# 将注意力分数转化为概率分布，突出关键元素，压缩次要元素。
# 提供数值稳定性和良好的梯度特性，有助于训练收敛。

# 能保证weight > 0
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights (naive softmax):", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# 使用PyTorch的内置函数
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights (PyTorch softmax):", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 计算context vector
"""
通过将嵌入的输入 token 𝑥(𝑖)  与对应的注意力权重相乘，然后对所得的向量求和，计算上下文向量𝑧(2)。
因此，上下文向量𝑧(2)是所有输入向量的加权和，具体方法是将每个输入向量与其对应的注意力权重相乘。
"""

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("Context vector:", context_vec_2)

## ------------------- 计算所有的context vector -------------------

## step1: 计算attention score
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
# matrix multiplication
# attn_scores = inputs @ inputs.T

## step2: 计算attention weights(归一化attention score)
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

## step3: 计算context vector
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)