import torch

from data_loader import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# initial positional embedding
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
# 正如我们所见，TokenID张量是8 × 4维的，这意味着数据批次包含八个文本样本，每个样本有四个令牌。
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
# Inputs shape:
#  torch.Size([8, 4])
# torch.Size([8, 4, 256])
print(token_embeddings.shape)

# GPT model的绝对嵌入编码方法，我们需要另外一个嵌入层
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embedding = pos_embedding_layer(torch.arange(context_length))
print(pos_embedding.shape)
input_embeddings = token_embeddings + pos_embedding
# torch.Size([8, 4, 256])
print(input_embeddings.shape)