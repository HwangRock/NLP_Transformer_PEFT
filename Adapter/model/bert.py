from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

model.eval()

# 예시 문장. 나중에 dataset으로 바꾸면 됨
sentences = [
    "Hello, this is a test sentence.",
    "Another example sentence for testing.",
]

inputs = tokenizer(
    sentences,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state

print("Embeddings shape:", embeddings.shape)
