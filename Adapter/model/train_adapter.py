import json
import torch
import torch.nn.functional as F
from bert_adapter_model import build_bert_with_adapter
from torch.utils.tensorboard import SummaryWriter


def main():
    config_path = r"./../config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model, adapter = build_bert_with_adapter(config)

    model.to(device)
    adapter.to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
    sentences = config["sentences"]  # 나중에 dataset 추가하고 변경해야함!!!!

    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_length"],
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    writer = SummaryWriter(log_dir="./runs/adapter_experiment")

    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        model.train()

        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

        cls_emb1 = embeddings[0, 0, :]
        cls_emb2 = embeddings[1, 0, :]

        cos_sim = F.cosine_similarity(cls_emb1.unsqueeze(0), cls_emb2.unsqueeze(0))
        loss = 1 - cos_sim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{config['num_epochs']} - Loss: {loss.item():.4f}")
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("CosineSimilarity/train", cos_sim.item(), epoch)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(adapter.state_dict(), f"adapter_best_epoch{epoch + 1}.pth")
            print(f"New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
