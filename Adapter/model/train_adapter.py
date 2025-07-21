import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from bert_adapter_model import build_bert_with_adapter


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
    writer = SummaryWriter(log_dir="./runs/adapter_experiment")

    classifier = nn.Linear(config["hidden_size"], 1).to(device)
    criterion = nn.BCEWithLogitsLoss()

    dataset = config["dataset"]

    inputs1 = tokenizer(
        [x["sent1"] for x in dataset],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_length"],
    ).to(device)

    inputs2 = tokenizer(
        [x["sent2"] for x in dataset],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_length"],
    ).to(device)

    labels = torch.tensor([x["label"] for x in dataset], dtype=torch.float).unsqueeze(1).to(device)

    best_f1 = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        classifier.train()

        cls_emb1 = model(**inputs1).last_hidden_state[:, 0, :]
        cls_emb2 = model(**inputs2).last_hidden_state[:, 0, :]

        features = torch.abs(cls_emb1 - cls_emb2)

        logits = classifier(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        true = labels.detach().cpu().numpy()

        f1 = f1_score(true, preds)
        acc = accuracy_score(true, preds)

        print(f"Epoch {epoch + 1}/{config['num_epochs']} - Loss: {loss.item():.4f} - F1: {f1:.4f} - Acc: {acc:.4f}")
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("F1/train", f1, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(adapter.state_dict(), f"adapter_best_epoch{epoch + 1}.pth")
            print(f"Best model saved at epoch {epoch + 1} (F1: {f1:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
