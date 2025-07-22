import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from bert_adapter_model import build_bert_with_adapter
from datasets import load_dataset


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

    dataset = load_dataset("glue", "sst2")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    train_inputs = tokenizer(
        train_data["sentence"],
        padding=True,
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt"
    ).to(device)
    train_labels = torch.tensor(train_data["label"], dtype=torch.float).unsqueeze(1).to(device)
    best_f1 = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        classifier.train()

        cls_emb = model(**train_inputs).last_hidden_state[:, 0, :]
        logits = classifier(cls_emb)
        loss = criterion(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        true = train_labels.detach().cpu().numpy()

        f1 = f1_score(true, preds)
        acc = accuracy_score(true, preds)

        print(f"Epoch {epoch + 1}/{config['num_epochs']} - Loss: {loss.item():.4f} - F1: {f1:.4f} - Acc: {acc:.4f}")
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("F1/train", f1, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)

        model.eval()
        classifier.eval()

        with torch.no_grad():
            val_inputs = tokenizer(
                val_data["sentence"],
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt"
            ).to(device)

            val_labels = torch.tensor(val_data["label"], dtype=torch.float).unsqueeze(1).to(device)

            val_cls_emb = model(**val_inputs).last_hidden_state[:, 0, :]
            val_logits = classifier(val_cls_emb)

            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
            val_true = val_labels.cpu().numpy()

            val_f1 = f1_score(val_true, val_preds)
            val_acc = accuracy_score(val_true, val_preds)

            print(f"→ Validation - F1: {val_f1:.4f} - Acc: {val_acc:.4f}")
            writer.add_scalar("F1/val", val_f1, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(adapter.state_dict(), f"adapter_best_epoch{epoch + 1}.pth")
                print(f"Best model saved at epoch {epoch + 1} (val_F1: {val_f1:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
