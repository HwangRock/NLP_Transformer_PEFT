import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from bert_adapter_model import build_bert_with_adapter
from datasets import load_dataset
from Adapter.utils.SST2Dataset import SST2Dataset


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

    train_encodings = tokenizer(train_data["sentence"], padding=True, truncation=True,
                                max_length=config["max_length"], return_tensors="pt")
    val_encodings = tokenizer(val_data["sentence"], padding=True, truncation=True,
                              max_length=config["max_length"], return_tensors="pt")

    train_labels = torch.tensor(train_data["label"], dtype=torch.float)
    val_labels = torch.tensor(val_data["label"], dtype=torch.float)

    train_dataset = SST2Dataset(train_encodings, train_labels)
    val_dataset = SST2Dataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    best_f1 = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        classifier.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            logits = classifier(cls_emb)
            loss = criterion(logits, batch["labels"].unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch["labels"].cpu().numpy().reshape(-1, 1)
            all_preds.extend(preds)
            all_labels.extend(labels)

        train_f1 = f1_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} - Loss: {avg_loss:.4f} - F1: {train_f1:.4f} - Acc: {train_acc:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        model.eval()
        classifier.eval()
        val_preds = []
        val_labels_all = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                logits = classifier(cls_emb)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels = batch["labels"].cpu().numpy().reshape(-1, 1)

                val_preds.extend(preds)
                val_labels_all.extend(labels)

        val_f1 = f1_score(val_labels_all, val_preds)
        val_acc = accuracy_score(val_labels_all, val_preds)

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
