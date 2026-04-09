import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report

import config


# ── Track metrics during training ─────────────────────────────────────────────

class History:
    """Stores loss and accuracy for each epoch (train + val)."""

    def __init__(self):
        self.train_loss, self.train_acc = [], []
        self.val_loss,   self.val_acc   = [], []

    def record(self, train_loss, train_acc, val_loss, val_acc):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)


# ── Evaluate model on a DataLoader ────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    """
    Runs the model on all batches in loader without updating weights.
    Returns: loss, accuracy, predicted labels, true labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():   # no gradient needed during evaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)     # class with the highest score

            total_loss += loss.item() * labels.size(0)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ── Save all results for a domain ─────────────────────────────────────────────

def save_results(domain, history, test_preds, test_labels):
    """
    Saves to results/<domain>/:
      - loss_curves.png     — train/val loss and accuracy over epochs
      - confusion_matrix.png
      - metrics.json        — final accuracy numbers
    """
    out_dir = os.path.join(config.RESULTS_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)

    _plot_curves(history, domain, out_dir)
    _plot_confusion_matrix(test_preds, test_labels, domain, out_dir)
    _save_metrics(history, test_preds, test_labels, domain, out_dir)

    print(f"Results saved to {out_dir}")


def _plot_curves(history, domain, out_dir):
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history.train_loss, label="Train")
    ax1.plot(epochs, history.val_loss,   label="Val")
    ax1.set_title(f"{domain} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history.train_acc, label="Train")
    ax2.plot(epochs, history.val_acc,   label="Val")
    ax2.set_title(f"{domain} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(preds, labels, domain, out_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CIFAR10_CLASSES,
        yticklabels=config.CIFAR10_CLASSES,
        ax=ax,
    )
    ax.set_title(f"{domain} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _save_metrics(history, preds, labels, domain, out_dir):
    report = classification_report(
        labels, preds,
        target_names=config.CIFAR10_CLASSES,
        output_dict=True,
    )
    metrics = {
        "domain":        domain,
        "best_val_acc":  round(max(history.val_acc), 4),
        "final_val_acc": round(history.val_acc[-1], 4),
        "test_acc":      round(float((preds == labels).mean()), 4),
        "per_class_f1":  {cls: round(report[cls]["f1-score"], 4) for cls in config.CIFAR10_CLASSES},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test accuracy: {metrics['test_acc']:.4f}")
