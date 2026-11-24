import torch
import matplotlib
matplotlib.use("Agg")  # save the chart to a file, never open a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

DOWN, UP = 0, 1


# 1. Run the model over a loader and collect every true label and every guess
def get_predictions(model, loader):
    model.eval()  # test mode: no learning
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():  # skip the nudge math, faster
        for images, labels in loader:
            scores = model(images)  # 2 scores per picture: (DOWN, UP)
            preds = scores.argmax(dim=1)  # pick whichever score is bigger
            probs = torch.softmax(scores, dim=1)[:, UP]  # how sure the model is that it's UP (0 to 1)
            all_labels += labels.tolist()
            all_preds += preds.tolist()
            all_probs += probs.tolist()
    return all_labels, all_preds, all_probs


# 2. The baseline: ignore the picture, always guess UP
def always_up_baseline(labels):
    # This "dumb" guess scores exactly the % of UP images in the test set.
    # The CNN is only useful if it beats this number.
    return [UP] * len(labels)


# 3. Turn labels + guesses into scores
def safe_divide(a, b):
    return a / b if b > 0 else 0.0  # avoid crashing when a count is 0


def compute_metrics(labels, preds):
    # Confusion matrix counts ("positive" = UP)
    tp = sum(1 for y, p in zip(labels, preds) if y == UP and p == UP)      # said UP, was UP
    tn = sum(1 for y, p in zip(labels, preds) if y == DOWN and p == DOWN)  # said DOWN, was DOWN
    fp = sum(1 for y, p in zip(labels, preds) if y == DOWN and p == UP)    # said UP, was DOWN
    fn = sum(1 for y, p in zip(labels, preds) if y == UP and p == DOWN)    # said DOWN, was UP

    precision = safe_divide(tp, tp + fp)  # when it says UP, how often is it right?
    recall = safe_divide(tp, tp + fn)     # of all real UPs, how many did it catch?
    return {
        "accuracy": safe_divide(tp + tn, len(labels)),  # % of all guesses that were right
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),  # mix of precision + recall
        "down_recall": safe_divide(tn, tn + fp),  # of all real DOWNs, how many did it catch?
        "pct_guessed_up": safe_divide(tp + fp, len(labels)),  # if ~100%, the model just copies the baseline
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": len(labels),
    }


# 4. Print the CNN and the baseline side by side
def print_report(cnn, base):
    rows = [
        ("Accuracy", "accuracy"),
        ("Precision (UP)", "precision"),
        ("Recall (UP)", "recall"),
        ("F1 (UP)", "f1"),
        ("Recall (DOWN)", "down_recall"),
        ("% guessed UP", "pct_guessed_up"),
        ("ROC-AUC", "roc_auc"),
    ]
    print("\n" + "=" * 46)
    print(f"TEST RESULTS  ({cnn['n']} test images)")
    print("=" * 46)
    print(f"{'Metric':<18}{'CNN':>12}{'Always-UP':>14}")
    print("-" * 46)
    for name, key in rows:
        print(f"{name:<18}{cnn[key]:>11.1%}{base[key]:>13.1%}")
    print("-" * 46)

    print("\nCNN confusion matrix (rows = real, columns = guessed):")
    print(f"{'':<12}{'guess DOWN':>12}{'guess UP':>10}")
    print(f"{'real DOWN':<12}{cnn['tn']:>12}{cnn['fp']:>10}")
    print(f"{'real UP':<12}{cnn['fn']:>12}{cnn['tp']:>10}")

    gap = cnn["accuracy"] - base["accuracy"]
    print(f"\nCNN vs always-UP: {gap * 100:+.1f} percentage points of accuracy")
    if gap > 0:
        print("The CNN beats the baseline.")
    else:
        print("The CNN does NOT beat the baseline (it is no better than always guessing UP).")
    if cnn["pct_guessed_up"] > 0.95:
        print("Warning: the CNN guesses UP almost every time, so it is basically copying the baseline.")


# 5. Save a confusion matrix picture for the slides
def plot_confusion_matrix(m, save_path="confusion_matrix.png", title="CNN on the test set"):
    grid = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(grid, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 14},
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], ax=ax)  # Seaborn heatmap
    ax.set_xlabel("Guessed")
    ax.set_ylabel("Real")
    ax.set_title(f"{title}  (accuracy {m['accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# 6. Do everything: predict, score, compare, print, save
def evaluate(model, test_loader, plot_path="confusion_matrix.png"):
    labels, preds, probs = get_predictions(model, test_loader)
    if not labels:
        raise ValueError("The test loader is empty.")
    cnn = compute_metrics(labels, preds)
    base = compute_metrics(labels, always_up_baseline(labels))
    # ROC-AUC: 0.5 = coin flip, 1.0 = perfect. Needs both UP and DOWN in the test set.
    cnn["roc_auc"] = roc_auc_score(labels, probs) if len(set(labels)) == 2 else 0.5
    base["roc_auc"] = 0.5  # always-UP gives every image the same score, so it can't rank them
    print_report(cnn, base)
    if plot_path:
        plot_confusion_matrix(cnn, plot_path)
        print(f"\nSaved {plot_path}")SS
    return cnn, base


if __name__ == "__main__":
    from CNN_build import StockCNN  # Arnav
    from preprocessing_skeleton import get_dataloaders  # Dale (same imports as train.py)

    _, _, test_loader = get_dataloaders()  # only the test loader; train/val belong to train.py

    model = StockCNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))  # weights saved by train.py
    evaluate(model, test_loader)