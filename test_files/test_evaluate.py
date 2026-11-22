
import os
import sys

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Allow Python to import evaluate.py from the parent Team-10 folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import compute_metrics, always_up_baseline, evaluate


# ---------------------------------
# 1. Check the math on a tiny hand-made example
# ---------------------------------
labels = [1, 1, 1, 0, 0]  # 3 UP, 2 DOWN
preds = [1, 1, 0, 1, 0]   # 2 right UP, 1 missed UP, 1 wrong UP, 1 right DOWN
m = compute_metrics(labels, preds)
assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 1, 1, 1)
assert abs(m["accuracy"] - 3 / 5) < 1e-9
assert abs(m["precision"] - 2 / 3) < 1e-9
assert abs(m["recall"] - 2 / 3) < 1e-9

# The always-UP baseline should score exactly the % of UP labels
base = compute_metrics(labels, always_up_baseline(labels))
assert abs(base["accuracy"] - 3 / 5) < 1e-9
assert base["recall"] == 1.0 and base["down_recall"] == 0.0
print("Math checks passed!")


# ---------------------------------
# 2. Temporary CNN model (same as in test_train.py)
# ---------------------------------
class StockCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 64 * 64, 2)

    def forward(self, x):
        return self.fc(self.flatten(x))


# ---------------------------------
# 3. Fake test set + run the full evaluation
# ---------------------------------
torch.manual_seed(0)
images = torch.randn(20, 3, 64, 64)  # 20 fake RGB images
labels = torch.randint(0, 2, (20,))  # 0 = DOWN, 1 = UP
test_loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)

model = StockCNN()
weights = "test_files/test_model.pth"  # saved by test_train.py, if you ran it
if os.path.exists(weights):
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    print("Loaded", weights)

cnn, base = evaluate(model, test_loader, plot_path="test_files/test_confusion_matrix.png")
assert cnn["n"] == 20
assert os.path.exists("test_files/test_confusion_matrix.png")

print("\nEvaluation test successful!")