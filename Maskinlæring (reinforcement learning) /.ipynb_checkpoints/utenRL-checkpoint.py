# Lager scriptet som kjører alle tre vindustyper separat og plottet sammen
script_code = """
# ============================================================
# Sammenligning av Sliding Window-strategier (ingen RL)
# Kjører Fixed, Variable og Weighted windows separat
# Trener hver strategi på samme datasett med egen modell
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# === Brukervalg av datasett ===
choice = "plateau"  # <-- kan endres til "gaussian" eller "sinusoidal"

file_map = {
    "plateau": "plateau_log.csv",
    "gaussian": "gaussian_curve.csv",
    "sinusoidal": "sinusoidal_log.csv"
}

filename = file_map[choice]
if not os.path.exists(filename):
    raise FileNotFoundError(f"Filen '{filename}' finnes ikke.")

# === Last og normaliser data ===
df = pd.read_csv(filename)
scaler = MinMaxScaler()
normalized = scaler.fit_transform(df[['Y']].values).astype(np.float32)

# === Sliding window-strategier ===
def fixed_window(data, idx, size=20):
    if idx < size: return data[0:idx+1]
    return data[idx-size+1:idx+1]

def variable_window(data, idx, min_size=10, max_size=30):
    size = random.randint(min_size, max_size)
    if idx < size: return data[0:idx+1]
    return data[idx-size+1:idx+1]

def weighted_window(data, idx, size=20):
    if idx < size: window = data[0:idx+1]
    else: window = data[idx-size+1:idx+1]
    weights = np.linspace(0.1, 1.0, len(window)).reshape(-1, 1)
    return window * weights

# === LSTM-modell ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# === Treningsfunksjon for en strategi ===
def train_strategy(strategy_fn, name, episodes=1):
    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    predictions, actuals = [], []

    for ep in range(episodes):
        for t in range(30, len(normalized)-1):
            window = strategy_fn(normalized, t)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            y_true = torch.tensor(normalized[t+1], dtype=torch.float32).unsqueeze(0)

            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions.append(y_pred.item())
            actuals.append(y_true.item())

    return predictions, actuals

# === Kjør alle tre strategier ===
pred_fixed, actual = train_strategy(fixed_window, "Fixed")
pred_variable, _ = train_strategy(variable_window, "Variable")
pred_weighted, _ = train_strategy(weighted_window, "Weighted")

# === Plot alle tre i samme graf ===
plt.figure(figsize=(14, 6))
plt.plot(actual[:300], label='Actual', linewidth=2)
plt.plot(pred_fixed[:300], label='Fixed Window')
plt.plot(pred_variable[:300], label='Variable Window')
plt.plot(pred_weighted[:300], label='Weighted Window')
plt.title(f"Sliding Window Comparison on {choice.capitalize()} Data")
plt.xlabel("Time Step")
plt.ylabel("Normalized CPU Utilization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# Lagre som fil
script_path = "/mnt/data/adaptive_window_comparison.py"
with open(script_path, "w") as f:
    f.write(script_code)

script_path
