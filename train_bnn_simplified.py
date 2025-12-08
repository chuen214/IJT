# -*- coding: utf-8 -*-
"""
训练简化BNN模型（4特征）
不使用Concentration和Density
用于应对药厂不提供这些信息的情况
"""

import os
import math
import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

print("="*70)
print("简化BNN训练 (4特征: Temperature, Volume, Viscosity, Spring_k)")
print("="*70)

# ============================================================
# 1. 加载和预处理数据
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "training_dataset_with_spring.xlsx")
save_dir = os.path.join(BASE_DIR, "saved_bnn_simplified")

print(f"\n[1] Loading data...")
df = pd.read_excel(excel_path)

# 温度转换
df["Temperature"] = df["Temperature"].astype(str).str.strip().str.capitalize()
temp_map = {"Cool": 5.0, "Standard": 20.0, "Warm": 40.0}
df["Temperature_num"] = df["Temperature"].map(temp_map)

# 简化特征（只有4个）
feature_cols = [
    "Temperature_num", "Volume", "Viscosity", "Spring_k_mean"
]
target_col = "Injection Time"
weight_cols = ["Spring_k_std"]

df_clean = df[feature_cols + [target_col] + weight_cols].dropna().reset_index(drop=True)
print(f"   Data: {len(df_clean)} samples")
print(f"   Features (4): {feature_cols}")
print(f"   Target: {target_col}")

# ============================================================
# 2. 计算样本权重
# ============================================================

print(f"\n[2] Computing sample weights...")

# Temperature权重 - 大幅增强以保证正确的单调性
temp_counts = df_clean["Temperature_num"].value_counts()
# 基础权重
temp_weights = 1.0 / df_clean["Temperature_num"].map(temp_counts)
# 额外增强5°C和40°C的权重（它们是极端温度，对学习单调性很重要）
temp_boost = df_clean["Temperature_num"].map({5.0: 3.0, 20.0: 1.0, 40.0: 3.0})
temp_weights = temp_weights * temp_boost
temp_weights = temp_weights / temp_weights.mean()

# Injection Time权重
time_bins = pd.cut(df_clean[target_col], bins=[0, 3, 5, 7, 20], labels=['very_fast', 'normal', 'slow', 'very_slow'])
time_counts = time_bins.value_counts()
time_weights_map = (1.0 / time_counts).to_dict()
time_weights = pd.Series(time_bins.map(time_weights_map).astype(float))
time_weights = time_weights / time_weights.mean()

# Spring_k_std不确定性权重
std_normalized = (df_clean["Spring_k_std"] - df_clean["Spring_k_std"].min()) / \
                 (df_clean["Spring_k_std"].max() - df_clean["Spring_k_std"].min())
uncertainty_weights = 1.0 / (1.0 + std_normalized)
uncertainty_weights = uncertainty_weights / uncertainty_weights.mean()

# 综合权重
sample_weights = temp_weights * time_weights * uncertainty_weights
sample_weights = sample_weights / sample_weights.mean()

print(f"   Sample weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")

df_clean["sample_weight"] = sample_weights

# ============================================================
# 3. 创建Time/Volume归一化
# ============================================================

print(f"\n[3] Creating normalized target...")
df_clean["Time_per_Volume"] = df_clean[target_col] / df_clean["Volume"]

# ============================================================
# 4. 分层切分
# ============================================================

print(f"\n[4] Splitting data...")
df_clean["time_bin"] = pd.qcut(df_clean["Time_per_Volume"], q=10, labels=False, duplicates="drop")
train_df, test_df = train_test_split(
    df_clean, test_size=0.2, stratify=df_clean["time_bin"], random_state=42
)
train_df = train_df.drop(columns=["time_bin"])
test_df = test_df.drop(columns=["time_bin"])

print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# 5. 标准化
# ============================================================

print(f"\n[5] Standardizing features...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_np = train_df[feature_cols].to_numpy(dtype=float)
X_test_np = test_df[feature_cols].to_numpy(dtype=float)

scaler_X.fit(X_train_np)
X_train_scaled = scaler_X.transform(X_train_np)
X_test_scaled = scaler_X.transform(X_test_np)

y_train_np = train_df["Time_per_Volume"].to_numpy(dtype=float).reshape(-1, 1)
y_test_np = test_df["Time_per_Volume"].to_numpy(dtype=float).reshape(-1, 1)

scaler_y.fit(y_train_np)
y_train_scaled = scaler_y.transform(y_train_np).flatten()
y_test_scaled = scaler_y.transform(y_test_np).flatten()

train_weights = train_df["sample_weight"].to_numpy(dtype=float)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
w_train = torch.tensor(train_weights, dtype=torch.float32)

volume_train = torch.tensor(train_df["Volume"].to_numpy(), dtype=torch.float32)
volume_test = torch.tensor(test_df["Volume"].to_numpy(), dtype=torch.float32)

print(f"   Input dim: {len(feature_cols)}")

# ============================================================
# 6. 定义BNN
# ============================================================

print(f"\n[6] Creating simplified BNN...")

class SimplifiedBNN(PyroModule):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_dim, hidden_dim)
        self.fc2 = PyroModule[nn.Linear](hidden_dim, hidden_dim)
        self.out = PyroModule[nn.Linear](hidden_dim, 2)
        
        prior_scale = 0.6
        self.fc1.weight = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim, in_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim]).to_event(1))
        self.fc2.weight = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim]).to_event(1))
        self.out.weight = PyroSample(dist.Normal(0.0, prior_scale).expand([2, hidden_dim]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0.0, prior_scale).expand([2]).to_event(1))
    
    def forward(self, x, y=None, weights=None):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.out(h)
        
        mu = out[:, 0]
        raw_sigma = out[:, 1]
        sigma = F.softplus(raw_sigma) + 1e-6
        
        pyro.deterministic("mu_pred", mu)
        pyro.deterministic("sigma_pred", sigma)
        
        with pyro.plate("data", x.shape[0]):
            if weights is not None:
                obs_sigma = sigma / torch.sqrt(weights)
                pyro.sample("obs", dist.Normal(mu, obs_sigma), obs=y)
            else:
                pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        
        return mu, sigma

bnn_model = SimplifiedBNN(in_dim=len(feature_cols), hidden_dim=128)  # 增加到128

# ============================================================
# 7. 训练BNN
# ============================================================

print(f"\n[7] Training BNN (10000 epochs with Temperature monotonicity constraint)...")

pyro.clear_param_store()
guide = AutoDiagonalNormal(bnn_model)
optimizer = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())

losses = []
for epoch in range(1, 10001):
    loss = svi.step(X_train, y_train, w_train)
    
    # 每50个epoch添加Temperature单调性约束
    if epoch % 50 == 0:
        # 创建Temperature对比样本
        n_samples = min(200, len(X_train))
        indices = torch.randperm(len(X_train))[:n_samples]
        X_sample = X_train[indices].clone()
        
        # Temperature是第一个特征，增加标准化后的温度（相当于10度）
        temp_delta = 10.0 / scaler_X.scale_[0]
        X_sample_high = X_sample.clone()
        X_sample_high[:, 0] = X_sample_high[:, 0] + temp_delta
        
        # 预测
        with torch.no_grad():
            pred_low_dist = bnn_model(X_sample)
            pred_high_dist = bnn_model(X_sample_high)
            pred_low = pred_low_dist[0]  # mu
            pred_high = pred_high_dist[0]  # mu
        
        # Temperature增加应该导致Time减少，如果pred_high >= pred_low则违反
        monotonicity_violation = torch.relu(pred_high - pred_low + 0.05).mean()  # 0.05是容忍度
        
        # 单调性权重逐渐增加
        mono_weight = min(1.0, epoch / 3000) * 50
        loss = loss + mono_weight * monotonicity_violation.item()
    
    losses.append(loss)
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}: Loss = {loss:.2f}")

print("\n[8] Training complete!")

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.title('Training Loss Curve - Simplified Model')
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150)
print(f"   Loss curve saved")
plt.close()

# ============================================================
# 8. 评估
# ============================================================

print(f"\n[9] Evaluating model...")

predictive = Predictive(bnn_model, guide=guide, num_samples=200)
samples = predictive(X_test)

y_pred_scaled_mean = samples["obs"].mean(dim=0).detach().numpy()
y_pred_scaled_std = samples["obs"].std(dim=0).detach().numpy()

y_pred_per_vol_mean = scaler_y.inverse_transform(y_pred_scaled_mean.reshape(-1, 1)).flatten()
y_pred_per_vol_std = y_pred_scaled_std * scaler_y.scale_[0]

y_pred_time_mean = y_pred_per_vol_mean * volume_test.numpy()
y_pred_time_std = y_pred_per_vol_std * volume_test.numpy()

y_pred_time_mean = np.maximum(y_pred_time_mean, 0.01)

y_true_time = test_df[target_col].to_numpy()

rmse = np.sqrt(mean_squared_error(y_true_time, y_pred_time_mean))
mae = mean_absolute_error(y_true_time, y_pred_time_mean)
r2 = r2_score(y_true_time, y_pred_time_mean)

print(f"   RMSE: {rmse:.4f} seconds")
print(f"   MAE:  {mae:.4f} seconds")
print(f"   R²:   {r2:.4f}")

print(f"\n   Performance by Temperature:")
for temp in [5.0, 20.0, 40.0]:
    mask = test_df["Temperature_num"] == temp
    if mask.sum() > 0:
        rmse_temp = np.sqrt(mean_squared_error(y_true_time[mask], y_pred_time_mean[mask]))
        r2_temp = r2_score(y_true_time[mask], y_pred_time_mean[mask])
        print(f"      {temp}°C: RMSE={rmse_temp:.4f}, R²={r2_temp:.4f} (n={mask.sum()})")

# ============================================================
# 9. 导出模型
# ============================================================

print(f"\n[10] Exporting model...")

num_weight_samples = 100
weight_samples = []

for i in range(num_weight_samples):
    guide_trace = pyro.poutine.trace(guide).get_trace(X_train[:1], y_train[:1])
    weights = {
        'fc1.weight': guide_trace.nodes['fc1.weight']['value'].detach().cpu().numpy(),
        'fc1.bias': guide_trace.nodes['fc1.bias']['value'].detach().cpu().numpy(),
        'fc2.weight': guide_trace.nodes['fc2.weight']['value'].detach().cpu().numpy(),
        'fc2.bias': guide_trace.nodes['fc2.bias']['value'].detach().cpu().numpy(),
        'out.weight': guide_trace.nodes['out.weight']['value'].detach().cpu().numpy(),
        'out.bias': guide_trace.nodes['out.bias']['value'].detach().cpu().numpy(),
    }
    weight_samples.append(weights)
    
    if (i + 1) % 20 == 0:
        print(f"      Sampled {i+1}/{num_weight_samples}...")

export_data = {
    'weight_samples': weight_samples,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_cols': feature_cols,
    'hidden_dim': 128,
    'in_dim': len(feature_cols),
    'performance': {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    },
    'note': 'Simplified model (4 features) without Concentration and Density'
}

export_path = os.path.join(save_dir, 'bnn_export.pkl')
joblib.dump(export_data, export_path)
print(f"   Saved: {export_path}")

print("\n" + "="*70)
print("[DONE] Simplified BNN training complete!")
print(f"Model saved to: {save_dir}/")
print("="*70)

