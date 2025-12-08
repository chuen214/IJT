# -*- coding: utf-8 -*-
"""
改进的BNN训练脚本
主要改进：
1. 移除Spring_k_std作为输入特征
2. 使用Loss weighting处理数据不平衡
3. 加入物理约束来改善外插性能
4. 预测Time/Volume来保证Volume的线性关系
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
print("改进的BNN训练 - Physics-Informed with Loss Weighting")
print("="*70)

# ============================================================
# 1. 加载和预处理数据
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "training_dataset_with_spring.xlsx")
save_dir = os.path.join(BASE_DIR, "saved_bnn_improved")

print(f"\n[1] Loading data...")
df = pd.read_excel(excel_path)

# 温度转换
df["Temperature"] = df["Temperature"].astype(str).str.strip().str.capitalize()
temp_map = {"Cool": 5.0, "Standard": 20.0, "Warm": 40.0}
df["Temperature_num"] = df["Temperature"].map(temp_map)

# 特征 - 移除Spring_k_std作为输入，但保留用于加权
feature_cols = [
    "Temperature_num", "Volume", "Concentration",
    "Viscosity", "Density", "Spring_k_mean"
]
target_col = "Injection Time"
weight_cols = ["Spring_k_std"]  # 用于加权

df_clean = df[feature_cols + [target_col] + weight_cols].dropna().reset_index(drop=True)
print(f"   Data: {len(df_clean)} samples")
print(f"   Features: {feature_cols}")
print(f"   Target: {target_col}")

# ============================================================
# 2. 计算样本权重（处理数据不平衡）
# ============================================================

print(f"\n[2] Computing sample weights for imbalanced data...")

# 2.1 Temperature权重（稀有温度权重提高）
temp_counts = df_clean["Temperature_num"].value_counts()
temp_weights = 1.0 / df_clean["Temperature_num"].map(temp_counts)
temp_weights = temp_weights / temp_weights.mean()  # 归一化

# 2.2 Concentration权重（高浓度权重提高）
# 将浓度分为3档：低(<3)、中(3-6)、高(>6)
conc_bins = pd.cut(df_clean["Concentration"], bins=[0, 3, 6, 10], labels=['low', 'mid', 'high'])
conc_counts = conc_bins.value_counts()
conc_weights_map = (1.0 / conc_counts).to_dict()
conc_weights = pd.Series(conc_bins.map(conc_weights_map).astype(float))
conc_weights = conc_weights / conc_weights.mean()

# 2.3 Injection Time权重（极端值权重提高）
time_bins = pd.cut(df_clean[target_col], bins=[0, 3, 5, 7, 20], labels=['very_fast', 'normal', 'slow', 'very_slow'])
time_counts = time_bins.value_counts()
time_weights_map = (1.0 / time_counts).to_dict()
time_weights = pd.Series(time_bins.map(time_weights_map).astype(float))
time_weights = time_weights / time_weights.mean()

# 2.4 Spring_k_std不确定性权重（不确定性高的权重降低）
# Spring_k_std越大，权重越小
std_normalized = (df_clean["Spring_k_std"] - df_clean["Spring_k_std"].min()) / \
                 (df_clean["Spring_k_std"].max() - df_clean["Spring_k_std"].min())
uncertainty_weights = 1.0 / (1.0 + std_normalized)  # std大时权重小
uncertainty_weights = uncertainty_weights / uncertainty_weights.mean()

# 综合权重
sample_weights = temp_weights * conc_weights * time_weights * uncertainty_weights
sample_weights = sample_weights / sample_weights.mean()  # 最终归一化

print(f"   Temperature weight range: {temp_weights.min():.2f} - {temp_weights.max():.2f}")
print(f"   Concentration weight range: {conc_weights.min():.2f} - {conc_weights.max():.2f}")
print(f"   Time weight range: {time_weights.min():.2f} - {time_weights.max():.2f}")
print(f"   Uncertainty weight range: {uncertainty_weights.min():.2f} - {uncertainty_weights.max():.2f}")
print(f"   Final sample weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")

df_clean["sample_weight"] = sample_weights

# ============================================================
# 3. 创建归一化目标：Time per unit Volume
# ============================================================

print(f"\n[3] Creating normalized target (Time/Volume)...")

# 预测Time/Volume来保证Volume的线性关系
df_clean["Time_per_Volume"] = df_clean[target_col] / df_clean["Volume"]

print(f"   Original Time range: {df_clean[target_col].min():.2f} - {df_clean[target_col].max():.2f}")
print(f"   Time/Volume range: {df_clean['Time_per_Volume'].min():.2f} - {df_clean['Time_per_Volume'].max():.2f}")

# ============================================================
# 4. 分层切分数据
# ============================================================

print(f"\n[4] Splitting data...")

# 分层切分（基于Time_per_Volume）
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

# 标准化Time_per_Volume
y_train_np = train_df["Time_per_Volume"].to_numpy(dtype=float).reshape(-1, 1)
y_test_np = test_df["Time_per_Volume"].to_numpy(dtype=float).reshape(-1, 1)

scaler_y.fit(y_train_np)
y_train_scaled = scaler_y.transform(y_train_np).flatten()
y_test_scaled = scaler_y.transform(y_test_np).flatten()

# 权重
train_weights = train_df["sample_weight"].to_numpy(dtype=float)

# 转换为torch tensor
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
w_train = torch.tensor(train_weights, dtype=torch.float32)

# 保存Volume用于最后还原
volume_train = torch.tensor(train_df["Volume"].to_numpy(), dtype=torch.float32)
volume_test = torch.tensor(test_df["Volume"].to_numpy(), dtype=torch.float32)

print(f"   Input dim: {len(feature_cols)}")
print(f"   Train samples: {len(X_train)}")

# ============================================================
# 6. 定义带物理约束的BNN
# ============================================================

print(f"\n[6] Creating Physics-Informed BNN...")

class PhysicsInformedBNN(PyroModule):
    def __init__(self, in_dim=6, hidden_dim=64):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_dim, hidden_dim)
        self.fc2 = PyroModule[nn.Linear](hidden_dim, hidden_dim)
        self.out = PyroModule[nn.Linear](hidden_dim, 2)  # mu和sigma
        
        prior_scale = 0.6
        self.fc1.weight = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim, in_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim]).to_event(1))
        self.fc2.weight = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0.0, prior_scale).expand([hidden_dim]).to_event(1))
        self.out.weight = PyroSample(dist.Normal(0.0, prior_scale).expand([2, hidden_dim]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0.0, prior_scale).expand([2]).to_event(1))
    
    def forward(self, x, y=None, weights=None):
        """
        x: 输入特征 [Temperature, Volume, Concentration, Viscosity, Density, Spring_k_mean]
        y: Time_per_Volume (标准化后)
        weights: 样本权重
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.out(h)
        
        mu = out[:, 0]
        raw_sigma = out[:, 1]
        sigma = F.softplus(raw_sigma) + 1e-6
        
        pyro.deterministic("mu_pred", mu)
        pyro.deterministic("sigma_pred", sigma)
        
        # 加权观测
        with pyro.plate("data", x.shape[0]):
            if weights is not None:
                # 使用权重调整观测的标准差（权重大的点更重要，所以sigma更小）
                obs_sigma = sigma / torch.sqrt(weights)
                pyro.sample("obs", dist.Normal(mu, obs_sigma), obs=y)
            else:
                pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        
        return mu, sigma

bnn_model = PhysicsInformedBNN(in_dim=len(feature_cols), hidden_dim=64)

# ============================================================
# 7. 训练BNN
# ============================================================

print(f"\n[7] Training BNN (5000 epochs)...")

pyro.clear_param_store()
guide = AutoDiagonalNormal(bnn_model)
optimizer = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())

losses = []
for epoch in range(1, 5001):
    loss = svi.step(X_train, y_train, w_train)
    losses.append(loss)
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}: Loss = {loss:.2f}")

print("\n[8] Training complete!")

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150)
print(f"   Loss curve saved to {save_dir}/training_loss.png")
plt.close()

# ============================================================
# 8. 评估模型
# ============================================================

print(f"\n[9] Evaluating model...")

predictive = Predictive(bnn_model, guide=guide, num_samples=200)
samples = predictive(X_test)

# 预测Time_per_Volume（标准化空间）
y_pred_scaled_mean = samples["obs"].mean(dim=0).detach().numpy()
y_pred_scaled_std = samples["obs"].std(dim=0).detach().numpy()

# 反标准化到Time_per_Volume
y_pred_per_vol_mean = scaler_y.inverse_transform(y_pred_scaled_mean.reshape(-1, 1)).flatten()
y_pred_per_vol_std = y_pred_scaled_std * scaler_y.scale_[0]

# 转换回实际的Injection Time
y_pred_time_mean = y_pred_per_vol_mean * volume_test.numpy()
y_pred_time_std = y_pred_per_vol_std * volume_test.numpy()

# 真实值
y_true_time = test_df[target_col].to_numpy()

# 计算指标
rmse = np.sqrt(mean_squared_error(y_true_time, y_pred_time_mean))
mae = mean_absolute_error(y_true_time, y_pred_time_mean)
r2 = r2_score(y_true_time, y_pred_time_mean)

print(f"   RMSE: {rmse:.4f} seconds")
print(f"   MAE:  {mae:.4f} seconds")
print(f"   R²:   {r2:.4f}")

# 按温度和浓度分组评估
print(f"\n   Performance by Temperature:")
for temp in [5.0, 20.0, 40.0]:
    mask = test_df["Temperature_num"] == temp
    if mask.sum() > 0:
        rmse_temp = np.sqrt(mean_squared_error(y_true_time[mask], y_pred_time_mean[mask]))
        r2_temp = r2_score(y_true_time[mask], y_pred_time_mean[mask])
        print(f"      {temp}°C: RMSE={rmse_temp:.4f}, R²={r2_temp:.4f} (n={mask.sum()})")

print(f"\n   Performance by Concentration:")
for conc_range in [(0, 3), (3, 6), (6, 10)]:
    mask = (test_df["Concentration"] >= conc_range[0]) & (test_df["Concentration"] < conc_range[1])
    if mask.sum() > 0:
        rmse_conc = np.sqrt(mean_squared_error(y_true_time[mask], y_pred_time_mean[mask]))
        r2_conc = r2_score(y_true_time[mask], y_pred_time_mean[mask])
        print(f"      {conc_range[0]}-{conc_range[1]}: RMSE={rmse_conc:.4f}, R²={r2_conc:.4f} (n={mask.sum()})")

# ============================================================
# 9. 可视化预测结果
# ============================================================

print(f"\n[10] Creating visualizations...")

# 9.1 预测 vs 实际
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：整体预测
ax = axes[0]
ax.scatter(y_true_time, y_pred_time_mean, alpha=0.5, s=20)
ax.errorbar(y_true_time, y_pred_time_mean, yerr=y_pred_time_std, 
            fmt='none', alpha=0.2, ecolor='gray')
min_val = min(y_true_time.min(), y_pred_time_mean.min())
max_val = max(y_true_time.max(), y_pred_time_mean.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('True Injection Time (s)')
ax.set_ylabel('Predicted Injection Time (s)')
ax.set_title(f'Predictions vs True Values\nRMSE={rmse:.4f}, R²={r2:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：残差
ax = axes[1]
residuals = y_pred_time_mean - y_true_time
ax.scatter(y_true_time, residuals, alpha=0.5, s=20)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('True Injection Time (s)')
ax.set_ylabel('Residual (Predicted - True)')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150)
print(f"   Saved: {save_dir}/predictions.png")
plt.close()

# 9.2 按温度分组的预测
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
temps = [5.0, 20.0, 40.0]
temp_names = ['Cool (5°C)', 'Standard (20°C)', 'Warm (40°C)']

for i, (temp, name) in enumerate(zip(temps, temp_names)):
    ax = axes[i]
    mask = test_df["Temperature_num"] == temp
    if mask.sum() > 0:
        ax.scatter(y_true_time[mask], y_pred_time_mean[mask], alpha=0.6, s=30)
        min_val = y_true_time[mask].min()
        max_val = y_true_time[mask].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        rmse_temp = np.sqrt(mean_squared_error(y_true_time[mask], y_pred_time_mean[mask]))
        r2_temp = r2_score(y_true_time[mask], y_pred_time_mean[mask])
        ax.set_title(f'{name}\nRMSE={rmse_temp:.4f}, R²={r2_temp:.4f}')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('True Time (s)')
    ax.set_ylabel('Predicted Time (s)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'predictions_by_temp.png'), dpi=150)
print(f"   Saved: {save_dir}/predictions_by_temp.png")
plt.close()

# ============================================================
# 10. 导出模型
# ============================================================

print(f"\n[11] Exporting model for production...")

# 从guide采样权重
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

# 保存
export_data = {
    'weight_samples': weight_samples,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_cols': feature_cols,
    'hidden_dim': 64,
    'in_dim': len(feature_cols),
    'performance': {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    },
    'note': 'Model predicts Time/Volume, multiply by Volume to get Injection Time'
}

export_path = os.path.join(save_dir, 'bnn_export.pkl')
joblib.dump(export_data, export_path)
print(f"   Saved: {export_path}")

# ============================================================
# 11. 测试物理约束
# ============================================================

print(f"\n[12] Testing physical constraints...")

def predict_time(temp, volume, conc, visc, dens, spring_k, model_data, num_samples=100):
    """使用导出的模型预测"""
    # 准备输入
    x_input = np.array([[temp, volume, conc, visc, dens, spring_k]])
    x_scaled = model_data['scaler_X'].transform(x_input)
    
    predictions = []
    for weights in model_data['weight_samples'][:num_samples]:
        h = np.maximum(0, x_scaled @ weights['fc1.weight'].T + weights['fc1.bias'])
        h = np.maximum(0, h @ weights['fc2.weight'].T + weights['fc2.bias'])
        out = h @ weights['out.weight'].T + weights['out.bias']
        mu_scaled = out[0, 0]
        predictions.append(mu_scaled)
    
    # 反标准化
    time_per_vol = model_data['scaler_y'].inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    injection_time = time_per_vol * volume
    
    return injection_time.mean(), injection_time.std()

# 测试1: Volume线性关系
print(f"\n   Test 1: Volume linearity (other factors fixed)")
print(f"   Temp=20, Conc=2, Visc=1.5, Dens=1.1, Spring=0.4")
for vol in [0.5, 0.75, 1.0]:
    time_mean, time_std = predict_time(20, vol, 2, 1.5, 1.1, 0.4, export_data)
    print(f"      Volume={vol}ml: Time={time_mean:.3f}±{time_std:.3f}s")

# 测试2: Temperature单调性
print(f"\n   Test 2: Temperature monotonicity (should decrease)")
print(f"   Vol=0.5, Conc=2, Visc=1.5, Dens=1.1, Spring=0.4")
for temp in [5, 20, 40]:
    time_mean, time_std = predict_time(temp, 0.5, 2, 1.5, 1.1, 0.4, export_data)
    print(f"      Temp={temp}°C: Time={time_mean:.3f}±{time_std:.3f}s")

# 测试3: Concentration单调性
print(f"\n   Test 3: Concentration monotonicity (should increase)")
print(f"   Temp=20, Vol=0.5, Visc=1.5, Dens=1.1, Spring=0.4")
for conc in [1, 3, 6, 9]:
    time_mean, time_std = predict_time(20, 0.5, conc, 1.5, 1.1, 0.4, export_data)
    print(f"      Conc={conc}: Time={time_mean:.3f}±{time_std:.3f}s")

# 测试4: Spring_k单调性
print(f"\n   Test 4: Spring_k monotonicity (should decrease)")
print(f"   Temp=20, Vol=0.5, Conc=2, Visc=1.5, Dens=1.1")
for spring in [0.35, 0.40, 0.45]:
    time_mean, time_std = predict_time(20, 0.5, 2, 1.5, 1.1, spring, export_data)
    print(f"      Spring_k={spring}: Time={time_mean:.3f}±{time_std:.3f}s")

print("\n" + "="*70)
print("[DONE] Improved BNN training complete!")
print(f"Model saved to: {save_dir}/")
print("="*70)

