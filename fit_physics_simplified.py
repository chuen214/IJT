# -*- coding: utf-8 -*-
"""
为简化模型拟合物理公式（4特征）
不使用Concentration和Density
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib
import os

print("="*70)
print("拟合物理公式（简化模型，4特征）")
print("="*70)

# 加载训练数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "training_dataset_with_spring.xlsx")

df = pd.read_excel(excel_path)
df["Temperature"] = df["Temperature"].astype(str).str.strip().str.capitalize()
temp_map = {"Cool": 5.0, "Standard": 20.0, "Warm": 40.0}
df["Temperature_num"] = df["Temperature"].map(temp_map)

feature_cols = ["Temperature_num", "Volume", "Viscosity", "Spring_k_mean"]
target_col = "Injection Time"

df_clean = df[feature_cols + [target_col]].dropna()

print(f"\n[1] 加载数据: {len(df_clean)} 样本")

# 提取数据
temp = df_clean["Temperature_num"].values
vol = df_clean["Volume"].values
visc = df_clean["Viscosity"].values
spring = df_clean["Spring_k_mean"].values
time = df_clean[target_col].values

print(f"\n[2] 拟合物理公式（简化版）...")
print("   公式: Time = k0 × (Volume^α1 × Viscosity^α2) / (Temperature^β × Spring_k^γ)")

# 定义物理模型
def physics_model(params, temp, vol, visc, spring):
    k0, alpha1, alpha2, beta, gamma = params
    
    # 防止除零和负数
    temp = np.maximum(temp, 1.0)
    spring = np.maximum(spring, 0.01)
    
    # 物理公式
    time_pred = k0 * (vol**alpha1 * visc**alpha2) / (temp**beta * spring**gamma)
    
    return time_pred

# 定义损失函数（MSE）
def loss_function(params):
    time_pred = physics_model(params, temp, vol, visc, spring)
    mse = np.mean((time_pred - time)**2)
    return mse

# 初始参数猜测
initial_params = [
    5.0,    # k0: 基准系数
    1.0,    # alpha1: Volume指数
    0.5,    # alpha2: Viscosity指数
    0.5,    # beta: Temperature指数（反比）
    1.0     # gamma: Spring_k指数（反比）
]

# 参数边界
bounds = [
    (0.1, 50.0),    # k0
    (0.8, 1.2),     # alpha1: Volume应该接近线性
    (0.0, 2.0),     # alpha2: Viscosity
    (0.0, 2.0),     # beta: Temperature（反比）
    (0.0, 3.0)      # gamma: Spring_k（反比）
]

# 优化
print("   开始优化...")
result = minimize(loss_function, initial_params, method='L-BFGS-B', bounds=bounds)

if result.success:
    print("   ✓ 优化成功!")
else:
    print("   ⚠ 优化可能未完全收敛")

# 最优参数
k0, alpha1, alpha2, beta, gamma = result.x

print(f"\n[3] 拟合结果:")
print(f"   k0 (基准系数) = {k0:.4f}")
print(f"   α1 (Volume指数) = {alpha1:.4f}")
print(f"   α2 (Viscosity指数) = {alpha2:.4f}")
print(f"   β (Temperature指数) = {beta:.4f}")
print(f"   γ (Spring_k指数) = {gamma:.4f}")

# 评估拟合质量
time_pred = physics_model(result.x, temp, vol, visc, spring)
rmse = np.sqrt(np.mean((time_pred - time)**2))
mae = np.mean(np.abs(time_pred - time))
r2 = 1 - np.sum((time - time_pred)**2) / np.sum((time - time.mean())**2)

print(f"\n[4] 拟合质量:")
print(f"   RMSE = {rmse:.4f} 秒")
print(f"   MAE = {mae:.4f} 秒")
print(f"   R² = {r2:.4f}")

# 测试单调性
print(f"\n[5] 测试单调性:")

def predict_physics(temp, vol, visc, spring):
    return physics_model(result.x, 
                        np.array([temp]), 
                        np.array([vol]), 
                        np.array([visc]), 
                        np.array([spring]))[0]

# Spring_k单调性
print("   Spring_k单调性:")
spring_values = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.79]
prev_time = None

for s in spring_values:
    t = predict_physics(20, 0.75, 1.5, s)
    trend = ""
    if prev_time is not None:
        trend = "⬇️" if t < prev_time else "⬆️"
    print(f"      Spring_k={s:.2f}: Time={t:.3f}s {trend}")
    prev_time = t

# Volume线性关系
print("\n   Volume线性关系:")
volume_values = [0.5, 0.75, 1.0, 1.5, 2.0]
base_time = None

for v in volume_values:
    t = predict_physics(20, v, 1.5, 0.4)
    if base_time is None:
        base_time = t
        ratio_str = "baseline"
    else:
        actual_ratio = t / base_time
        expected_ratio = v / 0.5
        ratio_str = f"{actual_ratio:.2f}x (expected {expected_ratio:.2f}x)"
    print(f"      Volume={v}ml: Time={t:.3f}s ({ratio_str})")

# 保存参数
physics_params = {
    'k0': k0,
    'alpha1': alpha1,
    'alpha2': alpha2,
    'beta': beta,
    'gamma': gamma,
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'model_type': 'simplified'  # 标记为简化模型
}

save_path = os.path.join(BASE_DIR, "physics_model_params_simplified.pkl")
joblib.dump(physics_params, save_path)
print(f"\n[6] 参数已保存到: {save_path}")

print("\n" + "="*70)
print("完成!")
print("="*70)

