# -*- coding: utf-8 -*-
"""
重新拟合物理公式 - 强制Density正相关
同时尽量保持其他参数不变
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib
import os

print("="*70)
print("重新拟合物理公式（强制Density正相关）")
print("="*70)

# 加载训练数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "training_dataset_with_spring.xlsx")

df = pd.read_excel(excel_path)
df["Temperature"] = df["Temperature"].astype(str).str.strip().str.capitalize()
temp_map = {"Cool": 5.0, "Standard": 20.0, "Warm": 40.0}
df["Temperature_num"] = df["Temperature"].map(temp_map)

feature_cols = ["Temperature_num", "Volume", "Concentration", "Viscosity", "Density", "Spring_k_mean"]
target_col = "Injection Time"

df_clean = df[feature_cols + [target_col]].dropna()

print(f"\n[1] 加载数据: {len(df_clean)} 样本")

# 提取数据
temp = df_clean["Temperature_num"].values
vol = df_clean["Volume"].values
conc = df_clean["Concentration"].values
visc = df_clean["Viscosity"].values
dens = df_clean["Density"].values
spring = df_clean["Spring_k_mean"].values
time = df_clean[target_col].values

print(f"\n[2] 拟合物理公式（带约束）...")
print("   公式: Time = k0 × (Volume^α1 × Concentration^α2 × Viscosity^α3 × Density^α4) / (Temperature^β × Spring_k^γ)")

# 定义物理模型
def physics_model(params, temp, vol, conc, visc, dens, spring):
    k0, alpha1, alpha2, alpha3, alpha4, beta, gamma = params
    
    # 防止除零和负数
    temp = np.maximum(temp, 1.0)
    spring = np.maximum(spring, 0.01)
    
    # 物理公式
    time_pred = k0 * (vol**alpha1 * conc**alpha2 * visc**alpha3 * dens**alpha4) / (temp**beta * spring**gamma)
    
    return time_pred

# 定义损失函数（MSE + 正则化）
def loss_function(params):
    time_pred = physics_model(params, temp, vol, conc, visc, dens, spring)
    mse = np.mean((time_pred - time)**2)
    
    # 添加正则化：鼓励参数接近原始值（除了alpha4）
    # 原始参数（除了alpha4）
    original_params = [2.6755, 0.8, 0.042, 0.488, 0.0, 0.035, 0.890]
    regularization = 0
    
    for i, (p, p_orig) in enumerate(zip(params, original_params)):
        if i != 4:  # 不对alpha4进行正则化
            regularization += 0.1 * (p - p_orig)**2
    
    return mse + regularization

# 初始参数（使用原来的，但alpha4改为0.5）
initial_params = [
    2.6755,  # k0
    0.8,     # alpha1: Volume
    0.042,   # alpha2: Concentration
    0.488,   # alpha3: Viscosity
    0.5,     # alpha4: Density（改为0.5，强制正相关）
    0.035,   # beta: Temperature
    0.890    # gamma: Spring_k
]

# 参数边界（强制alpha4>0）
bounds = [
    (0.1, 50.0),     # k0
    (0.7, 1.0),      # alpha1: Volume（保持接近原值）
    (0.0, 0.2),      # alpha2: Concentration（保持接近原值）
    (0.3, 0.7),      # alpha3: Viscosity（保持接近原值）
    (0.3, 1.5),      # alpha4: Density（强制>0，正相关）
    (0.0, 0.2),      # beta: Temperature（保持接近原值）
    (0.7, 1.2)       # gamma: Spring_k（保持接近原值）
]

# 优化
print("   开始优化...")
result = minimize(loss_function, initial_params, method='L-BFGS-B', bounds=bounds)

if result.success:
    print("   ✓ 优化成功!")
else:
    print("   ⚠ 优化可能未完全收敛")

# 最优参数
k0, alpha1, alpha2, alpha3, alpha4, beta, gamma = result.x

print(f"\n[3] 拟合结果（对比）:")
print(f"   参数              旧值      新值      变化")
print(f"   {'='*50}")
print(f"   k0 (基准系数)    2.6755    {k0:.4f}   {k0-2.6755:+.4f}")
print(f"   α1 (Volume)      0.8000    {alpha1:.4f}   {alpha1-0.8:+.4f}")
print(f"   α2 (Concentration) 0.0420  {alpha2:.4f}   {alpha2-0.042:+.4f}")
print(f"   α3 (Viscosity)   0.4881    {alpha3:.4f}   {alpha3-0.488:+.4f}")
print(f"   α4 (Density)     0.0000    {alpha4:.4f}   {alpha4-0.0:+.4f}  ← 修正！")
print(f"   β (Temperature)  0.0349    {beta:.4f}   {beta-0.035:+.4f}")
print(f"   γ (Spring_k)     0.8898    {gamma:.4f}   {gamma-0.890:+.4f}")

# 评估拟合质量
time_pred = physics_model(result.x, temp, vol, conc, visc, dens, spring)
rmse = np.sqrt(np.mean((time_pred - time)**2))
mae = np.mean(np.abs(time_pred - time))
r2 = 1 - np.sum((time - time_pred)**2) / np.sum((time - time.mean())**2)

print(f"\n[4] 拟合质量:")
print(f"   RMSE = {rmse:.4f} 秒")
print(f"   MAE = {mae:.4f} 秒")
print(f"   R² = {r2:.4f}")

# 测试单调性
print(f"\n[5] 测试单调性:")

def predict_physics(temp, vol, conc, visc, dens, spring):
    return physics_model(result.x, 
                        np.array([temp]), 
                        np.array([vol]), 
                        np.array([conc]), 
                        np.array([visc]), 
                        np.array([dens]), 
                        np.array([spring]))[0]

# Density单调性（重点测试）
print("   Density单调性 (其他参数固定):")
density_values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5]
prev_time = None

for d in density_values:
    t = predict_physics(20, 0.75, 2.0, 1.5, d, 0.4)
    trend = ""
    if prev_time is not None:
        trend = "⬆️" if t > prev_time else "⬇️"
    print(f"      Density={d:.1f}: Time={t:.3f}s {trend}")
    prev_time = t

# Spring_k单调性（确保不受影响）
print("\n   Spring_k单调性 (其他参数固定):")
spring_values = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.79]
prev_time = None

for s in spring_values:
    t = predict_physics(20, 0.75, 2.0, 1.5, 1.1, s)
    trend = ""
    if prev_time is not None:
        trend = "⬇️" if t < prev_time else "⬆️"
    print(f"      Spring_k={s:.2f}: Time={t:.3f}s {trend}")
    prev_time = t

# Volume线性关系（确保不受影响）
print("\n   Volume线性关系 (其他参数固定):")
volume_values = [0.5, 0.75, 1.0, 1.5, 2.0]
base_time = None

for v in volume_values:
    t = predict_physics(20, v, 2.0, 1.5, 1.1, 0.4)
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
    'alpha3': alpha3,
    'alpha4': alpha4,
    'beta': beta,
    'gamma': gamma,
    'rmse': rmse,
    'mae': mae,
    'r2': r2
}

# 备份旧参数
old_params_path = os.path.join(BASE_DIR, "physics_model_params_old.pkl")
if os.path.exists(os.path.join(BASE_DIR, "physics_model_params.pkl")):
    import shutil
    shutil.copy(os.path.join(BASE_DIR, "physics_model_params.pkl"), old_params_path)
    print(f"\n[6] 旧参数已备份到: physics_model_params_old.pkl")

save_path = os.path.join(BASE_DIR, "physics_model_params.pkl")
joblib.dump(physics_params, save_path)
print(f"[7] 新参数已保存到: {save_path}")

print("\n" + "="*70)
print("完成！Density现在应该有正相关了。")
print("="*70)

