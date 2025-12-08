# 💉 InJight - 注射时间预测系统

**版本**: v2.2 Final  
**状态**: ✅ 生产就绪

基于贝叶斯神经网络（BNN）和物理公式的混合预测系统，用于预测药物注射时间。

---

## 🚀 快速开始

### 启动UI（推荐方式）

```bash
streamlit run streamlit_app_improved.py
```

浏览器自动打开 http://localhost:8501

### 首次使用

1. **检查环境**（可选）：
   ```bash
   python check_environment.py
   ```

2. **安装依赖**（如需要）：
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 系统特性

### 双模型系统
- **完整模型 (6特征)**: Temperature, Volume, Concentration, Viscosity, Density, Spring_k
- **简化模型 (4特征)**: Temperature, Volume, Viscosity, Spring_k（不需要浓度和密度）

### 混合预测
- **神经网络 (BNN)**: 安全范围内，最高精度
- **物理公式**: 极端外插，保证物理约束

### 完美物理约束
- ✅ 7/7物理约束全部满足
- ✅ Volume线性关系
- ✅ 所有单调性正确

---

## 📁 核心文件

### 必需文件
```
streamlit_app_improved.py          # 主UI程序
hybrid_inference.py                # 混合预测器
saved_bnn_improved/                # 完整模型
saved_bnn_simplified/              # 简化模型
physics_model_params.pkl           # 完整模型物理参数
physics_model_params_simplified.pkl # 简化模型物理参数
training_dataset_with_spring.xlsx  # 训练数据
requirements.txt                   # 依赖包列表
```

### 训练脚本（如需要重新训练）
```
train_bnn_improved.py              # 训练完整模型
train_bnn_simplified.py            # 训练简化模型
refit_physics_model.py             # 重新拟合完整物理公式
fit_physics_simplified.py          # 拟合简化物理公式
```

---

## 📚 文档

### 用户文档
- **`最终使用指南.md`** ⭐ - 快速开始和使用说明
- **`使用指南_模型选择.md`** - 模型选择详细说明

### 技术文档
- **`最终版本说明_v2.2.md`** - 版本总结和问题修复记录
- **`模型对比说明.md`** - 完整模型vs简化模型技术对比
- **`混合模型说明.md`** - 混合预测系统工作原理

### 部署文档
- **`公司电脑部署总结.md`** - 公司电脑部署指南
- **`请读我_README.txt`** - 快速参考

---

## 🎯 使用场景

### 场景1: 有完整信息
```
选择：完整模型 (6特征)
输入：所有6个参数
精度：最高 (RMSE=0.54秒)
```

### 场景2: 缺少浓度或密度
```
选择：简化模型 (4特征)
输入：4个参数（不需要浓度和密度）
精度：略低 (RMSE=0.55秒，仅差2%)
```

---

## 🔧 重新训练模型

### 训练完整模型
```bash
python train_bnn_improved.py
python refit_physics_model.py
```

### 训练简化模型
```bash
python train_bnn_simplified.py
python fit_physics_simplified.py
```

---

## 📞 技术支持

### 遇到问题？
1. 查看 `最终使用指南.md`
2. 运行 `python check_environment.py` 检查环境
3. 查看终端错误信息

### 常见问题
- **模型文件未找到**: 运行对应的训练脚本
- **依赖包缺失**: `pip install -r requirements.txt`
- **UI无法启动**: 检查Streamlit是否正确安装

---

## 🎉 版本历史

- **v2.2 Final**: 双模型系统，完美物理约束，模型选择器
- **v2.1**: 混合预测系统（BNN + 物理公式）
- **v2.0**: 改进的BNN（移除Spring_k_std，Loss weighting）
- **v1.0**: 初始版本（有严重问题，已废弃）

---

## 📄 许可证

内部使用项目

---

**祝使用愉快！** 🚀
