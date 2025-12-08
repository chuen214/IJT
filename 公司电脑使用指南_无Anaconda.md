# InJight 公司电脑使用指南（无Anaconda版本）

**版本**: v2.2  
**适用**: Windows系统，标准Python（无需Anaconda）  
**最后更新**: 2025-12-02

---

## 📋 目录

1. [前置要求](#前置要求)
2. [首次设置](#首次设置)
3. [日常使用](#日常使用)
4. [常见问题](#常见问题)
5. [文件说明](#文件说明)

---

## ✅ 前置要求

### 必需软件

1. **Python 3.8 - 3.11**
   - 检查是否已安装：
     ```powershell
     python --version
     ```
   - 如果未安装，从 [python.org](https://www.python.org/downloads/) 下载安装
   - ⚠️ **重要**：安装时勾选 "Add Python to PATH"

2. **pip**（通常随Python一起安装）
   - 检查：
     ```powershell
     pip --version
     ```

### 系统要求

- Windows 10/11
- 至少 500MB 可用磁盘空间
- 网络连接（首次安装依赖时需要）

---

## 🚀 首次设置

### 步骤1: 复制文件夹

将整个 `InJight` 文件夹复制到公司电脑的任意位置，例如：
```
C:\Users\你的用户名\Desktop\InJight
```

### 步骤2: 打开PowerShell

1. 在 `InJight` 文件夹内，按住 `Shift` + 右键
2. 选择 "在此处打开 PowerShell 窗口" 或 "在此处打开终端"

### 步骤3: 创建虚拟环境

```powershell
python -m venv venv
```

**预期输出**：
```
（无错误信息，创建成功）
```

**耗时**：约30秒

### 步骤4: 激活虚拟环境

```powershell
.\venv\Scripts\Activate.ps1
```

**预期输出**：
```
(venv) PS C:\...\InJight>
```

**注意**：如果出现 "无法加载脚本" 错误，执行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
然后重新执行激活命令。

### 步骤5: 安装依赖包

```powershell
pip install -r requirements.txt
```

**预期输出**：
```
Collecting streamlit>=1.28.0
Collecting torch>=2.0.0
...
Successfully installed streamlit-1.51.0 torch-2.9.1 ...
```

**耗时**：约5-10分钟（取决于网络速度）

**如果网络慢**，可以使用国内镜像：
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤6: 验证安装

```powershell
python check_environment.py
```

**预期输出**：
```
[OK] Python Version
[OK] pip
[OK] streamlit
[OK] torch
[OK] pyro-ppl
...
[SUCCESS] All checks passed!
```

### 步骤7: 测试启动UI

```powershell
streamlit run streamlit_app_improved.py
```

**预期输出**：
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

浏览器会自动打开，显示InJight界面。

**如果浏览器未自动打开**：
- 手动访问：http://localhost:8501

---

## 📝 日常使用

### 方法1: 使用批处理文件（推荐）

**双击**：`start_ui_venv.bat`

这会自动：
1. 激活虚拟环境
2. 启动UI
3. 打开浏览器

### 方法2: 手动启动

1. 打开PowerShell（在InJight文件夹内）
2. 激活虚拟环境：
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. 启动UI：
   ```powershell
   streamlit run streamlit_app_improved.py
   ```

---

## 🛠️ 常见问题

### 问题1: "python不是内部或外部命令"

**原因**：Python未安装或未添加到PATH

**解决**：
1. 检查Python是否安装：
   ```powershell
   py --version
   ```
2. 如果 `py` 命令可用，使用 `py` 代替 `python`：
   ```powershell
   py -m venv venv
   py -m pip install -r requirements.txt
   ```

### 问题2: "无法加载脚本，因为在此系统上禁止运行脚本"

**原因**：PowerShell执行策略限制

**解决**：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
然后重新激活虚拟环境。

### 问题3: "ModuleNotFoundError: No module named 'xxx'"

**原因**：依赖包未安装或虚拟环境未激活

**解决**：
1. 确保虚拟环境已激活（提示符前有 `(venv)`）
2. 重新安装依赖：
   ```powershell
   pip install -r requirements.txt
   ```

### 问题4: "端口8501已被占用"

**原因**：另一个Streamlit应用正在运行

**解决**：
1. 关闭其他Streamlit应用
2. 或使用其他端口：
   ```powershell
   streamlit run streamlit_app_improved.py --server.port 8502
   ```

### 问题5: 模型文件未找到

**错误信息**：
```
FileNotFoundError: saved_bnn_improved/bnn_export.pkl
```

**原因**：模型文件缺失

**解决**：
1. 检查模型文件是否存在：
   ```
   saved_bnn_improved/bnn_export.pkl
   saved_bnn_simplified/bnn_export.pkl
   physics_model_params.pkl
   physics_model_params_simplified.pkl
   ```
2. 如果缺失，需要重新训练（见下方）

### 问题6: 安装依赖时网络超时

**解决**：使用国内镜像
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 🔄 重新训练模型（如需要）

### 训练完整模型

```powershell
# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 训练BNN
python train_bnn_improved.py

# 拟合物理公式
python refit_physics_model.py
```

**耗时**：约5-10分钟

### 训练简化模型

```powershell
# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 训练BNN
python train_bnn_simplified.py

# 拟合物理公式
python fit_physics_simplified.py
```

**耗时**：约5-10分钟

---

## 📁 文件说明

### 必需文件（运行系统）

| 文件 | 说明 | 必需 |
|------|------|------|
| `streamlit_app_improved.py` | 主UI程序 | ✅ |
| `hybrid_inference.py` | 混合预测器 | ✅ |
| `saved_bnn_improved/bnn_export.pkl` | 完整模型 | ✅ |
| `saved_bnn_simplified/bnn_export.pkl` | 简化模型 | ✅ |
| `physics_model_params.pkl` | 完整物理参数 | ✅ |
| `physics_model_params_simplified.pkl` | 简化物理参数 | ✅ |
| `requirements.txt` | 依赖列表 | ✅ |

### 可选文件

| 文件 | 说明 | 何时需要 |
|------|------|---------|
| `training_dataset_with_spring.xlsx` | 训练数据 | 重新训练时 |
| `train_bnn_improved.py` | 训练脚本 | 重新训练时 |
| `check_environment.py` | 环境检查 | 排查问题时 |

---

## 🎯 快速参考

### 首次设置（一次性）

```powershell
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证
python check_environment.py

# 5. 启动UI
streamlit run streamlit_app_improved.py
```

### 日常使用

```powershell
# 方法1: 双击 start_ui_venv.bat

# 方法2: 手动
.\venv\Scripts\Activate.ps1
streamlit run streamlit_app_improved.py
```

---

## 📊 系统要求总结

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10/11 |
| **Python版本** | 3.8 - 3.11 |
| **磁盘空间** | 至少500MB |
| **内存** | 至少2GB可用 |
| **网络** | 首次安装时需要 |

---

## 🔐 公司电脑特殊注意事项

### 1. 防火墙/杀毒软件

如果UI无法启动，可能是防火墙阻止：
- 允许Python和Streamlit通过防火墙
- 或将端口8501加入白名单

### 2. 代理设置

如果公司网络需要代理：
```powershell
# 设置pip代理
pip install -r requirements.txt --proxy http://代理地址:端口
```

### 3. 权限问题

如果遇到权限错误：
- 以管理员身份运行PowerShell
- 或使用用户目录下的虚拟环境

### 4. 离线安装（如无网络）

如果公司电脑无法联网，需要：
1. 在有网络的电脑上下载所有依赖包
2. 使用 `pip download` 下载wheel文件
3. 复制到公司电脑后使用 `pip install` 安装

---

## 📞 获取帮助

### 1. 查看文档

- `最终使用指南.md` - 完整使用说明
- `使用指南_模型选择.md` - 模型选择说明
- `README.md` - 项目说明

### 2. 运行诊断

```powershell
python check_environment.py
```

### 3. 查看错误信息

- 终端中的错误信息
- Streamlit界面中的错误提示

---

## ✅ 验证清单

首次设置完成后，确认以下项目：

- [ ] Python已安装（`python --version`）
- [ ] 虚拟环境已创建（`venv`文件夹存在）
- [ ] 虚拟环境可激活（提示符前有`(venv)`）
- [ ] 依赖包已安装（`check_environment.py`通过）
- [ ] UI可启动（浏览器显示InJight界面）
- [ ] 模型文件存在（4个.pkl文件）
- [ ] 可以正常预测（输入参数后无错误）

---

## 🎉 完成！

如果所有检查项都通过，系统已准备就绪！

**现在可以开始使用InJight进行注射时间预测了！** 🚀

---

## 📝 更新记录

- **2025-12-02**: 创建无Anaconda版本使用指南
- 适用于InJight v2.2 Final

---

**如有问题，请参考其他文档或联系技术支持。**

