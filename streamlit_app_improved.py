# -*- coding: utf-8 -*-
"""
改進的Streamlit UI - BNN注射時間預測
使用新的6特徵模型（移除了Spring_k_std）
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# 页面配置
st.set_page_config(
    page_title="InJight - 注射時間預測系統",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .improvement-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 標題
st.markdown('<div class="main-header">InJight 注射時間預測系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基於貝葉斯神經網絡的智能預測', unsafe_allow_html=True)

# ============================================================
# 載入模型
# ============================================================
# 注意：模型載入已移至側邊欄部分

# ============================================================
# 預測函數
# ============================================================

@st.cache_resource
def load_hybrid_predictor(model_type='full'):
    """載入混合預測器"""
    from hybrid_inference import HybridPredictor
    try:
        if model_type == 'full':
            predictor = HybridPredictor(
                bnn_model_dir="saved_bnn_improved",
                physics_params_path="physics_model_params.pkl"
            )
        else:  # simplified
            predictor = HybridPredictor(
                bnn_model_dir="saved_bnn_simplified",
                physics_params_path="physics_model_params_simplified.pkl"
            )
        # 验证模型类型
        is_simplified = predictor.is_simplified
        expected_features = 4 if is_simplified else 6
        actual_features = len(predictor.bnn_model.get('feature_cols', []))
        if actual_features != expected_features:
            raise ValueError(f"模型類型不匹配: is_simplified={is_simplified}, 特徵數={actual_features}, 期望={expected_features}")
        return predictor, None
    except Exception as e:
        return None, str(e)

# 模型選擇將在側邊欄進行

def predict_injection_time(temperature, volume, concentration, viscosity, 
                          density, spring_k_mean, num_samples=100, model_type='full'):
    """使用混合預測系統（BNN + 物理公式）"""
    
    # 溫度轉換
    if isinstance(temperature, str):
        temp_map = {"Cool (5°C)": 5.0, "Standard (20°C)": 20.0, "Warm (40°C)": 40.0}
        temperature = temp_map.get(temperature, 20.0)
    
    # 根據模型類型選擇預測器（混合）
    predictor_result = load_hybrid_predictor(model_type)
    if predictor_result[1] is not None:
        raise ValueError(f"模型載入失敗: {predictor_result[1]}")
    predictor = predictor_result[0]
    
    # 验证predictor类型是否正确
    if predictor.is_simplified != (model_type == 'simplified'):
        raise ValueError(f"模型類型不匹配: 請求{model_type}, 實際{'simplified' if predictor.is_simplified else 'full'}")
    
    # 使用混合預測器
    mean, std, method, level, warnings = predictor.predict(
        temperature, volume, concentration, viscosity, 
        density, spring_k_mean, num_samples
    )
    
    # 生成預測分布（用於可視化）
    # 使用正態分布近似
    predictions = np.random.normal(mean, std, num_samples)
    predictions = np.maximum(predictions, 0.01)
    
    # 返回預測、方法和警告
    return predictions, method, level, warnings

# ============================================================
# 側邊欄 - 模型信息
# ============================================================

with st.sidebar:
    st.header("📊 模型選擇")
    
    # 模型選擇器
    model_choice = st.radio(
        "選擇預測模型",
        options=['完整模型 (6特徵)', '簡化模型 (4特徵)'],
        index=0,
        help="完整模型需要濃度和密度；簡化模型不需要",
        key='model_choice_radio'
    )
    
    # 根據選擇載入對應的模型
    model_type = 'full' if '完整' in model_choice else 'simplified'
    
    # 存儲到session state供其他部分使用
    st.session_state['model_type'] = model_type
    hybrid_predictor, hybrid_error = load_hybrid_predictor(model_type)
    
    if hybrid_error:
        st.error(f"❌ 模型載入失敗: {hybrid_error}")
        st.stop()
    
    # 顯示當前模型信息
    st.markdown("---")
    st.markdown(f"### 當前模型: {'完整' if model_type == 'full' else '簡化'}")
    
    if model_type == 'full':
        st.markdown("**輸入特徵 (6個)**:")
        st.markdown("✓ Temperature, Volume")
        st.markdown("✓ **Concentration, Density**")
        st.markdown("✓ Viscosity, Spring_k")
    else:
        st.markdown("**輸入特徵 (4個)**:")
        st.markdown("✓ Temperature, Volume")
        st.markdown("✓ Viscosity, Spring_k")
        st.markdown("⚠️ 不需要濃度和密度")
    
    st.markdown("---")
    
    # 載入對應模型的性能數據
    try:
        if model_type == 'full':
            model_data_path = Path(__file__).parent / "saved_bnn_improved" / "bnn_export.pkl"
        else:
            model_data_path = Path(__file__).parent / "saved_bnn_simplified" / "bnn_export.pkl"
        
        model_data = joblib.load(model_data_path)
        
        st.markdown("### BNN性能指標")
        perf = model_data['performance']
        st.metric("RMSE", f"{perf['rmse']:.4f} 秒")
        st.metric("MAE", f"{perf['mae']:.4f} 秒")
        st.metric("R²", f"{perf['r2']:.4f}")
    except Exception as e:
        st.warning("無法載入性能指標")
    
    st.markdown("---")
    
    st.markdown("### 方法切換規則")
    st.markdown("**安全範圍** → 🧠 神經網絡")
    st.markdown("**小外插** → 🧠 神經網絡 + ⚠️")
    st.markdown("**極端外插** → 🔬 物理公式")
    
    st.markdown("---")
    
    st.markdown("### 輸入特徵範圍")
    
    # 定義訓練範圍和外插範圍
    if model_type == 'full':
        feature_ranges = {
            "Temperature (°C)": {
                "訓練範圍": "5 - 40",
                "可外插範圍": "0 - 60"
            },
            "Volume (ml)": {
                "訓練範圍": "0.5 - 0.75",
                "可外插範圍": "0.1 - 3.0"
            },
            "Concentration": {
                "訓練範圍": "0.5 - 9.6",
                "可外插範圍": "0.1 - 20"
            },
            "Viscosity": {
                "訓練範圍": "1.0 - 3.0",
                "可外插範圍": "0.5 - 10"
            },
            "Density": {
                "訓練範圍": "0.995 - 1.01",
                "可外插範圍": "0.7 - 1.3"
            },
            "Spring K": {
                "訓練範圍": "0.37 - 0.42",
                "可外插範圍": "0.2 - 0.8"
            }
        }
    else:  # simplified
        feature_ranges = {
            "Temperature (°C)": {
                "訓練範圍": "5 - 40",
                "可外插範圍": "0 - 60"
            },
            "Volume (ml)": {
                "訓練範圍": "0.5 - 0.75",
                "可外插範圍": "0.1 - 3.0"
            },
            "Viscosity": {
                "訓練範圍": "1.0 - 3.0",
                "可外插範圍": "0.5 - 10"
            },
            "Spring K": {
                "訓練範圍": "0.37 - 0.42",
                "可外插範圍": "0.2 - 0.8"
            }
        }
    
    for feat_name, ranges in feature_ranges.items():
        st.markdown(f"**{feat_name}**")
        st.markdown(f"✓ 訓練: {ranges['訓練範圍']}")
        st.markdown(f"⚠️ 外插: {ranges['可外插範圍']}")
        st.markdown("")

# ============================================================
# 主界面 - 標籤頁
# ============================================================

# 獲取當前選擇的模型類型
model_type = st.session_state.get('model_type', 'full')

tab1, tab2, tab3, tab4 = st.tabs(["🎯 單次預測", "🔬 敏感性分析", "🎯 反向求解彈簧強度", "ℹ️ 使用說明"])

# ============================================================
# Tab 1: 單次預測
# ============================================================

with tab1:
    st.header("單次預測")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("輸入參數")
        
        temperature = st.selectbox(
            "溫度 (Temperature)",
            ["Cool (5°C)", "Standard (20°C)", "Warm (40°C)"],
            index=1,
            help="藥物儲存/使用溫度"
        )
        
        volume = st.number_input(
            "體積 (Volume, ml)",
            min_value=0.1,
            max_value=3.0,
            value=0.5,
            step=0.05,
            help="注射體積，訓練數據範圍：0.5-0.75ml，可外插至3.0ml"
        )
        
        # 只有完整模型才需要濃度和密度
        if model_type == 'full':
            concentration = st.number_input(
                "濃度 (Concentration)",
                min_value=0.1,
                max_value=20.0,
                value=2.0,
                step=0.1,
                help="藥物濃度，訓練數據範圍：0.5-9.6，可外插至20"
            )
            
            density = st.number_input(
                "密度 (Density)",
                min_value=0.5,
                max_value=3.0,
                value=1.1,
                step=0.01,
                help="藥物密度，訓練數據範圍：0.995-1.01，可外插至3.0"
            )
        else:
            # 簡化模型使用默認值（不顯示）
            concentration = 2.0  # 默認值（不使用）
            density = 1.1  # 默認值（不使用）
            st.info("💡 簡化模型不需要濃度和密度信息")
        
        viscosity = st.number_input(
            "粘度 (Viscosity)",
            min_value=0.5,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="藥物粘度，訓練數據範圍：1.0-3.0，可外插至10"
        )
        
        spring_k_mean = st.number_input(
            "彈簧強度 (Spring K)",
            min_value=0.2,
            max_value=0.8,
            value=0.4,
            step=0.01,
            help="彈簧強度平均值，訓練數據範圍：0.37-0.42，可外插至0.8"
        )
        
        num_samples = st.slider(
            "採樣數量 (Uncertainty Samples)",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="更多採樣數量提供更準確的不確定性估計"
        )
        
        predict_btn = st.button("🚀 開始預測", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("預測結果")
        
        if predict_btn:
            with st.spinner("正在預測..."):
                predictions, method, level, pred_warnings = predict_injection_time(
                    temperature, volume, concentration, 
                    viscosity, density, spring_k_mean, num_samples, model_type
                )
                
                mean_time = predictions.mean()
                std_time = predictions.std()
                ci_lower = np.percentile(predictions, 2.5)
                ci_upper = np.percentile(predictions, 97.5)
                
                # 顯示使用的方法
                method_badge = {
                    'bnn': "🧠 神經網絡 (BNN)",
                    'physics': "🔬 物理公式",
                    'mc': "🧠 MC Dropout"
                }.get(method, method)
                level_badge = {
                    'safe': '✅ 安全範圍',
                    'mild': '⚠️ 小外插',
                    'extreme': '🔴 極端外插'
                }.get(level, level)
                
                if model_type == 'mc':
                    model_name = "MC Dropout (6特徵，無物理)"
                else:
                    model_name = "完整模型 (6特徵)" if model_type == 'full' else "簡化模型 (4特徵)"
                st.info(f"**使用模型**: {model_name} | **預測方法**: {method_badge} | **數據範圍**: {level_badge}")
                
                # 顯示主要結果
                st.markdown("### 📈 預測注射時間")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("平均值", f"{mean_time:.3f} 秒")
                with col_b:
                    st.metric("標準差", f"{std_time:.3f} 秒")
                with col_c:
                    st.metric("變異係數", f"{(std_time/mean_time)*100:.1f}%")
                
                st.markdown("### 📊 置信區間")
                col_d, col_e = st.columns(2)
                with col_d:
                    st.metric("95% CI 下界", f"{ci_lower:.3f} 秒")
                with col_e:
                    st.metric("95% CI 上界", f"{ci_upper:.3f} 秒")
                
                # 預測分布圖
                st.markdown("### 📉 預測分布")
                
                fig = go.Figure()
                
                # 直方圖
                fig.add_trace(go.Histogram(
                    x=predictions,
                    name="預測分布",
                    nbinsx=30,
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                # 添加均值線
                fig.add_vline(
                    x=mean_time, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"均值: {mean_time:.3f}s",
                    annotation_position="top"
                )
                
                # 添加置信區間
                fig.add_vrect(
                    x0=ci_lower, x1=ci_upper,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text="95% CI",
                    annotation_position="top left"
                )
                
                fig.update_layout(
                    title="注射時間預測分布",
                    xaxis_title="注射時間 (秒)",
                    yaxis_title="頻次",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 顯示警告信息
                if pred_warnings:
                    warning_text = "\n\n".join([f"• {w}" for w in pred_warnings])
                    if level == 'extreme':
                        st.error(f"🔴 **極端外插警告**\n\n{warning_text}\n\n已自動切換到物理公式，保證預測的物理合理性。")
                    elif level == 'mild':
                        st.warning(f"⚠️ **小範圍外插警告**\n\n{warning_text}\n\n使用神經網絡預測，但不確定性可能增加。")
                
                # 額外的不確定性警告
                if std_time / mean_time > 0.5:
                    st.warning(f"⚠️ **不確定性較高**：變異係數 = {(std_time/mean_time)*100:.1f}%，建議謹慎使用預測結果。")

# ============================================================
# Tab 2: 敏感性分析
# ============================================================

with tab2:
    st.header("敏感性分析")
    st.markdown("分析各輸入參數對注射時間的影響")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("基準條件")
        
        base_temp = st.number_input("基準溫度", value=20.0, key="sens_temp")
        base_vol = st.number_input("基準體積", value=0.5, key="sens_vol")
        
        # 只有完整模型才顯示濃度和密度
        if model_type == 'full':
            base_conc = st.number_input("基準濃度", value=2.0, key="sens_conc")
            # 將默認密度放回訓練安全範圍，避免敏感性分析全程被判為極端外插而強制用物理公式
            base_dens = st.number_input("基準密度", value=1.0, key="sens_dens")
        else:
            base_conc = 2.0  # 默認值
            base_dens = 1.1  # 默認值
        
        base_visc = st.number_input("基準粘度", value=1.5, key="sens_visc")
        base_spring = st.number_input("基準彈簧強度", value=0.4, key="sens_spring")
        
    # 根據模型類型選擇可變參數
    if model_type == 'full':
        available_params = ["Temperature", "Volume", "Concentration", "Viscosity", "Density", "Spring_k_mean"]
    else:
        available_params = ["Temperature", "Volume", "Viscosity", "Spring_k_mean"]
    
    param_to_vary = st.selectbox(
        "選擇要變化的參數",
        available_params
    )
    
    analyze_btn = st.button("🔬 開始分析", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("分析結果")
        
        if analyze_btn:
            with st.spinner("正在分析..."):
                # 定義參數變化範圍（擴展到外插範圍）
                param_ranges = {
                    "Temperature": np.linspace(0, 60, 15),
                    "Volume": np.linspace(0.3, 2.5, 15),
                    "Viscosity": np.linspace(0.8, 8.0, 15),
                    "Spring_k_mean": np.linspace(0.3, 0.8, 15)
                }
                
                # 定義訓練範圍（用於標註）
                training_ranges = {
                    "Temperature": (5.0, 40.0),
                    "Volume": (0.5, 0.75),
                    "Viscosity": (1.0, 3.0),
                    "Spring_k_mean": (0.37, 0.42)
                }
                
                # 完整模型才有的參數
                if model_type == 'full':
                    param_ranges["Concentration"] = np.linspace(0.5, 15, 15)
                    param_ranges["Density"] = np.linspace(0.7, 1.3, 15)
                    training_ranges["Concentration"] = (0.5, 9.6)
                    training_ranges["Density"] = (0.995, 1.01)
                
                values = param_ranges[param_to_vary]
                means = []
                stds = []
                
                # 根据模型类型构建base_params
                base_params = {
                    "Temperature": base_temp,
                    "Volume": base_vol,
                    "Viscosity": base_visc,
                    "Spring_k_mean": base_spring
                }
                
                # 完整模型才需要Concentration和Density
                if model_type == 'full':
                    base_params["Concentration"] = base_conc
                    base_params["Density"] = base_dens
                
                for val in values:
                    params = base_params.copy()
                    params[param_to_vary] = val
                    
                    predictions, method, level, _ = predict_injection_time(
                        params["Temperature"],
                        params["Volume"],
                        params.get("Concentration", 2.0),  # 簡化模型使用默認值
                        params["Viscosity"],
                        params.get("Density", 1.0),
                        params["Spring_k_mean"],
                        num_samples=100,
                        model_type=model_type
                    )
                    
                    means.append(predictions.mean())
                    stds.append(predictions.std())
                
                # 绘图
                fig = go.Figure()
                
                # 單條「混合模型預測」曲線（已在後端平滑混合）
                fig.add_trace(go.Scatter(
                    x=values,
                    y=np.array(means),
                    mode='lines+markers',
                    name='混合模型預測',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8, symbol='circle')
                ))
                
                # 添加不確定性區間
                fig.add_trace(go.Scatter(
                    x=np.concatenate([values, values[::-1]]),
                    y=np.concatenate([
                        np.array(means) + np.array(stds),
                        (np.array(means) - np.array(stds))[::-1]
                    ]),
                    fill='toself',
                    fillcolor='rgba(0,100,250,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='±1 標準差',
                    showlegend=True
                ))
                
                # 添加訓練範圍標註
                train_min, train_max = training_ranges[param_to_vary]
                fig.add_vrect(
                    x0=train_min, x1=train_max,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text="訓練範圍",
                    annotation_position="top left"
                )
                
                fig.update_layout(
                    title=f"{param_to_vary} 對注射時間的影響（含外插範圍）",
                    xaxis_title=param_to_vary,
                    yaxis_title="注射時間 (秒)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 顯示數據表
                results_df = pd.DataFrame({
                    param_to_vary: values,
                    'Predicted_Time': means,
                    'Std_Dev': stds
                })
                st.dataframe(results_df)

# ============================================================
# Tab 3: 反向求解彈簧強度
# ============================================================

with tab3:
    st.header("反向求解彈簧強度")
    st.markdown("根據目標注射時間範圍，反向計算需要的彈簧強度")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("固定條件")
        
        inv_temp = st.selectbox(
            "溫度 (Temperature)",
            ["Cool (5°C)", "Standard (20°C)", "Warm (40°C)"],
            index=1,
            key="inv_temp"
        )
        inv_vol = st.number_input("體積 (Volume, ml)", value=0.5, min_value=0.3, max_value=2.5, step=0.05, key="inv_vol")
        
        # 根據模型類型顯示不同輸入
        if model_type == 'full':
            inv_conc = st.number_input("濃度 (Concentration)", value=2.0, min_value=0.3, max_value=15.0, step=0.1, key="inv_conc")
            inv_dens = st.number_input("密度 (Density)", value=1.1, min_value=0.7, max_value=1.3, step=0.1, key="inv_dens")
        else:
            inv_conc = 2.0  # 默認值
            inv_dens = 1.1  # 默認值
        
        inv_visc = st.number_input("粘度 (Viscosity)", value=1.5, min_value=0.8, max_value=8.0, step=0.1, key="inv_visc")
        
        st.markdown("---")
        st.subheader("目標時間範圍")
        
        target_time_min = st.number_input("最小時間 (秒)", value=0.0, min_value=0.0, max_value=20.0, step=0.5, key="target_min")
        target_time_max = st.number_input("最大時間 (秒)", value=10.0, min_value=0.0, max_value=20.0, step=0.5, key="target_max")
        
        if target_time_min >= target_time_max:
            st.error("⚠️ 最大時間必須大於最小時間")
        
        solve_btn = st.button("🎯 開始求解", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("求解結果")
        
        if solve_btn and target_time_min < target_time_max:
            with st.spinner("正在求解最優彈簧強度..."):
                # 溫度轉換
                if isinstance(inv_temp, str):
                    temp_map = {"Cool (5°C)": 5.0, "Standard (20°C)": 20.0, "Warm (40°C)": 40.0}
                    inv_temp_num = temp_map.get(inv_temp, 20.0)
                else:
                    inv_temp_num = inv_temp
                
                # 載入預測器
                predictor_result = load_hybrid_predictor(model_type)
                if predictor_result[1] is not None:
                    st.error(f"❌ 模型載入失敗: {predictor_result[1]}")
                else:
                    predictor = predictor_result[0]
                    
                    # 定義目標函數
                    def predict_time(spring_k):
                        """預測注射時間"""
                        mean, std, _, _, _ = predictor.predict(
                            inv_temp_num, inv_vol, inv_conc, inv_visc, inv_dens, spring_k
                        )
                        return mean
                    
                    # 使用二分搜索找到滿足條件的Spring_k範圍
                    # Spring_k越大，Time越小
                    spring_k_min, spring_k_max = 0.3, 0.8
                    
                    # 找到使Time = target_time_max的Spring_k (下界)
                    left, right = spring_k_min, spring_k_max
                    for _ in range(50):  # 50次二分搜索
                        mid = (left + right) / 2
                        time_pred = predict_time(mid)
                        if time_pred > target_time_max:
                            left = mid  # Time太大，需要更大的Spring_k
                        else:
                            right = mid
                    spring_k_lower = (left + right) / 2
                    
                    # 找到使Time = target_time_min的Spring_k (上界)
                    left, right = spring_k_min, spring_k_max
                    for _ in range(50):
                        mid = (left + right) / 2
                        time_pred = predict_time(mid)
                        if time_pred > target_time_min:
                            left = mid
                        else:
                            right = mid
                    spring_k_upper = (left + right) / 2
                    
                    # 驗證結果
                    time_at_lower = predict_time(spring_k_lower)
                    time_at_upper = predict_time(spring_k_upper)
                    
                    # 顯示結果
                    st.markdown("### 🎯 求解成功")
                    
                    st.markdown(f"""
                    **目標時間範圍**: {target_time_min:.2f}s - {target_time_max:.2f}s
                    
                    **推薦彈簧強度範圍**:
                    """)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("下限 Spring_k", f"{spring_k_lower:.4f}", f"→ {time_at_lower:.2f}s")
                    with col_b:
                        st.metric("上限 Spring_k", f"{spring_k_upper:.4f}", f"→ {time_at_upper:.2f}s")
                    
                    # 推薦中間值
                    spring_k_mid = (spring_k_lower + spring_k_upper) / 2
                    time_at_mid = predict_time(spring_k_mid)
                    
                    st.markdown("---")
                    st.markdown("### 📌 推薦值")
                    st.metric(
                        "推薦 Spring_k (中間值)",
                        f"{spring_k_mid:.4f}",
                        f"預測時間: {time_at_mid:.2f}s"
                    )
                    
                    # 驗證範圍
                    if spring_k_lower > spring_k_upper:
                        st.warning("⚠️ 在當前條件下，無法找到滿足目標時間範圍的彈簧強度")
                    elif spring_k_lower < 0.32 or spring_k_upper > 0.8:
                        st.warning("⚠️ 計算的彈簧強度超出合理範圍 (0.32-0.8)，結果可能不可靠")
                    
                    # 繪製Spring_k vs Time曲線
                    st.markdown("---")
                    st.markdown("### 📊 Spring_k 對注射時間的影響")
                    
                    import plotly.graph_objects as go
                    
                    # 生成曲線數據
                    spring_k_range = np.linspace(0.3, 0.8, 30)
                    times = [predict_time(k) for k in spring_k_range]
                    
                    fig = go.Figure()
                    
                    # 主曲線
                    fig.add_trace(go.Scatter(
                        x=spring_k_range,
                        y=times,
                        mode='lines',
                        name='預測時間',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # 目標範圍
                    fig.add_hrect(
                        y0=target_time_min, y1=target_time_max,
                        fillcolor="green", opacity=0.2,
                        layer="below", line_width=0,
                        annotation_text="目標範圍",
                        annotation_position="right"
                    )
                    
                    # 標記推薦點
                    fig.add_trace(go.Scatter(
                        x=[spring_k_mid],
                        y=[time_at_mid],
                        mode='markers',
                        name='推薦值',
                        marker=dict(color='red', size=12, symbol='star')
                    ))
                    
                    # 標記範圍邊界
                    fig.add_trace(go.Scatter(
                        x=[spring_k_lower, spring_k_upper],
                        y=[time_at_lower, time_at_upper],
                        mode='markers',
                        name='邊界值',
                        marker=dict(color='orange', size=10)
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Spring_k (彈簧強度)",
                        yaxis_title="Injection Time (秒)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 詳細信息
                    with st.expander("📋 查看詳細計算"):
                        st.markdown(f"""
                        **固定條件**:
                        - Temperature: {inv_temp_num}°C
                        - Volume: {inv_vol}ml
                        - Concentration: {inv_conc}
                        - Viscosity: {inv_visc}
                        - Density: {inv_dens}
                        
                        **求解結果**:
                        - Spring_k 下限: {spring_k_lower:.4f} → Time = {time_at_lower:.3f}s
                        - Spring_k 推薦: {spring_k_mid:.4f} → Time = {time_at_mid:.3f}s
                        - Spring_k 上限: {spring_k_upper:.4f} → Time = {time_at_upper:.3f}s
                        
                        **說明**:
                        - Spring_k 越大，注射時間越短
                        - 推薦值取範圍中點，可根據實際需求微調
                        - 如果需要更快的注射，選擇較大的 Spring_k
                        - 如果需要更慢的注射，選擇較小的 Spring_k
                        """)

# ============================================================
# Tab 4: 使用說明
# ============================================================

with tab4:
    st.header("使用說明")
    
    st.markdown("""
    ### 📖 系統概述
    
    InJight v2.1 是一個基於**混合預測系統**的注射時間預測系統。
    
    **混合預測策略：**
    - 🧠 **神經網絡 (BNN)**: 用於安全範圍和小範圍外插（最準確）
    - 🔬 **物理公式**: 用於極端外插（保證單調性和物理合理性）
    - 🤖 **自動切換**: 系統根據輸入參數自動選擇最合適的方法
    
    **主要特點：**
    - ✅ **智能方法選擇**: 自動在神經網絡和物理公式間切換
    - ✅ **完美單調性**: 極端外插時保證所有物理約束
    - ✅ **Volume線性關係**: 體積與時間保持線性關係
    - ✅ **不確定性量化**: 提供預測的置信區間
    
    ---
    
    ### 🎯 輸入參數說明
    
    | 參數 | 說明 | 訓練範圍 | 單位 |
    |------|------|----------|------|
    | **Temperature** | 藥物儲存/使用溫度 | 5-40 | °C |
    | **Volume** | 注射體積 | 0.5-0.75 | ml |
    | **Concentration** | 藥物濃度 | 0.5-9.6 | - |
    | **Viscosity** | 藥物粘度 | - | - |
    | **Density** | 藥物密度 | - | - |
    | **Spring_k_mean** | 彈簧強度平均值 | 0.37-0.42 | - |
    
    ---
    
    ### ✅ 物理約束驗證
    
    改進的模型滿足以下物理約束：
    
    1. **Injection Time > 0** (非負性)
    2. **Temperature ↑ → Time ↓** (溫度升高，粘度降低，時間縮短)
    3. **Volume ↑ → Time ↑** (體積增加，時間線性增加)
    4. **Concentration ↑ → Time ↑** (濃度升高，粘度增加，時間增加)
    5. **Viscosity ↑ → Time ↑** (粘度升高，流動阻力增加)
    6. **Density ↑ → Time ↑** (密度升高，質量增加)
    7. **Spring_k ↑ → Time ↓** (彈簧強度增加，推動力增大)
    
    ---
    
    ### 📊 預測解讀
    
    - **平均值**: 預測的最可能注射時間
    - **標準差**: 預測的不確定性
    - **95% 置信區間**: 有95%的概率，真實值在此區間內
    - **變異係數**: 相對不確定性 (Std/Mean×100%)
    
    ---
    
    ### ⚠️ 注意事項
    
    1. **外插警告**: 當輸入參數超出訓練範圍時，系統會發出警告
    2. **不確定性**: 外插預測的不確定性通常更大
    3. **物理合理性**: 即使外插，預測仍滿足物理約束
    4. **Volume線性關係**: 體積加倍時，時間約加倍（誤差<15%）
    
    ---
    
    ### 🆚 與舊模型對比
    
    | 指標 | 新模型 (v2.0) | 舊模型 (v1.0) |
    |------|---------------|---------------|
    | **輸入特徵數** | 6 | 7 (含Spring_k_std) |
    | **Volume線性關係** | ✅ 滿足 (1.44x) | ❌ 不滿足 (1.01x) |
    | **Temperature單調性** | ✅ 滿足 | ❌ 不滿足 |
    | **Concentration單調性** | ✅ 滿足 | ❌ 完全相反 |
    | **外插性能** | ✅ 合理 | ❌ 幾乎不變 |
    | **物理約束** | ✅ 全部滿足 | ❌ 多數不滿足 |
    
    ---
    
    ### 📞 技術支持
    
    如有問題或建議，請聯繫開發團隊。
    
    **版本**: v2.0 (改進版)  
    **更新日期**: 2025-11-30
    """)

# ============================================================
# 頁腳
# ============================================================

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; padding: 1rem;">'
    '© 2025 InJight System | Powered by Bayesian Neural Network v2.0 '
    '<span class="improvement-badge">✨ Physics-Informed</span>'
    '</div>',
    unsafe_allow_html=True
)



