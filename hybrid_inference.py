# -*- coding: utf-8 -*-
"""
混合推理系统
- 安全范围：使用BNN神经网络（最准确）
- 小外插：使用BNN神经网络（带警告）
- 极端外插：使用物理公式（保证单调性）
"""

import os
import numpy as np
import joblib

class HybridPredictor:
    """混合预测器"""
    
    def __init__(self, bnn_model_dir="saved_bnn_improved", physics_params_path="physics_model_params.pkl"):
        """
        初始化混合预测器
        
        参数:
            bnn_model_dir: BNN模型目录
            physics_params_path: 物理模型参数文件
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 加载BNN模型
        bnn_path = os.path.join(BASE_DIR, bnn_model_dir, "bnn_export.pkl")
        self.bnn_model = joblib.load(bnn_path)
        print(f"✓ BNN模型加载成功")
        
        # 加载物理模型参数
        physics_path = os.path.join(BASE_DIR, physics_params_path)
        self.physics_params = joblib.load(physics_path)
        print(f"✓ 物理模型参数加载成功")
        
        # 检查是否为简化模型
        feature_cols = self.bnn_model.get('feature_cols', [])
        self.is_simplified = len(feature_cols) == 4
        
        # 定义安全范围（训练数据范围）
        self.safe_ranges = {
            'Temperature': (5.0, 40.0),
            'Volume': (0.5, 0.75),
            'Viscosity': (1.0, 3.0),
            'Spring_k_mean': (0.37, 0.42)
        }
        
        # 定义小外插范围（可以使用BNN，但不确定性增加）
        self.mild_extrapolation_ranges = {
            'Temperature': (0.0, 60.0),
            'Volume': (0.3, 1.2),
            'Viscosity': (0.8, 4.0),
            'Spring_k_mean': (0.32, 0.50)
        }
        
        # 完整模型才有的特征
        if not self.is_simplified:
            self.safe_ranges['Concentration'] = (0.5, 9.6)
            self.safe_ranges['Density'] = (0.995, 1.01)  # 实际训练范围！
            self.mild_extrapolation_ranges['Concentration'] = (0.3, 12.0)
            self.mild_extrapolation_ranges['Density'] = (0.994, 1.015)  # 极度严格！
        
        print(f"✓ 混合预测器初始化完成")
    
    def check_extrapolation_level(self, temperature, volume, concentration, 
                                  viscosity, density, spring_k_mean):
        """
        检查外插程度
        
        返回:
            level: 'safe', 'mild', 'extreme'
            details: 超出范围的参数列表
        """
        # 根据模型类型构建参数字典
        params = {
            'Temperature': temperature,
            'Volume': volume,
            'Viscosity': viscosity,
            'Spring_k_mean': spring_k_mean
        }
        
        # 完整模型才检查Concentration和Density
        if not self.is_simplified:
            params['Concentration'] = concentration
            params['Density'] = density
        
        # 检查是否在安全范围内
        in_safe_range = True
        in_mild_range = True
        out_of_safe = []
        out_of_mild = []
        
        for name, value in params.items():
            safe_min, safe_max = self.safe_ranges[name]
            mild_min, mild_max = self.mild_extrapolation_ranges[name]
            
            if value < safe_min or value > safe_max:
                in_safe_range = False
                out_of_safe.append(name)
            
            if value < mild_min or value > mild_max:
                in_mild_range = False
                out_of_mild.append(name)
        
        # 特殊处理：Density训练范围太窄，仅在极端外插时强制使用物理公式
        # 完整模型才有Density
        if not self.is_simplified and 'Density' in params:
            # 只有当Density超出mild范围(0.994-1.015)时才强制使用物理公式
            if density < 0.994 or density > 1.015:
                in_mild_range = False
                if 'Density' not in out_of_mild:
                    out_of_mild.append('Density')
        
        if in_safe_range:
            return 'safe', []
        elif in_mild_range:
            return 'mild', out_of_safe
        else:
            return 'extreme', out_of_mild
    
    def predict_bnn(self, temperature, volume, concentration, viscosity, 
                   density, spring_k_mean, num_samples=100):
        """使用BNN进行预测"""
        # 使用初始化时检测的模型类型
        if self.is_simplified:
            # 简化模型：只需要4个特征（Temperature, Volume, Viscosity, Spring_k_mean）
            x_input = np.array([[temperature, volume, viscosity, spring_k_mean]])
        else:
            # 完整模型：6个特征
            x_input = np.array([[temperature, volume, concentration, viscosity, density, spring_k_mean]])
        
        x_scaled = self.bnn_model['scaler_X'].transform(x_input)
        
        predictions = []
        for weights in self.bnn_model['weight_samples'][:num_samples]:
            h = np.maximum(0, x_scaled @ weights['fc1.weight'].T + weights['fc1.bias'])
            h = np.maximum(0, h @ weights['fc2.weight'].T + weights['fc2.bias'])
            out = h @ weights['out.weight'].T + weights['out.bias']
            mu_scaled = out[0, 0]
            predictions.append(mu_scaled)
        
        predictions = np.array(predictions)
        time_per_vol = self.bnn_model['scaler_y'].inverse_transform(predictions.reshape(-1, 1)).flatten()
        injection_time = time_per_vol * volume
        injection_time = np.maximum(injection_time, 0.01)
        
        return injection_time.mean(), injection_time.std()
    
    def predict_physics(self, temperature, volume, concentration, viscosity, 
                       density, spring_k_mean):
        """使用物理公式进行预测"""
        params = self.physics_params
        
        # 防止除零
        temperature = max(temperature, 1.0)
        spring_k_mean = max(spring_k_mean, 0.01)
        
        # 检查是否为简化模型（通过参数是否存在来判断）
        # 简化模型：k0, alpha1, alpha2, beta, gamma (5个)
        # 完整模型：k0, alpha1, alpha2, alpha3, alpha4, beta, gamma (7个)
        is_simplified = ('alpha3' not in params) or ('alpha4' not in params)
        
        if is_simplified:
            # 简化模型物理公式（4特征）
            time = params['k0'] * \
                   (volume ** params['alpha1'] * 
                    viscosity ** params['alpha2']) / \
                   (temperature ** params['beta'] * 
                    spring_k_mean ** params['gamma'])
        else:
            # 完整模型物理公式（6特征）
            time = params['k0'] * \
                   (volume ** params['alpha1'] * 
                    concentration ** params['alpha2'] * 
                    viscosity ** params['alpha3'] * 
                    density ** params['alpha4']) / \
                   (temperature ** params['beta'] * 
                    spring_k_mean ** params['gamma'])
        
        time = max(time, 0.01)
        
        # 估算不确定性（基于拟合误差）
        std = params['rmse'] * 1.5  # 外插时不确定性增加
        
        return time, std
    
    def predict(self, temperature, volume, concentration, viscosity, 
               density, spring_k_mean, num_samples=100, force_method=None):
        """
        智能混合预测
        
        参数:
            force_method: 强制使用的方法 ('bnn' 或 'physics')，None则自动选择
        
        返回:
            mean: 预测均值
            std: 预测标准差
            method: 使用的方法 ('bnn' 或 'physics')
            level: 外插程度 ('safe', 'mild', 'extreme')
            warnings: 警告信息列表
        """
        # 检查外插程度
        level, out_params = self.check_extrapolation_level(
            temperature, volume, concentration, viscosity, density, spring_k_mean
        )
        
        warnings = []
        
        # 决定使用哪个方法
        if force_method:
            method = force_method
        elif level == 'extreme':
            method = 'physics'
            warnings.append("检测到极端外插，使用物理公式保证单调性")
        else:
            method = 'bnn'
            if level == 'mild':
                warnings.append("检测到小范围外插，预测不确定性可能增加")
        
        # 进行预测
        if method == 'bnn':
            mean, std = self.predict_bnn(
                temperature, volume, concentration, viscosity, 
                density, spring_k_mean, num_samples
            )
        else:
            mean, std = self.predict_physics(
                temperature, volume, concentration, viscosity, 
                density, spring_k_mean
            )
        
        # 添加具体的外插参数警告
        if out_params:
            param_names = {
                'Temperature': '温度',
                'Volume': '体积',
                'Concentration': '浓度',
                'Viscosity': '粘度',
                'Density': '密度',
                'Spring_k_mean': '弹簧强度'
            }
            out_param_names = [param_names.get(p, p) for p in out_params]
            warnings.append(f"超出训练范围的参数: {', '.join(out_param_names)}")
        
        return mean, std, method, level, warnings

# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("混合预测系统测试")
    print("="*70)
    
    # 初始化预测器
    predictor = HybridPredictor()
    
    print(f"\n{'='*70}")
    print("测试场景")
    print(f"{'='*70}\n")
    
    test_cases = [
        {
            'name': '场景1: 安全范围内',
            'params': (20, 0.5, 2.0, 1.5, 1.1, 0.40),
            'expected_method': 'bnn'
        },
        {
            'name': '场景2: 小外插 (Volume=1.0)',
            'params': (20, 1.0, 2.0, 1.5, 1.1, 0.40),
            'expected_method': 'bnn'
        },
        {
            'name': '场景3: 极端外插 (Spring_k=0.60)',
            'params': (20, 0.75, 2.0, 1.5, 1.1, 0.60),
            'expected_method': 'physics'
        },
        {
            'name': '场景4: 极端外插 (Spring_k=0.79)',
            'params': (20, 0.75, 2.0, 1.5, 1.1, 0.79),
            'expected_method': 'physics'
        },
        {
            'name': '场景5: 多参数外插',
            'params': (5, 2.0, 8.0, 3.0, 1.5, 0.50),
            'expected_method': 'physics'
        }
    ]
    
    for case in test_cases:
        print(f"【{case['name']}】")
        print(f"   输入: Temp={case['params'][0]}°C, Vol={case['params'][1]}ml, " +
              f"Conc={case['params'][2]}, Visc={case['params'][3]}, " +
              f"Dens={case['params'][4]}, Spring={case['params'][5]}")
        
        mean, std, method, level, warnings = predictor.predict(*case['params'])
        
        print(f"   预测: {mean:.3f} ± {std:.3f} 秒")
        print(f"   方法: {method.upper()} ({'神经网络' if method == 'bnn' else '物理公式'})")
        print(f"   级别: {level.upper()}")
        
        if warnings:
            for warning in warnings:
                print(f"   ⚠️  {warning}")
        
        print()
    
    # 测试Spring_k单调性（使用混合方法）
    print(f"{'='*70}")
    print("Spring_k单调性测试（自动切换方法）")
    print(f"{'='*70}\n")
    
    spring_values = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.79]
    prev_time = None
    
    for spring in spring_values:
        mean, std, method, level, _ = predictor.predict(5, 0.75, 5.0, 1.5, 1.1, spring)
        
        trend = ""
        if prev_time is not None:
            trend = "⬇️" if mean < prev_time else "⬆️"
        
        method_str = "BNN" if method == 'bnn' else "物理"
        print(f"   Spring_k={spring:.2f}: Time={mean:.3f}±{std:.3f}s {trend} [{method_str}]")
        prev_time = mean
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")

