# -*- coding: utf-8 -*-
"""Environment Check Script for Injection Time Predictor"""

import sys
import os

print('='*70)
print('Environment Check for Injection Time Predictor')
print('='*70)
print()

passed = 0
failed = 0

def check(name, ok, msg=''):
    global passed, failed
    if ok:
        print(f'[OK] {name}')
        passed += 1
        return True
    else:
        print(f'[FAIL] {name}')
        if msg:
            print(f'      {msg}')
        failed += 1
        return False

# 1. Python Version
print('[1/7] Checking Python Version...')
v = sys.version_info
check(
    f'Python {v.major}.{v.minor}.{v.micro}',
    (3, 8) <= (v.major, v.minor) <= (3, 11),
    'Require Python 3.8-3.11'
)
print()

# 2. Required Packages
print('[2/7] Checking Required Packages...')
packages = {
    'streamlit': 'streamlit',
    'plotly': 'plotly',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'torch': 'torch',
    'pyro': 'pyro-ppl',
    'joblib': 'joblib',
    'sklearn': 'scikit-learn',
    'openpyxl': 'openpyxl'
}

for pkg, install_name in packages.items():
    try:
        __import__(pkg)
        check(pkg, True)
    except ImportError:
        check(pkg, False, f'Run: pip install {install_name}')
print()

# 3. Project Files
print('[3/7] Checking Project Files...')
files = [
    'training_dataset_with_spring.xlsx',
    'streamlit_app_smart.py',
    'simple_inference.py',
    'train_and_export.py',
    'start_ui.bat'
]

for f in files:
    check(f, os.path.exists(f), 'File is missing')
print()

# 4. Model Files
print('[4/7] Checking Model Files...')
model_file = 'saved_bnn_prod/bnn_export.pkl'
model_exists = os.path.exists(model_file)
check(model_file, model_exists, 'Run: python train_and_export.py')

if model_exists:
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    check(f'Model size: {size_mb:.1f} MB', size_mb > 0.1, 'Model file too small')
print()

# 5. Test Inference Module
print('[5/7] Testing Inference Module...')
try:
    from simple_inference import SimpleBNNPredictor
    if model_exists:
        predictor = SimpleBNNPredictor(model_file)
        check('SimpleBNNPredictor loaded', True)
        
        # Quick test
        test_params = {
            'Temperature_num': 20.0,
            'volume': 0.6,
            'concentration': 5.0,
            'viscosity': 1.0,
            'density': 1.0,
            'spring_k_mean': 0.39,
            'spring_k_std': 0.03
        }
        result = predictor.predict(test_params, num_samples=10)
        check('Test prediction', 2.0 <= result['mean'] <= 8.0, 'Prediction out of range')
    else:
        check('SimpleBNNPredictor', False, 'Model file missing')
except Exception as e:
    check('SimpleBNNPredictor', False, str(e))
print()

# 6. Documentation
print('[6/7] Checking Documentation...')
docs = ['README.md', 'START.md', 'requirements.txt']
for d in docs:
    check(d, os.path.exists(d))
print()

# 7. Data Analysis
print('[7/7] Checking Data Analysis...')
eda_dir = 'EDA_with_spring_output'
check(eda_dir, os.path.isdir(eda_dir))
print()

# Summary
print('='*70)
print(f'Results: {passed} Passed, {failed} Failed')
print('='*70)

if failed == 0:
    print()
    print('[SUCCESS] All checks passed!')
    print('Your environment is ready to use.')
    print()
    print('Next steps:')
    print('  1. Double-click: start_ui.bat')
    print('  2. Or run: streamlit run streamlit_app_smart.py')
    print('  3. Visit: http://localhost:8502')
    print()
else:
    print()
    print('[ERROR] Some checks failed.')
    print('Please fix the issues above and run this script again.')
    print()
    print('Common fixes:')
    print('  - Install missing packages: pip install -r requirements.txt')
    print('  - Train model: python train_and_export.py')
    print('  - Check file paths and permissions')
    print()

print('='*70)
sys.exit(0 if failed == 0 else 1)