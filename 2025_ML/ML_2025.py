# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:24:10 2025

@author: jiangchen
"""

import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
import shap  # 用于SHAP分析
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import RFE
import traceback
from scipy import stats

# -------------------------- Basic Configuration --------------------------
RANDOM_SEED = 42  # Global random seed to ensure reproducibility

# -------------------------- 1. Data Preprocessing --------------------------

# Load data
def load_and_preprocess_data():
    """Unified data preprocessing function to ensure consistency in dataset preparation"""
    try:
        # Load data
        train_data = pd.read_excel('train_data.xlsx')
        test_data = pd.read_excel('test_data.xlsx')
            
        # 区分有序和无序分类变量（根据业务逻辑调整）
        ordinal_cols = ['Disease stage', 'RPR Titer Baseline']  # 有序：有等级关系（如疾病阶段I < II < III）
        nominal_cols = ['gender']  # 无序：无等级关系（如性别男/女）
        le_dict = {}  # 存储有序变量的编码器，用于可能的逆转换

        # 合并所有数据以保证编码一致性
        all_data = pd.concat([train_data.drop('cure and infection', axis=1), 
                     test_data.drop('cure and infection', axis=1)])

        # 处理每个数据集
        datasets = {'train': train_data, 'test': test_data}
        processed_datasets = {}

        for name, data in datasets.items():
            X = data.drop('cure and infection', axis=1)
            y = data['cure and infection']
        
            # 处理目标变量
            if y.dtype == 'object':
                y = y.map({'cured': 0, 'Cured': 0, 'serologically cured': 0, 
                       'infected': 1, 'Infected': 1, 'currently infected': 1})
                y = y.astype(int)
        
            # 反转目标变量（补充依据：前期分析显示80%特征与原始目标负相关，p<0.05）
            y = 1 - y
        
            # 处理有序分类变量：标签编码（保留等级关系）
            for col in ordinal_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    le.fit(all_data[col].astype(str))  # 用所有数据拟合编码器
                    X[col] = le.transform(X[col].astype(str))
                    le_dict[col] = le
        
            # 处理无序分类变量：独热编码（消除虚假等级关系）
            if nominal_cols:
                X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)  # drop_first避免多重共线性
            
            # Remove ID column and unnecessary columns
            if 'number' in X.columns:
                X = X.drop('number', axis=1)
            # Remove 'Unnamed: 11' column if exists
            if 'Unnamed: 11' in X.columns:
                X = X.drop('Unnamed: 11', axis=1)
            
            processed_datasets[f'X_{name}'] = X
            processed_datasets[f'y_{name}'] = y
        
        # Feature standardization - Fit only on training data to avoid data leakage
        scaler = StandardScaler()
        X_train = processed_datasets['X_train']
        scaler.fit(X_train)
        
        # Apply scaling to all datasets
        for name in ['train', 'test']:
            if f'X_{name}' in processed_datasets:
                processed_datasets[f'X_{name}_scaled'] = pd.DataFrame(
                    scaler.transform(processed_datasets[f'X_{name}']),
                    columns=processed_datasets[f'X_{name}'].columns
                )
        
        # Create non-scaled version for Naive Bayes
        for name in ['train', 'test']:
            if f'X_{name}' in processed_datasets:
                processed_datasets[f'X_{name}_non_scaled'] = processed_datasets[f'X_{name}'].copy()
        
        print(f"Data loading completed.")
        print(f"Training set size: {processed_datasets['X_train'].shape}")
        print(f"Test set size: {processed_datasets['X_test'].shape}")
        
        return processed_datasets
        
    except Exception as e:
        raise ValueError(f"数据加载失败，请检查train_data.xlsx和test_data.xlsx是否存在且格式正确：{str(e)}")

# -------------------------- 2. Model and Parameter Grid Initialization --------------------------
def init_models_and_params():
    """Initialize all models and parameter grids with consistent configurations"""
    models = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss'),
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced'),
        "SVM": SVC(random_state=RANDOM_SEED, class_weight='balanced', probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight='balanced'),
        "Neural Network": MLPClassifier(random_state=RANDOM_SEED, max_iter=1000),
        "Naive Bayes": GaussianNB()
    }
    
    # Parameter grids for grid search - consistent across all scripts
    param_grids = {
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        },
        "XGBoost": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.03, 0.1]
        },
        "Logistic Regression": {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        },
        "SVM": {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        "Decision Tree": {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        },
        "Neural Network": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh']
        },
        "Naive Bayes": {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
    }
    
    # Calculate the number of parameter combinations for each model
    param_counts = {
        model_name: np.prod([len(params) for params in param_grid.values()])
        for model_name, param_grid in param_grids.items()
    }
    
    return models, param_grids, param_counts

# -------------------------- 3. Nested Cross-validation with All Features --------------------------
def run_nested_cv(X_train, y_train, models, param_grids, param_counts, X_train_non_scaled=None, export_dir=None):
    # 新增参数检查
    if X_train is None:
        raise ValueError("X_train 为 None，请检查数据输入")
    if y_train is None:
        raise ValueError("y_train 为 None，请检查数据输入")
    if X_train_non_scaled is not None and X_train_non_scaled.empty:
        raise ValueError("X_train_non_scaled 为空，请检查数据预处理逻辑")
def run_nested_cv(X_train, y_train, models, param_grids, param_counts, X_train_non_scaled=None, export_dir=None):
    """Perform nested cross-validation using all features"""
    # Cross-validation settings
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    
    best_models = {}
    cv_details = []
    grid_search_summary = []
    outer_validation_results = {model_name: {'y_true': [], 'y_pred_proba': []} for model_name in models.keys()}
    fold_auc_scores = {model_name: [] for model_name in models.keys()}
    fold_validation_data = {model_name: [] for model_name in models.keys()}
    
    print("\n=== Starting Nested Cross-validation with All Features ===")
    
    for model_name in models.keys():
        print(f"\nProcessing {model_name}...")
        model = models[model_name]
        param_grid = param_grids[model_name]
        param_count = param_counts[model_name]

        outer_scores = []
        fold_details = []

        # Outer cross-validation loop
        for fold_idx, (train_outer_idx, val_outer_idx) in enumerate(outer_cv.split(X_train, y_train)):
            print(f"  Processing outer fold {fold_idx + 1}/5...")
            
            # 分割外部训练集和验证集
            if model_name == "Naive Bayes" and X_train_non_scaled is not None:
                X_outer_train = X_train_non_scaled.iloc[train_outer_idx]
                X_outer_val = X_train_non_scaled.iloc[val_outer_idx]
            else:
                X_outer_train = X_train.iloc[train_outer_idx]
                X_outer_val = X_train.iloc[val_outer_idx]
            
            y_outer_train = y_train.iloc[train_outer_idx]
            y_outer_val = y_train.iloc[val_outer_idx]

            # 内部交叉验证调参
            inner_results = []
            fold_best_params = None
            fold_best_auc = 0
            
            param_combinations = ParameterGrid(param_grid)
            for params_idx, params in enumerate(param_combinations):
                # 创建当前参数的模型
                if model_name == "XGBoost":
                    current_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
                elif model_name == "SVM":
                    current_model = SVC(** params, random_state=RANDOM_SEED, probability=True, class_weight='balanced')
                elif model_name == "Naive Bayes":
                    current_model = GaussianNB(**params)
                else:
                    model_class = model.__class__
                    current_model = model_class(** params, random_state=RANDOM_SEED)
                
                # 内部CV循环
                inner_scores = []
                for inner_idx, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
                    X_inner_train = X_outer_train.iloc[train_inner_idx]
                    y_inner_train = y_outer_train.iloc[train_inner_idx]
                    X_inner_val = X_outer_train.iloc[val_inner_idx]
                    y_inner_val = y_outer_train.iloc[val_inner_idx]
                    
                    # 训练模型
                    try:
                        if model_name == "XGBoost":
                            current_model.fit(X_inner_train, y_inner_train, verbose=False)
                        else:
                            current_model.fit(X_inner_train, y_inner_train)
                        
                        # 计算内部AUC
                        try:
                            y_val_proba = current_model.predict_proba(X_inner_val)[:, 1]
                        except AttributeError:
                            decision_scores = current_model.decision_function(X_inner_val)
                            if len(decision_scores.shape) > 1:
                                decision_scores = decision_scores[:, 1]
                            y_val_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
                        
                        inner_auc = roc_auc_score(y_inner_val, y_val_proba)
                        inner_scores.append(inner_auc)
                    except Exception as e:
                        print(f"Inner loop error: {str(e)}")
                        inner_scores.append(0)
                
                # 记录当前参数的平均内部AUC
                avg_inner_auc = np.mean(inner_scores)
                inner_results.append({'params': params, 'avg_auc': avg_inner_auc})
            
            # 找到内部CV的最佳参数（注意：这行必须在参数循环之外！）
            best_inner_result = max(inner_results, key=lambda x: x['avg_auc'])
            fold_best_params = best_inner_result['params']
            
            # 用最佳参数训练模型
            if model_name == "XGBoost":
                best_inner_model = xgb.XGBClassifier(**fold_best_params, random_state=RANDOM_SEED)
                best_inner_model.fit(X_outer_train, y_outer_train, verbose=False)
            elif model_name == "Naive Bayes":
                best_inner_model = GaussianNB(** fold_best_params)
                best_inner_model.fit(X_outer_train, y_outer_train)
            else:
                model_class = model.__class__
                best_inner_model = model_class(**fold_best_params, random_state=RANDOM_SEED)
                best_inner_model.fit(X_outer_train, y_outer_train)
            
            # 多指标评估外部验证集
            try:
                y_outer_proba = best_inner_model.predict_proba(X_outer_val)[:, 1]
            except AttributeError:
                decision_scores = best_inner_model.decision_function(X_outer_val)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                y_outer_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
            
            y_outer_pred = best_inner_model.predict(X_outer_val)
            
            # 计算评估指标
            outer_auc = roc_auc_score(y_outer_val, y_outer_proba)
            accuracy = accuracy_score(y_outer_val, y_outer_pred)
            precision = precision_score(y_outer_val, y_outer_pred, zero_division=0)
            recall = recall_score(y_outer_val, y_outer_pred, zero_division=0)
            f1 = f1_score(y_outer_val, y_outer_pred, zero_division=0)
            
            # 计算混淆矩阵和特异性
            try:
                cm = confusion_matrix(y_outer_val, y_outer_pred)
                if cm.shape == (1, 1):
                    tn, fp, fn, tp = cm[0,0], 0, 0, 0
                else:
                    tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            except Exception as e:
                print(f"混淆矩阵计算错误: {e}")
                tn, fp, fn, tp = 0, 0, 0, 0
                specificity = 0
            
            # 收集结果
            outer_validation_results[model_name]['y_true'].extend(y_outer_val)
            outer_validation_results[model_name]['y_pred_proba'].extend(y_outer_proba)
            fold_auc_scores[model_name].append(outer_auc)
            fold_validation_data[model_name].append({
                'y_true': y_outer_val,
                'y_pred_proba': y_outer_proba,
                'fold_idx': fold_idx,
                'auc': outer_auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1
            })
            outer_scores.append(outer_auc)
            
            # 记录折叠详情
            fold_details.append({
                'Model': model_name,
                'Outer_Fold': fold_idx + 1,
                'Inner_Fold_Count': 3,
                'Outer_Val_AUC': outer_auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1': f1,
                'Best_Params': str(fold_best_params)
            })
        
        # 在整个外部循环结束后，根据所有外部折叠的平均表现选择最佳参数
        # 创建参数统计字典，记录每个参数组合在所有折叠中的表现
        param_performance = {}
        
        # 遍历所有折叠，收集参数性能数据
        for idx, detail in enumerate(fold_details):
            params_str = detail['Best_Params']
            if params_str not in param_performance:
                param_performance[params_str] = []
            param_performance[params_str].append(outer_scores[idx])
        
        # 计算每个参数组合的平均AUC和标准差
        param_avg_auc = {}
        for params_str, scores in param_performance.items():
            param_avg_auc[params_str] = np.mean(scores)
        
        # 找到平均AUC最高的参数组合
        best_params_str = max(param_avg_auc, key=param_avg_auc.get)
        best_avg_auc = param_avg_auc[best_params_str]
        
        # 将字符串形式的参数转回字典
        import ast
        try:
            best_fold_params = ast.literal_eval(best_params_str)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse best parameters for {model_name}, using last fold's parameters")
            # 如果解析失败，使用AUC最高的折叠参数作为备选
            best_fold_idx = np.argmax(outer_scores)
            for detail in fold_details:
                if detail['Outer_Fold'] == best_fold_idx + 1:
                    try:
                        best_fold_params = ast.literal_eval(detail['Best_Params'])
                    except (ValueError, SyntaxError):
                        best_fold_params = fold_best_params
                    break
        
        print(f"{model_name} - Selected best parameters with average AUC: {best_avg_auc:.3f}")
        
        # 准备完整训练集
        if model_name == "XGBoost" and X_train_non_scaled is not None:
            X_full_train = X_train_non_scaled
        else:
            X_full_train = X_train
        
        # 重新训练模型
        if model_name == "XGBoost":
            final_model = xgb.XGBClassifier(**best_fold_params, random_state=RANDOM_SEED)
            final_model.fit(X_full_train, y_train, verbose=False)
        elif model_name == "Naive Bayes":
            final_model = GaussianNB(**best_fold_params)
            final_model.fit(X_full_train, y_train)
        else:
            model_class = model.__class__
            final_model = model_class(**best_fold_params, random_state=RANDOM_SEED)
            final_model.fit(X_full_train, y_train)
        
        best_models[model_name] = final_model

        # 计算平均性能（属于model_name循环内部）
        avg_auc = np.mean(outer_scores)
        grid_search_summary.append({
            'Model': model_name,
            'Total_Params': param_count,
            'Mean_Outer_AUC': avg_auc,
            'Std_Outer_AUC': np.std(outer_scores),
            'Best_Params': str(fold_best_params)
        })
        cv_details.extend(fold_details)

        print(f"{model_name} - Mean Outer AUC: {avg_auc:.3f} (±{np.std(outer_scores):.3f})")
        print(f"  Best Parameters: {fold_best_params}")

    # 转换为DataFrame（属于函数内部，不在任何循环内）
    df_cv_details = pd.DataFrame(cv_details)
    df_grid_summary = pd.DataFrame(grid_search_summary)
    
    # 导出结果（属于函数内部）
    if export_dir:
        nested_cv_excel_path = os.path.join(export_dir, 'nested_cv_results.xlsx')
        with pd.ExcelWriter(nested_cv_excel_path, engine='openpyxl') as writer:
            df_cv_details.to_excel(writer, sheet_name='CV_Details', index=False)
            df_grid_summary.to_excel(writer, sheet_name='CV_Summary', index=False)
        print(f"Nested cross-validation results saved: {nested_cv_excel_path}")
        
        # 绘制箱线图
        plt.figure(figsize=(12, 8))
        df_cv_details.boxplot(column='Outer_Val_AUC', by='Model', grid=False)
        plt.title('Nested Cross-Validation AUC Results by Model', fontsize=14)
        plt.suptitle('')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('AUC Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        boxplot_path = os.path.join(export_dir, 'nested_cv_boxplot.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Nested cross-validation boxplot saved: {boxplot_path}")
        
        # 绘制柱状图
        plt.figure(figsize=(12, 8))
        sorted_df = df_grid_summary.sort_values('Mean_Outer_AUC', ascending=False)
        bars = plt.bar(sorted_df['Model'], sorted_df['Mean_Outer_AUC'], yerr=sorted_df['Std_Outer_AUC'],
                      capsize=5, alpha=0.7)
        plt.title('Mean AUC Comparison Across Models (Nested CV)', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Mean AUC Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, auc_val in zip(bars, sorted_df['Mean_Outer_AUC']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{auc_val:.3f}', ha='center', va='bottom')
                      
        plt.tight_layout()
        barplot_path = os.path.join(export_dir, 'nested_cv_barplot.png')
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Nested cross-validation barplot saved: {barplot_path}")

    # 转换为numpy数组
    for model_name in outer_validation_results:
        outer_validation_results[model_name]['y_true'] = np.array(outer_validation_results[model_name]['y_true'])
        outer_validation_results[model_name]['y_pred_proba'] = np.array(outer_validation_results[model_name]['y_pred_proba'])
        
    # 计算中位数AUC数据集
    median_auc_datasets = {}
    for model_name in fold_validation_data:
        sorted_folds = sorted(fold_validation_data[model_name], key=lambda x: x['auc'])
        median_idx = len(sorted_folds) // 2
        median_auc_datasets[model_name] = sorted_folds[median_idx]
        
    return best_models, df_cv_details, df_grid_summary, outer_validation_results, median_auc_datasets

# -------------------------- 4. Retrain Best Models on Full Training Set --------------------------
def retrain_best_models(best_models, X_train, y_train, X_train_non_scaled=None):
    """Retrain best models on the full training set with consistent configurations"""
    print("\n=== Retraining Best Models on Full Training Set ===")
    retrained_models = {}
    
    for model_name, model in best_models.items():
        print(f"Retraining {model_name}...")
        try:
            # Create a new instance with best parameters and fixed random seed
            if model_name == "XGBoost":
                # 创建参数字典，避免random_state重复传递
                model_params = model.get_params()
                model_params['random_state'] = RANDOM_SEED  # 确保使用指定的随机种子
                new_model = xgb.XGBClassifier(**model_params)
            elif model_name == "Naive Bayes":
                # Naive Bayes does not accept random_state parameter
                new_model = GaussianNB(**model.get_params())
            else:
                # 为所有其他模型采用一致的参数设置方法，避免random_state重复传递
                model_params = model.get_params()
                if 'random_state' in model_params:
                    model_params['random_state'] = RANDOM_SEED  # 更新或添加random_state参数
                new_model_class = model.__class__
                new_model = new_model_class(**model_params)
            
            # Train on complete dataset using appropriate data
            if model_name == "Naive Bayes" and X_train_non_scaled is not None:
                new_model.fit(X_train_non_scaled, y_train)
            else:
                new_model.fit(X_train, y_train)
            
            retrained_models[model_name] = new_model
        except Exception as e:
            print(f"Error retraining {model_name}: {str(e)}")
            retrained_models[model_name] = model  # If retraining fails, use the original model
    
    return retrained_models

# -------------------------- 5. Evaluate Models on All Datasets --------------------------
def evaluate_models_on_all_datasets(models, processed_datasets, export_dir=None):
    """Evaluate best models on all datasets (train, test)"""
    print("\n=== Evaluating Models on All Datasets ===")
    
    # Define datasets to evaluate
    datasets_to_evaluate = [
        {'name': 'train', 'X': processed_datasets['X_train_scaled'], 'y': processed_datasets['y_train'],
         'X_non_scaled': processed_datasets['X_train_non_scaled']},
        {'name': 'test', 'X': processed_datasets['X_test_scaled'], 'y': processed_datasets['y_test'],
         'X_non_scaled': processed_datasets['X_test_non_scaled']}
    ]
    
    # Evaluate each model on every dataset
    all_evaluation_results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}:")
        
        for dataset in datasets_to_evaluate:
            dataset_name = dataset['name']
            
            # Use appropriate data for prediction based on model type
            if model_name == "Naive Bayes":
                X_data = dataset['X_non_scaled']
            else:
                X_data = dataset['X']
            
            y_data = dataset['y']
            
            try:
                # 对于SVM模型，添加特殊处理逻辑
                if model_name == "SVM":
                    try:
                        y_pred_proba = model.predict_proba(X_data)[:, 1]
                    except AttributeError:
                        try:
                            # 如果predict_proba失败，使用decision_function并归一化到[0,1]范围
                            decision_scores = model.decision_function(X_data)
                            if len(decision_scores.shape) > 1:
                                decision_scores = decision_scores[:, 0]
                            # 归一化处理
                            y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
                        except Exception as e:
                            print(f"  Error getting prediction scores for {model_name}: {str(e)}")
                            raise
                else:
                    y_pred_proba = model.predict_proba(X_data)[:, 1]
                     
                y_pred = model.predict(X_data)
                
                # Calculate confusion matrix to get specificity
                cm = confusion_matrix(y_data, y_pred)
                tn = cm[0, 0]  # True Negative
                fp = cm[0, 1]  # False Positive
                fn = cm[1, 0]  # False Negative
                tp = cm[1, 1]  # True Positive
                
                # Calculate specificity
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Calculate AIC and AICc values
                n_features = X_data.shape[1]
                n_samples = len(y_data)
                aic, aicc = calculate_aic(y_data, y_pred_proba, n_features, n_samples)
                
                # Calculate evaluation metrics
                metrics = {
                    'Model': model_name,
                    'Dataset': dataset_name.capitalize(),
                    'AUC': roc_auc_score(y_data, y_pred_proba),
                    'Accuracy': accuracy_score(y_data, y_pred),
                    'Precision': precision_score(y_data, y_pred, zero_division=0),
                    'Recall': recall_score(y_data, y_pred, zero_division=0),
                    'Specificity': specificity,
                    'F1': f1_score(y_data, y_pred, zero_division=0),
                    'AIC': aic,
                    'AICc': aicc,
                    'Samples': len(y_data),
                    'Positive_Samples': sum(y_data),
                    'Positive_Ratio': np.mean(y_data),
                    'Feature_Count': n_features
                }
                
                all_evaluation_results.append(metrics)
                
                # Print results for current dataset
                print(f"  {dataset_name.capitalize()} set - AUC: {metrics['AUC']:.4f}, "
                      f"Accuracy: {metrics['Accuracy']:.4f}, AICc: {metrics['AICc']:.2f}")
                
            except Exception as e:
                print(f"  Error evaluating {model_name} on {dataset_name} set: {str(e)}")
    
    # Convert results to DataFrame for output
    df_evaluation = pd.DataFrame(all_evaluation_results)
    
    # Reorder columns for better readability
    column_order = ['Model', 'Dataset', 'AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1',
                    'AIC', 'Samples', 'Positive_Samples', 'Positive_Ratio', 'Feature_Count']
    df_evaluation = df_evaluation[column_order]
    
    # If export directory is provided, save evaluation results to Excel
    if export_dir:
        evaluation_excel_path = os.path.join(export_dir, 'model_evaluation_all_datasets.xlsx')
        with pd.ExcelWriter(evaluation_excel_path, engine='openpyxl') as writer:
            df_evaluation.to_excel(writer, sheet_name='Evaluation_Results', index=False)
            
            # Create pivot table for comparison
            pivot_table = df_evaluation.pivot_table(
                index='Model',
                columns='Dataset',
                values=['AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']
            )
            pivot_table.to_excel(writer, sheet_name='Evaluation_Pivot')
        
        print(f"\nAll dataset model evaluation results saved: {evaluation_excel_path}")
    
    return df_evaluation

# -------------------------- 6. Calculate AIC and AICc Values --------------------------
def calculate_aic(y_true, y_pred_proba, n_features, n_samples=None):
    """Calculate Akaike Information Criterion (AIC) and corrected AIC (AICc) for classification models
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    n_features : int
        Number of features used in the model
    n_samples : int, optional
        Number of samples in the dataset
    
    Returns:
    --------
    tuple
        (aic, aicc) - AIC and AICc values
    """
    # Calculate log likelihood for binary classification
    # Use vectorized operations for better performance
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    
    # Calculate log likelihood using vectorized operations
    ll = np.sum(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    
    # Calculate number of parameters
    k = n_features + 1  # n_features + 1 (intercept) for classification models
    
    # Calculate AIC
    aic = -2 * ll + 2 * k
    
    # Calculate AICc if n_samples is provided
    if n_samples is None:
        # If n_samples is not provided, estimate it from y_true length
        n_samples = len(y_true)
    
    # Boundary check: if n <= k + 1, return NaN for AICc
    if n_samples <= k + 1:
        aicc = float('nan')
        print(f"Warning: Sample size (n={n_samples}) too small for AICc calculation (n <= k+1={k+1})")
    else:
        # Calculate AICc using the formula: AICc = AIC + (2k² + 2k)/(n - k - 1)
        aicc = aic + (2 * k * (k + 1)) / (n_samples - k - 1)
    
    # For large samples (n > k + 40), AIC and AICc are almost identical, return AIC as AICc
    if n_samples > k + 40:
        aicc = aic
    
    return aic, aicc

# -------------------------- 7. Plotting Functions --------------------------

# 修改后的绘图函数：将中文替换为英文
def plot_roc_curve(y_true, y_pred_proba, title, export_path=None):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.3, 1.05])  # 调整纵坐标范围，从0.3开始以更好地展示曲线细节
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}


# 修改后的绘图函数：将中文替换为英文
def plot_pr_curve(y_true, y_pred_proba, title, export_path=None):
    """绘制精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (Area = {pr_auc:.2f})')
    plt.axhline(y=np.mean(y_true), color='r', linestyle='--', label=f'Baseline ({np.mean(y_true):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.3, 1.05])  # 调整纵坐标范围，从0.3开始以更好地展示曲线细节
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {'precision': precision, 'recall': recall, 'auc': pr_auc}


# 修改后的绘图函数：将中文替换为英文
def plot_calibration_curve(y_true, y_pred_proba, title, export_path=None):
    """绘制校准曲线"""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', color='red', lw=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {'prob_true': prob_true, 'prob_pred': prob_pred}


# 修改后的绘图函数：将中文替换为英文
def plot_dca_curve(y_true, y_pred_proba, title, export_path=None):
    """绘制决策曲线分析(DCA)曲线"""
    thresholds = np.arange(0, 1.01, 0.01)
    net_benefit = [calculate_net_benefit(y_true, y_pred_proba, t) for t in thresholds]
    net_benefit_all_patients = [np.mean(y_true) - (1 - np.mean(y_true)) * (t / (1 - t)) if t < 1 else 0 for t in thresholds]
    net_benefit_none = np.zeros_like(thresholds)
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, net_benefit, color='darkorange', lw=2, label='Model')
    plt.plot(thresholds, net_benefit_all_patients, color='blue', lw=2, linestyle='--', label='Treat All')
    plt.plot(thresholds, net_benefit_none, color='black', lw=2, linestyle=':', label='Treat None')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(title)
    plt.ylim(bottom=-0.1, top=0.6)
    plt.legend(loc="lower left")
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {'thresholds': thresholds, 'net_benefit': net_benefit}


# 新增：跨模型曲线整合函数，用于将所有模型的曲线整合在一个图表中
def plot_roc_curve_models(model_results, title, export_path=None, dataset_type='val'):
    """将所有模型的ROC曲线整合在一个图表中"""
    plt.figure(figsize=(12, 10))
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_index = 0
    
    for model_name, result in model_results.items():
        if dataset_type == 'val':
            y_true = result['val_predictions']['true']
            y_pred_proba = result['val_predictions']['all']
        elif dataset_type == 'test':
            y_true = result['test_predictions']['true']
            y_pred_proba = result['test_predictions']['all']
        else:
            raise ValueError("dataset_type must be 'val' or 'test'")
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # 绘制曲线
        plt.plot(fpr, tpr, color=colors[color_index % len(colors)], lw=2, 
                 label=f'{model_name} (Area = {roc_auc:.3f})')
        color_index += 1
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pr_curve_models(model_results, title, export_path=None, dataset_type='val'):
    """将所有模型的PR曲线整合在一个图表中"""
    plt.figure(figsize=(12, 10))
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_index = 0
    
    for model_name, result in model_results.items():
        if dataset_type == 'val':
            y_true = result['val_predictions']['true']
            y_pred_proba = result['val_predictions']['all']
        elif dataset_type == 'test':
            y_true = result['test_predictions']['true']
            y_pred_proba = result['test_predictions']['all']
        else:
            raise ValueError("dataset_type must be 'val' or 'test'")
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # 绘制曲线
        plt.plot(recall, precision, color=colors[color_index % len(colors)], lw=2, 
                 label=f'{model_name} (Area = {pr_auc:.3f})')
        color_index += 1
    
    # 计算所有模型的平均基线
    if dataset_type == 'val':
        all_y_true = np.concatenate([result['val_predictions']['true'] for result in model_results.values()])
    elif dataset_type == 'test':
        all_y_true = np.concatenate([result['test_predictions']['true'] for result in model_results.values()])
    plt.axhline(y=np.mean(all_y_true), color='r', linestyle='--', label=f'Baseline ({np.mean(all_y_true):.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.3, 1.05])  # 调整纵坐标范围，从0.3开始以更好地展示曲线细节
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_curve_models(model_results, title, export_path=None, dataset_type='val'):
    """将所有模型的校准曲线整合在一个图表中"""
    plt.figure(figsize=(12, 10))
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'x', '*', '+', 'p', 'h', '8']
    color_index = 0
    
    for model_name, result in model_results.items():
        if dataset_type == 'val':
            y_true = result['val_predictions']['true']
            y_pred_proba = result['val_predictions']['all']
        elif dataset_type == 'test':
            y_true = result['test_predictions']['true']
            y_pred_proba = result['test_predictions']['all']
        else:
            raise ValueError("dataset_type must be 'val' or 'test'")
        
        # 计算校准曲线
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        # 绘制曲线
        plt.plot(prob_pred, prob_true, marker=markers[color_index % len(markers)], 
                 color=colors[color_index % len(colors)], lw=2, label=model_name)
        color_index += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dca_curve_models(model_results, title, export_path=None, dataset_type='val'):
    """将所有模型的DCA曲线整合在一个图表中"""
    plt.figure(figsize=(12, 10))
    thresholds = np.arange(0, 1.01, 0.01)
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_index = 0
    
    # 计算所有模型的平均y_true值，用于绘制基线
    if dataset_type == 'val':
        all_y_true = np.concatenate([result['val_predictions']['true'] for result in model_results.values()])
    elif dataset_type == 'test':
        all_y_true = np.concatenate([result['test_predictions']['true'] for result in model_results.values()])
    else:
        raise ValueError("dataset_type must be 'val' or 'test'")
    mean_y_true = np.mean(all_y_true)
    
    for model_name, result in model_results.items():
        if dataset_type == 'val':
            y_true = result['val_predictions']['true']
            y_pred_proba = result['val_predictions']['all']
        elif dataset_type == 'test':
            y_true = result['test_predictions']['true']
            y_pred_proba = result['test_predictions']['all']
        else:
            raise ValueError("dataset_type must be 'val' or 'test'")
        
        # 计算DCA曲线
        net_benefit = [calculate_net_benefit(y_true, y_pred_proba, t) for t in thresholds]
        
        # 绘制曲线
        plt.plot(thresholds, net_benefit, color=colors[color_index % len(colors)], lw=2, 
                 label=model_name)
        color_index += 1
    
    # 绘制基线
    net_benefit_all_patients = [mean_y_true - (1 - mean_y_true) * (t / (1 - t)) if t < 1 else 0 for t in thresholds]
    net_benefit_none = np.zeros_like(thresholds)
    
    plt.plot(thresholds, net_benefit_all_patients, color='blue', lw=2, linestyle='--', label='Treat All')
    plt.plot(thresholds, net_benefit_none, color='black', lw=2, linestyle=':', label='Treat None')
    
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(title)
    plt.ylim(bottom=-0.1, top=0.6)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_net_benefit(y_true, y_pred_proba, threshold):
    """计算给定阈值下的净收益"""
    y_pred = y_pred_proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    if threshold == 1:  # 避免除以零
        return (tp / n) if n > 0 else 0
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit


def calculate_optimal_threshold_with_ci(best_model_name, best_model, processed_datasets, export_dir=None):

    print(f"\n=== Calculating Optimal Threshold with Confidence Intervals for {best_model_name} ===")
    
    # 创建导出目录
    if export_dir:
        optimal_threshold_dir = os.path.join(export_dir, 'Optimal_Threshold_Analysis')
        os.makedirs(optimal_threshold_dir, exist_ok=True)
    else:
        optimal_threshold_dir = None
    
    # 获取测试集数据
    if best_model_name == "Naive Bayes":
        X_test = processed_datasets['X_test_non_scaled']
    else:
        X_test = processed_datasets['X_test_scaled']
    y_test = processed_datasets['y_test']
    
    # 获取预测概率
    try:
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # 处理SVM等可能不支持predict_proba的情况
        try:
            if hasattr(best_model, 'decision_function'):
                decision_values = best_model.decision_function(X_test)
                if len(decision_values.shape) > 1:
                    decision_values = decision_values[:, 0]
                # 使用sigmoid函数将决策值转换为概率
                y_pred_proba = 1 / (1 + np.exp(-decision_values))
            else:
                print(f"Warning: {best_model_name} has neither predict_proba nor decision_function. Using default probabilities.")
                y_pred_proba = np.zeros(len(X_test)) + 0.5
        except Exception as e:
            print(f"Error in getting prediction probabilities: {str(e)}")
            y_pred_proba = np.zeros(len(X_test)) + 0.5
    
    # 计算不同阈值的净获益值
    thresholds = np.linspace(0, 1, 101)
    net_benefits = [calculate_net_benefit(y_test, y_pred_proba, t) for t in thresholds]
    
    # 计算treat all曲线的净获益值
    positive_rate = np.mean(y_test)
    treat_all_net_benefits = [positive_rate - (1 - positive_rate) * (t / (1 - t)) if t < 1 else 0 for t in thresholds]
    
    # 定位净获益峰值对应的最优阈值
    max_nb_idx = np.argmax(net_benefits)
    optimal_threshold = thresholds[max_nb_idx]
    max_net_benefit = net_benefits[max_nb_idx]
    
    # 使用自助法计算置信区间
    print("Calculating 95% confidence intervals using bootstrap...")
    n_bootstrap = 1000
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        if i % 200 == 0:
            print(f"  Bootstrap iteration {i}/{n_bootstrap}")
        
        # 有放回地采样
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_boot = y_test.iloc[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        
        # 计算每个阈值的净获益
        try:
            nb_boot = [calculate_net_benefit(y_boot, y_pred_proba_boot, t) for t in thresholds]
            bootstrap_results.append(nb_boot)
        except Exception as e:
            # 处理可能的异常
            continue
    
    # 确保有足够的bootstrap结果
    if len(bootstrap_results) < 10:
        print("Warning: Not enough valid bootstrap results. Using simplified confidence interval calculation.")
        # 使用简化方法计算置信区间
        ci_lower = np.maximum(-0.1, max_net_benefit - 1.96 * np.std(net_benefits))
        ci_upper = np.minimum(0.6, max_net_benefit + 1.96 * np.std(net_benefits))
        threshold_ci = {
            'lower': max(0, optimal_threshold - 0.05),
            'upper': min(1, optimal_threshold + 0.05)
        }
    else:
        # 计算每个阈值的置信区间
        bootstrap_results = np.array(bootstrap_results)
        ci_lower = np.percentile(bootstrap_results, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_results, 97.5, axis=0)
        
        # 计算净获益峰值的置信区间
        max_nb_indices = np.argmax(bootstrap_results, axis=1)
        max_thresholds = thresholds[max_nb_indices]
        threshold_ci = {
            'lower': np.percentile(max_thresholds, 2.5),
            'upper': np.percentile(max_thresholds, 97.5)
        }
        
        # 计算置信区间宽度并选择宽度较窄的阈值
        ci_width = ci_upper - ci_lower
        # 找出净获益在前10%且置信区间宽度最小的阈值
        top_10_percent_threshold = sorted(list(zip(net_benefits, ci_width, thresholds)), 
                                          key=lambda x: (-x[0], x[1]))[:10]
        
        # 如果有多个良好的阈值选项，选择置信区间最窄的
        if len(top_10_percent_threshold) > 1:
            best_threshold_by_ci = min(top_10_percent_threshold, key=lambda x: x[1])
            if best_threshold_by_ci[2] != optimal_threshold:
                print(f"Selecting threshold with narrower CI: {best_threshold_by_ci[2]:.3f} (vs {optimal_threshold:.3f})")
                optimal_threshold = best_threshold_by_ci[2]
                opt_idx = np.argmin(np.abs(thresholds - optimal_threshold))
                max_net_benefit = net_benefits[opt_idx]
    
    # 结果可视化
    print("Generating visualization plots...")
    
    # 1. 带有置信区间的DCA曲线
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, net_benefits, color='darkorange', lw=2, label='Model Net Benefit')
    if 'ci_lower' in locals() and 'ci_upper' in locals():
        plt.fill_between(thresholds, ci_lower, ci_upper, color='orange', alpha=0.2, label='95% Confidence Interval')
    
    # 绘制基线
    net_benefit_all_patients = [np.mean(y_test) - (1 - np.mean(y_test)) * (t / (1 - t)) if t < 1 else 0 for t in thresholds]
    net_benefit_none = np.zeros_like(thresholds)
    
    plt.plot(thresholds, net_benefit_all_patients, color='blue', lw=2, linestyle='--', label='Treat All')
    plt.plot(thresholds, net_benefit_none, color='black', lw=2, linestyle=':', label='Treat None')
    
    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    plt.title(f'{best_model_name} - DCA Curve with 95% CI', fontsize=16)
    plt.ylim(bottom=-0.1, top=0.6)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if optimal_threshold_dir:
        dca_with_ci_path = os.path.join(optimal_threshold_dir, f'{best_model_name}_dca_with_ci.png')
        plt.savefig(dca_with_ci_path, dpi=300, bbox_inches='tight')
        print(f"DCA curve with CI saved to: {dca_with_ci_path}")
    plt.close()
    
    # 2. 阈值与净获益关系图（放大版）
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, net_benefits, color='darkorange', lw=2)
    if 'ci_lower' in locals() and 'ci_upper' in locals():
        plt.fill_between(thresholds, ci_lower, ci_upper, color='orange', alpha=0.2)
    
    # 添加置信区间标注（如果有）
    if 'threshold_ci' in locals():
        plt.hlines(y=max_net_benefit, xmin=threshold_ci['lower'], xmax=threshold_ci['upper'], 
                   color='green', linestyle='--', lw=2, label=f'Threshold 95% CI: [{threshold_ci["lower"]:.3f}, {threshold_ci["upper"]:.3f}]')
    
    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    plt.title(f'{best_model_name} - Threshold vs Net Benefit', fontsize=16)
    # 调整y轴范围以突出显示最大值区域
    y_min = max(-0.1, max_net_benefit - 0.2)
    y_max = min(0.6, max_net_benefit + 0.1)
    plt.ylim(bottom=y_min, top=y_max)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if optimal_threshold_dir:
        threshold_vs_nb_path = os.path.join(optimal_threshold_dir, f'{best_model_name}_threshold_vs_net_benefit.png')
        plt.savefig(threshold_vs_nb_path, dpi=300, bbox_inches='tight')
        print(f"Threshold vs Net Benefit plot saved to: {threshold_vs_nb_path}")
    plt.close()
    
    # 3. 置信区间宽度与阈值关系图
    if 'ci_width' in locals():
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, ci_width, color='purple', lw=2)
        plt.xlabel('Threshold Probability', fontsize=14)
        plt.ylabel('95% Confidence Interval Width', fontsize=14)
        plt.title(f'{best_model_name} - CI Width vs Threshold', fontsize=16)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if optimal_threshold_dir:
            ci_width_path = os.path.join(optimal_threshold_dir, f'{best_model_name}_ci_width_vs_threshold.png')
            plt.savefig(ci_width_path, dpi=300, bbox_inches='tight')
            print(f"CI Width vs Threshold plot saved to: {ci_width_path}")
        plt.close()
    
    # 保存结果到Excel
    if optimal_threshold_dir:
        results_df = pd.DataFrame({
            'Threshold': thresholds,
            'Net_Benefit': net_benefits
        })
        
        if 'ci_lower' in locals() and 'ci_upper' in locals():
            results_df['CI_Lower'] = ci_lower
            results_df['CI_Upper'] = ci_upper
            results_df['CI_Width'] = ci_upper - ci_lower
        
        excel_path = os.path.join(optimal_threshold_dir, f'{best_model_name}_threshold_analysis_results.xlsx')
        results_df.to_excel(excel_path, index=False)
        print(f"Threshold analysis results saved to: {excel_path}")
        
        # 保存最优阈值信息
        optimal_info = {
            'Model': best_model_name,
            'Optimal_Threshold': optimal_threshold,
            'Max_Net_Benefit': max_net_benefit,
            'Threshold_95CI_Lower': threshold_ci['lower'] if 'threshold_ci' in locals() else None,
            'Threshold_95CI_Upper': threshold_ci['upper'] if 'threshold_ci' in locals() else None,
            'Positive_Rate': np.mean(y_test),
            'Sample_Size': len(y_test)
        }
        
        optimal_info_df = pd.DataFrame([optimal_info])
        optimal_info_path = os.path.join(optimal_threshold_dir, f'{best_model_name}_optimal_threshold_info.xlsx')
        optimal_info_df.to_excel(optimal_info_path, index=False)
        print(f"Optimal threshold information saved to: {optimal_info_path}")
    
    print(f"\nOptimal threshold analysis completed for {best_model_name}:")
    print(f"- Optimal threshold: {optimal_threshold:.3f}")
    print(f"- Maximum net benefit: {max_net_benefit:.4f}")
    if 'threshold_ci' in locals():
        print(f"- 95% confidence interval for threshold: [{threshold_ci['lower']:.3f}, {threshold_ci['upper']:.3f}]")
    
    # 计算净获益有效范围（考虑两条参考线并确保区间内所有阈值净获益持续稳定高于参考线）
    net_benefits_array = np.array(net_benefits)
    treat_all_array = np.array(treat_all_net_benefits)
    treat_none_array = np.zeros_like(thresholds)
    
    # 找出模型净获益同时大于treat all和treat none曲线的阈值
    # 优先使用置信区间下限进行评估
    if 'ci_lower' in locals():
        # 使用置信区间下限来评估有效范围
        ci_lower_array = np.array(ci_lower)
        better_than_both = (ci_lower_array > treat_all_array) & (ci_lower_array > treat_none_array)
    else:
        # 如果没有置信区间数据，回退到使用净获益值
        better_than_both = (net_benefits_array > treat_all_array) & (net_benefits_array > treat_none_array)
    
    # 计算连续有效区间 - 确保区间内所有阈值都满足条件
    if np.any(better_than_both):
        # 找到所有满足条件的阈值点
        valid_indices = np.where(better_than_both)[0]
        
        if len(valid_indices) > 0:
            # 计算连续区间
            # 首先将阈值点分组为连续的区间
            continuous_intervals = []
            start_idx = 0  # 使用valid_indices的索引位置，而不是值
            
            for i in range(1, len(valid_indices)):
                if valid_indices[i] != valid_indices[i-1] + 1:
                    # 区间中断，记录前一个连续区间
                    continuous_intervals.append((valid_indices[start_idx], valid_indices[i-1]))
                    start_idx = i
            # 添加最后一个区间
            continuous_intervals.append((valid_indices[start_idx], valid_indices[-1]))
            
            # 计算每个连续区间的净获益平均值
            interval_scores = []
            for start, end in continuous_intervals:
                if 'ci_lower' in locals():
                    # 使用置信区间下限计算平均值
                    ci_lower_array = np.array(ci_lower)
                    interval_ci_lower = ci_lower_array[start:end+1]
                    avg_ci_lower = np.mean(interval_ci_lower)
                    # 检查区间内所有点的置信区间下限是否也大于参考线
                    lower_better_than_both = (ci_lower_array[start:end+1] > treat_all_array[start:end+1]) & \
                                            (ci_lower_array[start:end+1] > treat_none_array[start:end+1])
                    # 优先考虑置信区间下限的性能
                    avg_nb = avg_ci_lower
                    # 如果区间内所有点的置信区间下限都满足条件，给予额外分数
                    if np.all(lower_better_than_both):
                        avg_nb += 0.01  # 加分，优先选择置信区间也满足条件的区间
                else:
                    # 如果没有置信区间数据，回退到使用净获益值
                    interval_nb = net_benefits_array[start:end+1]
                    avg_nb = np.mean(interval_nb)
                interval_scores.append((avg_nb, start, end))
            
            # 选择净获益最高的连续区间
            interval_scores.sort(reverse=True, key=lambda x: x[0])
            best_avg_nb, best_start_idx, best_end_idx = interval_scores[0]
            
            # 计算最佳连续区间的阈值范围
            valid_threshold_range = {
                'lower': thresholds[best_start_idx],
                'upper': thresholds[best_end_idx]
            }
            
            # 统计有效范围内的阈值数量
            valid_threshold_count = best_end_idx - best_start_idx + 1
            valid_threshold_percentage = (valid_threshold_count / len(thresholds)) * 100
        else:
            valid_threshold_range = None
            valid_threshold_count = 0
            valid_threshold_percentage = 0
    else:
        valid_threshold_range = None
        valid_threshold_count = 0
        valid_threshold_percentage = 0
    
    # 打印有效阈值范围信息
    if valid_threshold_range:
        print(f"\nNet Benefit Effective Range (model > both treat all and treat none):")
        print(f"- Continuous threshold range: [{valid_threshold_range['lower']:.3f}, {valid_threshold_range['upper']:.3f}]")
        print(f"- Number of valid thresholds in range: {valid_threshold_count}/{len(thresholds)}")
        print(f"- Percentage of valid thresholds: {valid_threshold_percentage:.1f}%")
        
        # 计算区间内的平均净获益
        range_indices = np.where((thresholds >= valid_threshold_range['lower']) & 
                                (thresholds <= valid_threshold_range['upper']))[0]
        if len(range_indices) > 0:
            avg_net_benefit_in_range = np.mean(net_benefits_array[range_indices])
            print(f"- Average net benefit in range: {avg_net_benefit_in_range:.4f}")
            
            # 检查置信区间情况（如果有）
            if 'ci_lower' in locals() and 'ci_upper' in locals():
                ci_lower_array = np.array(ci_lower)
                # 检查区间内所有点的置信区间下限是否也大于参考线
                lower_better_than_both = (ci_lower_array[range_indices] > treat_all_array[range_indices]) & \
                                        (ci_lower_array[range_indices] > treat_none_array[range_indices])
                if np.all(lower_better_than_both):
                    print(f"- All thresholds in range have CI lower bound > both reference lines (highly reliable)")
                else:
                    partially_reliable = np.sum(lower_better_than_both) / len(range_indices) * 100
                    print(f"- {partially_reliable:.1f}% of thresholds in range have CI lower bound > both reference lines")
    else:
        print("\nNo continuous threshold range where model net benefit exceeds both treat all and treat none net benefits.")
    
    # 更新最优阈值信息，添加有效范围信息
    if optimal_threshold_dir and 'optimal_info' in locals():
        # 计算区间内的平均净获益
        avg_net_benefit_in_range = None
        if valid_threshold_range:
            range_indices = np.where((thresholds >= valid_threshold_range['lower']) & 
                                    (thresholds <= valid_threshold_range['upper']))[0]
            if len(range_indices) > 0:
                avg_net_benefit_in_range = np.mean(net_benefits_array[range_indices])
        
        # 检查置信区间可靠性
        ci_reliability_percentage = None
        if 'ci_lower' in locals() and 'ci_upper' in locals() and valid_threshold_range:
            range_indices = np.where((thresholds >= valid_threshold_range['lower']) & 
                                    (thresholds <= valid_threshold_range['upper']))[0]
            if len(range_indices) > 0:
                ci_lower_array = np.array(ci_lower)
                lower_better_than_both = (ci_lower_array[range_indices] > treat_all_array[range_indices]) & \
                                        (ci_lower_array[range_indices] > treat_none_array[range_indices])
                ci_reliability_percentage = np.sum(lower_better_than_both) / len(range_indices) * 100
        
        optimal_info.update({
            'Valid_Threshold_Range_Lower': valid_threshold_range['lower'] if valid_threshold_range else None,
            'Valid_Threshold_Range_Upper': valid_threshold_range['upper'] if valid_threshold_range else None,
            'Valid_Threshold_Count': valid_threshold_count,
            'Valid_Threshold_Percentage': valid_threshold_percentage,
            'Avg_Net_Benefit_in_Range': avg_net_benefit_in_range,
            'CI_Reliability_Percentage': ci_reliability_percentage,
            'Reference_Lines_Considered': 'Both Treat All and Treat None'
        })
        
        # 重新保存更新后的最优阈值信息
        optimal_info_df = pd.DataFrame([optimal_info])
        optimal_info_path = os.path.join(optimal_threshold_dir, f'{best_model_name}_optimal_threshold_info.xlsx')
        optimal_info_df.to_excel(optimal_info_path, index=False)
    
    # 更新Excel结果表，添加两条参考线数据和连续有效区间标记
    if optimal_threshold_dir and 'results_df' in locals():
        results_df['Treat_All_Net_Benefit'] = treat_all_net_benefits
        results_df['Treat_None_Net_Benefit'] = treat_none_array
        results_df['Better_Than_Treat_All'] = net_benefits_array > treat_all_array
        results_df['Better_Than_Treat_None'] = net_benefits_array > treat_none_array
        results_df['Better_Than_Both'] = better_than_both
        
        # 添加是否在连续有效区间内的标记
        in_valid_range = np.zeros_like(thresholds, dtype=bool)
        if valid_threshold_range:
            range_mask = (thresholds >= valid_threshold_range['lower']) & \
                         (thresholds <= valid_threshold_range['upper'])
            in_valid_range[range_mask] = True
        results_df['In_Valid_Continuous_Range'] = in_valid_range
        
        results_df.to_excel(excel_path, index=False)
    
    # 返回结果
    return {
        'optimal_threshold': optimal_threshold,
        'max_net_benefit': max_net_benefit,
        'threshold_ci': threshold_ci if 'threshold_ci' in locals() else None,
        'thresholds': thresholds,
        'net_benefits': net_benefits,
        'treat_all_net_benefits': treat_all_net_benefits,
        'valid_threshold_range': valid_threshold_range,
        'valid_threshold_count': valid_threshold_count,
        'valid_threshold_percentage': valid_threshold_percentage,
        'ci_lower': ci_lower if 'ci_lower' in locals() else None,
        'ci_upper': ci_upper if 'ci_upper' in locals() else None
    }


def evaluate_all_features(processed_datasets, best_models, median_auc_datasets, export_dir=None):
    """使用全部特征评估模型性能并绘制相关图表"""
    print("\n=== Evaluating Models with All Features ===")
    
    # 为每个模型创建评估结果
    evaluation_results = {}
    
    for model_name in best_models.keys():
        print(f"\nProcessing {model_name}...")
        
        # 获取训练数据
        if model_name == "Naive Bayes":
            X_train = processed_datasets['X_train_non_scaled']
        else:
            X_train = processed_datasets['X_train_scaled']
        y_train = processed_datasets['y_train']
        
        # 获取中位数AUC的外部验证结果
        median_dataset = median_auc_datasets[model_name]
        y_true_val = median_dataset['y_true']
        y_pred_proba_all = median_dataset['y_pred_proba']
        print(f"  Using median AUC dataset (Fold {median_dataset['fold_idx']+1}, AUC: {median_dataset['auc']:.4f})")
        
        # 获取交叉验证的验证集索引
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        val_indices_list = [val_idx for _, val_idx in skf.split(X_train, y_train)]
        val_indices = val_indices_list[median_dataset['fold_idx']]  # 使用中位数AUC对应的折叠索引
        
        # 使用最佳参数训练模型（全部特征）
        # 根据不同模型类型使用不同的数据处理方式
        if model_name == "XGBoost":
            # XGBoost模型使用标准化数据
            model_params = best_models[model_name].get_params()
            model_params['random_state'] = RANDOM_SEED
            final_model = xgb.XGBClassifier(**model_params)
            final_model.fit(processed_datasets['X_train_scaled'], y_train, verbose=False)
    
        elif model_name == "Naive Bayes":
            # 朴素贝叶斯模型使用非标准化数据
            final_model = GaussianNB(**best_models[model_name].get_params())
            final_model.fit(processed_datasets['X_train_non_scaled'], y_train)

        elif model_name == "SVM":
            # SVM模型需要特别设置probability=True以支持predict_proba
            model_class = best_models[model_name].__class__
            model_params = best_models[model_name].get_params()
            model_params['random_state'] = RANDOM_SEED
            model_params['probability'] = True  # 确保启用概率预测
            final_model = model_class(**model_params)
            final_model.fit(processed_datasets['X_train_scaled'], y_train)

        else:
            # 其他所有模型使用标准化数据
            model_class = best_models[model_name].__class__
            model_params = best_models[model_name].get_params()
            model_params['random_state'] = RANDOM_SEED
            final_model = model_class(**model_params)
            final_model.fit(processed_datasets['X_train_scaled'], y_train)
        
        # 创建导出目录
        if export_dir:
            model_export_dir = os.path.join(export_dir, model_name)
            os.makedirs(model_export_dir, exist_ok=True)
        else:
            model_export_dir = None
        
        # 绘制ROC曲线
        plot_roc_curve(
            y_true_val, y_pred_proba_all,
            f'{model_name} - ROC Curve (All Features)',
            os.path.join(model_export_dir, 'roc_all_features.png') if model_export_dir else None
        )
        
        # 绘制PR曲线
        plot_pr_curve(
            y_true_val, y_pred_proba_all,
            f'{model_name} - PR Curve (All Features)',
            os.path.join(model_export_dir, 'pr_all_features.png') if model_export_dir else None
        )
        
        # 绘制校准曲线
        plot_calibration_curve(
            y_true_val, y_pred_proba_all,
            f'{model_name} - Calibration Curve (All Features)',
            os.path.join(model_export_dir, 'calibration_all_features.png') if model_export_dir else None
        )
        
        # 绘制DCA曲线
        plot_dca_curve(
            y_true_val, y_pred_proba_all,
            f'{model_name} - DCA Curve (All Features)',
            os.path.join(model_export_dir, 'dca_all_features.png') if model_export_dir else None
        )
        
        # 在测试集上进行预测
        if model_name == "Naive Bayes":
            X_test = processed_datasets['X_test_non_scaled']
        else:
            X_test = processed_datasets['X_test_scaled']
        y_test = processed_datasets['y_test']
        
        # 获取测试集预测概率 - 添加健壮的错误处理
        try:
            y_pred_proba_test = final_model.predict_proba(X_test)[:, 1]
        except AttributeError:
            # 处理SVM等可能不支持predict_proba的情况
            try:
                # 尝试使用decision_function并转换为概率
                if hasattr(final_model, 'decision_function'):
                    decision_values = final_model.decision_function(X_test)
                    if len(decision_values.shape) > 1:
                        decision_values = decision_values[:, 0]
                    # 使用sigmoid函数将决策值转换为概率
                    y_pred_proba_test = 1 / (1 + np.exp(-decision_values))
                else:
                    # 如果两者都不可用，返回默认概率
                    print(f"Warning: {model_name} has neither predict_proba nor decision_function. Using default probabilities.")
                    y_pred_proba_test = np.zeros(len(X_test)) + 0.5
            except Exception as e:
                print(f"Error in getting prediction probabilities for {model_name}: {str(e)}")
                y_pred_proba_test = np.zeros(len(X_test)) + 0.5
        
        # 绘制测试集的ROC曲线
        plot_roc_curve(
            y_test, y_pred_proba_test,
            f'{model_name} - ROC Curve (Test Set)',
            os.path.join(model_export_dir, 'test_roc_curve.png') if model_export_dir else None
        )
        
        # 绘制测试集的PR曲线
        plot_pr_curve(
            y_test, y_pred_proba_test,
            f'{model_name} - PR Curve (Test Set)',
            os.path.join(model_export_dir, 'test_pr_curve.png') if model_export_dir else None
        )
        
        # 绘制测试集的校准曲线
        plot_calibration_curve(
            y_test, y_pred_proba_test,
            f'{model_name} - Calibration Curve (Test Set)',
            os.path.join(model_export_dir, 'test_calibration_curve.png') if model_export_dir else None
        )
        
        # 绘制测试集的DCA曲线
        plot_dca_curve(
            y_test, y_pred_proba_test,
            f'{model_name} - DCA Curve (Test Set)',
            os.path.join(model_export_dir, 'test_dca_curve.png') if model_export_dir else None
        )
        
        # 保存评估结果
        evaluation_results[model_name] = {
            'all_features': X_train.columns.tolist(),
            'all_model': final_model,
            'val_predictions': {
                'all': y_pred_proba_all,
                'true': y_true_val
            },
            'test_predictions': {
                'all': y_pred_proba_test,
                'true': y_test
            }
        }
    
    print("\nAll features evaluation completed.")
    return evaluation_results

# -------------------------- 7. SHAP Analysis --------------------------
def perform_shap_analysis(best_model_name, best_model, X_train, X_test, y_test, X_test_non_scaled=None, export_dir=None):
    """对最佳模型执行增强的SHAP分析"""
    print(f"\n=== Performing Enhanced SHAP Analysis on {best_model_name} ===")
    
    # 确定用于SHAP分析的数据
    if best_model_name == "Naive Bayes" and X_test_non_scaled is not None:
        X_shap_test = X_test_non_scaled
        X_shap_train = X_train  # For LinearExplainer, use scaled data
        X_shap_test_display = X_test  # For better visualization of scaled features
    else:
        X_shap_test = X_test
        X_shap_train = X_train
        X_shap_test_display = X_test
    
    try:
        # 根据模型类型创建适当的SHAP解释器
        if best_model_name in ['Random Forest', 'XGBoost']:
            explainer = shap.TreeExplainer(best_model)
        elif best_model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(best_model, X_shap_train)
        elif best_model_name == 'SVM':
            # 确保SVM模型正确支持predict_proba
            sample_size = min(100, len(X_shap_train))
            # 使用partial函数确保返回概率的第二列
            def predict_proba_wrapper(X):
                return best_model.predict_proba(X)[:, 1:2]  # 返回二维数组
            explainer = shap.KernelExplainer(
                predict_proba_wrapper, 
                X_shap_train.sample(sample_size, random_state=RANDOM_SEED)
            )
        elif best_model_name in ['Naive Bayes', 'Decision Tree', 'Neural Network']:
            # 使用样本提高速度
            sample_size = min(100, len(X_shap_train))
            explainer = shap.KernelExplainer(
                best_model.predict_proba, 
                X_shap_train.sample(sample_size, random_state=RANDOM_SEED)
            )
        else:
            print(f"No SHAP explainer available for {best_model_name}")
            return None
        
        if explainer is not None:
            # 计算SHAP值
            shap_values = explainer.shap_values(X_shap_test)
            
            # 处理不同的SHAP值格式
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            elif shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # For certain models
            
            # SHAP摘要图（蜂群图）
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap_test_display, feature_names=X_shap_test_display.columns, show=False)
            plt.title(f'{best_model_name} SHAP Summary Plot')
            plot_path = os.path.join(export_dir, f'{best_model_name}_SHAP_Summary_Plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved: {plot_path}")
            
            # 计算预测以选择代表性样本
            if best_model_name == "Naive Bayes" and X_test_non_scaled is not None:
                y_pred_proba = best_model.predict_proba(X_test_non_scaled)[:, 1]
            else:
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Generate SHAP waterfall plots for multiple representative samples
            if len(X_shap_test) > 0:
                # 获取基值
                if np.isscalar(explainer.expected_value):
                    base_value = explainer.expected_value
                else:
                    base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                
                # 选择3个代表性样本：最高概率、最低概率和中等概率
                sample_indices = []
                sample_types = []
                
                # 最高概率样本
                if len(y_pred_proba) > 0:
                    highest_prob_idx = y_pred_proba.argsort()[-1]
                    sample_indices.append(highest_prob_idx)
                    sample_types.append('highest_prob')
                
                # 最低概率样本
                if len(y_pred_proba) > 1:
                    lowest_prob_idx = y_pred_proba.argsort()[0]
                    if lowest_prob_idx != highest_prob_idx:
                        sample_indices.append(lowest_prob_idx)
                        sample_types.append('lowest_prob')
                
                # 中等概率样本
                if len(y_pred_proba) > 2:
                    middle_idx = len(y_pred_proba) // 2
                    if middle_idx not in sample_indices:
                        sample_indices.append(middle_idx)
                        sample_types.append('middle_prob')
                
                # Ensure we have at least one sample
                if not sample_indices:
                    sample_indices = [0]
                    sample_types = ['first']
                
                # 为选定的样本生成瀑布图
                for i, (sample_index, sample_type) in enumerate(zip(sample_indices, sample_types)):
                    try:
                        shap_single_sample = shap_values[sample_index]
                        
                        # 创建SHAP解释对象
                        shap_exp = shap.Explanation(
                            values=shap_single_sample,
                            base_values=base_value,
                            data=X_shap_test_display.iloc[sample_index],
                            feature_names=X_shap_test_display.columns.tolist()
                        )
                        
                        # Plot waterfall plot with improved settings
                        plt.figure(figsize=(14, 10))
                        # 在瀑布图中显示更多特征
                        shap.plots.waterfall(shap_exp, max_display=15, show=False)
                        
                        # Add detailed title with sample information
                        if sample_type == 'highest_prob':
                            title = f'{best_model_name} SHAP Waterfall Plot - Highest Probability Sample'
                        elif sample_type == 'lowest_prob':
                            title = f'{best_model_name} SHAP Waterfall Plot - Lowest Probability Sample'
                        elif sample_type == 'middle_prob':
                            title = f'{best_model_name} SHAP Waterfall Plot - Medium Probability Sample'
                        else:
                            title = f'{best_model_name} SHAP Waterfall Plot (Sample {sample_index})'
                        
                        # 在标题中添加预测概率
                        pred_prob = y_pred_proba[sample_index]
                        true_label = y_test.iloc[sample_index]
                        # 将数值标签映射到其实际含义
                        label_meaning = 'Serologically Cured' if true_label == 0 else 'Currently Infected'  # 0=Cured, 1=Infected
                        plt.title(f"{title}\nSample Index: {sample_index}, Prediction Probability: {pred_prob:.4f}, True Label: {true_label} ({label_meaning})", fontsize=14)
                        
                        # 保存图像
                        plot_path = os.path.join(export_dir, f'{best_model_name}_SHAP_Waterfall_Plot_{sample_type}.png')
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Enhanced SHAP waterfall plot saved: {plot_path}")
                    except Exception as e:
                        print(f"Error generating waterfall plot for sample {sample_index}: {str(e)}")
            
            # 带有蜂群图和条形图的组合SHAP图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap_test_display, feature_names=X_shap_test_display.columns, plot_type="dot", show=False, color_bar=True)
            plt.gca().set_position([0.5, 0.5, 0.65, 0.65])
            ax1 = plt.gca()
            ax2 = ax1.twiny()
            shap.summary_plot(shap_values, X_shap_test_display, plot_type="bar", show=False)
            plt.gca().set_position([0.5, 0.5, 0.65, 0.65])
            
            # 自定义样式
            ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)
            bars = ax2.patches  # 获取所有条形对象
            for bar in bars:
                bar.set_alpha(0.2)
            
            ax1.set_xlabel('Shapley Value Contribution (Beeswarm Plot)', fontsize=10)
            ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=10)
            ax2.xaxis.set_label_position('top')  # 将标签移到顶部
            ax2.xaxis.tick_top()  # Move ticks to the top
            ax1.set_ylabel('Features', fontsize=10)
            plt.tight_layout()
            
            plot_path = os.path.join(export_dir, f"{best_model_name}_SHAP_combined.pdf")
            plt.savefig(plot_path, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Combined SHAP plot saved: {plot_path}")
        
        # Calculate feature importance based on SHAP values
        if isinstance(shap_values, np.ndarray):
            feature_importance = np.abs(shap_values).mean(0)
            feature_importance_df = pd.DataFrame({
                'Feature': X_shap_test.columns,
                'SHAP_Importance': feature_importance,
                'Rank': range(1, len(feature_importance) + 1)  # 添加排名列
            }).sort_values('SHAP_Importance', ascending=False)
            
            # 将特征重要性保存到Excel
            excel_path = os.path.join(export_dir, f'{best_model_name}_feature_importance.xlsx')
            feature_importance_df.to_excel(excel_path, index=False)
            print(f"Enhanced feature importance saved to: {excel_path}")
            
            return feature_importance_df
        
    except Exception as e:
        print(f"Error in SHAP analysis for {best_model_name}: {str(e)}")
        # 对于SHAP不能很好工作的模型（如Naive Bayes），使用替代重要性度量
        if best_model_name == "Naive Bayes":
            # 使用变异系数作为替代重要性度量
            if hasattr(best_model, 'theta_'):
                # 对于GaussianNB
                mean_vals = best_model.theta_[1]  # Mean for positive class
                var_vals = best_model.sigma_  # Variance
                cv_vals = np.sqrt(var_vals) / mean_vals if len(mean_vals) > 0 and np.any(mean_vals) else np.zeros_like(var_vals)
                cv_vals = np.nan_to_num(cv_vals, nan=0.0, posinf=0.0, neginf=0.0)
                
                feature_importance_df = pd.DataFrame({
                    'Feature': X_shap_test.columns,
                    'SHAP_Importance': cv_vals,
                    'Rank': range(1, len(cv_vals) + 1)  # 添加排名列
                }).sort_values('SHAP_Importance', ascending=False)
                
                # 将特征重要性保存到Excel
                excel_path = os.path.join(export_dir, f'{best_model_name}_feature_importance.xlsx')
                feature_importance_df.to_excel(excel_path, index=False)
                print(f"Naive Bayes feature importance (based on coefficient of variation) saved: {excel_path}")
                
                return feature_importance_df
        
    return None

# -------------------------- 递归特征消除(RFE)分析 --------------------------
def perform_rfe_analysis(best_model_name, best_model, X_train, y_train, X_test, y_test, export_dir):
    """对最佳模型执行递归特征消除(RFE)分析，跟踪性能变化"""
    print(f"\n=== Performing Recursive Feature Elimination (RFE) on {best_model_name} ===")
    
    # 创建RFE导出目录
    rfe_export_dir = os.path.join(export_dir, 'RFE_Analysis')
    os.makedirs(rfe_export_dir, exist_ok=True)
    
    try:
        # 定义不同的特征数量
        min_features = 1
        max_features = min(20, X_train.shape[1])  # 限制最大特征数量为20或总特征数
        
        # 为不同类型的模型选择适当的estimator
        if best_model_name == 'Logistic Regression':
            # 使用逻辑回归作为estimator，适合RFE
            estimator = LogisticRegression(random_state=RANDOM_SEED)
        elif best_model_name == 'Random Forest':
            # 使用随机森林作为estimator
            estimator = RandomForestClassifier(random_state=RANDOM_SEED)
        elif best_model_name == 'XGBoost':
            # 使用XGBoost作为estimator
            estimator = xgb.XGBClassifier(random_state=RANDOM_SEED)
        else:
            # 对于其他模型，使用逻辑回归作为默认estimator
            estimator = LogisticRegression(random_state=RANDOM_SEED)
            print(f"Using Logistic Regression as estimator for RFE analysis of {best_model_name}")
        
        # 存储不同特征数量下的性能指标
        rfe_results = []
        feature_rankings = {}
        
        # 从全特征开始逐步减少特征数量
        for n_features in range(max_features, min_features - 1, -1):
            print(f"Running RFE with {n_features} features...")
            
            # 创建RFE选择器
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            
            # 拟合RFE
            rfe.fit(X_train, y_train)
            
            # 获取选中的特征
            selected_features = X_train.columns[rfe.support_]
            
            # 在测试集上评估性能
            y_pred_proba = rfe.predict_proba(X_test)[:, 1] if hasattr(rfe, 'predict_proba') else None
            
            if y_pred_proba is not None:
                # 计算AUC
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # 计算精确率-召回率曲线的AUC
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc_score = auc(recall, precision)
                
                # 计算准确率
                y_pred = rfe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # 存储结果
                rfe_results.append({
                    'n_features': n_features,
                    'AUC': auc_score,
                    'PR_AUC': pr_auc_score,
                    'Accuracy': accuracy,
                    'Selected_Features': selected_features.tolist()
                })
                
                # 存储特征排名
                if n_features == max_features:
                    # 只在使用全特征时记录排名
                    feature_rankings = {}
                    for i, col in enumerate(X_train.columns):
                        feature_rankings[col] = rfe.ranking_[i]
        
        # 转换结果为DataFrame
        rfe_results_df = pd.DataFrame(rfe_results)
        
        # 保存RFE结果到Excel
        excel_path = os.path.join(rfe_export_dir, f'{best_model_name}_RFE_results.xlsx')
        rfe_results_df.to_excel(excel_path, index=False)
        print(f"RFE results saved to: {excel_path}")
        
        # 保存特征排名到Excel
        feature_rankings_df = pd.DataFrame({
            'Feature': list(feature_rankings.keys()),
            'Ranking': list(feature_rankings.values())
        }).sort_values('Ranking')
        
        ranking_excel_path = os.path.join(rfe_export_dir, f'{best_model_name}_feature_rankings.xlsx')
        feature_rankings_df.to_excel(ranking_excel_path, index=False)
        print(f"Feature rankings saved to: {ranking_excel_path}")
        
        # 绘制RFE性能曲线
        plot_rfe_performance_curve(rfe_results_df, best_model_name, rfe_export_dir)
        
        return rfe_results_df, feature_rankings_df
        
    except Exception as e:
        print(f"Error in RFE analysis for {best_model_name}: {str(e)}")
        return None, None

def plot_rfe_performance_curve(rfe_results_df, model_name, export_dir):
    """绘制递归特征消除过程中的性能变化曲线"""
    try:
        plt.figure(figsize=(12, 8))
        
        # 绘制AUC曲线
        plt.plot(rfe_results_df['n_features'], rfe_results_df['AUC'], 'o-', color='blue', linewidth=2, markersize=8, label='ROC AUC')
        
        # 绘制PR AUC曲线
        plt.plot(rfe_results_df['n_features'], rfe_results_df['PR_AUC'], 's-', color='green', linewidth=2, markersize=8, label='PR AUC')
        
        # 绘制准确率曲线
        plt.plot(rfe_results_df['n_features'], rfe_results_df['Accuracy'], '^-', color='red', linewidth=2, markersize=8, label='Accuracy')
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置标题和标签
        plt.title(f'{model_name} Performance vs. Number of Features (RFE)', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Features', fontsize=14)
        plt.ylabel('Performance Metric', fontsize=14)
        
        # 反转x轴，使特征数量从左到右递减
        plt.gca().invert_xaxis()
        
        # 添加图例
        plt.legend(fontsize=12)
        
        # 保存图像
        plot_path = os.path.join(export_dir, f'{model_name}_RFE_performance_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"RFE performance curve saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error plotting RFE performance curve: {str(e)}")

def generate_rfe_feature_table(rfe_results_df, feature_rankings_df, model_name, export_dir):
    """生成递归特征消除过程中的详细特征选择表格"""
    try:
        # 创建一个综合表格
        summary_data = []
        
        # 为每个特征数量级别添加信息
        for _, row in rfe_results_df.iterrows():
            n_features = row['n_features']
            selected_features = row['Selected_Features']
            auc_score = row['AUC']
            
            # 获取这些特征的排名
            for feature in selected_features:
                if feature in feature_rankings_df['Feature'].values:
                    ranking = feature_rankings_df[feature_rankings_df['Feature'] == feature]['Ranking'].values[0]
                    
                    summary_data.append({
                        'Number_of_Features': n_features,
                        'Feature': feature,
                        'Ranking': ranking,
                        'AUC_at_This_Feature_Count': auc_score,
                        'Included': True
                    })
        
        # 转换为DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # 按特征数量和排名排序
        summary_df = summary_df.sort_values(['Number_of_Features', 'Ranking'], ascending=[False, True])
        
        # 保存表格
        excel_path = os.path.join(export_dir, f'{model_name}_RFE_feature_selection_details.xlsx')
        summary_df.to_excel(excel_path, index=False)
        
        print(f"RFE feature selection details table saved to: {excel_path}")
        
        return summary_df
        
    except Exception as e:
        print(f"Error generating RFE feature table: {str(e)}")
        return None

# -------------------------- 主函数执行 --------------------------
def main():
    """主函数：运行完整的模型训练和评估流程（仅使用全部特征）"""
    # 创建导出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join('model_results', timestamp)
    os.makedirs(export_dir, exist_ok=True)
    print(f"Results will be exported to: {export_dir}")
    
    # 1. 数据预处理
    processed_datasets = load_and_preprocess_data()
    
    # 2. 初始化模型和参数
    models, param_grids, param_counts = init_models_and_params()
    
    # 3. 运行嵌套交叉验证（仅使用全部特征）
    best_models, df_cv_details, df_grid_summary, outer_validation_results, median_auc_datasets = run_nested_cv(
        processed_datasets['X_train_scaled'],
        processed_datasets['y_train'],
        models,
        param_grids,
        param_counts,
        processed_datasets['X_train_non_scaled'],
        export_dir
    )
    
    # 4. 在完整训练集上重训练最佳模型
    retrained_models = retrain_best_models(
        best_models,
        processed_datasets['X_train_scaled'],
        processed_datasets['y_train'],
        processed_datasets['X_train_non_scaled']
    )
    
    # 5. 在所有数据集上评估模型
    evaluation_results = evaluate_models_on_all_datasets(
        retrained_models,
        processed_datasets,
        export_dir
    )
    
    # 6. 评估全部特征的模型性能并生成图表
    all_feature_results = evaluate_all_features(
        processed_datasets,
        retrained_models,
        median_auc_datasets,
        export_dir
    )
    
    # 创建跨模型曲线整合图目录
    cross_model_plots_dir = os.path.join(export_dir, 'cross_model_plots')
    os.makedirs(cross_model_plots_dir, exist_ok=True)
    
    # 生成所有模型的全部特征曲线整合图 - 验证集
    plot_roc_curve_models(
        all_feature_results,
        'ROC Curves of All Models (Validation Set)',
        os.path.join(cross_model_plots_dir, 'roc_validation_set_all_models.png'),
        dataset_type='val'
    )
    
    plot_pr_curve_models(
        all_feature_results,
        'PR Curves of All Models (Validation Set)',
        os.path.join(cross_model_plots_dir, 'pr_validation_set_all_models.png'),
        dataset_type='val'
    )
    
    plot_calibration_curve_models(
        all_feature_results,
        'Calibration Curves of All Models (Validation Set)',
        os.path.join(cross_model_plots_dir, 'calibration_validation_set_all_models.png'),
        dataset_type='val'
    )
    
    plot_dca_curve_models(
        all_feature_results,
        'DCA Curves of All Models (Validation Set)',
        os.path.join(cross_model_plots_dir, 'dca_validation_set_all_models.png'),
        dataset_type='val'
    )
    
    # 生成所有模型的全部特征曲线整合图 - 测试集
    plot_roc_curve_models(
        all_feature_results,
        'ROC Curves of All Models (Test Set)',
        os.path.join(cross_model_plots_dir, 'roc_test_set_all_models.png'),
        dataset_type='test'
    )
    
    plot_pr_curve_models(
        all_feature_results,
        'PR Curves of All Models (Test Set)',
        os.path.join(cross_model_plots_dir, 'pr_test_set_all_models.png'),
        dataset_type='test'
    )
    
    plot_calibration_curve_models(
        all_feature_results,
        'Calibration Curves of All Models (Test Set)',
        os.path.join(cross_model_plots_dir, 'calibration_test_set_all_models.png'),
        dataset_type='test'
    )
    
    plot_dca_curve_models(
        all_feature_results,
        'DCA Curves of All Models (Test Set)',
        os.path.join(cross_model_plots_dir, 'dca_test_set_all_models.png'),
        dataset_type='test'
    )
    
    # 7. 对性能最佳的模型执行SHAP分析
    # 找出性能最佳的模型（基于嵌套交叉验证的平均AUC）
    best_model_name = df_grid_summary.loc[df_grid_summary['Mean_Outer_AUC'].idxmax(), 'Model']
    best_model = retrained_models[best_model_name]
    
    # 计算最佳模型的最优阈值及其置信区间
    print(f"\n=== Evaluating Optimal Threshold for Best Model: {best_model_name} ===")
    optimal_threshold_results = calculate_optimal_threshold_with_ci(best_model_name, best_model, processed_datasets, export_dir)
    
    print(f"\n=== Performing SHAP Analysis on Best Model: {best_model_name} ===")
    # 创建SHAP分析导出目录
    shap_export_dir = os.path.join(export_dir, 'SHAP_Analysis')
    os.makedirs(shap_export_dir, exist_ok=True)
    
    # 执行SHAP分析
    perform_shap_analysis(
        best_model_name,
        best_model,
        processed_datasets['X_train_scaled'],
        processed_datasets['X_test_scaled'],
        processed_datasets['y_test'],
        processed_datasets['X_test_non_scaled'],
        shap_export_dir
    )
    
    print("\nCross-model plots have been generated!")
    # 8. 对性能最佳的模型执行递归特征消除(RFE)分析
    print(f"\n=== Performing Recursive Feature Elimination (RFE) on Best Model: {best_model_name} ===")
    
    # 执行RFE分析
    rfe_results_df, feature_rankings_df = perform_rfe_analysis(
        best_model_name,
        best_model,
        processed_datasets['X_train_scaled'],
        processed_datasets['y_train'],
        processed_datasets['X_test_scaled'],
        processed_datasets['y_test'],
        export_dir
    )
    
    # 如果RFE分析成功，生成详细的特征选择表格
    if rfe_results_df is not None and feature_rankings_df is not None:
        generate_rfe_feature_table(rfe_results_df, feature_rankings_df, best_model_name, os.path.join(export_dir, 'RFE_Analysis'))
    
    print("\nAll analyses completed successfully!")


if __name__ == "__main__":
    main()