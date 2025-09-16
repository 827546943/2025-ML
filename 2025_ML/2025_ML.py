# -*- coding: utf-8 -*-
"""
Unified Model Parameter Tuning, Evaluation, and SHAP Analysis Script
Integrated functionality from parameter selection, parameter optimization with model evaluation, and SHAP analysis scripts
"""

import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import copy
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
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
        
        # Process categorical variables
        categorical_cols = ['gender', 'Disease stage', 'RPR Titer Baseline']
        le_dict = {}  # Store label encoders for possible inverse transformation
        
        # Combine all data for consistent label encoding
        all_data = pd.concat([train_data.drop('cure and infection', axis=1), 
                             test_data.drop('cure and infection', axis=1)])
        
        # Process each dataset separately
        datasets = {'train': train_data, 'test': test_data}
        
        processed_datasets = {}
        for name, data in datasets.items():
            X = data.drop('cure and infection', axis=1)
            y = data['cure and infection']
            
            # Process categorical variables
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    # Use all data to fit encoders to handle all possible categories
                    le.fit(all_data[col].astype(str))
                    X[col] = le.transform(X[col].astype(str))
                    le_dict[col] = le
            
            # Remove ID column if exists
            if 'number' in X.columns:
                X = X.drop('number', axis=1)
            
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
        print(f"Data loading error: {str(e)}")
        # Provide sample data for debugging with fixed random seed
        np.random.seed(RANDOM_SEED)
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_test = pd.DataFrame(np.random.randn(50, 5), columns=[f'feature_{i}' for i in range(5)])
        y_test = pd.Series(np.random.randint(0, 2, 50))
        
        # Create processed dataset structure
        processed_datasets = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_train_scaled': X_train.copy(), 'X_test_scaled': X_test.copy(),
            'X_train_non_scaled': X_train.copy(), 'X_test_non_scaled': X_test.copy()
        }
        
        print("Using sample data for testing")
        return processed_datasets

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

# -------------------------- Helper Function for Inner Loop Feature Selection --------------------------
def inner_loop_feature_selection(model_name, model, X_inner_train, y_inner_train, X_inner_val, y_inner_val, use_non_scaled=False):
    """Perform feature selection within inner CV loop using SHAP importance"""
    # Train initial model with all features
    if model_name == "XGBoost":
        model.fit(X_inner_train, y_inner_train, verbose=False)
    else:
        model.fit(X_inner_train, y_inner_train)
    
    # Get feature importance based on SHAP values or model's built-in importance
    try:
        if model_name == "XGBoost":
            # Use built-in feature importance for XGBoost
            importance = model.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            # Fallback for models without built-in importance (like Naive Bayes)
            importance = np.ones(X_inner_train.shape[1])
        
        # Create sorted feature list based on importance
        sorted_feature_indices = np.argsort(importance)[::-1]
        sorted_features = X_inner_train.columns[sorted_feature_indices].tolist()
        total_features = len(sorted_features)
        
        # Store results for this fold
        fold_selection_results = []
        best_auc = 0
        best_feature_count = 0
        
        # Evaluate with increasing number of features
        for num_features in range(1, total_features + 1):
            selected_features = sorted_features[:num_features]
            
            # Prepare data with selected features
            X_selected_train = X_inner_train[selected_features]
            X_selected_val = X_inner_val[selected_features]
            
            # Create and train new model with selected features
            if model_name == "XGBoost":
                # 创建参数字典，避免random_state重复传递
                model_params = model.get_params()
                model_params['random_state'] = RANDOM_SEED  # 确保使用指定的随机种子
                temp_model = xgb.XGBClassifier(**model_params)
                temp_model.fit(X_selected_train, y_inner_train, verbose=False)
            elif model_name == "Naive Bayes":
                temp_model = GaussianNB(**model.get_params())
                temp_model.fit(X_selected_train, y_inner_train)
            elif model_name == "SVM":
                # 确保SVM模型设置probability=True以支持predict_proba
                model_params = model.get_params()
                model_params['random_state'] = RANDOM_SEED
                model_params['probability'] = True  # 明确设置probability=True
                temp_model = SVC(**model_params)
                temp_model.fit(X_selected_train, y_inner_train)
            else:
                model_class = model.__class__
                model_params = model.get_params()
                model_params['random_state'] = RANDOM_SEED
                temp_model = model_class(**model_params)
                temp_model.fit(X_selected_train, y_inner_train)
            
            # Evaluate on validation set
            y_val_proba = temp_model.predict_proba(X_selected_val)[:, 1]
            auc_score = roc_auc_score(y_inner_val, y_val_proba)
            
            # Update best results
            if auc_score > best_auc:
                best_auc = auc_score
                best_feature_count = num_features
            
            # Record results
            fold_selection_results.append({
                'Num_Features': num_features,
                'AUC': auc_score,
                'Selected_Features': selected_features
            })
        
        return fold_selection_results, best_feature_count, best_auc
        
    except Exception as e:
        print(f"Error in inner loop feature selection: {str(e)}")
        # Fallback: return full feature results
        try:
            y_val_proba = model.predict_proba(X_inner_val)[:, 1]
        except AttributeError:
            # For models like SVC that don't support predict_proba by default
            # Use decision_function and normalize
            decision_scores = model.decision_function(X_inner_val)
            if len(decision_scores.shape) > 1:  # For multi-class models
                decision_scores = decision_scores[:, 1]
            # Normalize to [0, 1] range
            y_val_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        
        auc_score = roc_auc_score(y_inner_val, y_val_proba)
        return [{'Num_Features': X_inner_train.shape[1], 'AUC': auc_score, 'Selected_Features': X_inner_train.columns.tolist()}], \
               X_inner_train.shape[1], auc_score

# -------------------------- 3. Enhanced Nested Cross-validation with Inner Loop Feature Selection --------------------------
def run_nested_cv(X_train, y_train, models, param_grids, param_counts, X_train_non_scaled=None, export_dir=None):
    """Perform nested cross-validation with feature selection in inner loop"""
    # Cross-validation settings
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    
    best_models = {}
    cv_details = []
    grid_search_summary = []
    # Stores feature selection curves for each model and fold
    all_feature_selection_results = {model_name: [] for model_name in models.keys()}
    
    print("\n=== Starting Nested Cross-validation with Inner Loop Feature Selection ===")
    print("Feature selection will be performed within each inner CV fold to find optimal feature count")
    
    for model_name in models.keys():
        print(f"\nProcessing {model_name}...")
        model = models[model_name]
        param_grid = param_grids[model_name]
        param_count = param_counts[model_name]

        outer_scores = []
        fold_details = []
        model_feature_selection_results = []

        # Outer cross-validation loop
        for fold_idx, (train_outer_idx, val_outer_idx) in enumerate(outer_cv.split(X_train, y_train)):
            print(f"  Processing outer fold {fold_idx + 1}/5...")
            
            # Special handling for Naive Bayes - use non-scaled data
            if model_name == "Naive Bayes" and X_train_non_scaled is not None:
                X_outer_train = X_train_non_scaled.iloc[train_outer_idx]
                X_outer_val = X_train_non_scaled.iloc[val_outer_idx]
            else:
                X_outer_train = X_train.iloc[train_outer_idx]
                X_outer_val = X_train.iloc[val_outer_idx]
            
            y_outer_train = y_train.iloc[train_outer_idx]
            y_outer_val = y_train.iloc[val_outer_idx]

            # Inner cross-validation for parameter tuning and feature selection
            inner_results = []
            fold_best_params = None
            fold_best_auc = 0
            fold_best_feature_count = 0
            
            # We'll perform a simplified grid search manually to integrate feature selection
            from sklearn.model_selection import ParameterGrid
            param_combinations = ParameterGrid(param_grid)
            
            for params_idx, params in enumerate(param_combinations):
                # Create model with current parameters
                if model_name == "XGBoost":
                    current_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED)
                elif model_name == "Naive Bayes":
                    current_model = GaussianNB(**params)
                else:
                    model_class = model.__class__
                    current_model = model_class(**params, random_state=RANDOM_SEED)
                
                # Inner CV loop with feature selection
                inner_scores = []
                inner_feature_counts = []
                
                for inner_idx, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
                    # Split inner data
                    X_inner_train = X_outer_train.iloc[train_inner_idx]
                    y_inner_train = y_outer_train.iloc[train_inner_idx]
                    X_inner_val = X_outer_train.iloc[val_inner_idx]
                    y_inner_val = y_outer_train.iloc[val_inner_idx]
                    
                    # Perform feature selection within inner fold
                    fold_selection_results, best_feature_count, best_auc = inner_loop_feature_selection(
                        model_name, current_model, X_inner_train, y_inner_train, X_inner_val, y_inner_val
                    )
                    
                    inner_scores.append(best_auc)
                    inner_feature_counts.append(best_feature_count)
                    
                    # Store feature selection results for visualization
                    for result in fold_selection_results:
                        result['Outer_Fold'] = fold_idx + 1
                        result['Inner_Fold'] = inner_idx + 1
                        result['Parameters'] = str(params)
                        model_feature_selection_results.append(result)
                
                # Calculate average inner score for this parameter set
                avg_inner_auc = np.mean(inner_scores)
                avg_feature_count = int(np.mean(inner_feature_counts))
                
                inner_results.append({
                    'params': params,
                    'avg_auc': avg_inner_auc,
                    'avg_feature_count': avg_feature_count
                })
            
            # Find best parameters and feature count from inner CV
            best_inner_result = max(inner_results, key=lambda x: x['avg_auc'])
            fold_best_params = best_inner_result['params']
            fold_best_feature_count = best_inner_result['avg_feature_count']
            
            # Create model with best parameters
            if model_name == "XGBoost":
                best_inner_model = xgb.XGBClassifier(**fold_best_params, random_state=RANDOM_SEED)
                best_inner_model.fit(X_outer_train, y_outer_train, verbose=False)
            elif model_name == "Naive Bayes":
                best_inner_model = GaussianNB(**fold_best_params)
                best_inner_model.fit(X_outer_train, y_outer_train)
            else:
                model_class = model.__class__
                best_inner_model = model_class(**fold_best_params, random_state=RANDOM_SEED)
                best_inner_model.fit(X_outer_train, y_outer_train)
            
            # Get feature importance and select optimal features
            try:
                if model_name == "XGBoost":
                    importance = best_inner_model.feature_importances_
                elif hasattr(best_inner_model, 'feature_importances_'):
                    importance = best_inner_model.feature_importances_
                elif hasattr(best_inner_model, 'coef_'):
                    importance = np.abs(best_inner_model.coef_[0])
                else:
                    importance = np.ones(X_outer_train.shape[1])
                
                # Select top N features
                sorted_feature_indices = np.argsort(importance)[::-1]
                sorted_features = X_outer_train.columns[sorted_feature_indices].tolist()
                optimal_features = sorted_features[:fold_best_feature_count]
                
                # Train final model with optimal features
                X_outer_train_optimal = X_outer_train[optimal_features]
                X_outer_val_optimal = X_outer_val[optimal_features]
                
                if model_name == "XGBoost":
                    final_model = xgb.XGBClassifier(**fold_best_params, random_state=RANDOM_SEED)
                    final_model.fit(X_outer_train_optimal, y_outer_train, verbose=False)
                elif model_name == "Naive Bayes":
                    final_model = GaussianNB(**fold_best_params)
                    final_model.fit(X_outer_train_optimal, y_outer_train)
                else:
                    model_class = model.__class__
                    final_model = model_class(**fold_best_params, random_state=RANDOM_SEED)
                    final_model.fit(X_outer_train_optimal, y_outer_train)
                
                # Evaluate on outer validation set
                try:
                    y_outer_proba = final_model.predict_proba(X_outer_val_optimal)[:, 1]
                except AttributeError:
                    # For models like SVC that don't support predict_proba by default
                    decision_scores = final_model.decision_function(X_outer_val_optimal)
                    if len(decision_scores.shape) > 1:
                        decision_scores = decision_scores[:, 1]
                    # Normalize to [0, 1] range
                    y_outer_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
                
                outer_auc = roc_auc_score(y_outer_val, y_outer_proba)
                
            except Exception as e:
                print(f"Error in feature selection for outer fold {fold_idx + 1}: {str(e)}")
                # Fallback: use all features
                try:
                    y_outer_proba = best_inner_model.predict_proba(X_outer_val)[:, 1]
                except AttributeError:
                    decision_scores = best_inner_model.decision_function(X_outer_val)
                    if len(decision_scores.shape) > 1:
                        decision_scores = decision_scores[:, 1]
                    # Normalize to [0, 1] range
                    y_outer_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
                
                outer_auc = roc_auc_score(y_outer_val, y_outer_proba)
            
            outer_scores.append(outer_auc)

            # Record detailed fold information
            fold_details.append({
                'Model': model_name,
                'Outer_Fold': fold_idx + 1,
                'Inner_Fold_Count': 3,
                'Optimal_Feature_Count': fold_best_feature_count,
                'Outer_Val_AUC': outer_auc,
                'Best_Params': str(fold_best_params)
            })
        
        # Store feature selection results for this model
        all_feature_selection_results[model_name] = model_feature_selection_results
        
        # Retrain final model on full training set with optimal parameters
        if model_name == "XGBoost" and X_train_non_scaled is not None:
            X_full_train = X_train_non_scaled
        else:
            X_full_train = X_train
        
        if model_name == "XGBoost":
            final_model = xgb.XGBClassifier(**fold_best_params, random_state=RANDOM_SEED)
            final_model.fit(X_full_train, y_train, verbose=False)
        elif model_name == "Naive Bayes":
            final_model = GaussianNB(**fold_best_params)
            final_model.fit(X_full_train, y_train)
        else:
            model_class = model.__class__
            final_model = model_class(**fold_best_params, random_state=RANDOM_SEED)
            final_model.fit(X_full_train, y_train)
        
        best_models[model_name] = final_model

        # Calculate average performance
        avg_auc = np.mean(outer_scores)

        # Record grid search summary
        grid_search_summary.append({
            'Model': model_name,
            'Total_Params': param_count,
            'Mean_Outer_AUC': avg_auc,
            'Std_Outer_AUC': np.std(outer_scores),
            'Best_Params': str(fold_best_params)
        })

        # Extend overall CV details
        cv_details.extend(fold_details)

        print(f"{model_name} - Mean Outer AUC: {avg_auc:.3f} (±{np.std(outer_scores):.3f})")
        print(f"  Best Parameters: {fold_best_params}")

    # Convert to DataFrame for output
    df_cv_details = pd.DataFrame(cv_details)  # Detailed nested CV results
    df_grid_summary = pd.DataFrame(grid_search_summary)  # Grid search summary
    
    # Export nested CV results if export directory is provided
    if export_dir:
        # Save detailed nested CV results to Excel
        nested_cv_excel_path = os.path.join(export_dir, 'nested_cv_results.xlsx')
        with pd.ExcelWriter(nested_cv_excel_path, engine='openpyxl') as writer:
            df_cv_details.to_excel(writer, sheet_name='CV_Details', index=False)
            df_grid_summary.to_excel(writer, sheet_name='CV_Summary', index=False)
        print(f"Nested cross-validation results saved: {nested_cv_excel_path}")
        
        # Create boxplot for nested CV results
        plt.figure(figsize=(12, 8))
        df_cv_details.boxplot(column='Outer_Val_AUC', by='Model', grid=False)
        plt.title('Nested Cross-Validation AUC Results by Model', fontsize=14)
        plt.suptitle('')  # Remove default pandas title
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('AUC Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        boxplot_path = os.path.join(export_dir, 'nested_cv_boxplot.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Nested cross-validation boxplot saved: {boxplot_path}")
        
        # Create bar chart for mean AUC comparison
        plt.figure(figsize=(12, 8))
        sorted_df = df_grid_summary.sort_values('Mean_Outer_AUC', ascending=False)
        bars = plt.bar(sorted_df['Model'], sorted_df['Mean_Outer_AUC'], yerr=sorted_df['Std_Outer_AUC'],
                      capsize=5, alpha=0.7)
        plt.title('Mean AUC Comparison Across Models (Nested CV)', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Mean AUC Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for bar, auc_val in zip(bars, sorted_df['Mean_Outer_AUC']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{auc_val:.3f}', ha='center', va='bottom')
                     
        plt.tight_layout()
        
        barplot_path = os.path.join(export_dir, 'nested_cv_barplot.png')
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Nested cross-validation barplot saved: {barplot_path}")

    return best_models, df_cv_details, df_grid_summary, all_feature_selection_results

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
                
                # Calculate AIC value
                n_features = X_data.shape[1]
                aic = calculate_aic(y_data, y_pred_proba, n_features)
                
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
                    'Samples': len(y_data),
                    'Positive_Samples': sum(y_data),
                    'Positive_Ratio': np.mean(y_data),
                    'Feature_Count': n_features
                }
                
                all_evaluation_results.append(metrics)
                
                # Print results for current dataset
                print(f"  {dataset_name.capitalize()} set - AUC: {metrics['AUC']:.4f}, "
                      f"Accuracy: {metrics['Accuracy']:.4f}, AIC: {metrics['AIC']:.2f}")
                
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

# -------------------------- 6. Calculate AIC Values --------------------------
def calculate_aic(y_true, y_pred_proba, n_features):
    """Calculate Akaike Information Criterion (AIC) for classification models"""
    # Calculate log likelihood for binary classification
    ll = 0
    for i in range(len(y_true)):
        if y_true.iloc[i] == 1:
            ll += np.log(y_pred_proba[i] + 1e-15)  # Add small value to avoid log(0)
        else:
            ll += np.log(1 - y_pred_proba[i] + 1e-15)
    
    # AIC = -2*log_likelihood + 2*number_of_parameters
    # For classification models, number of parameters is n_features + 1 (intercept)
    aic = -2 * ll + 2 * (n_features + 1)
    return aic

# -------------------------- 7. Plot Confusion Matrix --------------------------
def plot_confusion_matrix(models, X_test, y_test, export_dir, X_test_non_scaled=None, best_model_name=None, X_train=None, y_train=None, best_model_name_by_cv=None, inner_loop_best_feature_counts=None):
    """Plot confusion matrix for the specified best model or best model based on test set AUC and AIC"""
    # 初始化变量，避免UnboundLocalError
    best_model = None
    best_auc = 0.0
    y_pred = None
    fallback_model_used = False
    
    try:
        # 首先检查models是否为空
        if not models:
            print("Error: No models available to plot confusion matrix.")
            return None
        
        # 默认使用第一个模型作为后备
        default_model_name = next(iter(models.keys()))
        default_model = models[default_model_name]
        
        # 如果提供了最佳模型名称，直接使用
        if best_model_name and best_model_name in models:
            best_model = models[best_model_name]
            # 计算该模型的测试集AUC
            if best_model_name == "Naive Bayes" and X_test_non_scaled is not None:
                try:
                    y_pred_proba = best_model.predict_proba(X_test_non_scaled)[:, 1]
                except AttributeError:
                    decision_scores = best_model.decision_function(X_test_non_scaled)
                    if len(decision_scores.shape) > 1:
                        decision_scores = decision_scores[:, 1]
                    # Normalize to [0, 1] range
                    y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
            else:
                try:
                    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                except AttributeError:
                    decision_scores = best_model.decision_function(X_test)
                    if len(decision_scores.shape) > 1:
                        decision_scores = decision_scores[:, 1]
                    # Normalize to [0, 1] range
                    y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
            best_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nUsing specified best model: {best_model_name} with Test Set AUC: {best_auc:.4f}")
        else:
            # 使用与identify_best_model相同的逻辑选择最佳模型（基于AIC和过拟合检查）
            print(f"\n=== Selecting Best Model for Confusion Matrix (using AIC-based selection) ===")
            model_performances = {}
        
        # 默认使用所有特征，除非提供了内部循环最佳特征数量
        if inner_loop_best_feature_counts is None:
            inner_loop_best_feature_counts = {}
            # 计算默认特征数量
            default_n_features = X_train.shape[1] if X_train is not None else X_test.shape[1]
            for model_name in models.keys():
                inner_loop_best_feature_counts[model_name] = default_n_features  # 默认使用所有特征
            
            for model_name, model in models.items():
                # 根据模型类型使用适当的数据进行预测
                if model_name == "Naive Bayes" and X_test_non_scaled is not None:
                    try:
                        y_pred_proba_test = model.predict_proba(X_test_non_scaled)[:, 1]
                        # 计算训练集性能
                        y_pred_proba_train = model.predict_proba(X_train)[:, 1] if X_train is not None else None
                    except AttributeError:
                        try:
                            decision_scores_test = model.decision_function(X_test_non_scaled)
                            if len(decision_scores_test.shape) > 1:
                                decision_scores_test = decision_scores_test[:, 1]
                            # Normalize to [0, 1] range
                            y_pred_proba_test = (decision_scores_test - decision_scores_test.min()) / (decision_scores_test.max() - decision_scores_test.min() + 1e-10)
                            
                            # 计算训练集性能
                            if X_train is not None:
                                decision_scores_train = model.decision_function(X_train)
                                if len(decision_scores_train.shape) > 1:
                                    decision_scores_train = decision_scores_train[:, 1]
                                y_pred_proba_train = (decision_scores_train - decision_scores_train.min()) / (decision_scores_train.max() - decision_scores_train.min() + 1e-10)
                            else:
                                y_pred_proba_train = None
                        except Exception as e:
                            print(f"  Error getting prediction scores for {model_name}: {str(e)}")
                            continue
                else:
                    try:
                        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
                        # 计算训练集性能
                        y_pred_proba_train = model.predict_proba(X_train)[:, 1] if X_train is not None else None
                    except AttributeError:
                        try:
                            decision_scores_test = model.decision_function(X_test)
                            if len(decision_scores_test.shape) > 1:
                                decision_scores_test = decision_scores_test[:, 1]
                            # Normalize to [0, 1] range
                            y_pred_proba_test = (decision_scores_test - decision_scores_test.min()) / (decision_scores_test.max() - decision_scores_test.min() + 1e-10)
                            
                            # 计算训练集性能
                            if X_train is not None:
                                decision_scores_train = model.decision_function(X_train)
                                if len(decision_scores_train.shape) > 1:
                                    decision_scores_train = decision_scores_train[:, 1]
                                y_pred_proba_train = (decision_scores_train - decision_scores_train.min()) / (decision_scores_train.max() - decision_scores_train.min() + 1e-10)
                            else:
                                y_pred_proba_train = None
                        except Exception as e:
                            print(f"  Error getting prediction scores for {model_name}: {str(e)}")
                            continue
                
                try:
                    auc_test = roc_auc_score(y_test, y_pred_proba_test)
                    auc_train = roc_auc_score(y_train, y_pred_proba_train) if y_pred_proba_train is not None and y_train is not None else None
                    # 计算训练集和测试集的AUC差距，用于评估过拟合
                    auc_gap = auc_train - auc_test if auc_train is not None else 0
                except Exception as e:
                    print(f"  Error calculating AUC for {model_name}: {str(e)}")
                    continue
                
                try:
                    # 获取特征数量（使用内部循环最佳特征数量）
                    if model_name in inner_loop_best_feature_counts:
                        n_features = inner_loop_best_feature_counts[model_name]
                        print(f"  Using best feature count ({n_features}) for {model_name}")
                    else:
                        n_features = X_test.shape[1] if model_name != "Naive Bayes" or X_test_non_scaled is None else X_test_non_scaled.shape[1]
                    
                    # 计算AIC值（使用测试集预测概率和特征数量）
                    aic_value = calculate_aic(y_test, y_pred_proba_test, n_features)
                    
                    model_performances[model_name] = {
                        'AUC_Test': auc_test,
                        'AUC_Train': auc_train,
                        'AUC_Gap': auc_gap,
                        'AIC': aic_value
                    }
                    
                    print(f"{model_name}: ")
                    print(f"  Test AUC = {auc_test:.4f}, AIC: {aic_value:.2f}")
                    if auc_train is not None:
                        print(f"  Train AUC = {auc_train:.4f}, AUC Gap = {auc_gap:.4f}")
                except Exception as e:
                    print(f"  Error calculating metrics for {model_name}: {str(e)}")
                    continue
            
            # 如果model_performances为空，则使用第一个可用的模型
            if not model_performances:
                print("Warning: No model performances calculated. Using first available model.")
                if models:
                    best_model_name = next(iter(models.keys()))
                    best_model = models[best_model_name]
                else:
                    print("Error: No models available to plot confusion matrix.")
                    return None
            else:
                # 如果提供了基于交叉验证的最佳模型名称，我们仍然使用它，但增加过拟合检查
                if best_model_name_by_cv and best_model_name_by_cv in model_performances:
                    best_model_name = best_model_name_by_cv
                    perf = model_performances[best_model_name]
                    print(f"\nSelected Best Model by Nested CV Average AUC: {best_model_name}")
                    print(f"  Test AUC = {perf['AUC_Test']:.4f}")
                    if perf['AUC_Gap'] is not None:
                        print(f"  AUC Gap = {perf['AUC_Gap']:.4f}")
                        
                        # 检查是否存在过拟合风险
                        if perf['AUC_Gap'] > 0.15:  # 如果训练集和测试集的AUC差距超过0.15，认为有过拟合风险
                            print(f"  WARNING: {best_model_name} has overfitting risk (AUC Gap > 0.15).")
                            # 寻找AIC值最小的模型作为替代选项
                            alternative_model_name = min(model_performances, key=lambda x: model_performances[x]['AIC'])
                            alt_perf = model_performances[alternative_model_name]
                            
                            # 如果替代模型的AIC值更小，且测试集AUC不低于最佳模型的95%，则使用替代模型
                            if alt_perf['AIC'] < perf['AIC'] and alt_perf['AUC_Test'] >= perf['AUC_Test'] * 0.95:
                                print(f"  Alternative model {alternative_model_name} selected due to lower AIC and controlled overfitting.")
                                print(f"  Alternative model Test AUC = {alt_perf['AUC_Test']:.4f}, AIC = {alt_perf['AIC']:.2f}")
                                best_model_name = alternative_model_name
                else:
                    # 不使用交叉验证结果时，直接选择AIC值最小的模型
                    best_model_name = min(model_performances, key=lambda x: model_performances[x]['AIC'])
                    perf = model_performances[best_model_name]
                    print(f"\nBest Performing Model with AIC-based Selection: {best_model_name}")
                    print(f"  Test AUC = {perf['AUC_Test']:.4f}, AIC = {perf['AIC']:.2f}")
                    if perf['AUC_Gap'] is not None:
                        print(f"  AUC Gap = {perf['AUC_Gap']:.4f}")
                
                # 确保best_model_name在models中
                if best_model_name in models:
                    best_model = models[best_model_name]
                    best_auc = model_performances[best_model_name]['AUC_Test']
                    print(f"Selected {best_model_name} as best model with Test Set AUC: {best_auc:.4f}")
                else:
                    print(f"Warning: {best_model_name} not found in models. Using first available model.")
                    if models:
                        best_model_name = next(iter(models.keys()))
                        best_model = models[best_model_name]
                        # 尽量计算该模型的AUC
                        if best_model_name in model_performances:
                            best_auc = model_performances[best_model_name]['AUC_Test']
        
        # 确保best_model已定义
        if best_model is None:
            print("Error: No valid best model found. Using first available model.")
            if models:
                best_model_name = next(iter(models.keys()))
                best_model = models[best_model_name]
            else:
                print("Error: No models available to plot confusion matrix.")
                return None
        
        # Generate confusion matrix for the best model
        # 使用内部循环最佳特征数量进行特征选择
        if best_model_name in inner_loop_best_feature_counts and inner_loop_best_feature_counts[best_model_name] > 0:
            n_features_to_use = inner_loop_best_feature_counts[best_model_name]
            print(f"Using {n_features_to_use} best features for confusion matrix prediction with {best_model_name}")
            
            # 根据模型类型选择适当的数据并应用特征选择
            if best_model_name == "Naive Bayes" and X_test_non_scaled is not None:
                # 对非标准化测试数据应用特征选择
                if n_features_to_use < X_test_non_scaled.shape[1]:
                    # 这里假设特征是按重要性排序的，取前n_features_to_use个特征
                    X_test_selected = X_test_non_scaled.iloc[:, :n_features_to_use]
                else:
                    X_test_selected = X_test_non_scaled
                try:
                    y_pred = best_model.predict(X_test_selected)
                except Exception as e:
                    print(f"Error predicting with {best_model_name}: {str(e)}")
                    # 使用所有特征作为后备
                    y_pred = best_model.predict(X_test_non_scaled)
            else:
                # 对标准化测试数据应用特征选择
                if n_features_to_use < X_test.shape[1]:
                    # 这里假设特征是按重要性排序的，取前n_features_to_use个特征
                    X_test_selected = X_test.iloc[:, :n_features_to_use]
                else:
                    X_test_selected = X_test
                try:
                    # 确保best_model不为None
                    if best_model is not None:
                        y_pred = best_model.predict(X_test_selected)
                    else:
                        print("Error: best_model is None, cannot make predictions.")
                        # 使用默认模型
                        y_pred = default_model.predict(X_test_selected)
                        fallback_model_used = True
                except Exception as e:
                    print(f"Error predicting with {best_model_name}: {str(e)}")
                    # 使用所有特征作为后备
                    if best_model is not None:
                        y_pred = best_model.predict(X_test)
                    else:
                        y_pred = default_model.predict(X_test)
                        fallback_model_used = True
        else:
            # 如果没有提供内部循环最佳特征数量，使用所有特征
            if best_model_name == "Naive Bayes" and X_test_non_scaled is not None:
                try:
                    if best_model is not None:
                        y_pred = best_model.predict(X_test_non_scaled)
                    else:
                        y_pred = default_model.predict(X_test_non_scaled)
                        fallback_model_used = True
                except Exception as e:
                    print(f"Error predicting with {best_model_name}: {str(e)}")
                    # 使用标准化数据作为后备
                    try:
                        if best_model is not None:
                            y_pred = best_model.predict(X_test)
                        else:
                            y_pred = default_model.predict(X_test)
                            fallback_model_used = True
                    except Exception as e2:
                        print(f"Error predicting with standardized data: {str(e2)}")
                        # 无法预测，返回None
                        return None
            else:
                try:
                    if best_model is not None:
                        y_pred = best_model.predict(X_test)
                    else:
                        y_pred = default_model.predict(X_test)
                        fallback_model_used = True
                except Exception as e:
                    print(f"Error predicting with {best_model_name}: {str(e)}")
                    # 无法预测，返回None
                    return None
        
        # 确保y_pred已定义
        if y_pred is None:
            print("Error: Failed to generate predictions.")
            # 尝试使用默认模型作为最后的后备
            try:
                y_pred = default_model.predict(X_test)
                fallback_model_used = True
            except Exception as e:
                print(f"Failed to make predictions with default model: {str(e)}")
                return None
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cured', 'Infected'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        
        # 根据是否使用了后备模型来设置标题
        if fallback_model_used:
            plt.title(f'{best_model_name} Confusion Matrix (Fallback Model)\nTest Set AUC: {best_auc:.4f}', fontsize=14)
        else:
            plt.title(f'{best_model_name} Confusion Matrix (Best Model)\nTest Set AUC: {best_auc:.4f}', fontsize=14)
        plt.tight_layout()
        
        cm_save_path = os.path.join(export_dir, f'{best_model_name}_confusion_matrix.png')
        plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Best model confusion matrix saved: {cm_save_path}")
        
        return best_model_name
        
    except Exception as e:
        print(f"Error in plot_confusion_matrix: {str(e)}")
        # 作为最后的后备，尝试使用第一个可用模型
        if models:
            best_model_name = next(iter(models.keys()))
            print(f"Attempting to use {best_model_name} as fallback.")
            try:
                best_model = models[best_model_name]
                # 简单预测
                if best_model_name == "Naive Bayes" and X_test_non_scaled is not None:
                    y_pred = best_model.predict(X_test_non_scaled)
                else:
                    y_pred = best_model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cured', 'Infected'])
                disp.plot(cmap=plt.cm.Blues, values_format='d')
                plt.title(f'{best_model_name} Confusion Matrix (Fallback Model)', fontsize=14)
                plt.tight_layout()
                
                cm_save_path = os.path.join(export_dir, f'{best_model_name}_confusion_matrix_fallback.png')
                plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Fallback model confusion matrix saved: {cm_save_path}")
                return best_model_name
            except Exception as fallback_e:
                print(f"Fallback failed: {str(fallback_e)}")
        
        # 如果所有尝试都失败，返回None
        print("Failed to plot confusion matrix.")
        return None

# -------------------------- 8. ROC曲线计算 --------------------------
def calculate_roc_curve(models, X_test, y_test, X_test_non_scaled=None):
    """Calculate ROC curve data for all models"""
    roc_results = {}
    for model_name, model in models.items():
        # Use appropriate data for prediction
        if model_name == "Naive Bayes" and X_test_non_scaled is not None:
            try:
                y_pred_proba = model.predict_proba(X_test_non_scaled)[:, 1]
            except AttributeError:
                decision_scores = model.decision_function(X_test_non_scaled)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        else:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                decision_scores = model.decision_function(X_test)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        
        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        roc_results[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}
    return roc_results

# -------------------------- 9. Calculate PR Curve --------------------------
def calculate_pr_curve(models, X_test, y_test, X_test_non_scaled=None):
    """Calculate precision-recall curve data for all models"""
    pr_results = {}
    for model_name, model in models.items():
        # Use appropriate data for prediction
        if model_name == "Naive Bayes" and X_test_non_scaled is not None:
            try:
                y_pred_proba = model.predict_proba(X_test_non_scaled)[:, 1]
            except AttributeError:
                decision_scores = model.decision_function(X_test_non_scaled)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        else:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                decision_scores = model.decision_function(X_test)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        pr_results[model_name] = {'precision': precision, 'recall': recall, 'pr_auc': pr_auc}
    return pr_results

# -------------------------- 10. Calculate Calibration Curve --------------------------
def calculate_calibration_curve(models, X_test, y_test, X_test_non_scaled=None):
    """Calculate calibration curve data for all models"""
    calib_results = {}
    for model_name, model in models.items():
        # 使用适当的数据进行预测
        if model_name == "Naive Bayes" and X_test_non_scaled is not None:
            try:
                y_pred_proba = model.predict_proba(X_test_non_scaled)[:, 1]
            except AttributeError:
                decision_scores = model.decision_function(X_test_non_scaled)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        else:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                decision_scores = model.decision_function(X_test)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        
        # 计算校准曲线
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        calib_results[model_name] = {'prob_true': prob_true, 'prob_pred': prob_pred}
    return calib_results

# -------------------------- 11. DCA计算 --------------------------
def calculate_dca(y_true, y_pred_proba, thresholds):
    """Calculate decision curve analysis metrics"""
    n = len(y_true)
    net_benefit = []
    for pt in thresholds:
        if (1 - pt) == 0:  # Avoid division by zero
            net_benefit.append(0)
            continue
        
        pred_pos = y_pred_proba >= pt
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        
        # 计算净收益
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit

# -------------------------- 12. Plotting Functions --------------------------
def plot_roc_curve(roc_results, save_path, best_model_name=None):
    """Plot ROC curves for all models with best model highlighted"""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    # 使用深色颜色映射并调整范围，使颜色更深
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_results)))
    model_names = list(roc_results.keys())
    
    for i, model_name in enumerate(model_names):
        fpr = roc_results[model_name]['fpr']
        tpr = roc_results[model_name]['tpr']
        auc_score = roc_results[model_name]['auc']
        
        # 突出显示最佳模型
        if best_model_name and model_name == best_model_name:
            plt.plot(fpr, tpr, color='red', linewidth=2, 
                     label=f'{model_name} (AUC={auc_score:.4f})')
        else:
            plt.plot(fpr, tpr, color=colors[i], linewidth=2, alpha=1.0, 
                     label=f'{model_name} (AUC={auc_score:.4f})')
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curves of All Models (Test Set)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {save_path}")

def plot_pr_curve(pr_results, y_test, save_path, best_model_name=None):
    """Plot PR curves for all models with best model highlighted"""
    plt.figure(figsize=(10, 8))
    prevalence = np.mean(y_test)
    plt.axhline(y=prevalence, color='k', linestyle='--', 
                label=f'Random Guess (Prevalence={prevalence:.3f})')
    # 使用深色颜色映射并调整范围，使颜色更深
    colors = plt.cm.Set1(np.linspace(0, 1, len(pr_results)))
    model_names = list(pr_results.keys())
    
    for i, model_name in enumerate(model_names):
        precision = pr_results[model_name]['precision']
        recall = pr_results[model_name]['recall']
        pr_auc = pr_results[model_name]['pr_auc']
        
        # 突出显示最佳模型
        if best_model_name and model_name == best_model_name:
            plt.plot(recall, precision, color='red', linewidth=2, alpha=1.0,
                     label=f'{model_name} (PR-AUC={pr_auc:.4f})')
        else:
            plt.plot(recall, precision, color=colors[i], linewidth=2, alpha=1.0,
                     label=f'{model_name} (PR-AUC={pr_auc:.4f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves of All Models (Test Set)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved: {save_path}")

def plot_calibration_curve(calib_results, save_path):
    """Plot calibration curves for all models"""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    # 使用深色颜色映射并调整范围，使颜色更深
    colors = plt.cm.Set1(np.linspace(0, 1, len(calib_results)))
    model_names = list(calib_results.keys())
    
    for i, model_name in enumerate(model_names):
        prob_true = calib_results[model_name]['prob_true']
        prob_pred = calib_results[model_name]['prob_pred']
        valid_idx = ~np.isnan(prob_true) & ~np.isnan(prob_pred)
        if np.sum(valid_idx) > 0:
            plt.plot(prob_pred[valid_idx], prob_true[valid_idx], 
                     color=colors[i], marker='o', ms=6, linewidth=2, alpha=1.0,
                     label=model_name)
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Observed Probability', fontsize=12)
    plt.title('Calibration Curves of All Models (Test Set)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved: {save_path}")

def plot_dca_curve(models, X_test, y_test, save_path, X_test_non_scaled=None):
    """Plot DCA curves for all models"""
    plt.figure(figsize=(10, 8))
    thresholds = np.linspace(0, 0.99, 100)
    plt.axhline(y=0, color='k', linestyle='--', label='No Treatment')
    
    # All treatment line
    prevalence = np.mean(y_test)
    all_treat_nb = [prevalence - (1-prevalence)*(pt/(1-pt)) for pt in thresholds]
    plt.plot(thresholds, all_treat_nb, color='gray', linestyle='--', label='Treat All')
    
    # 使用深色颜色映射并调整范围，使颜色更深
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    model_names = list(models.keys())
    
    for i, model_name in enumerate(model_names):
        # Use appropriate data for prediction
        if model_name == "Naive Bayes" and X_test_non_scaled is not None:
            try:
                y_pred_proba_pos = models[model_name].predict_proba(X_test_non_scaled)[:, 1]
            except AttributeError:
                decision_scores = models[model_name].decision_function(X_test_non_scaled)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba_pos = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        else:
            try:
                y_pred_proba_pos = models[model_name].predict_proba(X_test)[:, 1]
            except AttributeError:
                decision_scores = models[model_name].decision_function(X_test)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[:, 1]
                # Normalize to [0, 1] range
                y_pred_proba_pos = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        
        net_benefit = calculate_dca(y_true=y_test, y_pred_proba=y_pred_proba_pos, thresholds=thresholds)
        plt.plot(thresholds, net_benefit, color=colors[i], linewidth=2, alpha=1.0, label=model_name)
    
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis of All Models (Test Set)', fontsize=14, fontweight='bold')
    plt.ylim(bottom=-0.1, top=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"DCA curve saved: {save_path}")

# -------------------------- 13. Identify Best Performing Model --------------------------
def identify_best_model(retrained_models, X_test, y_test, X_train, y_train, X_test_non_scaled=None, best_model_name_by_cv=None, inner_loop_best_feature_counts=None):
    """Identify the best performing model with AIC-based selection. 
    If best_model_name_by_cv is provided, use that model. Otherwise, identify based on test set AIC 
    and additional metrics to control overfitting."""
    print("\n=== Evaluating Models on Test Set with AIC-based Selection ===")
    model_performances = {}
    
    # 定义包含最佳特征子集和模型的字典
    models_with_optimal_features = {}
    
    for model_name, model in retrained_models.items():
        # 获取内部循环最佳特征数量，如果未提供则使用所有特征
        if inner_loop_best_feature_counts and model_name in inner_loop_best_feature_counts:
            best_feature_count = inner_loop_best_feature_counts[model_name]
            print(f"  Using {best_feature_count} features for {model_name} from inner loop selection")
        else:
            best_feature_count = X_train.shape[1]  # 默认使用所有特征
            print(f"  Using all {best_feature_count} features for {model_name} (no inner loop data available)")
        
        # 尝试获取模型的特征重要性
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importances = np.abs(model.coef_[0])
            elif hasattr(model, 'dual_coef_'):
                feature_importances = np.abs(model.dual_coef_[0])
            else:
                # 如果无法获取特征重要性，则使用所有特征
                print(f"  Warning: Cannot get feature importances for {model_name}, using all features")
                feature_indices = np.arange(X_train.shape[1])
                X_train_optimal = X_train
                X_test_optimal = X_test
                
                # 重新训练模型
                new_model = copy.deepcopy(model)
                new_model.fit(X_train_optimal, y_train)
                models_with_optimal_features[model_name] = {
                    'model': new_model,
                    'best_feature_count': best_feature_count,
                    'selected_features': feature_indices
                }
                continue
            
            # 选择最重要的特征
            if best_feature_count > len(feature_importances):
                best_feature_count = len(feature_importances)
                print(f"  Adjusted best feature count for {model_name} to {best_feature_count}")
            
            top_feature_indices = np.argsort(feature_importances)[::-1][:best_feature_count]
            
            # 提取最佳特征子集
            X_train_optimal = X_train[:, top_feature_indices]
            X_test_optimal = X_test[:, top_feature_indices]
            
            # 重新训练模型
            new_model = copy.deepcopy(model)
            new_model.fit(X_train_optimal, y_train)
            
            models_with_optimal_features[model_name] = {
                'model': new_model,
                'best_feature_count': best_feature_count,
                'selected_features': top_feature_indices
            }
            
        except Exception as e:
            print(f"  Error during feature selection for {model_name}: {str(e)}")
            # 如果特征选择失败，使用原始模型和所有特征
            models_with_optimal_features[model_name] = {
                'model': model,
                'best_feature_count': X_train.shape[1],
                'selected_features': np.arange(X_train.shape[1])
            }
            continue
        
        # 根据模型类型使用适当的数据进行预测
        current_model = models_with_optimal_features[model_name]['model']
        best_feature_count = models_with_optimal_features[model_name]['best_feature_count']
        
        if model_name == "Naive Bayes" and X_test_non_scaled is not None:
            # 注意：Naive Bayes需要非标准化数据，这里可能需要特殊处理特征选择
            try:
                # 使用原始模型的预测（因为Naive Bayes的特征选择比较特殊）
                y_pred_proba_test = model.predict_proba(X_test_non_scaled)[:, 1]
                y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            except AttributeError:
                try:
                    decision_scores_test = model.decision_function(X_test_non_scaled)
                    if len(decision_scores_test.shape) > 1:
                        decision_scores_test = decision_scores_test[:, 1]
                    # Normalize to [0, 1] range
                    y_pred_proba_test = (decision_scores_test - decision_scores_test.min()) / (decision_scores_test.max() - decision_scores_test.min() + 1e-10)
                    
                    decision_scores_train = model.decision_function(X_train)
                    if len(decision_scores_train.shape) > 1:
                        decision_scores_train = decision_scores_train[:, 1]
                    y_pred_proba_train = (decision_scores_train - decision_scores_train.min()) / (decision_scores_train.max() - decision_scores_train.min() + 1e-10)
                except Exception as e:
                    print(f"  Error getting prediction scores for {model_name}: {str(e)}")
                    continue
        else:
            try:
                y_pred_proba_test = current_model.predict_proba(X_test_optimal)[:, 1]
                y_pred_proba_train = current_model.predict_proba(X_train_optimal)[:, 1]
            except AttributeError:
                try:
                    decision_scores_test = current_model.decision_function(X_test_optimal)
                    if len(decision_scores_test.shape) > 1:
                        decision_scores_test = decision_scores_test[:, 1]
                    # Normalize to [0, 1] range
                    y_pred_proba_test = (decision_scores_test - decision_scores_test.min()) / (decision_scores_test.max() - decision_scores_test.min() + 1e-10)
                    
                    decision_scores_train = current_model.decision_function(X_train_optimal)
                    if len(decision_scores_train.shape) > 1:
                        decision_scores_train = decision_scores_train[:, 1]
                    y_pred_proba_train = (decision_scores_train - decision_scores_train.min()) / (decision_scores_train.max() - decision_scores_train.min() + 1e-10)
                except Exception as e:
                    print(f"  Error getting prediction scores for {model_name}: {str(e)}")
                    continue
        
        try:
            auc_test = roc_auc_score(y_test, y_pred_proba_test)
            auc_train = roc_auc_score(y_train, y_pred_proba_train)
            # 计算训练集和测试集的AUC差距，用于评估过拟合
            auc_gap = auc_train - auc_test
        except Exception as e:
            print(f"  Error calculating AUC for {model_name}: {str(e)}")
            continue
        
        # Also calculate accuracy
        try:
            if model_name == "Naive Bayes" and X_test_non_scaled is not None:
                y_pred_test = model.predict(X_test_non_scaled)
                y_pred_train = model.predict(X_train)
            else:
                y_pred_test = current_model.predict(X_test_optimal)
                y_pred_train = current_model.predict(X_train_optimal)
            
            acc_test = accuracy_score(y_test, y_pred_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_gap = acc_train - acc_test
            
            # 计算AIC值（使用测试集预测概率和特征数量）
            aic_value = calculate_aic(y_test, y_pred_proba_test, best_feature_count)
            
            # AIC值越小表示模型越好，这里我们转换为得分（-AIC）以便与原有逻辑兼容
            aic_based_score = -aic_value
            
            model_performances[model_name] = {
                'AUC_Test': auc_test, 
                'AUC_Train': auc_train,
                'AUC_Gap': auc_gap,
                'Accuracy_Test': acc_test, 
                'Accuracy_Train': acc_train,
                'Accuracy_Gap': acc_gap,
                'AIC': aic_value,
                'AIC_Based_Score': aic_based_score
            }
            
            print(f"{model_name}: ")
            print(f"  Test AUC = {auc_test:.4f}, Train AUC = {auc_train:.4f}, AUC Gap = {auc_gap:.4f}")
            print(f"  Test Accuracy = {acc_test:.4f}, Train Accuracy = {acc_train:.4f}, Accuracy Gap = {acc_gap:.4f}")
            print(f"  AIC: {aic_value:.2f}, AIC-Based Score: {aic_based_score:.2f}")
        except Exception as e:
            print(f"  Error calculating metrics for {model_name}: {str(e)}")
            continue
    
    # 更新retrained_models为使用最佳特征子集的模型
    updated_retrained_models = {}
    for model_name, model_data in models_with_optimal_features.items():
        updated_retrained_models[model_name] = model_data['model']
    
    # 如果提供了基于交叉验证的最佳模型名称，我们仍然使用它，但增加过拟合检查
    try:
        if best_model_name_by_cv and best_model_name_by_cv in updated_retrained_models:
            best_model_name = best_model_name_by_cv
            # 安全检查：确保模型名称在model_performances字典中
            if best_model_name not in model_performances:
                print(f"Warning: {best_model_name} not found in model_performances, selecting based on AIC")
                if model_performances:
                    best_model_name = min(model_performances, key=lambda x: model_performances[x]['AIC'])
                elif updated_retrained_models:
                    best_model_name = list(updated_retrained_models.keys())[0]
                else:
                    best_model_name = None
            
            # 再次检查best_model_name是否有效
            if best_model_name and best_model_name in model_performances:
                perf = model_performances[best_model_name]
                best_feature_count = models_with_optimal_features[best_model_name]['best_feature_count']
                print(f"\nSelected Best Model by Nested CV Average AUC: {best_model_name}")
                print(f"  Using {best_feature_count} features from inner loop selection")
                print(f"  Test AUC = {perf['AUC_Test']:.4f}, Train AUC = {perf['AUC_Train']:.4f}, AUC Gap = {perf['AUC_Gap']:.4f}")
                
                # 检查是否存在过拟合风险
                if perf['AUC_Gap'] > 0.15:  # 如果训练集和测试集的AUC差距超过0.15，认为有过拟合风险
                    print(f"  WARNING: {best_model_name} has overfitting risk (AUC Gap > 0.15).")
                    # 寻找AIC值最小的模型作为替代选项
                    try:
                        alternative_model_name = min(model_performances, key=lambda x: model_performances[x]['AIC'])
                        if alternative_model_name in model_performances:
                            alt_perf = model_performances[alternative_model_name]
                            
                            # 如果替代模型的AIC值更小，且测试集AUC不低于最佳模型的95%，则使用替代模型
                            if alt_perf['AIC'] < perf['AIC'] and alt_perf['AUC_Test'] >= perf['AUC_Test'] * 0.95:
                                print(f"  Alternative model {alternative_model_name} selected due to lower AIC and controlled overfitting.")
                                print(f"  Alternative model Test AUC = {alt_perf['AUC_Test']:.4f}, AIC = {alt_perf['AIC']:.2f}")
                                best_model_name = alternative_model_name
                                if best_model_name in models_with_optimal_features:
                                    best_feature_count = models_with_optimal_features[best_model_name]['best_feature_count']
                    except Exception as e:
                        print(f"  Error finding alternative model: {str(e)}")
            else:
                print(f"Warning: Cannot access performance data for model {best_model_name}")
    except Exception as e:
        print(f"Error in best model selection process: {str(e)}")
        # 回退到选择第一个可用模型
        if updated_retrained_models:
            best_model_name = list(updated_retrained_models.keys())[0]
            print(f"  Falling back to first available model: {best_model_name}")
        else:
            best_model_name = None
            print("  No models available for selection")
    else:
        # 不使用交叉验证结果时，直接选择AIC值最小的模型
        try:
            best_model_name = min(model_performances, key=lambda x: model_performances[x]['AIC'])
            perf = model_performances[best_model_name]
            best_feature_count = models_with_optimal_features[best_model_name]['best_feature_count']
            print(f"\nBest Performing Model with AIC-based Selection: {best_model_name}")
            print(f"  Using {best_feature_count} features from inner loop selection")
            print(f"  Test AUC = {perf['AUC_Test']:.4f}, Train AUC = {perf['AUC_Train']:.4f}, AUC Gap = {perf['AUC_Gap']:.4f}")
            print(f"  AIC: {perf['AIC']:.2f}, AIC-Based Score: {perf['AIC_Based_Score']:.2f}")
        except Exception as e:
            print(f"Error selecting best model: {str(e)}")
            # 回退到选择第一个可用模型
            if updated_retrained_models:
                best_model_name = list(updated_retrained_models.keys())[0]
                perf = {'AUC_Test': 0, 'AUC_Train': 0, 'AUC_Gap': 0, 'AIC': float('inf'), 'AIC_Based_Score': 0}
                best_feature_count = models_with_optimal_features.get(best_model_name, {}).get('best_feature_count', 'all')
                print(f"  Falling back to first available model: {best_model_name}")
            else:
                print("  No models available for selection")
                best_model_name = None
                perf = None
                best_feature_count = 'all'
    
    # 确保返回安全，即使best_model_name为None
    if best_model_name and best_model_name in updated_retrained_models:
        best_model = updated_retrained_models[best_model_name]
    elif updated_retrained_models:
        # 回退到第一个可用模型
        best_model_name = list(updated_retrained_models.keys())[0]
        best_model = updated_retrained_models[best_model_name]
        print(f"  Final fallback to first available model: {best_model_name}")
    else:
        best_model = None
        print("  No models available for return")
    
    return best_model_name, best_model, model_performances

# -------------------------- 14. SHAP Analysis --------------------------
def perform_shap_analysis(best_model_name, best_model, X_train, X_test, y_test, X_test_non_scaled=None, export_dir=None):
    """Perform enhanced SHAP analysis on the best model"""
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
        elif best_model_name in ['SVM', 'Naive Bayes', 'Decision Tree', 'Neural Network']:
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

# -------------------------- 15. Feature Selection Based on SHAP Importance --------------------------
def feature_selection_by_shap(best_model_name, best_model, X_train, X_test, y_train, y_test, 
                             X_train_non_scaled=None, X_test_non_scaled=None, 
                             feature_importance_df=None, export_dir=None, best_model_test_auc=None):
    """Perform enhanced feature selection based on SHAP importance scores with detailed performance tracking"""
    if feature_importance_df is None:
        print("No feature importance data available for selection.")
        return
    
    print(f"\n=== Performing Enhanced Feature Selection for {best_model_name} Based on SHAP Importance ===")
    print("This process shows how model performance changes as we add features based on SHAP values from most to least important.")
    
    # 获取按重要性排序的特征名称
    sorted_features = feature_importance_df['Feature'].tolist()
    total_features = len(sorted_features)
    
    # 存储结果
    selection_results = []
    best_auc = 0
    best_feature_count = 0
    best_features = []
    
    # 为了与测试集评估保持一致性，对全特征集使用完全相同的模型实例
    if best_model_test_auc is not None:
        print(f"\nUsing original test set evaluation for consistency. Test set AUC: {best_model_test_auc:.4f}")
        
    # 创建一个具有最佳参数的新模型实例
    if best_model_name == "XGBoost":
        # 创建参数字典，避免random_state重复传递
        model_params = best_model.get_params()
        model_params['random_state'] = RANDOM_SEED  # 确保使用指定的随机种子
        base_model = xgb.XGBClassifier(**model_params)
    elif best_model_name == "Naive Bayes":
        # Naive Bayes不接受random_state参数
        base_model = GaussianNB(**best_model.get_params())
    else:
        base_model_class = best_model.__class__
        # 获取模型参数并更新random_state
        model_params = best_model.get_params()
        model_params['random_state'] = RANDOM_SEED
        base_model = base_model_class(**model_params)
    
    # Evaluate models with increasing number of features
    print("\nProgress: Number of Features | AUC Score | Accuracy | AUC Change")
    print("--------------------------------------------------------------------")
    prev_auc = 0
    
    for num_features in range(1, total_features + 1):
        # Select top N features
        selected_features = sorted_features[:num_features]
        
        # 准备具有选定特征的训练和测试数据
        if best_model_name == "Naive Bayes" and X_train_non_scaled is not None:
            X_train_selected = X_train_non_scaled[selected_features]
            X_test_selected = X_test_non_scaled[selected_features]
        else:
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
        
        # 用选定的特征训练模型
        try:
            if best_model_name == "Naive Bayes":
                # Naive Bayes不接受random_state参数
                model = GaussianNB(**base_model.get_params())
            else:
                # 获取模型参数并更新random_state
                model_params = base_model.get_params()
                model_params['random_state'] = RANDOM_SEED
                model = base_model.__class__(**model_params)
            model.fit(X_train_selected, y_train)
        
            # 在测试集上评估
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, model.predict(X_test_selected))
        
            # Calculate AUC change from previous feature count
            auc_change = auc_score - prev_auc if prev_auc > 0 else auc_score
            prev_auc = auc_score
        
            # Record results with more detailed information
            selection_results.append({
                'Num_Features': num_features,
                'AUC': auc_score,
                'AUC_Change': auc_change,
                'Accuracy': accuracy,
                'Selected_Features': ', '.join(selected_features),
                'Top_Feature': selected_features[0],  # Most important feature in this step
                'Marginal_Feature': selected_features[-1]  # Feature added in this step
            })
        
            # 更新最佳模型
            if auc_score > best_auc or (auc_score == best_auc and num_features < best_feature_count):
                best_auc = auc_score
                best_feature_count = num_features
                best_features = selected_features
        
            # 打印进度，带有变化指示符
            change_indicator = "+" if auc_change > 0 else "" if auc_change == 0 else "-"
            print(f"{num_features:12d} | {auc_score:.4f} | {accuracy:.4f} | {change_indicator}{auc_change:.4f}")
            
            # 存储全特征集AUC用于比较
            if num_features == total_features:
                full_feature_auc = auc_score
        except Exception as e:
            print(f"Error during feature selection with {num_features} features: {str(e)}")
            
    # 确保full_feature_auc有值，即使全特征集评估失败
    if 'full_feature_auc' not in locals():
        full_feature_auc = best_model_test_auc if best_model_test_auc is not None else 0
    
    # 将结果转换为DataFrame
    df_selection_results = pd.DataFrame(selection_results)
    
    # Save detailed feature selection results
    if export_dir:
        detailed_results_path = os.path.join(export_dir, f'{best_model_name}_feature_selection_detailed_results.xlsx')
        df_selection_results.to_excel(detailed_results_path, index=False)
        print(f"\nDetailed feature selection results saved to: {detailed_results_path}")
    
    # Generate feature selection performance curve
    if export_dir and not df_selection_results.empty:
        # 为图表创建特征名称标签
        feature_labels = [f'{i}. {sorted_features[i-1]}' for i in range(1, total_features + 1)]
        
        plt.figure(figsize=(14, 7))  # 增加宽度以适应特征名称
        plt.plot(feature_labels, df_selection_results['AUC'], 'o-', linewidth=2, color='blue')
        plt.axhline(y=full_feature_auc, color='red', linestyle='--', alpha=0.5, label=f'Full Features AUC ({full_feature_auc:.4f})')
        # 找到最佳特征对应的索引位置
        best_feature_idx = best_feature_count - 1
        plt.axvline(x=best_feature_idx, color='green', linestyle='--', alpha=0.5, label=f'Best Feature Set ({best_feature_count} features)')
        plt.title(f'{best_model_name} Feature Selection Performance Curve\nBest AUC: {best_auc:.4f} (using {best_feature_count} features)', fontsize=14)
        plt.xlabel('Features (Ranked by SHAP Importance)', fontsize=12)
        plt.ylabel('AUC Score', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45, ha='right')  # 旋转标签以提高可读性
        plt.tight_layout()
        
        # 解释AUC值差异的原因
        print("\nNote: The AUC values in this feature selection curve may differ from the AUC values in the main evaluation plot.")
        print("Reason: This curve shows performance of models retrained with feature subsets, while the main plot shows performance of the original full model.")
        print("Even with the same features, model retraining can lead to slight variations in performance metrics.")
        
        performance_curve_path = os.path.join(export_dir, f'{best_model_name}_feature_selection_performance_curve.png')
        plt.savefig(performance_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature selection performance curve saved: {performance_curve_path}")
    
    # 保存最佳特征集
    if export_dir and best_features:
        best_feature_set = pd.DataFrame({'Feature': best_features, 'Rank': range(1, len(best_features) + 1)})
        best_feature_set_path = os.path.join(export_dir, f'{best_model_name}_best_feature_set.xlsx')
        best_feature_set.to_excel(best_feature_set_path, index=False)
        print(f"Best feature set saved: {best_feature_set_path}")
    
    # Generate feature selection summary
    if export_dir:
        summary_data = {
            'Model': [best_model_name],
            'Total_Features': [total_features],
            'Best_Feature_Count': [best_feature_count],
            'Best_AUC': [best_auc],
            'Full_Feature_AUC': [full_feature_auc],
            'AUC_Improvement': [best_auc - full_feature_auc],
            'Top_3_Features': [', '.join(best_features[:3]) if len(best_features) >= 3 else ', '.join(best_features)]
        }
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(export_dir, f'{best_model_name}_feature_selection_summary.xlsx')
        df_summary.to_excel(summary_path, index=False)
        print(f"Feature selection summary saved: {summary_path}")
    
    return best_features

# -------------------------- 16. Visualization of Inner Loop Feature Selection Results --------------------------
def visualize_inner_loop_feature_selection(model_name, feature_selection_results, export_dir=None):
    """Visualize feature selection results from inner loop with AUC change curves"""
    if not feature_selection_results:
        print(f"No feature selection results available for {model_name}.")
        return
    
    print(f"\n=== Visualizing Inner Loop Feature Selection Results for {model_name} ===")
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(feature_selection_results)
    
    # For each fold and parameter combination, plot the AUC curve
    if export_dir:
        # First, create a summary by averaging across folds and parameters
        df_summary = df_results.groupby('Num_Features').agg({
            'AUC': ['mean', 'std']
        }).reset_index()
        df_summary.columns = ['Num_Features', 'Mean_AUC', 'Std_AUC']
        
        # Generate summary plot
        plt.figure(figsize=(12, 7))
        
        # Plot mean AUC curve with error bars
        plt.plot(df_summary['Num_Features'], df_summary['Mean_AUC'], 'o-', linewidth=2, color='blue',
                 label='Mean AUC across folds and parameters')
        plt.fill_between(df_summary['Num_Features'], 
                        df_summary['Mean_AUC'] - df_summary['Std_AUC'], 
                        df_summary['Mean_AUC'] + df_summary['Std_AUC'], 
                        alpha=0.2, color='blue', label='±1 Std Dev')
        
        # Find best feature count (where mean AUC is highest)
        best_idx = df_summary['Mean_AUC'].idxmax()
        best_feature_count = df_summary.loc[best_idx, 'Num_Features']
        best_auc = df_summary.loc[best_idx, 'Mean_AUC']
        
        # Add reference lines
        plt.axhline(y=best_auc, color='red', linestyle='--', alpha=0.5, 
                   label=f'Best Mean AUC: {best_auc:.4f}')
        plt.axvline(x=best_feature_count, color='green', linestyle='--', alpha=0.5, 
                   label=f'Optimal Feature Count: {best_feature_count}')
        
        # Add titles and labels
        plt.title(f'{model_name} Inner Loop Feature Selection - AUC vs. Number of Features\n'
                 f'(Average across {len(df_results.Outer_Fold.unique())} outer folds and {len(df_results.Parameters.unique())} parameter sets)',
                 fontsize=14)
        plt.xlabel('Number of Features', fontsize=12)
        plt.ylabel('AUC Score', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the plot
        curve_path = os.path.join(export_dir, f'{model_name}_inner_loop_feature_selection_auc_curve.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Inner loop feature selection AUC curve saved: {curve_path}")
        
        # Save detailed results to Excel
        detailed_results_path = os.path.join(export_dir, f'{model_name}_inner_loop_feature_selection_details.xlsx')
        df_results.to_excel(detailed_results_path, index=False)
        print(f"Detailed inner loop feature selection results saved: {detailed_results_path}")
        
        # Save summary to Excel
        summary_path = os.path.join(export_dir, f'{model_name}_inner_loop_feature_selection_summary.xlsx')
        df_summary.to_excel(summary_path, index=False)
        print(f"Summary of inner loop feature selection results saved: {summary_path}")

# -------------------------- 17. Create Timestamped Directory --------------------------
def create_timestamped_directory(base_name):
    """Create timestamped directory for consistent output organization"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = f"{base_name}_{timestamp}"
    
    # 如果目录已存在，添加计数器后缀
    counter = 1
    temp_dir = export_dir
    while os.path.exists(temp_dir):
        temp_dir = f"{base_name}_{timestamp}_{counter}"
        counter += 1
    
    export_dir = temp_dir
    os.makedirs(export_dir)
    print(f"New output directory created: {export_dir}")
    return export_dir

# -------------------------- 17. Main Function --------------------------
def main():
    """Unified main function that coordinates the entire parameter tuning, model evaluation, and SHAP analysis process"""
    print("===== Starting Unified Model Parameter Tuning, Evaluation, and SHAP Analysis ====")
    
    # Configure matplotlib font to support display
    plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
    # Create main output directory
    main_export_dir = create_timestamped_directory("unified_tuning_results")
    
    # ===================== Parameter Selection Section =====================
    print("\n\n===== Starting Parameter Selection Process ====")
    # 1. 加载和预处理数据
    processed_datasets = load_and_preprocess_data()
    
    # 2. Initialize Models and Parameter Grids
    models, param_grids, param_counts = init_models_and_params()
    
    # 3. Run Nested Cross-Validation for Grid Search
    X_train_scaled = processed_datasets['X_train_scaled']
    y_train = processed_datasets['y_train']
    X_train_non_scaled = processed_datasets['X_train_non_scaled']
    
    best_models, df_cv_details, df_grid_summary, all_feature_selection_results = run_nested_cv(
        X_train_scaled, y_train, models, param_grids, param_counts, X_train_non_scaled, main_export_dir
    )
    
    # 4. 在完整训练集上重训练最佳模型
    retrained_models = retrain_best_models(best_models, X_train_scaled, y_train, X_train_non_scaled)

    # ===================== Parameter Optimization and Model Evaluation Section =====================
    print("\n\n===== Starting Parameter Optimization and Model Evaluation ====")
    # Prepare test data for curve generation
    X_test_scaled = processed_datasets['X_test_scaled']
    y_test = processed_datasets['y_test']
    X_test_non_scaled = processed_datasets['X_test_non_scaled']

    # 从嵌套交叉验证结果中找到平均外部验证AUC最高的模型
    print("\n=== Finding Best Model by Nested CV Average AUC ===")
    # 按照Mean_Outer_AUC降序排序
    sorted_cv_results = df_grid_summary.sort_values('Mean_Outer_AUC', ascending=False)
    best_model_name_by_cv = sorted_cv_results.iloc[0]['Model']
    best_mean_outer_auc = sorted_cv_results.iloc[0]['Mean_Outer_AUC']

    print(f"Models sorted by Nested CV Mean Outer AUC:")
    for _, row in sorted_cv_results.iterrows():
        print(f"  {row['Model']}: Mean Outer AUC = {row['Mean_Outer_AUC']:.4f} (±{row['Std_Outer_AUC']:.4f})")

    print(f"\nBest Model by Nested CV: {best_model_name_by_cv}, Mean Outer AUC = {best_mean_outer_auc:.4f}")

    # 提取每个模型的内部循环最佳特征数量
    inner_loop_best_feature_counts = {}
    for model_name in retrained_models.keys():
        if model_name in all_feature_selection_results and all_feature_selection_results[model_name]:
            df_results = pd.DataFrame(all_feature_selection_results[model_name])
            df_summary = df_results.groupby('Num_Features').agg({'AUC': ['mean', 'std']}).reset_index()
            df_summary.columns = ['Num_Features', 'Mean_AUC', 'Std_AUC']
            best_idx = df_summary['Mean_AUC'].idxmax()
            best_feature_count = df_summary.loc[best_idx, 'Num_Features']
            inner_loop_best_feature_counts[model_name] = best_feature_count
            print(f"Model {model_name}: Best feature count from inner loop = {best_feature_count}")
        else:
            inner_loop_best_feature_counts[model_name] = X_train_scaled.shape[1]  # 默认使用所有特征
            print(f"Model {model_name}: No inner loop feature selection results, using all features")

    # 使用嵌套交叉验证结果选择最佳模型（基于内部循环最佳特征数量）
    best_model_name, best_model, model_performances = identify_best_model(
        retrained_models, X_test_scaled, y_test, X_train_scaled, y_train, X_test_non_scaled, best_model_name_by_cv, inner_loop_best_feature_counts
    )
    
    # 5. 在所有数据集上评估模型（使用经过特征选择的模型）
    # 从identify_best_model函数中，我们可以通过model_performances获取每个模型的最佳特征数量
    # 重新运行evaluate_models_on_all_datasets，确保Feature_Count列使用最佳特征数量
    from collections import defaultdict
    
    # 创建一个新的字典，包含使用最佳特征子集的模型
    updated_retrained_models = {}
    for model_name, model in retrained_models.items():
        # 这里我们需要重新创建使用最佳特征数量的模型
        # 由于我们无法直接获取identify_best_model中创建的updated_retrained_models
        # 我们重新执行特征选择过程
        best_feature_count = inner_loop_best_feature_counts[model_name]
        
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importances = np.abs(model.coef_[0])
            elif hasattr(model, 'dual_coef_'):
                feature_importances = np.abs(model.dual_coef_[0])
            else:
                # 如果无法获取特征重要性，则使用所有特征
                updated_retrained_models[model_name] = model
                continue
            
            # 选择最重要的特征
            if best_feature_count > len(feature_importances):
                best_feature_count = len(feature_importances)
                
            top_feature_indices = np.argsort(feature_importances)[::-1][:best_feature_count]
            
            # 提取最佳特征子集
            X_train_optimal = X_train_scaled[:, top_feature_indices]
            
            # 重新训练模型
            if model_name == "XGBoost":
                new_model = xgb.XGBClassifier(**model.get_params())
            elif model_name == "Naive Bayes":
                new_model = GaussianNB(**model.get_params())
            else:
                new_model_class = model.__class__
                new_model = new_model_class(**model.get_params())
            
            new_model.fit(X_train_optimal, y_train)
            updated_retrained_models[model_name] = new_model
        except Exception:
            # 如果特征选择失败，使用原始模型
            updated_retrained_models[model_name] = model
    
    # 在所有数据集上评估经过特征选择的模型
    df_evaluation = evaluate_models_on_all_datasets(updated_retrained_models, processed_datasets, main_export_dir)
    
    # 计算测试集上的曲线数据（使用经过特征选择的模型）
    print("\n=== Calculating Curve Data on Test Set ===")
    roc_results = calculate_roc_curve(updated_retrained_models, X_test_scaled, y_test, X_test_non_scaled)
    pr_results = calculate_pr_curve(updated_retrained_models, X_test_scaled, y_test, X_test_non_scaled)
    calib_results = calculate_calibration_curve(updated_retrained_models, X_test_scaled, y_test, X_test_non_scaled)
    
    # 6. Visualize inner loop feature selection results
    print("\n\n===== Visualizing Inner Loop Feature Selection Results ====")
        
    # 只对最终确定的最佳模型生成内部循环特征选择AUC曲线
    if best_model_name in all_feature_selection_results:
        visualize_inner_loop_feature_selection(
            best_model_name, 
            all_feature_selection_results[best_model_name], 
            main_export_dir
        )
        print(f"Visualized inner loop feature selection for final best model: {best_model_name}")
    else:
        print(f"No feature selection results found for final best model: {best_model_name}")
        
    # Generate and save all curves with best model highlighted
    print("\n=== Generating and Saving All Curves ====")
    plot_roc_curve(roc_results, os.path.join(main_export_dir, 'roc_curve.png'))
    plot_pr_curve(pr_results, y_test, os.path.join(main_export_dir, 'pr_curve.png'))
    plot_calibration_curve(calib_results, os.path.join(main_export_dir, 'calibration_curve.png'))
    plot_dca_curve(updated_retrained_models, X_test_scaled, y_test, os.path.join(main_export_dir, 'dca_curve.png'), X_test_non_scaled)
        
    # Generate confusion matrix for best model
    print("\n=== Generating Confusion Matrix for Best Model ===")
    plot_confusion_matrix(
        updated_retrained_models, 
        X_test_scaled, 
        y_test, 
        main_export_dir, 
        X_test_non_scaled, 
        X_train=X_train_scaled, 
        y_train=y_train, 
        best_model_name_by_cv=best_model_name,  # 使用通过嵌套交叉验证选择的最佳模型名称
        inner_loop_best_feature_counts=inner_loop_best_feature_counts  # 添加内部循环最佳特征数量参数
    )
        
    # ===================== SHAP Analysis Section =====================
    print("\n\n===== Starting SHAP Analysis Process ====")
        
    # 执行SHAP分析
    feature_importance_df = perform_shap_analysis(
        best_model_name, best_model, X_train_scaled, X_test_scaled, y_test,
        X_test_non_scaled, main_export_dir
    )
        
    # Perform feature selection based on SHAP importance
    if feature_importance_df is not None:
        # 添加安全检查，确保best_model_name存在于model_performances字典中
        if best_model_name and best_model_name in model_performances:
            best_test_auc = model_performances[best_model_name]['AUC_Test']
        else:
            # 如果不存在，使用默认值0.5（AUC的随机性能）
            best_test_auc = 0.5
            print(f"Warning: Cannot access performance data for model {best_model_name}, using default AUC value of 0.5")
        
        best_features = feature_selection_by_shap(
            best_model_name, best_model, X_train_scaled, X_test_scaled, y_train, y_test,
            X_train_non_scaled, X_test_non_scaled, feature_importance_df, 
            main_export_dir, best_test_auc
        )
        
    # Output all generated files
    print("\n=== All Generated Files ===")
    for file in sorted(os.listdir(main_export_dir)):
        print(f"- {os.path.join(main_export_dir, file)}")
        
    print("\nUnified model parameter tuning, evaluation, and SHAP analysis has been successfully completed!")

# Execute main process
if __name__ == "__main__":
    main()