import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import time

import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif, chi2, VarianceThreshold, SequentialFeatureSelector, SelectFromModel, f_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def apply_ieee_style():
    """Applies IEEE academic style to matplotlib plots."""
    plt.rcParams.update({
        'figure.figsize': (3.5, 3.0),
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'axes.prop_cycle': plt.cycler(color=['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])
    })

apply_ieee_style()

def print_memory_usage(step=""):
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2
        print(f"[Memory] {step}: {mem:.2f} MB")
    except ImportError:
        print("[Memory] psutil not installed. Run 'pip install psutil' to monitor memory.")

def clear_memory(device=None):
    """Forcibly clears system memory and PyTorch cache (CUDA/MPS)."""
    gc.collect()
    if device:
        device_type = device.split(':')[0] if isinstance(device, str) else device.type
        if device_type == 'cuda':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif device_type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

def clean_database(db_path, image_save_path=None, do_scale=True, scaler_type='standard', fix_skewness=False, split_ratios=(0.8, 0.9)):
    # low_memory=false handles mixed types
    df = pd.read_csv(db_path, low_memory=False)

    # 0. DROP ARTIFACTS EARLY
    # Drop 'Unnamed' columns (index artifacts) immediately so they don't prevent duplicate removal
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        print(f"[Preprocessing] Dropping artifact columns: {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)

    # Remove duplicate rows to prevent data leakage and bias
    print(f"Original shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    # Drop rows where the target is missing (essential for supervised learning)
    df.dropna(subset=['Attack Type'], inplace=True)
    print(f"Shape after removing duplicates: {df.shape}")

    # 1. TARGETS: Must be removed from X (features) because they are the answer (y) or proxies to it.
    target_cols = ['Attack Type', 'Label', 'Attack Tool']

    # 2. METADATA & TOPOLOGY: Specific to the testbed (e.g., Row ID, Timestamp, VLANs).
    # We drop these to prevent "Shortcut Learning". Feature selection algorithms often select these
    # because they accidentally correlate with the target in a static dataset, but they fail in real networks.
    metadata_cols = ['Seq', 'RunTime', 'sVid', 'dVid', 'SrcTCPBase', 'DstTCPBase', 'sHops', 'dHops']
    
    drop_cols = target_cols + metadata_cols
    
    # Only drop columns that actually exist in the dataset to avoid errors
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=existing_drop_cols, axis=1)
    y = df['Attack Type']
    
    print("\n[Dataset Info] Attack Type Distribution (Counts):")
    print(y.value_counts())
    print(f"[Preprocessing] Final feature set ({len(X.columns)}): {list(X.columns)}")
    
    del df
    
    # Replace infinite values (e.g., in 'Rate' columns) with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    db_len = len(y)
    indices = np.arange(db_len)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    
    split_1 = int(db_len * split_ratios[0])
    split_2 = int(db_len * split_ratios[1])
    train_indices = indices[:split_1]
    test_indices = indices[split_1 : split_2]
    val_indices = indices[split_2:]
    
    X_train = X.iloc[train_indices].copy()
    y_train = y.iloc[train_indices].copy()
    X_val = X.iloc[val_indices].copy()
    y_val = y.iloc[val_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_test = y.iloc[test_indices].copy()
    
    # Verify that the split preserves class distribution
    print("\n[Verification] Class distribution (normalized) across splits:")
    dist_comparison = pd.DataFrame({
        'Original': y.value_counts(normalize=True),
        'Train': y_train.value_counts(normalize=True),
        'Val': y_val.value_counts(normalize=True),
        'Test': y_test.value_counts(normalize=True)
    })
    print(dist_comparison)
    
    # Visualize the class distribution to confirm stratification
    ax = dist_comparison.plot(kind='bar', figsize=(3.5, 3.5), width=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_xlabel('Attack Type', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.legend(title='Dataset Split')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if image_save_path:
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        plt.savefig(os.path.join(image_save_path, 'class_distribution_check.pdf'))
        print(f"Saved distribution plot to {os.path.join(image_save_path, 'class_distribution_check.pdf')}")

    plt.show()
    plt.close()
    
    # Free up memory by deleting the large original DataFrames
    print_memory_usage("Before GC")
    del X, y
    gc.collect()
    print_memory_usage("After GC")

    # Impute missing values with median for numerical columns
    # CRITICAL: Calculate median ONLY on Training data to avoid leakage
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    train_medians = X_train[numeric_cols].median()
    
    X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
    X_val[numeric_cols] = X_val[numeric_cols].fillna(train_medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(train_medians)

    # Fallback: If a column was all NaNs, median is NaN. Fill remaining numerical NaNs with 0.
    X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
    X_val[numeric_cols] = X_val[numeric_cols].fillna(0)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

    # Fill remaining (categorical) NaNs with a placeholder
    X_train.fillna('Unknown', inplace=True)
    X_val.fillna('Unknown', inplace=True)
    X_test.fillna('Unknown', inplace=True)

    # Encoding non-numerical data
    columns_to_treat = []
    for column in X_train.columns:
        if X_train[column].dtype == 'O':
            columns_to_treat.append(column)
            
    for column in columns_to_treat:
        # Fit encoder on all unique values to ensure consistent mapping and avoid crashes on unseen labels
        # (This is a minor acceptable compromise for static datasets)
        all_values = pd.concat([X_train[column], X_val[column], X_test[column]]).astype(str)
        encoder = LabelEncoder()
        encoder.fit(all_values)
        
        X_train[column] = encoder.transform(X_train[column].astype(str))
        X_val[column] = encoder.transform(X_val[column].astype(str))
        X_test[column] = encoder.transform(X_test[column].astype(str))
        
    output_encoder = LabelEncoder()
    # Fit output encoder on all targets (safe as targets are usually known)
    output_encoder.fit(pd.concat([y_train, y_val, y_test]))
    y_train = output_encoder.transform(y_train)
    y_val = output_encoder.transform(y_val)
    y_test = output_encoder.transform(y_test)
    
    # --- Preprocessing: Skewness Correction OR Scaling ---
    # These are typically mutually exclusive. PowerTransformer with standardize=True already
    # scales the data (mean=0, std=1). Applying another scaler can distort the distribution.
    if fix_skewness and do_scale:
        print("[Warning] Both 'fix_skewness' and 'do_scale' are True. "
              "PowerTransformer will be used as it also standardizes the data.")

    if fix_skewness:
        print("[Preprocessing] Applying PowerTransformer (Yeo-Johnson) to fix skewness and standardize data...")
        
        # Visualization: Before PowerTransformer
        if image_save_path:
            skew_vals = X_train[numeric_cols].skew().abs().sort_values(ascending=False)
            top_skewed_cols = skew_vals.head(3).index.tolist()
            
            if top_skewed_cols:
                print(f"[Visualization] Plotting top 3 skewed features before transformation: {top_skewed_cols}")
                plt.figure(figsize=(3.5, 6))
                for i, col in enumerate(top_skewed_cols):
                    plt.subplot(3, 1, i+1)
                    sns.histplot(X_train[col], kde=True, color='#D55E00', bins=30)
                plt.tight_layout()
                plt.savefig(os.path.join(image_save_path, 'distribution_before_pt.pdf'))
                plt.show()
                plt.close()

        # Yeo-Johnson handles positive/negative values and standardizes data (0 mean, 1 std)
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train[numeric_cols] = pt.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = pt.transform(X_val[numeric_cols])
        X_test[numeric_cols] = pt.transform(X_test[numeric_cols])

        # Visualization: After PowerTransformer
        if image_save_path and 'top_skewed_cols' in locals() and top_skewed_cols:
            print(f"[Visualization] Plotting top 3 skewed features after transformation")
            plt.figure(figsize=(3.5, 6))
            for i, col in enumerate(top_skewed_cols):
                plt.subplot(3, 1, i+1)
                sns.histplot(X_train[col], kde=True, color='#0072B2', bins=30)
            plt.tight_layout()
            plt.savefig(os.path.join(image_save_path, 'distribution_after_pt.pdf'))
            plt.show()
            plt.close()

    elif do_scale:
        print(f"[Preprocessing] Applying {scaler_type} scaling...")
        if scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            # Default to MinMaxScaler
            scaler = MinMaxScaler()
            
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Ensure float32 type
    for column in X_train.columns:
        if X_train[column].dtype != 'object':
            X_train[column] = X_train[column].astype('float32')
            X_val[column] = X_val[column].astype('float32')
            X_test[column] = X_test[column].astype('float32')
            
    X_train.info()

    return X_train, X_val, X_test, y_train, y_val, y_test, output_encoder

def analyze_correlations(X, file_path=None, version='v1', threshold=0.95):
    """
    Plots the correlation matrix and identifies highly correlated features.
    """
    plt.figure(figsize=(4, 3.5))
    corr_matrix = X.corr()
    
    # Plot heatmap
    # Mask the upper triangle to make it easier to read
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    ax = sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, square=True, linewidths=0.5, linecolor='gray', cbar=True)
    
    ax.grid(False)
    plt.xticks(rotation=90, ha='center', fontsize=8, fontname='Times New Roman')
    plt.yticks(fontsize=8, fontname='Times New Roman')
    
    if file_path:
        plt.savefig(os.path.join(file_path, f"{version}_correlation_matrix.pdf"), bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    # Identify collinear features
    corr_abs = corr_matrix.abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"\n[Analysis] Features with correlation > {threshold}:")
    for col in to_drop:
        correlated_cols = upper.index[upper[col] > threshold].tolist()
        print(f"  - {col} is correlated with {correlated_cols}")
        
    return to_drop

def print_evaluation_metrics(y_val, y_pred, training_time, prediction_time, output_encoder, file_path, version, results_file_name, cm_title, notes=""):
    # Ensure inputs are numpy arrays (handle DataFrames/Series)
    if hasattr(y_val, 'values'):
        y_val = y_val.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    classification_rep = classification_report(y_val, y_pred, target_names=output_encoder.classes_, digits=8, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    print(f"Accuracy: {accuracy:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall: {recall:.8f}")
    print(f"F1 Score: {f1:.8f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"latency per sample: {prediction_time/len(y_val):.8f} seconds")
    print(f"\nClassification Report: \n{classification_rep}")

    # Get current time for the log
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_safe = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    full_path = os.path.join(file_path, version + '_' + results_file_name)

    # 'a' mode creates the file if missing, appends if present
    with open(full_path, 'a') as f:
        # 1. Add a separator and Timestamp so you know WHEN this run happened
        f.write(f"{'='*90}\n")
        f.write(f"Experiment Run: {timestamp}\n")
        f.write(f"{'='*90}\n")
        
        # Add notes section
        if notes:
            f.write(f"Notes:\n{notes}\n\n")

        # 2. Write your metrics
        f.write(f"Accuracy: {accuracy:.8f}\n")
        f.write(f"Precision: {precision:.8f}\n")
        f.write(f"Recall: {recall:.8f}\n")
        f.write(f"F1 Score: {f1:.8f}\n")
        f.write(f"Training Time: {training_time:.4f} seconds\n")
        f.write(f"Prediction Time: {prediction_time:.4f} seconds\n")
        f.write(f"Latency per sample: {prediction_time/len(y_val):.8f} seconds\n")
        f.write(f"\nClassification Report: \n{classification_rep}\n")
        f.write(f"\nConfusion Matrix:\n{np.array2string(cm, separator=', ')}\n")

        # 3. Add a blank line at the end for spacing
        f.write("\n")
        
    # Plot confusion matrix
    plt.figure(figsize=(4, 3.5))
    
    # IEEE Style Heatmap
    ax = sns.heatmap(cm, annot=True, fmt='d', cbar=True, cmap='Blues',
                xticklabels=output_encoder.classes_, yticklabels=output_encoder.classes_,
                annot_kws={"size": 6}, linewidths=0.5, linecolor='gray')
    
    # Explicitly remove grid (crucial for heatmaps when global grid is on)
    ax.grid(False)
    
    plt.xlabel('Predicted Class', fontsize=10, fontname='Times New Roman')
    plt.ylabel('True Class', fontsize=10, fontname='Times New Roman')
    plt.xticks(rotation=90, ha='center', fontsize=8, fontname='Times New Roman')
    plt.yticks(fontsize=8, fontname='Times New Roman')
    
    # Save confusion matrix plot
    # Use timestamp to prevent overwriting and include model identifier from results filename
    model_id = os.path.splitext(results_file_name)[0]
    plt.savefig(os.path.join(file_path, f"{version}_{model_id}_{timestamp_safe}_confusion_matrix.pdf"), format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    return accuracy, precision, recall, f1

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=128):
    # Ensure inputs are numpy arrays (handle DataFrames)
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_val, 'values'):
        X_val = X_val.values

    # Reshape to (N, 1, F) for consistency across LSTM, GRU, CNN
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    
    train_features = torch.tensor(X_train_reshaped, dtype=torch.float32)
    train_targets = torch.tensor(y_train, dtype=torch.long)
    val_features = torch.tensor(X_val_reshaped, dtype=torch.float32)
    val_targets = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_and_evaluate_pytorch_model(model, train_loader, val_loader, num_epochs=10, device='cpu', model_name="Model", use_amp=False):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Determine device type for AMP (Automatic Mixed Precision)
    if isinstance(device, str):
        device_type = device.split(':')[0]
    else:
        device_type = device.type

    # GradScaler is primarily for CUDA. For MPS (Mac M-series), standard float32 is often fast enough.
    # If using CUDA, we initialize the scaler.
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device_type == 'cuda') else None
    
    if use_amp and device_type == 'mps':
        print("[Warning] Mixed Precision (AMP) enabled on MPS. Gradient Scaling is disabled, which may cause instability. Recommend use_amp=False for M-series chips.")
    
    start_time = time.time()
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixed Precision Context
            # Use generic torch.autocast which supports 'cuda', 'cpu', and 'mps' (PyTorch 2.1+)
            if use_amp:
                # MPS/CUDA use float16, CPU uses bfloat16 usually
                amp_dtype = torch.float16 if device_type != 'cpu' else torch.bfloat16
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale gradients to prevent underflow in float16
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc_history.append(correct / total)
        
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_acc_history.append(correct_val / total_val)
        
    training_time = time.time() - start_time
    
    start_time = time.time()
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
    prediction_time = time.time() - start_time
    y_pred = np.array(all_preds)
    
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.close()
    
    # Clear GPU/MPS cache after training to release memory back to the OS
    clear_memory(device)
    
    return y_pred, training_time, prediction_time

# --------------------------------------------------------------------------------------
# The following functions are implemented inside the (compare_feature_selection_methods) function
def perform_filter_feature_selection(X_train, y_train, n_features_to_select=20, method='anova'):
    """
    Performs Filter-based feature selection.
    Filter methods are faster than RFE but evaluate features independently.
    Supported methods: 'pearson', 'anova', 'chi2', 'mutual_info', 'variance', 'fisher'
    """
    print(f"\n[Filter] Starting {method.upper()} Feature Selection to select top {n_features_to_select} features...")
    start_time = time.time()
    
    if method == 'pearson':
        # f_regression is based on correlation between feature and target
        score_func = f_regression
    elif method == 'anova':
        score_func = f_classif
    elif method == 'mutual_info':
        score_func = mutual_info_classif
    elif method == 'chi2':
        score_func = chi2
        # Chi2 requires non-negative values
        if (X_train < 0).any().any():
            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            selector = SelectKBest(score_func=score_func, k=n_features_to_select)
            selector.fit(X_train_scaled, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()
            elapsed_time = time.time() - start_time
            print(f"[Filter] Completed in {elapsed_time:.2f} seconds.")
            print(f"[Filter] Selected Features: {selected_features}")
            return selected_features, elapsed_time
        
    elif method == 'variance':
        # VarianceThreshold doesn't take k, so we find the threshold manually
        variances = X_train.var()
        threshold = variances.sort_values(ascending=False).iloc[n_features_to_select-1]
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        elapsed_time = time.time() - start_time
        print(f"[Filter] Completed in {elapsed_time:.2f} seconds.")
        print(f"[Filter] Selected Features: {selected_features}")
        return selected_features, elapsed_time
    elif method == 'fisher':
        # Manual Fisher Score implementation
        # F = sum(n_i * (mu_i - mu)^2) / sum(n_i * var_i)
        classes = np.unique(y_train)
        scores = []
        for col in X_train.columns:
            feature = X_train[col]
            mu = feature.mean()
            numerator = 0
            denominator = 0
            for c in classes:
                feat_c = feature[y_train == c]
                n_c = len(feat_c)
                mu_c = feat_c.mean()
                var_c = feat_c.var() if len(feat_c) > 1 else 0
                numerator += n_c * (mu_c - mu)**2
                denominator += n_c * var_c
            score = numerator / denominator if denominator != 0 else 0
            scores.append(score)
        
        indices = np.argsort(scores)[::-1][:n_features_to_select]
        selected_features = X_train.columns[indices].tolist()
        elapsed_time = time.time() - start_time
        print(f"[Filter] Completed in {elapsed_time:.2f} seconds.")
        print(f"[Filter] Selected Features: {selected_features}")
        return selected_features, elapsed_time
    else:
        raise ValueError(f"Unknown filter method: {method}")
        
    selector = SelectKBest(score_func=score_func, k=n_features_to_select)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    elapsed_time = time.time() - start_time
    print(f"[Filter] Completed in {elapsed_time:.2f} seconds.")
    print(f"[Filter] Selected Features: {selected_features}")
    
    return selected_features, elapsed_time

def perform_wrapper_feature_selection(X_train, y_train, n_features_to_select=20, method='rfe'):
    """
    Performs Wrapper-based feature selection.
    Supported methods: 'forward', 'backward', 'stepwise', 'rfe', 'genetic', 'annealing'
    """
    print(f"\n[Wrapper] Starting {method.upper()} Feature Selection to select top {n_features_to_select} features...")
    start_time = time.time()
    # Use DecisionTree instead of RandomForest for Wrapper methods to significantly reduce runtime
    estimator = DecisionTreeClassifier(random_state=42)
    
    if method == 'rfe':
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=0.1)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_].tolist()
        
    elif method == 'forward':
        sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction='forward', cv=2, n_jobs=-1)
        sfs.fit(X_train, y_train)
        selected_features = X_train.columns[sfs.get_support()].tolist()
        
    elif method == 'backward':
        sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction='backward', cv=2, n_jobs=-1)
        sfs.fit(X_train, y_train)
        selected_features = X_train.columns[sfs.get_support()].tolist()
        
    elif method == 'stepwise':
            # FIX: Use mlxtend for true Stepwise (Floating Forward) Selection
            from mlxtend.feature_selection import SequentialFeatureSelector as SFS
            
            # floating=True turns on "Stepwise" (Forward add, then conditional remove)
            sfs = SFS(estimator, 
                    k_features=n_features_to_select, 
                    forward=True, 
                    floating=True,  # <--- This makes it Stepwise
                    scoring='accuracy',
                    cv=2,
                    n_jobs=-1)
            
            sfs = sfs.fit(X_train, y_train)
            selected_features = list(sfs.k_feature_names_)
        
    elif method == 'genetic':
            # UPGRADE: Parameters increased for scientific validity
            population_size = 50  # Was 10 (Too small)
            generations = 20      # Was 3 (Too short)
            mutation_rate = 0.1
            
            n_features = X_train.shape[1]
            # Initialize random population
            population = [np.random.choice([0, 1], size=n_features, p=[0.8, 0.2]) for _ in range(population_size)]
            
            best_score = 0
            best_mask = population[0]
            
            for gen in range(generations):
                scores = []
                for individual in population:
                    cols = [i for i, x in enumerate(individual) if x == 1]
                    if len(cols) == 0: 
                        scores.append(0)
                        continue
                    
                    # Check to prevent errors if 0 features selected
                    X_subset = X_train.iloc[:, cols]
                    estimator.fit(X_subset, y_train)
                    scores.append(estimator.score(X_subset, y_train))
                
                # Elitism: Keep best 2
                sorted_idx = np.argsort(scores)[::-1]
                best_gen_score = scores[sorted_idx[0]]
                
                # Track global best
                if best_gen_score > best_score:
                    best_score = best_gen_score
                    best_mask = population[sorted_idx[0]]
                
                # Simple Tournament Selection & Crossover
                new_pop = [population[i] for i in sorted_idx[:2]] # Keep top 2
                
                while len(new_pop) < population_size:
                    # Tournament size 3
                    parents = []
                    for _ in range(2):
                        candidates = np.random.choice(len(population), 3, replace=False)
                        parent_idx = candidates[np.argmax([scores[i] for i in candidates])]
                        parents.append(population[parent_idx])
                    
                    p1, p2 = parents
                    cut = np.random.randint(0, n_features)
                    child = np.concatenate([p1[:cut], p2[cut:]])
                    
                    # Mutation
                    if np.random.rand() < mutation_rate:
                        m_idx = np.random.randint(0, n_features)
                        child[m_idx] = 1 - child[m_idx]
                    new_pop.append(child)
                population = new_pop
                
            selected_indices = [i for i, x in enumerate(best_mask) if x == 1]
            # Safety check: if GA selected too many, take top N based on single-feature importance or just truncate
            if len(selected_indices) > n_features_to_select:
                selected_indices = selected_indices[:n_features_to_select]
                
            selected_features = X_train.columns[selected_indices].tolist()

    elif method == 'annealing':
        # Simplified Simulated Annealing
        current_mask = np.zeros(X_train.shape[1])
        current_mask[:n_features_to_select] = 1
        np.random.shuffle(current_mask)
        
        best_mask = current_mask.copy()
        best_score = 0
        
        cols = [i for i, x in enumerate(current_mask) if x == 1]
        estimator.fit(X_train.iloc[:, cols], y_train)
        current_score = estimator.score(X_train.iloc[:, cols], y_train)
        
        for i in range(20): # Limited iterations for speed
            new_mask = current_mask.copy()
            # Flip one bit
            idx = np.random.randint(0, len(new_mask))
            new_mask[idx] = 1 - new_mask[idx]
            
            cols = [i for i, x in enumerate(new_mask) if x == 1]
            if len(cols) == 0: continue
            
            estimator.fit(X_train.iloc[:, cols], y_train)
            new_score = estimator.score(X_train.iloc[:, cols], y_train)
            
            if new_score > current_score:
                current_mask = new_mask
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_mask = new_mask
            else:
                # Acceptance probability
                if np.random.rand() < np.exp((new_score - current_score) * 100):
                    current_mask = new_mask
                    current_score = new_score
                    
        selected_indices = [i for i, x in enumerate(best_mask) if x == 1]
        selected_features = X_train.columns[selected_indices[:n_features_to_select]].tolist()
        
    else:
        raise ValueError(f"Unknown wrapper method: {method}")
        
    elapsed_time = time.time() - start_time
    print(f"[Wrapper] Completed in {elapsed_time:.2f} seconds.")
    print(f"[Wrapper] Selected Features: {selected_features}")
    return selected_features, elapsed_time

def perform_embedded_feature_selection(X_train, y_train, n_features_to_select=20, method='rf'):
    """
    Performs Embedded feature selection.
    Supported methods: 'lasso', 'ridge', 'elastic_net', 'rf', 'gradient_boosting'
    """
    print(f"\n[Embedded] Starting {method.upper()} Feature Selection to select top {n_features_to_select} features...")
    start_time = time.time()
    X_train_to_fit = X_train

    if method == 'lasso':
        # L1 Regularization
        model = LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42, max_iter=10000)
    elif method == 'ridge':
        # L2 Regularization (Ridge) - selects based on coef magnitude
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, random_state=42, max_iter=10000)
    elif method == 'elastic_net':
        # Elastic Net
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1, random_state=42, max_iter=10000)
    elif method == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif method == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown embedded method: {method}")

    # Fit model
    model.fit(X_train_to_fit, y_train)
    
    # Select features
    # Use prefit=True since the model is already trained, which is more efficient.
    selector = SelectFromModel(model, max_features=n_features_to_select, threshold=-np.inf, prefit=True)
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Fallback if too many features selected (SelectFromModel threshold logic can be tricky)
    if len(selected_features) > n_features_to_select:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use the absolute value of coefficients
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(X_train.shape[1])
            
        indices = np.argsort(importances)[::-1][:n_features_to_select]
        selected_features = X_train.columns[indices].tolist()
    
    elapsed_time = time.time() - start_time
    print(f"[Embedded] Completed in {elapsed_time:.2f} seconds.")
    print(f"[Embedded] Selected Features: {selected_features}")
    return selected_features, elapsed_time
# --------------------------------------------------------------------------------------
# The following functions are implemented inside the (perform_voting_feature_selection) function
def compare_feature_selection_methods(X_train, y_train, X_val, y_val, n_features=20, sample_size=None, random_state=42):
    """
    Runs multiple feature selection methods on a sample of data and evaluates them.
    Useful for deciding which method works best for your specific dataset.
    """
    results = {}
    feature_sets = {}
    execution_times = {}
    
    # Set seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Downsample for speed during feature selection (optional)
    if sample_size and sample_size < len(X_train):
        print(f"\n[Comparison] Downsampling training data to {sample_size} samples for feature selection...")
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sel = X_train.iloc[indices].copy()
        y_sel = y_train[indices].copy()
    else:
        print(f"\n[Comparison] Using full training data ({len(X_train)} samples) for feature selection.")
        X_sel = X_train
        y_sel = y_train

    print(f"\n{'='*40}\nComparing Feature Selection Methods\n{'='*40}")
    
    eval_model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    #eval_model = MLPClassifier(hidden_layer_sizes=(128, 96, 64, 48), max_iter=300, activation='relu', solver='adam', alpha=0.005, verbose=True, random_state=random_state)

    # --- Filter Methods ---
    filter_methods = ['pearson', 'anova', 'chi2', 'mutual_info', 'variance', 'fisher']
    for method in filter_methods:
        print(f"\n--- Filter Method: {method} ---")
        try:
            feats, exec_time = perform_filter_feature_selection(X_sel, y_sel, n_features_to_select=n_features, method=method)
            eval_model.fit(X_train[feats], y_train)
            acc = eval_model.score(X_val[feats], y_val)
            results[f'Filter ({method})'] = acc
            feature_sets[f'Filter ({method})'] = feats
            execution_times[f'Filter ({method})'] = exec_time
            print(f"   -> Validation Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"   -> Failed: {e}")

    # --- Wrapper Methods ---
    wrapper_methods = ['forward', 'backward', 'stepwise', 'rfe', 'genetic', 'annealing']
    for method in wrapper_methods:
        print(f"\n--- Wrapper Method: {method} ---")
        feats, exec_time = perform_wrapper_feature_selection(X_sel, y_sel, n_features_to_select=n_features, method=method)
        eval_model.fit(X_train[feats], y_train)
        acc = eval_model.score(X_val[feats], y_val)
        results[f'Wrapper ({method})'] = acc
        feature_sets[f'Wrapper ({method})'] = feats
        execution_times[f'Wrapper ({method})'] = exec_time
        print(f"   -> Validation Accuracy: {acc:.4f}")

    # --- Embedded Methods ---
    embedded_methods = ['lasso', 'ridge', 'elastic_net', 'rf', 'gradient_boosting']
    for method in embedded_methods:
        print(f"\n--- Embedded Method: {method} ---")
        feats, exec_time = perform_embedded_feature_selection(X_sel, y_sel, n_features_to_select=n_features, method=method)
        eval_model.fit(X_train[feats], y_train)
        acc = eval_model.score(X_val[feats], y_val)
        results[f'Embedded ({method})'] = acc
        feature_sets[f'Embedded ({method})'] = feats
        execution_times[f'Embedded ({method})'] = exec_time
        print(f"   -> Validation Accuracy: {acc:.4f}")

    # Summary
    print(f"\n{'='*40}\nSummary of Validation Accuracy\n{'='*40}")
    # Sort results by accuracy
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    for method, score in sorted_results.items():
        print(f"{method}: {score:.4f}")
        
    # Check for similar accuracies and advise
    if len(results) > 1 and np.std(list(results.values())) < 0.001:
        print("\n[Advice] All methods achieved very similar accuracy.")
        print("         This suggests the dataset has strong signal or high redundancy.")
        print("         To differentiate methods, try reducing 'n_features' (e.g., to 5 or 10).")

    return sorted_results, feature_sets, execution_times

def plot_feature_selection_comparison(results, file_path=None, version='v1'):
    """
    Plots the validation accuracy of different feature selection methods.
    """
    if not results:
        print("No results to plot.")
        return

    # Convert dictionary to DataFrame
    df_results = pd.DataFrame(list(results.items()), columns=['Method', 'Accuracy'])
    
    # Sort by Accuracy
    df_results = df_results.sort_values(by='Accuracy', ascending=False)

    # Set plot style (IEEE Standard)
    plt.figure(figsize=(3.5, 4.5)) # Width 3.5 (column), Height 4.5 (for 17 methods)
    
    # Create bar chart (Horizontal for better readability of method names)
    ax = sns.barplot(x='Accuracy', y='Method', data=df_results, color='#0072B2', zorder=2)
    
    plt.xlabel('Validation Accuracy', fontsize=10, fontname='Times New Roman')
    plt.ylabel('Method', fontsize=10, fontname='Times New Roman')
    plt.xticks(fontsize=8, fontname='Times New Roman')
    plt.yticks(fontsize=8, fontname='Times New Roman')
    
    # Set X-axis limit to show differences clearly
    min_acc = df_results['Accuracy'].min()
    if min_acc > 0.9:
        plt.xlim(0.9, 1.01)
    else:
        plt.xlim(0, 1.01)

    # Grid and Spines
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=8, fontname='Times New Roman')
        
    plt.tight_layout()
    
    if file_path:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        filename = f"{version}_feature_selection_comparison.pdf"
        plt.savefig(os.path.join(file_path, filename), format='pdf', bbox_inches='tight')
        print(f"Saved plot to {os.path.join(file_path, filename)}")
        
    plt.show()
    plt.close()
# --------------------------------------------------------------------------------------
# The following functions i can choose from for feature selection and dimensionality reduction
def perform_voting_feature_selection(X_train, y_train, X_val, y_val, n_features=20, sample_size=None, file_path=None, version='v1', random_state=42):
    """
    Selects features based on the best performing method (Accuracy).
    If ties occur, selects the method with the lowest execution time.
    """
    # Run comparison to get scores, feature sets, and execution times
    sorted_results, feature_sets, execution_times = compare_feature_selection_methods(X_train, y_train, X_val, y_val, n_features, sample_size, random_state)
    
    # Plot results if path is provided
    if file_path:
        plot_feature_selection_comparison(sorted_results, file_path, version)
    
    print(f"\n{'='*40}\nSelecting Best Feature Selection Method\n{'='*40}")
    
    # Find max accuracy
    if not sorted_results:
        print("No results found. Returning empty feature list.")
        return [], sorted_results

    max_accuracy = max(sorted_results.values())
    
    # Find all methods with that accuracy (using small epsilon for float comparison)
    candidates = [method for method, acc in sorted_results.items() if acc >= max_accuracy - 1e-9]
    
    print(f"Highest Validation Accuracy: {max_accuracy:.4f}")
    print(f"Candidates with top accuracy: {candidates}")
    
    best_method = candidates[0]
    
    if len(candidates) > 1:
        print(f"Tie detected. Selecting method with lowest execution time...")
        # Sort candidates by execution time
        best_method = min(candidates, key=lambda m: execution_times.get(m, float('inf')))
        min_time = execution_times.get(best_method, 0)
        print(f"Winner: {best_method} (Time: {min_time:.4f}s)")
    else:
        print(f"Winner: {best_method}")

    selected_features = feature_sets[best_method]
    print(f"Selected Features ({len(selected_features)}): {selected_features}")
    
    return selected_features, sorted_results

def log_metrics(model_results, model_name, accuracy, precision, recall, f1, train_time, pred_time):
    """
    Calculates metrics and appends them to the model_results list.
    Uses 'weighted' average for multi-class classification.
    """
    
    # Check if model already exists in list and update it, otherwise append
    for result in model_results:
        if result['Model'] == model_name:
            result.update({
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Training Time (s)': train_time,
                'Prediction Time (s)': pred_time
            })
            return

    model_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training Time (s)': train_time,
        'Prediction Time (s)': pred_time
    })

def plot_individual_metrics(model_results, save_dir=None, version='v1'):
    """
    Plots individual metrics for model comparison.
    """
    if not model_results:
        print("No results to plot.")
        return
        
    df_results = pd.DataFrame(model_results)
    
    # Define metrics to plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)', 'Prediction Time (s)']
    
    # Set style
    
    for metric in metrics:
        if metric not in df_results.columns:
            continue
            
        # Create a new figure for every metric
        plt.figure(figsize=(4, 3.5))
        
        # Create bar chart
        # hue='Model' assigns colors, legend=False hides the redundant legend
        ax = sns.barplot(x='Model', y=metric, data=df_results, hue='Model', legend=False, zorder=2)
        
        # Titles and Labels
        plt.ylabel(metric, fontsize=10, fontname='Times New Roman')
        plt.xlabel('Model', fontsize=10, fontname='Times New Roman')
        plt.xticks(rotation=90, fontsize=8, fontname='Times New Roman', ha='right')
        plt.yticks(fontsize=8, fontname='Times New Roman')
        
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on top of bars
        for container in ax.containers:
            # Use 4 decimal places for accuracy/f1, 2 decimals for time
            fmt = '%.4f' if 'Time' not in metric else '%.2f'
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=8, fontname='Times New Roman')
            
        # Adjust layout
        plt.tight_layout()
        
        # Optional: Save for Thesis/Paper
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = f"{version}_comparison_{metric.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pdf"
            plt.savefig(os.path.join(save_dir, filename), format='pdf', bbox_inches='tight')
            print(f"Saved {filename}")
            
        plt.show()
        plt.close()
