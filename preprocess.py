import numpy as np
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from utils import print_memory_usage

def preprocessing(X_train, X_val, X_test, y_train, y_val, y_test, output_encoder, file_path, version, sampling_method='smote', plot_distributions=True):
    
    # --- GLOBAL IEEE STYLE SETTINGS ---
    # Apply this once so all charts look professional
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (3.5, 3.0) # Matches IEEE Column Width
    })

    ## 1. Data Balance Plotting (Before Sampling)
    if plot_distributions:
        label_frequencies = []
        labels = []
        for attack in np.unique(y_train):
            frequency = len(y_train[y_train == attack])
            label_frequencies.append(frequency)
            labels.append(str(output_encoder.inverse_transform(np.expand_dims(np.array(attack), axis = 0))[0]))
        
        sorted_indices = np.argsort(label_frequencies)[::-1]
        label_frequencies = [label_frequencies[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        
        # Create Plot
        fig, ax = plt.subplots() # Use fig, ax for better control
        
        # CRITICAL FIX: Send Grid to Background
        ax.set_axisbelow(True) 
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Plot Bars
        ax.bar(x=labels, height=label_frequencies, color='#0072B2')
        
        # Styling
        ax.set_xlabel('Attack Types')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=90, ha='center') # Vertical labels
        
        # Clean Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(file_path, version + '_' + 'attack_type_distribution.pdf'), format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    ## Sampling Logic
    # (Assuming print_memory_usage is defined elsewhere in your code)
    # print_memory_usage(f"Before Sampling ({sampling_method})") 
    
    if sampling_method == 'smote':
        sampler = SMOTE(random_state=42)
    elif sampling_method == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif sampling_method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        sampler = None

    if sampler:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        # print_memory_usage(f"After Sampling ({sampling_method})")
    else:
        X_resampled, y_resampled = X_train, y_train
        # print_memory_usage("No Sampling applied")

    ## 2. Data Balance Plotting (After Sampling)
    if plot_distributions and sampler:
        label_frequencies = []
        labels = []
        for attack in np.unique(y_resampled):
            frequency = len(y_resampled[y_resampled == attack])
            label_frequencies.append(frequency)
            labels.append(str(output_encoder.inverse_transform(np.expand_dims(np.array(attack), axis = 0))[0]))
        
        sorted_indices = np.argsort(label_frequencies)[::-1]
        label_frequencies = [label_frequencies[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        
        # Create Plot
        fig, ax = plt.subplots()
        
        # CRITICAL FIX: Send Grid to Background
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Plot Bars
        ax.bar(x=labels, height=label_frequencies, color='#0072B2')
        
        # Styling
        ax.set_xlabel('Attack Types')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=90, ha='center')
        
        # Clean Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(file_path, version + '_' + f'attack_type_distribution_with_{sampling_method}.pdf'), format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    # Return numpy arrays
    X_train_out = (X_resampled.values if hasattr(X_resampled, 'values') else X_resampled).astype(np.float32)
    X_val_out = (X_val.values if hasattr(X_val, 'values') else X_val).astype(np.float32)
    X_test_out = (X_test.values if hasattr(X_test, 'values') else X_test).astype(np.float32)

    # Ensure targets are numpy arrays
    y_resampled = y_resampled.values if hasattr(y_resampled, 'values') else y_resampled
    y_val = y_val.values if hasattr(y_val, 'values') else y_val
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    
    print(f"\n[Preprocessing] Final dataset sizes (after {sampling_method}):")
    print(f"  - Training samples: {len(y_resampled)}")
    print(f"  - Validation samples: {len(y_val)}")
    print(f"  - Test samples: {len(y_test)}")
    
    return X_train_out, X_val_out, X_test_out, y_resampled, y_val, y_test