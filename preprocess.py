import numpy as np
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from utils import print_memory_usage


def preprocessing(X_train, X_val, X_test, y_train, y_val, y_test, output_encoder, file_path, version, sampling_method='smote', plot_distributions=True):
        
        ## Data balance Plotting
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
            plt.figure(figsize=(3.5, 3.0))
            plt.bar(x = labels, height = label_frequencies, color='#0072B2')
            plt.xlabel('Attack Types')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(file_path, version + '_' + 'attack_type_distribution.pdf'), format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()

        ## Sampling
        print_memory_usage(f"Before Sampling ({sampling_method})")
        
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
            print_memory_usage(f"After Sampling ({sampling_method})")
        else:
            X_resampled, y_resampled = X_train, y_train
            print_memory_usage("No Sampling applied")

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
            plt.figure(figsize=(3.5, 3.0))
            plt.bar(x = labels, height = label_frequencies, color='#0072B2')
            plt.xlabel('Attack Types')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(file_path, version + '_' + f'attack_type_distribution_with_{sampling_method}.pdf'), format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()

        # Return numpy arrays for compatibility with PyTorch DataLoaders
        # We skip PCA to preserve the features selected by the Voting mechanism
        X_train_out = (X_resampled.values if hasattr(X_resampled, 'values') else X_resampled).astype(np.float32)
        X_val_out = (X_val.values if hasattr(X_val, 'values') else X_val).astype(np.float32)
        X_test_out = (X_test.values if hasattr(X_test, 'values') else X_test).astype(np.float32)
        
        return X_train_out, X_val_out, X_test_out, y_resampled, y_val, y_test