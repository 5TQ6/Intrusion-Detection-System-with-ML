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
            return selected_features
        
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
        return selected_features
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
        return selected_features
    else:
        raise ValueError(f"Unknown filter method: {method}")
        
    selector = SelectKBest(score_func=score_func, k=n_features_to_select)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    elapsed_time = time.time() - start_time
    print(f"[Filter] Completed in {elapsed_time:.2f} seconds.")
    print(f"[Filter] Selected Features: {selected_features}")
    
    return selected_features

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
    return selected_features

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
        model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42, max_iter=10000)
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
    
    return selected_features
# --------------------------------------------------------------------------------------
# The following functions are implemented inside the (perform_voting_feature_selection) function
def compare_feature_selection_methods(X_train, y_train, X_val, y_val, n_features=20, sample_size=None, random_state=42):
    """
    Runs multiple feature selection methods on a sample of data and evaluates them.
    Useful for deciding which method works best for your specific dataset.
    """
    results = {}
    feature_sets = {}
    
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
    
    # Use a simple Random Forest for evaluation
    eval_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    # --- Filter Methods ---
    filter_methods = ['pearson', 'anova', 'chi2', 'mutual_info', 'variance', 'fisher']
    for method in filter_methods:
        print(f"\n--- Filter Method: {method} ---")
        try:
            feats = perform_filter_feature_selection(X_sel, y_sel, n_features_to_select=n_features, method=method)
            eval_model.fit(X_train[feats], y_train)
            acc = eval_model.score(X_val[feats], y_val)
            results[f'Filter ({method})'] = acc
            feature_sets[f'Filter ({method})'] = feats
            print(f"   -> Validation Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"   -> Failed: {e}")

    # --- Wrapper Methods ---
    wrapper_methods = ['forward', 'backward', 'stepwise', 'rfe', 'genetic', 'annealing']
    for method in wrapper_methods:
        print(f"\n--- Wrapper Method: {method} ---")
        feats = perform_wrapper_feature_selection(X_sel, y_sel, n_features_to_select=n_features, method=method)
        eval_model.fit(X_train[feats], y_train)
        acc = eval_model.score(X_val[feats], y_val)
        results[f'Wrapper ({method})'] = acc
        feature_sets[f'Wrapper ({method})'] = feats
        print(f"   -> Validation Accuracy: {acc:.4f}")

    # --- Embedded Methods ---
    embedded_methods = ['lasso', 'ridge', 'elastic_net', 'rf', 'gradient_boosting']
    for method in embedded_methods:
        print(f"\n--- Embedded Method: {method} ---")
        feats = perform_embedded_feature_selection(X_sel, y_sel, n_features_to_select=n_features, method=method)
        eval_model.fit(X_train[feats], y_train)
        acc = eval_model.score(X_val[feats], y_val)
        results[f'Embedded ({method})'] = acc
        feature_sets[f'Embedded ({method})'] = feats
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

    return sorted_results, feature_sets

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
def perform_voting_feature_selection(X_train, y_train, X_val, y_val, n_features=20, sample_size=None, top_k=3, file_path=None, version='v1', random_state=42):
    """
    Selects features based on a majority vote from the top K performing feature selection methods.
    """
    # Run comparison to get scores and feature sets
    sorted_results, feature_sets = compare_feature_selection_methods(X_train, y_train, X_val, y_val, n_features, sample_size, random_state)
    
    # Plot results if path is provided
    if file_path:
        plot_feature_selection_comparison(sorted_results, file_path, version)
    
    # --- Debugging/Verification Block ---
    print(f"\n{'='*40}\n[Debugging] Feature Selection Verification\n{'='*40}")
    
    # 1. Print Feature Lists for Top 5
    debug_top_k = 5
    top_methods_debug = list(sorted_results.keys())[:debug_top_k]
    
    for i, method in enumerate(top_methods_debug):
        feats = feature_sets[method]
        print(f"\nRank {i+1}: {method} (Acc: {sorted_results[method]:.4f})")
        print(f"  Features: {feats}")

    # 2. Check for Identity & 3. Calculate Overlap
    if len(top_methods_debug) >= 2:
        best_method = top_methods_debug[0]
        best_feats = set(feature_sets[best_method])
        
        print(f"\n[Comparison] Comparing against Rank 1 ({best_method}):")
        
        for i in range(1, len(top_methods_debug)):
            current_method = top_methods_debug[i]
            current_feats = set(feature_sets[current_method])
            
            if best_feats == current_feats:
                print(f"  - Rank {i+1} ({current_method}) selected the EXACT same features as Rank 1.")
            else:
                # Jaccard Index
                intersection = len(best_feats.intersection(current_feats))
                union = len(best_feats.union(current_feats))
                jaccard = intersection / union if union > 0 else 0
                print(f"  - Rank {i+1} ({current_method}) overlap: {intersection} common features (Jaccard: {jaccard:.2%})")
    
    print(f"{'='*40}\n")
    # --- End Debugging Block ---

    print(f"\n{'='*40}\nStarting Voting Feature Selection (Top {top_k} Methods)\n{'='*40}")
    
    # Get top k methods
    top_methods = list(sorted_results.keys())[:top_k]
    print(f"Top {top_k} methods selected for voting:")
    for m in top_methods:
        print(f"  - {m} (Accuracy: {sorted_results[m]:.4f})")
    
    # Collect all features from top methods
    all_selected_features = []
    for method in top_methods:
        all_selected_features.extend(feature_sets[method])
        
    # Analyze overlap to explain similar accuracies
    if len(top_methods) >= 2:
        set1 = set(feature_sets[top_methods[0]])
        set2 = set(feature_sets[top_methods[1]])
        overlap = len(set1.intersection(set2))
        print(f"[Voting] Overlap between top 2 methods ({top_methods[0]} & {top_methods[1]}): {overlap}/{n_features} features in common.")

    # Count occurrences
    feature_counts = Counter(all_selected_features)
    
    # Majority vote (>= ceil(top_k / 2))
    majority_threshold = (top_k // 2) + 1
    voting_features = [feat for feat, count in feature_counts.items() if count >= majority_threshold]
    
    print(f"\n[Voting] Features selected by majority ({majority_threshold}+ votes): {voting_features}")
    print(f"[Voting] Total features selected: {len(voting_features)}")
    
    # Return both the selected features and the results for analysis/plotting
    return voting_features, sorted_results
