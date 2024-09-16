from scipy.stats import pearsonr, ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

def compute_categorical_similarity(col_real, col_synthetic):
    try:
        p_real = pd.Series(col_real).value_counts(normalize=True)
        p_synthetic = pd.Series(col_synthetic).value_counts(normalize=True)
        u = (p_real * np.log(p_real / p_synthetic)).sum()
        return 1 - u
    except Exception as e:
        print(f"Error in compute_categorical_similarity: {e}")
        return 0

def column_similarity(real_data, synthetic_data):
    try:
        similarities = []
        for col_real, col_synthetic in zip(real_data, synthetic_data):
            correlation, _ = pearsonr(col_real, col_synthetic)
            similarities.append(correlation)
        return np.mean(similarities)
    except Exception as e:
        print(f"Error in column_similarity: {e}")
        return 0

def correlation_similarity(real_data, synthetic_data):
    try:
        real_corr = np.corrcoef(real_data, rowvar=False)
        synthetic_corr = np.corrcoef(synthetic_data, rowvar=False)
        correlation, _ = pearsonr(real_corr.flatten(), synthetic_corr.flatten())
        return correlation
    except Exception as e:
        print(f"Error in correlation_similarity: {e}")
        return 0

def jensen_shannon_similarity(real_data, synthetic_data):
    try:
        similarities = []
        for col_real, col_synthetic in zip(real_data.T, synthetic_data.T):
            p_real = np.histogram(col_real, bins=10, density=True)[0]
            p_synthetic = np.histogram(col_synthetic, bins=10, density=True)[0]
            similarity = 1 - jensenshannon(p_real, p_synthetic)
            similarities.append(similarity)
        return np.mean(similarities)
    except Exception as e:
        print(f"Error in jensen_shannon_similarity: {e}")
        return 0

def kolmogorov_smirnov_similarity(real_data, synthetic_data):
    try:
        similarities = []
        for col_real, col_synthetic in zip(real_data.T, synthetic_data.T):
            similarity, _ = ks_2samp(col_real, col_synthetic)
            similarities.append(1 - similarity)
        return np.mean(similarities)
    except Exception as e:
        print(f"Error in kolmogorov_smirnov_similarity: {e}")
        return 0

def propensity_mean_absolute_similarity(real_data, synthetic_data):
    try:
        X = np.vstack([real_data, synthetic_data])
        y = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        error = mean_absolute_error(y_test, y_pred_proba)
        return 1 - error
    except Exception as e:
        print(f"Error in propensity_mean_absolute_similarity: {e}")
        return 0

def resemblance_measure(real_data, synthetic_data):
    global n
    n = 0
    resemblance_score = 0
    
    for func in [
        column_similarity, 
        correlation_similarity, 
        jensen_shannon_similarity, 
        kolmogorov_smirnov_similarity, 
        propensity_mean_absolute_similarity
    ]:
        result = func(real_data, synthetic_data)
        if result != 0:
            resemblance_score += result
            n += 1
    
    # Avoid division by zero
    if n == 0:
        print("No resemblance measures could be computed.")
        return 0

    resemblance_score /= n
    print(f"Resemblance Score: {resemblance_score}")
    return resemblance_score