from scipy.stats import pearsonr, ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def compute_categorical_similarity(col_real, col_synthetic):
    # Compute Theil's U for categorical features
    p_real = pd.Series(col_real).value_counts(normalize=True)
    p_synthetic = pd.Series(col_synthetic).value_counts(normalize=True)
    u = (p_real * np.log(p_real / p_synthetic)).sum()
    return 1 - u
def column_similarity(real_data, synthetic_data):
    similarities = []
    for col_real, col_synthetic in zip(real_data, synthetic_data):
        correlation, _ = pearsonr(col_real, col_synthetic)
        similarity = correlation
        similarities.append(similarity)
    return np.mean(similarities)
def correlation_similarity(real_data, synthetic_data):
    real_corr = np.corrcoef(real_data, rowvar=False)
    synthetic_corr = np.corrcoef(synthetic_data, rowvar=False)
    print(synthetic_corr)
    correlation, _ = pearsonr(real_corr.flatten(), synthetic_corr.flatten())
    return correlation
def jensen_shannon_similarity(real_data, synthetic_data):
    similarities = []
    for col_real, col_synthetic in zip(real_data.T, synthetic_data.T):
        # Compute probability distributions and Jensen-Shannon divergence
        p_real = np.histogram(col_real, bins=10, density=True)[0]
        p_synthetic = np.histogram(col_synthetic, bins=10, density=True)[0]
        similarity = 1 - jensenshannon(p_real, p_synthetic)
        similarities.append(similarity)
    return np.mean(similarities)
def kolmogorov_smirnov_similarity(real_data, synthetic_data):
    similarities = []
    for col_real, col_synthetic in zip(real_data.T, synthetic_data.T):
        # Compute cumulative distributions and Kolmogorov-Smirnov distance
        similarity, _ = ks_2samp(col_real, col_synthetic)
        similarity = 1 - similarity
        similarities.append(similarity)
    return np.mean(similarities)
def propensity_mean_absolute_similarity(real_data, synthetic_data):
    # Train XGBoost classifier to discriminate between real and synthetic samples
    X = np.vstack([real_data, synthetic_data])
    y = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    # Compute mean absolute error of classifier probabilities
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    error = mean_absolute_error(y_test, y_pred_proba)
    return 1 - error