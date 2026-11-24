"""
Exploratory Data Analysis Generator for NOCHILL-WAVAX (Dune ID: 4834833)
duneml_s3_eda.py
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
import umap
import phate
import config

def compute_r2(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 10:
        return 0.0
    x_clean, y_clean = x[mask], y[mask]
    if np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return 0.0
    r, _ = stats.pearsonr(x_clean, y_clean)
    return float(r ** 2)

def compute_robust_r2(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 20:
        return 0.0
    x_c = x[mask]
    y_c = y[mask]
    # Outlier trimmed correlation
    q_low_x, q_high_x = np.percentile(x_c, [2.5, 97.5])
    q_low_y, q_high_y = np.percentile(y_c, [2.5, 97.5])
    inlier_mask = (x_c >= q_low_x) & (x_c <= q_high_x) & (y_c >= q_low_y) & (y_c <= q_high_y)
    if inlier_mask.sum() < 10:
        return compute_r2(x, y)
    r, _ = stats.pearsonr(x_c[inlier_mask], y_c[inlier_mask])
    return float(r ** 2)

def run_eda_pipeline():
    print(f"Loading training data from {config.TRAINING_DATA_CSV}...")
    df = pd.read_csv(config.TRAINING_DATA_CSV)
    
    targets = [
        'future_return_1h', 'future_return_6h', 'future_return_12h',
        'log_future_return_1h', 'log_future_return_6h', 'log_future_return_12h'
    ]
    
    # Identify feature columns
    excluded_cols = set(targets + ['time', 'price', 'log_price', 'log_return'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in excluded_cols]
    
    print(f"Identified {len(features)} engineered numeric features across {len(targets)} targets.")

    # Compute 3D UMAP and PHATE embeddings on key features
    print("Computing 3D UMAP and PHATE manifold embeddings...")
    embedding_features = [
        c for c in features if not c.endswith('_v') and not c.startswith('bin_')
    ][:25] # Select top 25 non-velocity base features
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[embedding_features].fillna(0.0))
    
    # Downsample for manifold computation stability if needed
    reducer_umap = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_emb = reducer_umap.fit_transform(X_scaled)
    df['umap_1'] = umap_emb[:, 0]
    df['umap_2'] = umap_emb[:, 1]
    df['umap_3'] = umap_emb[:, 2]
    
    reducer_phate = phate.PHATE(n_components=3, k=15, a=20, t='auto', n_jobs=1, verbose=0, random_state=42)
    phate_emb = reducer_phate.fit_transform(X_scaled)
    df['phate_1'] = phate_emb[:, 0]
    df['phate_2'] = phate_emb[:, 1]
    df['phate_3'] = phate_emb[:, 2]
    
    print("Computing correlation matrix and redundancy detection (|corr| > 0.85)...")
    corr_df = df[features].corr().fillna(0.0)
    redundant_pairs = []
    seen_pairs = set()
    for i, col1 in enumerate(features):
        for j, col2 in enumerate(features):
            if i >= j:
                continue
            c_val = float(corr_df.loc[col1, col2])
            if abs(c_val) > 0.85:
                redundant_pairs.append({
                    "feature1": col1,
                    "feature2": col2,
                    "correlation": round(c_val, 4)
                })
    redundant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    # Prepare correlation dictionary
    corr_dict = {}
    for col1 in features:
        corr_dict[col1] = {col2: round(float(corr_df.loc[col1, col2]), 4) for col2 in features}
        
    print("Computing per-target metrics and histograms...")
    feature_metrics = {}
    hist_data = {}
    
    for tgt in targets:
        y = df[tgt].values
        pos_mask = y > 0
        neg_mask = y <= 0
        
        feature_metrics[tgt] = {}
        hist_data[tgt] = {}
        
        # Fast MI computation on subset of features or full
        mi_scores = mutual_info_regression(df[features].fillna(0.0), y, random_state=42, n_neighbors=3)
        
        for idx, feat in enumerate(features):
            x = df[feat].values
            
            # Linear & Robust R2
            r2_val = compute_r2(x, y)
            robust_r2_val = compute_robust_r2(x, y)
            mi_val = float(mi_scores[idx])
            
            x_pos = x[pos_mask]
            x_neg = x[neg_mask]
            
            # Kruskal-Wallis test
            if len(x_pos) > 5 and len(x_neg) > 5 and np.var(x_pos) > 1e-12 and np.var(x_neg) > 1e-12:
                try:
                    stat, pval = stats.kruskal(x_pos, x_neg)
                    anova_p = float(pval)
                except Exception:
                    anova_p = 1.0
            else:
                anova_p = 1.0
                
            pos_mean = float(np.mean(x_pos)) if len(x_pos) > 0 else 0.0
            pos_var = float(np.var(x_pos)) if len(x_pos) > 0 else 0.0
            neg_mean = float(np.mean(x_neg)) if len(x_neg) > 0 else 0.0
            neg_var = float(np.var(x_neg)) if len(x_neg) > 0 else 0.0
            
            feature_metrics[tgt][feat] = {
                "r2": round(r2_val, 6),
                "robust_r2": round(robust_r2_val, 6),
                "mi": round(mi_val, 6),
                "anova_p": anova_p,
                "pos_mean": round(pos_mean, 6),
                "pos_var": round(pos_var, 6),
                "neg_mean": round(neg_mean, 6),
                "neg_var": round(neg_var, 6)
            }
            
            # Histogram generation (1%-trimmed)
            p1 = np.percentile(x, 1)
            p99 = np.percentile(x, 99)
            if p1 >= p99:
                p99 = p1 + 1.0
                
            bins = np.linspace(p1, p99, 16)
            hist_bins = []
            for b_idx in range(len(bins) - 1):
                b_start = bins[b_idx]
                b_end = bins[b_idx + 1]
                in_bin = (x >= b_start) & (x < b_end if b_idx < len(bins) - 2 else x <= b_end)
                pos_c = int(np.sum(in_bin & pos_mask))
                neg_c = int(np.sum(in_bin & neg_mask))
                hist_bins.append({
                    "range": f"{b_start:.3f} - {b_end:.3f}",
                    "pos": pos_c,
                    "neg": neg_c
                })
            hist_data[tgt][feat] = hist_bins
            
    print("Computing PCA pairwise and manifold pairwise R² scores...")
    pca_base_cols = [f'pca_pnl_{i+1}' for i in range(config.N_PCA_COMP)]
    pca_vel_cols = [f'pca_pnl_{i+1}_v' for i in range(config.N_PCA_COMP)]
    
    r2_groups = {
        "pca_base": [],
        "pca_velocity": [],
        "umap_phate": []
    }
    
    # Base PCA pairs
    for i in range(len(pca_base_cols)):
        for j in range(i + 1, len(pca_base_cols)):
            f1, f2 = pca_base_cols[i], pca_base_cols[j]
            r2_val = compute_r2(df[f1].values, df[f2].values)
            r2_groups["pca_base"].append({
                "group": "BASE",
                "label": f"[BASE] {f1} vs {f2}",
                "f1": f1,
                "f2": f2,
                "r2": round(r2_val, 6)
            })
            
    # Velocity PCA pairs
    for i in range(len(pca_vel_cols)):
        for j in range(i + 1, len(pca_vel_cols)):
            f1, f2 = pca_vel_cols[i], pca_vel_cols[j]
            r2_val = compute_r2(df[f1].values, df[f2].values)
            r2_groups["pca_velocity"].append({
                "group": "VELOCITY",
                "label": f"[VELOCITY] {f1} vs {f2}",
                "f1": f1,
                "f2": f2,
                "r2": round(r2_val, 6)
            })
            
    # Manifold pairs
    for emb_name, p_cols in [("UMAP", ['umap_1', 'umap_2', 'umap_3']), ("PHATE", ['phate_1', 'phate_2', 'phate_3'])]:
        for i in range(len(p_cols)):
            for j in range(i + 1, len(p_cols)):
                f1, f2 = p_cols[i], p_cols[j]
                r2_val = compute_r2(df[f1].values, df[f2].values)
                r2_groups["umap_phate"].append({
                    "group": emb_name,
                    "label": f"[{emb_name}] {f1} vs {f2}",
                    "f1": f1,
                    "f2": f2,
                    "r2": round(r2_val, 6)
                })

    print("Generating sampled scatter data and PnL bin timeline...")
    # Sample 240 rows evenly for scatter visualizations
    sample_step = max(1, len(df) // 240)
    df_sample = df.iloc[::sample_step].copy()
    
    scatter_cols = ['time', 'price'] + targets + features + ['umap_1', 'umap_2', 'umap_3', 'phate_1', 'phate_2', 'phate_3']
    scatter_data = df_sample[scatter_cols].to_dict(orient="records")
    
    bin_cols = [c for c in df.columns if c.startswith('bin_')]
    bin_timeline = df_sample[['time', 'price'] + bin_cols].to_dict(orient="records")
    
    eda_payload = {
        "targets": targets,
        "features": features,
        "corr_features": features,
        "scatter_data": scatter_data,
        "feature_metrics": feature_metrics,
        "hist_data": hist_data,
        "r2_groups": r2_groups,
        "correlation_matrix": corr_dict,
        "redundant_pairs": redundant_pairs,
        "bin_features": bin_cols,
        "bin_timeline": bin_timeline
    }
    
    with open(config.EDA_FEATURES_JSON, 'w', encoding='utf-8') as f:
        json.dump(eda_payload, f, indent=2)
    print(f"Saved complete EDA payload to {config.EDA_FEATURES_JSON}")

if __name__ == "__main__":
    run_eda_pipeline()
