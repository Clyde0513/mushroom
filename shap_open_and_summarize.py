import joblib
import pandas as pd
import numpy as np
import sys

MODEL_PATH = 'best_mushroom_model_oof.joblib'
ENG_CSV = 'mushroom_engineered.csv'

print('Loading model and engineered data...')
try:
    # Ensure TargetEncoderOOF class is available in this process namespace so
    # joblib/pickle can resolve the class if the saved pipeline referenced
    # __main__.TargetEncoderOOF (common when saved from a notebook).
    try:
        from mushroom_transformers import TargetEncoderOOF
        import __main__ as _main
        setattr(_main, 'TargetEncoderOOF', TargetEncoderOOF)
    except Exception:
        # fallback: ignore if module not present; load may still succeed
        pass

    model = joblib.load(MODEL_PATH)
except Exception as e:
    print('ERROR: could not load model:', e, file=sys.stderr)
    sys.exit(2)

try:
    df = pd.read_csv(ENG_CSV)
except Exception as e:
    print('ERROR: could not load engineered CSV:', e, file=sys.stderr)
    sys.exit(2)

cat_for_te = ['cap_shape_surface', 'odor_capcolor']
num_features = ['odor_freq', 'odor_poison_rate', 'has_stalk_root']
featurenames = [f'{c}_te_transform' for c in cat_for_te] + num_features

print('Preparing transformed feature DataFrame...')
X = df[cat_for_te + num_features].copy()

try:
    pre = model.named_steps['pre']
    X_trans = pre.transform(X)
except Exception as e:
    print('ERROR: transform failed:', e, file=sys.stderr)
    sys.exit(3)

X_arr = np.asarray(X_trans)
if X_arr.ndim == 3 and X_arr.shape[2] == 1:
    X_arr = X_arr.squeeze(2)

if X_arr.ndim != 2 or X_arr.shape[1] != len(featurenames):
    print(f'WARN: transformed shape {X_arr.shape} != expected (*,{len(featurenames)})', file=sys.stderr)

Xdf = pd.DataFrame(X_arr, columns=featurenames, index=df.index)

# compute shap
print('Computing SHAP values...')
try:
    import shap
    clf = model.named_steps['clf']
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xdf)
except Exception as e:
    print('ERROR computing SHAP:', e, file=sys.stderr)
    sys.exit(4)

# normalize to 2D shap array sv
if isinstance(shap_values, list):
    sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) -> pick last class
        if sv.shape[0] == Xdf.shape[0] and sv.shape[1] == Xdf.shape[1]:
            sv = sv[:, :, -1]
        elif sv.shape[0] != Xdf.shape[0] and sv.shape[1] == Xdf.shape[0] and sv.shape[2] == Xdf.shape[1]:
            sv = sv[-1, :, :]
        else:
            sv = sv[..., -1]

sv = np.asarray(sv, dtype=float)
if sv.shape != Xdf.shape:
    if sv.T.shape == Xdf.shape:
        sv = sv.T
    else:
        print('ERROR: final shap array shape mismatch', sv.shape, Xdf.shape, file=sys.stderr)
        sys.exit(5)

# feature importance
mean_abs = np.abs(sv).mean(axis=0)
imp = pd.DataFrame({'feature': Xdf.columns, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
print('\nTop features by mean(|SHAP|):')
print(imp.head(10).to_string(index=False))

# For top 3 features produce numeric dependence summaries
from scipy.stats import pearsonr, spearmanr

n_top = min(3, Xdf.shape[1])
print('\nPer-feature dependence summaries (top features):')
for i, row in imp.head(n_top).iterrows():
    fname = row['feature']
    fvals = Xdf[fname].values.astype(float)
    shap_f = sv[:, Xdf.columns.get_loc(fname)]
    # correlations
    try:
        p_corr = pearsonr(fvals, shap_f)[0]
    except Exception:
        p_corr = np.nan
    try:
        s_corr = spearmanr(fvals, shap_f).correlation
    except Exception:
        s_corr = np.nan
    # binned means
    try:
        bins = np.percentile(fvals, [0,25,50,75,100])
        # ensure unique bin edges
        bins = np.unique(bins)
        if len(bins) > 1:
            digit = np.digitize(fvals, bins, right=True)
            grp = pd.DataFrame({'feat': fvals, 'shap': shap_f, 'bin': digit})
            bmeans = grp.groupby('bin')['feat','shap'].agg(['mean','count']).round(4)
        else:
            bmeans = pd.DataFrame({'feat_mean':[fvals.mean()],'shap_mean':[shap_f.mean()]})
    except Exception:
        bmeans = None

    direction = 'increasing' if (np.nanmean(shap_f) > 0) else 'decreasing'
    print(f"\n- {fname}: mean|shap|={row['mean_abs_shap']:.4f}, mean(shap)={np.nanmean(shap_f):.4f} ({direction})")
    print(f"  Pearson(shap,feat)={p_corr:.3f}, Spearman(shap,feat)={s_corr:.3f}")
    if isinstance(bmeans, pd.DataFrame):
        print('  Binned feature vs SHAP (per-bin mean and count):')
        # print a compact representation
        try:
            if 'feat' in bmeans.columns:
                be = bmeans
            else:
                be = bmeans
            print(be.to_string())
        except Exception:
            print('   (failed to tabulate binned means)')

# Interaction notes: compute correlation matrix of shap contributions between top-5 features
k = min(5, Xdf.shape[1])
top_feats = imp.head(k)['feature'].tolist()
shap_subset = pd.DataFrame(sv[:, [Xdf.columns.get_loc(c) for c in top_feats]], columns=top_feats)
shap_corr = shap_subset.corr()
print('\nSHAP-value correlation between top features:')
print(shap_corr.round(3).to_string())
