import pandas as pd
import numpy as np
import joblib
import sys

from pathlib import Path

OUT_DIR = Path('.')
ENG_CSV = Path('mushroom_engineered.csv')
MODEL_PATH = Path('best_mushroom_model_oof.joblib')

if not ENG_CSV.exists():
    print(f'ERROR: engineered CSV not found at {ENG_CSV}.', file=sys.stderr)
    sys.exit(2)
if not MODEL_PATH.exists():
    print(f'ERROR: saved model not found at {MODEL_PATH}.', file=sys.stderr)
    sys.exit(2)

print('Loading engineered data and model...')
df = pd.read_csv(ENG_CSV)

# Use the reusable transformer implementation so joblib can unpickle pipelines
from mushroom_transformers import TargetEncoderOOF

model = joblib.load(MODEL_PATH)

# feature lists must match notebook
cat_for_te = ['cap_shape_surface', 'odor_capcolor']
num_features = ['odor_freq', 'odor_poison_rate', 'has_stalk_root']
featurenames = [f'{c}_te_transform' for c in cat_for_te] + num_features

# Prepare X for transformation
X = df[cat_for_te + num_features].copy()
y = df['label_num'].astype('int').copy()

# Transform using pipeline preprocessor
try:
    X_trans = model.named_steps['pre'].transform(X)
except Exception as e:
    print('ERROR: pipeline preprocessor transform failed:', e, file=sys.stderr)
    sys.exit(3)

# Ensure 2D array and build DataFrame
X_arr = np.asarray(X_trans)
if X_arr.ndim != 2 or X_arr.shape[1] != len(featurenames):
    print(f'ERROR: transformed array shape {X_arr.shape} != expected (*, {len(featurenames)})', file=sys.stderr)
    # try to coerce common shapes
    try:
        # if shape is (n_samples, n_features, 1) squeeze
        if X_arr.ndim == 3 and X_arr.shape[2] == 1:
            X_arr = X_arr.squeeze(2)
    except Exception:
        pass

Xdf = pd.DataFrame(X_arr, columns=featurenames, index=X.index)

# Correlation checks
corr_feats = ['odor_capcolor_te_transform', 'odor_poison_rate', 'odor_freq']
print('\nPearson correlation:')
print(Xdf[corr_feats].corr(method='pearson').to_string())
print('\nSpearman correlation:')
print(Xdf[corr_feats].corr(method='spearman').to_string())

# SHAP dependence plot for top feature
try:
    import shap
    import matplotlib.pyplot as plt
    print('\nComputing SHAP values (this may take a moment)...')
    clf = model.named_steps['clf']
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xdf)
    # normalize to 2D array of shap values for class of interest (class-1)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 2:
            sv = shap_arr
        elif shap_arr.ndim == 3:
            # (n_samples, n_features, n_classes) -> pick last class
            if shap_arr.shape[0] == Xdf.shape[0] and shap_arr.shape[1] == Xdf.shape[1]:
                sv = shap_arr[:, :, -1]
            elif shap_arr.shape[0] != Xdf.shape[0] and shap_arr.shape[1] == Xdf.shape[0] and shap_arr.shape[2] == Xdf.shape[1]:
                sv = shap_arr[-1, :, :]
            else:
                sv = shap_arr[..., -1]
        else:
            raise ValueError('unexpected shap_values ndim')
    sv = np.asarray(sv)
    # dependence plot for odor_capcolor_te_transform
    dep_png = OUT_DIR / 'shap_dependence_odor_capcolor.png'
    plt.figure(figsize=(6,4))
    shap.dependence_plot('odor_capcolor_te_transform', sv, Xdf, show=False)
    plt.tight_layout()
    plt.savefig(dep_png, dpi=150)
    plt.close()
    print('Saved SHAP dependence plot to', dep_png)

    # SHAP summary plot (bar) for top features
    try:
        summary_png = OUT_DIR / 'shap_summary_bar.png'
        plt.figure(figsize=(6,4))
        shap.summary_plot(sv, Xdf, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(summary_png, dpi=150)
        plt.close()
        print('Saved SHAP summary bar plot to', summary_png)
    except Exception as e:
        print('SHAP summary bar plot failed:', e, file=sys.stderr)

    # dependence plots for top 2 features by mean(|SHAP|)
    try:
        import numpy as _np
        mean_abs = _np.abs(sv).mean(axis=0)
        top_idx = list(_np.argsort(mean_abs)[-2:][::-1])
        top_features = [Xdf.columns[i] for i in top_idx]
        for fname in top_features:
            fn_png = OUT_DIR / f'shap_dependence_{fname}.png'
            plt.figure(figsize=(6,4))
            shap.dependence_plot(fname, sv, Xdf, show=False)
            plt.tight_layout()
            plt.savefig(fn_png, dpi=150)
            plt.close()
            print('Saved SHAP dependence plot to', fn_png)
    except Exception as e:
        print('SHAP top-feature dependence plots failed:', e, file=sys.stderr)
except Exception as e:
    print('SHAP dependence plot failed:', e, file=sys.stderr)

# Misclassification by odor_readable (use same train/test split as notebook)
try:
    from sklearn.model_selection import train_test_split
    X_model = df[cat_for_te + num_features].copy()
    y_model = df['label_num'].astype('int').copy()
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42, stratify=y_model)
    preds = model.predict(X_test)
    test_idx = X_test.index
    test_df = df.loc[test_idx].copy()
    test_df['y_true'] = y_test.values
    test_df['y_pred'] = preds
    test_df['misclassified'] = test_df['y_true'] != test_df['y_pred']
    # group by odor_readable
    grp = test_df.groupby('odor_readable').agg(
        total_test=('misclassified','size'),
        misclassified_count=('misclassified','sum')
    )
    grp['misclassification_rate'] = (grp['misclassified_count'] / grp['total_test']).round(3)
    grp = grp.sort_values(['misclassification_rate','misclassified_count'], ascending=[False, False])
    print('\nMisclassification summary by odor_readable (test set):')
    print(grp.head(20).to_string())
    # show a few sample misclassified rows
    mis_samples = test_df[test_df['misclassified']].copy()
    if len(mis_samples) > 0:
        print('\nExamples of misclassified samples (first 8):')
        cols_show = ['odor_readable','cap_color_readable','label','y_true','y_pred']
        for c in cols_show:
            if c not in mis_samples.columns:
                cols_show.remove(c)
        print(mis_samples[cols_show].head(8).to_string(index=False))
    else:
        print('\nNo misclassifications on the test set.')
except Exception as e:
    print('Misclassification analysis failed:', e, file=sys.stderr)

except Exception as e:
    print('Misclassification analysis failed:', e, file=sys.stderr)

# --- Additional sanity checks requested ---
try:
    print('\nTransformed feature sample (first 6 rows):')
    print(Xdf[featurenames].head(6).to_string())
    print('\nTransformed feature summary:')
    print(Xdf[featurenames].describe().to_string())
    te_col = 'odor_capcolor_te_transform'
    if te_col in Xdf.columns:
        print(f"\nUnique count for {te_col}:", Xdf[te_col].nunique())
        print(f"Sample unique values (first 10): {list(Xdf[te_col].unique()[:10])}")
    else:
        print(f"\n{te_col} not in transformed DataFrame columns")

    # Confusion matrix on test set
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, preds)
    print('\nConfusion matrix (rows=true, cols=predicted):')
    print(cm)
except Exception as e:
    print('Sanity checks failed:', e, file=sys.stderr)

print('\nDone.')
