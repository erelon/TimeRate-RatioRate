#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

# LightGBM classifier (install with: pip install lightgbm)
from lightgbm import LGBMClassifier

# Optional: UMAP (install with pip install umap-learn)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed. Run 'pip install umap-learn' for UMAP visualizations.")

RESULTS_DIR = "results"
WINNERS_CSV = os.path.join(RESULTS_DIR, "param_sweep_winners.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)

def load_dataset():
    dfw = pd.read_csv(WINNERS_CSV)

    # Filter out ties - only analyze clear wins
    # Also filter out 3-way ties altogether
    if "outcome" in dfw.columns:
        n_total = len(dfw)
        n_3way = len(dfw[dfw["outcome"] == "tie_3way"])
        n_2way = len(dfw[dfw["outcome"] == "tie_2way"])
        dfw = dfw[dfw["outcome"] == "win"].copy()
        print(f"Filtered out {n_2way} 2-way ties and {n_3way} 3-way ties from {n_total} total trials ({len(dfw)} wins remaining)")

    # Also drop any rows where winner is NaN (extra safety)
    dfw = dfw[dfw["winner"].notna()].copy()

    # Reset index to ensure proper alignment after filtering
    dfw = dfw.reset_index(drop=True)

    # expand params_json into columns
    params = dfw["params_json"].apply(json.loads)
    dfp = pd.json_normalize(params.tolist())
    df = pd.concat([dfw.drop(columns=["params_json"]), dfp], axis=1)

    # label
    y = df["winner"].astype(str)
    # Filter out any "nan" string values that might have slipped through
    valid_mask = (y != "nan") & (y != "None") & (y != "")
    df = df[valid_mask].copy()
    y = y[valid_mask].copy()

    X = df.drop(columns=["winner", "runner_up", "runner_up_avg_rate",
                         "best_avg_rate", "best_converged_at",
                         "outcome", "tied_agents"], errors="ignore")
    # trial_id is an identifier, not a feature
    X = X.drop(columns=["trial_id"], errors="ignore")

    # Remove columns with a single unique value (no variance)
    nunique = X.nunique()
    low_variance_cols = nunique[nunique <= 1].index.tolist()
    if low_variance_cols:
        print(f"Dropping low-variance columns: {low_variance_cols}")
        X = X.drop(columns=low_variance_cols)

    return X, y, df

def make_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

def plot_feature_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series):
    # Fit and extract names (including one-hot expanded)
    pipe.fit(X, y)
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]

    # get feature names
    num_cols = pre.transformers_[0][2]
    cat_pipe = pre.transformers_[1][1]
    cat_cols = pre.transformers_[1][2]
    ohe = cat_pipe.named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feat_names = list(num_cols) + cat_names

    importances = model.feature_importances_
    idx = np.argsort(importances)[-25:]  # top 25

    plt.figure()
    plt.barh(np.array(feat_names)[idx], importances[idx])
    plt.title("Top feature importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lgbm_feature_importance.png"), dpi=200)
    plt.close()

def plot_confusion(pipe: Pipeline, X: pd.DataFrame, y: pd.Series):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    pipe.fit(Xtr, ytr)
    disp = ConfusionMatrixDisplay.from_estimator(pipe, Xte, yte)
    disp.ax_.set_title("Confusion matrix (hold-out)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

def plot_small_tree(X: pd.DataFrame, y: pd.Series):
    pre = make_preprocessor(X)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0)
    pipe = Pipeline([("pre", pre), ("model", tree)])
    pipe.fit(X, y)

    # Plot tree (note: feature names are expanded; keep max_depth small)
    plt.figure(figsize=(18, 10))
    plot_tree(pipe.named_steps["model"], filled=True, fontsize=8)
    plt.title("DecisionTree (depth<=4) — rough regime rules")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "decision_tree.png"), dpi=200)
    plt.close()

    # Also export human-readable rules (on transformed space; still useful)
    rules = export_text(pipe.named_steps["model"])
    with open(os.path.join(OUT_DIR, "decision_tree_rules.txt"), "w", encoding="utf-8") as f:
        f.write(rules)

def plot_pca_map(X: pd.DataFrame, y: pd.Series):
    pre = make_preprocessor(X)
    Z = pre.fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    Z2 = pca.fit_transform(Z.toarray() if hasattr(Z, "toarray") else Z)

    labels = y.astype(str).values
    classes = sorted(np.unique(labels))

    plt.figure()
    for c in classes:
        m = labels == c
        plt.scatter(Z2[m, 0], Z2[m, 1], s=12, alpha=0.7, label=c)
    plt.title("PCA map of env regimes (colored by winner)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_regime_map.png"), dpi=200)
    plt.close()


def plot_lda_map(X: pd.DataFrame, y: pd.Series):
    """
    Linear Discriminant Analysis - specifically designed to maximize class separation.
    Projects data onto axes that maximize between-class variance / within-class variance.
    """
    pre = make_preprocessor(X)
    Z = pre.fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z

    labels = y.astype(str).values
    classes = sorted(np.unique(labels))
    n_classes = len(classes)

    # LDA can have at most n_classes - 1 components
    n_components = min(2, n_classes - 1)

    if n_components < 1:
        print("LDA requires at least 2 classes, skipping LDA plot")
        return

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    Z_lda = lda.fit_transform(Z, labels)

    # Calculate explained variance ratio
    explained_var = lda.explained_variance_ratio_ if hasattr(lda, 'explained_variance_ratio_') else None

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, c in enumerate(classes):
        m = labels == c
        if n_components == 1:
            # 1D plot with jitter for visibility
            plt.scatter(Z_lda[m, 0], np.random.normal(i, 0.1, m.sum()),
                       s=20, alpha=0.7, label=c, color=colors[i])
        else:
            plt.scatter(Z_lda[m, 0], Z_lda[m, 1], s=20, alpha=0.7, label=c, color=colors[i])

    title = "LDA projection (maximizes class separation)"
    if explained_var is not None:
        title += f"\nLD1: {explained_var[0]:.1%}" + (f", LD2: {explained_var[1]:.1%}" if len(explained_var) > 1 else "")
    plt.title(title)
    plt.xlabel("LD1")
    plt.ylabel("LD2" if n_components > 1 else "Class (jittered)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lda_regime_map.png"), dpi=200)
    plt.close()


def plot_tsne_map(X: pd.DataFrame, y: pd.Series):
    """
    t-SNE - good at preserving local structure and revealing clusters.
    """
    pre = make_preprocessor(X)
    Z = pre.fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z

    # For large datasets, use PCA first to speed up t-SNE
    if Z.shape[1] > 50:
        pca = PCA(n_components=50, random_state=0)
        Z = pca.fit_transform(Z)

    # Adjust perplexity based on sample size
    n_samples = Z.shape[0]
    perplexity = min(30, max(5, n_samples // 10))

    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity, n_iter=1000)
    Z_tsne = tsne.fit_transform(Z)

    labels = y.astype(str).values
    classes = sorted(np.unique(labels))

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, c in enumerate(classes):
        m = labels == c
        plt.scatter(Z_tsne[m, 0], Z_tsne[m, 1], s=20, alpha=0.7, label=c, color=colors[i])

    plt.title(f"t-SNE projection (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_regime_map.png"), dpi=200)
    plt.close()


def plot_umap_map(X: pd.DataFrame, y: pd.Series):
    """
    UMAP - often better than t-SNE at preserving both local and global structure.
    """
    if not HAS_UMAP:
        print("Skipping UMAP plot (umap-learn not installed)")
        return

    pre = make_preprocessor(X)
    Z = pre.fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z

    # Adjust n_neighbors based on sample size
    n_samples = Z.shape[0]
    n_neighbors = min(15, max(5, n_samples // 20))

    umap = UMAP(n_components=2, random_state=0, n_neighbors=n_neighbors, min_dist=0.1)
    Z_umap = umap.fit_transform(Z)

    labels = y.astype(str).values
    classes = sorted(np.unique(labels))

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, c in enumerate(classes):
        m = labels == c
        plt.scatter(Z_umap[m, 0], Z_umap[m, 1], s=20, alpha=0.7, label=c, color=colors[i])

    plt.title(f"UMAP projection (n_neighbors={n_neighbors})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "umap_regime_map.png"), dpi=200)
    plt.close()


def plot_3d_pca(X: pd.DataFrame, y: pd.Series):
    """
    3D PCA plot - might reveal structure not visible in 2D.
    """
    from mpl_toolkits.mplot3d import Axes3D

    pre = make_preprocessor(X)
    Z = pre.fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z

    pca = PCA(n_components=3, random_state=0)
    Z3 = pca.fit_transform(Z)

    labels = y.astype(str).values
    classes = sorted(np.unique(labels))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, c in enumerate(classes):
        m = labels == c
        ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], s=20, alpha=0.7, label=c, color=colors[i])

    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained[1]:.1%})')
    ax.set_zlabel(f'PC3 ({explained[2]:.1%})')
    ax.set_title(f"3D PCA (total explained: {sum(explained):.1%})")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_3d_regime_map.png"), dpi=200)
    plt.close()


def plot_cluster_winner_mix(X: pd.DataFrame, y: pd.Series, k: int = 6):
    pre = make_preprocessor(X)
    Z = pre.fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z

    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    cl = km.fit_predict(Z)

    dfc = pd.DataFrame({"cluster": cl, "winner": y.astype(str).values})
    mix = (
        dfc.groupby(["cluster", "winner"]).size()
        .unstack(fill_value=0)
        .sort_index()
    )

    mix_norm = mix.div(mix.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(mix_norm))
    for col in mix_norm.columns:
        vals = mix_norm[col].values
        plt.bar(mix_norm.index.astype(str), vals, bottom=bottom, label=col)
        bottom += vals
    plt.title(f"Winner mix per cluster (k={k})")
    plt.xlabel("Cluster")
    plt.ylabel("Fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cluster_winner_mix.png"), dpi=200)
    plt.close()

def main():
    X, y, df = load_dataset()
    pre = make_preprocessor(X)

    # LightGBM classifier instead of RandomForest
    n_classes = len(np.unique(y))
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=n_classes,
        n_jobs=-1,
        random_state=0,
    )

    pipe = Pipeline([("pre", pre), ("model", lgbm)])

    # Quick sanity: is it predictable at all?
    scores = cross_val_score(pipe, X, y, cv=5)
    with open(os.path.join(OUT_DIR, "cv_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"5-fold CV accuracy (LightGBM): mean={scores.mean():.3f}, std={scores.std():.3f}\n")

    plot_confusion(pipe, X, y)
    plot_feature_importance(pipe, X, y)
    plot_small_tree(X, y)

    # Dimensionality reduction visualizations
    print("Generating PCA plot...")
    plot_pca_map(X, y)

    print("Generating LDA plot (best for classification separation)...")
    plot_lda_map(X, y)

    print("Generating t-SNE plot...")
    plot_tsne_map(X, y)

    print("Generating UMAP plot...")
    plot_umap_map(X, y)

    print("Generating 3D PCA plot...")
    plot_3d_pca(X, y)

    plot_cluster_winner_mix(X, y, k=6)

    print(f"\nWrote visualizations to: {OUT_DIR}")
    print("Files:")
    print("  - lgbm_feature_importance.png")
    print("  - decision_tree.png, decision_tree_rules.txt")
    print("  - confusion_matrix.png")
    print("  - pca_regime_map.png (2D PCA)")
    print("  - lda_regime_map.png (LDA - maximizes class separation)")
    print("  - tsne_regime_map.png (t-SNE - local structure)")
    print("  - umap_regime_map.png (UMAP - local + global structure)")
    print("  - pca_3d_regime_map.png (3D PCA)")
    print("  - cluster_winner_mix.png")


if __name__ == "__main__":
    main()
