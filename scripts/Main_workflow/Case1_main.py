import os
import pandas as pd
import numpy as np
import yaml
import logging
import shap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib as mpl

from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     LeaveOneOut, KFold, train_test_split)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.inspection import PartialDependenceDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import (make_scorer, r2_score, mean_squared_error)
from sklearn.linear_model import LinearRegression  # for coefficient-based interpretation
from xgboost import XGBRegressor

##############################################################################
# GPU Check
##############################################################################
def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available. Using GPU for computations.")
        return torch.device("cuda")
    else:
        print("GPU is not available. Using CPU for computations.")
        return torch.device("cpu")

##############################################################################
# 1) Load Configuration
##############################################################################
def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configuration_path.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

##############################################################################
# 2) Outlier Detection
##############################################################################
def detect_and_remove_outliers(X_combined_features, config):
    if X_combined_features.empty:
        raise ValueError("Input feature data is empty. Please check columns or dataset.")
    if not config["preprocessing"].get("remove_outliers", True):
        print("Outlier removal disabled in config.")
        return X_combined_features, np.arange(X_combined_features.shape[0])

    n_components = config["preprocessing"].get("pca_components", 2)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(X_combined_features)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA Explained Variance ({n_components} comps): {explained_variance:.2f}%")

    try:
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        outlier_labels = iso_forest.fit_predict(reduced_data)
    except Exception as e:
        raise RuntimeError(f"Isolation Forest failed: {e}")

    inlier_indices = np.where(outlier_labels == 1)[0]
    X_cleaned = X_combined_features.iloc[inlier_indices]
    print(f"Outliers removed: {len(outlier_labels) - len(inlier_indices)}")
    return X_cleaned, inlier_indices

from mpl_toolkits.mplot3d import Axes3D  
def visualize_pca_3d(X_combined, config, output_path="pca_3d_plot.svg"):

    X_cleaned, inlier_indices = detect_and_remove_outliers(X_combined, config)
    inlier_set = set(inlier_indices)

    # Perform a 3D PCA for visualization
    pca_3 = PCA(n_components=3)
    pca_3_data = pca_3.fit_transform(X_combined)  # shape => (n_samples, 3)
    explained_variance = pca_3.explained_variance_ratio_
    print(f"[3D PCA] Explained Variance: {100*np.sum(explained_variance):.2f}%")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = pca_3_data[:, 0]
    ys = pca_3_data[:, 1]
    zs = pca_3_data[:, 2]
    for i in range(X_combined.shape[0]):
        if i in inlier_set:
            ax.scatter(xs[i], ys[i], zs[i],
                       c='lightblue', marker='o', alpha=0.8)
        else:
            ax.scatter(xs[i], ys[i], zs[i],
                       c='salmon', marker='^', alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA Visualization (Inliers vs Outliers)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format="svg", dpi=300)
    plt.close()
    print(f"3D PCA plot saved to: {output_path}")
##############################################################################
# 3) Data Loading & Preprocessing
##############################################################################
def load_and_preprocess_data_for_pathways(config):
    """
    Loads chemical features from config["data"]["chemical_path"]
    and pathway data from config["data"]["pathway_path"].
    Builds X from the PNEC columns named f\"{col}_pnec_ratio\",
    ignoring chemicals that lack them. Returns (X_cleaned, Y_cleaned).
    """
    from sklearn.preprocessing import MinMaxScaler

    chemical_path = config["data"]["chemical_path"]
    pathway_path  = config["data"]["pathway_path"]
    epsilon       = config["preprocessing"]["epsilon"]
    apply_scaler  = config["preprocessing"].get("min_max_scaling", False)

    # Read Excel files
    chem_df     = pd.read_excel(chemical_path)
    pathway_df  = pd.read_excel(pathway_path)

    # Column ranges from config
    x_conc_start, x_conc_end = config["columns"]["X_concentrations_cols"]
    x_pnec_start, x_pnec_end = config["columns"]["X_pnec_ratios_cols"]
    y_pwy_start, y_pwy_end   = config["columns"]["Y_pathways_cols"]

    # Slice chemical data
    X_concentrations = chem_df.iloc[:, x_conc_start - 1 : x_conc_end - 1]
    X_pnec_ratios    = chem_df.iloc[:, x_pnec_start - 1 : x_pnec_end - 1]

    # Slice pathways
    if y_pwy_end == -1:
        Y_pathways = pathway_df.iloc[:, y_pwy_start - 1 :]
    else:
        Y_pathways = pathway_df.iloc[:, y_pwy_start - 1 : y_pwy_end]

    # Log transform the ratio data
    X_pnec_log  = np.log1p(X_pnec_ratios + epsilon)

    # Construct X features ONLY from matching \"col_pnec_ratio\" columns
    combined_features = {}
    for conc_col in X_concentrations.columns:

        ratio_col = f"{conc_col}_pnec_ratio"
        if ratio_col in X_pnec_log.columns:

            combined_features[ratio_col] = X_pnec_log[ratio_col]
        else:
            print(f"Skipping {conc_col}: no matching ratio column '{ratio_col}' in X_pnec_ratios.")

    X_combined = pd.DataFrame(combined_features)

    # Optional scaling
    if apply_scaler:
        scaler = MinMaxScaler()
        X_combined = pd.DataFrame(
            scaler.fit_transform(X_combined),
            columns=X_combined.columns
        )

    # Log transform Y
    Y_pathways_log = np.log1p(Y_pathways + epsilon)

    # Outlier detection on final X
    X_cleaned, inlier_idx = detect_and_remove_outliers(X_combined, config)
    Y_cleaned = Y_pathways_log.iloc[inlier_idx]

    return X_cleaned, Y_cleaned

##############################################################################
# 4) Create Short Codes for Pathways Only
##############################################################################
def generate_pathway_mappings(Y_pathways, output_dir):
    """
    Create short codes for each pathway. 
    Skips chemical mapping altogether, as requested.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Create short codes for pathways
    path_shortcodes = {pwy: f"P{idx+1}" for idx, pwy in enumerate(Y_pathways.columns)}
    path_map_df = pd.DataFrame(list(path_shortcodes.items()), columns=["Full Pathway Name", "Short Code"])
    path_map_path = os.path.join(output_dir, "pathway_name_mapping.csv")
    path_map_df.to_csv(path_map_path, index=False)
    return path_shortcodes

##############################################################################
# Optional) EDA using CCA
##############################################################################
from sklearn.cross_decomposition import CCA

def run_cca_biplot(
    X,
    Y,
    pathway_shortcodes,
    n_components=2,
    output_dir=None
):
    """
    Perform CCA on X (chemicals) vs Y (pathways), then create a 2D scatter labeling each chemical and pathway point.
    """

    # 1) Fit the CCA
    cca_model = CCA(n_components=n_components)
    cca_model.fit(X, Y)

    # 2) Transform data
    X_c, Y_c = cca_model.transform(X, Y)
    # Usually shape = (n_samples, n_components), if each row is a sample

    # 3) Only plot if we have at least 2 components
    if n_components < 2:
        print("n_components < 2, skipping 2D scatter plot.")
        return X_c, Y_c, cca_model

    # Create figure
    plt.figure(figsize=(12, 10))

    # Scatter the chemical canonical scores in blue
    plt.scatter(X_c[:, 0], X_c[:, 1],
                color='lightblue', alpha=0.6, label="Chemicals")

    # Scatter the pathway canonical scores in red
    plt.scatter(Y_c[:, 0], Y_c[:, 1],
                color='salmon', alpha=0.6, label="Pathways")

    # 4) Label each chemical point
    for i in range(len(X_c)):
        # By default, we use X.columns[i] if X is a DataFrame with column names as chemicals.
        # But if X is transposed (row=chemical), you might do X.index[i].
        chem_name = X.columns[i] if hasattr(X, 'columns') else f"C{i+1}"
        chem_label = chem_name.replace("_combined", "")
        plt.text(X_c[i, 0], X_c[i, 1],
                 chem_label,
                 color='navy', fontsize=18)

    # 5) Label each pathway point
    for i in range(len(Y_c)):
        # If Y has columns that represent each pathway:
        if hasattr(Y, 'columns') and i < len(Y.columns):
            full_name = Y.columns[i]
            short_label = pathway_shortcodes.get(full_name, full_name)
        else:
            short_label = f"P{i+1}"
        plt.text(Y_c[i, 0], Y_c[i, 1],
                 short_label,
                 color='darkred', fontsize=18)

    plt.title("CCA Analysis: Chemical vs. Pathway", fontsize=18)
    plt.xlabel("Canonical Component 1", fontsize=18)
    plt.ylabel("Canonical Component 2", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 6) Save figure
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "cca_biplot_labeled.svg")
    else:
        save_path = "cca_biplot_labeled.svg"

    plt.savefig(save_path, format="svg", dpi=300)
    plt.close()
    print(f"CCA biplot saved to: {save_path}")

    return X_c, Y_c, cca_model


##############################################################################
# 5) RFE
##############################################################################
def select_features_with_rfe(X_combined, y_summed, n_top_chemicals):
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rfe = RFE(estimator=rf_model, n_features_to_select=n_top_chemicals)
    rfe.fit(X_combined, y_summed)
    top_feats = X_combined.columns[rfe.support_].tolist()
    logging.info(f"Top RFE Features: {top_feats}")
    print(f"Top RFE Features: {top_feats}")
    return top_feats

##############################################################################
# 6) Evaluate Model with k-fold
##############################################################################
def evaluate_model_with_cv(model, X, y, cv=10):
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
    r2_scores   = cross_val_score(model, X, y, cv=cv, scoring='r2')
    rmse_scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    mean_r2 = np.mean(r2_scores)
    mean_rmse = -np.mean(rmse_scores)
    return mean_r2, mean_rmse

##############################################################################
# 7) Model Training & Pathway Selection
##############################################################################
def select_top_chemicals_and_pathways(
    X_combined, Y_pathways_log,
    n_top_chemicals=10, n_top_paths=10,
    cv=10, output_dir="./",
    export_tree=True,
    tree_index=0,
    tree_filename="xgboost_tree"
):

    # --- existing code for feature selection and model training ---
    y_summed = Y_pathways_log.sum(axis=1)
    top_feats = select_features_with_rfe(X_combined, y_summed, n_top_chemicals)
    X_reduced = X_combined[top_feats]

    xgb_model = XGBRegressor(random_state=42, objective="reg:squarederror", eval_metric="rmse")
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'lambda': [1, 2, 5, 10],
        'alpha': [0, 1, 2, 3],
    }
    grid_xgb = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid_xgb,
        cv=cv, n_jobs=-1, verbose=2
    )
    grid_xgb.fit(X_reduced, y_summed)
    best_xgb = grid_xgb.best_estimator_

    mean_r2, mean_rmse = evaluate_model_with_cv(best_xgb, X_reduced, y_summed, cv=cv)

    y_train_pred = best_xgb.predict(X_reduced)
    train_r2 = r2_score(y_summed, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_summed, y_train_pred))

    feat_importances = pd.Series(best_xgb.feature_importances_, index=top_feats)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sorted_imp = feat_importances.sort_values(ascending=True)
    sns.barplot(x=sorted_imp.values, y=sorted_imp.index, palette='Blues')
    plt.title("Feature Importances (XGBoost)", fontsize=20)
    plt.xlabel("Importance", fontsize=18)
    plt.ylabel("Chemical Feature", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances_xgb.svg"), dpi=300)
    plt.close()

    path_importances = []
    for path_col in Y_pathways_log.columns:
        path_model = XGBRegressor(random_state=42, objective="reg:squarederror", eval_metric="rmse")
        path_model.fit(X_reduced, Y_pathways_log[path_col])
        path_importances.append((path_col, path_model.feature_importances_.sum()))
    path_imp_df = pd.DataFrame(path_importances, columns=["Pathway", "Importance"])
    top_paths = path_imp_df.nlargest(n_top_paths, "Importance")["Pathway"].tolist()

    logging.info(f"Training R²: {train_r2:.4f}, Training RMSE: {train_rmse:.4f}")
    logging.info(f"CV R²: {mean_r2:.4f}, CV RMSE: {mean_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    print(f"CV R²: {mean_r2:.4f}, RMSE: {mean_rmse:.4f}")

    # --- new code to optionally export a single XGBoost tree ---
    if export_tree:
        try:
            import xgboost as xgb
            import graphviz

            dot_data = xgb.to_graphviz(
                best_xgb,
                num_trees=tree_index,
                rankdir="UT",         # top-down layout
                yes_color="#2E86C1",
                no_color="#E74C3C"
            )
            render_path = dot_data.render(
                filename=os.path.join(output_dir, tree_filename),
                format="svg",
                cleanup=True
            )
            print(f"SVG tree saved to {render_path}")
        except ImportError as e:
            print("Could not export tree - missing packages? Error:", e)

    return top_feats, top_paths, feat_importances, mean_r2, mean_rmse, train_r2, train_rmse

##############################################################################
# 8) Negative Coefficients => “Bad” (Linear Model)
##############################################################################
def analyze_coeff_sign(X, Y, output_dir, label="Pathway"):

    os.makedirs(output_dir, exist_ok=True)

    # If Y is multi-col, sum across them (like an overall path effect)
    if Y.ndim > 1 and Y.shape[1] > 1:
        Y_sum = Y.sum(axis=1)
    else:
        # If single col, use as is
        Y_sum = Y.squeeze()

    linreg = LinearRegression()
    linreg.fit(X, Y_sum)
    coefs = linreg.coef_  # array of shape (n_features,)

    # Build table
    results = []
    for feat, coef in zip(X.columns, coefs):
        # negative => "bad"
        direction = "Bad" if coef < 0 else "Good/Neutral"
        results.append((feat, coef, direction))

    df_results = pd.DataFrame(results, columns=["Feature", "Coefficient", "EffectDirection"])
    df_results.sort_values("Coefficient", inplace=True)
    out_csv = os.path.join(output_dir, f"coeff_sign_{label.lower()}.csv")
    df_results.to_csv(out_csv, index=False)

    print(f"Coefficient sign analysis saved to {out_csv}")
    logging.info(f"Coefficient sign analysis saved to {out_csv}")

##############################################################################
# 9) SHAP & PDP
##############################################################################
def interpret_model_with_shap_and_pdp(model, X_features, top_chemicals, config):
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Performing SHAP analysis...")
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_features)

        # SHAP summary
        shap.summary_plot(shap_values, X_features, show=False, max_display=config["shap"]["max_display"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_plot.svg"), dpi=300)
        plt.close()

        # SHAP dependence
        for feature in top_chemicals:
            interaction_index = config["shap"].get("interaction_index", None)
            shap.dependence_plot(
                feature,
                shap_values,
                X_features,
                interaction_index=interaction_index,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature}.svg"), dpi=300)
            plt.close()

        # Mean Absolute SHAP Value Bar Plot
        abs_shap_values = np.abs(shap_values).mean(axis=0)
        mean_abs_shap_df = pd.DataFrame({
            "Feature": X_features.columns,
            "MeanAbsSHAP": abs_shap_values
        }).sort_values(by="MeanAbsSHAP", ascending=False)

        plt.figure(figsize=(12, 12))
        sns.barplot(
            x="MeanAbsSHAP",
            y="Feature",
            data=mean_abs_shap_df,
            orient="h",
            palette="viridis"
        )
        plt.title("Mean Absolute SHAP Values", fontsize=18)
        plt.xlabel("MeanAbsSHAP", fontsize=18)
        plt.ylabel("Feature", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mean_abs_shap_values.svg"), dpi=300)
        plt.close()

    except Exception as e:
        logging.error(f"SHAP analysis failed: {e}")

    logging.info("Performing Partial Dependence Plots...")
    for chemical in top_chemicals:
        try:
            PartialDependenceDisplay.from_estimator(model, X_features, [chemical], grid_resolution=50)
            plt.title(f"Partial Dependence Plot: {chemical}")
            plt.xlabel(chemical, fontsize=18)
            plt.ylabel("Predicted Response", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pdp_{chemical}.svg"), dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"PDP generation failed for {chemical}: {e}")

##############################################################################
# 10) Build & Plot Networks
##############################################################################
from scipy.stats import spearmanr
def build_networks(X, Y, feature_importances, shap_values, config):
    imp_thresh  = config['correlation_analysis']['importance_threshold']
    corr_thresh = config['correlation_analysis']['correlation_threshold']
    shap_thresh = config['correlation_analysis']['shap_threshold']
    alpha       = config['correlation_analysis'].get('pval_alpha', 0.05)

    G_correlation = nx.Graph()
    G_importance = nx.Graph()
    G_shap = nx.Graph()

    chemicals = X.columns
    pathways  = Y.columns

    # Correlation-based
    for chem in chemicals:
        for pwy in pathways:
            x = X[chem]
            y = Y[pwy]
            corr, pval = spearmanr(x, y)
            
            if abs(corr) > corr_thresh and pval < alpha:
                G_correlation.add_edge(chem, pwy, weight=corr)

    # Feature importance-based
    for chem, imp in feature_importances.items():
        if imp > imp_thresh:
            for pwy in pathways:
                x = X[chem]
                y = Y[pwy]
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                else:
                    corr = 0
                if abs(corr) > corr_thresh:
                    G_importance.add_edge(chem, pwy, weight=imp)

    # SHAP-based
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_imp_series = pd.Series(mean_abs_shap, index=X.columns)
    for chem, shap_imp in shap_imp_series.items():
        if shap_imp > shap_thresh:
            for pwy in pathways:
                x = X[chem]
                y = Y[pwy]
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                else:
                    corr = 0
                if abs(corr) > corr_thresh:
                    G_shap.add_edge(chem, pwy, weight=shap_imp)

    return G_correlation, G_importance, G_shap

def plot_and_save_network(graph, chemicals, pathways, output_dir, title, config, threshold):
    os.makedirs(output_dir, exist_ok=True)
    filtered_edges = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] >= threshold]
    filtered_graph = graph.edge_subgraph(filtered_edges).copy()
    filtered_graph.add_nodes_from(graph.nodes)

    degrees = dict(filtered_graph.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_n_nodes = config['visualization'].get('top_n_nodes', 50)
    top_nodes = {node for node, _ in sorted_nodes[:top_n_nodes]}
    filtered_graph = filtered_graph.subgraph(top_nodes).copy()

    chemicals = [c for c in chemicals if c in filtered_graph.nodes]
    pathways  = [p for p in pathways if p in filtered_graph.nodes]

    figsize = config['visualization']['figsize']
    layout_type = config['visualization'].get('layout', 'kamada_kawai')
    if layout_type == 'spring':
        pos = nx.spring_layout(filtered_graph)
    elif layout_type == 'circular':
        pos = nx.circular_layout(filtered_graph)
    else:
        pos = nx.kamada_kawai_layout(filtered_graph)

    node_sizes = {node: 500 + 10*filtered_graph.degree(node) for node in filtered_graph.nodes}
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        filtered_graph, pos,
        nodelist=chemicals, node_shape='s', node_color='lightblue',
        node_size=[node_sizes[node] for node in chemicals], label='Chemicals', ax=ax
    )
    nx.draw_networkx_nodes(
        filtered_graph, pos,
        nodelist=pathways, node_shape='o', node_color='salmon',
        node_size=[node_sizes[node] for node in pathways], label='Pathways', ax=ax
    )

    edge_weights = [d['weight'] for _,_,d in filtered_graph.edges(data=True)]
    if edge_weights:
        norm = mpl.colors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_colors = [plt.cm.viridis(norm(w)) for w in edge_weights]
        nx.draw_networkx_edges(filtered_graph, pos, edge_color=edge_colors, alpha=0.6, width=2, ax=ax)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("Edge Weight", fontsize=20)

    node_labels = {n: n.replace("_combined", "") for n in filtered_graph.nodes}
    nx.draw_networkx_labels(filtered_graph, pos, labels=node_labels, font_size=18, font_color='black', ax=ax)
    custom_legend = [
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='lightblue', markersize=18, label='Chemicals'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='salmon',   markersize=18, label='Pathways')
    ]
    ax.legend(handles=custom_legend, loc='upper right', fontsize=14)
    plt.title(title, fontsize=18)

    save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_network.svg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Network plot saved to: {save_path}")

##############################################################################
# MAIN
##############################################################################
def main():
    logging.basicConfig(level=logging.INFO)
    device = check_gpu()
    logging.info(f"Device in use: {device}")

    # Load config
    config = load_config()
    out_dir = config["data"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Step A: Load & Preprocess
    X_combined, Y_pathways_log = load_and_preprocess_data_for_pathways(config)
    
    # Step B: Generate short codes ONLY for pathways
    generate_pathway_mappings(Y_pathways_log, out_dir)
    
    pathway_shortcodes = generate_pathway_mappings(Y_pathways_log, out_dir)
    pca_3d_path = os.path.join(out_dir, "pca_3d_plot.svg")

    visualize_pca_3d(
        X_combined=X_combined,
        config=config,
        output_path=pca_3d_path
    )
    # 4) Run the labeled 2D CCA scatter
    X_c, Y_c, cca = run_cca_biplot(
        X=X_combined,
        Y=Y_pathways_log,
        pathway_shortcodes=pathway_shortcodes,
        n_components=2,
        output_dir=out_dir
    )
    # Step C: RFE + XGB => top chemicals & pathways
    (top_chems, top_paths, feat_importances,
     mean_r2, mean_rmse, tr_r2, tr_rmse) = select_top_chemicals_and_pathways(
         X_combined,
         Y_pathways_log,
         n_top_chemicals=config["feature_selection"]["top_n_chemicals"],
         n_top_paths=config["feature_selection"]["top_n_args"],  # reusing top_n_args for path
         cv=5,
         output_dir=out_dir
         
    )
    logging.info(f"Top Chemicals: {top_chems}")
    logging.info(f"Top Pathways: {top_paths}")

    # Step D: Linear Model Coeff => negative => bad
    logging.info("Analyzing coefficient signs (negative => bad)...")
    # Example: sum top pathways as the Y target
    Y_sum = Y_pathways_log[top_paths].sum(axis=1)
    X_reduced = X_combined[top_chems]

    analyze_coeff_sign(
        X_reduced,
        Y_sum,
        output_dir=out_dir,
        label="Pathway"
    )

    # Step E: Train final XGB => SHAP & PDP
    logging.info("Final XGB model for SHAP/PDP...")
    final_xgb = XGBRegressor(random_state=42, objective="reg:squarederror", eval_metric="rmse")
    final_xgb.fit(X_reduced, Y_sum)
    interpret_model_with_shap_and_pdp(final_xgb, X_reduced, top_chems, config)

    # Step F: Build & Save Networks
    shap_vals = shap.TreeExplainer(final_xgb).shap_values(X_reduced)
    G_correlation, G_importance, G_shap = build_networks(
        X_reduced, 
        Y_pathways_log[top_paths],
        feat_importances,
        shap_vals,
        config
    )
    plot_and_save_network(G_correlation, top_chems, top_paths, out_dir,
                          "Correlation-Based Network", config,
                          config["correlation_analysis"]["correlation_threshold"])
    plot_and_save_network(G_importance, top_chems, top_paths, out_dir,
                          "Feature Importance-Based Network", config,
                          config["correlation_analysis"]["importance_threshold"])
    plot_and_save_network(G_shap, top_chems, top_paths, out_dir,
                          "SHAP-Based Network", config,
                          config["correlation_analysis"]["shap_threshold"])

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
