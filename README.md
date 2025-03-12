# Environmental Risk Assessment of Emerging Contaminants (ERAEC)
Basic machine learning workflow and configuration files for chemical-biological data integration and pattern discovery purposes
This project explores how machine learning (ML) can identify and prioritize emerging contaminants (ECs) that disrupt microbial functions in various environments. It focuses on pattern discovery rather than predictive modeling, demonstrating how ML-based workflows highlight chemicals most likely driving community shifts or pathway disturbances.

Key Objectives
Data Integration: Combine EC concentration data (often scaled by PNEC) with metagenomic or omics-derived endpoints (e.g., ARG abundance, metabolic pathways).

Modeling Framework: Apply gradient boosting or random forest models with feature selection, cross-validation, and interpretability tools (SHAP, PDP).

Methodology Highlights
Preprocessing: Log-transformed concentration/PNEC ratios, normalized biological endpoints, and filtering of low-detection features.
Cross-Validation: Typically 10-fold or leave-one-out to estimate alignment (R²) between EC features and microbial response.
Interpretability: XGBoost’s built-in feature importance, SHAP summary plots, and partial dependence curves to clarify top chemical drivers and their concentration thresholds.

Case Studies:
Landfill Leachate: Highly contaminated environment analyzed for carbon/nitrogen pathways.
Yangtze River Sediment: Broader antibiotic and heavy metal contamination profiles linked to ARG dissemination.

# Repository Contents
scripts/: Python scripts for data preprocessing, model training, and interpretability plots.

data/: Example or template for the type of data used (EC concentrations, metagenomic relative abundance suitable for cross-sample comparison such as 16S rRNA normalized ARGs abundance).

figures/: Selected outputs, including correlation networks, SHAP plots, and PDPs.

docs/: Additional documentation or references for environment configuration.

# OS & ENV
Win11 & python 3.12.7

Packages requirements:
pandas
numpy
scikit-learn
tensorflow
keras-tuner
matplotlib
seaborn
xgboost
pyyaml
networkx
econml 

# Contact & Contributing
Contributors: Hanlin Cui, Bin Liang, Aijie Wang
OrganizationL: State Key Laboratory of Urban Water Resource and Environment, School of Civil & Environmental Engineering, Harbin Institute of Technology.
Contributions: Feel free to open issues or pull requests for improved data integration, advanced ML models, or additional visualization features.
For questions or collaboration, contact 21B929086@stu.hit.edu.cn; liangbin1214@hit.edu.cn
