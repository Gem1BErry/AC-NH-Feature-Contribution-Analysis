# AC-NA-Feature-Contribution-Analysis

This repository contains the complete code and data used in my master’s thesis:

> **"The Contribution of Subjective Experiences in Video Games to Players’ Well-Being: A Case Study of Animal Crossing: New Horizons"**

📌 This repository link is referenced in the thesis to ensure transparency, replicability, and academic integrity.

---

## 📘 Project Overview

The research investigates:
- How well subjective experience features predict players’ well-being
- Which machine learning model performs best
- How feature importance varies, using SHAP analysis
- Whether gender and age moderate the relationship using Hierarchical Linear Modeling

---

## 📂 Data

- `RawData.csv` contains the cleaned and anonymized dataset used for modeling.
- It includes gameplay data and survey responses collected from players of *Animal Crossing: New Horizons*.
- Due to privacy and scope, only derived features and de-identified data are included.

---

## 💡 How to Use

You can view the analysis process by reading the files below:

## How to Use


- 🔹 [Preprocessing](src/preprocessing.md): Data cleaning and feature construction
- 🔹 [EDA](src/EDA.md): Exploratory data analysis and initial data inspection
- 🔹 [Tuning](src/tuning.md): Hyperparameter tuning for Random Forest and XGBoost models  
- 🔹 [Regression Diagnostics](src/regression%20diagnostics.md): Assessing model assumptions and performance diagnostics  
- 🔹 [Modeling](src/modeling.md): Training and evaluating multiple regression models  
- 🔹 [SHAP](src/shap.md): Feature importance analysis using SHAP values  
- 🔹 [Subgroup Analysis](src/subgroup_analysis.md): Analyzing effects within demographic subgroups  



All visualizations and model outputs can be found in the `results/` folder.




