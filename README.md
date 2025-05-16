# AC-NA-Feature-Contribution-Analysis

This repository contains the complete code and data used in my master’s thesis:

> **"The Contribution of Subjective Experiences in Video Games to Players’ Well-Being: A Case Study of Animal Crossing: New Horizons"**

It explores how different subjective experience features—such as autonomy, competence, relatedness, enjoyment, and extrinsic motivation—contribute to players’ well-being, and whether these relationships are moderated by demographic factors like gender and age.

📌 This repository link is referenced in the thesis to ensure transparency, replicability, and academic integrity.

---

## 📘 Project Overview

The research investigates:
- How well subjective experience features predict players’ well-being
- Whether game duration ("Hours") improves prediction performance
- Which machine learning model performs best (Linear Regression, Random Forest, XGBoost)
- How feature importance varies, using SHAP analysis
- Whether gender and age moderate the relationship using Hierarchical Linear Modeling (HLM)

---

---

## 📂 Data

- `RawData.csv` contains the cleaned and anonymized dataset used for modeling.
- It includes gameplay data and survey responses collected from players of *Animal Crossing: New Horizons*.
- Due to privacy and scope, only derived features and de-identified data are included.

---

## 💡 How to Use

This repository is **not designed for direct execution**, but provides **step-by-step code walkthroughs** in markdown format.

You can view the analysis process by reading the files below:

- 🔹 [Preprocessing](src/preprocessing.md): Data cleaning, feature construction, and well-being score calculation  
- 🔹 [Modeling](src/modeling.md): Training and evaluating multiple regression models  
- 🔹 [SHAP](src/shap.md): Feature importance analysis using SHAP values  
- 🔹 [HLM](src/hlm_gender_age.md): Subgroup analysis using Hierarchical Linear Modeling

All visualizations and model outputs can be found in the `results/` folder.




