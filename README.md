# FlexiSAF - Generative AI & Data Science (Advanced)

## 📌 Deliverable Overview
This repository folder contains the final deliverables for the Advanced Machine Learning module. The task involves building, evaluating, and documenting two advanced predictive models tailored for the EdTech sector (specifically aligned with FlexiSAF's SAFSIMS platform).

### 📁 Repository Structure
* `project1_early_warning_xai.ipynb` - Early Warning System model using XGBoost & SHAP.
* `project2_enrollment_forecasting_rnn.ipynb` - Enrollment Demand Forecasting model using a Recurrent Neural Network (LSTM).
* `data.csv` - Dataset for Project 1 (Student Dropout Data).
* `student_placement_timeseries.csv` - Panel dataset for Project 2 (Student Performance Timeseries).
* `README.md` - Project documentation.

---

## 🚀 Project 1: Early Warning System with Explainable AI
**Objective:** Predict student dropout risk before it happens and explain the reasoning to educators.
* **Algorithm:** XGBoost (Ensemble Learning) optimized for class imbalance.
* **Explainability:** Utilizes SHAP (SHapley Additive exPlanations) to generate Global Feature Importance and Local Waterfall plots.
* **EdTech Impact:** Allows school administrators to transition from black-box predictions to actionable, transparent insights for personalized student intervention.

## 📈 Project 2: Enrollment Demand Forecasting (RNN)
**Objective:** Forecast future school enrollment using historical student trajectories to aid in resource planning.
* **Algorithm:** Long Short-Term Memory (LSTM) Neural Network.
* **Approach:** Uses a sliding-window time-series approach to predict individual student CGPA trajectories, aggregating the results to forecast the volume of active, returning students.
* **EdTech Impact:** Provides school administrators with data-driven projections for classroom capacity planning, teacher recruitment, and budget forecasting.

---

## 🔗 Required Links & Evidence
* **Live Demo / Hosted Link:** [Insert Link Here, or remove if N/A]
* **Presentation / Recording:** [Insert Link Here, or remove if N/A]
* **Figma Prototype:** [Insert Link Here, or remove if N/A]

## 🛠️ How to Run Locally
1. Ensure Python 3.11+ is installed.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap tensorflow jupyter`
3. Open the notebooks in Jupyter or VS Code and execute all cells.
