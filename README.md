# Anomaly Detection in Ship Engine Functionality

An unsupervised machine learning project exploring anomaly detection techniques for monitoring ship engine functionality.

## Overview

Ship engine malfunctions can contribute to operational downtime, increased fuel consumption, maintenance costs, and safety risks. This project develops an anomaly detection framework to identify unusual engine behaviour from continuously monitored engine parameters.

The analysis compares a classical statistical approach with two unsupervised machine learning techniques:

* Interquartile Range (IQR)
* One-Class Support Vector Machine (One-Class SVM)
* Isolation Forest

Principal Component Analysis (PCA) was also used to visualise the detected anomalies in two dimensions.

## Dataset

The dataset contains **19,535 observations** across six numerical engine-monitoring features:

* Engine RPM
* Lubrication oil pressure
* Fuel pressure
* Coolant pressure
* Lubrication oil temperature
* Coolant temperature

The dataset contained no missing values or duplicate observations.

## Methodology

### 1. Exploratory Data Analysis

The dataset was first inspected to understand its structure, distributions, descriptive statistics, missing values, and potential extreme observations.

Visualisations were created to examine the distributions of the six engine parameters.

### 2. Statistical Anomaly Detection — IQR

The Interquartile Range method was used to identify observations with feature values outside the expected range.

Each of the six features was assessed individually, and an observation was classified as anomalous when at least two of its six features were identified as outliers.

This produced an anomaly rate of **2.16%**, within the expected 1–5% range.

### 3. One-Class SVM

One-Class SVM was used to identify observations that differed from the majority of the data without requiring labelled anomalies.

After parameter tuning, the final configuration used:

* `gamma = 0.3`
* `nu = 0.03`

The model identified approximately **3.02%** of observations as anomalous.

### 4. Isolation Forest

Isolation Forest was used as a second unsupervised anomaly detection approach. The model identifies observations that are easier to isolate from the rest of the dataset.

The final configuration used:

* `n_estimators = 100`
* `contamination = 0.03`

The model identified approximately **3.005%** of observations as anomalous.

### 5. PCA Visualisation

Principal Component Analysis was applied to reduce the six-dimensional feature space to two principal components.

The resulting 2D visualisations were used to examine the distribution of normal and anomalous observations identified by the machine learning models.

The PCA plots showed that anomalous observations were generally more dispersed and appeared towards the peripheries of the main cluster.

## Results

| Method           | Anomaly Rate |
| ---------------- | -----------: |
| IQR              |        2.16% |
| One-Class SVM    |       ~3.02% |
| Isolation Forest |      ~3.005% |

The statistical and machine learning approaches produced broadly consistent anomaly patterns.

IQR provided a straightforward statistical baseline, while the unsupervised machine learning approaches provided greater flexibility for identifying more complex patterns.

## Key Takeaways

* Statistical and unsupervised ML methods produced broadly consistent anomaly detection results.
* IQR provided a simple and interpretable baseline.
* One-Class SVM and Isolation Forest identified approximately 3% of observations as anomalous after parameter tuning.
* PCA provided an intuitive way to visualise the detected anomalies.
* PCA only captures linear relationships, so techniques such as UMAP or t-SNE could be explored for more complex non-linear structures.

## Recommendations

Potential extensions to this project include:

* Combining statistical and machine learning approaches into a hybrid anomaly detection system.
* Automating anomaly monitoring and alerting.
* Periodically reviewing model parameters as engine operating conditions change.
* Exploring non-linear dimensionality reduction techniques.
* Integrating anomaly detection with predictive maintenance workflows.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

## Project Structure

```text
anomaly-detection-ship-engine/
│
├── README.md
├── anomaly_detection_ship_engine_analysis.ipynb
└── anomaly_detection_ship_engine_report.pdf
```

## Project Files

**Notebook:** Contains the exploratory data analysis, anomaly detection implementation, model tuning, and visualisations.

**Report:** Provides the written analysis, interpretation of results, conclusions, and recommendations.

## Author

**Jovan Surya**

Data Science, Machine Learning & AI
