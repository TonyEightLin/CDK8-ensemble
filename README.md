# CDK8-Ensemble

This repository holds the data and code associated with the manuscript **A consensus machine learning model to generate
a
focused screening library for the identification of CDK8 inhibitors**. Scripts include the generation of Machine
Learning
models and the generated similarity matrix. Datasets used for model training and testing are also provided.

The main scripts for the project can be found under the **cdk8classifier/** folder. Datasets for training or generating
figures associated with the manuscript can be found under the **resources/** folder. The **configs.yaml** file contains
parameter suggestions generated from hyperopt tuning.

This project generated models for classification models for CDK8 inhibitors. Six models were generated:

- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)
- Random Forest (RF)
- C-Support Vector (SVC)
- XGBoost (XG)
- Multilayer Perceptron (MLP)

**Citation:**

The manuscript can be found at [Protein Science](https://onlinelibrary.wiley.com/doi/10.1002/pro.5007) (Open Access) and
cite as:

Lin TE, Yen D,HuangFu W-C, Wu Y-W, Hsu J-Y, Yen S-C, et al.An ensemble machine learning model generates a focused
screening library for the identification ofCDK8 inhibitors. Protein Science. 2024;33(6):
e5007. https://doi.org/10.1002/pro.5007

Thank you for visiting! 