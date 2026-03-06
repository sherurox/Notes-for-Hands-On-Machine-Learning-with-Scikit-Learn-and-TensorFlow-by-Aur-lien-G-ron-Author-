# Hands-On Machine Learning — Study Notes & Foundations Map

My chapter-wise study notebook for **Hands-On Machine Learning with Scikit-Learn and TensorFlow** by Aurélien Géron. This repo contains detailed notes, code snippets, and a complete concept map covering the foundational ML topics from Chapters 1–9.

## AI Book Tutor

I built a custom ChatGPT tutor trained on this book to help you study interactively. Ask it questions, get explanations, or quiz yourself on any chapter.

**[Hands-On Machine Learning Book Tutor — by Shreyas Khandale](https://chatgpt.com/g/g-698e63c74aa081919bfe725df449a1f2-hands-on-machine-learning-book-tutor)**

---

## What's Inside

- **9 folders**, one per chapter (1–9)
- Each folder contains **one `.ipynb` notebook** with detailed notes and important code snippets
- Concept maps, core ideas, and recall summaries for every chapter

> **Copyright note:** This repo contains **my own notes and code**. It does not include the original book text or the full book PDF.

---

## The Big Picture — What the Whole Book Teaches

```
Data → Model → Training → Evaluation → Improvement
```

Machine Learning Foundations cover a pipeline that starts with understanding the landscape, moves through data handling and model training, and ends with advanced techniques like ensemble methods, dimensionality reduction, and unsupervised learning.

---

## Complete Machine Learning Foundations Map (Ch. 1–9)

```
Machine Learning Foundations
│
├── 1. ML Landscape (Chapter 1)
│   ├── What is Machine Learning
│   ├── Why use ML
│   ├── Types of ML
│   │   ├── Supervised Learning
│   │   ├── Unsupervised Learning
│   │   ├── Semi-Supervised Learning
│   │   └── Reinforcement Learning
│   ├── Learning Systems
│   │   ├── Batch Learning
│   │   └── Online Learning
│   └── ML Challenges
│       ├── Bad data
│       ├── Overfitting
│       └── Underfitting
│
├── 2. ML Project Pipeline (Chapter 2)
│   ├── Frame the problem
│   ├── Get the data
│   ├── Train/test split
│   ├── Explore data
│   ├── Prepare data
│   │   ├── Feature engineering
│   │   ├── Scaling
│   │   └── Encoding
│   ├── Train models
│   ├── Fine-tune models
│   └── Evaluate final model
│
├── 3. Classification (Chapter 3)
│   ├── Binary classification
│   ├── Evaluation Metrics
│   │   ├── Confusion matrix
│   │   ├── Precision
│   │   ├── Recall
│   │   └── F1 score
│   ├── Precision-Recall trade-off
│   ├── ROC Curve & AUC
│   ├── Multiclass classification
│   └── Multilabel classification
│
├── 4. Training Models (Chapter 4)
│   ├── Linear Regression
│   │   ├── Normal Equation
│   │   └── Gradient Descent
│   ├── Gradient Descent Types
│   │   ├── Batch GD
│   │   ├── Stochastic GD
│   │   └── Mini-Batch GD
│   ├── Polynomial Regression
│   ├── Learning Curves
│   │   ├── Underfitting
│   │   └── Overfitting
│   ├── Regularization
│   │   ├── Ridge
│   │   ├── Lasso
│   │   └── Elastic Net
│   └── Logistic Regression
│       ├── Sigmoid
│       └── Softmax
│
├── 5. Support Vector Machines (Chapter 5)
│   ├── Linear SVM
│   ├── Maximum Margin Classifier
│   ├── Soft Margin (C parameter)
│   ├── Nonlinear SVM
│   │   └── Kernel Trick
│   ├── Kernels
│   │   ├── Polynomial
│   │   └── RBF
│   └── SVM Regression (SVR)
│
├── 6. Decision Trees (Chapter 6)
│   ├── Tree Structure
│   │   ├── Root
│   │   ├── Nodes
│   │   └── Leaves
│   ├── CART Algorithm
│   ├── Impurity Measures
│   │   ├── Gini
│   │   └── Entropy
│   ├── Tree Regularization
│   │   ├── max_depth
│   │   ├── min_samples_split
│   │   └── min_samples_leaf
│   └── Decision Tree Regression
│
├── 7. Ensemble Learning (Chapter 7)
│   ├── Voting Classifiers
│   │   ├── Hard Voting
│   │   └── Soft Voting
│   ├── Bagging
│   │   └── Random Forest
│   ├── Boosting
│   │   ├── AdaBoost
│   │   └── Gradient Boosting
│   └── Stacking
│
├── 8. Dimensionality Reduction (Chapter 8)
│   ├── Curse of Dimensionality
│   ├── PCA
│   │   ├── Principal components
│   │   ├── Explained variance
│   │   └── Covariance / eigenvectors
│   ├── PCA Variants
│   │   ├── Incremental PCA
│   │   └── Randomized PCA
│   ├── Kernel PCA
│   └── Manifold Learning
│       └── LLE
│
└── 9. Unsupervised Learning (Chapter 9)
    ├── Clustering
    │   ├── K-Means
    │   ├── Elbow Method
    │   ├── Silhouette Score
    │   ├── K-Means++
    │   └── Mini-Batch K-Means
    ├── Density Clustering
    │   └── DBSCAN
    └── Probabilistic Clustering
        └── Gaussian Mixture Models
            ├── EM algorithm
            └── BIC / AIC
```

---

## Chapter-by-Chapter Breakdown

### Chapter 1 — The Machine Learning Landscape

**Core Idea:** ML is about learning patterns from data, improving with experience, and avoiding explicit rule programming.

**Concept Map:**

```
Machine Learning Landscape
│
├── What is Machine Learning?
│   ├── Learning patterns from data
│   ├── Improves with experience
│   └── Avoids explicit rule programming
│
├── Why Use Machine Learning?
│   ├── Problems too complex for rules
│   ├── Adapting to changing environments
│   └── Discover hidden patterns in data
│
├── Types of Machine Learning Systems
│   ├── By Supervision
│   │   ├── Supervised Learning
│   │   │   ├── Classification
│   │   │   └── Regression
│   │   ├── Unsupervised Learning
│   │   │   ├── Clustering
│   │   │   ├── Dimensionality Reduction
│   │   │   └── Anomaly Detection
│   │   ├── Semi-Supervised Learning
│   │   └── Reinforcement Learning
│   │       ├── Agent
│   │       ├── Environment
│   │       └── Reward signal
│   ├── By Training Method
│   │   ├── Batch Learning
│   │   └── Online Learning
│   └── By Learning Style
│       ├── Instance-Based Learning
│       └── Model-Based Learning
│
└── Main Challenges of Machine Learning
    ├── Insufficient training data
    ├── Non-representative data
    ├── Poor-quality data
    ├── Irrelevant features
    ├── Overfitting (model too complex)
    └── Underfitting (model too simple)
```

**20-Second Recall:**
> ML types (supervised, unsupervised, semi-supervised, reinforcement) → Learning systems (batch vs online, instance vs model based) → Challenges (bad data, overfitting, underfitting)

---

### Chapter 2 — End-to-End Machine Learning Project

**Core Idea:** Machine Learning is not just training a model. It is a pipeline.

```
Problem → Data → Exploration → Preparation → Model → Tuning → Evaluation
```

**Concept Map:**

```
End-to-End Machine Learning Project
│
├── Look at the Big Picture
│   ├── Define objective
│   ├── Identify ML task (Regression / Classification)
│   └── Choose performance measure (e.g., RMSE)
│
├── Get the Data
│   ├── Download dataset
│   ├── Load dataset
│   └── Explore dataset structure
│
├── Create Test Set
│   ├── Train/Test split
│   └── Stratified sampling
│
├── Discover & Visualize Data
│   ├── Data visualization
│   ├── Correlation analysis
│   └── Identify useful patterns
│
├── Prepare the Data
│   ├── Data Cleaning (handle missing values)
│   ├── Feature Engineering (create new features)
│   ├── Feature Scaling (Normalization / Standardization)
│   └── Categorical Encoding (Ordinal / One-Hot)
│
├── Select and Train Models
│   ├── Linear Regression
│   ├── Decision Tree
│   └── Random Forest
│
├── Fine-Tune the Model
│   ├── Cross-validation
│   ├── Grid Search
│   └── Randomized Search
│
├── Analyze Best Model
│   ├── Feature importance
│   └── Error analysis
│
└── Evaluate on Test Set
    └── Final performance evaluation
```

**20-Second Recall:**
> 1. Look at the big picture → 2. Get the data → 3. Create test set → 4. Explore & visualize → 5. Prepare the data → 6. Train models → 7. Fine-tune models → 8. Evaluate final model

---

### Chapter 3 — Classification

**Core Idea:** Classification is not only about predicting classes. The main challenge is evaluating models correctly using precision, recall, F1 score, and ROC curves.

**Concept Map:**

```
Classification
│
├── Binary Classification
│   ├── Example: digit 5 vs not-5
│   └── Target labels: True / False
│
├── Classification Algorithms
│   ├── SGD Classifier
│   └── Random Forest Classifier
│
├── Performance Evaluation
│   ├── Cross-Validation
│   ├── Confusion Matrix
│   │   ├── True Positive (TP)
│   │   ├── True Negative (TN)
│   │   ├── False Positive (FP)
│   │   └── False Negative (FN)
│   ├── Precision → TP / (TP + FP)
│   ├── Recall → TP / (TP + FN)
│   └── F1 Score → harmonic mean of precision & recall
│
├── Precision vs Recall Trade-off
│   └── Adjusting classification threshold
│
├── ROC Curve
│   ├── True Positive Rate (Recall)
│   ├── False Positive Rate
│   └── AUC (Area Under Curve)
│
├── Multiclass Classification
│   ├── One-vs-Rest (OvR)
│   └── One-vs-One (OvO)
│
└── Multi-Label Classification
    └── Multiple labels per instance
```

**20-Second Recall:**
> Binary classification → Evaluation metrics (confusion matrix, precision, recall, F1) → Precision-Recall trade-off → ROC curve & AUC → Multiclass → Multilabel

---

### Chapter 4 — Training Models

**Core Idea:** Models are trained by minimizing a cost function. Main tools: Gradient Descent, Regularization, and Logistic Regression for classification.

**Concept Map:**

```
Training Models
│
├── Linear Regression
│   ├── Normal Equation (analytical solution)
│   └── Gradient Descent
│       ├── Batch Gradient Descent
│       ├── Stochastic Gradient Descent
│       └── Mini-Batch Gradient Descent
│       └── Learning Rate
│           ├── Too small → slow convergence
│           └── Too large → divergence
│
├── Polynomial Regression
│   └── Modeling nonlinear relationships
│
├── Learning Curves
│   ├── Underfitting (model too simple)
│   └── Overfitting (model too complex)
│
├── Regularization
│   ├── Ridge Regression (L2)
│   ├── Lasso Regression (L1)
│   └── Elastic Net
│
└── Logistic Regression
    ├── Sigmoid Function
    ├── Binary Classification
    └── Softmax Regression (Multiclass)
```

**20-Second Recall:**
> Linear Regression (Normal Equation / Gradient Descent) → Polynomial Regression → Learning Curves (underfitting / overfitting) → Regularization (Ridge, Lasso, Elastic Net) → Logistic Regression (sigmoid / softmax)

---

### Chapter 5 — Support Vector Machines

**Core Idea:** SVM finds the decision boundary that maximizes the margin between classes. For nonlinear problems, kernels transform data into higher dimensions where linear separation becomes possible.

**Concept Map:**

```
Support Vector Machines (SVM)
│
├── Linear SVM Classification
│   ├── Decision Boundary
│   ├── Maximum Margin
│   └── Support Vectors
│
├── Soft Margin Classification
│   ├── Handling Outliers
│   └── Regularization Parameter (C)
│       ├── Large C → fewer violations
│       └── Small C → wider margin
│
├── Nonlinear Classification
│   ├── Polynomial Features
│   └── Kernel Trick
│
├── Kernel Functions
│   ├── Polynomial Kernel
│   ├── RBF (Gaussian) Kernel
│   └── Similarity Features
│
├── SVM Regression (SVR)
│   ├── Linear SVR
│   └── Polynomial Kernel SVR
│
└── Key Hyperparameters
    ├── C (regularization)
    ├── gamma (RBF kernel)
    └── degree (polynomial kernel)
```

**20-Second Recall:**
> Linear SVM (maximum margin) → Soft Margin (C controls violations) → Nonlinear SVM (kernel trick) → Kernels (polynomial, RBF) → SVM Regression (SVR)

---

### Chapter 6 — Decision Trees

**Core Idea:** Decision Trees recursively split the dataset into smaller subsets based on feature values to create simple decision rules for prediction.

**Concept Map:**

```
Decision Trees
│
├── Decision Tree Structure
│   ├── Root Node
│   ├── Internal Nodes
│   ├── Branches
│   └── Leaf Nodes (predictions)
│
├── How Trees Make Decisions
│   └── Feature-based splitting
│
├── CART Algorithm
│   └── Classification and Regression Trees
│
├── Impurity Measures
│   ├── Gini Impurity
│   └── Entropy
│
├── Tree Depth
│   ├── Shallow trees → underfitting
│   └── Deep trees → overfitting
│
├── Regularization (Controlling Tree Growth)
│   ├── max_depth
│   ├── min_samples_split
│   ├── min_samples_leaf
│   └── max_features
│
├── Decision Tree Regression
│   └── Predicts numeric values
│
└── Key Characteristics
    ├── Interpretable models
    ├── No feature scaling required
    └── Sensitive to data variations
```

**20-Second Recall:**
> Tree structure (root, nodes, leaves) → CART algorithm → Impurity measures (Gini, Entropy) → Tree depth (underfitting/overfitting) → Regularization → Decision Tree Regression

---

### Chapter 7 — Ensemble Learning and Random Forests

**Core Idea:** Instead of relying on a single model, ensemble learning combines many models to produce a stronger and more stable predictor. Weak learners + combination → strong learner.

**Concept Map:**

```
Ensemble Learning
│
├── Idea of Ensemble Methods
│   ├── Combine multiple models
│   └── Improve prediction performance
│
├── Voting Classifiers
│   ├── Hard Voting (majority vote)
│   └── Soft Voting (average predicted probabilities)
│
├── Bagging (Bootstrap Aggregating)
│   ├── Train models on bootstrap samples
│   ├── Parallel training
│   └── Reduce variance
│
├── Random Forest
│   ├── Ensemble of Decision Trees
│   ├── Random feature selection
│   ├── Bagging-based method
│   └── Feature importance estimation
│
├── Boosting
│   ├── Sequential training
│   ├── Focus on previous errors
│   ├── AdaBoost (reweights misclassified samples)
│   └── Gradient Boosting (learns from residual errors)
│
└── Stacking
    ├── Combine predictions of multiple models
    └── Meta-model learns final prediction
```

**20-Second Recall:**
> Voting classifiers → Bagging (Random Forest) → Boosting (AdaBoost, Gradient Boosting) → Stacking

---

### Chapter 8 — Dimensionality Reduction

**Core Idea:** High-dimensional data often lies near a lower-dimensional structure. Dimensionality reduction finds this structure while preserving as much information as possible.

**Concept Map:**

```
Dimensionality Reduction
│
├── Why Reduce Dimensions?
│   ├── Faster training
│   ├── Data visualization
│   └── Curse of dimensionality
│
├── Projection Methods
│   └── Data projected onto lower-dimensional subspace
│
├── Principal Component Analysis (PCA)
│   ├── Principal Components (directions of maximum variance)
│   ├── Explained Variance Ratio (measure of information preserved)
│   ├── Choosing Number of Components (cumulative explained variance)
│   └── PCA Mathematics
│       ├── Covariance matrix
│       ├── Eigenvectors
│       └── Eigenvalues
│
├── PCA Variants
│   ├── Incremental PCA (handles large datasets)
│   └── Randomized PCA (faster approximate PCA)
│
├── Kernel PCA
│   ├── Nonlinear dimensionality reduction
│   └── Kernel trick
│
└── Manifold Learning
    ├── Manifold Hypothesis (high-dimensional data lies on low-dimensional manifold)
    └── Locally Linear Embedding (LLE)
        ├── Preserves local relationships
        └── Unfolds nonlinear manifolds
```

**20-Second Recall:**
> Why reduce dimensions? (curse of dimensionality) → PCA (principal components, explained variance, covariance & eigenvectors) → PCA variants (Incremental, Randomized) → Kernel PCA → Manifold Learning (LLE)

---

### Chapter 9 — Unsupervised Learning Techniques

**Core Idea:** Unsupervised learning discovers hidden structure in unlabeled data. The chapter focuses mainly on centroid-based clustering (K-Means), density-based clustering (DBSCAN), and probabilistic clustering (Gaussian Mixtures).

**Concept Map:**

```
Unsupervised Learning
│
├── Clustering
│   ├── K-Means Clustering
│   │   ├── Centroids
│   │   ├── Inertia (cost function)
│   │   └── Iterative optimization
│   ├── Choosing Number of Clusters
│   │   ├── Elbow Method
│   │   └── Silhouette Score
│   ├── K-Means Improvements
│   │   ├── K-Means++
│   │   └── Mini-Batch K-Means
│   └── Applications
│       └── Image Segmentation
│
├── Density-Based Clustering
│   └── DBSCAN
│       ├── ε (epsilon radius)
│       ├── min_samples
│       ├── Core points
│       ├── Border points
│       └── Noise points (outliers)
│
└── Gaussian Mixture Models (GMM)
    ├── Probabilistic Clustering (soft cluster assignments)
    ├── Gaussian Components
    │   ├── Mean (μ)
    │   ├── Covariance (Σ)
    │   └── Mixture weight (π)
    ├── Expectation-Maximization (EM)
    │   ├── E-step
    │   └── M-step
    └── Model Selection
        ├── AIC
        └── BIC
```

**20-Second Recall:**
> Clustering (K-Means, Elbow Method, Silhouette Score, K-Means++, Mini-Batch) → DBSCAN (density-based clusters, noise detection) → Gaussian Mixture Models (soft clustering, EM algorithm)

---

## Repo Structure

```
├── Chapter 1 — The Machine Learning Landscape/
│   └── Chapter 1 — The Machine Learning Landscape.ipynb
├── Chapter 2 — End-to-End ML Project/
│   └── Chapter 2 — End-to-End ML Project.ipynb
├── Chapter 3 — Classification/
│   └── Chapter 3 — Classification.ipynb
├── Chapter 4 — Training Models/
│   └── Chapter 4 — Training Models.ipynb
├── Chapter 5 — Support Vector Machines/
│   └── Chapter 5 — Support Vector Machines.ipynb
├── Chapter 6 — Decision Trees/
│   └── Chapter 6 — Decision Trees.ipynb
├── Chapter 7 — Ensemble Learning and Random Forests/
│   └── Chapter 7 — Ensemble Learning and Random Forests.ipynb
├── Chapter 8 — Dimensionality Reduction/
│   └── Chapter 8 — Dimensionality Reduction.ipynb
├── Chapter 9 — Unsupervised Learning Techniques/
│   └── Chapter 9 — Unsupervised Learning Techniques.ipynb
└── README.md
```

---

## Future Additions

If I build mini-projects for exercises, I'll link them from this README later (with code, datasets, and results).

---

**Made by [Shreyas Khandale](https://github.com/shreyaskhandale)**
