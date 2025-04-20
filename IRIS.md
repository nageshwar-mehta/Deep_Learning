# **Iris Flower Classification using RBF Neural Network**  

## **Table of Contents**  
1. [Problem Statement](#problem-statement)  
2. [Dataset Overview](#dataset-overview)  
3. [Environment Setup](#environment-setup)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
   - [Statistical Summary](#statistical-summary)  
   - [Data Visualization](#data-visualization)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Model Architecture](#model-architecture)  
   - [Initial RBF Network](#initial-rbf-network)  
   - [Improved RBF Network](#improved-rbf-network)  
7. [Training & Evaluation](#training--evaluation)  
8. [Results](#results)  
9. [Conclusion](#conclusion)  

---

## **1. Problem Statement**  
The goal of this project is to classify iris flowers into three species (**setosa, versicolor, virginica**) based on four morphological features:  
- **Sepal length (cm)**  
- **Sepal width (cm)**  
- **Petal length (cm)**  
- **Petal width (cm)**  

We implement a **Radial Basis Function (RBF) Neural Network** to achieve high classification accuracy.  

---

## **2. Dataset Overview**  
The dataset used is the **Iris dataset** from `scikit-learn`, containing:  
- **150 samples** (50 per class)  
- **4 numerical features** (all in cm)  
- **3 target classes**:  
  - **Setosa**  
  - **Versicolor**  
  - **Virginica**  

### **Key Characteristics**  
✅ **Balanced dataset** (33.33% per class)  
✅ **No missing values**  
✅ **Features follow normal distribution**  

---

## **3. Environment Setup**  
### **Required Libraries**  
```python
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow plotly_express scipy
```  

### **Libraries Used**  
- **Data Handling**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`, `plotly_express`  
- **Machine Learning**: `scikit-learn`  
- **Deep Learning**: `tensorflow`  

---

## **4. Exploratory Data Analysis (EDA)**  

### **Statistical Summary**  
| Feature          | Mean   | Std Dev | Min  | 25%  | 50%  | 75%  | Max  |
|------------------|--------|---------|------|------|------|------|------|
| Sepal Length (cm)| 5.84   | 0.83    | 4.3  | 5.1  | 5.8  | 6.4  | 7.9  |
| Sepal Width (cm) | 3.06   | 0.44    | 2.0  | 2.8  | 3.0  | 3.3  | 4.4  |
| Petal Length (cm)| 3.76   | 1.77    | 1.0  | 1.6  | 4.35 | 5.1  | 6.9  |
| Petal Width (cm) | 1.20   | 0.76    | 0.1  | 0.3  | 1.3  | 1.8  | 2.5  |  

### **Data Visualization**  
#### **1. Histograms & Probability Density Functions (PDFs)**  
- **Petal measurements** show clear separation between species.  
- **Sepal measurements** have some overlap between species.  

#### **2. Box Plots**  
- **Setosa** has smaller petals compared to the other two species.  
- **Virginica** has the largest petals on average.  

---

## **5. Data Preprocessing**  
1. **Encoding**:  
   - One-hot encoding applied to target labels (`LabelBinarizer`).  
2. **Scaling**:  
   - Features standardized using `StandardScaler` (mean=0, std=1).  
3. **Train-Test Split**:  
   - **80% training (120 samples)**  
   - **20% testing (30 samples)**  
4. **Validation Split**:  
   - **30% of training data** used for validation during training.  

---

## **6. Model Architecture**  

### **Initial RBF Network**  
```python
Sequential([
    Flatten(input_shape=(4,)),
    RBFLayer(10, gamma=0.5),
    Dense(3, activation="softmax")
])
```  
- **Optimizer**: Adam  
- **Loss**: Categorical crossentropy  
- **Early Stopping**: Patience=5  

### **Improved RBF Network**  
```python
Sequential([
    Flatten(input_shape=(4,)),
    RBFLayer(16, gamma=0.5),
    Dense(32, activation="relu"),  # Additional hidden layer
    Dense(3, activation="softmax")
])
```  
- **Optimizer**: Adam  
- **Loss**: Categorical crossentropy  
- **Callbacks**:  
  - Early Stopping (patience=10)  
  - Model Checkpointing  

---

## **7. Training & Evaluation**  

### **Training Process**  
- **Epochs**: 500 (early stopping applied)  
- **Batch Size**: 4 (initial), 8 (improved)  
- **Validation Split**: 30% (initial), 20% (improved)  

### **Performance Metrics**  
| Model          | Test Accuracy | Test Loss |
|----------------|---------------|-----------|
| **Initial RBF**| 100%          | 0.0910    |
| **Improved RBF**| 100%         | 0.0541    |  

### **Confusion Matrix**  
Both models achieved **perfect classification** on the test set.  

---

## **8. Results**  
✅ **100% accuracy** on test data for both models.  
✅ **Improved model** converges faster and has lower loss.  
✅ **No overfitting** observed (training & validation curves align).  

---

## **9. Conclusion**  
- The **RBF Neural Network** is highly effective for iris classification.  
- The **improved architecture** (with an additional dense layer) performs slightly better.  
- Future work could explore **hyperparameter tuning** (e.g., gamma value, number of RBF units).  

---
### **How to Run the Code**  
1. Install all dependencies .  
2. Run the Jupyter notebook or Python script.  

---
**📌 Note**: The full code and visualizations are available in the provided notebook.  

---
**Author**: Nageshwar Kumar  
**Date**: 20-04-2025  


