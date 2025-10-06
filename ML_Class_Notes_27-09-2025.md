Of course. You are right, the formulas can be made much clearer. Here is the revised version with improved formatting for all mathematical equations.

---

# Machine Learning Fundamentals (ML-11)

*Date: 27-09-2025*

This document covers the foundational concepts of machine learning, including data handling, model evaluation, and an introduction to linear regression.

## 1. Introduction to Machine Learning

A brief overview of fundamental ML concepts, including how data is prepared for training.

The first step in any machine learning project is to split the available data into separate sets for training and testing the model.

-   **Training Set:** The subset of data used to train the model. The algorithm learns the underlying patterns and relationships from this data.
-   **Test Set:** The subset of data used to evaluate the performance of the trained model. This data is "unseen" by the model during training and provides an unbiased estimate of its performance on new, real-world data.

***Example of Data Splitting:***

Given 7 data points:
-   Data points 1, 2, 3, 4 → **Train Set**
-   Data points 5, 6, 7 → **Test Set**

## 2. The Role of Error in Learning

To learn, a model needs a mechanism to measure its mistakes and correct them.

The core of machine learning is an iterative process where the algorithm adjusts its internal parameters based on a calculated **error**. This error, or "difference," is the discrepancy between the model's prediction (**ŷ**) and the actual value (**y**). By minimizing this error over time, the model learns.

*(Diagram: A flow chart shows that a model takes inputs (x1, x2), produces a prediction (ŷ), which is then compared to the true value (y). The resulting "difference" is fed back into the model to help it adjust and improve.)*

## 3. Loss vs. Cost Functions

These functions mathematically define the error of a model.

-   **Loss Function:** Measures the error for a **single data point**. It quantifies how far a single prediction is from its actual value.
-   **Cost Function:** Measures the aggregate error over an **entire dataset** (e.g., the training or test set). It is typically the average or sum of the loss functions for all examples in the set.

***Example of Error Calculation:***

| Actual Value (y) | Predicted Value (ŷ) | Difference (y - ŷ) |
| :--------------- | :------------------ | :------------------- |
| 10               | 5                   | 5                    |
| 15               | 7                   | 8                    |
| 20               | 19                  | 1                    |
| 40               | 50                  | -10                  |

The cost function aggregates these individual differences into a single numerical score (e.g., Training cost = 180, Test cost = 100).

## 4. Model Performance Diagnosis

Understanding how a model performs on training vs. test data helps diagnose common issues like overfitting and underfitting.

### Overfitting

**Overfitting** occurs when a model performs exceptionally well on the training data but poorly on unseen test data. This indicates that the model has "memorized" the training set, including its noise, instead of learning the underlying general patterns.

-   **Condition:** `Training Loss <<< Test/Validation Loss`
-   **Example:** A model achieves 100% accuracy on the training set but only 30% on the test set.

*(Diagram: A complex, wavy line fits every training data point perfectly but misses the general trend of the test data points.)*

### Underfitting

**Underfitting** occurs when a model performs poorly on both the training and the test data. This suggests the model is too simple to capture the underlying structure of the data.

-   **Condition:** Both `Training Loss` and `Test/Validation Loss` are high.

*(Diagram: A simple straight line fails to capture the underlying curve in both the training and test data.)*

### Ideal Case (Good Fit)

An **ideal model** generalizes well from the training data to unseen data. Its performance on the training and test sets is comparable and strong.

-   **Condition:** `Training Loss ≈ Test/Validation Loss` (and both are low).
-   **Example:** Train accuracy is 92%, and Test accuracy is 87%.

## 5. Evaluation Metrics

Evaluation metrics are quantitative measures used to assess the performance of a machine learning model.

### Regression Metrics

Used for tasks where the output is a continuous value (e.g., predicting a price).

1.  **Mean Squared Error (MSE):** The average of the squared differences between predicted and actual values. It penalizes larger errors more heavily.
    ```
    MSE = (1/n) * Σ (yᵢ - ŷᵢ)²
    ```
2.  **Root Mean Squared Error (RMSE):** The square root of the MSE. It brings the metric back to the same units as the target variable.
    ```
    RMSE = √[ (1/n) * Σ (yᵢ - ŷᵢ)² ]
    ```
3.  **Mean Absolute Error (MAE):** The average of the absolute differences between predicted and actual values. It is less sensitive to outliers than MSE.
    ```
    MAE = (1/n) * Σ |yᵢ - ŷᵢ|
    ```
    ***Where:***
    -   `n`: The total number of data points.
    -   `Σ`: The summation symbol (sum of all instances).
    -   `yᵢ`: The actual value for the i-th data point.
    -   `ŷᵢ`: The predicted value for the i-th data point.

### Classification Metrics

Used for tasks where the output is a discrete category (e.g., "spam" or "not spam"). These are often derived from a **Confusion Matrix**.

A confusion matrix summarizes the performance of a classification model:
-   **True Positives (TP):** Correctly predicted positive cases.
-   **True Negatives (TN):** Correctly predicted negative cases.
-   **False Positives (FP):** Incorrectly predicted positive cases (Type I error).
-   **False Negatives (FN):** Incorrectly predicted negative cases (Type II error).

***Example: Spam Email Classifier***

|          | Predicted: Spam | Predicted: Non-Spam |
| :------- | :-------------- | :------------------ |
| **Actual: Spam** | **600 (TP)**    | 300 (FN)            |
| **Actual: Non-Spam** | 100 (FP)    | **9000 (TN)**   |

1.  **Accuracy:** The ratio of correct predictions to the total number of predictions.
    -   **Formula:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
    -   **Example:** `(600 + 9000) / (600 + 9000 + 100 + 300) = 0.96` or **96%**

2.  **Precision:** Of all the predictions for the positive class, how many were correct?
    -   **Formula:** `Precision = TP / (TP + FP)`
    -   **Example:** `600 / (600 + 100) = 0.857` or **85.7%**

3.  **Recall (Sensitivity):** Of all the actual positive cases, how many did the model identify correctly?
    -   **Formula:** `Recall = TP / (TP + FN)`
    -   **Example:** `600 / (600 + 300) = 0.667` or **66.7%**

4.  **F1 Score:** The harmonic mean of Precision and Recall, balancing both metrics.
    -   **Formula:** `F1 Score = 2 * (Precision * Recall) / (Precision + Recall)`

## 6. Linear Regression

An introduction to one of the simplest and most widely used regression algorithms.

Linear regression models the relationship between a dependent variable (`y`) and one or more independent variables (`x`) by fitting a linear equation to the observed data.

-   **Simple Linear Regression Formula:** `y = mx + c`
    -   `y`: Target variable
    -   `x`: Feature variable
    -   `m`: Slope (weight)
    -   `c`: Intercept (bias)

### Linear vs. Non-Linear Models

-   **Linear Model:** The equation is a linear combination of its parameters.
    -   *Example:* `y = m₁x₁ + m₂x₂ + m₃x₃ + c`
-   **Non-Linear Model:** The parameters are not combined linearly.
    -   *Example:* `y = (m₀ + m₁x₁) / (1 + m₂x₂)`

**Note on Polynomial Regression:** While it can model non-linear relationships, it is still considered a **linear model** because the equation `y = m₁x + m₂x² + c` is linear *with respect to its parameters* (`m₁`, `m₂`).

## 7. The Model Training Process

A high-level overview of how a model like linear regression learns from data.

The typical workflow involves several stages:

1.  **Data Processing:** Raw data is processed through **Exploratory Data Analysis (EDA)** and **Feature Engineering (FE)** to create a clean, final dataset.
2.  **Data Splitting:** The final data is split into training and testing sets.
3.  **Iterative Training Loop (on the training set):**
    a. **Initialize Parameters:** Start with initial values for the model parameters (e.g., `m=0`, `c=0`).
    b. **Predict:** Make a prediction (`ŷ`) using the current parameters.
    c. **Calculate Error (Loss):** Compute the error between the prediction and the actual value (e.g., `error = y - ŷ`).
    d. **Update Parameters:** Use the error to adjust the parameters (`m` and `c`) in a direction that reduces the error. This adjustment is often done using an algorithm like **Gradient Descent**.
    e. **Repeat:** Continue this loop until the model's performance converges and the cost function is minimized.

## 8. Sample Dataset

A sample dataset for a regression task, such as predicting salary.

| Year-of-exp | Completed-project | Salary-k |
| :---------- | :---------------- | :------- |
| 1           | 0                 | 39.764   |
| 2           | 1                 | 48.400   |
| 3           | 1                 | 56.978   |
| 4           | 2                 | 68.240   |
| 5           | 3                 | 77.867   |
| 6           | 4                 | 85.022   |

---

## Summary

-   **Data Splitting** (Train/Test) is crucial for building and evaluating models that can generalize to new data.
-   Models learn by iteratively minimizing an **error** (or **cost function**), which measures the difference between predictions and actual values.
-   **Overfitting** (memorizing training data) and **Underfitting** (model too simple) are common problems diagnosed by comparing training and test performance.
-   **Evaluation Metrics** are used to quantify model performance. The choice of metric depends on the task (Regression vs. Classification).
-   **Linear Regression** is a fundamental algorithm that models relationships using a straight line, optimized to minimize error.

## References

-   [Gradient Descent 3D Visualization](https://aero-learn.imperial.ac.uk/vis/Machine%20Learning/gradient_descent_3d.html)
-   [Derivative Cheat Sheet](https://worksheets.clipart-library.com/images2/derivative-cheat-sheet/derivative-cheat-sheet-17.png)