

# Machine Learning Notes: Logistic Regression

> 📅 **Date:** 11-OCT-2025

## Agenda

This session covers the fundamentals and implementation of Logistic Regression.
- **Logistic Regression Theory**
- **Implementation of LR**
- **Model Evaluation (ROC, AUC)**
- **Cross-Validation**

---

## 1. Introduction to Logistic Regression

Logistic Regression is a fundamental algorithm used for **classification** tasks. While its name includes "regression," it's used to predict a categorical outcome (e.g., Yes/No, 0/1, True/False).

It starts with a linear equation, similar to Linear Regression:
`z = mx + c` (or more generally, `z = β₀ + β₁x₁ + ...`)

However, the output of this linear equation (`z`) can be any real number, from -∞ to +∞. To convert this into a probability suitable for classification, we use an **activation function**.

### The Sigmoid (or Logistic) Function

In Logistic Regression, the activation function is the **Sigmoid function**. Its primary role is to "squash" the output of the linear equation into a defined range.

-   **Input Range:** Takes any real number (`-∞` to `+∞`).
-   **Output Range:** Produces a value between `0` and `1`.

This output can be interpreted as the probability of the positive class.


*A visual representation of the process.*

The flow is as follows:
1.  Calculate `z = mx + c`.
2.  Apply the sigmoid function to `z` to get a probability, `ŷ` (y-hat).
3.  Apply a **threshold** (commonly 0.5) to the probability to make a final class prediction.
    -   If `ŷ > 0.5`, predict class `1`.
    -   If `ŷ <= 0.5`, predict class `0`.

#### Example: Predicting Height

Let's say we want to classify a person as "tall" (1) or "short" (0) based on some features.

-   **Input:** Height features.
-   **Output (`y`):** A binary value: `0` (short) or `1` (tall).

The model's output, `sigmoid(z) = ŷ`, would be a probability.
-   If `ŷ = 0.8`, it means there is an 80% probability that the person is tall.
-   If `ŷ = 0.2`, it means there is a 20% probability that the person is tall.

---

## 2. The Cost Function: Why Mean Squared Error (MSE) Fails

In Linear Regression, the **Mean Squared Error (MSE)** is a common cost function used to measure the model's error.

**MSE Formula:**
```
MSE = (1/m) * Σ(yᵢ - ŷᵢ)²
```

However, for a classification problem like Logistic Regression, **MSE is not suitable**. If we were to use MSE, the resulting cost function would be non-convex, making it difficult for optimization algorithms like Gradient Descent to find the global minimum. The notes simply state that "MSE will not work."

---

## 3. The Logistic Regression Cost Function (Log Loss)

To solve this, Logistic Regression uses a different cost function, often called **Log Loss** or **Binary Cross-Entropy**. This function heavily penalizes confident but incorrect predictions.

**Log Loss Formula:**
$$
J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right]
$$

Let's analyze this with the examples from the notes (`m=1` for simplicity).

### Cost Function Analysis

#### Case 1: Confident but Wrong Prediction
-   **True Label (`y`):** `0`
-   **Predicted Probability (`y_pred`):** `0.9` (Model is 90% sure the class is 1)

**Calculation:**
```
Cost = - [ (0 * log(0.9)) + (1 - 0) * log(1 - 0.9) ]
     = - [ 0 + 1 * log(0.1) ]
     = -log(0.1)
     = -(-2.3025)
     = 2.3025
```
This is a **high cost**, reflecting the high penalty for being very confident in the wrong prediction.
> For comparison, the MSE would be `(0 - 0.9)² = 0.81`. The Log Loss penalty is much steeper.

#### Case 2: Low Confidence and Correct Prediction
-   **True Label (`y`):** `0`
-   **Predicted Probability (`y_pred`):** `0.1` (Model is 90% sure the class is 0)

**Calculation:**
```
Cost = - [ (0 * log(0.1)) + (1 - 0) * log(1 - 0.1) ]
     = - [ 0 + 1 * log(0.9) ]
     = -log(0.9)
     = -(-0.10536)
     = 0.10536
```
This is a **low cost**, rewarding the model for making a correct prediction with high confidence.

---

## 4. Flow of a Logistic Regression Model

The end-to-end process of training a Logistic Regression model can be summarized in these steps:

1.  **Input Features:** Start with your input data (`x₁, x₂, x₃, x₄, ...`).
2.  **Linear Combination:** Calculate the intermediate output `z` using the linear equation `z = mx + c`.
3.  **Activation:** Apply the **sigmoid** function to `z` to get the predicted probability `ŷ`.
4.  **Loss Calculation:** Use the **Log Loss** function `J(y, ŷ)` to calculate the error between the prediction and the true label.
5.  **Parameter Update:** Use an optimizer (like Gradient Descent) to update the model's **learnable parameters (`m`, `c` or `β`)** to minimize the loss.
6.  **Iteration:** Repeat steps 2-5 until the loss converges or a stopping criterion is met.
7.  **Done:** The model is trained and ready for predictions.

---

## 5. Out of Context Topic: Data and Model Cards

The notes briefly touch upon documentation practices for machine learning projects.

### Data Card
A Data Card provides essential information about the dataset used.
-   Info about the data
-   Sources
-   Number of rows & columns
-   Features (including their range and type)
-   Information on missing values

### Model Card
A Model Card provides essential information about the trained model.
-   Performance metrics on a test set
-   Link to the test set on which evaluation was performed
-   Framework used (e.g., `sklearn`)
-   Evaluation details (e.g., confusion matrix)
-   Date of creation/training

---

## 6. Practical Implementation: Preprocessing and Pipelines

The notes outline a practical workflow using `sklearn` pipelines for data preprocessing, which is a crucial step before model training.

### Building a Preprocessing Pipeline
Often, different preprocessing steps are needed for different columns. A pipeline helps automate this. The notes describe a dynamic approach:

-   **Define Column Groups:** Separate columns based on the required processing (e.g., `imputation_cols`, `numerical_cols`).
-   **Create Transformers:** Define the steps, such as `Imputation` and `Scaling`.
-   **Conditional Logic:** The pipeline can be built conditionally. For example, if imputation columns exist, create a sub-pipeline for imputation and scaling on those columns.
-   **Handling Remainders:** For columns that don't fit into the specified transformation groups, you can define a `remainder` strategy:
    -   `remainder='drop'`: Discard these columns.
    -   `remainder='passthrough'`: Keep these columns as they are.



### Integrating Preprocessing with the Model

The preprocessor (e.g., a `ColumnTransformer` or a full `Pipeline`) is typically integrated with the final model.

1.  **Transform Data:** The pipeline first applies all the preprocessing steps (e.g., imputation for feature B, scaling for A, C, D, E) to the raw training data (`X_train`).
2.  **Train Model:** The transformed data is then fed into the Logistic Regression model (`clf`) for training using the `clf.fit()` method.



### The Prediction Process

Once the model is trained, making a prediction on new data involves these steps:

1.  **Calculate `z`:** Compute the linear combination using the learned weights (and bias).
    `z = X_test · weights + bias`
2.  **Calculate Probability `p`:** Apply the sigmoid function to `z`.
    `p = sigmoid(z)`
3.  **Apply Threshold:** Convert the probability into a final class label.
    `ŷ = 1 if p > 0.5 else 0`