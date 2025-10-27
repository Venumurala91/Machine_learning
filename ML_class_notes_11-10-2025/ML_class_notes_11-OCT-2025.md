
# 🧠 Logistic Regression: From Theory to Implementation

*Date: 11-OCT-2025*

This document covers the theory behind Logistic Regression, its cost function, its implementation workflow, and practical considerations using Scikit-learn pipelines.

## Agenda
- **Logistic Regression Theory:** Understanding the core concepts.
- **Cost Function:** Why Mean Squared Error (MSE) fails and why Log Loss is used.
- **Implementation Flow:** The step-by-step process of training a model.
- **Preprocessing Pipelines:** Using Scikit-learn to prepare data for modeling.
- **Model & Data Documentation:** Best practices with Model and Data Cards.

> 💡 **Topics for Future Discussion:** The agenda also mentioned `ROC/AUC` and `Cross-validation`, which are crucial for evaluating and validating classification models. These will be covered in a subsequent session.

---

## 1. What is Logistic Regression?

While "regression" is in the name, Logistic Regression is fundamentally a **classification** algorithm. It's used to predict a discrete, categorical outcome (e.g., Yes/No, True/False, 0/1).

It works by taking a linear equation and passing its output through a special activation function called the **Sigmoid function**.

### ### The Core Components

1.  **Linear Equation:** At its heart, it starts with a simple linear formula, just like Linear Regression.
    $$
    z = mx + c \quad (\text{or more generally, } z = W^T X + b)
    $$

2.  **Activation (Logistic) Function:** The output `z` can be any real number from `-∞` to `+∞`. To convert this into a probability, we use the **Sigmoid function**.

    ![Description: A diagram showing the Sigmoid function mapping an input range of (-∞, +∞) to an output range of (0, 1)](https://i.imgur.com/h2sW73D.png)

    -   **Input Range:** `(-∞, +∞)`
    -   **Output Range:** `(0, 1)`

This output represents the *probability* of the positive class (e.g., the probability of being "tall" is 80%).

3.  **Thresholding for Classification:** To get a final class label (0 or 1), we apply a decision threshold to the probability. The most common threshold is 0.5.

    -   If probability `ŷ > 0.5`, predict class `1`.
    -   If probability `ŷ <= 0.5`, predict class `0`.

### ### The Prediction Flow at a Glance



**Example:** Predicting if a person is "tall" (1) or "short" (0) based on some features.
- The model calculates `z`.
- `sigmoid(z)` produces `ŷ` (our predicted probability).
- If `ŷ = 0.8`, the model predicts an 80% probability that the person is tall. Since 0.8 > 0.5, the final prediction is class `1` (tall).
- If `ŷ = 0.2`, the model predicts a 20% probability of being tall. Since 0.2 < 0.5, the final prediction is class `0` (short).

---

## 2. The Cost Function: Why Not MSE?

In Linear Regression, we use the **Mean Squared Error (MSE)** to measure how far our predictions are from the actual values.

$$
\text{MSE} = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

However, **MSE is not suitable for Logistic Regression**. If you plot the MSE cost for a classification problem, it results in a "non-convex" function with many local minima. This makes it very difficult for optimization algorithms like Gradient Descent to find the one best set of parameters (the global minimum).

> 💡 We need a different cost function that is convex and penalizes wrong predictions more effectively.

### ### Introducing Log Loss (Binary Cross-Entropy)

Logistic Regression uses a cost function called **Log Loss** or **Binary Cross-Entropy**. It's designed to heavily penalize models that are confident about an incorrect prediction.

The cost function `J(β)` is defined as:
$$
J(\beta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right]
$$

Let's break down its behavior with two examples where the actual class `y` is `0`.

#### **Case 1: High Penalty for a Confident but Wrong Prediction**
- **Actual `y`**: `0`
- **Predicted `ŷ`**: `0.9` (The model is 90% sure the class is 1, which is wrong)

Plugging this into the formula (for a single instance, `m=1`):
`Cost = - [ (0 * log(0.9)) + (1 - 0) * log(1 - 0.9) ]`
`Cost = - [ 0 + 1 * log(0.1) ]`
`Cost = -log(0.1) ≈ 2.3025`

This is a **very high cost**. For comparison, the MSE would be `(0 - 0.9)² = 0.81`.

#### **Case 2: Low Penalty for a Confident and Correct Prediction**
- **Actual `y`**: `0`
- **Predicted `ŷ`**: `0.1` (The model is 90% sure the class is 0, which is correct)

Plugging this into the formula:
`Cost = - [ (0 * log(0.1)) + (1 - 0) * log(1 - 0.1) ]`
`Cost = - [ 0 + 1 * log(0.9) ]`
`Cost = -log(0.9) ≈ 0.10536`

This is a **very low cost**, as expected.

| Case | Actual (y) | Predicted (ŷ) | Log Loss Cost | MSE Cost | Outcome |
| :--- | :--------: | :-----------: | :-----------: | :------: | :------ |
| 1    |     0      |      0.9      |   **2.3025**  |   0.81   | High Cost (Punished) |
| 2    |     0      |      0.1      |   **0.10536** |   0.01   | Low Cost (Rewarded)  |

---

## 3. The Complete Logistic Regression Workflow

Training a Logistic Regression model is an iterative process, typically using an optimization algorithm like Gradient Descent.

Here is the high-level flow:

1.  **Initialize Parameters:** Start with initial values for the model's weights (`m` or `W`) and bias (`c` or `b`).
2.  **Compute Linear Output:** For each data point, calculate the linear combination: `z = W·X + b`.
3.  **Apply Sigmoid:** Pass `z` through the sigmoid function to get the predicted probabilities: `ŷ = sigmoid(z)`.
4.  **Calculate Loss:** Use the Log Loss function to compute the total cost over all data points based on `y` and `ŷ`.
5.  **Update Parameters:** Calculate the gradient of the loss function with respect to the parameters and update them to minimize the loss.
6.  **Iterate:** Repeat steps 2-5 until the cost function converges (i.e., the parameters stabilize and the loss is minimized).
7.  **Done:** The final parameters represent the trained model.

---

## 4. Practical Implementation with Scikit-learn

In practice, we use libraries like Scikit-learn, which handle the complex optimization for us. Our main job becomes preparing the data correctly.

### ### Data Preprocessing with Pipelines

Real-world data is messy. It often requires multiple preprocessing steps like handling missing values (imputation) and feature scaling. A Scikit-learn `Pipeline` combined with a `ColumnTransformer` is the best way to organize these steps.

**Scenario:** We have a dataset with numerical columns. Some need missing values imputed, while all need to be scaled.

-   `imputation_cols = [1, 2, 3, 4, 6]` (Columns that might have missing values)
-   `numerical_cols = [1, 2, 3, 4, 5, 6]` (All numerical columns)



A `ColumnTransformer` lets you apply different transformations to different columns.
-   **Transformer 1:** Apply imputation *and* scaling to columns `[1, 2, 3, 4, 6]`.
-   **Transformer 2:** Apply only scaling to the remaining numerical column `[5]`.
-   **`remainder` parameter:** You can choose to `'drop'` any columns not specified or `'passthrough'` to keep them as-is.

This entire multi-step process can be encapsulated in a single `Pipeline` object.

### ### Training and Prediction

Once the preprocessing pipeline is defined, it can be combined with the model.

![Description: A diagram showing the full ML workflow. X_train is fed into a pipeline that handles imputation and scaling. The transformed data is then used by clf.fit() along with y_train to train the logistic regression model.](https://i.imgur.com/2sA7wE5.png)

1.  **Define the Model:** `clf = LogisticRegression()`
2.  **Combine with Pipeline:** Create a master pipeline that includes the preprocessing `ColumnTransformer` and the `clf` model.
3.  **Fit the Model:** Call `pipeline.fit(X_train, y_train)`. The pipeline automatically applies all the transformations to the training data before feeding it to the model.
4.  **Predict:** When you call `pipeline.predict(X_test)`, it automatically applies the *same* learned transformations to the new data before making a prediction.

---

## 5. Model and Data Documentation

Good documentation is crucial for reproducible and responsible machine learning.

### ### Data Card

A Data Card is a short document that provides context about a dataset. It should include:
-   **Source:** Where did the data come from?
-   **Content:** What do the rows and columns represent?
-   **Features:** Details about each feature (e.g., data type, range, definition).
-   **Limitations:** Are there known issues, like missing values or biases?

### ### Model Card

A Model Card is a document that provides information about a trained model's performance and behavior.
-   **Performance:** Key metrics on a test set (e.g., accuracy, precision, recall, confusion matrix).
-   **Evaluation Data:** A link or description of the test set used.
-   **Intended Use:** What is the model designed to do?
-   **Limitations & Biases:** In what scenarios might the model perform poorly?

---

## Summary

### Key Takeaways
-   **Logistic Regression is for Classification:** It predicts probabilities for categorical outcomes.
-   **Sigmoid is Key:** It transforms a linear output into a probability between 0 and 1.
-   **Log Loss is the Right Tool:** It's the cost function used to train the model, as it correctly penalizes confident wrong predictions.
-   **Pipelines are Essential:** Use Scikit-learn's `Pipeline` and `ColumnTransformer` to build robust and reproducible data preprocessing workflows.
-   **Document Everything:** Use Data Cards and Model Cards to ensure your work is transparent and understandable.

### Further Reading
1.  **ROC Curves and AUC:** Learn how to evaluate the performance of a classification model across different thresholds.
2.  **Cross-Validation:** Understand this powerful technique for getting a more reliable estimate of model performance.
3.  **Regularization in Logistic Regression:** Explore L1 and L2 regularization (used in `sklearn.linear_model.LogisticRegression`) to prevent overfitting.