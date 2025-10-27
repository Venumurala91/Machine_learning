

# Machine Learning Class Notes: 05-10-2025

## Table of Contents
1.  [R-Squared (R²) Score](#1-r-squared-r²-score)
2.  [The Bias & Variance Tradeoff](#2-the-bias--variance-tradeoff)
3.  [L1 & L2 Regularization](#3-l1--l2-regularization)
4.  [Logistic Regression](#4-logistic-regression)

---

## 1. R-Squared (R²) Score

The **R-Squared (R²)** score is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. In simpler terms, it tells you how well your model's predictions fit the actual data compared to just predicting the average value.

In simple terms:

It measures how close the predictions are to the actual data.

R² = 1 → Perfect model (predictions match actual values exactly)

R² = 0 → Model is no better than predicting the mean

R² < 0 → Model performs worse than simply predicting the mean

### The Intuition

Imagine we are trying to predict a person's `weight` based on their `height`.

1.  **Baseline Model (Using the Mean):** The simplest possible prediction we can make is to always guess the average weight, regardless of height. The error in this model is the variation of each data point from this average line. This is called the **Total Sum of Squares (SST)** or `var(mean)`.

    
    *This diagram shows the total variation of the data around the mean (average) weight.*

2.  **Our Regression Model (The Fit Line):** A better approach is to fit a regression line (`y = mx + c`) to the data. The error in this model is the distance from each data point to our fit line (these distances are called **residuals**). The variation here is the **Sum of Squared Errors (SSE)** or `var(fit)`.

    
    *This diagram shows the variation of the data around our model's best-fit line. The goal is for this variation to be much smaller than the variation around the mean.*

### The Formula

R² quantifies the improvement of our fit line over the simple mean line. It represents the percentage of total variation that our model successfully explains.

The formula is:
```
R² = (Variance_around_mean - Variance_around_fit) / Variance_around_mean
```
Which can be written as:
```
R² = (SST - SSE) / SST   or   R² = 1 - (SSE / SST)
```

> **Key takeaway:** R² tells us how much of the variation in the person's `weight` we can explain by taking their `height` into account.

### Example Calculation

Let's assume from our data:
-   Variance around the mean (`var(mean)`): **72**
-   Variance around our fit line (`var(fit)`): **24**

```
R² = (72 - 24) / 72 = 48 / 72 ≈ 0.66
```
An R² score of 0.66 means that **our model explains approximately 66% of the variation in the person's weight**. There is a 66% reduction in variance when we use our model instead of just using the average weight.

---

## 2. The Bias & Variance Tradeoff

In machine learning, our goal is to build a model that not only performs well on the data it was trained on (`train` data) but also generalizes well to new, unseen data (`test` data). Bias and Variance are two types of errors that prevent us from achieving this.

### What is Bias?

**Bias** is the error resulting from overly simplistic assumptions in the learning algorithm. A high-bias model fails to capture the underlying patterns in the data and performs poorly on both training and test sets. This is known as **underfitting**.

-   **Example:** Using a simple linear model to fit complex, non-linear data.
-   **Characteristics:**
    -   High error on the training set.
    -   High error on the test set.
    -   `Loss(Train) ≈ 100`, `Loss(Test) ≈ 98` (Both are high).


*This model has high bias because the straight line is too simple to capture the curved relationship in the data.*

### What is Variance?

**Variance** is the error resulting from a model that is too complex and sensitive to the small fluctuations in the training data. A high-variance model captures noise from the training data, causing it to perform exceptionally well on the training set but poorly on the test set. This is known as **overfitting**.

-   **Example:** Using a high-degree polynomial model that wiggles to fit every single training point.
-   **Characteristics:**
    -   Very low error on the training set.
    -   High error on the test set.
    -   `Loss(Train) ≈ 0`, `Loss(Test) ≈ 70` (A large gap between train and test performance).


*This model has low bias (it fits the training data perfectly) but high variance, as it will perform poorly on new test data (green dots).*

### The Tradeoff

The ultimate goal is to find a balance between bias and variance.
> **Ideal Model:** A model with **low bias** (it's complex enough to capture the true relationship) and **low variance** (it's not so complex that it models the noise). This is the "sweet spot" of the tradeoff.

---

## 3. L1 & L2 Regularization

When a model is too complex (high variance/overfitting), its coefficients (slopes) often take on very large values. **Regularization** is a technique used to prevent overfitting by adding a penalty term to the model's loss function. This penalty discourages the model from learning overly complex patterns.

> **Analogy:** Think of a driving test on a race track. The goal isn't just to complete the track with zero error (overfitting the track), but to do so while keeping the car controlled (keeping coefficients small) so you can handle any track (new data).

### The Problems with Vanilla Linear Regression
Standard Linear Regression (using OLS or SGD) can overfit if:
-   The model is too complex (too many features).
-   Features are highly correlated (multicollinearity).

### L1 Regularization (Lasso Regression)

L1 Regularization adds a penalty equal to the **sum of the absolute values** of the coefficients.

**New Loss Function:**
```
Loss_L1 = MSE + λ * Σ|βᵢ|
```
-   `MSE`: The original Mean Squared Error.
-   `βᵢ`: The coefficient of the i-th feature.
-   `λ` (lambda): The regularization parameter. It controls the strength of the penalty.

**Key Property:** L1 regularization can shrink some coefficients to **exactly zero**. This makes it very useful for **feature selection**, as it effectively removes unimportant features from the model.

### L2 Regularization (Ridge Regression)

L2 Regularization adds a penalty equal to the **sum of the squared values** of the coefficients.

**New Loss Function:**
```
Loss_L2 = MSE + λ * Σ(βᵢ)²
```

**Key Property:** L2 regularization forces coefficients to be small but does **not** shrink them to exactly zero. It is effective at handling multicollinearity (highly correlated features) by distributing the coefficient values among them.

| Method | Base Loss | Regularization | Key Feature |
| :--- | :--- | :--- | :--- |
| **Vanilla LR** | SSE / MSE | None | Prone to overfitting |
| **Lasso (L1)** | SSE / MSE | `λ * Σ|βᵢ|` | Can perform feature selection |
| **Ridge (L2)** | SSE / MSE | `λ * Σ(βᵢ)²` | Manages multicollinearity well |

---

## 4. Logistic Regression

Despite its name, Logistic Regression is a model used for **classification** tasks, not regression. It's used to predict a categorical outcome (e.g., Yes/No, Sick/Healthy, Spam/Not Spam).

### The Problem with Linear Regression for Classification

A linear regression model outputs a continuous value (from -∞ to +∞). This is not suitable for a classification task where we need a probability (a value between 0 and 1).

`ŷ = m₁x₁ + m₂x₂ + ...` → This output `ŷ` can be any number.

### The Solution: The Sigmoid Function

Logistic Regression solves this by taking the output of a linear equation and passing it through a special function called the **Sigmoid** (or **Logistic**) function.

**The Sigmoid Function:**
```
sigmoid(z) = 1 / (1 + e⁻ᶻ)
```
where `z` is the output of the linear part (`z = mx + c`).



This function takes any real number `z` and squashes it into a value between 0 and 1.

### Interpretation

The output of the sigmoid function is interpreted as the **probability** of the positive class.

-   If `sigmoid(z)` is close to 1, the model predicts the positive class (e.g., "Sick").
-   If `sigmoid(z)` is close to 0, the model predicts the negative class (e.g., "Healthy").

**Example:**
-   Input: Temperature = 39.9°C
-   Linear part `z` is calculated.
-   `sigmoid(z)` outputs `0.8`.
-   **Interpretation:** There is an **80% probability** that the person is sick.

A decision threshold (usually 0.5) is used to make a final classification:
> If `sigmoid(z) > 0.5`, predict `1` (Yes/Positive).
>
> Else, predict `0` (No/Negative).




| Concept                    | What It Does                          | Type                   | Key Idea                                  |
| -------------------------- | ------------------------------------- | ---------------------- | ----------------------------------------- |
| **R² Score**               | Checks how well regression model fits | Regression             | Closer to 1 = better fit                  |
| **Bias–Variance Tradeoff** | Balances underfitting vs overfitting  | Concept                | Too simple = bias, Too complex = variance |
| **L1/L2 Regularization**   | Controls model complexity             | Regression/Classifiers | Penalize large weights                    |
| **Logistic Regression**    | Predicts probabilities (Yes/No)       | Classification         | Uses sigmoid function                     |
