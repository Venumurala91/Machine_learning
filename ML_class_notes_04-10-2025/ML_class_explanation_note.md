

# Machine Learning Class Notes (04-10-2025)

## Agenda

- **Topic**: Practical Linear Regression
- **Methods**:
    - **Gradient Descent**: An iterative optimization algorithm.
        - **Batch Gradient Descent**: Uses the entire dataset to compute the gradient at each step.
        - **Stochastic Gradient Descent (SGD)**: Uses a single data point to compute the gradient at each step, making it faster but more noisy.

---

## 1. The Machine Learning Workflow

A typical ML project follows these general stages.

### 1.1. Data Preparation

The initial data is usually split into two main files:
1.  `train.csv`: Contains both **features** and the **target column**. This data is used to train the model.
2.  `test.csv`: Contains only the **features**. The model will predict the target for this data.

The goal is to train a model that can take input features (`X`) and make a prediction (`ŷ`) that is as close as possible to the actual target value (`y`).

### 1.2. Standard ML Pipeline

A common pipeline for building a model can be visualized as:

> **Data** → **EDA** (Exploratory Data Analysis) → **FE** (Feature Engineering) → **ML Model**

Key steps within this pipeline include:

-   **Feature Engineering (FE)**:
    -   **Encoding**: Converting categorical data into a numerical format.
        -   **OHE**: One-Hot Encoding
        -   **TE**: Target Encoding
    -   **Outlier Handling**:
        -   **Good**: Sometimes outliers contain valuable information and should be kept.
        -   **Bad**: In other cases, they are noise and should be removed.

### 1.3. Data Splitting (`train_test_split`)

To evaluate a model's performance, we split the data. A common strategy is:
-   **Method**: Use a function like `train_test_split`.
-   **Ratio**: The data is often split into a training set and a testing set.
    -   Example: A 90/10 split, where 90% is used for training and validation, and 10% is held out as a final test set.
    -   The 90% training data can be further split (e.g., 70% for training, 30% for validation).



---

## 2. Important Concepts in Preprocessing

### 2.1. Handling Unseen Categories in Encoding

When using One-Hot Encoding, the encoder is "fit" on the training data. If the test data contains categories not seen during training, they cannot be encoded correctly.

**Example:**
-   **Training Data Categories**: `['Blue', 'Green', 'Red']`
-   After OHE, these might become:
    - `Blue` → `[1, 0, 0]`
    - `Green` → `[0, 1, 0]`
    - `Red` → `[0, 0, 1]`
-   **Test Data Categories**: `['Blue', 'Green', 'Red', 'Violet', 'Yellow']`
-   **Problem**: The model has never seen `Violet` or `Yellow`.
-   **Solution**: These unknown categories are typically ignored and encoded as all zeros: `[0, 0, 0]`.



### 2.2. `fit`, `transform`, and `fit_transform`

This is a critical concept to prevent **data leakage**.

-   **`fit()`**: The model (e.g., a `StandardScaler`) learns parameters from the data. For a scaler, this means calculating the mean and standard deviation. **This should only be done on the training data.**
-   **`transform()`**: The model applies the learned parameters to transform the data.
-   **`fit_transform()`**: A convenience method that performs `fit()` followed by `transform()` on the same data.

> 💡 **Golden Rule**:
> - Use **`fit_transform()` on the training data**.
> - Use only **`transform()` on the validation and test data**.

This ensures that the model parameters (like scaling factors) are learned *exclusively* from the training data, mimicking a real-world scenario where future data is unknown.



### 2.3. What is Data Leakage?

Data leakage occurs when information from outside the training dataset is used to create the model. Using `fit()` or `fit_transform()` on the entire dataset (including test data) is a common example. This leads to an overly optimistic performance evaluation, as the model has "cheated" by seeing the test data during training.

### 2.4. The `random_state` Parameter

-   **Purpose**: Many algorithms and functions in ML (like `train_test_split` or model initializations) have a random component. Setting a `random_state` (with an integer like `42`) acts as a **seed**, ensuring that the random numbers generated are the same every time the code is run.
-   **Benefit**: This guarantees **reproducibility** of results.

```python
# Example of using a seed for reproducibility
import numpy as np

# Without a seed, the results will change each time
print(np.random.randint(1, 100, 5))

# With a seed, the results are always the same
np.random.seed(42)
print(np.random.randint(1, 100, 5))
# Output will always be: [52, 93, 15, 72, 61]
```

---

## 3. Gradient Descent Deep Dive

### 3.1. The Update Rule

Gradient Descent works by iteratively adjusting the model's parameters (weights or coefficients, denoted by `β`) to minimize a cost function (like error).

The core update rule is:
$$ \beta_{\text{new}} = \beta_{\text{old}} - \alpha \cdot \nabla J(\beta) $$
Where:
-   $ \beta_{\text{new}} $ is the updated parameter.
-   $ \beta_{\text{old}} $ is the current parameter.
-   $ \alpha $ (alpha) is the **learning rate**.
-   $ \nabla J(\beta) $ is the **gradient** of the cost function (the direction of steepest ascent).

### 3.2. The Learning Rate ($\alpha$)

The learning rate is a hyperparameter that controls how large of a step we take during each update.

-   **Too High**: A large `α` can cause the algorithm to "overshoot" the minimum and fail to converge.
-   **Too Low**: A small `α` will make convergence very slow, requiring many more iterations.



---

## 4. Linear Regression: Gradient Descent vs. OLS

Linear Regression can be solved using two main approaches: an iterative method (Gradient Descent) and an analytical method (Ordinary Least Squares).

| Feature                | Gradient Descent (GD)                               | Ordinary Least Squares (OLS)                                |
| ---------------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| **Method**             | Iterative: Updates parameters over many iterations. | Analytical: Solves for parameters directly in a single step.|
| **Best For**           | Medium to very large datasets.                      | Small to medium-sized datasets.                             |
| **Computational Cost** | Efficient for large datasets with many features.    | Computationally expensive for large datasets.               |
| **Implementation**     | `sklearn.SGDRegressor`                              | `sklearn.LinearRegression`                                  |

### 4.1. The Cost Function

Both methods aim to minimize the **Sum of Squared Errors (SSE)**. The cost function for GD is often written as a scaled version, the **Mean Squared Error (MSE)**, for mathematical convenience.

-   **SSE (minimized by OLS)**:
    $$ J(\beta) = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 $$

-   **MSE (minimized by GD)**:
    $$ J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2 $$

### 4.2. The Formulas

#### Gradient Descent Update Rule (for Linear Regression)
This is an iterative process repeated multiple times.
$$ \beta_{\text{new}} = \beta_{\text{old}} - \alpha \cdot \frac{1}{m} X^T(X\beta - y) $$

#### OLS Closed-Form Solution
This is a direct calculation, also known as the **Normal Equation**.
$$ \beta = (X^T X)^{-1} X^T y $$

This formula is derived by taking the derivative of the cost function with respect to `β`, setting it to zero, and solving for `β`.