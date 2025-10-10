
### **`05-10-2025`**

## Agenda

*   R² score
*   Bias & Variance
*   L1 & L2 Regularization
*   practical
*   Logistic Regression (Depth)

---

## R-squared (R²) Score

R-squared is a metric used to evaluate the performance of a regression model. It represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

![A scatter plot showing data points for weight vs. height.](https://i.imgur.com/scatter_plot_initial.png "Initial Data Points")

![A scatter plot with a line of best fit (y=mx+c) drawn through the data points. The vertical distance from each point to the line is shown as a 'residual'.](https://i.imgur.com/scatter_plot_fit.png "Line of Best Fit")

To understand R², we first need to understand variance.

1.  **Variance around the Mean:** If we only use the mean of the weight to predict the weight, the error is the total variation in the data.
    *   Mean: `mean = Σyi / n`
    *   Sum of Squares around Mean `SS(mean)`: `Σ(data - mean)²`
    *   Variation around Mean: `SS(mean) / n`

    ![A scatter plot illustrating the calculation of the mean of the y-values (weight) and the variance of each data point from that mean.](https://i.imgur.com/variance_mean.png "Variance around the Mean")

2.  **Variance around the Fit:** This is the error of our model. It's the variation of the data points around the fitted regression line.
    *   Sum of Squares around Fit `SS(fit)`: `Σ(data - fit)²`

    ![A scatter plot showing the residuals (error) between the actual data points and the predicted values on the regression line.](https://i.imgur.com/variance_fit.png "Variance around the Fit")

### R² Formula

The R² score is the reduction in variance achieved by the model compared to just using the mean.

$$
R² = \frac{\text{var}(\text{mean}) - \text{var}(\text{fit})}{\text{var}(\text{mean})}
$$

In other words, R² tells us how much of the variation in person's weight we can explain by taking person's height into account.

**Example Calculation:**
*   `var(mean) = 72`
*   `var(fit) = 24`

$$
R² = \frac{72 - 24}{72} = 0.66 \approx 66\%
$$

**Interpretation:**
*   There is a 66% reduction in variance when we take person height into account.
*   Person height explains 66% of the variation in person weight.
*   R² tells us how much of the variation in the target variable is explained by the model. Higher is better, but context matters.

---

## Bias & Variance

These are two fundamental sources of error in a machine learning model.

![A scatter plot showing two sets of data points: 'Train' in blue and 'Test' in green.](https://i.imgur.com/train_test_data.png "Train vs. Test Data")

### Bias
The inability of a machine learning model to capture the true relationship in the data. High bias models are often too simple.

*   **High Bias (Underfitting):** The model is too simple to learn the underlying structure of the data.
    *   Example: Using a linear model for a non-linear relationship.
    *   It can't capture the relationship well.
    *   `Loss(Train): 100`
    *   `Loss(Test): 98`
    *   (Note: Both training and test loss are high).

![A graph showing a linear line failing to capture the trend of non-linear training data, indicating high bias.](https://i.imgur.com/high_bias.png "High Bias")

### Variance
The difference in fits between the training and test datasets. High variance models are often too complex and capture noise in the training data.

*   **High Variance (Overfitting):** The model is too flexible and fits the training data too closely, including its noise.
    *   Example: A complex polynomial model.
    *   It performs well on training data but poorly on unseen test data.
    *   It gives very different results on train vs. test sets.
    *   `Loss(Train): 0`
    *   `Loss(Test): 70`
    *   (Note: Training loss is very low, but test loss is high).

![A graph showing a highly flexible line perfectly fitting all training data points but likely failing on test data, indicating low bias but high variance.](https://i.imgur.com/high_variance.png "High Variance / Overfitting")

### The Bias-Variance Tradeoff

The goal is to find a balance between bias and variance.
*   `→` **Bias & variance tradeoff**
*   `→` **Low bias & low variance (ideal)**

---

## Regularization

A technique used to prevent overfitting by penalizing large coefficients in the model. The core idea is: **Don't just minimize error, keep the coefficients controlled.**

### Problems with Vanilla Linear Regression
Standard algorithms like Ordinary Least Squares (OLS) or Stochastic Gradient Descent (SGD) work perfectly if:
*   You have plenty of data compared to the number of features.
*   Features are not strongly correlated (no multicollinearity).
*   The model is not too complex (not too many parameters).

Regularization adds a penalty term to the loss function to control model complexity.

### L1 Regularization (Lasso)

Adds a penalty equal to the **sum of the absolute values** of the coefficients.

$$
\text{Loss}_{\text{new}} = \text{MSE} + \lambda \sum_{i=1}^{n} |\beta_i|
$$

*   **Multicollinearity:** Handles correlated features well.
*   **Feature Selection:** It can shrink some coefficients to exactly zero, effectively removing them from the model.

### L2 Regularization (Ridge)

Adds a penalty equal to the **sum of the squared values** of the coefficients.

$$
\text{Loss}_{\text{new}} = \text{MSE} + \lambda \sum_{i=1}^{n} \beta_i^2
$$

*   It shrinks coefficients towards zero but **never to exactly zero**.
*   It is effective at preventing overfitting when many features are present.

### Summary of Linear Regression Variants

*   **Loss(OLS):** `SSE` → `R² / F1`
*   **Loss(OLS) + L1:** `SSE + λ Σ|βi|` → `R² / F1`
*   **Loss(OLS) + L2:** `SSE + λ Σ(βi)²` → `R² / F1`
*   **Loss(GD) + L1:** `MSE + λ Σ|βi|` → `R² / F1`
*   **Loss(GD) + L2:** `MSE + λ Σ(βi)²` → `R² / F1`

---

## Logistic Regression

A classification algorithm used to predict a binary outcome (e.g., Yes/No, 1/0, True/False).

While linear regression outputs a continuous value (`-∞` to `+∞`), logistic regression outputs a probability between 0 and 1.

$$
\text{Linear Regression: } \hat{y} = x_1m_1 + x_2m_2
$$

This continuous output `ŷ` needs to be transformed into a probability. This is done using the **Sigmoid (or Logistic) function**.

### The Sigmoid Function

The sigmoid function takes any real-valued number and maps it to a value between 0 and 1.

$$
\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}
$$

Where `z` is the output of the linear part of the model (`z = mx + c`).

![A graph of the sigmoid function, an S-shaped curve that maps inputs from negative infinity to positive infinity into an output range of 0 to 1.](https://i.imgur.com/sigmoid_curve.png "Sigmoid Function")

*   A large positive `z` results in a value close to 1.
*   A large negative `z` results in a value close to 0.
*   `z = 0` results in a value of 0.5.

**Example Application:**
*   **Input:** Temperature = `39.9`
*   **Model Output (z):** `(Some value)`
*   **Sigmoid Output (ŷ):** `0.8`
*   **Interpretation:** There is an 80% chance the person is sick.

A decision boundary is used to convert this probability into a class label.
*   **Rule:** `IF sigmoid(z) > 0.5 THEN predict 1 ELSE predict 0`