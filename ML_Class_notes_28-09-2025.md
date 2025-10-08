
# Machine Learning Lecture Notes (28-09-2025)

## Agenda: MC-III
- Gradient & Gradient Descent
- Practical Linear Regression (from EDA to Model Prediction)

---

## 1. Linear Regression Fundamentals

Linear Regression is a supervised learning algorithm used to predict a continuous target variable (`y`) based on one or more predictor variables (`x`).

### Dataset Example

Here is a sample dataset with two features: "Years of Experience" (`x1`) and "Completed Projects" (`x2`), used to predict "Salary" (`y`).

| Years of exp (x1) | Completed projects (x2) | Salary.k (y) |
| :---------------- | :---------------------- | :----------- |
| 1 | 0 | 39.764 |
| 2 | 1 | 48.900 |
| 3 | 1 | 56.978 |
| 4 | 2 | 68.290 |
| 5 | 3 | 77.867 |
| 6 | 4 | 85.022 |

### The Model Equation

The goal is to find the best-fit line that describes the relationship between the features and the target. For a single feature:

```
y = mx + c
```

- `y`: The target variable (e.g., Salary)  
- `x`: The feature variable (e.g., Years of Experience)  
- `m`: The slope or coefficient, representing the weight of the feature  
- `c`: The intercept or bias, the value of `y` when `x` is 0  

Model prediction:

```
ŷ = mx + c
```

Error for a single data point:

```
Error = y - ŷ
```

*Diagram Description: The notes show a scatter plot of data points with a best-fit line. The line represents the model's predictions. The goal of linear regression is to find the line that minimizes the total distance (error) from all data points.*

---

## 2. Multiple Linear Regression and Matrix Operations

When we have multiple features (e.g., `x1`, `x2`), the equation expands.

### From Simple to Multiple Regression

**Previous (Simple):**

```
y = mx + c
```

**New (Multiple Features):**

```
y = m1*x1 + m2*x2 + ... + mn*xn + c
```

This can be expressed efficiently using vectors and matrices:

```
Y = MX + C
```

- `Y`: Vector of target values  
- `M`: Matrix of coefficients (`m1, m2, ...`)  
- `X`: Matrix of feature values  
- `C`: Intercept term  

### Matrix Multiplication

Matrix multiplication is fundamental to vectorized implementations of machine learning algorithms.

**Condition:** The number of columns in the first matrix must equal the number of rows in the second matrix.  
**Output Shape:** If Matrix A is `(r x c)` and Matrix B is `(c x k)`, the resulting matrix will be `(r x k)`.

**Example:**

Let Matrix A be `(2x3)` and Matrix B be `(3x2)`:

```
A = [[1, 2, 3],
     [4, 5, 6]]

B = [[7, 8],
     [9, 10],
     [11, 12]]
```

The resulting `(2x2)` matrix is:

```
[[ (1*7 + 2*9 + 3*11),  (1*8 + 2*10 + 3*12) ],
 [ (4*7 + 5*9 + 6*11),  (4*8 + 5*10 + 6*12) ]]
```

```
= [[58, 64],
   [139, 154]]
```

---

## 3. Gradient Descent

**Gradient Descent** is an iterative optimization algorithm used to find the minimum of a function. In machine learning, we use it to find the model parameters (`m` and `c`) that minimize the **cost function** (e.g., total error).

### The Cost Function

The cost function measures how well the model is performing. A common one is the **Mean Squared Error (MSE)**:

```
J = (1/n) * Σ (yi - ŷi)²
```

- `n`: Number of data points  
- `yi`: Actual value for the i-th data point  
- `ŷi`: Predicted value for the i-th data point  

Our goal is to find the parameters (`m0, m1, m2, ...`) that result in the lowest possible cost (`J`).

*Diagram Description: The notes show a 3D "bowl-shaped" plot representing the cost function. The two horizontal axes are model parameters (e.g., `m1`, `m0`), and the vertical axis is the cost. The lowest point corresponds to the minimum cost and the optimal parameters.*

### How Gradient Descent Works

1. **Initialize Parameters:** Start with random values for the parameters (`m` and `c`).  
2. **Calculate the Gradient:** Compute the gradient (the partial derivative) of the cost function with respect to each parameter. The gradient tells us the direction of the steepest ascent (uphill).  
3. **Update Parameters:** Take a step in the opposite direction of the gradient (downhill) to move towards the minimum. The size of this step is controlled by the **learning rate**.  
4. **Repeat:** Repeat steps 2 and 3 until the cost function converges to a minimum.

> **Key Idea:** The slope/gradient tells us the direction of the steepest ascent (uphill). To minimize our cost, we take a step in the opposite direction (downhill).

---

## 4. The Mathematics of Gradient Descent

To update our parameters, we need to calculate the derivative of the cost function.

### Cost Function in Vectorized Form

Let `β` be the vector of parameters (including the intercept). The cost function `J(β)` can be written as:

```
J(β) = (1 / 2m) * (Y - Xβ)ᵀ (Y - Xβ)
```

*(The `1/2` is added for mathematical convenience to simplify the derivative.)*

### Deriving the Gradient

We need to find the partial derivative of `J(β)` with respect to `β`, denoted `∂J/∂β`.

1. **Expand the expression:**

```
S(β) = (Y - Xβ)ᵀ (Y - Xβ)
S(β) = (Yᵀ - βᵀXᵀ)(Y - Xβ)
S(β) = YᵀY - YᵀXβ - βᵀXᵀY + βᵀXᵀXβ
```

Since `YᵀXβ` is a scalar, it equals its transpose `βᵀXᵀY`.

```
S(β) = YᵀY - 2YᵀXβ + βᵀXᵀXβ
```

2. **Take the derivative with respect to `β`:**

Using matrix calculus rules:

```
∂/∂β (YᵀY) = 0
∂/∂β (-2YᵀXβ) = -2XᵀY
∂/∂β (βᵀXᵀXβ) = 2XᵀXβ
```

3. **Combine the terms:**

```
∂S/∂β = -2XᵀY + 2XᵀXβ = 2Xᵀ(Xβ - Y)
```

4. **Final Gradient Formula:**

Since `ŷ = Xβ`, we can write this as `2Xᵀ(ŷ - Y)`.

Including the `(1/2m)` factor from the cost function:

```
∂J/∂β = (1/m) * Xᵀ(ŷ - Y)
```

In Python-like code:

```python
gradient = (X.T @ (y_pred - y)) / m
```

### The Parameter Update Rule

Finally, update parameters in each iteration:

```
β_new = β_old - (α * gradient)
```

- `β`: Vector of model parameters (`m0, m1, ...`)  
- `α`: Learning rate (e.g., 0.01), a small positive value controlling step size  

