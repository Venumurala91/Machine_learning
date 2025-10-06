
# Machine Learning I - Class Notes
**Date:** 21-09-2025

## 1. Why Does Machine Learning Exist?
*Summary: Machine Learning exists to solve complex problems where traditional rule-based programming is inefficient or impossible, by enabling systems to learn patterns directly from data.*

### Traditional Programming vs. Machine Learning

Traditional programming relies on developers explicitly writing rules to handle specific conditions. This approach, also known as "hard-coding," works well for simple, well-defined problems but becomes unmanageable as complexity grows.

**Example: A Rule-Based System**
A simple retail application might use the following rules:
```python
# Rule for senior discount
if customer_age > 60:
    apply_senior_discount()

# Rule for a special offer
if sales_total < 1000:
    print("No special offer applies.")
```
This system is rigid. Adding more rules for new scenarios makes the code increasingly complex and difficult to maintain.

### The Machine Learning Approach

Machine Learning is better suited for problems where the rules are not easily defined. Instead of hard-coding the logic, we let an algorithm learn the patterns from data.

**Example: Predicting a Purchase**
Imagine a clothing store wants to predict whether a customer will buy a winter jacket.

*   **Goal (Target):** Predict `True` (will buy) or `False` (will not buy).
*   **Input Data (Features):**
    *   `temperature`: 17°C
    *   `customer_age`: 30
    *   `customer_wore_jacket`: False
    *   `customer_visited_before`: True

A traditional approach would require a complex nested `if-else` structure that is nearly impossible to get right:
```python
if temperature < 20:
    if not customer_wore_jacket:
        if customer_visited_before:
            print("Customer will buy a jacket")
        else:
            # More rules needed...
            print(...)
```
With Machine Learning, a model learns the relationship between the features and the target outcome from historical sales data.

## 2. The Context of AI and Statistics
*Summary: ML is a subfield of AI that uses statistical principles to build models. Its growth has been fueled by the availability of large datasets and affordable computational power.*

### Relationship with Statistics
Statistics is the foundation of Machine Learning. It provides the methods for **Exploratory Data Analysis (EDA)**, inference, and model evaluation. The core idea is to make decisions and take actions that are backed by data analysis. As more features are added to a problem, statistical learning methods become far more effective than manual rule creation.

### The Rise of Modern Machine Learning
While the theoretical foundations were laid in the 1950s (e.g., neural networks in 1958), the practical application of ML has accelerated due to two main factors:
1.  **Increased Amount of Data:** The digital world generates vast amounts of data to train models.
2.  **Decreased Computation Cost:** The cost of processing power has dropped significantly, making it feasible to train complex models.

### What is Artificial Intelligence (AI)?
AI is the broad field of creating systems that can perform tasks that typically require human intelligence.

*   **AI Subfields:**
    *   **Rule-Based Systems:** Rely on logic and predefined rules.
    *   **Machine Learning (ML):** Systems that learn from data. This is the engine behind modern AI applications like ChatGPT, CoPilot, and Gemini.

*(Diagram: A Venn diagram shows that Artificial Intelligence (AI) is the largest circle. Inside it is a smaller circle for Machine Learning (ML). Inside ML is an even smaller circle for Deep Learning (DL). Within DL are specific applications like Computer Vision (CV) and Natural Language Processing (NLP). Reinforcement Learning (RL) is also shown as a part of ML.)*

*   **Machine Learning (ML):** A subset of AI where systems improve their performance on a task by learning from data, without being explicitly programmed for every case.
*   **Deep Learning (DL):** A subset of ML that uses multi-layered neural networks, inspired by the structure of the human brain.

---

## 3. Types of Machine Learning
*Summary: Machine learning is primarily categorized into three types: Supervised Learning (learning from labeled data), Unsupervised Learning (finding patterns in unlabeled data), and Reinforcement Learning (learning through trial and error).*

### Supervised Learning
In supervised learning, the model learns from data that is already labeled with the correct output. It's like a student learning with a supervisor who provides the answers.

*   **Goal:** Predict an output for new, unseen data.
*   **Key Terminology:**
    *   **Features:** The input variables used for prediction (also called *independent variables*).
    *   **Target (or Label):** The output variable you are trying to predict (also called the *dependent variable* or *ground truth*).

**Example: Predicting Income**
Given a dataset of employees, we want to predict their income.
*   **Features:** `age`, `total_years_experience`, `tech_field`
*   **Target/Label:** `income`

*(Diagram: A table shows columns for 'age', 'total-no-exp', and 'tech' grouped as independent 'features'. A separate column for 'income' is marked as the dependent 'label' to be predicted.)*

The model learns the relationship between the features and the income from the training data, and can then predict the income for a new employee.

### Unsupervised Learning
In unsupervised learning, the model works with unlabeled data and tries to find hidden patterns or structures on its own.

*   **Goal:** Discover underlying groups or patterns in the data (e.g., clustering, anomaly detection).

**Example: Customer Segmentation**
Given data about customers (`humidity`, `month` of purchase), an unsupervised model can segment them into groups like "winter shoppers," "summer shoppers," and "rainy-season shoppers" without being told about these categories beforehand.

*(Diagram: A scatter plot shows many data points. An unsupervised algorithm draws boundaries to separate the points into two distinct clusters, labeled 'A' and 'B'.)*

**Combining Unsupervised and Supervised Learning:**
Unsupervised learning can be a preliminary step for supervised learning.
1.  **Unsupervised Step:** Use a clustering algorithm to group a large set of unlabeled images into distinct clusters.
2.  **Labeling Step:** A human expert labels each cluster (e.g., "Dogs," "Cats").
3.  **Supervised Step:** Use this newly labeled dataset to train a supervised model that can classify new, unseen images as either a dog or a cat.

### Reinforcement Learning (RL)
Reinforcement learning is a more complex type of learning where an "agent" learns to make decisions by performing actions in an environment to achieve a goal.

*   **Goal:** The agent learns the best sequence of actions (a "policy") through trial and error, guided by a system of rewards and penalties.
*   **Core Concept:** An agent takes an **action**, receives a **reward** (for a good action) or a **penalty** (for a bad one), and adjusts its strategy to maximize its total reward over time.

*(Diagram: A simple grid-based game where an agent must navigate a maze with obstacles to reach an exit. A successful move yields a reward, while hitting a wall or a dead end results in a penalty.)*

*   **Applications:**
    *   Autonomous vehicles (e.g., Tesla)
    *   Robotics
    *   Game playing (e.g., Chess, Go)
    *   Resource optimization

---

## 4. Machine Learning Preparation: Data Splitting
*Summary: To accurately evaluate a model's performance on unseen data, the dataset must be split into separate sets for training, validation, and testing.*

### The Need for Splitting Data
You cannot evaluate your model's performance on the same data it was trained on. This would be like giving a student an exam with the exact same questions they studied—it only measures memorization, not true understanding (generalization).

*   **Overfitting:** A common problem where a model performs perfectly on the training data but fails on new, unseen data. This happens when the model memorizes the training data, including its noise and outliers, instead of learning the underlying pattern.

### Train-Test Split
The most basic way to split data is into two sets:
1.  **Training Set:** A subset of the data (e.g., 80%) used to train the model.
2.  **Testing Set:** The remaining portion (e.g., 20%) that is held back. This data is used only once, at the very end, to provide an unbiased evaluation of the final model's performance in the real world.

### Train-Validation-Test Split
A more robust approach, especially for tuning a model, is to use three sets:
1.  **Training Set:** Used to train the model (e.g., 70%).
2.  **Validation Set:** Used to tune the model's hyperparameters and make decisions about the model's architecture (e.g., 15%). Think of this as a "weekly quiz" to check progress and make improvements.
3.  **Testing Set:** Used for the final, unbiased evaluation of the trained model (e.g., 15%). This is the "final exam."

**Common Split Ratios:**
*   **70% Train / 15% Validation / 15% Test**
*   **80% Train / 10% Validation / 10% Test**

Using a validation set helps prevent overfitting by allowing you to see how your model performs on data it hasn't been trained on during the development process, reserving the test set for a truly final check.

---

## Summary of Key Concepts

*   **Why ML?** ML is used to solve problems with complex, non-obvious patterns that are difficult to program with explicit rules.
*   **AI, ML, and DL:** ML is a subset of AI that learns from data. DL is a subset of ML that uses neural networks.
*   **Types of Learning:**
    *   **Supervised:** Learning with labeled data (predicting a target).
    *   **Unsupervised:** Finding patterns in unlabeled data (grouping or clustering).
    *   **Reinforcement:** Learning through rewards and penalties (decision-making).
*   **Data Splitting:** To avoid overfitting and get a true measure of performance, data must be split into training, validation, and testing sets.
