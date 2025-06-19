# 🏠 Real Estate Price Prediction 📈

Welcome to the **Real Estate Price Prediction** project!  
This project predicts house prices using features like price per square foot, number of bedrooms (single/double), and more.  
Perfect for buyers, sellers, and real estate enthusiasts! 🚀

---

## 📋 Overview

- **Objective:**  
  Use machine learning to predict house prices based on factors such as area, price per sq. ft., bedrooms, and more.
- **Tech Stack:**  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)  
  Jupyter Notebook, pandas, scikit-learn, matplotlib, seaborn

---

## 🔍 Features Used

- 📏 **Price per Square Foot**
- 🛏️ **Number of Bedrooms (Single/Double)**
- 🛁 Bathrooms
- 📐 Area (sqft)
- 🌍 Location

---

## 🚦 How it Works

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('data.csv')

# Select features and target
X = data[['area_sqft', 'bedrooms', 'bathrooms', 'price_per_sqft']]
y = data['price']

# Model training
model = LinearRegression()
model.fit(X, y)
```

---

## 📊 Exploratory Data Analysis

- **Visualizations:**  
  - Scatter plots of price vs area, colored by bedroom type  
  - Bar charts comparing single vs double bedroom prices  
  - Correlation heatmaps

<details>
<summary>Sample code for EDA 📊</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='area_sqft', y='price', hue='bedroom_type', data=data)
plt.title("Price vs Area by Bedroom Type")
plt.show()
```
</details>

---

## 🏗️ Model Building

- **Algorithms Used:** Linear Regression, Random Forest, etc.
- **Evaluation Metrics:** RMSE, R² Score

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X)
print("RMSE:", mean_squared_error(y, y_pred, squared=False))
print("R² Score:", r2_score(y, y_pred))
```

---

## 📁 Files

- `Real-Estate-Price-Prediction.ipynb` – Main analysis and model notebook
- `data.csv` – Training dataset (if available)
- `README.md` – Project documentation

---

## 🚀 How to Run

1. **Clone the repo:**
    ```bash
    git clone https://github.com/boddapuchandrasekhar/Real-Estate-Price-Prediction.git
    cd Real-Estate-Price-Prediction
    ```
2. **Open in Jupyter Notebook or Colab**
3. **Run all cells to see analysis, training, and prediction results!**

---

## 🎨 Example Output

- 📈 Predicted house prices vs. actual prices
- 🏅 RMSE and R² scores for model performance
- 📊 Plots and charts for data insights

---

## 👤 Author

**Boddapu Chandra Sekhar**  
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=flat&logo=gmail&logoColor=white)](mailto:boddapuchandu2004@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/boddapuchandrasekhar)

---

## 🌟 Star this repo if you found it helpful!  
Let’s make real estate smarter together! 🏠✨
