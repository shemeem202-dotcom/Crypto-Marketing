# Crypto Marketing & Price Movement Prediction

## Project Overview

This project focuses on predicting the future movement of cryptocurrency prices using Machine Learning classification algorithms. The model analyzes historical cryptocurrency market data and predicts whether the closing price of a cryptocurrency will increase or decrease on the next day.

The project includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Training
- Hyperparameter Tuning
- Model Evaluation
- Visualization of Results

---

## Dataset

**Dataset:** `crypto_daily.csv`

The dataset contains daily cryptocurrency market information such as:

| Feature | Description |
|---------|-------------|
| date | Trading date |
| symbol | Cryptocurrency symbol |
| name | Cryptocurrency name |
| open | Opening price |
| high | Highest price |
| low | Lowest price |
| close | Closing price |
| volume | Trading volume |
| rsi_14 | 14-day Relative Strength Index |
| volatility_30d | 30-day volatility |

### Target Variable

```python
target = 1 → Next day's closing price is higher
target = 0 → Next day's closing price is lower
```

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset Shape & Information
- Missing Value Analysis
- Target Distribution
- Correlation Matrix
- Pair Plot Analysis
- Feature Distribution
- Outlier Detection

### Visualizations

- Count Plot
- Heatmap
- Pair Plot
- Histograms
- Boxplots

---

## Data Preprocessing

### Numerical Features

- Missing values filled using **Simple Imputer**
- Features scaled using **StandardScaler**

### Categorical Features

- Missing values handled
- Converted using **One Hot Encoding**

### Pipeline

The preprocessing steps are combined using:

```python
Pipeline()
ColumnTransformer()
```

---

## Machine Learning Algorithms

The following classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. Extra Trees Classifier

---

## Hyperparameter Tuning

To improve model performance:

```python
GridSearchCV
```

was used for hyperparameter optimization.

---

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix
- ROC-AUC Score

---

## Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
EDA
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Preprocessing Pipeline
       ↓
Model Training
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Prediction
```

---

## Project Structure

```text
Crypto-Marketing-Project/
│
├── Crypto Marketing.ipynb
├── crypto_daily.csv
├── README.md
│
├── images/
│   ├── heatmap.png
│   ├── pairplot.png
│   └── target_distribution.png
│
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/crypto-marketing-project.git
```

Move into the project directory:

```bash
cd crypto-marketing-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

## Future Improvements

- Deploy the model using Streamlit
- Add LSTM Deep Learning Model
- Integrate Real-Time Crypto API
- Build Interactive Dashboard
- Add Technical Indicators
