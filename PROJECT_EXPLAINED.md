# AQI Prediction Project — Complete Beginner's Guide
### Everything Explained From Scratch — What It Is, How It Works, and Why

---

## Table of Contents

1. [What Is This Project About?](#1-what-is-this-project-about)
2. [What is AQI?](#2-what-is-aqi)
3. [The Dataset — What Data Are We Working With?](#3-the-dataset--what-data-are-we-working-with)
4. [Step 1 — Loading the Data](#4-step-1--loading-the-data)
5. [Step 2 — Cleaning the Data](#5-step-2--cleaning-the-data)
6. [Step 3 — Exploratory Data Analysis (EDA)](#6-step-3--exploratory-data-analysis-eda)
7. [Step 4 — Preparing Data for Machine Learning](#7-step-4--preparing-data-for-machine-learning)
8. [Step 5 — The Three Machine Learning Models](#8-step-5--the-three-machine-learning-models)
9. [Step 6 — Evaluating the Models](#9-step-6--evaluating-the-models)
10. [Step 7 — Feature Importance](#10-step-7--feature-importance)
11. [Step 8 — Testing the Model on Custom Scenarios](#11-step-8--testing-the-model-on-custom-scenarios)
12. [Step 9 — The Early Warning System](#12-step-9--the-early-warning-system)
13. [Step 10 — The Interactive Map](#13-step-10--the-interactive-map)
14. [The Supporting Scripts](#14-the-supporting-scripts)
15. [Full Picture — How Everything Connects](#15-full-picture--how-everything-connects)
16. [Key Concepts Glossary](#16-key-concepts-glossary)

---

## 1. What Is This Project About?

This project uses **Machine Learning** to predict the **Air Quality Index (AQI)** of a location based on the concentration of various air pollutants.

Think of it like this:

> A weather station has sensors. Those sensors measure pollutant levels in the air (like dust, smoke, chemicals). You give those numbers to your trained ML model. The model tells you: *"Based on these readings, the AQI is approximately 280 — that's Poor air quality."*

### Real-World Use Case
- Government authorities can monitor air quality automatically without manually computing AQI.
- An early warning system can alert citizens when air quality is about to become dangerous.
- An interactive map can visualize which Indian cities have the worst air quality historically.

### What the Project Produces
| Output File | What It Is |
|---|---|
| `full_output.txt` | Summary of model results and test predictions |
| `eda_visualizations.png` | 4 EDA charts |
| `model_comparison.png` | Bar charts comparing 3 models |
| `feature_importance.png` | Which pollutants matter most |
| `best_model_analysis.png` | Actual vs Predicted scatter plot |
| `regplots.png` | AQI vs each pollutant regression lines |
| `boxplots.png` | Pollutant spread/distribution |
| `residual_analysis.png` | How far off the predictions are |
| `interactive_aqi_map.html` | Clickable map of Indian cities by AQI |
| `quick_test.py` | Script to test a single prediction |
| `test_aqi_model.py` | Script with 5 labelled test scenarios |

---

## 2. What is AQI?

**AQI stands for Air Quality Index.** It is a number that tells you how clean or polluted the air is and what health effects you might be concerned about.

### AQI Scale (India — CPCB Standard)

| AQI Range | Category | Health Impact |
|---|---|---|
| 0 – 50 | Good | Minimal Impact |
| 51 – 100 | Satisfactory | Minor breathing discomfort for sensitive people |
| 101 – 200 | Moderate | Breathing discomfort for people with lung/heart disease |
| 201 – 300 | Poor | Breathing discomfort for everyone on prolonged exposure |
| 301 – 400 | Very Poor | Respiratory illness on prolonged exposure |
| 401 – 500 | Severe | Serious health effects even on low exposure |
| 500+ | Hazardous / Emergency | Affects healthy people; serious risk for sensitive groups |

### How is AQI Officially Calculated?
In reality, AQI is computed using a special sub-index formula for each pollutant (PM2.5, PM10, SO2, NO2, CO, O3). The highest sub-index value becomes the overall AQI. This is a deterministic mathematical formula.

**What this project does differently:** Instead of using that formula, it trains a machine learning model to *learn* the relationship between pollutant values and AQI from historical data. The model learns patterns it has seen before and generalizes to new inputs.

---

## 3. The Dataset — What Data Are We Working With?

### File: `city_day-Copy1.csv`

This is a real-world dataset containing **daily air quality measurements** for **multiple Indian cities** from approximately **2015 to 2020**.

- **Total rows:** ~24,850 (each row = one city on one day)
- **Total columns:** 16

### Column-by-Column Breakdown

| Column | What It Means | Unit |
|---|---|---|
| `City` | Name of the Indian city | Text |
| `Date` | The date of measurement | YYYY-MM-DD |
| `PM2.5` | Particulate Matter ≤ 2.5 microns — fine dust, smoke, from vehicles/factories | µg/m³ |
| `PM10` | Particulate Matter ≤ 10 microns — coarser dust, pollen, construction dust | µg/m³ |
| `NO` | Nitric Oxide — from vehicle exhaust and industrial combustion | µg/m³ |
| `NO2` | Nitrogen Dioxide — byproduct of NO reacting with oxygen | µg/m³ |
| `NOx` | Nitrogen Oxides — total of NO + NO2 combined | µg/m³ |
| `NH3` | Ammonia — from agriculture, fertilizers, livestock | µg/m³ |
| `CO` | Carbon Monoxide — from incomplete combustion (cars, fires) | mg/m³ |
| `SO2` | Sulphur Dioxide — from burning coal, diesel, industrial processes | µg/m³ |
| `O3` | Ozone — forms when sunlight reacts with NO2 and VOCs | µg/m³ |
| `Benzene` | Volatile organic compound — from petrol, paint, tobacco smoke | µg/m³ |
| `Toluene` | Volatile organic compound — from paint, adhesives, solvents | µg/m³ |
| `Xylene` | Volatile organic compound — from printing, rubber, paint | µg/m³ |
| `AQI` | The computed Air Quality Index value (our target/output) | Number |
| `AQI_Bucket` | AQI category label (Good, Moderate, Poor, etc.) | Text |

### Sample Row (What One Day's Data Looks Like)
```
City: Ahmedabad
Date: 2015-01-01
PM2.5: 55.0,  PM10: 120.0,  NO: 10.2,  NO2: 28.5
NOx: 38.7,    NH3: 14.0,    CO: 1.8,   SO2: 18.3
O3: 42.1,     Benzene: 2.1, Toluene: 4.5, Xylene: 1.3
AQI: 164,     AQI_Bucket: Moderate
```

### Why Does This Dataset Have Missing Values?
Sensors break, go offline, or are not installed for certain pollutants in certain cities. So many cells are empty (NaN). Handling this is a critical part of data preprocessing.

---

## 4. Step 1 — Loading the Data

### Code (Notebook Cell 2)
```python
df = pd.read_csv('city_day-Copy1.csv')
print("Shape:", df.shape)
df.head()
```

### What is Happening Here?
- `pandas` is a Python library for working with tabular data (like Excel, but in code).
- `pd.read_csv()` reads the CSV file and puts all the data into a **DataFrame** — think of it as a smart table in memory.
- `df.shape` tells you `(rows, columns)` — we get `(24850, 16)`.
- `df.head()` shows you the first 5 rows to visually inspect the data.

### Libraries Imported in Cell 1
```python
import pandas as pd              # Data manipulation (tables)
import numpy as np               # Math operations on arrays
import matplotlib.pyplot as plt  # Drawing charts/graphs
import seaborn as sns            # Prettier statistical charts
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

`sklearn` (scikit-learn) is the machine learning library. It provides ready-made implementations of ML algorithms.

---

## 5. Step 2 — Cleaning the Data

### Code (Notebook Cell 4)
```python
# Drop rows where AQI is missing
df = df.dropna(subset=['AQI'])

# Fill missing pollutant values with column mean
pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
df[pollutant_cols] = df[pollutant_cols].fillna(df[pollutant_cols].mean())
```

### Why Do We Need to Clean Data?
Raw real-world data is almost always messy. Machine learning models cannot process missing values (NaN). You must make a decision about what to do with them.

### Two Strategies Used Here

**Strategy 1: Drop rows where AQI is missing**
- AQI is our target — the thing we are trying to predict. If a row has no AQI, it is useless for training because we have nothing to compare our prediction against.
- So we simply delete those rows.

**Strategy 2: Fill missing pollutant values with the column mean**
- For the input features (the 12 pollutants), we cannot just delete rows because we would lose too much data.
- Instead, we replace each missing value with the **average** of that entire column.
- For example, if PM2.5 has 500 missing values, every one of those 500 slots is filled with the average PM2.5 value across all other days.
- This is called **mean imputation**.

### Result After Cleaning
```
Shape after cleaning: (24850, 16)
Missing values remaining: 0
```

---

## 6. Step 3 — Exploratory Data Analysis (EDA)

### What is EDA?
Before building any model, data scientists look at the data visually to understand:
- What does the data look like?
- Are there patterns?
- Which pollutants seem most related to AQI?
- Are there weird outliers?

This phase is called **Exploratory Data Analysis (EDA)**.

### Four Charts Produced (saved as `eda_visualizations.png`)

---

### Chart 1: AQI Distribution (Histogram)
```python
axes[0,0].hist(df['AQI'], bins=50, color='steelblue', edgecolor='black')
```

**What it shows:** How many days have AQI values in each range (0-50, 50-100, etc.)

**What a histogram does:** It divides the range of values into "bins" (buckets) and counts how many values fall in each. If `bins=50`, the full AQI range is divided into 50 equal slices.

**What to look for:**
- Is the data skewed toward low or high AQI?
- Is it normally distributed (bell curve) or right-skewed (most days are okay but some are very bad)?
- A right-skewed distribution means most days have moderate AQI but a few extreme pollution days push the tail rightward.

---

### Chart 2: Top 10 Most Polluted Cities (Horizontal Bar Chart)
```python
top_cities = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
axes[0,1].barh(top_cities.index, top_cities.values, color='tomato')
```

**What it shows:** The 10 Indian cities with the highest *average* AQI across the entire dataset period.

**How it's computed:**
- `groupby('City')` — group all rows by city name
- `['AQI'].mean()` — for each city, calculate the average AQI
- `sort_values(ascending=False)` — sort from highest to lowest
- `.head(10)` — take only the top 10

**Expected result:** Cities like Delhi, Patna, Lucknow typically top these lists due to traffic, industrial activity, and geographic factors.

---

### Chart 3: Correlation Heatmap
```python
corr = df[corr_cols].corr()
sns.heatmap(corr[['AQI']].sort_values('AQI', ascending=False), annot=True, fmt='.2f', cmap='RdYlGn_r')
```

**What is correlation?** Correlation is a number between -1 and +1 that measures how strongly two variables move together:
- **+1.0** = Perfect positive relationship (as one goes up, so does the other)
- **0.0** = No relationship
- **-1.0** = Perfect negative relationship (as one goes up, the other goes down)

**What it shows here:** How strongly each pollutant is correlated with AQI.

**What to expect:** PM2.5 and CO typically show the highest correlation with AQI (close to 0.8–0.9) because they are the dominant components of the AQI formula. O3 might even show a weaker or negative correlation in some seasons.

**The `RdYlGn_r` colormap:** Red = high correlation (bad/high AQI), Green = low correlation. The `_r` reverses it so red means high.

---

### Chart 4: AQI Bucket Distribution (Bar Chart)
```python
bucket_counts = df['AQI_Bucket'].value_counts()
axes[1,1].bar(bucket_counts.index, bucket_counts.values, color='mediumpurple')
```

**What it shows:** How many data points fall in each AQI category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).

**Why it matters:** If 90% of your data is "Good" and only 10% is "Severe", your model will be very good at predicting Good days but terrible at predicting rare Severe days. This is called **class imbalance**.

---

## 7. Step 4 — Preparing Data for Machine Learning

### Code (Notebook Cell 6)
```python
features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
X = df[features]    # Input features (12 pollutants)
y = df['AQI']       # Target / output (the AQI value)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### X and y — The Most Important Concept in ML

In every supervised machine learning problem:
- **X** = The **inputs** (what you feed the model). Here: 12 pollutant readings.
- **y** = The **output** (what you want the model to predict). Here: the AQI value.

Think of it like: *"Given X (the pollutants), predict y (the AQI)."*

### Train-Test Split

We do NOT show the model all the data during training. We split it:

| Split | Size | Purpose |
|---|---|---|
| Training set | 80% (~19,880 rows) | The model learns from this data |
| Testing set | 20% (~4,970 rows) | We check how well the model performs on data it has never seen |

**Why keep test data secret from the model?**
If you studied from the exact same question paper as the exam, you would get 100% — but that doesn't mean you actually understand. The test set simulates real-world unseen data to give an honest evaluation.

**`random_state=42`:** This ensures the random split is reproducible. Every time you run the code, you get the exact same train/test split. Without it, results would differ each run.

---

## 8. Step 5 — The Three Machine Learning Models

The project trains and compares three different ML models. Here's what each one is and how it works.

---

### Model 1: Linear Regression

```python
LinearRegression()
```

**The Concept:** Fits a straight line (or hyperplane in multiple dimensions) through the data. It tries to find the best formula of the form:

```
AQI = w1×PM2.5 + w2×CO + w3×NO2 + ... + w12×Xylene + intercept
```

Where `w1, w2, ... w12` are **weights** (coefficients) that the model learns during training.

**Strengths:**
- Very fast to train
- Easy to interpret (you can see exactly which pollutant has what weight)
- Works well if the relationship is actually linear

**Weaknesses:**
- Assumes the relationship is linear — but real-world AQI is far more complex
- Cannot model interactions between features (e.g., "PM2.5 matters even more when CO is also high")
- Sensitive to outliers

**Result in this project:** R² = 0.8092 — decent but not great.

---

### Model 2: Random Forest

```python
RandomForestRegressor(n_estimators=100, random_state=42)
```

**The Concept:** A Random Forest is an **ensemble** of many Decision Trees.

**First, what is a Decision Tree?**
A decision tree asks a series of yes/no questions about the data:
```
Is PM2.5 > 150?
  YES → Is CO > 3.0?
           YES → Predict AQI = 380
           NO  → Predict AQI = 270
  NO  → Is NO2 > 30?
           YES → Predict AQI = 120
           NO  → Predict AQI = 60
```

**Then, what is a Random Forest?**
A Random Forest builds 100 decision trees (because `n_estimators=100`). Each tree:
1. Is trained on a **random subset** of the data (called bootstrapping)
2. At each split, considers only a **random subset** of features

Then for a prediction, all 100 trees vote and the average is taken.

**Why is this better than one tree?**
One tree can memorize the training data (overfit). 100 diverse trees each make slightly different errors — averaging them out produces a much more reliable prediction. This technique is called **Bagging** (Bootstrap Aggregating).

**Result in this project:** R² = 0.9105 — Best model. ✓

---

### Model 3: Gradient Boosting

```python
GradientBoostingRegressor(n_estimators=100, random_state=42)
```

**The Concept:** Another ensemble of decision trees, but built differently from Random Forest.

**How Boosting works (step by step):**
1. Train a small, weak decision tree on the data
2. Check where it made mistakes (calculate residuals = actual - predicted)
3. Train a second tree specifically to fix those mistakes
4. Add the two trees together
5. Check where the combined model still makes mistakes
6. Train a third tree to fix those mistakes
7. Repeat 100 times (`n_estimators=100`)

Each new tree *boosts* the performance by correcting previous errors. Trees are built **sequentially**, not independently.

**Key parameter — Learning Rate:** Controls how much each new tree contributes. Smaller = more careful, less risk of overfitting, but slower.

**Difference from Random Forest:**

| | Random Forest | Gradient Boosting |
|---|---|---|
| Tree building | Parallel (independent trees) | Sequential (each fixes previous) |
| Strategy | Averaging independent trees (Bagging) | Correcting errors step by step (Boosting) |
| Speed | Faster | Slower |
| Tendency | Less prone to overfitting | Can overfit if not tuned carefully |

**Result in this project:** R² = 0.8952 — Close second, but Random Forest wins here.

---

## 9. Step 6 — Evaluating the Models

### The Three Metrics

After training, we predict AQI for the test set and compare predictions to actual values.

---

### MAE — Mean Absolute Error

```
MAE = average of |actual AQI - predicted AQI|
```

**Plain English:** On average, how many AQI points is the model off by?

**Example:** If MAE = 20.60, it means on average the model's predictions are off by 20.6 AQI units.

**Lower is better.** MAE treats all errors equally regardless of size.

---

### RMSE — Root Mean Square Error

```
RMSE = square root of (average of (actual - predicted)²)
```

**Plain English:** Similar to MAE, but **punishes large errors more heavily**.

**Why?** Because squaring amplifies large errors. A prediction that's off by 100 contributes 10,000 to the average before square-rooting, while a prediction off by 10 contributes only 100.

**RMSE > MAE always.** A big gap between them means your model has some large outlier predictions.

**Lower is better.**

---

### R² Score — Coefficient of Determination

```
R² = 1 - (sum of squared errors / total variance in y)
```

**Plain English:** What percentage of the variation in AQI is explained by your model?

| R² Value | Meaning |
|---|---|
| 1.00 | Perfect — model explains all variation |
| 0.91 | Model explains 91% of AQI variation (Random Forest) |
| 0.81 | Model explains 81% of AQI variation (Linear Regression) |
| 0.50 | Model explains 50% — barely better than guessing the average |
| 0.00 | Model is no better than just predicting the average every time |
| < 0 | Model is actually worse than predicting the average |

**Higher is better. Maximum is 1.0.**

---

### Final Model Results

| Model | MAE | RMSE | R² | Verdict |
|---|---|---|---|---|
| Linear Regression | 31.20 | 59.11 | 0.8092 | Decent baseline |
| **Random Forest** | **20.60** | **40.49** | **0.9105** | **Best model** |
| Gradient Boosting | 23.71 | 43.80 | 0.8952 | Close second |

### Best Model Selection (Code)
```python
best_model_name = max(results, key=lambda x: results[x]['R2'])
```
This picks the model with the highest R² score — which is Random Forest.

---

### Model Comparison Charts (saved as `model_comparison.png`)
Three bar charts are generated, one each for MAE, RMSE, and R². This gives a visual comparison that's easier to present than raw numbers.

### Actual vs Predicted Scatter Plot (saved as `best_model_analysis.png`)
```python
plt.scatter(y_test, best_preds, alpha=0.3)
plt.plot([0, 2000], [0, 2000], 'r--', label='Perfect Prediction')
```
- Every dot is one test day. X-axis = real AQI. Y-axis = what the model predicted.
- The red dashed line is the "perfect line" — if the model were perfect, all dots would sit exactly on this line.
- Dots close to the line = accurate predictions. Dots far away = errors.
- `alpha=0.3` makes dots semi-transparent so overlapping areas show up as denser blobs.

---

## 10. Step 7 — Feature Importance

### Code
```python
importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=True)
importances.plot(kind='barh', color='steelblue')
```

### What is Feature Importance?

Random Forest can tell you how much each input feature contributed to making accurate predictions. It measures this by tracking how much each feature reduced prediction error across all the trees.

### Results (Feature Importance)

| Feature | Importance Score | Interpretation |
|---|---|---|
| **PM2.5** | **0.491** | By far the most important — alone drives ~49% of prediction quality |
| **CO** | **0.368** | Second most important — together with PM2.5 accounts for ~86% |
| NO | 0.037 | Minor contribution |
| PM10 | 0.037 | Minor contribution |
| O3 | 0.015 | Very small contribution |
| NOx | 0.012 | Very small |
| SO2 | 0.010 | Very small |
| NO2 | 0.008 | Minimal |
| Toluene | 0.007 | Minimal |
| Xylene | 0.006 | Minimal |
| Benzene | 0.005 | Near negligible |
| NH3 | 0.003 | Near negligible |

### Why Does This Matter?
- PM2.5 dominates because India's official AQI formula gives PM2.5 the highest weight — the model learned this from the data.
- CO is the second most important, especially in urban/traffic-heavy environments.
- VOCs (Benzene, Toluene, Xylene) contribute very little in this dataset, possibly because their readings are often low.

---

## 11. Step 8 — Testing the Model on Custom Scenarios

### Code (Notebook Cell 10 & `test_aqi_model.py`)
```python
test_cases = pd.DataFrame({
    'PM2.5': [250,  12,   80,  400,   5],
    'PM10':  [300,  25,  100,  500,  10],
    ...
})
predictions = best_model.predict(test_cases)
```

### What is This Doing?
After training the model, we invented 5 realistic scenarios — realistic pollutant values that might occur in different real-world situations — and let the model predict what AQI those conditions would produce.

### The 5 Test Scenarios

| Scenario | PM2.5 | CO | Predicted AQI | Category | Real-World Context |
|---|---|---|---|---|---|
| Heavy Pollution (Delhi Winter) | 250 | 5.0 | ~393 | Very Poor | Smog + winter temperature inversion traps pollution |
| Moderate Urban Day | 80 | 1.5 | ~172 | Moderate | Typical weekday in a mid-size Indian city |
| Clean Day (Hill Station) | 12 | 0.3 | ~36 | Good | Shimla or Ooty on a clear day |
| Extreme Industrial | 400 | 10.0 | ~508 | Hazardous | Near a factory complex or major fire |
| Near-Clean Rural | 5 | 0.1 | ~31 | Good | Remote village with minimal traffic |

### The `pd.cut()` Function — How Categories Are Assigned
```python
test_cases['Category'] = pd.cut(predictions,
    bins=[0, 50, 100, 200, 300, 400, 500, 9999],
    labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Severe','Hazardous'])
```
`pd.cut()` divides the predicted AQI into range-based buckets. For example, if prediction = 393, it falls in the 301–400 bin, labelled "Very Poor".

---

## 12. Step 9 — The Early Warning System

### Code (Notebook Cell 18 — `aqi_early_warning()` function)

```python
def aqi_early_warning(pm25=200, pm10=250, no2=50, co=4.0, o3=30, ...):
    input_data = pd.DataFrame([{...}])
    predicted_aqi = best_model.predict(input_data)[0]

    if predicted_aqi <= 50:
        level, action = "GOOD", "No action needed."
    elif predicted_aqi <= 100:
        level, action = "SATISFACTORY", "Sensitive individuals should limit exposure."
    ...
    print(f"Predicted AQI: {predicted_aqi:.1f}")
    print(f"Alert Level: {level}")
    print(f"Recommended: {action}")
```

### What is This?
This is the practical application of the trained model — a function that takes pollutant readings as input and gives back:
1. A predicted AQI number
2. An alert level (Good / Moderate / Poor / etc.)
3. A recommended action for citizens/authorities

### How Default Values Work
The function has default values for less critical pollutants (NO, NOx, NH3, SO2, Benzene, Toluene, Xylene). These defaults are set to the dataset mean for each pollutant, so you only **need** to supply the five key readings: PM2.5, PM10, NO2, CO, O3.

### The Three Demo Scenarios
```python
# Normal winter morning in Delhi (Moderate-Poor air)
aqi_early_warning(pm25=180, pm10=220, no2=45, co=3.0, o3=25)

# Clean day in a hill station (Good air)
aqi_early_warning(pm25=15, pm10=20, no2=8, co=0.3, o3=18)

# Industrial zone emergency (Hazardous air)
aqi_early_warning(pm25=350, pm10=420, no2=90, co=8.0, o3=55)
```

---

## 13. Step 10 — The Interactive Map

### Code (Notebook Cell 13 — `interactive_aqi_map.html`)

This section generates a complete HTML file — a webpage — that displays an **interactive map of India** where every city is shown as a coloured circle based on its average AQI.

### Technology Used: Leaflet.js
Leaflet is a JavaScript library for interactive maps. The Python code generates the HTML/JavaScript code for the map automatically.

### How the Map is Built (Step by Step)

1. **Compute average AQI per city:**
```python
city_data = df.groupby('City')['AQI'].mean().reset_index()
```

2. **Load city coordinates from a hardcoded dictionary:**
```python
coords = {
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    ...
}
```

3. **Assign a colour and radius based on AQI:**
```python
def get_color(aqi):
    if aqi <= 50:    return '#00e400', 'Good'       # Green
    elif aqi <= 100: return '#92d14f', 'Satisfactory'
    elif aqi <= 200: return '#ffff00', 'Moderate'    # Yellow
    elif aqi <= 300: return '#ff7e00', 'Poor'        # Orange
    elif aqi <= 400: return '#ff0000', 'Very Poor'   # Red
    else:            return '#8b0000', 'Hazardous'   # Dark red
```
The circle radius is also proportional to AQI: `radius = max(8, min(30, aqi / 15))`

4. **Write all of this into an HTML file** using Python string formatting (f-strings).

5. **Open the file in a browser** — you get a zoomable, clickable map of India where clicking a city shows its average AQI.

---

## 14. The Supporting Scripts

### `quick_test.py`
```python
test_input = pd.DataFrame([{
    "PM2.5": 150, "PM10": 200, "NO": 25, ...
}])
print("Expected output for these inputs: ~280-320 AQI (Poor range)")
```
A minimal standalone script to demonstrate how a single prediction would work. It builds the input DataFrame structure that you would feed the saved model. Note: the model is not actually saved to disk in this project (`pickle` is imported but never used), so this script is illustrative.

### `test_aqi_model.py`
```python
test_cases = [
    {"name": "Heavy Pollution - Delhi Winter", "PM2.5": 250, ..., "expected_category": "Very Poor"},
    {"name": "Clean Day - Hill Station", "PM2.5": 12, ..., "expected_category": "Good"},
    ...
]
```
Defines 5 named test scenarios with expected categories. This is used to validate that the model is producing sensible outputs — essentially manual unit tests for the ML model.

---

## 15. Full Picture — How Everything Connects

Here is the complete workflow from raw data to final outputs:

```
RAW DATA (city_day-Copy1.csv)
         │
         ▼
┌─────────────────────┐
│  Step 1: Load Data  │  → Read CSV into DataFrame (24850 rows × 16 cols)
└─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Step 2: Clean Data      │  → Drop rows with missing AQI
│                          │  → Fill missing pollutants with column mean
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Step 3: EDA             │  → 4 visualisation charts
│  (Explore + Understand)  │  → Histogram, Top Cities, Heatmap, Category Bar
└──────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Step 4: Prepare ML Data                     │
│  X = 12 pollutant columns                    │
│  y = AQI column                              │
│  Train/Test split: 80% train, 20% test       │
└──────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Step 5: Train 3 Models                                │
│  ├── Linear Regression    → R² = 0.8092               │
│  ├── Random Forest        → R² = 0.9105 (BEST) ✓      │
│  └── Gradient Boosting    → R² = 0.8952               │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Step 6: Evaluate + Compare                  │
│  → MAE, RMSE, R² for all 3 models            │
│  → Model comparison bar charts               │
│  → Actual vs Predicted scatter plot          │
│  → Residual analysis plots                   │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Step 7: Feature Importance                  │
│  → PM2.5 (49%), CO (37%) dominate            │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Step 8: Custom Test Scenarios               │
│  → 5 scenarios from Clean Rural to Hazardous │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Step 9: Early Warning System                │
│  → Function that takes pollutant inputs      │
│  → Returns predicted AQI + alert level       │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Step 10: Interactive HTML Map               │
│  → Leaflet.js map of Indian cities           │
│  → Color-coded by average AQI                │
└──────────────────────────────────────────────┘
```

---

## 16. Key Concepts Glossary

| Term | Plain English Definition |
|---|---|
| **Machine Learning** | Teaching a computer to find patterns in data and make predictions, without explicitly programming the rules |
| **Supervised Learning** | ML where you provide both inputs (X) and correct outputs (y) for the model to learn from |
| **Regression** | Predicting a continuous number (like AQI = 245), as opposed to classification (which category is it?) |
| **Feature** | An input variable used by the model (e.g., PM2.5, CO) |
| **Target** | The output variable the model is trying to predict (AQI) |
| **Training** | The process of showing the model data and letting it adjust its internal weights/rules |
| **Overfitting** | When a model memorizes training data and fails on new data — it's too specialized |
| **Underfitting** | When a model is too simple and fails to capture the real patterns |
| **Ensemble** | Combining multiple models to get a better result than any single model |
| **Bagging** | Building many independent models on random data subsets and averaging (Random Forest uses this) |
| **Boosting** | Building models sequentially where each one fixes the errors of the previous (Gradient Boosting uses this) |
| **Decision Tree** | A model that makes predictions by asking a series of yes/no questions |
| **MAE** | Mean Absolute Error — average prediction error in original units |
| **RMSE** | Root Mean Square Error — like MAE but penalizes large errors more |
| **R²** | How much of the variation in the target is explained by the model (0 to 1, higher is better) |
| **Feature Importance** | A score showing how much each input feature contributed to the model's accuracy |
| **Mean Imputation** | Replacing missing values with the average of that column |
| **Train-Test Split** | Dividing data into a portion for learning (train) and an unseen portion for honest evaluation (test) |
| **DataFrame** | A table in pandas (like an Excel sheet in Python memory) |
| **Random State** | A seed number that makes random operations reproducible |
| **µg/m³** | Micrograms per cubic meter — unit for measuring pollutant concentration in air |
| **Residual** | The difference between the actual value and the predicted value |
| **Correlation** | How strongly two variables move together (number from -1 to +1) |
| **Heteroscedasticity** | When prediction errors are not uniformly spread — they get bigger as the predicted value increases |
| **Hyperparameter** | A setting you choose before training (like `n_estimators=100`), unlike model weights which are learned |

---

*This document covers the entire project end-to-end — from raw CSV data to trained ML models, evaluation metrics, visualizations, the early warning system, and the interactive map.*
