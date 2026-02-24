# AQI Prediction Project — Viva Question Bank

> **Project:** Air Quality Index (AQI) Prediction using Machine Learning
> **Models Used:** Linear Regression, Random Forest, Gradient Boosting
> **Dataset:** Indian City Daily AQI (2015–2020) — 24,850 rows, 16 columns
> **Best Model:** Random Forest (R² = 0.9105, MAE = 20.60, RMSE = 40.49)

---

## Category 1: Conceptual Understanding

1. What is AQI and how is it calculated in the real world? Is your predicted AQI the same as the official AQI formula?
2. What is the difference between regression and classification? Why did you frame this as a regression problem instead of classifying directly into AQI buckets?
3. Explain the bias-variance tradeoff. Where does each of your three models (Linear Regression, Random Forest, Gradient Boosting) fall on that spectrum?
4. What does R² = 0.9105 actually mean in plain English? Can R² ever be negative, and when would that happen?
5. What is the difference between MAE and RMSE? Why is your RMSE (40.49) roughly double your MAE (20.60) for Random Forest — what does that gap tell you about your prediction errors?
6. What is an ensemble method? Which of your three models are ensemble methods, and what do they ensemble over?
7. Explain what overfitting is. How can you detect whether your Random Forest is overfitting on this dataset?
8. What is the Central Limit Theorem and does it matter for your model?
9. What does "variance explained" mean in the context of your R² scores?
10. Why are PM2.5 and PM10 treated as separate features when PM2.5 is a subset of PM10 by definition?

---

## Category 2: Code-Level Questions

11. In the data cleaning cell, you fill missing pollutant values with column means — why did you do this *after* dropping rows with missing AQI rather than before? Does the order matter?
12. You use `warnings.filterwarnings('ignore')` — what warnings are you suppressing, and is it good practice to do this in a production system?
13. In your `train_test_split`, you set `random_state=42`. What does this parameter do? What would happen to your results if you removed it?
14. You use `n_estimators=100` for both Random Forest and Gradient Boosting. Why 100? Did you try other values?
15. In the scatter plot for Actual vs Predicted, you draw the perfect prediction line as `plt.plot([0, 2000], [0, 2000])`. Why 2000? What if AQI values exceed 2000?
16. In your `pd.cut()` call for categorising predictions, you use `bins=[0, 50, 100, 200, 300, 400, 500, 9999]`. Why 9999 as the upper bound — is that justified?
17. In `quick_test.py`, you import `pickle` but never use it. Why is it there?
18. Your interactive map code hardcodes city coordinates in a dictionary. What happens for cities in the dataset that are not in your `coords` dictionary?
19. In the early warning system function, you set default values for less critical pollutants to dataset means. Is this statistically valid? What bias does this introduce?
20. You call `best_model.predict(test_cases)` on manually constructed DataFrames. How do you ensure the column order matches what the model was trained on?

---

## Category 3: Model / Algorithm Questions

21. Explain how a Random Forest works internally. How does it combine individual decision trees?
22. What is bagging vs boosting? Which of your models uses bagging and which uses boosting?
23. Why did Random Forest outperform Gradient Boosting here? Is that always expected?
24. Linear Regression got R² = 0.8092. Does that mean the relationship between pollutants and AQI is mostly linear? Why or why not?
25. What hyperparameters of Random Forest could you tune to potentially improve from R² = 0.9105? Name at least three.
26. What is the difference between `max_depth`, `min_samples_split`, and `min_samples_leaf` in a Random Forest?
27. Gradient Boosting uses a learning rate. What is it, and what would happen if you set it to 1.0 vs 0.01?
28. Did you perform any cross-validation? If not, how confident are you that your 80/20 split results are generalizable?
29. Could you use a neural network for this problem? Would it likely outperform Random Forest on this dataset size (24,850 rows)?
30. What loss function does `GradientBoostingRegressor` minimize by default? How does it differ from what `LinearRegression` minimizes?

---

## Category 4: Graphs & Result Interpretation

31. Your feature importance shows PM2.5 = 0.491 and CO = 0.368 — together they account for ~86% of importance. Does this concern you? What does it imply about the other 10 features?
32. In the correlation heatmap, which pollutant has the highest correlation with AQI? Is correlation the same as feature importance? Why might they differ?
33. Looking at your AQI distribution histogram, is the data normally distributed? How does the distribution shape affect model performance?
34. Your residual plot — are the residuals randomly scattered around zero? If you see a funnel shape (heteroscedasticity), what does that mean?
35. In the "Top 10 Most Polluted Cities" bar chart, which city has the highest average AQI? Does average AQI fairly represent a city's air quality?
36. Your box plots show outliers for some pollutants. How do outliers affect each of your three models differently?
37. In the Actual vs Predicted scatter plot, where does the model perform worst — at low AQI, mid AQI, or high AQI values? Why?
38. The AQI Bucket distribution bar chart — is the dataset balanced across categories? How does class imbalance affect a regression model?
39. Your regression line plots (AQI vs each pollutant) — for O3, is the relationship linear, or does it show a different pattern?
40. In your residual distribution histogram, is it centered at zero? Is the spread symmetric? What would a skewed residual distribution indicate?

---

## Category 5: Design Decisions (Why Did You Do This?)

41. Why did you choose these specific 12 pollutant features and not include `City` or `Date` as features?
42. Why did you use mean imputation for missing values instead of median, mode, or more advanced methods like KNN imputation?
43. Why didn't you normalize or standardize your features before training? Which of your models actually needs scaled features?
44. Why did you choose an 80/20 train-test split instead of 70/30 or 90/10?
45. Why didn't you use any feature selection or dimensionality reduction technique like PCA, given that some features contribute less than 1% importance?
46. Your early warning system uses the Random Forest model. Why not create a separate, simpler model specifically for real-time alerting?
47. Why did you build an interactive Leaflet map instead of a static matplotlib map? What additional value does it provide?
48. Why didn't you save and serialize your trained model using `pickle` or `joblib` for deployment?
49. You compute NOx as a feature alongside NO and NO2. Since NOx = NO + NO2, aren't you introducing multicollinearity? How does this affect each model?
50. Why didn't you include temporal features (month, season, year) even though your dataset has a `Date` column?

---

## Category 6: Limitations & Improvements

51. Your model is trained on Indian city data (2015–2020). Can it generalize to cities in Europe or China? Why or why not?
52. What are the main limitations of using mean imputation for missing data in this context?
53. The dataset has 24,850 rows. Is this enough to build a reliable model? What would you do with 100x more data?
54. Your model doesn't account for weather conditions (temperature, humidity, wind speed). How would adding these features likely improve performance?
55. How would you deploy this model as a real-time AQI prediction API? Describe the architecture.
56. What validation strategy would you use to ensure the model works well on future (unseen) time periods, not just a random 20% of historical data?
57. Your model predicts a continuous AQI number. How could you quantify the uncertainty of that prediction (e.g., predict AQI = 180 ± 25)?
58. If a new pollutant (e.g., microplastics in air) needed to be tracked, how would you modify your pipeline?
59. How would you handle concept drift — if pollution patterns change over the years, how would the model stay accurate?
60. What if the sensor that measures PM2.5 malfunctions and sends constant zeroes? How would your model behave?

---

## Category 7: Scenario-Based / What-if Questions

61. What if you remove PM2.5 entirely from the features? How much would R² drop, and why?
62. What if you train only on Delhi data and test on Bengaluru data? Would performance be better or worse, and why?
63. What if all pollutant values are set to zero? What AQI would your model predict — and should it be zero?
64. What if you change `test_size` from 0.2 to 0.5? How would MAE and R² likely change?
65. What happens if you increase `n_estimators` in Random Forest from 100 to 1000? Will R² always improve?
66. What if you accidentally include `AQI_Bucket` (the target category) as a feature? What would happen to your R² score?
67. What would happen if you trained on data from only one season (e.g., winter)? How would the model perform in summer?
68. If you replace mean imputation with dropping all rows that have any missing value, how many rows would you lose and how would it affect results?
69. What if your dataset had duplicate rows? How would that affect training and evaluation?
70. What if someone inputs negative pollutant values into your early warning system? Does your code handle that?

---

## Category 8: Advanced / Trap Questions

71. Your Random Forest R² is 0.91 on test data. What is the R² on training data? If it's 0.99, what does that tell you?
72. You split data randomly — but this is time-series data. Isn't a random split a form of data leakage? How does data from 2020 appearing in training help predict 2018 data in testing?
73. PM2.5 has importance 0.491, and AQI is formally *defined* using PM2.5 as a primary component. So isn't your model essentially learning a known formula? What's the ML contribution then?
74. If your model is just approximating the known AQI formula using ML, why not simply compute AQI using the official sub-index formula instead?
75. The `AQI_Bucket` column already exists in the dataset — meaning AQI was already computed. Why train an ML model to predict something that can be calculated deterministically?
76. You report R² = 0.9105. But did you check if this is statistically significant? How would you do that?
77. Can your model extrapolate — i.e., predict AQI for pollution levels far beyond what it has seen in training? What stops a tree-based model from doing this correctly?
78. Your feature importance sums to 1.0. Does this mean Toluene (0.007) is truly unimportant, or could it still have a causal effect on air quality that the model doesn't capture?
79. If two features are highly correlated (e.g., NO and NOx), how does Random Forest split importance between them? Does this make importance scores unreliable?
80. You claim this is an "Early Warning System." For a real early warning system, you need to predict *future* AQI. Your model predicts current AQI from current pollutant readings — isn't that nowcasting, not forecasting? What architectural change would make it a true early warning system?

---

## Quick Reference — Model Results

| Model              | MAE   | RMSE  | R²     |
|--------------------|-------|-------|--------|
| Linear Regression  | 31.20 | 59.11 | 0.8092 |
| Random Forest      | 20.60 | 40.49 | 0.9105 |
| Gradient Boosting  | 23.71 | 43.80 | 0.8952 |

## Quick Reference — Feature Importance (Random Forest)

| Feature  | Importance |
|----------|------------|
| PM2.5    | 0.491      |
| CO       | 0.368      |
| NO       | 0.037      |
| PM10     | 0.037      |
| O3       | 0.015      |
| NOx      | 0.012      |
| SO2      | 0.010      |
| NO2      | 0.008      |
| Toluene  | 0.007      |
| Xylene   | 0.006      |
| Benzene  | 0.005      |
| NH3      | 0.003      |

## Quick Reference — Test Case Predictions

| Scenario                      | PM2.5 | CO   | Predicted AQI | Category   |
|-------------------------------|-------|------|---------------|------------|
| Heavy Pollution (Delhi Winter)| 250   | 5.0  | ~393          | Very Poor  |
| Moderate Urban Day            | 80    | 1.5  | ~172          | Moderate   |
| Clean Day (Hill Station)      | 12    | 0.3  | ~36           | Good       |
| Extreme Industrial            | 400   | 10.0 | ~508          | Hazardous  |
| Near-Clean Rural              | 5     | 0.1  | ~31           | Good       |

---

*Total: 80 questions across 8 categories*
