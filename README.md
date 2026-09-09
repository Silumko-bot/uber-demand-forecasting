# Uber Pickup Demand Forecasting

## Project Overview

This project develops a one-day-ahead forecasting model for daily Uber pickup
demand using historical demand patterns and calendar-based features.

The aim is to understand whether historical pickup behaviour can be used to
improve short-term demand forecasts and support operational planning.

---

## 1. Uber Pickup Demand Forecasting

The project focuses on forecasting the next day's Uber pickup demand.

The analysis compares simple forecasting baselines with machine-learning models
to determine whether engineered historical demand features provide additional
predictive value.

---

## 2. Load Packages

The project uses:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Jupyter Notebook

---

## 3. Load Data

The dataset contains daily Uber pickup observations covering approximately six
months.

The main variables are:

- `date`
- `pickups`

These provide the foundation for the time-series analysis and forecasting
features developed later in the project.

---

## 4. Data Cleaning

The dataset was checked for:

- Missing values
- Duplicate records
- Invalid dates
- Missing dates in the time series
- Correct data types

The cleaned dataset was then sorted chronologically before analysis and
modelling.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Daily Demand

Daily Uber pickup demand fluctuates noticeably across the study period, while
the overall level of demand increases over time.

Repeated short-term rises and falls suggest that both recent demand behaviour
and longer-term trend may be useful for forecasting.

### 5.2 Calendar Features for EDA

Calendar-based variables were created to explore how demand changes across time.

### 5.3 Demand by Day of Week

Pickup demand varies noticeably throughout the week.

Thursday and Friday generally experience stronger demand, while Sunday records
lower average pickup activity.

This suggests that weekly seasonality could provide useful forecasting
information.

### 5.4 Weekday vs Weekend

Average pickup demand differs between weekdays and weekends, reinforcing the
importance of weekly demand patterns.

### 5.5 Monthly Demand

Average daily pickup volume increased considerably across the available months.

This upward trend is important because the final test period contains demand
levels that are higher than much of the earlier training data.

### 5.7 Weekly Seasonality

The time series shows a recurring weekly demand pattern alongside a broader
upward trend.

These findings support the use of lag, rolling, day-of-week and trend features
during modelling.

---

## 6. Feature Engineering

Historical pickup demand was transformed into features designed to capture
recent demand, weekly seasonality and longer-term movement.

Features included:

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`
- 7-day rolling mean
- 14-day rolling mean
- 7-day rolling standard deviation
- Weekend indicator
- Cyclical day-of-week features
- Time trend

Rolling features were shifted before calculation so that the current day's
target value was not used to predict itself.

---

## 7. Train/Test Split

Because this is a time-series forecasting problem, the dataset was not randomly
shuffled.

Earlier observations were used for model training, while the final period was
reserved as unseen test data.

This preserves the chronological structure of the problem and better reflects
real-world forecasting.

---

## 8. Evaluation Function

Models were evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **R²** — Coefficient of Determination

RMSE was used as the main model-comparison metric.

---

## 9. Naive Baseline

The naive baseline assumes that tomorrow's demand will be the same as today's.

This provides a simple benchmark that machine-learning models should ideally
outperform.

---

## 10. Seasonal Naive Baseline

The seasonal-naive forecast assumes that demand will be similar to the same day
one week earlier.

This provides a second benchmark that accounts for weekly seasonality.

---

## 11. Machine Learning Models

The project evaluates:

- Ridge Regression
- Random Forest
- Gradient Boosting

These models provide a mix of regularized linear and nonlinear forecasting
approaches.

---

## 12. Time-Series Cross-Validation

`TimeSeriesSplit` was used instead of random cross-validation.

This ensures that models are always trained on earlier observations and
validated on later observations, preventing future information from leaking
into model evaluation.

---

## 13. Visual CV Performance

Cross-validation performance was visualised to compare the stability and
forecasting error of the candidate models across time-series folds.

---

## 14. Tune Random Forest

Random Forest hyperparameters were tuned using time-series cross-validation.

Although tuning improved its cross-validation configuration, the tuned model
still struggled to generalise to the final unseen test period.

---

## 15. Fitting Models

The final candidate models were trained using the chronological training data.

---

## 16. Evaluating Models

Model predictions were evaluated against the unseen holdout period using MAE,
RMSE, MAPE and R².

The final results were:

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| **Ridge Regression** | **3,489.15** | **4,688.26** | **9.71%** | **0.24** |
| Naive Baseline | 3,992.67 | 5,429.02 | 12.42% | -0.02 |
| Seasonal Naive | 4,142.83 | 5,740.73 | 11.86% | -0.14 |
| Tuned Random Forest | 6,104.24 | 7,437.22 | 16.64% | -0.92 |
| Random Forest | 6,168.94 | 7,512.12 | 16.80% | -0.95 |
| Gradient Boosting | 6,550.65 | 8,036.23 | 17.97% | -1.24 |

---

## 17. Model Comparison Chart

The model-comparison chart shows that Ridge Regression produced the lowest
forecasting error on the final test period.

The more complex tree-based models performed substantially worse on the unseen
data.

---

## 18. Automatically Identify Best Holdout Performance

The evaluation results were sorted automatically to identify the model with the
lowest holdout error.

Ridge Regression was identified as the strongest-performing model.

---

## 19. Final Model Selection

**Ridge Regression** was selected as the final forecasting model.

It achieved:

- **MAE:** 3,489
- **RMSE:** 4,688
- **MAPE:** 9.71%
- **R²:** 0.24

The model also outperformed both simple forecasting benchmarks.

---

## 20. Calculating Improvement Over Baseline

Ridge Regression reduced RMSE by approximately **13.6% compared with the
previous-day naive baseline**.

This demonstrates that combining historical demand, seasonal patterns and trend
information added predictive value beyond simply using the previous day's
pickup volume.

---

## 21. Final Forecast DataFrame

A final forecast dataset was created containing:

- Date
- Actual pickup demand
- Predicted pickup demand

This dataset was used for the final prediction and error analysis.

---

## 22. Actual vs Predicted Demand

Ridge Regression follows the overall movement in demand reasonably well, but
its forecasts are smoother than the actual pickup values.

The largest prediction gaps occur during sudden high-demand periods.

---

## 23. Forecast Error Analysis

Forecast errors were calculated to understand where and how the final model
made mistakes.

Positive errors indicate underprediction, while negative errors indicate
overprediction.

---

## 24. Largest Forecast Errors

The largest forecasting errors were examined to identify the days where the
model struggled most.

The biggest errors were generally associated with sudden increases in demand
that were difficult to predict from historical patterns alone.

---

## 25. Forecast Error Over Time

Forecast errors fluctuate around zero, showing periods of both underprediction
and overprediction.

Several of the largest positive errors occur during high-demand periods,
indicating that Ridge Regression sometimes underestimates sharp increases in
pickup volume.

---

## 26. Error by Day of Week

Forecast accuracy varies across the week.

Wednesday was one of the most accurately predicted days, while **Saturday had
the largest average forecasting error**.

The model also showed higher errors toward Friday and Saturday, suggesting that
high-demand weekend periods are more difficult to forecast accurately.

---

## 28. Ridge Regression Coefficients

Previous-day demand (`lag_1`) was the strongest predictor in the final Ridge
Regression model.

Weekly seasonal features and the positive time-trend variable were also
important.

This suggests that short-term demand persistence, weekly seasonality and the
broader increase in demand all contributed to the final forecasts.

---

## 29. Business Insights

- **Demand varies throughout the week.** Thursday and Friday generally
  experience stronger pickup demand.

- **Recent demand is particularly important.** Previous-day pickup volume was
  the strongest predictor in the Ridge model.

- **Ridge Regression improved forecasting accuracy.** It reduced RMSE by
  approximately **13.6%** compared with the naive baseline.

- **High-demand periods remain challenging.** Friday and Saturday produced
  larger forecasting errors, particularly during sudden demand increases.

---

## 30. Limitations

- The dataset contains only approximately six months of daily observations.
- Longer-term annual seasonality cannot be reliably analysed.
- Daily totals hide within-day demand patterns.
- Weather, holidays, events and traffic information were unavailable.
- Pickup volume may not represent all potential customer demand.
- The forecasting setup assumes that previous actual demand is available before
  producing the next day's prediction.

---

## 31. Recommendations and Future Improvements

Future forecasting systems could be improved by incorporating:

- More historical observations
- Hourly pickup demand
- Weather information
- Public holidays
- Major events
- Geographic pickup zones
- Traffic conditions
- Regular walk-forward retraining

These factors may help explain the sudden demand spikes that historical pickup
patterns alone cannot capture.

---

## 32. Conclusion

This project developed a one-day-ahead forecasting model for daily Uber pickup
demand using historical demand and calendar-based information.

**Ridge Regression achieved the strongest final performance**, with:

- **MAE:** approximately 3,489 pickups
- **RMSE:** approximately 4,688 pickups
- **MAPE:** approximately 9.71%
- **R²:** approximately 0.24

Most importantly, Ridge Regression reduced RMSE by approximately **13.6%**
compared with the naive forecasting benchmark.

The project demonstrates that careful feature engineering, chronological
validation and comparison with strong baselines can be more valuable than model
complexity alone.

---

## Repository Structure

```text
Uber-Demand-Forecasting/
│
├── README.md
├── Uber_Demand_Forecasting.ipynb
└── data/
    └── pickups.csv
