
# Bike Rental Demand Prediction


## Different Phases

|**Phase**|**Name**|**Key Activity**|
|---|---|---|
|**Phase 1**|**Data Inspection**|Loading the UCI dataset, checking for nulls, and verifying data types.|
|**Phase 2**|**EDA (Exploration)**|Visualizing distributions and correlations to confirm if a GLM is the right tool.|
|**Phase 3**|**Feature Engineering**|Handling categorical variables (One-Hot Encoding) and dealing with multicollinearity (e.g., dropping `atemp`).|
|**Phase 4**|**Model Building**|Implementing the **Poisson** or **Negative Binomial** regression using `statsmodels`.|
|**Phase 5**|**Evaluation**|Using metrics like **Mean Poisson Deviance** or **RMSLE** to see how well the model predicts.|

## Phase 1: Data Inspection

- Dataset: [Bike Sharing Demand Dataset](https://www.kaggle.com/datasets/syedhaideralizaidi/bike-sharing-demand-dataset) 

```python
import pandas as pd

# Construct the full path to the file (adjust filename if yours is different)
file_path = '/content/drive/MyDrive/Bike Rental Project/Bike sharing.csv'

# Load the dataset
df = pd.read_csv(file_path)
```

```python
df.info()

df.describe()
```
- General idea about database columns.

```python
print(df.isnull().sum())
```
- Checking missing data at the columns. 

## Phase 2: Data Exploration
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['count'], kde=True, color='blue')
plt.title('Distribution of Bike Rental Counts')
plt.show()
```
- Distribution of count is skewed. 
  ![](https://github.com/Sayantan615/Bike-Rental-Predictions/blob/main/Bike%20Rental%20Demand%20using%20GLM%20right%20skewed%20possion%20like%20data%20distribution.png)
	- **X-axis (Horizontal): The Number of Bikes.** This represents the "Value" of the rental count (e.g., 0 bikes, 200 bikes, 500 bikes).
	- **Y-axis (Vertical): The Count of Instances (Frequency).** This represents **how many times** (how many hours in your dataset) that specific bike count occurred.
	- The tail stretches far to the right, it confirms that a **Poisson** or **Negative Binomial** GLM is better than standard linear regression.

```python
plt.figure(figsize=(8, 6))
correlation_matrix = df[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```
- **Multicollinearity Check**: temp and atemp features are highly correlated, we need to drop one.

```python
plt.figure(figsize=(8, 6))
correlation_matrix = df[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```
- Checking corelation between different features.

## Phase 3: Feature Engineering

```python
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract features (which are categorical)
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek

df_final.drop(['datetime'], axis=1, inplace=True)
```
- Dividing date time into multiple columns. And dropping the datetime column. 

```python
df['temp_workingday'] = df['temp'] * df['workingday']
df['temp_holiday'] = df['temp'] * df['holiday']
df['hum_weather'] = df['humidity'] * df['weather']
df['temp_squared'] = df['temp'] ** 2
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if x in [8, 17, 18] else 0)
df['log_windspeed'] = np.log1p(df['windspeed'])
df.head()
```
- Adding features for Augmented GLM
	- `temp_workingday` asks the question “Does ‘working day’ matter more when it’s hot?”

```python
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if x in [8, 17, 18] else 0)
```
- Adding an extra variable that gives extra weight to important hours. 

```python
df_final = pd.get_dummies(df, columns=['season', 'weather', 'hour', 'month', 'day_of_week'], drop_first=True)

df_final.head()
```
- **One-Hot Encoding** (or "Dummy Variables"). This creates a separate column for each categorical attribute with a $0$ or $1$.

```python
if 'atemp' in df.columns:
  df_final.drop('atemp', axis=1, inplace=True)

if 'windspeed' in df.columns:
  df_final.drop('windspeed', axis=1, inplace=True)

# Drop columns that are parts of the target or no longer needed
if 'casual' in df_final.columns:
  df_final.drop(['casual', 'registered'], axis=1, inplace=True)

if 'datetime' in df_final.columns:
  df_final.drop(['datetime'], axis=1, inplace=True)
```
- Discarded two more unnecessary columns. 
- `inplace=True` for permanent changes in the data frame. 

```python
from sklearn.model_selection import train_test_split

X = df_final.drop('count', axis=1) # Predictors
y = df_final['count']             # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Data is divided into train and test sets. 

```python
import statsmodels.api as sm

# Add constant to both Train and Test sets
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
```
- Adds a [[Linear Regression#Significance of Intercept ($ beta0$)|intercept]] ($\beta^0$). Column of 1s, model predicts a value for this, which is added to every prediction. 

```python
# Convert to clean numpy arrays to avoid 'Object/Boolean' conflicts
# We use float so the math is precise
y_clean = np.asarray(y_train).astype(float)
X_clean = np.asarray(X_train_const).astype(float)
X_test_clean = np.asarray(X_test_const).astype(float)
Y_test_clean = np.asarray(y_test).astype(float)
```
- Converting to clean numpy arrays to avoid ‘Object/Boolean’ conflicts, using float for precise calculations.

## Phase 4: Model Building

### Generalized Linear Models (GLMs)
Generalized Linear Models are used when underlying data (what you are trying to predict) cannot be defined by a Normal Distribution (bell curve).

>[!note] Interpret Coefficients
>In a GLM with a **Log Link**, the coefficients are "Multipliers." Because the math uses $e$ (Euler's number), a coefficient of $+0.1$ doesn't mean "add 0.1 bikes"; it means a **percentage change**.
>
|**If the Coef is...**|**The Math**|**The Interpretation**|
|---|---|---|
|**Positive (+0.05)**|$e^{0.05} \approx 1.05$|For every 1-unit increase, rentals go **UP** by 5%.|
|**Negative (-0.20)**|$e^{-0.20} \approx 0.81$|For every 1-unit increase, rentals go **DOWN** by 19% ($1 - 0.81$).|
|**Zero (0.00)**|$e^{0} = 1.00$|This feature has **NO effect** on bike rentals.|


### Model: GLM, Poisson distribution
```python
poisson_model = sm.GLM(y_clean, X_clean, family=sm.families.Poisson()).fit()

# Print the results with names
feature_names = ['const'] + list(X_train.columns)
print(poisson_model.summary(xname=feature_names))
```

#### Key Observations

- **Overdispersion: Deviance/DF Ratio is 41.8**
	- In standard Poisson model, we expect this ratio to be close to 1.0.
	- The model is underestimating the volatility/spikes in the data. The Mean != Variance.
	-  Moving on to a Negative Binomial GLM
- **Pseudo R-squared is 1.000**
	- In data science, a perfect score is almost always an error. It indicates Data Leakage–where the model has access to features that directly contain the answer it’s trying to predict
- **Log-Likelihood is -209,150**
	- This is probability score, while the number itself is hard to interpret alone, it serves as a baseline. 
	- If Negative Binomial predicts a less negative number then it is a better fit.
- **Convergence: No. of iterations 100**
	- The model timed out. The algorithm (IRLS) tried 100 times to find the best-fit line but couldn’t settle on one.
	- It should drop significantly usually to 10 or 20 iterations.


### Model: GLM, Neg. Binomial
```python
nb_model = sm.GLM(y_clean, X_clean, family=sm.families.NegativeBinomial(alpha=1.0)).fit()

print(nb_model.summary(xname=feature_names))
```

#### Key Observations
- **Overdispersion is solved: Deviance/DF ration is 0.36** (previous was 41.8)
	- The model is no longer overwhelmed by the spikes in bike rentals
- **Pseudo R-squared is 0.6345** (previous 1.000)
	- Good score for a real-world social behavior model. The model explains about 63% of the variation in bike rentals using only things like weather and time. 
- **Convergence is down to 16 iterations** (previous 100)
	- The algorithm found the answer very quickly, negative binomial is a better fit
- **Log-Likelihood scores are up: -50,112** (previous -209,150)
	- the closer to zero the better, model is more confident in its predictions
#### Plotting Residuals

```python
# Calculate residuals
predictions = nb_model.predict(sm.add_constant(X_test_clean))
residuals = y_test - predictions

# Plot Residuals
import matplotlib.pyplot as plt
plt.scatter(predictions, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Rentals')
plt.ylabel('Residuals (Error)')
plt.title('Residual Plot: Are the errors random?')
plt.show()
```

![[Bike Rental Demand using GLM residual plot 1.png#invert|center|400]] 

- The thickest part of the cloud is centered right on that red dashed line (0), which means on average the model is unbiased. But the Fan shape means heteroscedasticity. It means your model is very accurate when predicting low bike demands (like at 3 AM) but it becomes much more “uncertain” and has higher errors when predicting peak demand (5PM rush). 


### Model: GLM Log Gaussian

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Get predictions in Log Space
log_predictions = log_gaussian_model.predict(X_test_clean)

# 2. Transform predictions back to Real Bikes
final_predictions = np.expm1(log_predictions)

# 3. Calculate residuals on the SAME scale (Real Bikes)
# Ensure y_test is also a numpy array for consistent subtraction
residuals = np.asarray(y_test).flatten() - final_predictions

# 4. Plot
plt.figure(figsize=(10, 6))
plt.scatter(final_predictions, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Rentals (Real Bikes)')
plt.ylabel('Residuals (Error in Bikes)')
plt.title('Corrected Residual Plot (Original Scale)')
plt.show()
```

#### Key Observations

- Deviance/DF ration: $\frac{3321.4}{8656} \approx 0.3837$
	- Ratio $\approx 1$: The model fits well
	- Ratio $\gt 1$: Overdispersion
	- Ratio $\lt 1$: Underdispersion
	- In a Gaussian GLM, the interpretation is slightly different than in a Poisson model. 
		- It represent the variance in gaussian. 
		- Having a Deviance/DF ratio significantly **less than 1** in a Gaussian model is not "underdispersion" in a bad way. Instead, it indicates:
			- **High Precision:** Your residuals are small relative to the number of data points.
	    - **Stable Predictions:** Since $0.38$ is a relatively small variance for log-transformed counts, your model is consistently "tight" around the actual values.
- Pseudo R-squared: 0.9854
	- This indicates that your 51 features explain about 98.5% of the variance in the log-transformed rental counts.
- Convergence in down to 3 iterations.
	- Relationship between features and log-target is very stable and clean. 
- Log-Likelihood has increased -8159.5

#### Plotting Residuals

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Get predictions in Log Space
log_predictions = log_gaussian_model.predict(X_test_clean)

# 2. Transform predictions back to Real Bikes
final_predictions = np.expm1(log_predictions)

# 3. Calculate residuals on the SAME scale (Real Bikes)
# Ensure y_test is also a numpy array for consistent subtraction
residuals = np.asarray(y_test).flatten() - final_predictions

predictions = final_predictions

# 4. Plot
plt.figure(figsize=(10, 6))
plt.scatter(final_predictions, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Rentals (Real Bikes)')
plt.ylabel('Residuals (Error in Bikes)')
plt.title('Corrected Residual Plot (Original Scale)')
plt.show()
```




### [[Decision Trees]]

### Model: Decision Tree (Single Tree)

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# 1. Initialize the tree
# We set max_depth to keep it readable and prevent over-memorizing (overfitting)
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)

# 2. Fit the model using the Log-Target (since it worked best for you)
dt_model.fit(X_train, np.log1p(y_train))

# 3. Visualize the logic
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X_train.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Logic for Bike Rentals")
plt.show()
```
- `plot_tree`: generates a visualization flow-chart like diagram
- `max_depth`: curiosity limit,
	- low depth: the model is too simple (underfitting).
	- high depth: the model has memorized data instead of finding key splits, basically it will overfit. 
- `random_state`: It is like a seed, it ensures that the random choices made during the training process are the same every time you run the code.  

### Model: Random Forests

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Initialize the Forest
# n_estimators is the number of trees in your council
rf_model = RandomForestRegressor(n_estimators=100, 
                                 max_depth=10, 
                                 random_state=42, 
                                 n_jobs=-1) # Uses all your CPU cores for speed

# 2. Fit the model
rf_model.fit(X_train, np.log1p(y_train))

# 3. Predict and Inverse the Log
rf_predictions = rf_model.predict(X_test)

mae_score, rmsle_score = evaluate_model(y_test, rf_predictions, is_log_transformed=True)

print(f"Mean Absolute Error: {mae_score:.2f} bikes")
print(f"RMSLE: {rmsle_score:.4f}")
```

#### Hyperparameter Tuining
```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# 1. Define the "Search Space"
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None] # 'None' uses all features
}

# 2. Initialize the Search
rf = RandomForestRegressor(random_state=42)

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,     # How many random combinations to try
    cv=3,          # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1,     # Use all CPU cores
    scoring='neg_mean_absolute_error'
)

# 3. Run the search (Remember to use the Log of your target)
rf_random.fit(X_train, np.log1p(y_train))

rf_predictions = rf_random.predict(X_test)

mae_score, rmsle_score = evaluate_model(y_test, rf_predictions, is_log_transformed=True)

print(f"Mean Absolute Error: {mae_score:.2f} bikes")
print(f"RMSLE: {rmsle_score:.4f}")
```

### Model: Xgboost

```python
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
import numpy as np

# 1. Initialize the XGBoost Regressor
# We use 'reg:squarederror' for regression tasks
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,   # Number of trees to build
    learning_rate=0.05,  # How much each tree "fixes" the previous error
    max_depth=6,         # How "tall" the trees can grow
    subsample=0.8,       # Use 80% of data for each tree to prevent overfitting
    random_state=42
)

# 2. Fit the model
# Note: XGBoost handles raw counts well, but since you liked the
# Log-Gaussian results, training on the Log-Target is often even better!
y_train_log = np.log1p(y_train)
xgb_model.fit(X_train, y_train_log)

# 3. Predict and Inverse Transform
log_preds = xgb_model.predict(X_test)
final_preds = np.expm1(log_preds)
final_preds = np.maximum(final_preds, 0) # Ensure no negative bikes

# 4. Evaluate
mae = mean_absolute_error(y_test, final_preds)
rmsle = np.sqrt(mean_squared_log_error(y_test, final_preds))

print(f"XGBoost MAE: {mae:.2f}")
print(f"XGBoost RMSLE: {rmsle:.4f}")
```



## Phase 5: Evaluation and Interpretation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
import numpy as np

mae = mean_absolute_error(y_test_clean, predictions)
rmsle = np.sqrt(mean_squared_log_error(y_test_clean, predictions))

print(f"Mean Absolute Error: {mae:.2f} bikes")
print(f"RMSLE: {rmsle:.4f}")
```

### Model Comparison

| **Model**                 | **MAE (Bikes)** | **RMSLE**  | **Core Observation**                                                                   |
| ------------------------- | --------------- | ---------- | -------------------------------------------------------------------------------------- |
| **Poisson**               | 68.92           | 0.6647     | Baseline statistical model; struggles with variance in high-count data.                |
| **Neg. Binomial**         | 72.10           | 0.6683     | Better statistical theory (handled overdispersion) but slightly higher raw error.      |
| **Log-Gaussian**          | 70.35           | 0.6234     | Proved that log-transforming the target stabilizes percentage-based errors.            |
| **Decision Tree**         | 94.30           | 0.7866     | **Worst Performer.** Too brittle and prone to overfitting/underfitting simultaneously. |
| **Random Forest (Base)**  | 73.86           | 0.6271     | The power of "Bagging"; instantly stabilized the Decision Tree's mess.                 |
| **Random Forest (Tuned)** | 49.29           | 0.4660     | High performance through pruning; currently the most robust "stable" model.            |
| **XGBoost**               | **46.28**       | **0.4161** | Using sequential "Boosting" to fix remaining errors and minimize the log-loss.         |

- **Heteroscedasticity**: "Changing Volatility." Imagine you are predicting bike rentals. At 3:00 AM, you might predict 5 bikes and be off by 2. But at 5:00 PM, you might predict 500 bikes and be off by 100.
	- As the **count** gets bigger, the **error** (the "fan-out") also gets bigger. 
	- Standard models hate this because they assume the "noise" is the same everywhere.

- **Overdispersion**: In a perfect "Poisson" world, the average number of rentals should equal the "spread" (variance). If the average is 10, most days should be around 10.
	- **Overdispersion** happens when the average is 10, but some days are 0 and others are 50. The data is "spreading out" much more than the basic math expects.
	- This is why your **Negative Binomial** model was useful—it's designed to handle "wilder" data.

| **If you see...**            | **It means...**                                                                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **High MAE / High RMSLE**    | The model is guessing poorly everywhere.                                                                                                    |
| **Low MAE / High RMSLE**     | The model is accurate on large counts but making massive percentage errors on low counts (e.g., predicting 10 bikes when the actual was 1). |
| **Moderate MAE / Low RMSLE** | The model is very stable and reliable across all demand levels.                                                                             |
|                              |                                                                                                                                             |

- [[Mean Absolute Error (MAE)]]
	* **Definition:** Represents the average raw distance between your prediction and the actual count.
	* **In Context:** At your best score of **70.35**, the model is off by approximately 70 bikes.
	* **Relative Impact:** An error of 70 is more significant for low-volume hours (e.g., a 35% error for 200 rentals) but becomes much more acceptable during peak hours (e.g., a 9% error for 800 rentals).

- [[Mean Squared Error (MSE)|Root Mean Squared Logarithmic Error (RMSLE)]]: We should always strive to push this number **down toward zero**.
	* **Why it matters:** It recognizes that predicting 20 bikes for a 10-bike demand is a "disaster" (2x error), while predicting 510 for a 500-bike demand is a "huge success."
	* **Business Logic:** It is often harsher on **under-prediction** (predicting 40 for 100). In bike sharing, it is safer to have 10 extra bikes available (over-prediction) than to turn away 10 customers because you ran out of stock (under-prediction).

## [[Stacking (Stacked Generalization)]]
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV # A simple, stable linear model

# 1. Define your "Base Experts"
# Use the tuned parameters you found earlier
base_models = [
    ('rf', best_rf), # Your tuned Random Forest
    ('xgb', xgb_model) # Your tuned XGBoost
]

# 2. Initialize the Stacker
stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=RidgeCV(), # The Meta-Learner
    cv=5, # Internal validation to prevent overfitting
    n_jobs=-1
)

# 3. Fit (Again, use the Log-Target!)
stack_model.fit(X_train, np.log1p(y_train))

# 4. Predict
log_preds = stack_model.predict(X_test)
final_preds = np.expm1(log_preds)
```

| **Model**                 | **MAE (Bikes)** | **RMSLE**  | **Core Observation**                                                                                                   |
| ------------------------- | --------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Stacking (RF + XGB)**   | **45.97**       | **0.4150** | **The Final Winner.** The meta-model effectively balanced the stability of the Forest with the precision of the Boost. |

## Conclusion
I started with the most obvious choice for solving a regression problem, linear regression, but as it turned out that when the data is not normally distributed, then normal linear regression will fail, I tried GLM models, Possion, Negative binomial, and Log gaussion. 

While this gave good performance, as it turned out Decision Trees are pretty good for this kind of data. So I tried with single DT, Random Forests and Finally what gave the best results are Gradient Boosted Decision Trees (GBDTs).

Finally most solid results came from using staking of random forests and XGboost. 
