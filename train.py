import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from preprocessing_pipeline import preprocess
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. Preprocess Data ---
X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess()

# --- 2. Define Model and Hyperparameter Grid ---
# LightGBM often outperforms RandomForest on tabular data.
param_grid = {
    'n_estimators': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40, 50],  # Key tuning parameter for LGBM
    'max_depth': [-1, 10, 15],
    'reg_alpha': [0, 0.1, 0.5],      # L1 regularization
    'reg_lambda': [0, 0.1, 0.5]     # L2 regularization
}

lgbm = lgb.LGBMRegressor(random_state=42)

# --- 3. Run Randomized Search for Hyperparameter Tuning ---
search_lgbm = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_grid,
    n_iter=50,
    scoring='r2',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search_lgbm.fit(X_train, y_train)

print("Best Parameters:", search_lgbm.best_params_)
print(f"Best R2 Score on CV: {search_lgbm.best_score_:.4f}")

best_lgbm = search_lgbm.best_estimator_
y_pred = best_lgbm.predict(X_test)

# --- Evaluation ---
# Evaluate on the log-transformed scale (useful for model optimization)
mse_log = mean_squared_error(y_test, y_pred)
r2_log = r2_score(y_test, y_pred)

print(f"\n--- Test Set Performance (Log Scale) ---")
print(f"MSE (log): {mse_log:.4f}")
print(f"R2 Score (log): {r2_log:.4f}")

# Evaluate on the original dollar scale for better interpretation
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)
mae_real = mean_absolute_error(y_test_real, y_pred_real)

print(f"\n--- Test Set Performance (Original Scale) ---")
print(f"Mean Absolute Error: ${mae_real:,.2f}")

# Save the model and transformers
joblib.dump(best_lgbm, 'lightgbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\nModel, scaler, and label encoder saved successfully.")
