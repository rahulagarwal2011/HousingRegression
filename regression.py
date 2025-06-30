from utils import load_data, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np


df = load_data()


results = []


param_grids = {
    'LinearRegression': {
        'fit_intercept': [True, False],
        'positive': [True, False],  
        'n_jobs': [None, -1]
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr']
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random']
    }
}


models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}


dev_sizes = [0.2, 0.3, 0.4]


for name, model in models.items():
    print(f"\n========== Testing Model: {name} ==========")
    best_score = -np.inf
    best_dev_size = None
    best_model = None

    
    for dev_size in dev_sizes:
        print(f"\n--- Testing dev_size: {dev_size} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('MEDV', axis=1),
            df['MEDV'],
            test_size=dev_size,
            random_state=42
        )

        grid_params = param_grids.get(name, {})
        if grid_params:
            grid = GridSearchCV(model, grid_params, cv=5, scoring='r2')
            grid.fit(X_train, y_train)
            current_model = grid.best_estimator_
            preds = current_model.predict(X_test)
            mse, r2 = evaluate_model(y_test, preds)

            print(f"GridSearchCV tested parameters: {grid_params}")
            print(f"Best Params Found: {grid.best_params_}")
            print(f"R² Score: {r2:.4f}, MSE: {mse:.4f}")

        else:
            current_model = model.fit(X_train, y_train)
            preds = current_model.predict(X_test)
            mse, r2 = evaluate_model(y_test, preds)

            print(f"No hyperparameters to tune for {name}.")
            print(f"R² Score: {r2:.4f}, MSE: {mse:.4f}")

        
        if r2 > best_score:
            best_score = r2
            best_dev_size = dev_size
            best_model = current_model
            best_mse = mse

   
    print(f"\n---------Best configuration for {name}:")
    print(f"Best Dev Size: {best_dev_size}")
    if param_grids.get(name, {}):
        print(f"Best Hyperparameters: {best_model.get_params()}")
    print(f"Best R² Score: {best_score:.4f}, Best MSE: {best_mse:.4f}")

   
    results.append({
        'Model': name,
        'Best Dev Size': best_dev_size,
        'Best Hyperparameters': best_model.get_params() if param_grids.get(name, {}) else '-',
        'MSE': round(best_mse, 2),
        'R2': round(best_score, 2)
    })


print("\n=== 📊 Performance Comparison Summary ===")
for res in results:
    print(f"{res['Model']:20} | Dev Size: {res['Best Dev Size']} | MSE: {res['MSE']} | R2: {res['R2']} \nBest Params: {res['Best Hyperparameters']}\n")
