from utils import load_data, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

df = load_data()


models = {
    'LinearRegression': (LinearRegression(), {}),
    'Ridge': (Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}),
    'Lasso': (Lasso(), {'alpha': [0.01, 0.1, 1, 10, 100]})
}


dev_sizes = [0.2, 0.3, 0.4]


for name, (model, param_grid) in models.items():
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

        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
            grid.fit(X_train, y_train)
            current_model = grid.best_estimator_
            preds = current_model.predict(X_test)
            mse, r2 = evaluate_model(y_test, preds)

            print(f"GridSearchCV Best Params: {grid.best_params_}")
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


    print(f"\n-------Best configuration for {name}:")
    print(f"Best Dev Size: {best_dev_size}")
    if param_grid:
        print(f"Best Hyperparameters: {best_model.get_params()}")
    print(f"Best R² Score: {best_score:.4f}, Best MSE: {best_mse:.4f}")
