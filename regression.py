from utils import load_data, split_data, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

param_grids = {
    'LinearRegression': {},
    'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
    'Lasso': {'alpha': [0.01, 0.1, 1, 10, 100]}
}

for name, model in models.items():
    if param_grids[name]:
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best {name} params: {grid.best_params_}")
    else:
        best_model = model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    mse, r2 = evaluate_model(y_test, preds)
    print(f"{name} with tuning - MSE: {mse:.2f}, R²: {r2:.2f}")
