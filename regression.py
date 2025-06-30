from utils import load_data, split_data, train_model, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

for name, model in models.items():
    m = train_model(model, X_train, y_train)
    preds = m.predict(X_test)
    mse, r2 = evaluate_model(y_test, preds)
    print(f"{name} - MSE: {mse:.2f}, R²: {r2:.2f}")
