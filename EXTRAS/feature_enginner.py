from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [3, 6, 9, 12]
model = LinearRegression().fit(X, y)
print("Coefficient:", model.coef_, "Intercept:", model.intercept_)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X)
print("MSE:", mean_squared_error(y, y_pred))
print("R-squared:", r2_score(y, y_pred))
