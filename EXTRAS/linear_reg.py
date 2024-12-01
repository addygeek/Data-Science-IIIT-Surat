from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [3, 6, 9, 12]
model = LinearRegression().fit(X, y)
print("Coefficient:", model.coef_, "Intercept:", model.intercept_)
