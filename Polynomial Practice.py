
lm = LinearRegression()
lm.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 4)
polyx = polyreg.fit_transform(x)
polyreg.fit(polyx, y)

lm2 = LinearRegression()
lm2.fit(polyx, y)

lm.predict(6.5)

lm2.predict(polyreg.fit_transform(6.5))

---------------------------------------------------------------------------


lm = LinearRegression()
lm.fit(x, y)

polyreg = PolynomialFeatures(degree=4)
polyx = polyreg.fit_transform(x)
polyreg.fit(polyx, y)

lm2 = LineaRegression()
lm2.fit(x, y)

lm.predict(6.5)

lm2.predict(polyreg.fit_transform(6.5))



