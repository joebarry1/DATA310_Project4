### Q1 
The goal of regularization is to prevent overfitting through weighting of some regularization term, $\lambda$. In this case, answer B does not concern modelling data, and answer C concerns minimizing the MSE, which is not necessary the goal of regularization.

### Q2
Regularization is especially useful in high feature, low observation data, as this can be especially prone to overfitting.

### Q3
The L1 norm and L2 norm can be likened to the "taxicab" distance between points; of the distance if one only travelled directly along an axis. In the case of a taxi, this would be like only travelling along the streets or avenues of New York City. On the other had, the L2 norm simply takes the shortest path; "as the crow flies", in other words. Thus, even in the best case, these would be equal. In no case would the L1 norm yield a shorter distance than the L2 norm.

### Q4
The other answers do not make great mathematical sense, and every regularization technique we have seen in class has been of this form: minimize the MSE plus some penalty function. For instance, the elastic net regularization formula we looked at in class was given as: $$\text{minimize} \frac{1}{n}\sum_{i=1}^{n}\left(\text{Residual}_i\right)^2 + \alpha\left( \lambda\cdot \sum\limits_{j=1}^{p}|\beta_j| + (1-\lambda)\cdot\sum\limits_{j=1}^{p}\beta_j^2\right)$$

### Q5
Lasso Regularization can reduce unnecessary variables to have no effect on the final model if necessary, at sufficiently high levels of $\alpha$.

### Q6

First the necessary imports:


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
```

Now we load the data, separate features and targets, and scale X.


```python
data = load_boston()
X = data.data
X_names = data.feature_names
y = data.target
```


```python
scale = StandardScaler()
scale.fit_transform(X)
```




    array([[-0.41978194,  0.28482986, -1.2879095 , ..., -1.45900038,
             0.44105193, -1.0755623 ],
           [-0.41733926, -0.48772236, -0.59338101, ..., -0.30309415,
             0.44105193, -0.49243937],
           [-0.41734159, -0.48772236, -0.59338101, ..., -0.30309415,
             0.39642699, -1.2087274 ],
           ...,
           [-0.41344658, -0.48772236,  0.11573841, ...,  1.17646583,
             0.44105193, -0.98304761],
           [-0.40776407, -0.48772236,  0.11573841, ...,  1.17646583,
             0.4032249 , -0.86530163],
           [-0.41500016, -0.48772236,  0.11573841, ...,  1.17646583,
             0.44105193, -0.66905833]])



Now we will fit the model and calculate RMSE.


```python
lin_reg = LinearRegression()
lin_reg.fit(X,y)
y_pred = lin_reg.predict(X)
rmse = np.sqrt(mean_squared_error(y,y_pred))
print(rmse)
```

    4.679191295697281


There's our answer: $4.6792$.

### Q7
Once again, let's import anything we need which isn't yet imported.


```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
```

Now we can initialize the lasso regression and run a loop to fit the data in each split as done in class.


```python
las_reg = Lasso(alpha = .03)
kf = KFold(n_splits=10, random_state=1234,shuffle=True)
MSE_test = []
for idxTrain,idxTest in kf.split(X):
    Xtrain = X[idxTrain,:]
    Xtest = X[idxTest,:]
    ytrain = y[idxTrain]
    ytest = y[idxTest]
    las_reg.fit(Xtrain,ytrain)
    y_test_pred = las_reg.predict(Xtest)
    MSE_test.append(mean_squared_error(y_test_pred,ytest))
print(np.mean(MSE_test))
```

    24.742480813910557


Here is our answer: $24.7425$.

### Q8


```python
from sklearn.linear_model import ElasticNet
```

We can reuse a lot of the code from the previous question, just changing out the lasso regularization for elastic net.


```python
net_reg = ElasticNet(alpha = .05, l1_ratio = .9)
kf = KFold(n_splits=10, random_state=1234,shuffle=True)
MSE_test = []
for idxTrain,idxTest in kf.split(X):
    Xtrain = X[idxTrain,:]
    Xtest = X[idxTest,:]
    ytrain = y[idxTrain]
    ytest = y[idxTest]
    net_reg.fit(Xtrain,ytrain)
    y_test_pred = net_reg.predict(Xtest)
    MSE_test.append(mean_squared_error(y_test_pred,ytest))
print(np.mean(MSE_test))
```

    25.337273122846046


Answer: $25.3373$.

### Q9
This questions will follow similarly to Q6, except we will use the PolynomialFeatures function in sklearn.


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
polynomial_features= PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X)
```


```python
model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
print(rmse)
```

    2.5330156817505762


Our answer is $2.5530$.

### Q10

We will start with the model, which is similar to the last except for the added regularization.


```python
from sklearn.linear_model import Ridge
rid_reg = Ridge(alpha = .1)
```

We can reuse X_poly from Q9 as well.


```python
rid_reg.fit(X_poly, y)
```




    Ridge(alpha=0.1)



And now isolate the residuals.


```python
y_pred = rid_reg.predict(X_poly)
```

Now, we have to import some packages to produce the plot.


```python
import pylab 
import scipy.stats as stats
```


```python
stats.probplot(y_pred, dist="norm", plot=pylab)
pylab.show()
```


    
![png](output_38_0.png)
    


It is subjective, but I would say this is close enough to be a reasonable normal distribution.


```python

```
