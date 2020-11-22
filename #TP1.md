# TP1 - PROJECT : Linear Regression from Scratch

---

### Consulting the Combined Power Plant Dataset avaible on UCI machine learning repository

#### You have been contacted build a machine learning model to predict power output of a power plant given a set of readings from various sensors in a gas-fired power generation plant.

---

### The first step for building a machine learning application is understanding the business. 
#### In our case, we want to predict the power output of a gas-fired power generation plant based on the readings from various sensors. 
#### Peaking power plants are plants that supply power only occasionally. i.e. when there is high demand (peak demand) for electricity. Hence, the generated electricity is much more expensive than base load power plants. So it is important to understand and predict the power output to manage the plant connection to the power grid.

---

## Dataset Variables

### Features consist of hourly average ambient variables
####  Temperature (AT) in the range 1.81°C and 37.11°C
####  Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
####  Ambient Pressure (AP) in the range 992.89-1033.30 milibar
####  Relative Humidity (RH) in the range 25.56% to 100.16%
####  Net hourly electrical energy output (PE) 420.26-495.76 MW. The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization.

### This is a supervised machine learning problem since we have a labelled data set.


---

## Part1 - Linear Regression from Scratch

### Installation of libraries
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Obtention of the excel file
#### We have to install xlrd using pip before
```CMD
pip install xlrd
```
#### Then we can do this command line
```Python
df = pd.read_excel(open("D:\efrei\cours\s7\machine learning\lab1\CCPP\Folds5x2_pp.xlsx", 'rb'), index_col = 0, sheet_name = 'Sheet1')
df.head()
```

##### In this dataset, we have 5 features (so n = 5) and 9569 observations (so m = 9569)


#### We have to implement the vector of weights w
##### Because we do not have weights yet we instantiate it at 1
```Python
def weight(df):
    w = [1]
    for i in range (0, len(df.axes[1])):
        w.append(1)
    
    np.array(w)
    
    return np.transpose([w])


w = weight(df)
```

#### We have to implement the vector of observation x
```Python
def normalizeObservation(df, line):
    x = np.array(1)
    
    for i in df.loc[line]:
        x = np.append(x, i)
    
    return np.transpose([x])


x = normalizeObservation(df, 0)
```

#### We have to implement the matrix of observations X
```Python
def normalizeMatrix(df):    
    X = np.zeros( (len(df.axes[0]), len(df.axes[1])) )
    
    # for i < nb_columns
    # len(df.axes[1]) = 5
    for i in range (0, len(df.axes[1])):
                
        # for j < nb_lines
        # len(df.axes[0]) = 9568
        for j in range (0, len(df.axes[0])):
                        
            x = normalizeObservation(df, j)
            x = np.delete(x, i + 1)
            
            for k in range (0, len(x)):
                X[j][k] = x[k]
                
    return X



X = normalizeMatrix(df)
```

#### We have to implement the answer vector y
```Python
def normalizeFeature(df, col):
    y = []
    for i in df[col]:
        y.append(i)

    np.array(y)

    return np.transpose([y])


y = normalizeFeature(df, 'AT')
```

#### We have to implement the cost function J
```Python
def costFunction(df, multiplicator):
    w = weight(df)
    column_names = list(df.columns)
    result = 0
    
    y = normalizeFeature(df, 'PE')
    
    # len(df.axes[0]) = 9568
    for i in range (1, len(df.axes[0])):
        
        # len(df.axes[1]) = 5
        x = normalizeObservation(df, i)
        x = np.delete(x, len(df.axes[1]))
                
        # product of x * w
        # aim of the for : w0 + (x1 * w1) + (x2 * w2) + ... + (xn * wn)
        hwx = w[0]
        for i in range(0, len(x)):
            hwx = w[i] * x[i]
                          
                
        if(type(multiplicator) is not list):
            mult = (hwx - y[i])          
            result += (hwx - y[i]) * (hwx - y[i])
            
        else : 
            result += (hwx - y[i]) * multiplicator[i]
                
    
    result /= (2 * len(df.axes[0]))
    
    return result[0]



J = costFunction(df, -1)
```

#### We have to implement the Gradient Descent
```Python
def gradientDescent(df):
    m = len(df.axes[0])     # len(df.axes[0]) = 9568
    alpha = 0.03            # written in instructions
    result = 0
    
    wk = 3
    prev_wk = 10
    
    max_iterations = 1000   # written in instructions
    iterations = 0
    
    
    while( (prev_wk > alpha) | (iterations < max_iterations) ):
        prev_wk = wk        
                
        xk = normalizeObservation(df, iterations)
        xk = np.delete(xk, len(df.axes[1]))
        
        J = costFunction(df, xk)
        
        J = math.sqrt(J) 
        J /= math.sqrt(len(df.axes[0]))
        
        
        result = J * alpha
                
        wk -= result
        iterations += 1
    
    return wk[0]




wk = gradientDescent(df)
```




## Part2 - Compare to Scikit-Learn


## Part3 - Normal Equation