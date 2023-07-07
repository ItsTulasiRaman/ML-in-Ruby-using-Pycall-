**Using python functions in ruby:**

**Step 01: Install pycall**

- **In windows:**

  Open terminal (powershell/cmd) using run as administrator then type the command ``gem install pycall``

- **In linux:**

  Using the command ``sudo gem install pycall``

<br/>

**Step 02: Install necessary python packages locally using pip installer**

For this I've used 3 python packages: pandas, scikit-learn, lazypredict

Installing dependencies...

- **In windows:**
```
pip install pandas
pip install -U scikit-learn
pip install lazypredict
```

- **In linux:**
```
pip3 install pandas
pip3 install -U scikit-learn
pip3 install lazypredict
```

Note: If scikit-learn doesn’t work try ‘sklearn’ however sklearn package in deprecated and so not recommended to use.

Note: By default, `pip install` in linux would install packages in python2 environment. Using pip3 specifies the system to install packages in python3 environment.

<br/>

**Step 3: Importing python functions from local systems using pycall**

**For an instance here the import statement of python for the first 2 lines of code:**

```
import pandas
from sklearn import preprocessing as pre
```

**Calling it in ruby:**
```
pyimport 'pandas'
pyimport 'sklearn.preprocessing', as: 'pre'
```

Note: There are other possible ways to do this import. The above sample is not the only syntax.

<br/>

**Step 04: Accessing python functions in ruby:**

**Accessing a function without alias:**

```
df = pandas.read\_csv('./mushrooms.csv')
puts
puts "Head: "
puts df.head()
```

Here, we use the ```read\_csv``` function from pandas module to read the contents of the CSV file and store it into the variable `df`. Finally, we print the head of the DataFrame using df.head().


**Accessing a function with alias:**

puts

clf = lc.LazyClassifier.new(predictions: true)

models, predictions = clf.fit(x\_train, x\_test, y\_train, y\_test)

puts models

Here, the alias `lc` is used to create an instance for LazyClassifier with predictions set to True. Next, we use the fit method of LazyClassifier to train the models on the training data (x\_train and y\_train) and make predictions on the test data (x\_test). The models variable will contain a dictionary with the trained models. Finally, we print the models dictionary.

<br/>

**References and materials:**

<https://lazypredict.readthedocs.io/en/latest/>

<https://www.analyticsvidhya.com/blog/2021/05/lazy-predict-best-suitable-model-for-you/>

<br/>

**Clone and run the project:**
<https://github.com/ItsTulasiRaman/ML-in-Ruby-using-Pycall-.git/>
