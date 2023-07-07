require 'pycall/import'
include PyCall::Import

pyimport 'pandas'
pyimport 'sklearn.preprocessing', as: 'pre'
pyimport 'sklearn.model_selection', as: 'ms'
pyimport 'sklearn.linear_model', as: 'lm'
pyimport 'sklearn.neighbors', as: 'nb_neighbors'
pyimport 'sklearn.svm', as: 'svm'
pyimport 'sklearn.naive_bayes', as: 'nb'
pyimport 'sklearn.tree', as: 'tree'
pyimport 'sklearn.ensemble', as: 'ensemble'
pyimport 'sklearn.metrics', as: 'metrics'
pyimport "lazypredict.Supervised", as: 'lc' 

# Load Mushroom dataset
df = pandas.read_csv('./mushrooms.csv')

puts
puts "Head: "
puts df.head()

# Print the descriptive statistics of the DataFrame
puts
puts "Description: "
puts df.describe()

# Print the data types of the columns in the DataFrame
puts
puts "Information: "
puts df.info()

# Print the count of the different classes in the "class" column
puts
puts "Value Counts: "
puts df['class'].value_counts()

# Check for null values in the DataFrame
puts "No of null values (column-wise)"
puts df.isnull().sum()

# Fit the LabelEncoder to the DataFrame (categoial to numerical)
labelencoder = pre.LabelEncoder.new
df = df.apply(lambda { |column| labelencoder.fit_transform(column) })

# Drop the "veil-type" column
df = df.drop(["veil-type"], axis: 1)


# Get the "class" column as a NumPy array
y = df["class"].values

# Get all the other columns as a NumPy array
x = df.drop(["class"], axis:1).values

# Split the data into train and test sets
x_train, x_test, y_train, y_test = ms.train_test_split(x, y, random_state: 50, test_size: 0.3)

# Create a LogisticRegression model
lr = lm.LogisticRegression.new(solver: "liblinear")

# Fit the model to the training data
lr.fit(x_train, y_train)


# Print the test accuracy of the model
puts
puts "Logistic regression: "
puts "Test Accuracy: #{(lr.score(x_test, y_test) * 100).round(2)}%"

# Find the best K value for KNeighborsClassifier
best_Kvalue = 0
best_score = 0
(1..10).each do |i|
  knn = nb_neighbors.KNeighborsClassifier.new(n_neighbors: i)
  knn.fit(x_train, y_train)
  if knn.score(x_test, y_test) > best_score
    best_score = knn.score(x_test, y_test)
    best_Kvalue = i
  end
end

# Print the best K value and the test accuracy
puts
puts "Best KNN Value: #{best_Kvalue}"
puts "K nearest neighbours Classifier: "
puts "Test Accuracy: #{(best_score * 100).round(2)}%"

# Create a SVC model
svm_model = svm.SVC.new(random_state: 42, gamma: "auto")

# Fit the model to the training data
svm_model.fit(x_train, y_train)

# Print the test accuracy of the model
puts
puts "Support vector Classifier: "
puts "Test Accuracy: #{(svm_model.score(x_test, y_test) * 100).round(2)}%"


# Create a GaussianNB model
naive_bayes = nb.GaussianNB.new

# Fit the model to the training data
naive_bayes.fit(x_train, y_train)

# Print the test accuracy of the model
puts
puts "Naive Bayes Classifier: "
puts "Test Accuracy: #{(naive_bayes.score(x_test, y_test) * 100).round(2)}%"

# Create a DecisionTreeClassifier model
decision_tree = tree.DecisionTreeClassifier.new

# Fit the model to the training data
decision_tree.fit(x_train, y_train)

# Print the test accuracy of the model
puts
puts "Decision tree classifier: "
puts "Test Accuracy: #{(decision_tree.score(x_test, y_test) * 100).round(2)}%"


# Create a RandomForestClassifier model
rf = ensemble.RandomForestClassifier.new(n_estimators: 100, random_state: 50)

# Fit the model to the training data
rf.fit(x_train, y_train)


# Print the test accuracy of the model
puts
puts "Random Forest Classifier: "
puts "Test Accuracy: #{(rf.score(x_test, y_test) * 100).round(2)}%"

puts
clf = lc.LazyClassifier.new(predictions: true)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
puts models
