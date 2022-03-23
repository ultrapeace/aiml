from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

"""
Iris Plants Dataset, dataset contains 150 (50 in each of three classes)
Number of Attributes: 4 numeric, predictive attributes and the Class
"""
iris=datasets.load_iris()

""" 
The x variable contains the first four columns of the dataset 
(i.e. attributes) while y contains the labels.
"""
x = iris.data
y = iris.target
print ('sepal-length', 'sepal-width', 'petal-length', 'petal-width')
print(x)
print('class: 0-Iris-Setosa, 1- Iris-Versicolour, 2- Iris-Virginica')
print(y)

""" splits the dataset into 70% train data and 30% test data. This means that 
out of total 150 records,the training set will contain 105 records and 
the test set contains 45 of those records """
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#to Training the model and Nearest nighbors K=5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

#to make predictions on our test data
y_pred=classifier.predict(x_test)

""" For evaluating an algorithm, confusion matrix, precision, recall and 
f1 score are the most commonly used metrics."""
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))