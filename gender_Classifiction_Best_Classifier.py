from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# method to get the number of failures in the output
def getFailures(predictions, output):
	index = 0
	failedTests = 0
	for prediction in predictions:
		if output[index] != prediction:
			failedTests = failedTests + 1
		index = index +1

	return failedTests

classifiers =  ["Decision Trees", "Gradient Descent", "SVM failures", "Naive Bayes"]
failures = []

#Train each model and get the predictions
#store the failures of each classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
predictions = clf.predict(X)
failures.append(getFailures(predictions, Y))

clf = SGDClassifier(loss="hinge", penalty="l2")
clf = clf.fit(X, Y)
predictions = clf.predict(X)
failures.append(getFailures(predictions, Y))

clf = svm.SVC()
clf = clf.fit(X, Y)
predictions = clf.predict(X)
failures.append(getFailures(predictions, Y))

clf = GaussianNB()
clf = clf.fit(X, Y)
predictions = clf.predict(X)
failures.append(getFailures(predictions, Y))

# print the best classifier with minimum failures
print(classifiers[failures.index(min(failures))])