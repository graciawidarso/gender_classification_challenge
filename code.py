from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# CHALLENGE - create 3 more classifiers...
# 1. SVM
# 2. KNN
# 3. Neural Network

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40],[190, 90, 47], 
     [175, 64, 39],[177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 
     'female', 'male', 'male', 
     'female', 'female','female', 
     'male', 'male']

#classifier
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_knn = KNeighborsClassifier()
clf_nn = MLPClassifier()

#train the model
clf_tree = clf_tree.fit(X, Y)
clf_svm = clf_svm.fit(X,Y)
clf_knn = clf_knn.fit(X,Y)
clf_nn = clf_nn.fit(X,Y)

#prediction
pred_tree = clf_tree.predict(X)
pred_svm = clf_svm.predict(X)
pred_knn = clf_knn.predict(X)
pred_nn = clf_nn.predict(X)

#accuracy
acc_tree = accuracy_score(Y,pred_tree)
acc_svm = accuracy_score(Y,pred_svm)
acc_knn = accuracy_score(Y,pred_knn)
acc_nn = accuracy_score(Y,pred_nn)

#print the accuracy
tree = "Accuracy score of Decision Tree: {}".format(acc_tree)
svm = "Accuracy score of SVM: {}".format(acc_svm)
knn = "Accuracy score of KNearestNeighbors: {}".format(acc_knn)
nn = "Accuracy score of NeuralNetworks: {}".format(acc_nn)

print(tree)
print(svm)
print(knn)
print(nn)

#find the best accuracy
acc_list = [acc_svm, acc_knn, acc_nn]
model = {0: 'SVM', 
         1: 'KNearestNeighbors',
         2: 'Neural Network'}

acc = 0
best_model = ''
for i in acc_list:
    if i > acc:
        acc = i
        best_model = model[acc_list.index(i)]
print("The best model is {}, the accuracy: {}".format(best_model,acc))
