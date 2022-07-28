from sklearn.svm import  LinearSVC
from sklearn.metrics import accuracy_score

learn_data = [[0,0],[1,0],[0,1],[1,1]]
learn_label=[0,1,1,0]
clf = LinearSVC()

clf.fit(learn_data,learn_label)
test_data = [[0,0],[1,0],[0,1],[1,1]]
test_label = clf.predict(test_data)

print(test_data,"predict result:" ,test_label)
print(" accurate  = ",accuracy_score([0,1,1,0],test_label))