from sklearn import tree

features=[[150,0],[160,0],[155,0],[130,0],[170,0],[120,1],[110,1],[150,1],[125,1]]
label=[0,0,0,0,0,1,1,1,1]

classifier=tree.DecisionTreeClassifier()
classifier.fit(features,label)
print(classifier.predict([[130,0]]))