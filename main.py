import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import pandas as pd

ds = pd.read_csv('C:\\Users\\Manali\\Desktop\\HSNProjects\\ML\\Iris\\IRIS.csv')
ds.head()

%matplotlib inline
img=mimg.imread('C:\\Users\\Manali\\Desktop\\HSNProjects\\ML\\Iris\\iristypes.jpg')
plt.figure(figsize=(20,40))
plt.axis('off')
plt.imshow(img)

X = ds.iloc[:,:4].values
y = ds['species'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 75)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)

y_pred = nvclassifier.predict(X_test)
print(y_pred)

y_compare = np.vstack((y_test,y_pred)).T

y_compare[:5,:]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


a = cm.shape
cp = 0
fp = 0

for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            cp +=cm[row,c]
        else:
            fp += cm[row,c]
print('Correct: ', cp)
print('Wrong: ', fp)
print ('Accuracy of the Naive Bayes Clasification is: ', cp/(cm.sum()))