import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

a=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\abalone.csv')

print(a)
x=a.drop('Sex',axis=1)
y=a['Sex']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
ww=[]
w=[]
for k in range(3,20):
 w.append(k)
 model=KNeighborsClassifier(k)

 model.fit(x_train,y_train)

 s=model.score(x_train,y_train)
 ww.append(s)
sns.boxplot(y_train)
# plt.plot(w,ww,c='g')
# plt.xlabel('K',fontsize=16)
# plt.ylabel('Score',fontsize=16)
# plt.title('Score_Versus_K_Factor_for_abalone_Dataset',fontsize=16)
plt.show()