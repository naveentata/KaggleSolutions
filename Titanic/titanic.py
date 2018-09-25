#read data
import numpy as np
import pandas as pd

fi = pd.read_csv("train.csv")


pclass = list(fi["Pclass"])
sex = list(fi["Sex"])
age = list(fi["Age"])
sibsp = list(fi["SibSp"])
parch = list(fi["Parch"])
fare = list(fi["Fare"])
embarked = list(fi["Embarked"])


#print(fi.head(5))
print(fi.shape)
print(fi.columns)
a = list(fi.columns)
for column in a:
    fi[column] = fi[column].replace(r'\s+', np.nan, regex=True)
    fi[column] = fi[column].fillna(-1)

# print(fi.head(10))

print(pclass)

print(age)
print("===========================================================")
c = 0
for ag in range(len(age)):
    if str(age[ag])=="nan":
        age[ag]=-1
print(age)
print(c)
print("===========================================================")
for gen in sex:
    if gen == "male":
        sex[sex.index(gen)] = 1
    elif gen == "female":
        sex[sex.index(gen)] = 2
    else:
        sex[sex.index(gen)] = -1
print(sex)
for cla in embarked:
    if cla == "C":
        embarked[embarked.index(cla)]= 1
    elif cla == "S":
        embarked[embarked.index(cla)]= 2
    elif cla == "Q":
        embarked[embarked.index(cla)]= 3
    else:
        embarked[embarked.index(cla)]= -1
print(embarked)

pclass1 = np.array(pclass, dtype=np.float32)
sex1 = np.array(sex, dtype=np.float32)
age1 = np.array(age, dtype=np.float32)
sibsp1 = np.array(sibsp, dtype=np.float32)
parch1 = np.array(parch, dtype=np.float32)
fare1 = np.array(fare, dtype=np.float32)
embarked1 = np.array(embarked, dtype=np.float32)




X = np.column_stack((pclass1,sex1,age1,sibsp1,parch1,fare1,embarked1))

y = fi["Survived"]
Y  = np.array(y,dtype=np.int)


# k= fi["Cabin"].unique()
# print(k)

# data = pd.DataFrame({'Cabin': k })
# print(pd.get_dummies(data))

from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()
lm.fit(X,Y)
print(X,Y)

# =========================================================================================
fi1 = pd.read_csv("test.csv")

pid = list(fi1["PassengerId"])
pclass2 = list(fi1["Pclass"])
sex2 = list(fi1["Sex"])
age2 = list(fi1["Age"])
sibsp2 = list(fi1["SibSp"])
parch2 = list(fi1["Parch"])
fare2 = list(fi1["Fare"])
embarked2 = list(fi1["Embarked"])


#print(fi.head(5))
print(fi1.shape)
print(fi1.columns)
a = list(fi1.columns)
for column in a:
    fi1[column] = fi1[column].replace(r'\s+', np.nan, regex=True)
    fi1[column] = fi1[column].fillna(-1)

# print(fi.head(10))

for pc in pclass2:
    if pc == np.nan:
        pclass2[pclass2.index(pc)] =-1 
print(pclass2)

print(age2)
print("===========================================================")
c = 0
for ag in range(len(age2)):
    if str(age2[ag])=="nan":
        age2[ag]=-1
        c=c+1
print(age2)
print(c)
print("===========================================================")
for gen in sex2:
    if gen == "male":
        sex2[sex2.index(gen)] = 1
    elif gen == "female":
        sex2[sex2.index(gen)] = 2
    else:
        sex2[sex2.index(gen)] = -1
print(sex2)
for cla in embarked2:
    if cla == "C":
        embarked2[embarked2.index(cla)]= 1
    elif cla == "S":
        embarked2[embarked2.index(cla)]= 2
    elif cla == "Q":
        embarked2[embarked2.index(cla)]= 3
    else:
        embarked2[embarked2.index(cla)]= -1
print(embarked2)
for ag in range(len(fare2)):
    if str(fare2[ag])=="nan":
        fare2[ag]=-1

pclass3 = np.array(pclass2, dtype=np.float32)
sex3 = np.array(sex2, dtype=np.float32)
age3 = np.array(age2, dtype=np.float32)
sibsp3 = np.array(sibsp2, dtype=np.float32)
parch3 = np.array(parch2, dtype=np.float32)
fare3 = np.array(fare2, dtype=np.float32)
embarked3 = np.array(embarked2, dtype=np.float32)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(pclass3)
print(sex3)
print(age3)
print(sibsp3)
print(parch3)
print(fare3)
print(embarked3)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


p_id1 = []
predictions = []
X2 = np.column_stack((pclass3,sex3,age3,sibsp3,parch3,fare3,embarked3))
predictions = lm.predict(X2)
wri=[['PassengerId','Survived']]
for i in range(len(predictions)):
    wri.append([pid[i],predictions[i]])

import csv
myFile = open('submit.csv', 'w+')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(wri)
kip = lm.score(predictions,Y)
print(kip)