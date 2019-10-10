#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
#import pandas_profiling
import matplotlib.pyplot as plt


# In[2]:


test= pd.read_csv('test.csv')
train = pd.read_csv("train.csv")

#train.profile_report()
# In[3]:


print(train.shape)
print(test.shape)


# In[4]:


train
#test.head()


# In[5]:


#data_exploration
#sex
print(train.Sex.describe())
sex_pivot = train.pivot_table(values=['Survived'], index=['Sex'])
print(sex_pivot)
train.Sex.value_counts().plot.bar()
sex_pivot.plot.bar()
plt.show()  
#print(train['Age'].describe())
#print(train['Embarked'].describe()) #women are more likely to survive


# In[6]:


#Pclass
print(train.Pclass.describe())
print(train.Pclass.value_counts())
Pclass_pivot = train.pivot_table(index=['Pclass'], values= ['Survived'])
print(Pclass_pivot)
train.Pclass.value_counts().plot.bar()
Pclass_pivot.plot.bar()
plt.show()


# In[7]:


#Age   #We will divide age into columns using cut function
                             #count is 714 i.e. less than 891
survived = train[train['Survived']==1]
died = train[train['Survived']==0]
survived.Age.plot.hist(color='blue', bins = 50)
died.Age.plot.hist(color = 'red', bins = 50)                #in some age groups more people survived than died


# In[8]:


#Age
train['Age']= train['Age'].fillna(-0.5)
test['Age']= test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 13 , 19 , 35 , 60, 100]
label = ['Missing', 'Infant', 'Child', 'Teenager', 'Young', 'Adult', 'Old']
#print(len(label) == len(bins) - 1)
train['cut_age'] = pd.cut(train.Age,bins, labels= label )
test['cut_age'] = pd.cut(test.Age,bins, labels= label )
#print(train.head())
cut_age_pivot= train.pivot_table(values = ['Survived'], index= ['cut_age'])
train.cut_age.value_counts().plot.bar()
cut_age_pivot.plot.bar() 
print(cut_age_pivot)
                                                            #survival rate of certain class is more than  other


# In[9]:


#SibSp
print(train.SibSp.describe())
train.SibSp.value_counts().plot.bar()
train.pivot_table(values = 'Survived' , index = 'SibSp').plot.bar()


# In[10]:


#Parch
print(train.SibSp.describe())
train.Parch.value_counts().plot.bar()
train.pivot_table(values = 'Survived' , index = 'Parch').plot.bar()


# In[11]:


#Embarked
print(train.SibSp.describe())
train.Embarked.value_counts().plot.bar()
train.pivot_table(values = 'Survived' , index = 'Embarked').plot.bar()
train['Embarked'] = train['Embarked'].fillna('C')


# In[12]:


#PREPARING DATA FOR MACHINE LEARNING
train_final = train
test_final  = test

for column in ['cut_age', 'Pclass', 'Sex', 'Embarked']:
        dummies = pd.get_dummies(train[column], prefix=column)
        train_final = pd.concat([train_final , dummies], axis = 1) 
        
for column in ['cut_age', 'Pclass', 'Sex', 'Embarked']:
        dummies = pd.get_dummies(test[column], prefix=column)
        test_final = pd.concat([test_final , dummies], axis = 1)       

#print(len(train_final.columns.tolist()))        
train_final.columns.tolist()
        


# In[13]:


#CREATING MACHINE LEARNING MODEL
#LOGISTIC REGRESSION MODEL

def sigmoid(z):
    return 1/(1+ np.exp(-z))
              


# In[14]:


def h(theta, X):
     z = np.dot(X, theta)
     return sigmoid(z)
#h(np.array([1,1]), np.array([[1,1],[2,2]]))


# In[15]:


def cost_function(h, y,m):
    J = -(np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))/m
    return float(J)


# def gradient_descent(initial_theta, alpha,y,X):
#     theta = initial_theta
#     h1 = h(theta, X)
#     J = cost_function(h1,y)
#     while (J> 10**(-3)) :
#         theta -= alpha*np.dot(X.T, (h1-y))/m
#         h1 = h(theta, X)
#         J = cost_function(h1,y)
#     return theta   
#     
#             

# def predict(theta, X):
#     h_final = h(theta, X)
#     for i in range(m):
#        if(h_final[i]>=0.5):
#             h_final[i] = 1
#        else :
#             h_final[i] = 0
#     return h_final        
#             

# In[ ]:





# In[33]:


dummies_train = pd.get_dummies(train['cut_age'], prefix = 'age')
dummies_test = pd.get_dummies(test['cut_age'], prefix = 'age')
tr_final = pd.concat([train , dummies_train], axis=1)
ts_final = pd.concat([test, dummies_test], axis=1)

tr_final.head()
# In[34]:


tr_final= tr_final.drop(columns='cut_age')
ts_final= ts_final.drop(columns='cut_age')


# In[35]:


temp = tr_final.loc[:, 'age_Missing':'age_Old']
tr_final = tr_final[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
tr_final = pd.concat([tr_final, temp], axis =1)
#tr_final


# In[36]:


temp = ts_final.loc[:, 'age_Missing':'age_Old']
ts_final = ts_final[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
ts_final = pd.concat([ts_final, temp], axis =1)
#ts_final


# In[37]:


tr_final.loc[tr_final.Sex=='male', 'Sex'] = 0
tr_final.loc[tr_final.Sex=='female', 'Sex'] = 1
#tr_final
        


# In[ ]:





# In[38]:


ts_final.loc[ts_final.Sex=='male', 'Sex'] = 0
ts_final.loc[ts_final.Sex=='female', 'Sex'] = 1
#ts_final        


# In[39]:


tr_final.loc[tr_final['Embarked']=='S' , 'Embarked'] = 0
tr_final.loc[tr_final['Embarked']=='Q', 'Embarked'] = 1
tr_final.loc[tr_final['Embarked']=='C', 'Embarked'] = 2
#tr_final       


# In[40]:


ts_final.loc[ts_final['Embarked']=='S' , 'Embarked'] = 0
ts_final.loc[ts_final['Embarked']=='Q', 'Embarked'] = 1
ts_final.loc[ts_final['Embarked']=='C', 'Embarked'] = 2
#ts_final       


# In[64]:


#h1 = h(initial_theta, X)
y = train['Survived'].values
y=y.reshape(891,1)


# In[68]:


X1 = tr_final.values
X = np.ones((891,13))
X[:,1:] =X1
initial_theta = np.zeros((13,1))
h1 = h(initial_theta, X)
alpha = 0.0001
#y = train['Survived'].values

theta = initial_theta
for i in range(500000) :
    theta = theta - (alpha*np.dot(X.T, (h1-y)))/891
    h1 = h(theta, X)        
        


# In[69]:


theta


# In[70]:


final =h(theta, X)
#final


# In[71]:


for i in range(891):
       if(final[i]>=0.5):
            final[i] = 1
       else :
            final[i] = 0
final            
accuracy = final[final == y].shape[0]/891
accuracy


# In[72]:


X1 = ts_final.values
X = np.ones((418,13))
X[:,1:] =X1
output = h(theta, X)
for i in range(418):
       if(output[i]>=0.5):
            output[i] = 1
       else :
            output[i] = 0          
output = output.astype(int)
output


# In[73]:


output_final = pd.concat([test['PassengerId'], pd.DataFrame(output,columns= ['Survived'])], axis=1)
output_final.to_csv('output_final1.csv', index=False)


# In[ ]:





# In[ ]:




