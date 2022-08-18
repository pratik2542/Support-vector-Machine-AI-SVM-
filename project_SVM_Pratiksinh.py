# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:11:31 2022

@author: Pratik
"""


import pandas as pd, numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, precision_score, recall_score


data_grp3 = pd.read_csv("C:/Users/user/Downloads/KSI.csv")
data_grp3

# There are several columns consist of "Yes" and "<Null>" (where Null means No). 
# For these binary column, replace  "<Null>" with"No"
unwanted = ['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','PASSENGER','AG_DRIV','PEDESTRIAN']
data_grp3[unwanted]=data_grp3[unwanted].replace({'<Null>':'No', 'Yes':'Yes'})

# Replace other '<Null>' with nan, printing percentage of missing values for each feature
data_grp3.replace('<Null>', np.nan, inplace=True)
data_grp3.replace(' ',np.nan,inplace=True)
print(data_grp3.isna().sum()/len(data_grp3)*100)

# Dropping columns where missing values were greater than 80%
drop_column = ['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND']
data_grp3.drop(drop_column, axis=1, inplace=True)
#Drop irrelevant columns which are unique identifier
data_grp3.drop(['ObjectId','INDEX_'], axis=1, inplace=True)


print(data_grp3.shape)
print(data_grp3.isna().sum()/len(data_grp3)*100)
print(data_grp3.info())
print(data_grp3.select_dtypes(["object"]).columns)

# Neighbourhood is identical with Hood ID
data_grp3.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)
print(data_grp3.select_dtypes(["object"]).columns)

# extract features: weekday,day, month 
data_grp3['DATE'] = pd.to_datetime(data_grp3['DATE'])
data_grp3['WEEKDAY'] =data_grp3['DATE'].dt.dayofweek
data_grp3['DAY'] = pd.to_datetime(data_grp3['DATE']).dt.day
data_grp3['MONTH'] = data_grp3['DATE'].dt.month

#Drop Date
data_grp3.drop(['DATE'], axis=1, inplace=True)
data_grp3.columns

# Neighbourhood is identical with Hood ID, drop Neighbourhood
# X,Y are longitude and latitudes, dulicate, drop X and Y
data_grp3.drop(['NEIGHBOURHOOD','X','Y'], axis=1, inplace=True)
data_grp3.columns

data_grp3['STREET1'].value_counts()
data_grp3['POLICE_DIVISION'].value_counts() 
# remove other irrelevant columns or columns contain too many missing values
data_grp3.drop(['MANOEUVER','DRIVACT','DRIVCOND','INITDIR','STREET1','STREET2','WARDNUM','POLICE_DIVISION','DIVISION'], axis=1, inplace=True)
data_grp3.columns
data_grp3.info()

#Injury
ax=sns.catplot(x='INJURY', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("INJURY")

data_grp3['INJURY'].value_counts()



#Visualization

#Number of Unique accidents by Year
Num_accident = data_grp3.groupby('YEAR')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("years")
plt.ylabel('Accidents Numbers')
ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = list('red')   #red, green, blue, black, etc.
Num_accident.plot(
    kind='barh', 
    color='black',
    edgecolor='red'
)
plt.show()

#Check the relation between features and target
ax=sns.catplot(x='YEAR', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=45)
plt.title("Accidents in different years")

#Neighborhood
ax=sns.catplot(x='DISTRICT', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=45)
plt.title("Accidents in different day of a week")

#Vehicle type
ax=sns.catplot(x='VEHTYPE', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Vehicle type vs. occurance of accidents")

#LOCCOORD
ax=sns.catplot(x='LOCCOORD', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Location Coordinate")

#INVAGE
ax=sns.catplot(x='INVAGE', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Age of Involved Party")


# accident location 
#2D histogram
data_NonFatal = data_grp3[data_grp3['ACCLASS'] == 'Non-Fatal']
plt.hist2d(data_NonFatal['LATITUDE'], data_NonFatal['LONGITUDE'], bins=(40, 40), cmap=plt.cm.jet)
plt.title("2D histogram of Non-fatal accidents")
plt.xlabel("LATITUDE")
plt.ylabel("LONGITUDE")
plt.show()

data_Fatal = data_grp3[data_grp3['ACCLASS'] == 'Fatal']
plt.hist2d(data_Fatal['LATITUDE'], data_Fatal['LONGITUDE'], bins=(40, 40), cmap=plt.cm.jet)
plt.title("2D histogram of fatal accidents")
plt.xlabel("LATITUDE")
plt.ylabel("LONGITUDE")
plt.show()


# scatter plot of all fatal and non-fatal accidents
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data = data_grp3[data_grp3['ACCLASS'] == 'Non-Fatal'],alpha=0.3)
plt.title("Non-Fatal Accidents")
plt.show()
#scatter plot of fatal accidents
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data = data_grp3[data_grp3['ACCLASS'] == 'Fatal'],alpha=0.3)
plt.title("Fatal Accidents")
plt.show()

#Data Cleaning

print(data_grp3.isna().sum()/len(data_grp3)*100)

#catagorical feature, not make much sense if impute, so keep the features, just discard these rows with missing values
data_grp3.dropna(subset=['ROAD_CLASS', 'DISTRICT','VISIBILITY','RDSFCOND','LOCCOORD','IMPACTYPE','TRAFFCTL','INVTYPE'],inplace=True)
data_grp3.isnull().sum()

#target class
data_grp3['ACCLASS']=data_grp3['ACCLASS'].replace({'Non-Fatal':0, 'Fatal':1})
data_grp3['ACCLASS'].value_counts()  
#Changing the property damage and non-fatal columns to Non-Fatal¶
data_grp3['ACCLASS'] = np.where(data_grp3['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', data_grp3['ACCLASS'])
data_grp3['ACCLASS'] = np.where(data_grp3['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', data_grp3['ACCLASS'])

data_grp3['ACCLASS'].unique()

data_grp3['ACCLASS']=data_grp3['ACCLASS'].replace({'Non-Fatal':0, 'Fatal':1})
data_grp3['ACCLASS'].value_counts()  
#dataset is unbalanced

#Resampling- Upsampled
from sklearn.utils import resample
dataframe=data_grp3
df_majority = dataframe[dataframe.ACCLASS==0]
df_minority = dataframe[dataframe.ACCLASS==1]
df_majority, df_minority
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=14029,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
print(df_upsampled.ACCLASS.value_counts())

data_grp3=df_upsampled


#Test Train split
#Since the dataset is unbalanced, use straified split
X = data_grp3.drop(["ACCLASS"], axis=1)
y= data_grp3["ACCLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,stratify=y)
X_train, X_test, y_train, y_test

#impute
from sklearn.impute import SimpleImputer    
imputer = SimpleImputer(strategy="constant",fill_value='missing')  
data_tr=imputer.fit_transform(X_train)
data_tr= pd.DataFrame(data_tr, columns=X_train.columns)

print(data_tr.isna().sum()/len(data_tr)*100)

#numerical features
df1=data_grp3.drop(['ACCLASS'],axis=1)
num_data=df1.select_dtypes(include=[np.number]).columns
print(num_data)
data_num =data_tr[num_data] 
#standardize 
scaler = StandardScaler() #define the instance
scaled =scaler.fit_transform(data_num)
data_num_scaled= pd.DataFrame(scaled, columns=num_data)
print(data_num_scaled)

#categorical features
cat_data=df1.select_dtypes(exclude=[np.number]).columns
print(cat_data)
categoricalData =data_tr[cat_data]
print(categoricalData)

data_cat = pd.get_dummies(categoricalData, columns=cat_data, drop_first=True)
data_cat

X_train_prepared=pd.concat([data_num_scaled, data_cat], axis=1)
X_train_prepared.head()
X_train_prepared.columns
X_train_prepared.info()

#Feature Selection

#Feature selection by Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(LogisticRegression(solver='saga',penalty='l1'))
sel.fit(X_train_prepared, y_train)
selected_feat= X_train_prepared.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)


coefficient= pd.Series(sel.estimator_.coef_[0], index=X_train_prepared.columns)
#plot the selected features
fig = plt.gcf()
fig.set_size_inches(10, 20)
coefficient.plot(kind='barh')
plt.title("L1 coefficient")
plt.show()

abs_coefficient =abs(coefficient)
print(coefficient[coefficient==0])
print(coefficient[coefficient<0])
print(coefficient[coefficient>0])


#selected features

#numerical features
num_data=['ACCNUM', 'YEAR', 'TIME', 'HOUR', 'LATITUDE', 'LONGITUDE', 'WEEKDAY', 'DAY', 'MONTH']
data_num =data_tr[num_data] 
num_data=data_num.columns
print(num_data)

#categorical features

cat_data=['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','PASSENGER','AG_DRIV','PEDESTRIAN',
              'ROAD_CLASS', 'DISTRICT',  'TRAFFCTL','VISIBILITY', 'LIGHT', 'RDSFCOND','IMPACTYPE', 'INVAGE']
categoricalData =data_tr[cat_data]
print(categoricalData.columns)
data_cat = pd.get_dummies(categoricalData, columns=cat_data, drop_first=True)
data_cat

df=pd.concat([data_num, data_cat], axis=1)
df


# Pipelines


# build a pipeline for preprocessing the categorical attributes
cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant",fill_value='missing')),
        ('one_hot', OneHotEncoder(drop='first')),
    ])
# build a pipeline for preprocessing the numerical attributes
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
#full transformation Column Transformer
num_attribs = num_data
cat_attribs = cat_data

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])



# Model Training,Tuning and Testing
#SVM
from sklearn.svm import SVC
clf=SVC()
X_train_prepared = full_pipeline.fit_transform(X_train)
clf.fit(X_train_prepared, y_train)
#accuracy on training dataset
print("Training Accuracy",clf.score(X_train_prepared,y_train))

#test
X_test_prepared = full_pipeline.transform(X_test)
#predict
y_test_pred=clf.predict(X_test_prepared)

print("Testing accuracy", accuracy_score(y_test, y_test_pred))
print("precison",precision_score(y_test, y_test_pred))
print("recall",recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


#logistic
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=17)
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()



#test
X_test_prepared = full_pipeline.transform(X_test)
X_train_prepared.shape
lr.fit(X_train_prepared,y_train)
#predict
y_test_pred=lr.predict(X_test_prepared)
print("Training Accuracy",lr.score(X_train_prepared, y_train))

print("accuracy", accuracy_score(y_test, y_test_pred))
print("precison",precision_score(y_test, y_test_pred))
print("recall",recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


# Random Forest

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train_prepared, y_train)

rf_y_pred = rf.predict(X_test_prepared)
print('Accuracy of RandomForest is:', accuracy_score(y_test, rf_y_pred))


# Neural Networks
from sklearn.neural_network import MLPClassifier

clf_neural_network = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

print(clf_neural_network)
clf_neural_network.fit(X_train_prepared, y_train)
clf_neural_network.score(X_train_prepared, y_train)
y_pred = clf_neural_network.predict(X_test_prepared)
print('Accuracy of NN is:', accuracy_score(y_test, y_pred))

#knn
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
import sklearn.metrics as metrics

classifier = KNeighborsClassifier(n_neighbors=2)

classifier.fit(X_train_prepared, y_train)

y_pred = classifier.predict(X_test_prepared) 
y_scores = classifier.predict_proba(X_test_prepared) 
print('Classification Report(N): \n',classification_report(y_test, y_pred)) 
print('Accuracy(N): \n',metrics.accuracy_score(y_test, y_pred))


###
import pickle
#20. Save the model using the joblib (dump).

#21. Save the full pipeline using the joblib - (dump).
pickle.dump(full_pipeline, open('pipeline_group.pkl','wb'))
pickle.dump(y_test_pred, open('model_group.pkl','wb'))
 
import joblib
#21. Save the full pipeline using the joblib - (dump).
joblib.dump(y_test_pred, 'model_group.pkl')
