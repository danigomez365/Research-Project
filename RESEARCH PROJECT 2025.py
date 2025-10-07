#RESEARCH PROJECT 

# Data handling
import pandas as pd
import numpy as np

# Model building
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix
#visualization
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/danigomez/Downloads/dataverse_files 2/GEMM_v3_5.csv", low_memory=False)
df.head()

display(df.isnull().sum())
#decide what do with the null values- different columns
print(df.columns)
print(df)


df.drop(['ID', 'pilot', 'f1_date', 'f1_channel', 'f1_type', 'f2_date',
       'f2_channel', 'f2_type', 'f3_date', 'f3_channel', 'f3_type', 'f4_date',
       'f4_channel', 'f4_type', 'f5_date', 'f5_channel', 'f5_type', 'f6_date',
       'f6_channel', 'f6_type', 'f7_date', 'f7_channel', 'f7_type', 'f8_date',
       'f8_channel', 'f8_type', 'f9_date', 'f9_channel', 'f9_type', 'f10_date',
       'f10_channel', 'f10_type', 'f11_date', 'f11_channel', 'f11_type', 'app_retrieved', 'app_published', 'app_sent',
       'app_sent_channel', 'wp_nuts2', 'wp_nuts3','fullname','phenotype','extrahsrel', 'extraprodgrade', 'response', 'response_col',
       'anyinterest'], axis=1, inplace=True)


df = pd.get_dummies(df, columns=['occupation', 'comp_size', 'equal', 'anonym', 'wantsalary',
       'responsible', 'worktype', 'gender', 'grade', 'qualmismatch',
       'migrationstatus', 'ethnicity', 'native', 'ethnicity7', 'treat_perf',
       'treat_warmth', 'headscarf','religion','country', 'invitation'])

print(df.columns)


df.head()

#setting the independent variable 

X = df.drop(columns=['invitation_Invitation','invitation_No invitation'])

y = df['invitation_Invitation']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)

X.describe()



num_rows, num_columns = df.shape

# Print the number of features
num_features = num_columns - 1  
print("Number of features:", num_features)


#finding best parameters
from sklearn.model_selection import GridSearchCV

parameters = {
    'n_estimators': range(50, 201, 50),  
    'max_features': [11], # square root of number of features 
    'min_samples_leaf': range(1, 5)
}

rf=RandomForestClassifier(random_state=5)
grid_search= GridSearchCV(estimator=rf, param_grid=parameters)
grid_search.fit(X_train,y_train)
model = grid_search.best_estimator_

best_params = grid_search.best_params_
print("Best params:", best_params)

#accuracy score
from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred) 
print("Accuracy:", accuracy)

estimators_range = range(50, 201, 50)

# List to store cross-validation scores
cv_scores = []

# Perform cross-validation for each number of estimators
for n_estimators in estimators_range:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=5)
    scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot validation curve
import matplotlib.pyplot as plt

plt.plot(estimators_range, cv_scores)
plt.xlabel('Number of Estimators')
plt.ylabel('Cross-Validation Score')
plt.title('Validation Curve for Random Forest')
plt.show()







# In[2]:


# test for females
X_test_female = X_test[X_test['gender_Female'] == 1]
y_test_female = y_test.loc[X_test_female.index]

# predict
y_pred_female = model.predict(X_test_female)

#accuracy score 
accuracy_female = accuracy_score(y_test_female, y_pred_female)
print("Accuracy for females:", accuracy_female)


#count how many times the prediction is invitation_Invitation for females
invitation_female_count = (y_pred_female == 1).sum()

#print count of predictions 
print("Number of times 'invitation_Invitation' (1) is predicted for females: " + str(invitation_female_count))

# Test for males
X_test_male = X_test[X_test['gender_Male'] == 1]  
y_test_male = y_test.loc[X_test_male.index]

# Predict
y_pred_male = model.predict(X_test_male)

# Accuracy score
accuracy_male = accuracy_score(y_test_male, y_pred_male)
print("Accuracy for males:", accuracy_male)

#count how many times the prediction is invitation_Invitation for males 
invitation_male_count = (y_pred_male == 1).sum()

# print the count
print("Number of times invitation_Invitation is predicted for males: " + str(invitation_male_count))

#no religion
#  no religious affiliation 
X_test_no_religion = X_test[X_test['religion_No religious affiliation'] == 1]
y_test_no_religion = y_test.loc[X_test_no_religion.index]

# predict for no religion
y_pred_no_religion = model.predict(X_test_no_religion)

# accuracy for people with no religious affiliation
accuracy_no_religion = accuracy_score(y_test_no_religion, y_pred_no_religion)
print("Accuracy for people with no religious affiliation:", accuracy_no_religion)

#religion 
X_test_religion = X_test[X_test['religion_No religious affiliation'] == 0]
y_test_religion = y_test.loc[X_test_religion.index]

# Predict for religion
y_pred_religion = model.predict(X_test_religion)

# accuracy for people with religious affiliation
accuracy_religion = accuracy_score(y_test_religion, y_pred_religion)
print("Accuracy for people with religious affiliation:", accuracy_religion)


#count for invitaition_Invitation for no religious affiliation
invitation_no_religion_count = (y_pred_no_religion == 1).sum()

# Print the count
print("Number of times 'invitation_Invitation' is predicted for people with no religious affiliation: " + str(invitation_no_religion_count))

#count for invitation_Invitation for religious affiliation
invitation_religion_count = (y_pred_religion == 1).sum()

# Print the count
print("Number of times 'invitation_Invitation' is predicted for people with religious affiliation: " + str(invitation_religion_count))

#confusion matrix for females
cm_female = confusion_matrix(y_test_female, y_pred_female)


print("Confusion Matrix for Females:")
print(cm_female)


from sklearn.metrics import precision_score, recall_score

#precision for females 
precision = precision_score(y_test_female, y_pred_female)
print("Precision:", round(precision, 2))


#recall for females
recall = recall_score(y_test_female, y_pred_female)
print("Recall:", round(recall, 2))

from sklearn.metrics import precision_score, recall_score

#precision for females 
precision = precision_score(y_test_female, y_pred_female)
print("Precision:", round(precision, 2))


#recall for females
recall = recall_score(y_test_female, y_pred_female)
print("Recall:", round(recall, 2))

#precision and recall for males
precision = precision_score(y_test_male, y_pred_male)
print("Precision:", round(precision, 2))

recall = recall_score(y_test_male, y_pred_male)
print("Recall:", round(recall, 2))


#religious confusion matrix 
cm_religion = confusion_matrix(y_test_religion, y_pred_religion)

print("Confusion Matrix for religion:")
print(cm_religion)


#religious confusion matrix 
cm_religion = confusion_matrix(y_test_religion, y_pred_religion)

print("Confusion Matrix for religion:")
print(cm_religion)


#confusion matrix for non-religious 
cm_no_religion = confusion_matrix(y_test_no_religion, y_pred_no_religion)

print("Confusion Matrix for religion:")
print(cm_no_religion)


#precision and recall for non-religious 

precision = precision_score(y_test_no_religion, y_pred_no_religion)
print("Precision:", round(precision, 2))

recall = recall_score(y_test_no_religion, y_pred_no_religion)
print("Recall:", round(recall, 2))


# gender difference invitation ratio
gender_ratios = {
    "Female": df.loc[df['gender_Female'] == 1, 'invitation_Invitation'].mean(),
    "Male": df.loc[df['gender_Male'] == 1, 'invitation_Invitation'].mean(),
}
print("Gender based Invitation Ratios:")
print(gender_ratios)

#religious affiliation or non religious affiliation invitation ratio 
religion_columns = [
    'religion_Buddhist', 
    'religion_Christian', 
    'religion_Hindu', 
    'religion_Muslim', 
    'religion_No religious affiliation'
]

religion_ratios = {
    religion: df.loc[df[religion] == 1, 'invitation_Invitation'].mean()
    for religion in religion_columns
}
print("Religious affiliation Invitation Ratios:")
print(religion_ratios)

df['Religious'] = df[['religion_Buddhist', 'religion_Christian', 'religion_Hindu', 'religion_Muslim']].any(axis=1)

religion_status_ratios = {
    "Religious": df.loc[df['Religious'], 'invitation_Invitation'].mean(),
    "Non-Religious": df.loc[df['religion_No religious affiliation'] == 1, 'invitation_Invitation'].mean()
}
print("Religious vs Non-Religious Invitation Ratios:")
print(religion_status_ratios)


# # Part 2: Fairlearn

# In[3]:


get_ipython().system('pip install fairlearn')


from fairlearn.metrics import demographic_parity_difference, MetricFrame, selection_rate

# Gender-sensitive features (use only one column that splits the groups)
sensitive_gender = X_test['gender_Female']  # True = Female, False = Male

# Calculate Demographic Parity Difference
dpd_gender = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_gender)
print("Demographic Parity Difference (Gender):", round(dpd_gender, 3))

# Calculate Demographic Parity Ratio
gender_frame = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_gender)
rates = gender_frame.by_group
dpr_gender = min(rates) / max(rates)
print("Demographic Parity Ratio (Gender):", round(dpr_gender, 3))




# In[4]:


X_test['Religious'] = X_test[['religion_Buddhist', 'religion_Christian', 'religion_Hindu', 'religion_Muslim']].any(axis=1).astype(int)


from fairlearn.metrics import demographic_parity_difference, MetricFrame, selection_rate

# Religion-based fairness metrics
sensitive_religion = X_test['Religious']

dpd_religion = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_religion)
print("Demographic Parity Difference (Religion):", round(dpd_religion, 3))

religion_frame = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_religion)
dpr_religion = min(religion_frame.by_group) / max(religion_frame.by_group)
print("Demographic Parity Ratio (Religion):", round(dpr_religion, 3))


# In[5]:


#1 = Muslim, 0 = everyone else
sensitive_muslim = X_test['religion_Muslim']

dpd_muslim = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_muslim)
print("Demographic Parity Difference (Muslim vs Others):", round(dpd_muslim, 3))

mf_muslim = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_muslim)
dpr_muslim = min(mf_muslim.by_group) / max(mf_muslim.by_group)
print("Demographic Parity Ratio (Muslim vs Others):", round(dpr_muslim, 3))


# In[6]:


from fairlearn.metrics import demographic_parity_difference, MetricFrame, selection_rate

# 1 = Christian, 0 = everyone else
sensitive_christian = X_test['religion_Christian']

dpd_christian = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_christian)
print("Demographic Parity Difference (Christian vs Others):", round(dpd_christian, 3))

mf_christian = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_christian)
dpr_christian = min(mf_christian.by_group) / max(mf_christian.by_group)
print("Demographic Parity Ratio (Christian vs Others):", round(dpr_christian, 3))







