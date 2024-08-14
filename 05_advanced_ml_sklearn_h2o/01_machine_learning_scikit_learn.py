# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 5: ADVANCED MACHINE LEARNING 
# PART 1: SCIKIT LEARN PIPELINES
# ----

# Core
#%%
import pandas as pd
import numpy as np

# Pycaret
import pycaret.classification as clf

# Sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import *
#from sklearn.metrics import score
from sklearn.metrics import roc_auc_score, confusion_matrix

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Lead Scoring
import email_lead_scoring as els

# Leads Data

leads_df = els.db_read_and_process_els_data()
#%%
# 1.0 LOADING A PYCARET MODEL
mod_1 = clf.load_model('models/xgb_model_tuned')
mod_1
mod_2 = clf.load_model('models/ada_model_tuned')
clf.predict_model(mod_1, data=leads_df, raw_score=True)
?clf.predict_model
# 2.0 WHAT IS A SCIKIT LEARN PIPELINE?
# scikit-learn pipelines are a way to streamline the machine learning workflow, (i.e. data preprocessing, model building, model evaluation).
"""A Pipeline is created using the sklearn.pipeline.Pipeline class and consists of a series of steps, each of which is a tuple with a name and an estimator (a transformer or a model). The Pipeline object is then treated like a regular estimator—it has methods such as fit, predict, and score that can be used to train the pipeline and make predictions."""
"""Chaining Steps: You can chain together multiple steps, such as data preprocessing, feature scaling, and model training, into a single workflow.
Consistency: The same Pipeline can be applied to different datasets, making your code more consistent and reusable.
Code Readability and Modularity: The Pipeline provides a clear, linear structure to your code, making it easier to understand and maintain.
Cross-Validation and Hyperparameter Tuning: You can use the Pipeline with cross-validation and hyperparameter tuning methods (e.g., GridSearchCV) to optimize your entire workflow, not just the model.
Simplified Workflow: By combining multiple steps into one Pipeline, you can simplify your workflow and reduce the potential for errors."""

type(mod_1)

mod_1[0]
mod_1.__dict__.keys()


mod_1.__dict__["steps"]
mod_2.__dict__["steps"]

#check the length of the model or the number of steps in the model
len(mod_1)

#check the last step in the model
mod_1[len(mod_1)-1]

# 3.1 DATA PREPARATION

X = leads_df.drop(columns=['made_purchase', 'mailchimp_id', 'user_full_name', 'user_email', 'optin_time', 'email_provider'], axis =1)

Y = leads_df['made_purchase']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

X_train
X_test

Y_train
Y_test
# 3.2 CREATING A SKLEARN PIPELINE ----



# Instantiate an Encoder & Connect to a Column
enc = OneHotEncoder(handle_unknown='ignore')
?OneHotEncoder

transformer = make_column_transformer((enc, ['country_code']), remainder='passthrough')

# Make a Pipeline
pipeline_rf = make_pipeline(
    transformer,
    RandomForestClassifier(random_state=123)

)

# Fit & Predict with a Pipeline

pipeline_rf.fit(X_train, Y_train)

pipeline_rf.predict(X_test)
pipeline_rf.predict_proba(X_test)


# Metrics
pipeline_rf.score(X_test, Y_test)
predicted_class_rf = pipeline_rf.predict_proba(X_test)[:,1] > 0.035


roc_auc_score(
    y_true = Y_test,
    y_score = predicted_class_rf
)

confusion_matrix(
    y_true = Y_test,
    y_pred = predicted_class_rf
)

# 4.0 GRIDSEARCH -----

# Grid Search CV 
#Exhaustive search over specified parameter values for an estimator
?GridSearchCV
grid_xgb = GridSearchCV(
    estimator = XGBClassifier(),
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2, 0.35, 0.4, 0.6],
    },
    cv = 5,
    refit = True,
    scoring = 'roc_auc'
)

# Make A Pipeline With GridSearch

pipeline_xgb = make_pipeline(
    transformer,
    grid_xgb
)
pipeline_xgb
pipeline_xgb.fit(X_train, Y_train)
pipeline_xgb[1].best_params_

# Metrics

predicted_class_xgb = pipeline_xgb.predict_proba(X_test)[:,1] > 0.035

predicted_class_xgb

roc_auc_score(
    y_true = Y_test,
    y_score = predicted_class_xgb
)

confusion_matrix(
    y_true = Y_test,
    y_pred = predicted_class_xgb

)
# 5.0 PCYARET COMPARISON ----- 

Cleaned_data_df = leads_df.drop(columns=['made_purchase', 'mailchimp_id', 'user_full_name', 'user_email', 'optin_time', 'email_provider'], axis =1)
#mod_2.predict_proba(Cleaned_data.iloc[X_test.index])[:,1]

predicted_class_pycaret = mod_1.predict_proba(
    Cleaned_data_df.iloc[X_test.index]
)[:,1] > 0.60

predicted_class_pycaret

roc_auc_score(
    y_true = Y_test,
    y_score = predicted_class_pycaret
)

#Save and Loading 
import joblib
joblib.dump(pipeline_xgb, 'models/pipeline_xgb.pkl')
joblib.load('models/pipeline_xgb.pkl')

# CONCLUSIONS ----

# 1. See the benefit of using Pycaret (or AutoML)
# 2. Faster to get results than Scikit Learn
# 3. Handles a lot of the cumbersome preprocessing
# 4. The result is a Scikit Learn Pipeline


# %%
