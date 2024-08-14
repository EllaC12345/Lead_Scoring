# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 6: MLFLOW 
# PART 1: PYCARET INTEGRATION
# ----
#%%
# Core
import pandas as pd
import numpy as np
# Machine Learning
import pycaret.classification as clf
# MLFlow Models
import mlflow
from mlflow.tracking import MlflowClient
# Lead Scoring
import email_lead_scoring as els

# RECAP 

leads_tags_df = els.db_read_and_process_els_data()

leads_tags_df

#%%
# 1.0 PYCARET'S MLFLOW INTEGRATION ----

# When we setup pycaret, we setup logging experiments with MLFlow

#Job & Experiment Logging: 
#clf.setup()
    #n_jobs = -1,
    #session_id = 123,
    #log_experiment=True,
    #experiment_name = 'email_lead_scoring_1'

?clf.setup

# 2.0 MLFLOW UI ----

#!mlflow ui
# http://localhost:5000/

# how to run this in the terminal 
!mlflow ui

# how to kill it
pkill -f "mlflow ui"


# 1. GUI OVERVIEW: HOW TO FIND THE MODELS YOU'RE LOOKING FOR
# 2. SORTING: AUC 
# 3. SEARCHING: tags.Source = 'finalize_model'
# 4. DRILLING INTO MODELS

#%%
# 3.0 MLFLOW TRACKING & EXPERIMENT INTERFACE ----


# 3.1 TRACKING URI (FOLDER WHERE YOUR EXPERIMENTS & RUNS ARE STORED) ----

mlflow.get_tracking_uri()

# 3.2 WORKING WITH EXPERIMENTS (GROUP OF RUNS) ----

# Listing Experiments
#get the client first 
client = mlflow.MlflowClient()
client
#experiments = client.list_experiments()
mlflow.get_experiment("0")
mlflow.get_experiment_by_name('email_lead_scoring_0')


mlflow.__version__
# Programmatically Working With Experiments
#create the list of experiments
mlflow.search_experiments()
experiments = mlflow.search_experiments()
experiments

#retrieve different type of information(i.e. experiment id, location)
experiment_id_0 = experiments[0].experiment_id
experiment_location_0 = experiments[0].artifact_location
experiment_tags = experiments[0].tags


# 3.3 SEARCHING WITH THE EXPERIMENT NAME ----
logs_df = mlflow.search_runs(
experiment_ids=experiment_id_0
)

best_run_id = logs_df \
    .query("`tags.Source`  in ['finalize_model']") \
    .sort_values("metrics.AUC", ascending=False) \
    ['run_id'] \
    .values \
    [0]
best_run_id 
# pycaret interface to get experiments

clf.setup(data=leads_tags_df, target='made_purchase', log_experiment=True, experiment_name='email_lead_scoring_0', verbose = False)

clf.get_logs(experiment_name='email_lead_scoring_0')


#%%
# 4.0 WORKING WITH RUNS ----

# Finding Runs in Experiments
best_run_id
run = mlflow.get_run(run_id=best_run_id)
run

# Finding Runs from Run ID
run = mlflow.get_run(run_id=best_run_id)

#%%
# 5.0 MAKING PREDICTIONS ----

# Using the mlflow model 
import mlflow
#import mlflow.pyfunc
logged_model = 'runs:/c65bb5cc254d46c4a8be3fc3294800e5/model'

environment = mlflow.pyfunc.get_model_dependencies(logged_model)
# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)
?mlflow_load_pyfunc
loaded_model

# Predict on Pandas DataFrame
import pandas as pd

cleaned_leads_df = leads_tags_df.drop(columns=['made_purchase', 'mailchimp_id', 'user_full_name', 'user_email', 'optin_time', 'email_provider'], axis =1)
loaded_model.predict(cleaned_leads_df)

# Issue - Mlflow does not give probability. We need probabilities for lead scoring.

# Solution 1 - Extract out the sklearn model and use the 
#   sklearn .predict_proba() method

loaded_model._model_impl

loaded_model._model_impl.predict_proba(cleaned_leads_df
)[:,0]
# Solution 2 - Predict with Pycaret's prediction function in production
leads_tags_df

clf.predict_model(
    estimator = loaded_model._model_impl,
    data = leads_tags_df,
    raw_score = True
)

print (leads_tags_df.columns)

clf.load_model(
    model_name = f'mlruns/263870421031489217/{best_run_id}/artifacts/model/model'

)
# CONCLUSIONS ----
# 1. Pycaret simplifies the logging process for mlflow
# 2. Pycaret includes a lot of detail (saved metrics) by default
# 3. We haven't actually logged anything of our own 
# 4. Let's explore MLFlow deeper with H2O (We can actually use this process for Scikit Learn too!)

# %%
