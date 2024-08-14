# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 6: MLFLOW 
# PART 3: PREDICTION FUNCTION 
# ----
#%%
import pandas as pd
import mlflow
import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()


#%%
# 1.0 GETTING THE BEST RUN FOR PRODUCTION ----
#EXPERIMENT_NAME = 'email_lead_scoring_0'
EXPERIMENT_NAME = 'automl_lead_scoring_1'

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment
experiment_id = experiment.experiment_id

logs_df = mlflow.search_runs(experiment_id)
logs_df.head()

logs_df \
    .query("`tags.Source` in ['finalize_model', 'H2O_AutoML_Model']") \
    .query("`status` in 'FINISHED'")\
    .sort_values("metrics.auc", ascending=False) \
    ["run_id"] \
    .values[0]

logs_df['status'].unique()
# Function
def mlflow_get_best_run(
    experiment_name, n=1,
    metric= 'metrics.auc', # "metrics.auc", "metrics.AUC
    ascending=False,
    tag_source = ['finalize_model', 'H2O_AutoML_Model'],
    #status = 'FINISHED'
    ):
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
 
    experiment_id = experiment.experiment_id
    logs_df = mlflow.search_runs(experiment_id)
   
    
    best_run_id = logs_df \
        .query(f"`tags.Source` in {tag_source}") \
        .sort_values(metric, ascending=ascending) \
        ["run_id"] \
        .values \
        [0]
    return best_run_id , experiment_id 

mlflow_get_best_run('automl_lead_scoring_1')



#%%
# 2.0 PREDICT WITH THE MODEL (LEAD SCORING FUNCTION)


# Load model as a PyFuncModel.

# H2O
run_id = mlflow_get_best_run('automl_lead_scoring_1')
run_id
logged_model = f'runs:/{run_id[0]}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

cleaned_df = leads_df.drop(columns = ['mailchimp_id', 'optin_time', 'user_email', 'user_full_name', 'email_provider', 'made_purchase' ])
loaded_model.predict(cleaned_df)['p1']

###SKLEARN/Pycaret (Extract)

run_id = mlflow_get_best_run('automl_lead_scoring_1')
run_id
logged_model = f'runs:/{run_id[0]}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

loaded_model._model_impl.predict(cleaned_df)['p1']


# Function
def mlflow_score_leads( data, run_id):
    logged_model = f'runs:/{run_id[0]}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    #predict
    try:
        predictions_array = loaded_model.predict(pd.DataFrame(data))['p1']
    except:
        predictions_array = loaded_model._model_impl.predict(pd.DataFrame(data))['p1']
    
    predictions_series = pd.Series(predictions_array, name = "Score")
    
    ret = pd.concat([predictions_series, data], axis = 1)
    
    return ret

mlflow_score_leads(
    data = cleaned_df,
    run_id = mlflow_get_best_run('email_lead_scoring_0')
)

mlflow_score_leads(
    data = cleaned_df,
    run_id = mlflow_get_best_run('automl_lead_scoring_1')
)
# 3.0 TEST WORKFLOW ----
import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()
cleaned_df = leads_df.drop(columns = ['mailchimp_id', 'optin_time', 'user_email', 'user_full_name', 'email_provider', 'made_purchase' ])
best_run_id = els.mlflow_get_best_run('automl_lead_scoring_1')
best_run_id
els.mlflow_score_leads(
    data = cleaned_df,
    run_id = best_run_id
)
# %%
