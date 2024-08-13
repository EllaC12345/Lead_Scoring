import pandas as pd
import pycaret.classification as clf
import email_lead_scoring as els
import numpy as np
import mlflow
leads_df = els.db_read_and_process_els_data()

# MODEL LOAD FUNCTION ----
def model_score_leads(
    data,
    model_path = "models/blended_models_final"
):
    
    
    """
    Pycaret Model Lead Scoring Function:

    This function takes a pandas DataFrame containing leads data (`data`)
    and the path to the pycaret model (`model_path`) as input.
    It then performs the following steps:

    1. Loads the pycaret model.
    2. Generates predictions for the lead data using the loaded model.
    3. Transforms the predictions into lead scores (1 - prediction_score)
       where higher scores indicate higher likelihood of conversion.
    4. Concatenates the lead scores with the original data, creating
       a new DataFrame with scoring information.
    5. Returns the resulting DataFrame containing leads and their scores.

    Args:
        data (pandas.DataFrame): Leads data from email_lead_scoring.db_read_and_process_els_data().
        model_path (str, optional): Path to the pycaret model to load.
                                    Defaults to "models/blended_models_final".

    Returns:
        pandas.DataFrame: A new DataFrame containing the original lead data
                          alongside their corresponding lead scores.
    """

    # Load the pycaret model
    mod = clf.load_model(model_path)

    # Predict using the loaded model
    predictions_df = clf.predict_model(estimator=mod, data=data)

    # Ensure existence of 'prediction_score' and 'prediction_label' columns
    if 'prediction_score' not in predictions_df.columns:
        raise ValueError("The model prediction output lacks a 'prediction_score' column.")

    if 'prediction_label' not in predictions_df.columns:
        raise ValueError("The model prediction output lacks a 'prediction_label' column.")

    # Efficiently calculate lead scores, handling both 0 and 1 labels
    predictions_df['lead_score'] = 1 - predictions_df['prediction_score']

    # Concatenate lead scores with original data
    leads_scored_df = pd.concat([predictions_df['lead_score'], data], axis=1)

    return leads_scored_df
     

def mlflow_get_best_run(
    experiment_name, n=1,
    metric= ["metrics.auc"],
    ascending=False,
    tag_source = ['finalize_model', 'H2O_AutoML_Model']
    ):
    """_summary_
    Returns the best run from an MLflow experiment based on AUC and loss metrics.

    Args:
        experiment_name (_type_): MLFlow experiment name.
        n (int, optional): _description_. [n-1].
        metric (list, optional): _description_. Defaults to ["metrics.auc"].
        ascending (bool, optional): _description_. Defaults to False in descending order for AUC metrics and True for loss metrics which should be ascendingg
        tag_source (list, optional): _description_. Defaults to ['finalize_model', 'H2O_AutoML_Model'].

    Returns:
        _type_: _description_
    """

    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    logs_df = mlflow.search_runs(experiment_id)
    best_run_id = logs_df \
        .query(f"`tags.Source` in {tag_source}") \
        .sort_values(metric, ascending=ascending) \
        ["run_id"] \
        .values \
        [n-1]
    return best_run_id , experiment_id 

def mlflow_score_leads( data, run_id):
    
    """_summary_
     this function scores the leads using an MLflow run_id to select a model
     
    Args:
        data (_type_):(DataFrame): leads of data from email_lead_scoring.db_read_and_process_els_data().
        run_id (_type_): (string )An MLflow run_id to select a model.

    Returns:
        _Dataframe: a dataframe containing the original lead data alongside their corresponding lead scores.
    """
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
