
# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | MODEL LEAD SCORING FUNCTION
# ----

import pandas as pd
import pycaret.classification as clf
import email_lead_scoring as els
import numpy as np

leads_df = els.db_read_and_process_els_data()

# MODEL LOAD FUNCTION ----
def model_score_leads(
    data,
    model_path = "models/blended_models_final"
):
    mod = clf.load_model(model_path)
    
    # Predict
    predictions_df = clf.predict_model(estimator = mod, data = data)
    
    # FIX
    """.history/leads_scored_df = pd.concat(
        [1- predictions_df["prediction_score"], data], axis = 1
    )"""
    df = predictions_df
    predictions_df['prediction_score'] = np.where(df['prediction_label'] == 0, 1 - df['prediction_score'], df['prediction_score'])
    
    leads_scored_df = leads_scored_df = pd.concat([1-predictions_df['prediction_score'], data], axis = 1)
    return leads_scored_df
# TEST OUT
model_score_leads(data = leads_df)

import email_lead_scoring as els
leads_df = els.db_read_and_process_els_data()
leads_df
els.model_score_leads(
    data = leads_df,
    model_path = "models/catboost_model_tuned"
)