# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | PYCARET
# ----
#%%
# Core
import os
import pandas as pd
import numpy as np
import pycaret
import pycaret.classification as clf
import mlflow.pyfunc


# Lead Scoring
import email_lead_scoring as els

# RECAP ----
#%%
leads_df = els.db_read_and_process_els_data() 
leads_df
?clf.setup

leads_df.info()

#%%
# 1.0 PREPROCESSING (SETUP) ---- 
# - Infers data types (requires user to say yes)
# - Returns a preprocessing pipeline

# Remove unnecessary columns
df = leads_df \
    .drop(['mailchimp_id', 'user_full_name', 'user_email', 'optin_time', 'email_provider' ], axis = 1)

df.info()

# Numeric Features
_tag_mask = df.columns.str.match('tag_')

numeric_features = df.columns[_tag_mask].to_list()
numeric_features 
numeric_features.append('optin_days')
numeric_features

#Catgorical Features
categorical_features = ['country_code']

# Ordinal Features 
df['member_rating'].unique()
ordinal_features = {
    'member_rating': ['1', '2', '3', '4', '5']

}
df
?clf.setup
# classifier Setup
#%%
clf_1 = clf.setup(
    data = df,
    target = 'made_purchase',
    train_size = 0.8,
    preprocess = True,
    
    #Imputation
    imputation_type= 'simple',
    
    #Categorical
    categorical_features = categorical_features,
    #handle_unknown_categorical = ignore,
    categorical_imputation = 'most_frequent',
    #combine_rare_classes = True,
    outliers_method =  'iforest',
    outliers_threshold  = 0.05,
    #rare_class_threshold = 0.05,
    
    
    #Ordinary Features
    ordinal_features =  ordinal_features,
    #Numeric Features
    numeric_features = numeric_features,
    
    #K-Fold
    fold_strategy = 'stratifiedkfold',
    fold = 5,
    
    # job Experiment logging
    n_jobs= -1,
    session_id = 123,
    log_experiment = True,
    experiment_name = 'email_lead_scoring_0',
    
    #Silent: Turns off asking for data types inferred correctly
    verbose = False)

print(pycaret.__version__)

clf_1
#%%
# 2.0 GET CONFIGURATION ----
# - Understanding what Pycaret is doing underneath
# - Can extract pre/post transformed data
# - Get the Scikit Learn Pipeline


# Transformed Dataset
?clf.get_config




# Extract Scikit learn Pipeline
pipeline = clf.get_config('pipeline')
pipeline



# Check difference in columns
#pipeline.fit_transform(df)


# 3.0 MACHINE LEARNING (COMPARE MODELS) ----

# Available Models
clf.models()

# Running All Available Models
best_models = clf.compare_models(
    sort = 'AUC',
    n_select = 10,
    budget_time =  2
)



# Get the grid
clf.pull()


# Top 3 Models
best_models
len(best_models)
best_models[0]

# Make predictions
clf.predict_model(best_models[0])

clf.predict_model(
    estimator = best_models[0],
    data = df.iloc[[0]]
    
)



# Refits on Full Dataset
best_model_0_finalized = clf.finalize_model(best_models[0])

# Save / load model

os.mkdir('models')
clf.save_model(
    model = best_model_0_finalized,
    model_name = 'models/best_model_0'
)
clf.load_model('models/best_model_0')

#%%
# 4.0 PLOTTING MODEL PERFORMANCE -----


# Get all plots 
# - Note that this can take a long time for certain plots
# - May want to just plot individual (see that next)

clf.evaluate_model(best_model_0_finalized)  
# - ROC Curves & PR Curves
clf.plot_model(best_model_0_finalized, plot = 'auc')

clf.plot_model(best_model_0_finalized, plot = 'pr')

# Confusion Matrix / Error
clf.plot_model(
    best_models[1],
    plot = 'confusion_matrix',
    plot_kwargs={'percent': True}
)

# Gain/Lift Plots
clf.plot_model(best_models[1], plot = 'gain')

clf.plot_model(best_models[1], plot = 'lift')

# Feature Importance
clf.plot_model(best_models[1], plot = 'feature')
clf.plot_model(best_models[0], plot = 'feature_all')

# Shows the Precision/Recall/F1
clf.plot_model(best_models[1], plot = 'class_report')

# Get model parameters used
clf.plot_model(best_models[1], plot = 'parameter')

#%%
# 5.0 CREATING & TUNING INDIVIDUAL MODELS ----
clf.models()


# Create more models
catboost_model = clf.create_model(
    estimator = 'catboost'
)

# Tuning Models
catboost_model_tuned = clf.tune_model(
    estimator = catboost_model,
    n_iter = 10,
    optimize = 'AUC')


# Save ada tuned
catboost_model_tuned_finalized = clf.finalize_model(catboost_model_tuned)
clf.save_model(
    model = catboost_model_tuned_finalized,
    model_name = 'models/catboost_model_tuned' 
)
    
clf.load_model('models/catboost_model_tuned')
#%%
# 6.0 INTERPRETING MODELS ----
# - SHAP Package Integration
?clf.interpret_model

print(best_models)
best_models[9]

# 1. Summary Plot: Overall top features

clf.interpret_model(catboost_model_tuned, plot = 'summary')
# 2. Analyze Specific Features ----

# Our Exploratory Function
els.explore_sales_by_category(
    leads_df, 
    'member_rating', 
    sort_by='prop_in_group'
)

# Correlation Plot
clf.interpret_model(
    catboost_model_tuned,
    plot = 'correlation',
    feature = 'optin_days'
)

# Partial Dependence Plot
clf.interpret_model(
    catboost_model_tuned,
    plot = 'pdp',
    feature = 'tag_count',
    ice = True

)

# 3. Analyze Individual Observations

leads_df.iloc[[0]]

# Shap Force Plot

#from pycaret.classification import transform_target, transform_categorical

clf.interpret_model(
    catboost_model_tuned,
    plot = 'reason',
    X_new_sample = leads_df.iloc[[1]]
)
clf.predict_model(
    catboost_model_tuned,
    leads_df.iloc[[9]]

)

# 7.0 BLENDING MODELS (ENSEMBLES) -----
blended_models_finalized = clf.blend_models(
    best_models,
    optimize = 'AUC'
)



# 8.0 CALIBRATION ----
# - Improves the probability scoring (makes the probability realistic)

calibrated_models = []
for model in blended_models_finalized.estimators_:  # Access the list of models
    calibrated_model = clf.calibrate_model(
        estimator=model,
        method='sigmoid',  # Choose your calibration method (e.g., 'isotonic')
        verbose=True,
        return_train_score=False
    )
    calibrated_models.append(calibrated_model)
# - Calibration Plot
clf.plot_model(calibrated_models[2], plot='calibration')

for model in calibrated_models:
    clf.plot_model(model, plot='calibration'
    )

clf.plot_model(blended_models_finalized, plot='calibration')

# Create a New ensemble calibrated model

final_ensemble = clf.blend_models(
    calibrated_models
)

clf.plot_model( final_ensemble, plot = 'calibration')

# 9.0 FINALIZE MODEL ----
# - Equivalent of refitting on full dataset
blended_models_final = clf.finalize_model(final_ensemble
    )

# 10.0 MAKING PREDICTIONS & RANKING LEADS ----

# Prediction
predictions_df = clf.predict_model(
    estimator = blended_models_final,
    data = leads_df
)

predictions_df.query("prediction_label == 1")

predictions_df
# Scoring
leads_scored_df = pd.concat([1-predictions_df['prediction_score'], leads_df], axis = 1)
leads_scored_df.sort_values('prediction_score', ascending = False)

# SAVING / LOADING PRODUCTION MODELS -----
clf.save_model(
    model = blended_models_final,
    model_name = 'models/blended_models_final'
)


# CONCLUSIONS ----

# - We now have an email lead scoring model
# - Pycaret simplifies the process of building, selecting, improving machine learning models
# - Scikit Learn would take 1000's of lines of code to do all of this



# %%
