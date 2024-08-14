# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 3: DATA PROCESSING PIPELINE 
# ----

# LIBRARIES ----
#%%
# Core
import pandas as pd
import numpy as np
# EDA 
import re
import janitor as jn
import sweetviz as sv
# Email Lead Scoring
import email_lead_scoring as els

# Recap  ---
# - We see some promise in the data
# - Now we need to continue to develop features
# - Prepare for Machine Learning

leads_df = els.db_read_els_data()
leads_df
# 1.0 FEATURE EXPLORATION ----

# High Cardinality Features: Country Code
# higj_cardinality = a categorical feature with a large number of unique values. Solution is to group by and aggregate
#example high cardinality feature: Name
els.explore_sales_by_category(
    data = leads_df,
    category = 'country_code',
    sort_by = 'sales'
) \
.query('sales > 5') 


# Ordinal Feature
els.explore_sales_by_category(
    data = leads_df,
    category = 'member_rating',
    sort_by = 'sales'
)

# Interaction

els.explore_sales_by_numeric(
    data = leads_df,
    numeric = [ 'tag_count', 'member_rating']
)

# 2.0 BUILDING ENGINEERED FEATURES ----

# Date Features
date_max = leads_df['optin_time'].max()

date_min = leads_df['optin_time'].min()

time_range = date_max - date_min

time_range.days

leads_df['optin_days'] = (leads_df['optin_time'] - date_max).dt.days

leads_df.head()


# Email Features

leads_df['user_email']
'garrick.langworth@gmail.com'.split('@')[1]


leads_df['email_provider'] = leads_df['user_email'] \
    .apply(lambda x: x.split('@')[1])

leads_df
# Activity Features
leads_df['tag_count_by_optin_day'] = leads_df['tag_count'] / abs(leads_df['optin_days'] - 1)

leads_df['tag_count_by_optin_day']

# 3.0 ONE-TO-MANY FEATURES  ----
# - TAGS: 1 Customer can have many tags
# Specific Tag Features (Actions)
els.db_read_els_table_names()
tags_df = els.db_read_raw_els_table('tags')
tags_df

tags_wide_leads_df = tags_df \
    .assign(value = lambda x: 1) \
    . pivot_table(
        index = 'mailchimp_id',
        columns = 'tag',
        values = 'value',
    ) \
    . fillna(value = 0) \
    . pipe (
        func = jn.clean_names
    )
tags_wide_leads_df


# Merge Tags
## transfroms columns to series object, assign prefix Tag to each column using lambda functio, and converts to list
## the replacement will be automatically be made to the dataframe
tags_wide_leads_df.columns = tags_wide_leads_df.columns \
    . to_series() \
    .apply(func = lambda x: f'tag_{x}') \
    . to_list()

## remove mailchimp_id from index and reset index
tags_wide_leads_df = tags_wide_leads_df.reset_index()


# when merging without specifications, the data will be merged on the common column names
leads_tags_df  = leads_df \
    .merge(tags_wide_leads_df, how = 'left')

data = leads_tags_df

data
# Fill NA selectively
# - Fill NA for columns that match a regular expression
def fillna_regex(data, regex, value = 0, **kwargs):
    """Fill NA for columns that match a regular expression"""
    
    for col in data.columns:
        if re.match(pattern = regex, string = col):
            
            data[col] = data[col].fillna(value= value, **kwargs)
    return data 

leads_tags_df = fillna_regex(leads_tags_df, regex = 'tag_', value = 0)

leads_tags_df

# 4.0 ADJUSTING FEATURES ----
# - Country Code: Has high cardinality

# High Cardinality Features: Country Code
countries_to_keep = els.explore_sales_by_category(
    data = leads_tags_df,
    category = 'country_code'
) \
.query('sales >= 6') \
.index \
.to_list()

countries_to_keep

leads_tags_df
leads_tags_df['country_code'] = leads_tags_df['country_code'] \
    .apply(lambda x: x if x in countries_to_keep else 'Other')


leads_tags_df
# 5.0 EXPLORATORY REPORT PART 2

# Examine with Sweetviz
report = sv.analyze(
    source = leads_tags_df,
    target_feat = 'made_purchase',
    feat_cfg = sv.FeatureConfig(skip = ['mailchimp_id', 'user_email', 'optin_time', 'user_full_name'],
                                force_cat= ['member_rating', 'country_code', 'email_provider'])
    
)

report.show_html(
    filepath = "/Users/ellandalla/Desktop/Matt_D_Python/ds4b_2101_p/03_data_pipelines/subscriber_report_pipeline.html"
)
# CONCLUSIONS ----

# - Member Rating: 5 has 14% of purchasers within group
# - Country Code: US getting 11% conversion, AU getting 12%
# - Tag Count: 
#    - At zero tags, almost no chance of purchase
#    - At 5 tags, about 10% chance of purchase
#    - At 10 tags, about 35% chance of purchase
# - Optin Days: After 100 days, increases to 2% to 4-5% likelihood
# - Ratio of Tag Counts to Length of Time: Goes from zero to 15% at 0.10 (1 tag for every 10 days on list)
# - Attending Learning Labs & Webinars: Increases from 2% to 10%




# %%
