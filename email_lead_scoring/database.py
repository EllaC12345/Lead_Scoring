import pandas as pd
import numpy as np
import sqlalchemy as sql
from sqlalchemy import text

import re
import janitor as jn
from email_lead_scoring.exploratory import explore_sales_by_category



#import tables Names

def db_read_els_table_names(
    conn_str = "sqlite:///00_database/crm_database.sqlite"
):
    """AI is creating summary for db_read_els_table_names

    Args:
        conn_str (str, optional): [description]. Defaults to "sqlite:///00_database/crm_database.sqlite".

    Returns:
        [type]: List with table names
    """
    #connect to the engine
    engine = sql.create_engine(conn_str)
        
    # inspect the engine
    inspect = sql.inspect(engine)
    table_names = inspect.get_table_names()
    return table_names


## Pull Raw Data Per table

 #Get Raw Table data

def db_read_raw_els_table(
    table_name,
    conn_str = "sqlite:///00_database/crm_database.sqlite"
): 
    """AI is creating summary for db_read_raw_els_table:
    Function to read raw the table names on the crm database

    Args:
        table_name ([sql table]): [enter one of the table names from the crm database (Products, Subscribers, Tags, Transactions)]
        conn_str (str, optional): [connection string to the crm database]. Defaults to "sqlite:///00_database/crm_database.sqlite".

    Returns:
        [type]: pandas dataframe
    """
    #connect to the engine
    engine = sql.create_engine(conn_str)
    
    #Raw Data Collect
    with engine.connect() as conn:
        df = pd.read_sql(
            sql.text(f"SELECT * FROM {table_name}"),  # Use sql.text for the SQL statement
            con=conn
        )
    
    return df

# db_read_raw_els_table('Website')
df = db_read_raw_els_table('Website')


# Read & Combine Raw Data
def db_read_els_data(
    conn_str = 'sqlite:///00_database/crm_database.sqlite'
): 
    """AI is creating summary for db_read_els_data
    Function to read and combine raw data from various tables on the crm database

    Args:
        conn_str (str, optional): [description]. Defaults to 'sqlite:///00_database/crm_database.sqlite'.

    Returns:
        [pandas]: [dataframe]
    """
    
    #connect to the engine
    engine = sql.create_engine(conn_str)
    
    #Raw Data Collect
    with engine.connect() as conn:
        #Subscribers
        subscribers_df = pd.read_sql(
            sql=text("SELECT * FROM Subscribers"),
            con = conn
        )
        
        subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')
        subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64[ns]')
        subscribers_df['mailchimp_id'] = subscribers_df['mailchimp_id'].astype('int')
       
        #tags
        tags_df = pd.read_sql(
            sql=text("SELECT * FROM Tags"),
            con = conn
        )
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype('int')
    
        # Transactions
        transactions_df = pd.read_sql(
            sql=text("SELECT * FROM Transactions"),
            con = conn
        )
     
        # Merge Tag Counts
        user_events_df = tags_df \
            .groupby('mailchimp_id') \
            .agg(dict(tag = 'count')) \
            .set_axis(['tag_count'], axis=1) \
            .reset_index()
    
        subscribers_joined_df = subscribers_df.merge(
            right = user_events_df,
            how = 'left',
            left_on = 'mailchimp_id',
            right_on = 'mailchimp_id')\
                .fillna(dict(tag_count = 0))
        subscribers_joined_df['tag_count'] = subscribers_joined_df['tag_count'].astype('int')
        
        # Merge Taget Variables
        emails_made_purchase = transactions_df['user_email'].unique()
        subscribers_joined_df['made_purchase'] = subscribers_joined_df['user_email'] \
            .isin(emails_made_purchase) \
            .astype('int')
    return subscribers_joined_df    


## Process Data
def process_leads_tags(leads_df, tags_df):
    """AI is creating summary for process_leads_tags
    **Function to process the leads and tags data prepares for Machine Learnings Models ***

    Args:
        leads_df (dataframe): returns the output from the els.db_read_els_data() function
        tags_df (dataframe): returs the output from the els.db_read_raw_els_table('tags') function

    Returns:
        [dataframe]: leads and tags data combined and processed for Machine Learning Models
    """
    
    
    # Date Features
    date_max = leads_df['optin_time'].max()
    leads_df['optin_days'] = (leads_df['optin_time'] - date_max).dt.days
    
    # Email Features 
    leads_df['email_provider'] = leads_df['user_email'].map(lambda x: x.split('@')[1])
    
    # Activity Features
    leads_df['tag_count_by_optin_day'] = leads_df['tag_count'] / abs(leads_df['optin_days'] - 1)
    
    # Specific Tag Features 
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
    # Merge Tags
    tags_wide_leads_df.columns = tags_wide_leads_df.columns \
    . to_series() \
    .apply(func = lambda x: f'tag_{x}') \
    . to_list()

    ## remove mailchimp_id from index and reset index
    tags_wide_leads_df = tags_wide_leads_df.reset_index()


    # when merging without specifications, the data will be merged on the common column names
    leads_tags_df  = leads_df \
    .merge(tags_wide_leads_df, how = 'left')   
    
    # Fill NA Selectively
    # - Fill NA for columns that match a regular expression
    def fillna_regex(data, regex, value = 0, **kwargs):
        """Fill NA for columns that match a regular expression"""
        
        for col in data.columns:
            if re.match(pattern = regex, string = col):
                
                data[col] = data[col].fillna(value= value, **kwargs)
        return data 

    leads_tags_df = fillna_regex(leads_tags_df, regex = 'tag_', value = 0)
            
    import email_lead_scoring as els
    
    # High Cardinality Features: Country Code
    countries_to_keep = explore_sales_by_category(
        data = leads_tags_df,
        category = 'country_code'
    ) \
        .query('sales >= 6') \
        .index \
        .to_list()

    leads_tags_df['country_code'] = leads_tags_df['country_code'] \
    .apply(lambda x: x if x in countries_to_keep else 'Other')
    
    return leads_tags_df

# Final Pipeline

def db_read_and_process_els_data(
    conn_str = 'sqlite:///00_database/crm_database.sqlite'
    
):
    leads_df = db_read_els_data(conn_str = conn_str)
    
    tags_df = db_read_raw_els_table(
        conn_str = conn_str,
        table_name = 'Tags'
    )
    
    df = process_leads_tags(leads_df, tags_df)
    
    return df 
