# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 2: DATA IMPORT FUNCTIONS
# ----

# LIBRARIES ----
#%%
import pandas as pd
import numpy as np
import sqlalchemy as sql


# IMPORT RAW DATA ----

# Read & Combine Raw Data
def db_read_els_data(
    conn_str = 'sqlite:////Users/ellandalla/Desktop/Matt_D_Python/ds4b_2101_p/00_database/crm_database.sqlite'
):
    
    #connect to the engine
    engine = sql.create_engine(conn_str)
    
    #Raw Data Collect
    with engine.connect() as conn:
        #Subscribers
        subscribers_df = pd.read_sql(
            sql = "Select * From Subscribers",
            con = conn)
        
        subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')
        subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64[ns]')
        subscribers_df['mailchimp_id'] = subscribers_df['mailchimp_id'].astype('int')
       
        #tags
        tags_df = pd.read_sql(
            sql = "Select * From Tags",
            con = conn
        )
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype('int')
    
        # Transactions
        transactions_df = pd.read_sql(
            sql = "Select * From Transactions",
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
        
        # Merge Target Variables
        emails_made_purchase = transactions_df['user_email'].unique()
        subscribers_joined_df['made_purchase'] = subscribers_joined_df['user_email'] \
            .isin(emails_made_purchase) \
            .astype('int')
    return subscribers_joined_df


# Read Table Names
db_read_els_data().info() 

#%%
def db_read_els_table_names(
    conn_str = "sqlite:///00_database/crm_database.sqlite"
):
    #connect to the engine
    engine = sql.create_engine(conn_str)
        
    # inspect the engine
    inspect = sql.inspect(engine)
    table_names = inspect.get_table_names()
    return table_names

db_read_els_table_names()
# Get Raw Table data

def db_read_raw_els_table(
    table_name,
    conn_str = "sqlite:///00_database/crm_database.sqlite"
):
    #connect to the engine
    engine = sql.create_engine(conn_str)
    
    #Raw Data Collect
    with engine.connect() as conn:
        df = pd.read_sql(
            sql = f"Select * From {table_name}",
            con = conn)
    return df

db_read_raw_els_table('Website')


# TEST IT OUT -----




# %%
import email_lead_scoring as els

els.db_read_els_table_names()
els.db_read_els_data()
els.db_read_raw_els_table('Website')
# %%
