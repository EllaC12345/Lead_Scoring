# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 1: DATA UNDERSTANDING & KPIS
# ----

# GOAL: ----
# - Saw high costs, feedback showed problems
# - Now need to work with departments to collect data and develop project KPIs

# LIBRARIES ----
#%%
# Data Analysis:
import pandas as pd
import numpy as np
import plotly.express as px

# New Libraries:
import sweetviz as sv
import sqlalchemy as sql

# Email Lead Scoring: 
import email_lead_scoring as els


# ?els.cost_calc_monthly_unsub_cost_table

els.cost_simulate_unsub_cost() 

els.cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=np.linspace(0, 0.03, 5),
    customer_conversion_rate=np.linspace(0.4, 0.6, 3),
    sales_emails_per_month=5,
    unsub_rate_per_sales_email=0.001,
    email_list_size=1e5
) \
    .pipe(func=els.cost_plot_simulated_unsub_costs)


# 1.0 CONNECTING TO SQLITE DATABASE ----
url = "sqlite:////Users/ellandalla/Desktop/Matt_D_Python/ds4b_2101_p/00_database/crm_database.sqlite"

engine = sql.create_engine(url)
engine
conn = engine.connect()
inspect = sql.inspect(engine)
table_names = inspect.get_table_names()
table_names


# 2.0 COLLECT DATA ----

# Products ----

#Products
table_names[0]

Products_df = pd.read_sql(
    sql = "Select * From Products",
    con = conn
)

Products_df.head()
Products_df.shape
Products_df.info()
Products_df['product_id']= Products_df['product_id'].astype('int')

Products_df.info()

# Subscribers ----
table_names[1]

subscribers_df = pd.read_sql(
    sql = "Select * From Subscribers",
    con = conn)

subscribers_df.head()
subscribers_df.info()
subscribers_df.shape

subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')
subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64[ns]')
subscribers_df
# Tags ----
table_names[2]
tags_df = pd.read_sql(
    sql = "Select * From Tags",
    con = conn
)
tags_df.head()
tags_df.info()
tags_df.shape

tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype('int')
# Transactions ----
table_names[3]
transactions_df = pd.read_sql(
    sql = "Select * From Transactions",
    con = conn
)
transactions_df.head()
transactions_df.info()
transactions_df.shape

transactions_df['product_id'] = transactions_df['product_id'].astype('int')
transactions_df['purchased_at'] = transactions_df['purchased_at'].astype('datetime64[ns]')


# Website ----
table_names[4]
website_df = pd.read_sql(
    sql = "Select * From Website",
    con = conn
)
website_df.head()
website_df.info()
website_df.shape

website_df['date'] = website_df['date'].astype('datetime64[ns]')
website_df['pageviews'] = website_df['pageviews'].astype('int')
website_df['organicsearches'] = website_df['organicsearches'].astype('int')
website_df['sessions'] = website_df['sessions'].astype('int')

# Close Connection ----
# - Note: a better practice is to use `with`
conn.close()

with engine.connect() as conn:
    website_df = pd.read_sql(
        sql = "Select * From Website",
        con = conn
    )
#*** Better Practices

# 3.0 COMBINE & ORGANIZE DATA ----
# - Problem is related to probability of purchase from email list
# - Need to understand what increases probability of purchase
# - Learning Labs could be a key event
# - Website data would be interesting but can't link it to email
# - Products really aren't important to our initial question - just want to know if they made a purchase or not and identify which are likely

# Make Target Feature
subscribers_df

emails_made_purchase = transactions_df['user_email'].unique()

subscribers_df['made_purchase'] = subscribers_df['user_email'] \
    .isin(emails_made_purchase) \
    .astype(int)

# Who is purchasing? --- Count od subscribers who made a purchase
count_made_purchase1 = subscribers_df['made_purchase'].value_counts()
count_made_purchase = subscribers_df['made_purchase'].sum()
count_made_purchase
count_made_purchase1
total_subscribers = len(subscribers_df['made_purchase'])

count_made_purchase/total_subscribers

# By Geographic Regions (Countries)
#1. group subscribers by Country code
#2.aggregate by sum and mean of made_purchase (sum and proportion of subscribers who made a purchase)
#3. (set_axis) add new columns for proportion of overall sales and cumulative sum of sales
#4 .assign(prop_overall = lambda x: x['sales'] / sum(x['sales'])): calculate proportion of sales in each group compared to  overall sales accross all groups
#5. sort by sales   
#6
by_geography_df = subscribers_df \
    .groupby('country_code') \
    .agg(dict(made_purchase = ['sum', lambda x: sum(x) / len(x)])) \
    .set_axis(['sales', 'prop_in_group'], axis=1) \
    .assign(prop_overall = lambda x: x['sales'] / sum(x['sales'])) \
    .sort_values(by='sales', ascending=False) \
    .assign(prop_cumsum = lambda x: x['prop_overall'].cumsum() 
    ) 

by_geography_df

# - Top 80% countries 
by_geography_df[by_geography_df['prop_cumsum'] <= 0.8]

by_geography_df \
    .query("prop_cumsum <= 0.80")


# - High Conversion Countries (>8% conversion)
by_geography_df \
    .query("prop_in_group > 0.08")

by_geography_df.quantile(q=[0.10,0.50,0.90])

by_geography_df.mean()
# By Tags (Events)

tags_df

user_events_df = tags_df \
    .groupby('mailchimp_id') \
    .agg(dict(tag = 'count')) \
    .set_axis(['tag_count'], axis=1) \
    .reset_index()
user_events_df

# merge rows from subscribers_df with user_events_df based on mailchimp_id
# subscribers that don't have events will have a tag_count of 0
subscribers_joined_df = subscribers_df.merge(
    right = user_events_df,
    how = 'left',
    left_on = 'mailchimp_id',
    right_on = 'mailchimp_id')\
    .fillna(dict(tag_count = 0))
    


# Analyzing Tag Count Proportion

subscribers_joined_df\
    .groupby('made_purchase')\
    . quantile(q=[0.10, 0.50, 0.90],
    numeric_only=True
               )

# 4.0 SWEETVIZ EDA REPORT
report = sv.analyze(
    subscribers_joined_df,
    target_feat='made_purchase'
    
)
report.show_html(
    filepath = '/Users/ellandalla/Desktop/Matt_D_Python/ds4b_2101_p/02_data_understanding/subscribers_eda_report.html'
)



# 5.0 DEVELOP KPI'S ----
# - Reduce unnecessary sales emails by 30% while maintaing 99% of sales
# - Segment list - apply sales (hot leads), nuture (cold leads)

# EVALUATE PERFORMANCE -----
subscribers_joined_df[['made_purchase', 'tag_count']]\
    .groupby('made_purchase')\
    .agg(
        mean_tag_count = ('tag_count', 'mean'),
        median_tag_count = ('tag_count', 'median'),
        count_subscribers = ('tag_count', 'count')
    )

    

# WHAT COULD BE MISSED?
# - More information on which tags are most important







# %%
