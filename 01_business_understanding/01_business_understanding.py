# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----

#%%
# LIBRARIES ----x

import pandas as pd
import numpy as np
import janitor as jn
import plotly.express as px



# BUSINESS SCIENCE PROBLEM FRAMEWORK ----

# View Business as a Machine ----


# Business Units: 
#   - Marketing Department
#   - Responsible for sales emails  
# Project Objectives:
#   - Target Subscribers Likely To Purchase
#   - Nurture Subscribers to take actions that are known to increase probability of purchase
# Define Machine:
#   - Marketing sends out email blasts to everyone
#   - Generates Sales
#   - Also ticks some members off
#   - Members unsubscribe, this reduces email growth and profitability
# Collect Outcomes:
#   - Revenue has slowed, Email growth has slowed


# Understand the Drivers ----

#   - Key Insights:
#     - Company has Large Email List: 100,000 
#     - Email list is growing at 6,000/month less 3500 unsub for total of 2500
#     - High unsubscribe rates: 500 people per sales email
#   - Revenue:
#     - Company sales cycle is generating about $250,000 per month
#     - Average Customer Lifetime Value: Estimate $2000/customer
#   - Costs: 
#     - Marketing sends 5 Sales Emails Per Month
#     - 5% of lost customers likely to convert if nutured


#%%
# COLLECT OUTCOMES ----

email_list_size_1 = 100000
unsub_count_per_sales_email_1 = 500

unsub_rate_1 = unsub_count_per_sales_email_1 / email_list_size_1

unsub_rate_1

sales_emails_per_month_1 = 5
conversion_rate_1 = 0.05

lost_customer_1 = email_list_size_1 * unsub_rate_1 * conversion_rate_1 
lost_customer_1

average_customer_value_1 = 2000

lost_revenue_per_month_1 = lost_customer_1 * average_customer_value_1
lost_revenue_per_month_1
# No-growth scenario $3M cost
cost_no_growth_1 = lost_revenue_per_month_1 * 12
cost_no_growth_1

# Growth scenario
# amount = principle *((1+rate)**time)

growth_rate_1 = 3500/100000

100000 *((1+growth_rate_1)**0)

100000 *((1+growth_rate_1)**1)
100000 *((1+growth_rate_1)**6)
100000 *((1+growth_rate_1)**11)

# Cost Table
time = 12

#Period
period_series = pd.Series(np.arange(0,12), name = "Period")
len(period_series)
period_series
cost_table_df = period_series.to_frame()

# Email Size - No Growth
cost_table_df["Email_List_Size_No_Growth"] = np.repeat(email_list_size_1, time)
cost_table_df

# Lost Customers - No Growth
cost_table_df["Lost_Customers_No_Growth"] = cost_table_df['Email_List_Size_No_Growth'] * unsub_rate_1 * sales_emails_per_month_1 * conversion_rate_1
cost_table_df

# Lost Revenue - No Growth
cost_table_df["Lost_Revenue_No_Growth"] = cost_table_df['Lost_Customers_No_Growth'] * average_customer_value_1
cost_table_df


#Email List Size - Growth
cost_table_df["Email_List_Size_Growth"] = cost_table_df['Email_List_Size_No_Growth'] * ((1+growth_rate_1)**cost_table_df['Period'])
cost_table_df

px.line(
    data_frame = cost_table_df,
    y = ["Email_List_Size_No_Growth", "Email_List_Size_Growth"]
) \
.add_hline(y=0)

# Cost - With Growth ( cost due to Lost Customers)
cost_table_df["Cost_with_growth"] = cost_table_df['Email_List_Size_Growth'] * average_customer_value_1
cost_table_df

px.line(
    cost_table_df,
    y = ["Lost_Revenue_No_Growth", "Cost_with_growth"]
)\
.add_hline(y=0)

# Compare Cost - with/No Growth
cost_table_df[['Lost_Revenue_No_Growth', 'Cost_with_growth']].sum()

# If reduce unsubscribe rate by 30%
cost_table_df['Lost_Revenue_No_Growth'].sum() * 0.30
cost_table_df['Cost_with_growth'].sum() * 0.30

# 2.5% growth scenario: 
#   amount = principle * ((1+rate)**time)




# COST CALCULATION FUNCTIONS ----

# Function: Calculate Monthly Unsubscriber Cost Table ----

def cost_calc_monthly_cost_table(
    email_list_size = 1e5,
    email_list_growth_rate = 0.035,
    sales_emails_per_month = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate = 0.05,
    average_customer_value = 2000,
    n_periods = 12
):
    # PERIOD
    period_series = pd.Series(np.arange(0, n_periods), name = "Period") 
    cost_table2_df = period_series.to_frame()
    
    # Email Size - No Growth
    cost_table2_df["Email_List_Size_No_Growth"] = np.repeat(email_list_size, n_periods)
    
    # Lost Customers - No Growth
    cost_table2_df["Lost_Customers_No_Growth"] = cost_table2_df['Email_List_Size_No_Growth'] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate
    
    # Cost - No Growth
    cost_table2_df["Cost_No_Growth"] = cost_table2_df['Lost_Customers_No_Growth'] * average_customer_value * customer_conversion_rate
    
    # Email List Size - Growth
    cost_table2_df["Email_List_Size_Growth"] = cost_table2_df['Email_List_Size_No_Growth'] * ((1 + email_list_growth_rate)**cost_table2_df['Period']) 

    # Lost Customers - Growth
    cost_table2_df["Lost_Customers_Growth"] = cost_table2_df['Email_List_Size_Growth'] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate
    
    # Cost - With Growth
    cost_table2_df["Cost_With_Growth"] = cost_table2_df['Lost_Customers_Growth'] * average_customer_value * customer_conversion_rate
    
    return cost_table2_df

cost_table2_df = cost_calc_monthly_cost_table(
    email_list_size = 50000,
    sales_emails_per_month = 1,
    unsub_rate_per_sales_email = 0.001,
    n_periods = 24,
    #average_customer_value = 1500
    

)
print(cost_table2_df)
# Function: Summarize Cost ----
   
cost_table2_df[["Cost_No_Growth", "Cost_With_Growth"]] \
    .sum()\
    .to_frame() \
    .transpose()


def cost_total_unsub_cost(cost_table2_df):
    summary_df =  cost_table2_df[["Cost_No_Growth", "Cost_With_Growth"]] \
        .sum()\
        .to_frame() \
        .transpose()
    return summary_df

cost_total_unsub_cost(cost_table2_df)

# ARE OBJECTIVES BEING MET?
# - We can see a large cost due to unsubscription
# - However, some attributes may vary causing costs to change


# SYNTHESIZE OUTCOMES (COST SIMULATION) ----
# - Make a cartesian product of inputs that can vary
# - Use list comprehension to perform simulation
# - Visualize results

# Cartesian Product (Expand Grid)
?jn.expand_grid
#np.linspace(): the function returns an array of evenly spaced numbers over a specified interval( below example returns 10 numbers between 0 and 0.05)

data_dict = dict(
    email_list_monthly_growth_rate = np.linspace(0, 0.05, num =10),
    customer_conversion_rate = np.linspace(0.04, 0.06, num = 3)
)

parameter_grid_df = jn.expand_grid(others=data_dict)\
    .droplevel(level = 1, axis = 1)
parameter_grid_df


# List Comprehension (Simulate Costs)
def temporary_function(x,y):
    cost_table = cost_calc_monthly_cost_table(
        email_list_growth_rate = x,
        customer_conversion_rate = y
    )
    cost_summary_df = cost_total_unsub_cost(cost_table2_df)
    return cost_summary_df
temporary_function(0.10, y=0.10)

summary_list = [temporary_function(x, y) for x, y in zip(parameter_grid_df['email_list_monthly_growth_rate'], parameter_grid_df['customer_conversion_rate'])]

summary_list

simulation_results_df = pd.concat(summary_list, axis=0)\
    .reset_index() \
    .drop("index", axis = 1)\
    .merge(parameter_grid_df.reset_index(), left_index = True, right_index = True)
    
simulation_results_df

# Function
def cost_simulate_unsub_cost(
    email_list_monthly_growth_rate = [0,0.035],
    customer_conversion_rate = [0.04, 0.05,0.06],
    **kwargs
):
    
    #Parameter Grid
    data_dict = dict(
    email_list_monthly_growth_rate = email_list_monthly_growth_rate,
    customer_conversion_rate = customer_conversion_rate)

    parameter_grid_df = jn.expand_grid(others=data_dict)\
    .droplevel(level = 1, axis = 1)
    
    
    # Temporary Function
    # List Comprehension (Simulate Costs)
    def temporary_function(x,y):
        cost_table_df = cost_calc_monthly_cost_table(
            email_list_growth_rate = x,
            customer_conversion_rate = y,
            **kwargs
        )
        cost_summary_df = cost_total_unsub_cost(cost_table_df)
        return cost_summary_df
    
    # List Comprehension (Simulate Costs)

    summary_list = [temporary_function(x, y) for x, y in zip(parameter_grid_df['email_list_monthly_growth_rate'], parameter_grid_df['customer_conversion_rate'])]

    simulation_results_df = pd.concat(summary_list, axis=0)\
        .reset_index() \
        .drop("index", axis = 1)\
        .merge(parameter_grid_df.reset_index(), left_index = True, right_index = True)
    
    return simulation_results_df

cost_simulate_unsub_cost()

# VISUALIZE COSTS

simulation_results_wide_df = cost_simulate_unsub_cost(
    email_list_monthly_growth_rate = [0.01, 0.02],
    customer_conversion_rate = [0.04, 0.06],
    email_list_size = 100000)\
    .drop('Cost_No_Growth', axis = 1)\
    .pivot(
        index = 'email_list_monthly_growth_rate',
        columns = 'customer_conversion_rate',
        values = 'Cost_With_Growth'
    )

?px.imshow

px.imshow(
    simulation_results_wide_df,
    origin='lower',
    aspect='auto',
    title='Lead Cost Simulation',
    labels=dict(
        x ='Email List Growth Rate',
        y ='Customer Conversion Rate',
        color='Cost with Growth'
    )   
)
# Function: Plot Simulated Unsubscriber Costs
def cost_plot_simulated_unsub_costs(simulation_results):
    simulation_results_wide_df = simulation_results\
        .drop('Cost_No_Growth', axis = 1)\
        .pivot(
            index = 'email_list_monthly_growth_rate',
            columns = 'customer_conversion_rate',
            values = 'Cost_With_Growth'
        )
    
    fig = px.imshow(
        simulation_results_wide_df,
        origin='lower',
        aspect='auto',
        title='Lead Cost Simulation',
        labels=dict(
            x ='Customer Conversion Rate',
            y ='Monthly Email List Growth Rate',
            color='Cost of Unsubscription'
        )   
    )
    
    return fig

cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0.01, 0.02, 0.03],
    customer_conversion_rate=[0.04, 0.05, 0.06],
    email_list_size=100000)\
. pipe(cost_plot_simulated_unsub_costs)



# ARE OBJECTIVES BEING MET?
# - Even with simulation, we see high costs
# - What if we could reduce by 30% through better targeting?



# - What if we could reduce unsubscribe rate from 0.5% to 0.17% (marketing average)?
# - Source: https://www.campaignmonitor.com/resources/knowledge-base/what-is-a-good-unsubscribe-rate/



# HYPOTHESIZE DRIVERS

# - What causes a customer to convert of drop off?
# - If we know what makes them likely to convert, we can target the ones that are unlikely to nurture them (instead of sending sales emails)
# - Meet with Marketing Team
# - Notice increases in sales after webinars (called Learning Labs)
# - Next: Begin Data Collection and Understanding




# %%
