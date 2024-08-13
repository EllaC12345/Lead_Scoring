import pandas as pd
import numpy as np
import janitor as jn
import plotly.express as px
import pandas_flavor as pf



@pf.register_dataframe_method
def cost_calc_monthly_cost_table(
    email_list_size = 1e5,
    email_list_growth_rate = 0.035,
    sales_emails_per_month = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate = 0.05,
    average_customer_value = 2000,
    n_periods = 12
):
    """AI is creating summary for cost_calc_monthly_cost_table

    Args:
        email_list_size ([type], optional): [email list size]. Defaults to 1e5.
        email_list_growth_rate (float, optional): [monthly email list growth rate]. Defaults to 0.035.
        sales_emails_per_month (int, optional): [sales emails per month]. Defaults to 5.
        unsub_rate_per_sales_email (float, optional): [unsubscription rate per email]. Defaults to 0.005.
        customer_conversion_rate (float, optional): [Rate of email subscribers that convert to customers]. Defaults to 0.05.
        average_customer_value (int, optional): [Average Customer Value]. Defaults to 2000.
        n_periods (int, optional): [Number of Months for our cost Table ]. Defaults to 12.

    Returns:
        [type]: [Dataframe: a cost table with the following columns: Period, Email_List_Size_No_Growth, Lost_Customers_No_Growth, Cost_No_Growth, Email_List_Size_Growth, Lost_Customers_Growth, Cost_With_Growth]
    """
    # PERIOD
    period_series = pd.Series(np.arange(0, n_periods), name = "Period") 
    cost_table2_df = period_series.to_frame()
    
    # Email Size - No Growth
    cost_table2_df["Email_List_Size_No_Growth"] = np.repeat(email_list_size, n_periods)
    
    # Lost Customers - No Growth
    cost_table2_df["Lost_Customers_No_Growth"] = cost_table2_df['Email_List_Size_No_Growth'] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate
    
    # Cost - No Growth
    cost_table2_df["Cost_No_Growth"] = cost_table2_df['Lost_Customers_No_Growth'] * average_customer_value
    
    # Email List Size - Growth
    cost_table2_df["Email_List_Size_Growth"] = cost_table2_df['Email_List_Size_No_Growth'] * ((1 + email_list_growth_rate)**cost_table2_df['Period']) 

    # Lost Customers - Growth
    cost_table2_df["Lost_Customers_Growth"] = cost_table2_df['Email_List_Size_Growth'] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate
    
    # Cost - With Growth
    cost_table2_df["Cost_With_Growth"] = cost_table2_df['Lost_Customers_Growth'] * average_customer_value 
    
    return cost_table2_df



def cost_total_unsub_cost(cost_table2_df):
    """Takes input from cost_calc_monthly_cost_table() and produces a summary of the total costs

    Args:
        cost_table2_df ([dataframe]): [output from cost_calc_monthly_cost_table()]

    Returns:
        [dataframe]: [a summarized total cost of email unsubscriptions]
    """
    summary_df =  cost_table2_df[["Cost_No_Growth", "Cost_With_Growth"]] \
        .sum()\
        .to_frame() \
        .transpose()
    return summary_df


def cost_simulate_unsub_cost(
    email_list_monthly_growth_rate = [0,0.035],
    customer_conversion_rate = [0.04, 0.05,0.06],
    **kwargs
):
    """AI is creating summary for cost_simulate_unsub_cost
    Generate a cost simulation for different email list growth rates and customer conversion rates to simulate cost uncertainty.add()

    Args:
        email_list_monthly_growth_rate (list, optional): [list of values for email monthly growth rate to simulate  ]. Defaults to [0,0.035].
        customer_conversion_rate (list, optional): [list of values for customer conversion rate to simulate]. Defaults to [0.04, 0.05,0.06].

    Returns:
        [Dataframe]: [cartesian product of email_list_monthly_growth_rate and customer_conversion_rate with total unsubscribers cost simulation results]
    """
    
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


@pf.register_series_method
def cost_plot_simulated_unsub_costs(simulation_results):
    """AI is creating summary for cost_plot_simulated_unsub_costs:
    # Plot the simulation results  

    Args:
        simulation_results ([DataFrame]): [the output from cost_simulate_unsub_cost()]

    Returns:
        [plotly]: [Heatmap of the simulation results]
    """
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

