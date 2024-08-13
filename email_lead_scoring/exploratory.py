import pandas as pd
import numpy as np

# Function: Explore Sales By Category  

def explore_sales_by_category(data, category = 'country_code', sort_by = ['sales', 'prop_in_group' ]):
    """AI is creating summary for explore_sales_by_category
    Eplore Sales by Category

    Args:
        data ([dataframe]): [sunscribers_joined_df dataframe generated from db_read_els_data() function]
        category (str, optional): [categorical column  from the subcribers_joined dataframe, that provides more information about subcribers ('coubtry_code, members_rating)]. Defaults to 'country_code'.
        sort_by (list, optional): [methods to sort the table either by sales_amount or Prop_in_group = proprtion of sales by Category]. Defaults to ['Sales', 'prop_in_group' ].

    Returns:
        [dataframe]: [returns a dataframe with the following columns: sales, prop_in_group, prop_overall, prop_cumsum]
    """
    
    # handle sort_by
    if type(sort_by) is list:
        sort_by = sort_by[0]
    
        
    # Data Manipulation
    
    ret = data \
        .groupby(category) \
        .agg(dict(made_purchase = ['sum', lambda x: sum(x) / len(x)])) \
        .set_axis(['sales', 'prop_in_group'], axis=1) \
        .assign(prop_overall = lambda x: x['sales'] / sum(x['sales'])) \
        .sort_values(by= sort_by, ascending=False) \
        .assign(prop_cumsum = lambda x: x['prop_overall'].cumsum() 
        ) 
        
    return ret


# Function: Explore Sales by Numeric Feature
def explore_sales_by_numeric(
    data,
    numeric = 'tag_count',
    q = [0.10, 0.50, 0.90]
):
    """_summary_: Exploring the subscriber data using the column 'made_purchase' and any numeric column(s).

    Args:
        data (_dataframe_): _description_: [sunscribers_joined_df dataframe generated from db_read_els_data() function]
        numeric (str or list, optional): _description_. Defaults to 'tag_count'.
        q (list, optional): _description_. Defaults to [0.10, 0.50, 0.90].

    Returns:
        _type_: _description_
    """
    if type(numeric) is list:
        feature_list = ['made_purchase', *numeric ]
    else:
        feature_list = ['made_purchase', numeric]
        
    ret = data[feature_list] \
        .groupby('made_purchase') \
        .quantile(q=q, numeric_only=True)
        
    return ret
