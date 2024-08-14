# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 0: MACHINE LEARNING & API'S JUMPSTART 
# PART 3: STREAMLIT
# ----

# To run app (put this in Terminal):
#   streamlit run 00_jumpstart/03_streamlit_jumpstart.py

#%%
# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px

# 1.0 Title and Introduction
st.title("Business Dashboard")
st.write("""
         This dashboard provides insights into sales, customer demographics, and product performance. upload your data to get started. """)

#%%
# 2.0 Data Input
st.header("Upload Business Data")

upload_file = st.file_uploader("Choose a csv file", type="csv",
                               accept_multiple_files = False)
st.header
#%%
# 3.0 App Body 
#  What Happens Once Data Is Loaded?
data = pd.read_csv("/Users/ellandalla/Desktop/Matt_D_Python/ds4b_2101_p/00_jumpstart/data/sales.csv") 
data.head()

if upload_file:
    data = pd.read_csv(upload_file)
    st.write("Preview of the Uploaded Data:")
    st.write(data.head())
# * Sales insights
st.header("Sales Insights")
if 'sales_date' in data.columns and 'sales_amount' in data.columns:
    st.write("Sales Over Time")
    fig = px.line(data, x='sales_date', y='sales_amount', title='Sales Over Time')
    st.plotly_chart(fig)
else: 
    st.write("Please ensure the data has 'sales_date' and 'sales_amount' columns for Viz purposes.")

    
# * Customer Segmentation by Region
st.header("Customer Segmentation")
if 'region' in data.columns and 'sales_amount' in data.columns:
    st.write("Customer Segmentation by Region")
    fig = px.pie(data, names='region', values='sales_amount', title='Customer Segmentation by Region')
    st.plotly_chart(fig)
else: 
    st.write("Please ensure the data has 'region' and 'sales_amount' columns for Viz purposes.")

# * Product Analysis
st.header("Product Analysis")
if 'product' in data.columns and 'sales_amount' in data.columns:
    st.write("Top Products by Sales")
    top_products_df = data.groupby('product').agg({'sales_amount':'sum'}).nlargest(10, 'sales_amount')
    fig = px.bar( top_products_df, x= top_products_df.index, y='sales_amount', title='Top Products by Sales')
    st.plotly_chart(fig)
else:
    st.warning("Please ensure the data has 'product' and 'sales_amount' columns for Viz purposes.")

# * Feedback Form
st.header("Feedback(Your Opinion Matters)")
feedback = st.text_area("Please provide feedback on the dashboard")
if st.button("Submit"):
    st.write("Thank you for your feedback! We appreciate it.")

#%%
# 4.0 Footer
st.write("---")
st.write("this business dashboard is flexible and can be customized to meet your business needs. For more information, please visit [Business Science](https://www.business-science.io/).")


if __name__ == "__main__":
    pass
# %%
