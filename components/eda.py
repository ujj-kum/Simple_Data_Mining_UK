import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def display_basic_info(data):
    
    # Dataset Shape
    st.write(f"**Shape of the dataset:** {data.shape[0]} rows, {data.shape[1]} columns")

    # Column names and Data Types
    st.write("**Column names and Data Types:**")
    dtypes_df = pd.DataFrame(data.dtypes).T  # Convert to DataFrame and Transpose
    dtypes_df.index = ['dtype']  # Rename the index for clarity

    st.write( dtypes_df)

    # Missing Values
    missing_values = pd.DataFrame(data.isnull().sum()).T
    missing_values.index = ['Missing Values']
    st.write("**Missing Values Per Column:**")
    st.write(missing_values[missing_values > 0])

    # Duplicate Rows
    duplicates = data.duplicated().sum()
    st.write(f"**Number of Duplicate Rows:** {duplicates}")

    # Unique Values
    uniqe_data = pd.DataFrame(data.nunique()).T
    uniqe_data.index = ["Unique Data"]
    st.write("**Unique Values Per Column:**")
    st.write(uniqe_data)

def display_pairplot(data):
    st.header("Pairplot")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)  # Ensure it's a Python list
    
    # Check if there are at least two numerical columns
    if len(numerical_columns) > 1:
        # Multiselect widget for column selection
        selected_columns = st.multiselect(
            "Select columns for pairplot",
            numerical_columns,
            default=numerical_columns[:2]  # Default to the first two numerical columns
        )
        
        # Check if user has selected columns
        if selected_columns:
            # Create pairplot
            fig = sns.pairplot(data[selected_columns])
            st.pyplot(fig)
        else:
            st.warning("Please select at least one numerical column for the pairplot.")
    else:
        st.warning("Not enough numerical columns in the dataset to create a pairplot.")

def display_outlier_detection(data):

    # Identify numerical columns
    numerical_columns = data.select_dtypes(include=['number']).columns

    selected_columns = st.multiselect("Select columns for outlier detection", data.select_dtypes(include=['number']).columns)
    for col in selected_columns:
        st.write(f"Boxplot for `{col}`:")
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        st.pyplot(fig)
