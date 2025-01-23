import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


def display_basic_info(data):
    
    # Dataset Shape
    st.header("Basic Information")
    st.write(f"**Shape of the dataset:** {data.shape[0]} rows, {data.shape[1]} columns")

    # Column names and Data Types
    st.write("**Column names and Data Types:**")
    dtypes_df = pd.DataFrame(data.dtypes).transpose()  # Convert to DataFrame and Transpose
    dtypes_df.index = ['dtype']  # Rename the index for clarity

    st.write( dtypes_df)

    # Missing Values
    missing_values = pd.DataFrame(data.isnull().sum()).transpose()
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


def display_summary_statistics(data):

    # Summary Statistics for Numerical Columns
    st.write("### Summary Statistics (Numerical Columns)")
    numerical_summary = data.describe().transpose()
    st.write(numerical_summary)

    # Summary Statistics for Categorical Columns
    st.write("### Summary Statistics (Categorical Columns)")
    categorical_columns = data.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        categorical_summary = data[categorical_columns].describe().transpose()
        st.write(categorical_summary)
    else:
        st.write("No categorical columns found.")


# Display correlation matrix
def display_correlation_matrix(data):
    st.header("Correlation Matrix")
    numerical_columns = data.select_dtypes(include=['number'])
    if not numerical_columns.empty:
        corr_matrix = numerical_columns.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numerical columns for correlation matrix.")


# Display pairplot
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


def display_scatterplot(data):
    st.header("Scatter Plot")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)  # Ensure it's a Python list
    
    # Check if there are at least two numerical columns
    if len(numerical_columns) > 1:
        # Dropdowns for selecting x and y axes
        x_col = st.selectbox("Select X-axis column", numerical_columns)
        y_col = st.selectbox("Select Y-axis column", numerical_columns)
        
        # Create scatter plot
        fig, ax = plt.subplots()
        ax.scatter(data[x_col], data[y_col], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical columns in the dataset to create a scatter plot.")


def display_histogram(data):
    st.header("Histogram")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    
    # Check if there are numerical columns
    if numerical_columns:
        # Dropdown for selecting the column
        selected_column = st.selectbox("Select column for histogram", numerical_columns)
        
        # Create histogram
        fig, ax = plt.subplots()
        ax.hist(data[selected_column], bins=20, alpha=0.7, color='blue')
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram: {selected_column}")
        st.pyplot(fig)
    else:
        st.warning("No numerical columns in the dataset to create a histogram.")



def display_boxplot(data):
    st.header("Box Plot")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    
    # Check if there are numerical columns
    if numerical_columns:
        # Dropdown for selecting the column
        selected_column = st.selectbox("Select column for box plot", numerical_columns)
        
        # Create box plot
        fig, ax = plt.subplots()
        ax.boxplot(data[selected_column].dropna(), vert=True, patch_artist=True)
        ax.set_ylabel(selected_column)
        ax.set_title(f"Box Plot: {selected_column}")
        st.pyplot(fig)
    else:
        st.warning("No numerical columns in the dataset to create a box plot.")


# Display outlier detection
def display_outlier_detection(data):
    st.header("Outlier Detection")
    # Identify numerical columns
    numerical_columns = data.select_dtypes(include=['number']).columns
    if len(numerical_columns) > 0:
        selected_columns = st.multiselect("Select columns for outlier detection", data.select_dtypes(include=['number']).columns)
        for col in selected_columns:
            st.write(f"Boxplot for `{col}`:")
            fig, ax = plt.subplots()
            sns.boxplot(x=data[col], ax=ax)
            st.pyplot(fig)
    else:
            st.write("No numerical columns for outlier detection.")
