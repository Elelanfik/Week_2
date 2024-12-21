import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_univariate(data, column, title=None):
    plt.hist(data[column], bins=30, color='blue', alpha=0.6)
    plt.title(title if title else f"Univariate Analysis of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

# Function to plot bivariate analysis (assumed to be defined already)
def plot_bivariate(data, x_column, y_column, title=None):
    plt.scatter(data[x_column], data[y_column], color='brown', alpha=0.6)
    plt.title(title if title else f"Bivariate Analysis: {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)

def plot_column_frequency(data, column=None, top_n=10):
  
    top_values = data[column].value_counts().head(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_values.values, y=top_values.index)

    # Customize the plot
    plt.title(f'Top {top_n} Most Frequent {column} Types')
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.show()

def plot_histograms(data, numerical_columns, columns_per_plot=9):
    
    # Filter the numerical columns
    num_data = data[numerical_columns]
    
    # Calculate the number of plots needed
    num_plots = int(np.ceil(len(num_data.columns) / columns_per_plot))
    
    for i in range(num_plots):
        # Get the subset of columns for this figure
        subset_cols = num_data.iloc[:, i * columns_per_plot:(i + 1) * columns_per_plot]
        
        # Create a grid of histograms
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # 3x3 grid
        axes = axes.flatten()  # Flatten axes for easier iteration
        
        for j, col in enumerate(subset_cols.columns):
            ax = axes[j]
            subset_cols[col].hist(ax=ax, bins=20, edgecolor="blue")
            ax.set_title(col)
            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
        
        # Remove unused subplots
        for j in range(len(subset_cols.columns), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

def plot_graphical_analysis(data):
    # Identify continuous columns
    continuous_columns = data.select_dtypes(include=['number']).columns

    # Set up the plot style
    sns.set(style="whitegrid")
    
    # Plot histograms and boxplots side by side for continuous variables
    for col in continuous_columns:
        # Create a 1x2 grid for the plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram plot on the first axis
        sns.histplot(data[col], kde=True, bins=30, color='blue', edgecolor='black', ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")
        
        # Boxplot on the second axis
        sns.boxplot(x=data[col], color='green', ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
        axes[1].set_xlabel(col)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    # Violin plots for continuous variables (to see distribution across categories)
    for col in continuous_columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=data[col], color='purple')
        plt.title(f"Violin plot of {col}")
        plt.xlabel(col)
        plt.show()

def bivariate_analysis(data_tel):
    # Ensure data_tel has the necessary columns
    if 'Total DL (Bytes)' not in data_tel or 'Total UL (Bytes)' not in data_tel:
        raise ValueError("Dataset must contain 'Total DL (Bytes)' and 'Total UL (Bytes)' columns")
    
    # Calculate the total data (DL + UL)
    data_tel['Total Data'] = data_tel['Total DL (Bytes)'] + data_tel['Total UL (Bytes)']
    
    # Identify categorical columns
    categorical_columns = data_tel.select_dtypes(include=['object', 'category']).columns
    
    # Analyze categorical data
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=col, y='Total Data', data=data_tel, palette='viridis')
        plt.title(f"Total Data Usage by {col}")
        plt.xlabel(col)
        plt.ylabel("Total Data (DL + UL)")
        plt.xticks(rotation=45)
        plt.show()

        # Boxplot for categorical variables
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y='Total Data', data=data_tel, palette='Set2')
        plt.title(f"Boxplot of Total Data Usage by {col}")
        plt.xlabel(col)
        plt.ylabel("Total Data (DL + UL)")
        plt.xticks(rotation=45)
        plt.show()

    # Analyze continuous data
    continuous_columns = data_tel.select_dtypes(include=['number']).columns
    for col in continuous_columns:
        if col not in ['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']:
            continue
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data_tel[col], y=data_tel['Total Data'], palette="deep")
        plt.title(f"Scatter Plot: {col} vs Total Data Usage")
        plt.xlabel(col)
        plt.ylabel("Total Data (DL + UL)")
        plt.show()


def plot_correlation_matrix(data):
    # Define the columns for which we want to compute correlation
    columns_of_interest = ['Social Media data', 'Google data', 'Email data', 
                           'YouTube data', 'Netflix data', 'Gaming data', 'Other data']
    
    # Ensure that the specified columns exist in the dataset
    data_subset = data[columns_of_interest]
    
    # Compute the correlation matrix
    corr_matrix = data_subset.corr()

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=0.5, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
    plt.title("Correlation Matrix of Data Variables")
    plt.show()   
    return corr_matrix
     


