import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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

def analyze_correlation(data, columns, title="Correlation Matrix"):
    
    # Select relevant columns
    correlation_data = data[columns]
    
    # Compute the correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Display the correlation matrix
    print(correlation_matrix)
    
    # Heatmap visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()
    
    return correlation_matrix

def plot_top_customers(top_customers, metric_name, ylabel, save_path=None):
    """
    Plots the top 10 customers for a specific metric with better formatting.
    
    Parameters:
        top_customers (DataFrame): DataFrame containing the top 10 customers.
        metric_name (str): Name of the metric for the plot title.
        ylabel (str): Label for the y-axis.
        save_path (str): Path to save the plot as an image. If None, the plot is not saved.
    """
    # Reset index for plotting
    top_customers = top_customers.reset_index()
    
    # Shorten the MSISDN/Number to "Customer 1", "Customer 2", etc.
    top_customers['Customer Label'] = [f"Customer {i+1}" for i in range(len(top_customers))]
    
    # Convert large y-axis values to a readable format
    y_values = top_customers[top_customers.columns[1]]
    if y_values.max() > 1e6:
        y_values = y_values / 1e6
        ylabel += " (in millions)"
    elif y_values.max() > 1e3:
        y_values = y_values / 1e3
        ylabel += " (in thousands)"
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_customers['Customer Label'], y_values, color='skyblue')
    plt.title(f"Top 10 Customers by {metric_name}", fontsize=16)
    plt.xlabel("Customer", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
    else:
        plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_throughput_distribution_by_handset(self):
    """
    Analyzes and visualizes the distribution of average throughput for the top 5 handset types.
    """
    # Group by handset type and calculate average throughput
    throughput_per_handset = self.df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()

    # Sort by throughput and select the top 5 handset types
    throughput_sorted = throughput_per_handset.sort_values(ascending=False).head(5)
    selected_handset_types = throughput_sorted.index.tolist()

    print("\nSelected Handset Types for Throughput (Top 5 by Average Throughput):")
    print(selected_handset_types)

    # Filter the data for the selected handset types
    filtered_data = self.df[self.df['Handset Type'].isin(selected_handset_types)]

    # Plot histograms for each of the selected handset types
    plt.figure(figsize=(12, 6))
    for handset_type in selected_handset_types:
        subset = filtered_data[filtered_data['Handset Type'] == handset_type]
        sns.histplot(
            subset['Avg Bearer TP DL (kbps)'], 
            kde=True, 
            label=handset_type, 
            bins=20, 
            alpha=0.6
        )
    
    plt.title('Histogram of Average Throughput for Top 5 Handset Types')
    plt.xlabel('Average Throughput (kbps)')
    plt.ylabel('Frequency')
    plt.legend(title='Handset Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_retransmission_distribution_by_handset(self):
    """
    Analyzes and visualizes the distribution of average TCP retransmission for the top 5 handset types.
    """
    # Group by handset type and calculate average retransmissions
    retransmission_per_handset = self.df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()

    # Sort by retransmissions and select the top 5 handset types
    retransmission_sorted = retransmission_per_handset.sort_values(ascending=False).head(5)
    selected_handset_types = retransmission_sorted.index.tolist()

    print("\nSelected Handset Types for Retransmissions (Top 5 by Average Retransmissions):")
    print(selected_handset_types)

    # Filter the data for the selected handset types
    filtered_data = self.df[self.df['Handset Type'].isin(selected_handset_types)]

    # Plot histograms for each of the selected handset types
    plt.figure(figsize=(12, 6))
    for handset_type in selected_handset_types:
        subset = filtered_data[filtered_data['Handset Type'] == handset_type]
        sns.histplot(
            subset['TCP DL Retrans. Vol (Bytes)'], 
            kde=True, 
            label=handset_type, 
            bins=20, 
            alpha=0.6
        )
    
    plt.title('Histogram of Average TCP Retransmissions for Top 5 Handset Types')
    plt.xlabel('Average TCP Retransmission (Bytes)')
    plt.ylabel('Frequency')
    plt.legend(title='Handset Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

