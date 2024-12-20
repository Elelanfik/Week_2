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
