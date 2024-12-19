import matplotlib.pyplot as plt
import seaborn as sns

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