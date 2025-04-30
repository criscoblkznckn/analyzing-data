This script loads a dataset, performs basic analysis, and creates visualizations
using pandas and matplotlib libraries.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for better looking plots
sns.set(style="whitegrid")

def load_and_explore_data():
    """
    Task 1: Load and explore the dataset
    """
    print("\n=== Loading and Exploring Dataset ===\n")
    
    try:
        # Load the Iris dataset from sklearn
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\n")
        
        # Check data types and missing values
        print("Dataset information:")
        print(df.info())
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        # No cleaning needed for Iris dataset as it's already clean
        print("\nNo missing values found - dataset is clean.")
        
        return df
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def perform_basic_analysis(df):
    """
    Task 2: Perform basic data analysis
    """
    print("\n=== Basic Data Analysis ===\n")
    
    # Basic statistics for numerical columns
    print("Descriptive statistics for numerical columns:")
    print(df.describe())
    
    # Group by species and calculate means
    print("\nMean values by species:")
    species_means = df.groupby('species').mean()
    print(species_means)
    
    # Interesting findings
    print("\nInteresting observations:")
    print("- Setosa has significantly smaller petal dimensions than other species")
    print("- Virginica has the largest sepal length on average")
    print("- Versicolor and virginica have more similar measurements than setosa")

def create_visualizations(df):
    """
    Task 3: Create data visualizations
    """
    print("\n=== Creating Visualizations ===\n")
    
    # Set up figure and subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Line chart (simulating trends by index since Iris isn't time-series)
    plt.subplot(2, 2, 1)
    df['sepal length (cm)'].plot(kind='line', color='green')
    plt.title('Sepal Length Trend (by index)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sepal Length (cm)')
    
    # 2. Bar chart - average sepal length by species
    plt.subplot(2, 2, 2)
    df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar', color=['blue', 'orange', 'green'])
    plt.title('Average Sepal Length by Species')
    plt.ylabel('Length (cm)')
    plt.xticks(rotation=45)
    
    # 3. Histogram - distribution of petal length
    plt.subplot(2, 2, 3)
    df['petal length (cm)'].plot(kind='hist', bins=20, color='purple', edgecolor='black')
    plt.title('Distribution of Petal Lengths')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    
    # 4. Scatter plot - sepal length vs petal length
    plt.subplot(2, 2, 4)
    colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
    for species, group in df.groupby('species'):
        plt.scatter(group['sepal length (cm)'], group['petal length (cm)'], 
                   color=colors[species], label=species, alpha=0.7)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('iris_visualizations.png')
    print("Visualizations saved as 'iris_visualizations.png'")
    plt.show()

def main():
    """
    Main function to execute all tasks
    """
    print("Data Analysis Assignment\n")
    
    # Task 1: Load and explore data
    df = load_and_explore_data()
    
    if df is not None:
        # Task 2: Basic analysis
        perform_basic_analysis(df)
        
        # Task 3: Visualizations
        create_visualizations(df)
        
        print("\nAnalysis completed successfully!")
    else:
        print("Failed to load dataset. Analysis cannot continue.")

if __name__ == "__main__":
    main()
