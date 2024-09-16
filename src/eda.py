import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def perform_eda(df):
    # Print the first few rows of the dataframe
    print(df.head())
    
    # Print the data types of each column
    print(df.dtypes)

    # Data structure
    print(df.info())

    # Print the summary statistics of the dataframe
    print(df.describe())

    # Print the unique values and counts for each categorical column
    for col in df.select_dtypes(include=['object']).columns:
        print(f"Column: {col}")
        print(df[col].value_counts())
        print()

    # Histograms for numerical columns
    df[['TotalPremium', 'TotalClaims']].hist(bins=30, figsize=(10, 5))
    plt.show()

    # Correlation matrix for numeric columns only
    plt.figure(figsize=(15,10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

    # Print the number of missing values in each column
    print(df.isnull().sum())

if __name__ == "__main__":
    df = load_data('../data/transformed_data.csv')
    perform_eda(df)
