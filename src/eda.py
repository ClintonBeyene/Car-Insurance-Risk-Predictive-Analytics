import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def perform_eda(df):
    # Descriptive statistics
    print(df.describe())

    # Data structure
    print(df.info())

    # Missing values
    print(df.isnull().sum())

    # Histograms for numerical columns
    df[['TotalPremium', 'TotalClaims']].hist(bins=30, figsize=(10, 5))
    plt.show()

    # Correlation matrix for numeric columns only
    plt.figure(figsize=(15,10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

if __name__ == "__main__":
    df = load_data('../data/transformed_data.csv')
    perform_eda(df)
