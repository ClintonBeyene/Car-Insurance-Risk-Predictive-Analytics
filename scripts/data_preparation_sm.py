import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def feature_engineering(df):
    df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
    df['AgeOfVehicle'] = 2023 - df['VehicleIntroDate'].dt.year
    df['CapitalOutstanding'] = pd.to_numeric(df['CapitalOutstanding'], errors='coerce')
    df['VehicleValue'] = df['CapitalOutstanding'].apply(lambda x: 1 if x > 0 else 0)
    df['DriverExperience'] = 2023 - df['RegistrationYear']
    df['Region_VehicleType'] = df['Province'] + '_' + df['VehicleType']
    df['VehicleMake_CoverType'] = df['make'] + '_' + df['CoverType']
    df['CoverGroup_Premium'] = df['CoverGroup'] + '_' + df['TotalPremium'].astype(str)
    df['CoverGroup_Claims'] = df['CoverGroup'] + '_' + df['TotalClaims'].astype(str)
    
    # Drop the original date column to avoid conversion errors
    df = df.drop(columns=['VehicleIntroDate'])
    return df

def encode_categorical_data(df):
    # Convert categorical columns to numeric columns
    from sklearn.impute import SimpleImputer

    def impute_specific_column(df, column_name):
        imputer = SimpleImputer(strategy='mean')  
        df[[column_name]] = imputer.fit_transform(df[[column_name]])
        return df

    # Apply imputation to the 'CapitalOutstanding' column
    df = impute_specific_column(df, 'CapitalOutstanding')

    categorical_columns = ['TransactionMonth', 'IsVATRegistered', 'Citizenship', 'LegalType', 
                           'Title', 'Language', 'Bank', 'AccountType', 'MaritalStatus', 
                           'Gender', 'Country', 'Province', 'MainCrestaZone', 'SubCrestaZone', 
                           'ItemType','mmcode','make', 'Model', 'bodytype', 'NumberOfDoors', 
                           'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 'WrittenOff', 
                           'Rebuilt', 'Converted', 'CrossBorder', 'CoverCategory', 'CoverType', 
                           'CoverGroup', 'Section', 'Product', 'StatutoryClass', 
                           'StatutoryRiskType', 'VehicleMake_CoverType', 'CoverGroup_Premium',
                            'CoverGroup_Claims', 'VehicleType', 'Region_VehicleType', 
                            'VehicleMake_CoverType', 'CoverGroup_Premium', 'CoverGroup_Claims',
                            'VehicleType', 'TermFrequency', 'ExcessSelected']

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le


    return df

def scale_numerical_features(df):
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df
