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
    
    # Convert categorical columns to numeric columns
    categorical_columns = ['TransactionMonth', 'IsVATRegistered', 'Citizenship', 'LegalType', 
                           'Title', 'Language', 'Bank', 'AccountType', 'MaritalStatus', 
                           'Gender', 'Country', 'Province', 'MainCrestaZone', 'SubCrestaZone', 
                           'ItemType','mmcode','make', 'Model', 'bodytype', 'NumberOfDoors', 
                           'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 'WrittenOff', 
                           'Rebuilt', 'Converted', 'CrossBorder', 'CoverCategory', 'CoverType', 
                           'CoverGroup', 'Section', 'Product', 'StatutoryClass', 
                           'StatutoryRiskType', 'VehicleMake_CoverType', 'CoverGroup_Premium',
                            'CoverGroup_Claims', 'VehicleType']
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes
    
    # Convert TransactionMonth column to numeric column
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth']).dt.month
    
    # Calculate AgeOfVehicle column
    df['AgeOfVehicle'] = 2023 - df['RegistrationYear']
    
    return df

def encode_categorical_data(df):
    categorical_columns = ['Region_VehicleType', 'VehicleMake_CoverType', 'CoverGroup_Premium',
                          'CoverGroup_Claims','Language', 'Bank', 'AccountType', 'MaritalStatus', 'Gender', 
                           'Province', 'MainCrestaZone', 'SubCrestaZone', 'ItemType', 
                           'VehicleType', 'bodytype', 'AlarmImmobiliser', 'TrackingDevice',
                            'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 
                            'TermFrequency', 'ExcessSelected', 'CoverCategory', 'CoverType', 
                            'CoverGroup', 'Section', 'Product', 'StatutoryClass', 
                            'StatutoryRiskType', 'Region_VehicleType', 'VehicleMake_CoverType',
                              'CoverGroup_Premium', 'CoverGroup_Claims']
    already_encoded_columns = ['Language', 'Bank', 'AccountType', 'MaritalStatus', 'Gender',
                                'Province',  'MainCrestaZone', 'SubCrestaZone', 'ItemType',
                                'VehicleType', 'bodytype', 'AlarmImmobiliser', 'TrackingDevice',
                                'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'VehicleType' 
                                'CrossBorder', 'CoverCategory', 'CoverType', 'CoverGroup', 
                                'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType']
    new_categorical_columns = [column for column in categorical_columns if column not in already_encoded_columns]
    df = pd.get_dummies(df, columns=new_categorical_columns, drop_first=True, sparse=True)
    le = LabelEncoder()
    df['make'] = le.fit_transform(df['make'])
    df['Model'] = le.fit_transform(df['Model'])
    return df

def scale_numerical_features(df):
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df
