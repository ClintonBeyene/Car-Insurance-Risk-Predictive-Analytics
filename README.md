# Car Insurance Planning and Marketing Analysis
## Project Overview
AlphaCare Insurance Solutions (ACIS) is committed to developing cutting-edge risk and predictive analytics for car insurance planning and marketing in South Africa. This project aims to analyze historical insurance claim data to optimize marketing strategies and identify low-risk targets for premium reduction.

## Table of Contents
* Project Overview
* Business Objective
* Data Description
* Tasks Completed
* Exploratory Data Analysis (EDA)
* Visualizations
* Next Steps
* How to Run the Project
* Contributors
## Business Objective
The objective of this analysis is to help optimize the marketing strategy and discover “low-risk” targets for which the premium could be reduced, hence an opportunity to attract new clients.

## Data Description
The historical data spans from February 2014 to August 2015 and includes the following columns:

* **Insurance Policy Details**: UnderwrittenCoverID, PolicyID, TransactionMonth
* **Client Details**: IsVATRegistered, Citizenship, LegalType, Title, Language, Bank, AccountType, MaritalStatus, Gender
* **Client Location**: Country, Province, PostalCode, MainCrestaZone, SubCrestaZone
* **Car Details**: ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, Cylinders, Cubiccapacity, Kilowatts, Bodytype, NumberOfDoors, VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser, TrackingDevice, CapitalOutstanding, NewVehicle, WrittenOff, Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet
* **Plan Details**: SumInsured, TermFrequency, CalculatedPremiumPerTerm, ExcessSelected, CoverCategory, CoverType, CoverGroup, Section, Product, StatutoryClass, StatutoryRiskType
* **Payment & Claim Details**: TotalPremium, TotalClaims
## Tasks Completed
### Git and GitHub
* Created a Git repository for the project with a comprehensive README.
* Set up Git version control and created a branch named task-1 for day 1 analysis.
* Implemented CI/CD with GitHub Actions.
## Exploratory Data Analysis (EDA)
* **Data Summarization**: Calculated descriptive statistics for numerical features such as TotalPremium and TotalClaims.
* **Data Quality Assessment**: Checked for missing values and handled them appropriately.
* **Univariate Analysis**: Plotted histograms for numerical columns and bar charts for categorical columns to understand distributions.
* **Bivariate and Multivariate Analysis**: Explored correlations and associations between variables using scatter plots and correlation matrices.
## Visualizations
* Histograms: Created histograms for TotalPremium and TotalClaims.
* Correlation Heatmap: Generated a heatmap to explore relationships between numerical variables.
* Scatter Plots: Analyzed relationships between TotalPremium, TotalClaims, and other features.
## Next Steps
* Continue with A/B Hypothesis Testing to validate the following hypotheses:
** There are no risk differences across provinces.
** There are no risk differences between zip codes.
** There are no significant margin (profit) differences between zip codes.
** There are no significant risk differences between women and men.
* Develop machine learning models to predict total claims and optimal premium values based on various features.
* Report on the explaining power of important features influencing the models.
## How to Run the Project
1. Clone the repository:
``` bash
git clone https://github.com/yourusername/insurance-analysis.git
cd insurance-analysis
```

2. Create a virtual environment and install dependencies:
``` bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
3. Run the Jupyter notebooks for EDA and model development:
``` bash 
jupyter notebook
``` 
### Author: Clinton Beyene
