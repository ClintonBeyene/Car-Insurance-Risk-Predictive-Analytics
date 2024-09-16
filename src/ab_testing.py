# ab_testing.py

import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, ttest_ind, ranksums
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def select_metrics(df):
    # Selecting TotalPremium and TotalClaims as metrics
    return df[['TotalPremium', 'TotalClaims']]

def data_segmentation(df, feature, group_a, group_b):
    group_a_df = df[df[feature] == group_a]
    group_b_df = df[df[feature] == group_b]
    return group_a_df, group_b_df

def perform_t_test(group_a_df, group_b_df, metric):
    t_stat, p_value = stats.ttest_ind(group_a_df[metric], group_b_df[metric], equal_var=False)
    return t_stat, p_value

def perform_z_test(group_a_df, group_b_df, metric):
    mean_diff = group_a_df[metric].mean() - group_b_df[metric].mean()
    pooled_std = ((group_a_df[metric].std() ** 2 / len(group_a_df)) + (group_b_df[metric].std() ** 2 / len(group_b_df))) ** 0.5
    z_stat = mean_diff / pooled_std
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value

def perform_anova_test(df, feature, target):
    groups = [df[df[feature] == level][target] for level in df[feature].unique()]
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value

def perform_kruskal_test(df, feature, target):
    groups = [df[df[feature] == level][target] for level in df[feature].unique()]
    h_stat, p_value = kruskal(*groups)
    return h_stat, p_value

def perform_chi_squared_test(df, feature, target):
    contingency_table = pd.crosstab(df[feature], df[target])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p_value, dof, expected

def perform_wilcoxon_test(group_a_df, group_b_df, metric):
    w_stat, p_value = ranksums(group_a_df[metric], group_b_df[metric])
    return w_stat, p_value

def analyze_results(p_value, alpha=0.05):
    if p_value < alpha:
        return "Reject the null hypothesis"
    else:
        return "Fail to reject the null hypothesis"