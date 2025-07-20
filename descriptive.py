"""
Author: Lory H

Purpose: Population description (mean, sem or sd, count, percentages)

"""


import pandas as pd
from datetime import datetime
from datetime import date
import numpy as np
import math
#stats related
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scikit_posthocs as sp
import seaborn as sns

"""Functions"""

# Missing data summary

def missing_data(df, variable_list):
    records = []

    for var in variable_list:
        count = df[var].isna().sum()
        percent = (count / len(df)) * 100
        records.append({
            'variables': var,
            'missing_count': count,
            'missing_percent': percent
        })

    return pd.DataFrame(records)

# Descriptive summary data

def describe_variables(df, variable_list, var_type="ordinal"):
    records = []

    for var in variable_list:
        if var_type == "binary":
            count = df[var].sum()
            percent = (count / len(df[var])) * 100
            records.append({
                'variables': var,
                'value': 1,
                'count': count,
                'percent': percent
            })
            
        elif var_type == "categorical":
            value_counts = df[var].value_counts(dropna=False)
            total = len(df[var])
            for val, count in value_counts.items():
                percent = (count / total) * 100
                records.append({
                    'variables': var,
                    'value': val,
                    'count': count,
                    'percent': percent
                })
                
        elif var_type == "continuous":
            mean = df[var].mean()
            sem = df[var].sem()
            records.append({
                'variables': var,
                'mean': mean,
                'sem': sem
            })
        else:
            raise ValueError("var_type must be 'ordinal' or 'continuous'")

    return pd.DataFrame(records)


"""Importing your database"""

#Use path for easier access
path = "pathname of your file"

#Import data to dataframe
data = pd.read_excel (path+r'database.xlsx') 
list(data.columns)


# Define type of variables to include in the analysis

def get_variable_types(df, threshold=2, cat_max_unique=10):
    binary = [col for col in df.columns if df[col].dropna().nunique() == 2]
    categorical = [col for col in df.columns if threshold < df[col].dropna().nunique() <= cat_max_unique and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]))]
    continuous = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].dropna().nunique() > cat_max_unique]
    return binary, categorical, continuous

# Use function to create your different lists of variables

binary_vars, categorical_vars, continuous_vars = get_variable_types(data)

# Descriptive data
    
    # for binary variables
binary_df = describe_variables(data, binary_vars, var_type='binary')
    # for categorical variables
categorical_df = describe_variables(data, categorical_vars, var_type='categorical')
    # for continuous variables
continuous_df = describe_variables(data, continuous_vars, var_type='continuous')

    # combine the results into one dataframe
summary_df = pd.concat([binary_df, categorical_df, continuous_df])

# Missing data

    # calculate missing data and generate dataframes
results = []
for var in [binary_vars, categorical_vars, continuous_vars]:
    results += [missing_data(data, var)]

missing_binary_df, missing_categorical_df, missing_continuous_df = results

    # combine the results into one dataframe
missing_df = pd.concat([missing_binary_df, missing_categorical_df, missing_continuous_df])

# Merge all dataframe into one according to variable name (column name)
descriptive_table = pd.merge(summary_df, missing_df, on = 'variables')

# Export dataframe into excel
descriptive_table.to_excel(path+r'results/descriptive_summary.xlsx', index=False)
