"""
Created on April 06th 2025

Author: Lory H

Purpose: Study population description

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

# Convert list into dataframe
def ls_to_df(ls):
    new_df = []
    for df in ls :
        new_df += [pd.DataFrame(df)]
    return new_df

# Calculate missing data

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
        if var_type == "ordinal":
            count = df[var].sum()
            percent = (count / len(df)) * 100
            records.append({
                'variables': var,
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


# Descriptive data for each group compared
def desc_continuous(df, groups):
    mean = []
    sem = []
    for status in groups:
        #Calculate means and sem of continuous data
        result1 = []
        result2 = []
        for columns in [continuous_var]:
            result1 += [df[columns][df[groups] == status].mean(axis=0)]
            result2 += [df[columns][df[groups] == status].sem(axis=0)]
        mean += [result1]
        sem += [result2]
        
    return mean, sem

"""Importing Excel sheets"""

path = "write pathname here"

#Import data to dataframe
data = pd.read_excel (path+r'databasefile.xlsx') 
list(data.columns)


# Define type of variables to include in the analysis
ordinal_var = ['ordinal variable names']

continuous_var = ['continuous variable names']

# Missing data

    # calculate missing data and generate dataframes
results = []
for var in [ordinal_var, continuous_var]:
    results += [missing_data(data, var)]

missing_ordinal_df, missing_continuous_df = results

    # combine the results into one dataframe
missing_df = pd.concat([missing_ordinal_df, missing_continuous_df])


# Descriptive data

    # for ordinal variables
ordinal_summary_df = describe_variables(data, ordinal_var, var_type="ordinal")

    # for continuous variables
continuous_summary_df = describe_variables(data, continuous_var, var_type="continuous")

    # combine the results into one dataframe
summary_df = pd.concat([ordinal_summary_df, continuous_summary_df])


# Merge all descriptive data into one dataframe 
descriptive_table1 = pd.merge(missing_df, summary_df, on='variables', how='outer')


