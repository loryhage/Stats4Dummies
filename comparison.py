
"""
Created on Jul 15 2025

Author: Lory H

Purpose: Comparison of two groups
"""

import pandas as pd
from datetime import datetime
from datetime import date
import numpy as np
import math
#stats related
import scipy.stats as stats
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scikit_posthocs as sp
import seaborn as sns


"""Importing Data Tables"""

path = "pathname"

#Import data to dataframe
data = pd.read_excel (path+r'database.xlsx') 
list(data.columns)


"""Calculating descriptive data of groups compared"""

def descriptive_continuous(df, group_col, continuous_vars):
    results = []

    for var in continuous_vars:
        for group_val in df[group_col].dropna().unique():
            subset = df[df[group_col] == group_val][var].dropna()
            results.append({
                'dep_var' : group_col,
                'indep_var': var,
                'group': group_val,
                'mean': subset.mean(),
                'sem': subset.sem()
            })

    return pd.DataFrame(results)

def descriptive_ordinal(df, group_col, categorical_vars):
    results = []

    for var in categorical_vars:
        grouped = df.groupby(group_col)[var].value_counts(dropna=False).unstack(fill_value=0)
        totals = grouped.sum(axis=1)

        for group_val in grouped.index:
            for category_val in grouped.columns:
                count = grouped.loc[group_val, category_val]
                percent = (count / totals[group_val]) * 100 if totals[group_val] > 0 else np.nan

                results.append({
                    'dep_var' : group_col,
                    'group': group_val,
                    'indep_var': var,
                    'categories':category_val,
                    'count': count,
                    'percent': percent
                })

    return pd.DataFrame(results)



"""Statistical Tests"""
#Comparing binary vs ordinal variable
def fisher_test(df, dep_var, indep_var):
    # dep_var: List of dependent binary variable names
    # indep_var : List of ordinal variable names to test 
    var_pairs = [(indep, dep) for indep in indep_var for dep in dep_var]  
     
    results = []  
    for x, y in var_pairs:   
        # create contingency table with your pivot_table method
        contingency = df.pivot_table(index=x, columns=y, aggfunc=len).fillna(0).astype(int)
        # Check if contingency table is empty or too small for test
        if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        # Not enough data to perform chi-square test
            results.append({
                'indep_var': x,
                'dep_var': y,
                'stat': np.nan,
                'p_value': np.nan
            })
            continue
        # calculate fisher's exact test
        odds_ratio, p_value = stats.fisher_exact(contingency)
        # convert results to dataframe
        results.append({
            'indep_var': x,
            'dep_var': y,
            'stat': odds_ratio,
            'p_value': p_value
        })
    
    return pd.DataFrame(results)

#Comparing ordinal vs ordinal variables
def chi2_test(df, dep_var, indep_var):
        # dep_var: List of dependent ordinal variable names
        # indep_var : List of ordinal variable names to test 
        var_pairs = [(indep, dep) for indep in indep_var for dep in dep_var]  
        
        results = []
        
        for x, y in var_pairs:  
            # create contingency table
            contingency = pd.crosstab(df[x], df[y]).fillna(0).astype(int)
            
            # Check if contingency table is empty or too small for test
            if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
            # Not enough data to perform chi-square test
                results.append({
                    'indep_var': x,
                    'dep_var': y,
                    'stat': np.nan,
                    'p_value': np.nan
                })
                continue
            
            # run chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            # convert results to dataframe
            results.append({
                'indep_var': x,
                'dep_var': y,
                'stat': chi2,
                'p_value': p
            })
        
        return pd.DataFrame(results)

#Comparing binary vs continuous variables
def ttest_mannw(df, group_vars, continuous_vars):
  
    # group_vars: List of ordinal variable names to test
    # continuous_vars : List of continuous variable names to test
    var_pairs = [(group_var, cont_var) for group_var in group_vars for cont_var in continuous_vars]

    results = []
   
    # Step 1: Parametric assumption tests: normality and variance equality checks  
    for var, group_var in var_pairs:
        groups = df[group_var].dropna().unique()
        if len(groups) < 2:
            # Not enough groups to compare
            results.append({
                'dep_var': group_var,
                'indep_var': var,
                'normality_p': np.nan,
                'variance_p': np.nan,
                'test': 'none',
                'statistic': np.nan,
                'p_value': np.nan
            })
            continue

        group0 = df[df[group_var] == groups[0]][var].dropna()
        group1 = df[df[group_var] == groups[1]][var].dropna()

        # Normality test on residuals of OLS
        model = smf.ols(f'{var} ~ C({group_var})', data=df).fit()
        shapiro_p = stats.shapiro(model.resid)[1]

        # Variance homogeneity test
        if shapiro_p > 0.05:
            var_test = stats.bartlett(group0, group1)
        else:
            var_test = stats.levene(group0, group1)
        var_p = var_test.pvalue

    # Step 2: Performs t-test if parametric or Mann-Whitney test if non parametric data
 
        # Choose test
        if shapiro_p > 0.05 and var_p > 0.05:
            test_name = 't-test'
            stat, pval = stats.ttest_ind(group0, group1, nan_policy='omit')
        else:
            test_name = 'Mann-Whitney'
            stat, pval = stats.mannwhitneyu(group0, group1, alternative='two-sided')

        results.append({
            'dep_var': group_var,
            'indep_var': var,
            'normality_p': shapiro_p,
            'variance_p': var_p,
            'test': test_name,
            'statistic': stat,
            'p_value': pval
        })

    return pd.DataFrame(results)

#Comparing categorical vs continuous variables
def anova_kruskal(df, group_vars, continuous_vars):
    # group_vars: List of ordinal variable names to test
    # continuous_vars : List of continuous variable names to test
    var_pairs = [(group_var, cont_var) for group_var in group_vars for cont_var in continuous_vars]
    results = []

    
    for var, group_var in var_pairs:
        data_sub = df[[group_var, var]].dropna()
        groups = data_sub[group_var].unique()
        groups.sort()
        data_groups = [data_sub[data_sub[group_var] == g][var] for g in groups]
        # Check if there are at least 2 groups with data
        non_empty_groups = [g for g in data_groups if len(g) > 0]
        if len(non_empty_groups) < 2:
          # Not enough groups to test
          results.append({
              'dep_var': group_var,
              'indep_var': var,
              'normality_p': np.nan,
              'variance_p': np.nan,
              'test': 'none',
              'statistic': np.nan,
              'p_value': np.nan
          })
          continue

        # Fit OLS model for residuals normality test
        model = smf.ols(f'{var} ~ C({group_var})', data=data_sub).fit()
        shapiro_p = stats.shapiro(model.resid)[1]
        
        # Variance homogeneity test: Bartlett if normal residuals, else Levene
        if shapiro_p > 0.05:
            var_test = stats.bartlett(*data_groups)
        else:
            var_test = stats.levene(*data_groups)
        var_p = var_test.pvalue

        # Check if all values are identical across all groups (no variance)
        all_values = np.concatenate(data_groups)
        if np.all(all_values == all_values[0]):
            # all identical values, skip test
            results.append({
                'dep_var': group_var,
                'indep_var': var,
                'normality_p': shapiro_p,
                'variance_p': var_p,
                'test': 'No variance',
                'statistic': np.nan,
                'p_value': np.nan
            })
            continue  
       
        # Step 2: Performs anova if parametric or Kruskal-Wallis test if non parametric data
     
        # Choose test
        if shapiro_p > 0.05 and var_p > 0.05:
            test_name = 'ANOVA'
            anova_table = sm.stats.anova_lm(model, typ=2)
            stat = anova_table['F'][0]
            pval = anova_table['PR(>F)'][0]
        else:
            test_name = 'Kruskal-Wallis'
            stat, pval = stats.kruskal(*data_groups)
        
        results.append({
            'dep_var': group_var,
            'indep_var': var,
            'normality_p': shapiro_p,
            'variance_p': var_p,
            'test': test_name,
            'statistic': stat,
            'p_value': pval
        })
        
    return pd.DataFrame(results)

# Comparing continuous vs continuous variables
def spearman_corr(df, dep_vars, indep_vars):

