
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

"""Functions"""

def fisher_test(df, indep_var, dep_var):
    # indep_var : List of categorical variable names to test 
    # dep_var: List of binary variable names to test
    var_pairs = [(indep, dep) for indep in indep_var for dep in dep_var]
    
    results = []  
    for x, y in var_pairs:   
        # create contingency table with your pivot_table method
        contingency = df.pivot_table(index=x, columns=y, aggfunc=len).fillna(0).astype(int)
        # calculate fisher's exact test
        odds_ratio, p_value = stats.fisher_exact(contingency)
        # convert results to dataframe
        results.append({
            'indep_var': x,
            'dep_var': y,
            'odds_ratio': odds_ratio,
            'p_value': p_value
        })
    
    return pd.DataFrame(results)


def chi2_test(df, dep_var, indep_var):
        # dep_var: List of dependent ordinal variable names
        # indep_var : List of ordinal variable names to test 
        var_pairs = [(indep, dep) for indep in indep_var for dep in dep_var]  
        
        results = []
        
        for x, y in var_pairs:   
            # create contingency table
            contingency = pd.crosstab(df[x], df[y]).fillna(0).astype(int)
            
            # run chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            # convert results to dataframe
            results.append({
                'indep_var': x,
                'dep_var': y,
                'chi2_stat': chi2,
                'p_value': p
            })
        
        return pd.DataFrame(results)


def ttest_mannw(df, group_vars, continuous_vars):
  
    # group_var: List of ordinal variable names to test
    # continuous_var : List of continuous variable names to test
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
                'test': 'Not enough groups',
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


def anova_kruskal(df, group_vars, continuous_vars):
    # group_var: List of ordinal variable names to test
    # continuous_var : List of continuous variable names to test
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
              'test': np.nan,
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


def spearman_corr(df, dep_vars, indep_vars):
    # dep_var: List of ordinal variable names
    # indep_var : List of continuous variable names to test
    var_pairs = [(cont_indep, cont_dep) for cont_indep in indep_vars for cont_dep in dep_vars]
  
    results = []

    for x, y in var_pairs:
        corr, pval = stats.spearmanr(df[x], df[y], nan_policy='omit')
        results.append({
            'dep_var': x,
            'indep_var': y,
            'spearman_corr': corr,
            'p_value': pval
        })

    return pd.DataFrame(results)


