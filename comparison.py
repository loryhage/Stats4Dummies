"""Importing libraries"""

import pandas as pd
import numpy as np
import math
#stats related
import scipy.stats as stats
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scikit_posthocs as sp
import seaborn as sns
#format related
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font


#Description of groups
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

#Comparing binary vs ordinal variable
def fisher_test(df, dep_var, indep_var):
    # dep_var: List of dependent binary variable names
    # indep_var : List of ordinal variable names to test 
    results = []  
    for x in indep_var:
        for y in dep_var:
            contingency = pd.crosstab(df[x], df[y])
            if contingency.shape != (2, 2):
                results.append({'indep_var': x, 'dep_var': y, 'stat': np.nan, 'p_value': np.nan})
                continue
            odds_ratio, p_value = stats.fisher_exact(contingency.values)
            results.append({'indep_var': x, 'dep_var': y, 'stat': odds_ratio, 'p_value': p_value})
    return pd.DataFrame(results)

#Comparing ordinal vs ordinal variables
def chi2_test(df, dep_var, indep_var):
    # dep_var: List of dependent ordinal variable names
    # indep_var : List of ordinal variable names to test 
    results = []
    for x in indep_var:
        for y in dep_var:
            if df[x].nunique() < 3 or df[y].nunique() < 3:
                continue  # skip binary-like vars
            contingency = pd.crosstab(df[x], df[y])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                results.append({'indep_var': x, 'dep_var': y, 'stat': np.nan, 'p_value': np.nan})
                continue
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            results.append({'indep_var': x, 'dep_var': y, 'stat': chi2, 'p_value': p})
    return pd.DataFrame(results)

#Comparing binary vs continuous variables
def ttest_mannw(df, group_vars, continuous_vars):
    # group_vars: List of ordinal variable names to test
    # continuous_vars : List of continuous variable names to test
    var_pairs = [(group_var, cont_var) for group_var in group_vars for cont_var in continuous_vars]
    
    # Step 1: Parametric assumption tests: normality and variance equality checks  
    results = []
    for group_var, var in var_pairs:
        groups = df[group_var].dropna().unique()
        
        # Add this check here:
        if len(groups) != 2:
            # Skip or append NaNs, or log a warning
            results.append({
                'dep_var': group_var,
                'indep_var': var,
                'normality_p': np.nan,
                'variance_p': np.nan,
                'test': 'Skipped - not 2 groups',
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

""" Define type of variables to include in the analysis"""

def get_variable_types(df, threshold=2, cat_max_unique=10):
    binary = [col for col in df.columns if df[col].dropna().nunique() == 2]
    categorical = [col for col in df.columns if threshold < df[col].dropna().nunique() <= cat_max_unique and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]))]
    continuous = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].dropna().nunique() > cat_max_unique]
    return binary, categorical, continuous

# Use function to create your variable lists

binary_vars, categorical_vars, continuous_vars = get_variable_types(data)

# Delete categorical variables that include string (text)
categorical_vars = [col for col in categorical_vars if not pd.api.types.is_object_dtype(data[col])]

# Define your dependant (outcome) and independant (population) variables
dep_cont = [''] #subset of continuous_vars
dep_bin = [''] #subset of categorical_vars
dep_cat = [''] #subset of binary_vars

indep_cont = [''] #subset of continuous_vars
indep_bin = [''] #subset of categorical_vars
indep_cat = [''] #subset of binary_vars

"""Descriptive results between groups"""

# Continuous variables

results = []
for dep_var in dep_cont:  # iterate over all dependent continuous variables
    try:
        # test all independent continuous variables
        df_tmp = descriptive_continuous(data, dep_var, indep_cont) 
        if not df_tmp.empty:    # ignore empty dataframes
            results.append(df_tmp)
    except Exception as e:
        print(f"Error with variable {dep_var}: {e}") # raise error issue

continuous_desc = pd.concat(results, ignore_index=True) # combine all results


# Ordinal variables

results = []
for dep_var in dep_bin+dep_cat:  #iterate over all ordinal variables
    try:
        #test all independent ordinal variables
        df = descriptive_ordinal(data, dep_var, indep_bin+indep_cat)
        if not df.empty:    #ignore empty dataframes
            results.append(df)
    except:
        pass

ordinal_desc = pd.concat(results, ignore_index=True) #combine all results

"""Statistical test results between groups"""

"""categorical vs categorical"""

# Chi-square Test for categorical data
chi2_results = chi2_test(data, dep_bin+dep_cat, indep_bin+indep_cat)

# Fisher's Exact Test for binary data
fisher_results = fisher_test(data, dep_bin, indep_bin)

"""categorical vs continuous"""

# T-test or Mann-Whitney U test for continuous vs binary data 
continuous_bin_df =  ttest_mannw(data, binary_vars, continuous_vars)
print(continuous_bin_df)

# Anova or Kruskal-Wallis for continuous vs categorical data
continuous_cat_df =  anova_kruskal(data, categorical_vars, continuous_vars)
print(continuous_cat_df)

"""continuous vs continuous"""

# Spearman correlation for continuous vs continuous data
spearman_df = spearman_corr(data, dep_cont, indep_cont)
print(spearman_df)

"""Export results into dataframes"""

# Combine all ordinal variable results

    # add a common column to identify statistical test
chi2_results['stat_test'] = 'Chi-2'
fisher_results['stat_test'] = 'Fisher'

    # combine dataframes
ordinal_stats_df = pd.concat([
    chi2_results,
    fisher_results
],ignore_index=True)

# Combine all continuous variable results

# Add a common column to identify statistical test
continuous_bin_df['stat_test'] = 'T-test or Mann-Whitney'
continuous_cat_df['stat_test'] = 'ANOVA or Kruskal-Wallis'
spearman_df['stat_test'] = 'Spearman correlation'

# Merge all into a single DataFrame
continuous_stats_df = pd.concat([
    continuous_bin_df,
    continuous_cat_df,
    spearman_df
], ignore_index=True)

#Export all results into one excel with multiple sheets
with pd.ExcelWriter(path + r'results/statistical_summary.xlsx') as writer:
    continuous_desc.to_excel(writer, sheet_name='Continuous Descriptive', index=False)
    continuous_stats_df.to_excel(writer, sheet_name='Continuous Tests', index=False)
    ordinal_desc.to_excel(writer, sheet_name='Ordinal Descriptive', index=False)
    ordinal_stats_df.to_excel(writer, sheet_name='Ordinal Tests', index=False)
