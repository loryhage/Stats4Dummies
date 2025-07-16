
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

# Descriptive data for each group
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

"""Comparison of two groups : group1 vs group2"""

# Independant ordinal variables : Chi-square test

    # Create contingency tables for each variable
cont = []
for y in ordinal_var:
    x = group #dependant variable (binary data)
    cont += [data[[x, y]].pivot_table(index=x, columns=y, aggfunc=len).fillna(0).copy().astype(int)]

    # Calculate chi-2 test
Xh_res = []
for x in cont:
    Xh_res += [stats.chi2_contingency(x)] #output results: chi2 statistic, p-value, degree of freedom

ord_1, ord_2, ord_3 = Xh_res

    # Calculate percentage
allper_group1 = []
allper_group2 = []
for cat in df_col:
    per_group1 = []
    per_group2 = []
    sum_cat = []
    for row in range(len(cat)):
        per_group1 += [((cat[0].iloc[row]) / cat[0].sum())*100]
        per_group2 += [((cat[1].iloc[row]) / cat[1].sum())*100]
    allper_group1 += [per_group1]
    allper_group2 += [per_group2]

# Create dataframe for count and percentages of ordinal variables
  # Count
df_col = []
for df in cont :
    df_col += [pd.DataFrame(df).T]
  # Percentage
df_per_group1 = []
for df in allper_group1:
    df_per_group1 += [pd.DataFrame(df)]

df_per_inj = []
for df in allper_group2:
    df_per_group2 +=[pd.DataFrame(df)]

    # Merge percentages into one dataframe
array = []
for x in range(len(df_per_no)):
    array += [[df_per_group1[x], df_per_group2[x]]]

df_per = []
for df_group1, df_group2 in array:
    df_per += [pd.merge(df_group1, df_group2, left_index=True, right_index=True)]

per_df = []
for df in df_per:
    df.columns = [0, 1]
    per_df += [pd.DataFrame(df)]

  # Merge percentages into one dataframe for each variable
array = []
for x in range(len(per_df)):
    array += [[df_col[x], per_df[x]]]

  # Merge count and percentages into one dataframe
merged_df = []
for count, percentage in array :
     merged_df += [pd.concat([count, percentage.set_index(count.index[:len(percentage)])], axis=1)]

  # Create dataframe for each variable
cat_df = []
for df in merged_df:
    df.columns = ['count_0', 'count_1', 'percentage_0', 'percentage_1']
    cat_df += [pd.DataFrame(df)]
    
  # Combine all dataframes into one excel and export
dfs = cat_df
startrow = 0
with pd.ExcelWriter(path+r'data/categoricaltable.xlsx') as writer:
    for df in dfs:
        df.to_excel(writer, engine="xlsxwriter", startrow=startrow)
        startrow += (df.shape[0] + 2)

# Converting categorical data into dataframe

b = {'ordinal_var_1':ord_1.pvalue,'ordinal_var_2': ord_2.pvalue,'ordinale_var_3': ord_3.pvalue}
cat_p = pd.DataFrame.from_dict(b, orient='index')
cat_p.columns = ['p-value']
cat_p = cat_p.fillna(value=np.nan)
    
    #missing data
df_ordinal = pd.DataFrame(ordinal_na)
df_ordinal.columns = ['missing_count']
df_ordinal['missing_percentage'] = ordinal_perna
df_ordinal['p-value'] = cat_p
df_ordinal.index.names = ['variables']

#to excel
df_ordinal.to_excel(path+r'data/categorical_missing_pvalue.xlsx')

# Independant continuous variables : t-Test if data is normal, otherwise Mann-Whitney

  # Step 1: checking for normality

  # Create a function to calculate normality
def normality (df, column):
    n = smf.ols(column +" ~ C(group)", data= df).fit()
    shapiro = stats.shapiro(n.resid)
    return shapiro.pvalue

  # Iterate function on all data
result = []
for col in continuous_var :
    result += [normality(data, col)]

for n in range(len(result)):
    name = continuous_var
    if result[n] > 0.05:
        print(name[n], 'p-value =', result[n],': data is normal')
    else:
        print (name[n], 'p-value =', result[n],': data is not normal - use a non parametric test')

  # Create distribution plot
for cortisol in continuous_var :
    sns.displot(data, x=cortisol, kind="kde")

  # Step 2: checking variances 

column = continuous_var
var = []
for n in range(len(result)):
    col = data[column[n]].dropna()
    if result[n] > 0.05:
        # Bartlett for normal data
        var += [stats.bartlett(col[data[group] == 1],
                               col[data[group] == 0])]
    else:
        # Levene for non normal data
        var += [stats.levene(col[data[group] == 1],
                             col[data[group] == 0])]
print(var)

variances = []
for n in range(len(var)):
    variances += [var[n].pvalue]
       
    #all groups have equal variances = homogeneity
for n in range(len(variances)):
    col = continuous_var
    if variances[n] > 0.05:
        print(col[n],'variances are homogenous')
    else:
        print (col[n],'variances are not homogenous - use a non parametric test')

  # Step 3: parametric and non parametric tests
    
  # Mann-Whitney test for non parametric data

M_res = []
for x in ['nonp_var1','nonp_var2']:
    M_res += [stats.mannwhitneyu(data[x][data[group] == 0],
                                 data[x][data[group] == 1], 
                                 nan_policy='omit')] #returns: statistic, pvalue
var1, var2 = M_res

    # T-test for parametric data
for x in ['par_var1', 'par_var2']:
    T_res = stats.ttest_ind(data[x][data[group] == 0],
                           data[x][data[group] == 1],
                           nan_policy = 'omit') #returns: statistic, pvalue
var3, var4 = T_res

  # Add continuous results to dataframe : pvalue

a = {'var1' : var1.pvalue, 'var2' : var2.pvalue, 'var3' : var3.pvalue, 'var4' : var4.pvalue}
cont_p = pd.DataFrame.from_dict(a, orient='index')
cont_p.columns = ['p-value']

# Create dataframe with descriptive data

  # Calculate descriptive results (mean, sem) 
res_continuous =  desc_continuous(data, [0,1])

  # Separate results into variables
continuous_n = res_continuous[0]
Group1_cont_m, Group2_cont_m = res_continuous[1]
Group1_cont_sem, Group2_cont_sem = res_continuous[2]

  # Convert continuous data into dataframe    
df = ls_to_df([Group1_cont_m, Group2_cont_m]) #mean results

df_result = []
for dx in df :
    df_result += [dx.T]
group1_cont_df, group2_cont_df = df_result

group1_cont_df.columns = ['group1_mean']
group2_cont_df.columns = ['group2_mean']

df_continuous_m = pd.merge(group1_cont_df, group2_cont_df,left_index=True, right_index=True)

df = ls_to_df([NoInj_cont_sem, Inj_cont_sem]) #SEM results

df_result = []
for dx in df :
    df_result += [dx.T]
group1_cont_df, group2_cont_df = df_result

group1_cont_df.columns = ['group1_sem']
group2_cont_df.columns = ['group2_sem']

df_continuous_sem = pd.merge(group1_cont_df, group2_cont_df,left_index=True, right_index=True)

  # Merge mean and sem results into one dataframe
df_continuous = pd.merge(df_continuous_m, df_continuous_sem ,left_index=True, right_index=True)

  # Merge all results in one dataframe
df = pd.merge(df_continuous_missing, df_continuous, left_index=True, right_index=True, how='outer')
df_final = pd.merge(df, cont_p,left_index=True, right_index=True, how='outer')
df_final.index.names = ['variables']

  # Export dataframe to excel
df_final.to_excel(path+r'data/continuous_table.xlsx')
