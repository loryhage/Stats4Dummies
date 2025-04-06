"""
Created on June 18th 2023

Author: Lory H

Purpose: Univariate and multivariate analyses with odds ratios

Population : urologic injury vs no urologic injury

"""

import pandas as pd
import numpy as np
#stats related
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

"""Importing Excel sheets"""

#Set path of folder
path = '/Users/loryhage/Desktop/WORK/Research/1 Uro/UROplacenta/'

#Import data to dataframe
data = pd.read_excel (path+r'data/finaldatabase.xlsx') 
list(data.columns)

#replace "NA" with nan value in all columns
data.replace('na', np.nan, inplace=True)

#convert clavien dindo into float 

# For 3a and 3b to 31 and 32
data['clavien_dindo'] = np.where(data['clavien_dindo'] == '3a', 31, data['clavien_dindo'])
data['clavien_dindo'] = np.where(data['clavien_dindo'] == '3b', 32, data['clavien_dindo'])
# For '4a'/'4a\n' and '4b' to 41 and 42
data['clavien_dindo'] = np.where((data['clavien_dindo'] == '4a') | (data['clavien_dindo'] == '4a\n'), 41, data['clavien_dindo'])
data['clavien_dindo'] = np.where(data['clavien_dindo'] == '4b', 42, data['clavien_dindo'])

data['all_clavien_comp'] = np.where((data['clavien_dindo'] == 1) | (data['clavien_dindo'] == 2) | (data['clavien_dindo'] == 31) 
                                    | (data['clavien_dindo'] == 32)| (data['clavien_dindo'] == 41)| (data['clavien_dindo'] == 42) | (data['clavien_dindo'] == 5), 
                                    1, data['clavien_dindo'])

#Defining variables

group = 'urologic_injury'

continuous_var = ['hospit_postop','operative_time']

binary_var = ['delivery_type','conservative_csec','emergency_csec','hysterectomy_primary','hysterectomy_emergency','hysterectomy_secondary',
              'embolisation','blood_transfusion','ICU_all','vesical_invasion_preop','int_iliac_clipping','ureteral_catheter','cystoscopy',
              'vesical_invasion_intraop','balloon_catheter','death_m','cardiac_arrest','sepsis','renal_failure','haemoperitoneum',
              'uterine_rupture.1','vesical_rupture','calyceal_rupture','vesico_vag_fistula','iliac_vessel_injury',
              'vesical_suture_perop','partial_cystec','packing','thrombosis','occlusion_synd','PNA','pulmonary_embolism',
              'ureteral_reimp','hematuria','parietal_complications','clavien_dindo_binary', 'all_clavien_comp']
              
ordinal_var = ['EBL_cat', 'clavien_dindo']


"""Create MSKCC surgical complications categories"""

# Categories : cardiovascular, pulmonary, genitourinary, gastrointestinal, infection, parietal, general

# Cardiovascular : 'cardiac_arrest','haemoperitoneum','iliac_vessel_injury',int_iliac_clipping','thrombosis'
data['cardio_comp'] = np.where ((data['cardiac_arrest']==1) | (data['haemoperitoneum']==1 ) | (data['iliac_vessel_injury']==1)
                        | (data['int_iliac_clipping']==1) | (data['thrombosis']==1),
                        1,
                        0)

# Pulmonary : 'pulmonary_embolism'
data['pulmonary_comp'] = np.where (data['pulmonary_embolism']==1, 1, 0)

# Genitourinary : 'balloon_catheter', 'partial_cystec_postop','renal_failure','uterine_rupture.1',
# 'vesical_rupture','calyceal_rupture','vesico_vag_fistula','PNA','ureteral_reimp','hematuria'
data['urinary_comp'] = np.where ((data['balloon_catheter']==1 ) | (data['partial_cystec_postop']==1)
                        | (data['renal_failure']==1) | (data['uterine_rupture.1']==1) | (data['vesical_rupture']==1)
                        | (data['calyceal_rupture']==1) | (data['vesico_vag_fistula']==1) | (data['ureteral_reimp']==1)
                        | (data['PNA']==1) | (data['hematuria']==1),
                        1,
                        0)

# Gastrointestinal : 'occlusion_synd'
data['gastro_comp'] = np.where (data['occlusion_synd']==1, 1, 0)

# Infection : 'sepsis'
data['inf_comp'] = np.where (data['sepsis']==1, 1, 0)

# Parietal : 'parietal_complications'
data['parietal_comp'] = np.where (data['parietal_complications']==1, 1, 0)

# General : 'ICU_all', 'blood_transfusion', 'death_m'
data['general_comp'] = np.where ((data['ICU_all']==1) | (data['blood_transfusion']==1 ) | (data['death_m']==1),
                        1,
                        0)

# All complications
data['all_comp'] = np.where((data['cardio_comp']==1) | (data['pulmonary_comp']==1) | (data['urinary_comp']==1) | 
                    (data['gastro_comp']==1) | (data['inf_comp']==1) | (data['parietal_comp']==1) | (data['general_comp']==1),
                    1,
                    0)

# Variable to use
complications_var = ['cardio_comp', 'pulmonary_comp', 'urinary_comp', 'gastro_comp', 'inf_comp', 'parietal_comp', 'general_comp', 'all_comp']


"""Create function for converting list to dataframe"""
def ls_to_df(ls):
    new_df = []
    for df in ls :
        new_df += [pd.DataFrame(df)]
    return new_df

"""Calculate missing data and percentages"""
#count of missing data for each variable
null = []
for x in [binary_var, ordinal_var, continuous_var, complications_var]: 
    null += [data[x].isna().sum()]
binary_na, ordinal_na, continuous_na, complications_na = null

#percentage of missing data for each variable
per_null = []
for x in [binary_na, ordinal_na, continuous_na, complications_na]: 
    per_null += [(x/len(data.index))*100]
binary_perna, ordinal_perna, continuous_perna, complications_perna = per_null

"""Univariate analyses : Comparison of urologic vs non-urologic injury"""

# Descriptive statistical comparison between groups : dependant variable is a binary variable (urologic injury : yes/no)

# For independant binary variables : Chi-Square test

    #All binary variables : 
    #create contingency tables for each variable
cont_bin = []
for y in binary_var:
    x = 'urologic_injury'
    cont_bin += [data[[x, y]].pivot_table(index=x, columns=y, aggfunc=len).fillna(0).copy().astype(int)]

    #calculate chi-2 test
X_res = []
for x in cont_bin:
    X_res += [stats.chi2_contingency(x)] #results: chi2 statistic, p-value, degree of freedom
    
    #Surgical complications :
    #create contingency tables for each variable
cont_comp = []
for y in complications_var:
    x = 'urologic_injury'
    cont_comp += [data[[x, y]].pivot_table(index=x, columns=y, aggfunc=len).fillna(0).copy().astype(int)]

    #calculate chi-2 test
Xc_res = []
for x in cont_comp:
    Xc_res += [stats.chi2_contingency(x)] #results: chi2 statistic, p-value, degree of freedom


# For independant ordinal variables : Chi-Square test

    #create contingency tables for each variable
cont_ordinal = []
for y in ordinal_var :
    x = 'urologic_injury'
    cont_ordinal += [data[[x, y]].pivot_table(index=x, columns=y, aggfunc=len).fillna(0).copy().astype(int)]

    #calculate chi-2 test
Xh_res = []
for x in cont_ordinal:
    Xh_res += [stats.chi2_contingency(x)] #results: chi2 statistic, p-value, degree of freedom

"""Descriptive analysis of binary data"""

#Print n of groups
n = []
n += [{'n{}'.format(status): len(data[data['urologic_injury'] == status]) for status in [0,1]}]

#Calculate descriptive statistics for all binary data
count = []
for status in [0,1]:
    #Calculate count
    allcount = []
    for columns in binary_var:
        allcount += [data[columns][data['urologic_injury'] == status].eq(1).sum()]
    count += [allcount]
count_noinj, count_inj = count

#Calculate percentage of binary data in each group (dismiss missing data)
per_no = []
per_inj = []
for columns in binary_var:
    per_no += [(data[columns][data['urologic_injury'] == 0].eq(1).sum()) / (len(data[columns][data['urologic_injury'] == 0])) * 100]
    per_inj += [(data[columns][data['urologic_injury'] == 1].eq(1).sum()) / (len(data[columns][data['urologic_injury'] == 1])) * 100]


#Calculate descriptive statistics for surgical complications
count_comp = []
for status in [0,1]:
    #Calculate count
    allcount_comp = []
    for columns in complications_var:
        allcount_comp += [data[columns][data['urologic_injury'] == status].eq(1).sum()]
    count_comp += [allcount_comp]
count_comp_noinj, count_comp_inj = count_comp

#Calculate percentage of complications data in each group (dismiss missing data)
per_comp_no = []
per_comp_inj = []
for columns in complications_var:
    per_comp_no += [(data[columns][data['urologic_injury'] == 0].eq(1).sum()) / (len(data[columns][data['urologic_injury'] == 0])) * 100]
    per_comp_inj += [(data[columns][data['urologic_injury'] == 1].eq(1).sum()) / (len(data[columns][data['urologic_injury'] == 1])) * 100]


#Converting binary data into dataframe

#missing data

    #for count
df = ls_to_df([binary_na, continuous_na, complications_na])
df_binary,  df_continuous, df_complications = df
    #creating a dataframe of count
df_count= pd.concat(df)
df_count.columns = ['missing_count']
    #for percentage
df = ls_to_df([binary_perna, continuous_perna, complications_perna])
df_binary, df_continuous, df_complications = df
    #creating a dataframe of percentage
df_percentage = pd.concat(df)
df_percentage.columns = ['missing_percentage']
    #merging count and percentage df using index
df_missing = pd.merge(df_count, df_percentage, left_index=True, right_index=True)

#group data

    #for count
df = ls_to_df([count_noinj, count_inj, count_comp_noinj, count_comp_inj])
NoInj_binary_df, Inj_binary_df, NoInj_comp_df, Inj_comp_df = df

df_noinj_count = pd.concat([NoInj_binary_df, NoInj_comp_df], ignore_index=True)
df_inj_count = pd.concat([Inj_binary_df, Inj_comp_df], ignore_index=True)


df_noinj_count.columns = ['no_injury_count']
df_inj_count.columns = ['injury_count']

df_binary_c = pd.merge(df_noinj_count, df_inj_count, left_index=True, right_index=True)

    #for percentage
df = ls_to_df([per_no, per_inj, per_comp_no, per_comp_inj])
NoInj_binary_df, Inj_binary_df, NoInj_comp_df, Inj_comp_df = df

df_noinj_per = pd.concat([NoInj_binary_df, NoInj_comp_df], ignore_index=True)
df_inj_per = pd.concat([Inj_binary_df, Inj_comp_df], ignore_index=True)

df_noinj_per.columns = ['no_injury_percentage']
df_inj_per.columns = ['injury_percentage']

df_binary_p = pd.merge(df_noinj_per, df_inj_per, left_index=True, right_index=True)

    #merge count and percentage
df_binary = pd.merge(df_binary_c, df_binary_p,left_index=True, right_index=True)
allbinary_var = binary_var + complications_var
df_binary.insert(0, 'variables', allbinary_var)
df_binary.set_index('variables', inplace=True)

    #add pvalue into dataframe
    #for all binary
pvalue_bin = []
for p in X_res:
    pvalue_bin += [p.pvalue]
     #for surgical complications   
pvalue_comp = []
for p in Xc_res:
    pvalue_comp += [p.pvalue]
    #concat pvalue binary and complications
pvalue_all = pvalue_bin + pvalue_comp

    #merge all in one dataframe
df = pd.merge(df_missing, df_binary, left_index=True, right_index=True)
df['p-value'] = pvalue_all
df.insert(0, 'variables', allbinary_var)
df.set_index('variables', inplace=True)

#export dataframe to excel
df.to_excel(path+r'data/univariate_binary.xlsx')

"""Descriptive analysis of categorical data"""

#Create dataframe for count

df_col = []
for df in cont_ordinal :
    df_col += [pd.DataFrame(df).T]

    #calculate percentage
per_res = []
for x in [0,1]:
    allper = []
    for cat in df_col:
        per = []
        for row in range(len(cat)):
            per += [((cat[x].iloc[row]) / cat[x].sum())*100]
        allper += [per]
    per_res += [allper]

allper_no, allper_inj = per_res

 
#Create dataframe for percentage
df_per_no = []
for df in allper_no:
    df_per_no += [pd.DataFrame(df)]
df_per_inj = []
for df in allper_inj:
    df_per_inj +=[pd.DataFrame(df)]

    #merge percentage   
array = []
for x in range(len(df_per_no)):
    array += [[df_per_no[x], df_per_inj[x]]]
df_per = []
for df_no, df_inj in array:
    df_per += [pd.merge(df_no, df_inj, left_index=True, right_index=True)]
per_df = []
for df in df_per:
    df.columns = [0, 1]
    per_df += [pd.DataFrame(df)]
    #merge into one dataframe for each variable
array = []
for x in range(len(per_df)):
    array += [[df_col[x], per_df[x]]]
merged_df = []
for count, percentage in array :
     merged_df += [pd.concat([count, percentage.set_index(count.index[:len(percentage)])], axis=1)]
    #create dataframe for each variable
cat_df = []
for df in merged_df:
    df.columns = ['count_0', 'count_1', 'percentage_0', 'percentage_1']
    cat_df += [pd.DataFrame(df)]
    
 
#Combine all dataframes into one excel and export

dfs = cat_df
startrow = 0
with pd.ExcelWriter(path+r'data/univariate_cat.xlsx') as writer:
    for df in dfs:
        df.to_excel(writer, engine="xlsxwriter", startrow=startrow)
        startrow += (df.shape[0] + 2)

#Converting categorical stats into dataframe

    #pvalues

b = {'EBL_cat' : Xh_res[0].pvalue, 'clavien_dindo':Xh_res[1].pvalue}

cat_p = pd.DataFrame.from_dict(b, orient='index')
cat_p.columns = ['p-value']
cat_p = cat_p.fillna(value=np.nan)
    
    #add missing data
df_ordinal = pd.DataFrame(ordinal_na)
df_ordinal.columns = ['missing_count']
df_ordinal['missing_percentage'] = ordinal_perna
df_ordinal['p-value'] = cat_p
df_ordinal.insert(0, 'variables', ordinal_var)
df_ordinal.set_index('variables', inplace=True)

#to excel
df_ordinal.to_excel(path+r'data/univariate_cat_pvalue.xlsx')

"""Descriptive analysis of continuous variables"""

"""t-test if data is normal or mann-whitney if not"""

#Step 1: checking for normality

    #create function for normality
def normality (df, column):
    n = smf.ols(column +" ~ C(urologic_injury)", data= df).fit()
    shapiro = stats.shapiro(n.resid)
    return shapiro.pvalue

    #iterate function on all data
result = []
for col in continuous_var:
    result += [normality(data, col)]


    #group has a normal distribution = data is normal

for n in range(len(result)):
    name = continuous_var
    if result[n] > 0.05:
        print(name[n], 'p-value =', result[n],': data is normal')
    else:
        print (name[n], 'p-value =', result[n],': data is not normal - use a non parametric test')

    #distribution plot

for cortisol in continuous_var:
    sns.displot(data, x=cortisol, kind="kde")

#Step 2: checking variances 

column = continuous_var
var = []
for n in range(len(result)):
    col = data[column[n]].dropna()
    if result[n] > 0.05:
        #Bartlett for normal data
        var += [stats.bartlett(col[data['urologic_injury'] == 1],
                               col[data['urologic_injury'] == 0])]
    else:
        #Levene for non normal data
        var += [stats.levene(col[data['urologic_injury'] == 1],
                             col[data['urologic_injury'] == 0])]
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


#Step 3: non parametric tests
    
    # calculate mann-whitney test for non parametric data = all variables

M_res = []
for x in continuous_var:
    M_res += [stats.mannwhitneyu(data[x][data['urologic_injury'] == 0],
                                 data[x][data['urologic_injury'] == 1], 
                                 nan_policy='omit')] #returns: statistic, pvalue


"""Add descriptive statistics: mean sem"""

    #create function to calculate descriptive statistics (groups = 0, 1)
def desc_continuous(df, groups):
    #Print n of groups
    n = []
    n += [{'n{}'.format(status): len(df[df['urologic_injury'] == status]) for status in groups}]
    #Calculate descriptive statistics
    mean = []
    sem = []
    for status in groups:
        #Calculate means and sem of continuous data
        result1 = []
        result2 = []
        for columns in [continuous_var]:
            result1 += [df[columns][df['urologic_injury'] == status].mean(axis=0)]
            result2 += [df[columns][df['urologic_injury'] == status].sem(axis=0)]
        mean += [result1]
        sem += [result2]
        
    return n, mean, sem

res_continuous =  desc_continuous(data, [0,1])

    #seperate results into variables
continuous_n = res_continuous[0]
NoInj_cont_m, Inj_cont_m = res_continuous[1]
NoInj_cont_sem, Inj_cont_sem = res_continuous[2]

#Converting continuous data into dataframe

    #for mean
df = ls_to_df([NoInj_cont_m, Inj_cont_m])

df_result = []
for dx in df :
    df_result += [dx.T]
NoInj_cont_df, Inj_cont_df = df_result

NoInj_cont_df.columns = ['no_injury_mean']
Inj_cont_df.columns = ['injury_mean']

df_continuous_m = pd.merge(NoInj_cont_df, Inj_cont_df,left_index=True, right_index=True)

    #for SEM
df = ls_to_df([NoInj_cont_sem, Inj_cont_sem])

df_result = []
for dx in df :
    df_result += [dx.T]
NoInj_cont_df, Inj_cont_df = df_result

NoInj_cont_df.columns = ['no_injury_sem']
Inj_cont_df.columns = ['injury_sem']

df_continuous_sem = pd.merge(NoInj_cont_df, Inj_cont_df,left_index=True, right_index=True)

    #merge mean and sem
df_continuous = pd.merge(df_continuous_m, df_continuous_sem ,left_index=True, right_index=True)

#Add continuous results to dataframe : pvalue

pvalue_cont = []
for p in M_res:
    pvalue_cont += [p.pvalue]

    #merge all in one dataframe
df = pd.merge(df_missing, df_continuous, left_index=True, right_index=True)
df['p-value'] = pvalue_cont

#export dataframe to excel
df.to_excel(path+r'data/univariate_continuous.xlsx')

# count of total no_injury vs injury data : continuous_n to add in column title
print(continuous_n)



"""Multivariate analysis : odds ratios"""

#Impute missing data
def impute_data(df):
    df_imp = pd.DataFrame.copy(df)
    for col in list(df.columns):
        mean_col = df[col].mean()
        df_imp = df_imp.fillna(value = {col : mean_col})
    return df_imp, df_imp.isnull().sum()

    #use impute function
data_imp= impute_data(data)[0]

#Function to identify significant variables
def significant_values (table, var, pval):
    sig_val = []
    for x in range(len(table[var])): 
        if table[pval].iloc[x]<0.05:
            sig_val += [table[var][x]]
    return sig_val

    #identify significant descriptive variables
desc_cat = pd.read_excel(path+r'data/categorical_missing_pvalue.xlsx')
desc_cont = pd.read_excel(path+r'data/continuous_table.xlsx')

    #print significant variables into a list
sig_desc = []
for df in [desc_cat, desc_cont]:
    sig_desc += [significant_values(df, 'variables', 'p-value')]

    #merge lists into one
sig_d = []
for i in sig_desc:
    sig_d += [*i]

    #merge all variables to use into one dataframe
variables = continuous_var+binary_var+ordinal_var+complications_var+sig_d

#Logistic regression for odds ratios and CI
    
def log_model (data, endog, exog, name):
    #define: independant variables = data.exog (confounders) & dependant variable = data.endog (variable)
    x = data[exog]
    y = data[endog]
    #fit Logit model
    lm = sm.Logit(y, x)
    model = lm.fit(method='newton')
    model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR_'+name])
    model_odds['p-value_'+name]= model.pvalues
    model_odds[['2.5%_'+name, '97.5%_'+name]] = np.exp(model.conf_int())
    return  model_odds #return adjusted OR


    #Get univariate odds ratios
uni_OR = []
for var in variables:
    uni_OR += [log_model(data_imp, 'urologic_injury', var, 'not_adjusted')]

uni_var = pd.concat(uni_OR) #create dataframe with all results
uni_var['variables'] = uni_var.index #rename index column


#Select variables to include in multivariate analysis

    #identify significant univariate variables
sig_ls = significant_values(uni_var,'variables','p-value_not_adjusted')

    #Get multivariate odds ratios
adj_var = log_model(data_imp, 'urologic_injury', sig_ls, 'adjusted')

    #merge univariate and multivariate results
multivar = pd.merge(uni_var, adj_var, left_index=True, right_index=True)

    #create column for significant results
for OR in multivar['OR_adjusted']:
    multivar['significant'] = ['significant' if pval <= 0.05 else 'not significant' for pval in multivar['p-value_adjusted']]


#export dataframe to excel
multivar.to_excel(path+r'data/multivariate_analysis.xlsx')

#Create plot of Odds Ratios
fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(6, 4), dpi=150)
for idx, row in adj_var.iloc[::-1].iterrows():
    ci = [[row['OR_adjusted'] - row[::-1]['2.5%_adjusted']], [row['97.5%_adjusted'] - row['OR_adjusted']]]
    if row['p-value_adjusted'] <= 0.05 :
        plt.errorbar(x=[row['OR_adjusted']], y=[row.name], xerr=ci,ecolor='tab:red', capsize=3, 
                     linestyle='None', linewidth=1, marker="o", markersize=5, mfc="tab:red", mec="tab:red")
    else: 
        plt.errorbar(x=[row['OR_adjusted']], y=[row.name], xerr=ci, ecolor='tab:gray', capsize=3, 
                     linestyle='None', linewidth=1, marker="o", markersize=5, mfc="tab:gray", mec="tab:gray")
plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.xlabel('Odds Ratio and 95% Confidence Interval', fontsize=8)
plt.tight_layout()
plt.xlim([-5, 20])
plt.savefig(path+r'figures/adjusted_OR.png')
plt.show()

