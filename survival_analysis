"""
Created on April 5 2025

Author: Lory H

Purpose: Calculatin survival analyses using Kaplan Meier and Cox HR

"""

import pandas as pd
from datetime import datetime
from datetime import date
import numpy as np
import seaborn as sns

#survival analysis
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from ISLP.models import ModelSpec as MS
from lifelines.plotting import add_at_risk_counts

"""Importing Excel sheets"""
#Import data to dataframe
data = pd.read_excel (database)

#replace "NA" with nan value in all columns
data.replace('na', np.nan, inplace=True)

"""Calculate median time to event"""

LOCF = []
for i in range(len(data.patient_id)):                                              
    LOCF += [                                                               #calculate date to last follow up
        float("{:.2f}".format(                                              #keep 2 decimals
            (                                                               #convert to datetime format
                (datetime.date(data.x32.iloc[i])
                 - datetime.date(data.interv_date.iloc[i])).days            #output in days
                ) / (365.25)
            
        ))]

data['LOCF'] = LOCF
LOCF_med = data['LOCF'].median(axis=0)

"""Calculate time to 1st event or last follow up"""

def last_observation (df, subjects, event, last_event, last_observation, first_observation) :
    LOCF = []
    for i in range(len(df[subjects])):
        # if there is a complication
        if df[event].iloc[i] == 1 :                                                  
            LOCF += [                                                               #calculate date to complication
                float("{:.2f}".format(                                              #keep 2 decimals
                    (                                                               #convert to datetime format
                        (datetime.date(df[last_event].iloc[i])
                         - datetime.date(df[first_observation].iloc[i])).days          #output in days
                        ) / (365.25)
                    
                ))]
        else:                                                                       #if there is no complication
            LOCF += [                                                               #calculate date to last follow up
                float("{:.2f}".format(                                              #keep 2 decimals
                    (                                                               #convert to datetime format
                        (datetime.date(df[last_observation].iloc[i])
                         - datetime.date(df[first_observation].iloc[i])).days            #output in days
                        ) / (365.25)
                    
                ))]
    
    #create dictionary with all data
    a = {'patient_id': df[subjects], 'timetoevent_'+event : LOCF}
    results_df = pd.DataFrame.from_dict(a)
    results_df = results_df.fillna(value=np.nan)
    return results_df

df = data.copy()
for comp in ['event_type1', 'event_type2', 'event_type3'] :
   LOCF_df = last_observation(df, 'patient_id', comp, 'dateofevent', 'dateoflastfollowup', 'dateofstart')
   # Merge with data
   df = pd.merge(df, LOCF_df, on='patient_id', how='inner')
   df['timetoevent_'+comp] = np.where(
       df['timetoevent_'+comp] >= 0,
       df['timetoevent_'+comp],
       np.nan) 
   df = df.dropna(subset=['timetoevent_'+comp])

#Create column of the two groups according to one variable
df['group_variable'] = np.where(
    df['variable'] == 1,
    'group1',
    'group2')

# Export database for survival analysis into excel
df.to_excel(path+r'tables/data_survival.xlsx')

"""Kaplan Meier survival analysis"""

# Create a function of KM curve

def kaplan_curves (df, timetoevent, status, event, x, y, label_x, label_y) :
    # status is the variable analysed ; x is the data of group 1 (text or number) ; y is the data of group 2 (text or number)
    # label_x is the label of group 1 (text) ; label_y is the label of group 2 (text)
    
    # fix size of plots
    fig, ax = subplots(figsize=(8,8))
    
    # fit curve of group 1
    f1 = KaplanMeierFitter()
    f1.fit(df[timetoevent][df[status]== x], df[event][df[status]== x])    
    f1.plot(label= label_x, ax=ax, show_censors=True, censor_styles={'ms': 5, 'marker': '|'} , color='#0a2883', ci_alpha=0.1)
    
    # fit curve of group 2
    f2 = KaplanMeierFitter()
    f2.fit(df[timetoevent][df[status]== y], df[event][df[status]== y])    
    f2.plot(ax=ax, label=label_y, show_censors=True, censor_styles={'ms': 5, 'marker': '|'}  , color='#809fff', ci_alpha=0.1)
    
    # fix legend position
    ax.legend(loc='upper right', fontsize=12, frameon=False)
    
    # format plots and add labels/titles
    plt.xlabel("Time to Event (years)", **{'fontname':'Arial'},**{'fontsize':14})
    plt.ylabel("Survival Probability", **{'fontname':'Arial'},**{'fontsize':14})
    plt.xticks(**{'fontname':'Arial'}, **{'fontsize':12})
    plt.yticks(**{'fontname':'Arial'}, **{'fontsize':12})
    plt.axhline(0.5, color='black', linestyle='--', label='50% Survival')
    
    # add number of events under curves
    add_at_risk_counts(f1, f2,labels=[x+' group', y+' group'], 
                       **{'fontname':'Arial'}, **{'fontsize':12}, **{'horizontalalignment':'center'},
                       ax=ax, fig=fig)
        
    # compare survival analysis using Log rank test
    by_inter = {}
    for i, dg in df.groupby('intervention'):
        by_inter[i] = dg
    log_results = logrank_test(by_inter[x][timetoevent],
                               by_inter[y][timetoevent],
                               by_inter[x][event],
                               by_inter[y][event])
    return plt.tight_layout(), f1, f2, log_results.summary
# Create a function that uses KM function for each type of event

def survival_curves (data, LOCF) :
    curves = []
    stats = []
    median = []
    
    for complication in ['event_type1', 'event_type2', 'event_type3']:
        [curve, f1, f2, logrank] = kaplan_curves(data, 'timetoevent_'+complication, 'group_variable', complication, 'group1', 'group2', 'label1', 'label2')
        curves += [curve]
        plt.savefig(path+r'figures/'+complication+'_km.png', dpi=600)
        
        # Calculate survival percentage at median time and store in dictionary
        median.append({
            "complication": complication,
            group1+"_event_percentage": survival_at_median(f1),
            group2+"_event_percentage": survival_at_median(f2)
        })
        # Save into excel sheet
        median_df = pd.DataFrame(median)
        median_df.to_excel(path + r'tables/survival_medians.xlsx', index=False)
        
        # Save log rank results summary into excel sheet
        logrank.index = [complication] * len(logrank) # Add complication name as index
        stats += [logrank]
        logrank_df = pd.concat (stats)
        logrank_df.to_excel(path+r'tables/logrank.xlsx')
        
    return curves

def survival_at_median (kf) :
    # Calculate the percentage of patients with the event
    survival_probability = kf.predict(LOCF_med)
    # Calculate the percentage of patients with the event
    event_percentage = (1 - survival_probability) * 100
    # # Get the time point where survival drops to 50%
    # median_survival_time = kf.median_survival_time_  
    return event_percentage

# Use function to output all survival results

survival_results = survival_curves(df, LOCF_med)

"""Cox proportional hazards regression model"""

#Dataframe to include in analysis : impute missing data

# Convert variable into binary results and add into column
df['status'] = np.where(
    df['group_variable'] == 'group1',
    1,
    0)

# Define variables to include in univariate and multivariate analyses
variables = ['event','x', 'y', 'z']

inter_df = df[variables]
model_df = MS(variables, intercept=False).fit_transform(inter_df)

#Impute missing data
def impute_data(df):
    df_imp = pd.DataFrame.copy(df)
    for col in list(df.columns):
        mean_col = df[col].mean()
        df_imp = df_imp.fillna(value = {col : mean_col})
    return df_imp, df_imp.isnull().sum()

    #use impute function
imputed_df = impute_data(model_df)[0]
    #check variables for missing data
imputed_df.isnull().sum()
    
"""univariate HR"""

covariates = [x for x in variables if x != 'timetoevent' and x !='event']

uni_results = []
for var in covariates :
    cph = CoxPHFitter
    subset = imputed_df[[var, 'timetoevent', 'event']]
    uni_results += [cph(penalizer=0.1).fit(subset, duration_col='timetoevent', event_col='event')]
summary = []
for uni_HR in uni_results : 
    summary += [uni_HR.summary]
  
# Combine the results into a single DataFrame
univariate_summary = pd.concat(summary)
univariate_summary.to_excel(path+r'tables/Cox_univariate.xlsx')

"""multivariate HR"""

cph = CoxPHFitter
cox_fit = cph(penalizer=0.1).fit(imputed_df,'timetoevent','event')

results = pd.DataFrame(cox_fit.summary) # output corresponds to : exp(coef) = HR with CI 95% = exp(coef) lower 95% and exp(coef) upper 95%
# export into excel sheet
results.to_excel(path+r'tables/Cox_multivariate.xlsx')

    # Plot adjusted curves
plt.subplots(figsize=(10, 6))
cox_fit.plot()
plt.tight_layout()
plt.savefig(path+r'figures/CoxHR.png', dpi=300)

    # Plot partial effects on outcome (Cox-PH Regression)
    
cox_fit.plot_partial_effects_on_outcome(covariates = 'status', values = [1,0], cmap = 'tab20', plot_baseline=False)
plt.xlabel("Time to Event (years)", **{'fontname':'Arial'},**{'fontsize':12})
plt.ylabel("Survival Probability", **{'fontname':'Arial'},**{'fontsize':12})
plt.xticks(**{'fontname':'Arial'}, **{'fontsize':10})
plt.yticks(**{'fontname':'Arial'}, **{'fontsize':10})

plt.savefig(path+r'figures/Cox_intervention.png', dpi=600)
