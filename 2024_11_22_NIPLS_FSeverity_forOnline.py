#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:19:23 2023

@author: Lise Hobeika
"""
#%% Load packages and database
import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import seaborn as sns
import pingouin as pg
import itertools
from scipy import stats
from scipy.stats import zscore,  spearmanr
from pySankey.sankey import sankey
from sklearn.model_selection import  cross_validate#, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn import cross_decomposition , feature_selection , linear_model # impute, , , metrics

warnings.filterwarnings('ignore')

UKB = pd.read_csv('/Users/lise/ownCloud/Postdoc_acouphènes/Manips/UKBiobank/Data/2024_11_05_UKB_tinnitus.csv')

#%% PREPARE THE DATASET TO RUN THE MODEL

## Select all the features used in the model, divided by categories
Hearing        = ['Hearing_Difficulties_Self_Reported','Hearing_test_right','Hearing_test_left','Hearing_Difficulties_with_Background_noises','Hearing_Aid_Users','Cochlear_Implants', 'Noisy_Workplace','Loud_Music_Exposure']
Mood           = ['Frequency_Tiredness','Frequency_Depressed_Mood','Frequency_Tenseness','Frequency_Unenthousiasm', 'Mood_consultation_GP','Mood_consultation_Psychiatrist','Risk_Taking']
Neuroticism    = ['Worrier', 'Fed_up', 'Mood_swings', 'Miserableness', 'Loneliness_isolation', 'Guilty_Feelings',  'Tense',  'Suffer_from_nerves',  'Irritability', 'Sensitivity', 'Worry_too_long_after_embarrassment','Nervous_Feelings', 'Total_Neuroticism']
Life_Stressors = ['Serious_Trauma_to_Self', 'Serious_Trauma_to_Close_Relative', 'Death_of_Close_Relative', 'Death_of_Spouse_or_Partner', 'Martial_Separation_or_Divorce', 'Financial_Difficulties' , 'Life_Stressors_Last2years']
Sleep          = ['Insomnia', 'Difficulty_Getting_Up', 'Narcolepsy',  'Napping', 'Evening_person','Morning_person',  'Sleep_duration']  
Substance_Use  = ['Alcohol_Intake_comparison', 'Previous_Drinker','Past_Smoking', 'Daily_Smoker','Previous_Smoker','Hours_Home_Exposure_Smoking','Ever_smoked', 'Household_Smokers', 'Never_Drank','Occasional_Smoker',  'Alcohol_Intake'] 
Anthropometric = ['BMI', 'Weight', 'Gained_Weight','Lost_Weight','Systolic_Pressure', 'Diastolic_Pressure' , 'Pulse_Rate', 'Fractured_Broken_Bones'] 
Demographics   = ['White', 'Mixed', 'Asian','Black','Other','Age' ,'Sex'] 
Socioeconomics = ['Number_Vehicles', 'Household_Incomes', 'Living_with_Granchildren','Number_Houselhold','Able_to_confide', 'Living_with_Partner','Living_with_Related_Relatives','Living_with_Children','Living_with_Siblings','Living_with_Grandparents','Living_with_Unrelated_Relatives','Living_with_Parents','Frequency_social_visits']                   
Occupational   = [ 'Job_physical_work','Job_walking','Paid_Employment','Retired','Looking_home_or_family','Unable_work_sick_disable','Unemployed','College_University', 'Advanced_level', 'Ordinary_level', 'Certificate_secondary_education', 'Practical_career_diploma', 'Other_professional_qualifications']  
Physical_activity = ['Moderate_Activity_Min_per_Week', 'IPAQ_Low_Activity','IPAQ_High_Activity', 'At_or_Above_Activity_Recommendation'  , 'IPAQ_Moderate_Activity'  , 'Walking_Min_per_Week' , 'Hand_Grip_Strength']  

## MERGE THE CATEGORIES
categories_domains     = [Hearing,    Mood,   Neuroticism,   Life_Stressors,   Sleep,   Anthropometric,   Substance_Use,   Physical_activity,  Occupational,  Demographics,   Socioeconomics ]
categories_names       = ['Hearing', 'Mood' ,'Neuroticism' ,'Life Stressors' ,'Sleep', 'Anthropometric', 'Substance Use' ,'Physical activity','Occupational' ,'Demographics','Socioeconomics']
target_columns         =  Hearing  +  Mood  + Neuroticism  + Life_Stressors  + Sleep +  Anthropometric +  Substance_Use  + Physical_activity + Occupational  + Demographics + Socioeconomics

categories_nbvariables = [len(Hearing), len(Mood) , len(Neuroticism), len(Life_Stressors), len(Sleep), len(Anthropometric), len(Substance_Use), len(Physical_activity), len(Occupational), len(Demographics), len(Socioeconomics)   ]
categories_colors      =['#ddab0f', '#840000', '#bb3f3f', '#f4320c' , '#3e85c0', '#a2cffe', '#5a7d9a', '#0a437a', '#0a481e', '#65ab7c' , '#0a888a']


### Split data
# CREATE THE TRAINING DATASET # RAJOUTER UN FILTRE SUR LE FAIT QU'ON VEUT LES DONNEES DE FREQUENCE AU MEME MOMENT
dtrain0 = UKB[ (~UKB['Tinnitus_Severity_T0'].isna()) & (~UKB['Tinnitus_Severity_T0'].isna()) & (UKB['Tinnitus_Frequency_T1'].isna()) & (UKB['Tinnitus_Frequency_T2'].isna()) & (UKB['Tinnitus_Frequency_T3'].isna()) ]
target_columns_dtrain0 = [col + '_T0' for col in target_columns] + ['Tinnitus_Frequency_T0', 'Tinnitus_Severity_T0']
new_column_names = {col: col.replace('_T0', '') for col in target_columns_dtrain0}
dtrain0.rename(columns=new_column_names, inplace=True)

dtrain1 = UKB[ ( UKB['Tinnitus_Severity_T0'].isna()) & (~UKB['Tinnitus_Severity_T1'].isna()) & (~UKB['Tinnitus_Frequency_T1'].isna()) & (UKB['Tinnitus_Frequency_T2'].isna()) & (UKB['Tinnitus_Frequency_T3'].isna()) ]
target_columns_dtrain1 = [col + '_T1' for col in target_columns] + ['Tinnitus_Frequency_T1', 'Tinnitus_Severity_T1']
new_column_names = {col: col.replace('_T1', '') for col in target_columns_dtrain1}
dtrain1.rename(columns=new_column_names, inplace=True)

dtrain2 = UKB[ ( UKB['Tinnitus_Frequency_T0'].isna()) & (UKB['Tinnitus_Frequency_T1'].isna())  & (~UKB['Tinnitus_Severity_T2'].isna()) & (~UKB['Tinnitus_Frequency_T2'].isna()) & (UKB['Tinnitus_Frequency_T3'].isna()) ]
target_columns_dtrain2 = [col + '_T2' for col in target_columns] + ['Tinnitus_Frequency_T2', 'Tinnitus_Severity_T2']
new_column_names = {col: col.replace('_T2', '') for col in target_columns_dtrain2}
dtrain2.rename(columns=new_column_names, inplace=True)

dtrain3 = UKB[ ( UKB['Tinnitus_Frequency_T0'].isna()) & (UKB['Tinnitus_Frequency_T1'].isna())  & (UKB['Tinnitus_Frequency_T2'].isna()) & (~UKB['Tinnitus_Severity_T3'].isna()) & (~UKB['Tinnitus_Frequency_T3'].isna()) ]
target_columns_dtrain3 = [col + '_T3' for col in target_columns] + ['Tinnitus_Frequency_T3', 'Tinnitus_Severity_T3']
new_column_names = {col: col.replace('_T3', '') for col in target_columns_dtrain3}
dtrain3.rename(columns=new_column_names, inplace=True)

df_train_s = pd.concat([dtrain0, dtrain1, dtrain2, dtrain3], ignore_index=True)
df_train_s = df_train_s[ target_columns + ['Tinnitus_Frequency']  + ['Tinnitus_Severity'] ]


# CREATE THE TESTING DATASET
# PArticipants starting on T0

dtest01 = UKB[ (~UKB['Tinnitus_Severity_T0'].isna()) & (~UKB['Tinnitus_Severity_T1'].isna()) & (~UKB['Tinnitus_Frequency_T0'].isna()) & (~UKB['Tinnitus_Frequency_T1'].isna())  ]
dtest02 = UKB[ (~UKB['Tinnitus_Severity_T0'].isna()) & (UKB['Tinnitus_Frequency_T1'].isna()) & (~UKB['Tinnitus_Severity_T2'].isna()) & ( ~UKB['Tinnitus_Frequency_T0'].isna()) & (~UKB['Tinnitus_Frequency_T2'].isna()) ]
dtest03 = UKB[ (~UKB['Tinnitus_Severity_T0'].isna()) & (UKB['Tinnitus_Frequency_T1'].isna()) & (UKB['Tinnitus_Frequency_T2'].isna()) & (~UKB['Tinnitus_Severity_T3'].isna())  & ( ~UKB['Tinnitus_Frequency_T0'].isna()) & (~UKB['Tinnitus_Frequency_T3'].isna())]


target_columns_dtest0_baseline = [col + '_T0' for col in target_columns] + ['Tinnitus_Frequency_T0', 'Tinnitus_Severity_T0']
new_column_names = {col: col.replace('_T0', '') for col in target_columns_dtest0_baseline}
dtest01.rename(columns=new_column_names, inplace=True)
dtest02.rename(columns=new_column_names, inplace=True)
dtest03.rename(columns=new_column_names, inplace=True)

dtest01.rename(columns={'Tinnitus_Frequency_T1': 'Tinnitus_Frequency_FollowUp', 'Tinnitus_Severity_T1': 'Tinnitus_Severity_FollowUp'}, inplace=True)
dtest02.rename(columns={'Tinnitus_Frequency_T2': 'Tinnitus_Frequency_FollowUp', 'Tinnitus_Severity_T2': 'Tinnitus_Severity_FollowUp'}, inplace=True)
dtest03.rename(columns={'Tinnitus_Frequency_T3': 'Tinnitus_Frequency_FollowUp', 'Tinnitus_Severity_T3': 'Tinnitus_Severity_FollowUp'}, inplace=True)

# Participants starting on T1
dtest12 = UKB[ ( UKB['Tinnitus_Frequency_T0'].isna()) & (~UKB['Tinnitus_Severity_T1'].isna()) & (~UKB['Tinnitus_Severity_T2'].isna()) & (~UKB['Tinnitus_Frequency_T1'].isna()) & (~UKB['Tinnitus_Frequency_T2'].isna()) ]
dtest13 = UKB[ ( UKB['Tinnitus_Frequency_T0'].isna()) & (~UKB['Tinnitus_Severity_T1'].isna()) & (UKB['Tinnitus_Frequency_T2'].isna()) & (~UKB['Tinnitus_Severity_T3'].isna())  & (~UKB['Tinnitus_Frequency_T1'].isna()) & (~UKB['Tinnitus_Frequency_T3'].isna())]

target_columns_dtest1_baseline = [col + '_T1' for col in target_columns] + ['Tinnitus_Frequency_T1', 'Tinnitus_Severity_T1']
new_column_names = {col: col.replace('_T1', '') for col in target_columns_dtest1_baseline}
dtest12.rename(columns=new_column_names, inplace=True)
dtest13.rename(columns=new_column_names, inplace=True)

dtest12.rename(columns={'Tinnitus_Frequency_T2': 'Tinnitus_Frequency_FollowUp', 'Tinnitus_Severity_T2': 'Tinnitus_Severity_FollowUp'}, inplace=True)
dtest13.rename(columns={'Tinnitus_Frequency_T3': 'Tinnitus_Frequency_FollowUp', 'Tinnitus_Severity_T3': 'Tinnitus_Severity_FollowUp'}, inplace=True)


# Participants starting on T2
dtest23 = UKB[ ( UKB['Tinnitus_Frequency_T0'].isna()) & (UKB['Tinnitus_Frequency_T1'].isna()) & (~UKB['Tinnitus_Severity_T2'].isna()) & (~UKB['Tinnitus_Severity_T3'].isna()) & ( ~UKB['Tinnitus_Frequency_T2'].isna()) & (~UKB['Tinnitus_Frequency_T3'].isna())]
target_columns_dtest1_baseline = [col + '_T2' for col in target_columns] + ['Tinnitus_Frequency_T2', 'Tinnitus_Severity_T2']
new_column_names = {col: col.replace('_T2', '') for col in target_columns_dtest1_baseline}
dtest23.rename(columns=new_column_names, inplace=True)
dtest23.rename(columns={'Tinnitus_Frequency_T3': 'Tinnitus_Frequency_FollowUp', 'Tinnitus_Severity_T3': 'Tinnitus_Severity_FollowUp'}, inplace=True)


df_test_s = pd.concat([dtest01, dtest02, dtest03, dtest12, dtest13, dtest23 ], ignore_index=True)
df_test_s = df_test_s[ target_columns + ['Tinnitus_Severity'] + ['Tinnitus_Severity_FollowUp']  + ['Tinnitus_Frequency'] + ['Tinnitus_Frequency_FollowUp'] ]


## REMOVE ROWS WITH MORE THAN 20% MISSING VALUES / FILL MISS VALUES BY MEDIAN
threshold_train    = int(df_train_s.shape[1] * 0.20)
df_train_s_cleaned = df_train_s[df_train_s.isnull().sum(axis=1) <= threshold_train]

threshold_test = int(df_test_s.shape[1] * 0.20)
df_test_s_cleaned = df_test_s[df_test_s.isnull().sum(axis=1) <= threshold_test]

# Merge the training and testing datasets
df_combined = pd.concat([df_train_s_cleaned, df_test_s_cleaned], axis=0)

#% Change the number categories for tinnitus frequency 
# The category 'I used to have' will be changed by a new one : I don't right now, that Include 'Never' +  'I used to have'
# In term of correspondance:  
    # we had 0:never, 1:used to, 2: sometimes, 3: most of the time, 4: All the time
    # we changed to   0:Not now, 1: sometimes, 2: most of the time, 3: All the time

df_combined['Tinnitus_Frequency'] = df_combined['Tinnitus_Frequency'].replace(1,np.nan)
df_combined['Tinnitus_Frequency'] = df_combined['Tinnitus_Frequency'].replace(2,1)
df_combined['Tinnitus_Frequency'] = df_combined['Tinnitus_Frequency'].replace(3,2)
df_combined['Tinnitus_Frequency'] = df_combined['Tinnitus_Frequency'].replace(4,3)


df_combined['Tinnitus_Frequency_FollowUp'] = df_combined['Tinnitus_Frequency_FollowUp'].replace(1,0)
df_combined['Tinnitus_Frequency_FollowUp'] = df_combined['Tinnitus_Frequency_FollowUp'].replace(2,1)
df_combined['Tinnitus_Frequency_FollowUp'] = df_combined['Tinnitus_Frequency_FollowUp'].replace(3,2)
df_combined['Tinnitus_Frequency_FollowUp'] = df_combined['Tinnitus_Frequency_FollowUp'].replace(4,3)


# Replace the missing values per median
for column in target_columns:
    median_value = df_combined[column].median()
    df_combined[column].fillna(median_value, inplace=True)

# Zscore the entire dataset
df_combined_zscore                 = df_combined.copy()
df_combined_zscore[target_columns] = df_combined[target_columns].apply(zscore)


# Split the datasets back
df_train_s = df_combined_zscore.iloc[:len(df_train_s_cleaned), :]
df_test_s  = df_combined_zscore.iloc[len(df_train_s_cleaned):, :]

# Remove NA after removing particitipans who Used to have tinnitus
df_train_s = df_train_s [ ~df_train_s['Tinnitus_Frequency'].isna() ]
df_test_s  = df_test_s  [ ~df_test_s[ 'Tinnitus_Frequency'].isna() ]


#%% ###### RUN THE MODEL ON TINNITUS SEVERITY ######

# Define the data included in the model
X_train = df_train_s[target_columns]
y_train = df_train_s['Tinnitus_Severity']

X_test  = df_test_s[target_columns]
y_test  = df_test_s['Tinnitus_Severity']


### Create a PLS pipeline with standard scaling
pls = PLSRegression(n_components = 5)
scaler = StandardScaler()
pls_pipe = Pipeline([('pls', pls)])

# Cross validation
cv_results = cross_validate(pls_pipe, X_train, y_train, cv = 10, scoring='r2', return_estimator=True)

# Get the weights from the trained PLS models
weights      = [estimator.named_steps['pls'].coef_       for estimator in cv_results['estimator']]
loadings     = [estimator.named_steps['pls'].x_loadings_ for estimator in cv_results['estimator']]


# Create dataframe with the weights and loadings of the model
average_weights = np.mean(weights, axis=0)
weights_sev    = pd.DataFrame([target_columns,average_weights[0]],index=['names','Weights']).T

loadings_mean   = np.mean(np.stack([np.mean(loadings[i],axis=1) for i in range(10)],axis=1),axis=1)
loadings_freq   = pd.DataFrame([target_columns,loadings_mean],index=['names','Loading']).T

# Compute a signature for each subject using the dot product of the averaged weights and raw feature values
signature = np.dot(X_test, average_weights.T)
df_test_s['signature'] = signature


#%% FIGURE 2.A - WEIGHTS OF THE TINNITUS SEVERITY MODEL 
## Plot the weights organized by categories and size

categories_colors =['#ddab0f', '#840000', '#bb3f3f', '#f4320c' , '#3e85c0', '#a2cffe', '#5a7d9a', '#0a437a', '#0a481e', '#65ab7c' , '#0a888a']

# Initialize info for plot
fig, ax = plt.subplots(figsize=(5, 30))  # Set the initial size
list_x  = list()  # List of the column names for all the categories, which will be arranged in function of the weights
x       = list( range(0, len(weights_sev)))
istart  = 0 # index at which the new category begin in the list weights_sev.names

for i in range(0, len(categories_names)):
    
    common_indices = range(istart, istart + categories_nbvariables[i] ) 
    istart = istart  + categories_nbvariables[i]
    
    # Find the indices  
    icat_names   = weights_sev.names[common_indices]
    icat_weights = weights_sev.Weights[common_indices]
    
    # Arrange weight in descending order  
    sorted_pairs = sorted(zip(icat_weights, icat_names), reverse = True)
    sorted_weights, sorted_names =  zip(*sorted_pairs)

    # Create the dataframe  
    y = ([None] * len(weights_sev))
    for index, value in zip(common_indices, sorted_weights):
        y[index] = value
     
    modified_names = [name.replace('_T0', '').replace('_', ' ') for name in sorted_names]
    list_x = list_x  + list(modified_names[:]) # On fait une liste des noms des colonnes, qu'on rajoutera à la fin sur le plot

    # y[common_indices]  = weights_sev.Weights[common_indices]
    dataplot = pd.DataFrame({'x' : x, 'y' : y})
    
    # Plot the first set of bars at x-values 1, 2, 3, 4
    sns.barplot(x='y', y='x', data=dataplot,orient='horizontal', color=categories_colors[i])

fs = 18
plt.yticks(range(len(list_x)), list_x, fontsize = fs)
plt.xticks( fontsize = fs + 4)
sns.despine()
ax.tick_params(left=True, bottom=True, length=10, width=2), 
plt.xlabel('')



#%% FIGURE 1.B - ROC-AUC per level of presence
## Compute ROC curve and AUC

cat   = ['Mild','Moderate','Severe']
list_roc_full_model = []

# Create the ROC curve plot
plt.figure(figsize = (5,5))
fig, axs = plt.subplots(ncols=1, nrows=1,  figsize=(4, 4)) 

for i in [3,2,1]:

    condition = (y_test == 0) | (y_test == i)
    y_roc     = np.where(y_test[condition] == i, 1, 0)
    sign_roc  = df_test_s.signature[condition]
    
    fpr, tpr, _ = roc_curve(y_roc, sign_roc)
    roc_auc = auc(fpr, tpr)
    lw = 2  
    plt.plot(fpr, tpr,  lw=lw, label= cat[i-1] + ' = %0.2f' % auc(fpr, tpr),
    color  = sns.color_palette("seismic", 6)[i+2]) 
    list_roc_full_model.append(roc_auc)


fs = 14
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = fs)
plt.ylabel('True Positive Rate', fontsize = fs)
plt.title('Receiver Operating Characteristic', fontsize = fs)
plt.legend(loc='lower right', fontsize = fs)
plt.xticks(fontsize = fs)
plt.yticks(fontsize = fs)

# Show the plot
axs.tick_params(left=True, bottom=True), axs.spines[['top', 'right']].set_visible(False)#, axs.legend(frameon=False)
plt.legend(loc='lower right')  # Add a legend for the ROC curves


#%% Calculate Cohen's D for each level of tinnitus severity (mild, moderate, severe), compare to no distress

c = list()

for i in range(1,4):
    sig_notin      = df_test_s.signature[ ( y_test == 0)]
    sig_tin        = df_test_s.signature[ ( y_test == i)]
    cohen_d        = pg.compute_effsize(sig_tin, sig_notin, eftype='cohen')
   
    print(i,cohen_d)
    
    
#%% FIGURE 1.C - Variance explained by categories

list_category, list_category_p, l  = [], [] , []
ds_weights       = pd.DataFrame(average_weights, columns = target_columns)

for icat in range(0, len(categories_domains)): 
    icat_name     = categories_names[icat]
    icat_features = categories_domains[icat]
    isig          = np.dot(df_test_s[icat_features], ds_weights[icat_features].T)
    icorr         = stats.pearsonr(isig.flatten()  , df_test_s.Tinnitus_Severity)

    list_category.append(icorr[0]**2), list_category_p.append(icorr[1])  


sorted_pairs = sorted(zip(list_category, categories_names, categories_colors))
sorted_list_category, sorted_categories_names, sorted_categories_colors =  zip(*sorted_pairs)
s_categories = pd.Series(sorted_list_category, index = sorted_categories_names)
s_cat = s_categories

# And calculate the full explained variance
corr = stats.pearsonr(df_test_s['signature'],df_test_s['Tinnitus_Severity'] ) #Dot product with the selected weights, and compute correlation
print('Total Explained variance', round(100*corr[0]**2, 1))

#### FIGURE 1.C Explained Variance by Categories ####
fig, axs = plt.subplots(ncols=1, nrows=1,  figsize=(5,6)) 
axs.barh(y = s_cat.index, width = s_cat.values, color = sorted_categories_colors )
axs.set_xticks( [0, .02, .04]), axs.set_xticklabels(['0%', '2%', '4%'], fontsize = 16)
axs.xaxis.set_label_position('top'), axs.xaxis.tick_top()
axs.set_xlabel('Explained Variance by Category', fontsize = 16), axs.set_yticklabels(s_categories.index, fontsize = 16)
axs.spines[['right', 'bottom']].set_visible(False), axs.tick_params(left=True, top = True, length=4, width=1)


#%% FIGURE 1.D - ROC-AUC removing each category one by one

y_train = df_train_s['Tinnitus_Severity']
y_test  = df_test_s['Tinnitus_Severity']

categories_domains  = [Hearing ,    Mood ,   Neuroticism,   Life_Stressors,   Sleep,   Anthropometric,   Substance_Use,   Physical_activity,  Occupational,  Demographics,   Socioeconomics ]
cat                 = ['Mild','Moderate','Severe']


categories = [
    "Full Model", "Hearing", "Mood", "Neuroticism", "Life stressors", "Sleep",
    "Anthropometric", "Substance Use", "Physical Activity", "Occupational", "Demographics", "Socioeconomics", "4cats removed"
]

severe_values, moderate_values, mild_values   = [list_roc_full_model[0]],  [list_roc_full_model[1]],  [list_roc_full_model[2]]

# Run the model removing one category of features, to test its effects on the ROC-AUC Curves
for i in range(0, len(categories_domains)+1): 
    
    if i == len(categories_domains):
        icat = [Hearing ,    Mood ,   Neuroticism , Sleep]
        icat = list(itertools.chain(*icat))
    else:
    # Definition of the columns of the features, without the domain imputed
        icat = categories_domains[i]
        
    target_columns_imputed = list(set(target_columns) - set(icat))
    
    X_train = df_train_s[target_columns_imputed]
    X_test  = df_test_s[target_columns_imputed]
    
    ### Create a PLS pipeline with standard scaling
    pls = PLSRegression(n_components = 5)
    scaler = StandardScaler()
    pls_pipe = Pipeline([('pls', pls)])

    # Cross validation
    cv_results = cross_validate(pls_pipe, X_train, y_train, cv = 10, scoring='r2', return_estimator=True)

    # Get the weights from the trained PLS models
    weights      = [estimator.named_steps['pls'].coef_       for estimator in cv_results['estimator']]    
    average_weights_simplified = np.mean(weights, axis=0)
    

    # Compute a signature for each subject using the dot product of the averaged weights and raw feature values
    df_test_s['signature_simplified'] = np.dot(X_test, average_weights_simplified.T)
        
    list_roc = []
   
    for j in [3,2,1]:
    
        condition = (y_test == 0) | (y_test == j)
        y_roc     = np.where(y_test[condition] == j, 1, 0)
        sign_roc  = df_test_s.signature_simplified[condition]
        
        fpr, tpr, _ = roc_curve(y_roc, sign_roc)
        roc_auc = auc(fpr, tpr)
        list_roc.append(roc_auc)
    
    # Adding AUC-ROCs value to each category
    severe_values.append(list_roc[0])
    moderate_values.append(list_roc[1])
    mild_values.append(list_roc[2])  


# Preparation of the data for a stacked barplot
moderate_modified = np.array(moderate_values) - np.array(mild_values)  # 'A lot' minus 'Some'
severe_modified   = np.array(severe_values) - np.array(mild_values) - moderate_modified  # 'All' minus 'Some' and 'A lot'

# Convert categories to indices
x_indices = np.arange(len(categories))

# Create the figure and axis
plt.figure(figsize=(10, 6))
bar_width = 0.6

# Create the stacked bars
plt.bar(x_indices, mild_values,       width=bar_width, label='Some',  color=sns.color_palette("seismic", 6)[3])
plt.bar(x_indices, moderate_modified, width=bar_width, label='A lot', color=sns.color_palette("seismic", 6)[4], bottom = mild_values)
plt.bar(x_indices, severe_modified,   width=bar_width, label='All',   color=sns.color_palette("seismic", 6)[5], bottom = np.array(mild_values) + moderate_modified)

# Customize the plot
plt.xlabel('Categories')
plt.ylabel('ROC AUC')
plt.title('Stacked Bar Plot for Categories')
plt.ylim(0.50, 0.80)  # Adjust x-axis limits for better visualization if needed
plt.xticks(x_indices, categories, rotation=90, ha='right')  # Rotate category names for better readability
plt.legend()



#%% ###### LONGITUDINAL EVALUATION ##########
### FIGURE 1.E - Sankey Plot

UKB_longitudinal = df_test_s.copy()

# Category mapping and order
category_order = ['No', 'Mild', 'Moderate', 'Severe']
category_mapping = {i - 1: label for i, label in enumerate(category_order, 1)}



# Map and categorize data
UKB_longitudinal['Tinnitus_Severity'] = pd.Categorical(
    UKB_longitudinal['Tinnitus_Severity'].map(category_mapping),
    categories=category_order,
    ordered=True
)

UKB_longitudinal['Tinnitus_Severity_FollowUp'] = pd.Categorical(
    UKB_longitudinal['Tinnitus_Severity_FollowUp'].map(category_mapping),
    categories=category_order,
    ordered=True
)



# Map the index of count_T0 to category_order
index_mapping  = {0.0: 'No', 1.0: 'Mild', 2.0: 'Moderate', 3.0: 'Severe'}
count_T0  = UKB_longitudinal['Tinnitus_Severity'].value_counts()
count_T0  = count_T0.reindex(category_order)
labels_T0 = [f"{cat}\n({count:,})" for cat, count in zip(category_order, count_T0)]


count_T2  = UKB_longitudinal['Tinnitus_Severity_FollowUp'].value_counts()
count_T2  = count_T2.reindex(category_order)
labels_T2 = [f"{cat}\n({count:,})" for cat, count in zip(category_order, count_T2)]


UKB_longitudinal['Tinnitus_Severity'] = UKB_longitudinal['Tinnitus_Severity'].replace(
    dict(zip(category_order, labels_T0))
)

UKB_longitudinal['Tinnitus_Severity_FollowUp'] = UKB_longitudinal['Tinnitus_Severity_FollowUp'].replace(
    dict(zip(category_order, labels_T2))
)


# # Reorder 'Tinnitus_Severity' based on labels_T0
# UKB_longitudinal['Tinnitus_Severity'] = pd.Categorical(
#     UKB_longitudinal['Tinnitus_Severity'],
#     categories=labels_T0,
#     ordered=True
# )

# # Reorder 'Tinnitus_Severity' based on labels_T2
# UKB_longitudinal['Tinnitus_Severity_FollowUp'] = pd.Categorical(
#     UKB_longitudinal['Tinnitus_Severity_FollowUp'],
#     categories=labels_T2,
#     ordered=True
# )

# Color mapping
color_palette = sns.color_palette('seismic', 8)[4:8]#[::-1]
color_dict = {**dict(zip(labels_T0, [mpl.colors.to_hex(c) for c in color_palette])),
              **dict(zip(labels_T2, [mpl.colors.to_hex(c) for c in color_palette]))}

# Sankey plot
sankey(
    left=UKB_longitudinal['Tinnitus_Severity'], 
    right=UKB_longitudinal['Tinnitus_Severity_FollowUp'], 
    colorDict=color_dict
)


# Set figure size
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
sns.set(font="Arial", style='white')
fs = 8 
plt.gcf().set_size_inches(3, 6)



#%% ###### LONGITUDINAL EVALUATION ##########
### FIGURE 1.E - Sankey Plot

UKB_longitudinal = df_test_s.copy()

# Category mapping and order
category_order = ['No', 'Mild', 'Moderate', 'Severe']
category_mapping = {i - 1: label for i, label in enumerate(category_order, 1)}


# Map and categorize data
UKB_longitudinal['Tinnitus_Severity'] = pd.Categorical(
    UKB_longitudinal['Tinnitus_Severity'].map(category_mapping),
    categories=category_order,
    ordered=True
)

UKB_longitudinal['Tinnitus_Severity_FollowUp'] = pd.Categorical(
    UKB_longitudinal['Tinnitus_Severity_FollowUp'].map(category_mapping),
    categories=category_order,
    ordered=True
)

# Reorder the columns to match the specified category order
UKB_longitudinal['Tinnitus_Severity'] = pd.Categorical(UKB_longitudinal['Tinnitus_Severity'], categories=category_order, ordered=True)
UKB_longitudinal = UKB_longitudinal.sort_values(by='Tinnitus_Severity')



# Map the index of count_T0 to category_order
index_mapping  = {0.0: 'No', 1.0: 'Some of the time', 2.0: 'A lot of the time', 3.0: 'All the time'}
count_T0  = UKB_longitudinal['Tinnitus_Severity'].value_counts()
count_T0  = count_T0.reindex(category_order)
labels_T0 = [f"{cat}\n({count:,})" for cat, count in zip(category_order, count_T0)]


count_T2  = UKB_longitudinal['Tinnitus_Severity_FollowUp'].value_counts()
count_T2  = count_T2.reindex(category_order)
labels_T2 = [f"{cat}\n({count:,})" for cat, count in zip(category_order, count_T2)]


UKB_longitudinal['Tinnitus_Severity'] = UKB_longitudinal['Tinnitus_Severity'].replace(
    dict(zip(category_order, labels_T0))
)

UKB_longitudinal['Tinnitus_Severity_FollowUp'] = UKB_longitudinal['Tinnitus_Severity_FollowUp'].replace(
    dict(zip(category_order, labels_T2))
)

# Color mapping
color_palette = sns.color_palette('seismic', 8)[4:8]
color_dict = {**dict(zip(labels_T0, [mpl.colors.to_hex(c) for c in color_palette])),
              **dict(zip(labels_T2, [mpl.colors.to_hex(c) for c in color_palette]))}


# Sankey plot
sankey(
    left  = UKB_longitudinal['Tinnitus_Severity'], 
    right = UKB_longitudinal['Tinnitus_Severity_FollowUp'], 
    colorDict = color_dict
)

# Set figure size
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
sns.set(font="Arial", style='white')
fs = 8 
plt.gcf().set_size_inches(3, 6)



#%% FIGURE 1.F - Adjusted Risk Score across levels of Tinnitus evolution ####
## We adjust the risk scores to avoid biases (people who have tinnitus all the time are more likely to recover, than the contrary)
# Compute residuals of the risk score after controlling for number of the level of tinnitus presence at baseline

categories = ['Never', 'Some of the time', 'A lot of the time', 'All the time']

# Compute the change in Tinnitus Severity
df_test_s['Severity_evolution'] = df_test_s['Tinnitus_Severity_FollowUp'] - df_test_s['Tinnitus_Severity']

# Label for Change in Tinnitus presence level= -3: total recovery, 3: apparition of tinnitus all the time
Change = ['-3', '-2', '-1', '0', '1', '2', '3'] 

# Calculate the link between tinnitus presence and the signature, then adjust create the Signature residuals by removing the cubic regression
lr       = linear_model.LinearRegression()
reg_cub  = lr.fit(df_test_s['Tinnitus_Severity'].array.reshape(-1, 1)**2, df_test_s['signature'].array.reshape(-1, 1)).predict(df_test_s['Tinnitus_Severity'].array.reshape(-1, 1)**2)

df_test_s['Signature_residuals_cub'] = stats.zscore(df_test_s['signature'] - reg_cub[:,0])


### FIGURE ###
color_palette = sns.color_palette('RdBu', 8)[4:8][::-1]  +  sns.color_palette('seismic', 6)[3:6]

fig, axs = plt.subplots(ncols=1, nrows=1,  figsize=(6,6))
sns.barplot(ax = axs, data = df_test_s, x = 'Severity_evolution', y = 'Signature_residuals_cub', palette = color_palette )
axs.set_ylim(-0.55, 1.8), axs.set_yticks(np.arange(-1.2, 1.8, 0.4))
# axs.set_xticklabels(Change, fontsize = fs-2)
axs.set_ylabel('Adjusted Severity Risk Score', fontsize = fs), axs.set_xlabel('Tinnitus Severity Evolution', fontsize = fs)
axs.spines['bottom'].set_bounds((0, 8)), axs.spines['left'].set_bounds((-1.2, 1.8))
axs.tick_params(left=True, bottom=True, length=3, width=1), axs.spines[['top', 'right']].set_visible(False)
axs.spines[['top', 'left']].set_linewidth(1)
axs.tick_params(axis='y', labelsize=fs-2)


#%% FIGURE 1.G - Effect Sizes across Levels of Tinnitus evolution ####

## Evaluate Adjusted Risk Score - Evolution of tinnitus
# Estimating Cohen's d + AUC-ROC between People withtout tinnitus and those who evolved in tinnitus presence
# Apply 10,000 boostraps

def cohen_d(x,y): #Cohen_d with pooled standard deviation
        return (np.mean(x) - np.mean(y)) / math.sqrt(((x.shape[0] - 1) * np.std(x, ddof=1) + (y.shape[0] - 1) * np.std(y, ddof=1)) / (x.shape[0] + y.shape[0] - 2))

# Apply 10,000 boostraps
list_roc, list_cohen = [], [[], [],[]]
no_dist = df_test_s[(df_test_s.Tinnitus_Severity==0) & (df_test_s.Tinnitus_Severity_FollowUp==0)]

no_dist.insert(0, 'Group', 0)

rng = np.random.default_rng()

for i in np.arange(-3,4,1):
    distress = df_test_s[ (df_test_s['Severity_evolution']== i) ]
    distress.insert(0, 'Group', 1)
    list_cohen[0].append(cohen_d(distress.Signature_residuals_cub, no_dist.Signature_residuals_cub))
    bs = stats.bootstrap((distress.Signature_residuals_cub, no_dist.Signature_residuals_cub), cohen_d, confidence_level=0.95,method='percentile',random_state=rng,n_resamples=10000,vectorized=False)
    list_cohen[1].append(bs.confidence_interval[0]), list_cohen[2].append(bs.confidence_interval[1])
    no_dist  = df_test_s[(df_test_s.Severity_evolution==0) & (df_test_s.Tinnitus_Severity==0)]
    distress = df_test_s[ (df_test_s['Severity_evolution']== i) ]

    y_test_roc = df_test_s['Severity_evolution']
    condition  = (y_test_roc== 0) | (y_test_roc==i)
    y_roc = y_test_roc[condition]
    y_roc = np.where(y_test_roc[condition]==i,1,0)
    sig = df_test_s.Signature_residuals_cub
    sign_roc = sig[condition]
    
    if i<0: 
        y_roc = abs(y_roc-1)
        
    fpr, tpr, _ = roc_curve(y_roc, sign_roc)
    roc_auc = auc(fpr, tpr)
    list_roc.append(roc_auc)

Cohen_D = pd.DataFrame(data = list_cohen, index = ['Estimate', 'Low', 'High'], columns = Change).T
AUC_ROC = pd.DataFrame(data = list_roc, columns = ['AUC-ROC'], index = Change).T

#### FIGURE  ####
fig, axs = plt.subplots(ncols=1, nrows=2,  figsize=(5,10), gridspec_kw={'width_ratios': [1], 'height_ratios': [0.85, 0.15]})
fs=20
color1, color2 = sns.color_palette("RdBu", 7)[-1], sns.color_palette("seismic", 7)[-1]

# Cohen's D
for i,c in zip([0,3], [color1, color2]): 
    i2, xlabels = i+4, np.arange(-3,4,1)
    axs[0].fill_between(xlabels[i:i2], Cohen_D.Low[i:i2], Cohen_D.High[i:i2], color = c, alpha = 0.5)
    axs[0].plot(xlabels[i:i2], Cohen_D['Estimate'][i:i2], color = c)
axs[0].axvline(x = 0, ls = '-', color = 'black', lw = 1), axs[0].axhline(y = 0, ls = '--', color = 'grey', lw = 1)
axs[0].set_ylim((-1.3, 1.5)), axs[0].set_yticks(np.arange(-1.3, 1.5, 0.4)), axs[0].set_yticklabels(np.arange(-1.3, 1.5, 0.4).round(1), fontsize = fs)
axs[0].set_xticks(np.arange(-3,4,1))
axs[0].set_ylabel("Cohen's d (Evolution vs No Evolution)", fontsize = fs) #, axs[0].set_xlabel('Severity evolution', fontsize = fs)
axs[0].spines['bottom'].set_bounds((-3,3)), axs[0].spines['left'].set_bounds((-2,1.5))

# AUC-ROC
color_palette = sns.color_palette('RdBu', 8)[4:8][::-1]  +  sns.color_palette('seismic', 6)[3:6]
sns.barplot(ax = axs[1], data = AUC_ROC, palette = color_palette )
axs[1].set_ylim((0.5, 0.8)), axs[1].set_ylabel('AUC-ROC', fontsize = fs)
axs[1].set_yticks([0.50, 0.60, 0.70, 0.80]), axs[1].set_yticklabels([0.50, 0.60, 0.70, 0.80], fontsize = fs) ,  axs[1].set_xlabel('Tinnitus Severity Evolution', fontsize = fs)
for i in range(2):
    axs[i].set_xticklabels(Change, fontsize = fs)
    axs[i].tick_params(left=True, bottom=True, length=3, width=1) , axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].spines[['top', 'left']].set_linewidth(1)



#%% ###### SIMPLIFIED RISK SCORE ############
## Features selection
# Warning: this part is very long to run 

columns_to_remove = ['Total_Neuroticism', 'Hand_Grip_Strength', 'Hearing_test_right','Hearing_test_left', 'White', 'Mixed', 'Asian','Black','Other','Sex','Age' ,  'Household_Incomes']#,] 
target_columns_srs = [col for col in target_columns if col not in columns_to_remove]

X_train = df_train_s[target_columns_srs]
y_train = df_train_s['Tinnitus_Severity']

X_test  = df_test_s[target_columns_srs]
y_test  = df_test_s['Tinnitus_Severity']

plsreg = cross_decomposition.PLSRegression(n_components = 1, scale = False, max_iter=10000).fit(X_train, y_train)

range_score_tested = range(1,12)
nb_features, list_r = [], []

for ifeatures in range_score_tested: 
    sfs_forward   = feature_selection.SequentialFeatureSelector(plsreg, n_features_to_select=ifeatures, direction="backward").fit(X_train, y_train)
    list_features = list(X_train.columns[sfs_forward.get_support()])
    
    # Calculate the new signature for the short model
    weights_freq_srs = pd.DataFrame(plsreg.coef_.T, target_columns_srs)  
    selected_rows    = weights_freq_srs.loc[list_features].values.tolist()
    signature_srs    = np.dot(X_test[list_features], selected_rows)
    nb_features.append(ifeatures)

    r = spearmanr(signature_srs, y_test)
    list_r.append(r[0])
    
    print(ifeatures, list_features)
    
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(6, 8))
plt.plot(range_score_tested , list_r)
plt.xticks(range_score_tested)
plt.grid(True)


#%%  Calculate the Simplified risk score : look at the prediction accuracy with binarized values  

###### Look at the Features selected by the algorithm on the previous section
# 6 selected features
categories_short = ['Hearing_Difficulties_Self_Reported', 'Hearing_Difficulties_with_Background_noises', 'Frequency_Tiredness', 'Mood_swings', 'Sensitivity', 'Insomnia']

weights_short_binarized = [1, 1, 1, 1, 1, 1]

# This analysis is performed on the dataset without scoring. Therefore, we re-defined the datasets for traning and testing without the zscoring
UKB_short_test = df_combined.iloc[len(df_train_s_cleaned):, :]
UKB_short_test = UKB_short_test [ ~UKB_short_test['Tinnitus_Frequency'].isna() ]

X_test_short   = UKB_short_test[categories_short]

# Binarized the answers
X_test_short.Frequency_Tiredness  = [1 if x > 2 else 0 for x in X_test_short.Frequency_Tiredness]
X_test_short.Insomnia             = [1 if x > 2 else 0 for x in X_test_short.Insomnia]

# Calculate the simplified signature
UKB_short_test['Sig_short_bin'] = np.dot(X_test_short[categories_short], weights_short_binarized)

#%% FIGURE 3.B : Simplified risk score at baseline and its association with tinnitus Severity at follow UP- Odd Ratios

categories_severity = ['Severe', 'Moderate', 'Mild','No']
categories_RS       = ['0-1\nLow Risk', '2-3\nModerate Risk', '4-5\nHigh Risk', '6\nVery High Risk']

clf = linear_model.LogisticRegression(penalty='none')
CP_array, CP_array_risk, D_array, D = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4,4)), np.arange(4) 

for j in [3,2,1,0]:
    for i in[0,2,4]: # For loop
        dT0 = np.where((UKB_short_test['Sig_short_bin'] == i) |( UKB_short_test['Sig_short_bin'] == i+1) , 1, 0).reshape(-1, 1)
        dT2 = np.where(UKB_short_test['Tinnitus_Severity_FollowUp']==j, 1, 0)
        clf.fit(dT0 , dT2) 
        CP_array[3 - j, round(i/2) ] = clf.coef_[0][0] # (Betas, i.e. log odds) are append     
    
    dT0 = np.where( (UKB_short_test['Sig_short_bin'] == 6) , 1, 0).reshape(-1, 1)
    dT2 = np.where(UKB_short_test['Tinnitus_Severity_FollowUp'] == j, 1, 0)
    clf.fit(dT0 , dT2) 
    CP_array[3 - j, 3 ] = clf.coef_[0][0] # (Betas, i.e. log odds) are append     


idx, col = pd.Series([i  for i in categories_severity], name = 'Severity at T2'), pd.Series([i  for i in categories_RS], name = 'ShortRiskScore')
Sev_OR   = pd.DataFrame(data=np.exp(CP_array), index = idx, columns = col)

col_palette = LinearSegmentedColormap.from_list("custom", sns.color_palette('seismic', 100)[50:100] )

## Visualisation of odds ratios
fs = 12
fig, axs = plt.subplots(ncols=1, nrows=1,  figsize=(4, 4))
sns.heatmap( data = Sev_OR, fmt='.1f', cmap = col_palette, square=True, vmin=0, vmax=4, linewidths=0.25, annot=True, annot_kws={"fontsize": 8}, cbar = True, cbar_kws =  {'label':'Tinnitus Severity OR','ticks':np.asarray([0,4]), 'shrink':.75, 'use_gridspec':False, 'location':'top'})
plt.xlabel('Reduced Risk Score\nat Baseline', fontsize = fs)
plt.ylabel('Tinnitus Severity\nat Follow-up', fontsize = fs)


#%% FIGURE 3.C : Simplified risk score and its association with tinnitus Severity - BARPLOT

category_order_sev   = ['No','Mild','Moderate','Severe']
category_mapping_sev = {i-1: line for i, line in enumerate(category_order_sev, 1)}
 
fs = 12
fig, ax = plt.subplots(1, 1,figsize=(8,8))
sns.boxplot(x='Tinnitus_Severity', y='Sig_short_bin' , data = UKB_short_test ,palette = sns.color_palette('seismic', 8)[4:8] ) # palette = sns.color_palette('RdBu', 10)[6:10] ) #palette = sns.color_palette('RdBu', 10)[6:10] ) #, ])
ax.set_xticklabels(category_order_sev)
ax.set_ylabel('Reduced Risk Score'), ax.set_xlabel('Tinnitus Severity')


##### Calulate the ROC_AUC for each category
list_roc = []

y_test = UKB_short_test.Tinnitus_Severity
risk_scores_short_bin =  UKB_short_test.Sig_short_bin
for i in [3,2,1]:
    condition  = (y_test== 0) | (y_test==i)
    y_roc = y_test[condition]
    y_roc = np.where(y_test[condition]==i,1,0)
    sign_roc = risk_scores_short_bin[condition]
    
    fpr, tpr, _ = roc_curve(y_roc, sign_roc)
    roc_auc = auc(fpr, tpr)
    lw = 2  # Line width
    # plt.plot(fpr, tpr,  lw=lw, label= cat[i-1] + ' = %0.2f' % auc(fpr, tpr), color  = sns.color_palette("seismic", 6)[i+2])
    list_roc.append(roc_auc)
    print('ROC-AUC',category_order_sev[i], round(roc_auc, 3))

#%% FIGURE 3.D : Adjusted Simplified Risk Score across levels of Tinnitus evolution ####

# Compute the change in Tinnitus Severity
df_test_s['Severity_evolution'] = df_test_s['Tinnitus_Severity_FollowUp'] - df_test_s['Tinnitus_Severity']

# Label for Change in Tinnitus presence level= -3: total recovery, 3: apparition of tinnitus all the time
Change = ['-3', '-2', '-1', '0', '1', '2', '3'] 


# Calculate the link between tinnitus presence and the signature, then adjust create the Signature residuals by removing the cubic regression
lr      = linear_model.LinearRegression()
reg_cub = lr.fit(UKB_short_test['Tinnitus_Severity'].array.reshape(-1, 1)**2, UKB_short_test['Sig_short_bin'].array.reshape(-1, 1)).predict(UKB_short_test['Tinnitus_Severity'].array.reshape(-1, 1)**2)

UKB_short_test['Signature_residuals_cub'] = stats.zscore(pd.to_numeric(UKB_short_test['Sig_short_bin'] - reg_cub[:, 0], errors='coerce') )
UKB_short_test['Signature_residuals_cub'] = (UKB_short_test['Sig_short_bin'] - reg_cub[:,0] )

# Risk Score Evolution (Baseline & Follow-Up)
Risk_scores_res_mat, Risk_scores_res_cub_mat = np.zeros((4,4)), np.zeros((4,4))

for i in range(4): # For loop
    for j in range(4):
        # Use boolean indexing to select rows that meet the condition
        selected_rows = UKB_short_test[(UKB_short_test['Tinnitus_Severity'] == i) & (UKB_short_test['Tinnitus_Severity_FollowUp'] == j)]               
        Risk_scores_res_cub_mat[j, i] = np.mean(selected_rows['Signature_residuals_cub'])

col, idx = pd.Series(data=categories, name = 'Tinnitus Severitys\nat Baseline'), pd.Series(data=categories[::-1], name = 'Tinnitus Severity\nat Follow-Up')
Frequency_Spread_Res, Frequency_Spread_Res_Cub = pd.DataFrame(data = Risk_scores_res_mat[::-1], columns = col, index = idx), pd.DataFrame(data = Risk_scores_res_cub_mat[::-1], columns = col, index = idx)


#### FIGURE B2. Adjusted Risk Score across Change 
color_palette = sns.color_palette('RdBu', 8)[4:8][::-1]  +  sns.color_palette('seismic', 6)[3:6]

fig, axs = plt.subplots(ncols=1, nrows=1,  figsize=(3, 3))
sns.barplot(ax = axs, data = UKB_short_test, x = 'Severity_evolution', y = 'Signature_residuals_cub', palette = color_palette)
axs.set_xticklabels(Change)
axs.set_ylabel('Adjusted Short Risk Score'), axs.set_xlabel('Tinnitus Severity Evolution')
axs.tick_params(left=True, bottom=True, length=3, width=1), axs.spines[['top', 'right']].set_visible(False)
axs.spines[['top', 'left']].set_linewidth(1)



#%% FIGURE 3.E - Effect Sizes across Levels of Tinnitus evolution ####
## Evaluate the Cohen_d on the EVOLUTION of the SEVERITY
# Estimating Cohen's d + AUC-ROC -  Apply 10,000 boostraps
def cohen_d(x,y): #Cohen_d with pooled standard deviation
        return ( (np.mean(x) - np.mean(y)) / math.sqrt(((x.shape[0] - 1) * np.std(x, ddof=1) + (y.shape[0] - 1) * np.std(y, ddof=1)) / (x.shape[0] + y.shape[0] - 2)) )

# Apply 10,000 boostraps
list_roc, list_cohen = [], [[], [],[]]
no_dist = UKB_short_test[(UKB_short_test.Tinnitus_Severity==0) & (UKB_short_test.Tinnitus_Severity_FollowUp==0)]

#no_dist
no_dist.insert(0, 'Group', 0)
#CPF.insert(0, 'Group', 0)

rng = np.random.default_rng()

for i in np.arange(-3,4,1):
 #   dist  = UKB_T2[UKB_T2['ChronicSpread']==s]
    distress = UKB_short_test[ (UKB_short_test['Severity_evolution']== i) ]
    distress.insert(0, 'Group', 1)
    list_cohen[0].append(cohen_d(distress.Signature_residuals_cub, no_dist.Signature_residuals_cub))
    bs = stats.bootstrap((distress.Signature_residuals_cub, no_dist.Signature_residuals_cub), cohen_d, confidence_level=0.95,method='percentile',random_state=rng,n_resamples=10000,vectorized=False)
    list_cohen[1].append(bs.confidence_interval[0]), list_cohen[2].append(bs.confidence_interval[1])
    no_dist  = UKB_short_test[(UKB_short_test.Tinnitus_Severity==0) & (UKB_short_test.Tinnitus_Severity_FollowUp==0)]
    distress = UKB_short_test[(UKB_short_test['Severity_evolution']== i) ]

    y_test_roc = UKB_short_test['Severity_evolution']
    condition  = (y_test_roc== 0) | (y_test_roc==i)
    y_roc = y_test_roc[condition]
    y_roc = np.where(y_test_roc[condition]==i,1,0)
    sig = UKB_short_test.Signature_residuals_cub
    sign_roc = sig[condition]

    if i<0: 
        y_roc = abs(y_roc-1)
    fpr, tpr, _ = roc_curve(y_roc, sign_roc)
    roc_auc = auc(fpr, tpr)
    list_roc.append(roc_auc)

Cohen_D = pd.DataFrame(data = list_cohen, index = ['Estimate', 'Low', 'High'], columns = Change).T
AUC_ROC = pd.DataFrame(data = list_roc, columns = ['AUC-ROC'], index = Change).T


#### FIGURE  Effect Sizes across change ####
fig, axs = plt.subplots(ncols=1, nrows=2,  figsize=(5, 10), gridspec_kw={'width_ratios': [1], 'height_ratios': [0.85, 0.15]})
fs=20
color1, color2 = sns.color_palette("RdBu", 7)[-1], sns.color_palette("seismic", 7)[-1]
# Cohen's D
for i,c in zip([0,3], [color1, color2]): 
    i2, xlabels = i+4, np.arange(-3,4,1)
    axs[0].fill_between(xlabels[i:i2], Cohen_D.Low[i:i2], Cohen_D.High[i:i2], color = c, alpha = 0.5)
    axs[0].plot(xlabels[i:i2], Cohen_D['Estimate'][i:i2], color = c)
axs[0].axvline(x = 0, ls = '-', color = 'black', lw = 1), axs[0].axhline(y = 0, ls = '--', color = 'grey', lw = 1)
axs[0].set_ylim((-1.3, 1.5)), axs[0].set_yticks(np.arange(-1.3, 2, 0.4)), axs[0].set_yticklabels(np.arange(-1.3, 2, 0.4).round(1), fontsize = fs)
axs[0].set_xticks(np.arange(-3,4,1))
axs[0].set_ylabel("Cohen's d (Evolution vs No Distress)", fontsize = fs), axs[0].set_xlabel('Severity evolution', fontsize = fs)
axs[0].spines['bottom'].set_bounds((-3,3)), axs[0].spines['left'].set_bounds((-2,2))

# AUC-ROC
color_palette = sns.color_palette('RdBu', 8)[4:8][::-1]  +  sns.color_palette('seismic', 6)[3:6]
sns.barplot(ax = axs[1], data = AUC_ROC, palette = color_palette )
axs[1].set_ylim((0.5, 0.9)), axs[1].set_ylabel('AUC-ROC', fontsize = fs)
axs[1].set_yticks([0.50, 0.60, 0.70, 0.80]), axs[1].set_yticklabels([0.50, 0.60, 0.70, 0.80], fontsize = fs),
for i in range(2):
    axs[i].set_xticklabels(Change, fontsize = fs)
    axs[i].tick_params(left=True, bottom=True, length=3, width=1), axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].spines[['top', 'left']].set_linewidth(1)
