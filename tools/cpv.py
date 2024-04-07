# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:32:45 2024

@author: Utilisateur
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import scipy.stats as stats
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')

"""
Importation des tables
"""
data = pd.read_csv("../data/data_with_lots_agents_distance.csv", sep=";")

# Supprimer les lignes où le cpv est inconnu

"""
Notre label est le CPV (agents économiques)
Code CPV:  
    Les deux premiers chiffres identifient les divisions (XX000000-Y);
    Les trois premiers chiffres identifient les groupes (XXX00000-Y);
    Les quatre premiers chiffres identifient les classes (XXXX0000-Y);
    Les cinq premiers chiffres identifient les catégories (XXXXX000-Y);
"""

"""
Faire des stats sur les CPV
"""
# Utiliser ces nouvelles colonnes pour faire des stats par regroupement
def spv_desc(data, liste, columns):
    
    dict_stat_cpv = {}
    
    for element in liste:
        
        dict_stat_cpv[element] = {}
        
        for col in columns:
            
            data_v2 = data.copy()
            data_v2 = data_v2.drop_duplicates(subset=[col])
            groupes = data.groupby(element).agg({col: ['mean', 'median', 'min', 'max', 'count']})
            dict_stat_cpv[element][col] = groupes
    
    return dict_stat_cpv

# 'numberTendersSme','awardEstimatedPrice', 'lotsNumber', 
columns = ['awardEstimatedPrice', 'numberTendersSme', 'awardPrice','numberTenders','contractDuration','publicityDuration','distance','DELAY','ENVIRONMENTAL','OTHER','PRICE','SOCIAL','TECHNICAL']
liste = ["divisions", "groupes", "classes", "categories"]

dict_stat_cpv = spv_desc(data, liste, columns) 

"""
Graphique indicateur en fonction du cpv
"""
def graphic(data, columns):

    for col in columns:
        
        print("col = ", col)
        
        # Agréger les données en calculant la moyenne des prix pour chaque code CPV
        data_v2 = data.groupby('divisions')[col].mean().reset_index()

        # Tracer le barplot des prix en fonction des codes CPV
        plt.figure(figsize=(20, 12))
        sns.barplot(x='divisions', y=col, data=data_v2)
        plt.xlabel('Divisions', fontsize=20)
        plt.ylabel(f'mean({col})', fontsize=20)
        plt.xticks(rotation=90)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.savefig(f'../graphics/cpv/divisions-{col}.png')

graphic(data, columns)

"""
Vérification des conditions pour ANOVA
"""
def verif(data, col):
    
    # Sélection des données
    data_anova = data[['divisions', col]].dropna()

    # Vérification de l'égalité des variances
    groupes = data_anova['divisions'].unique()
    variances = []
    for groupe in groupes:
        variances.append(data_anova[data_anova['divisions'] == groupe][col].var())

    p_value_levene = stats.levene(*[data_anova[data_anova['divisions'] == groupe][col] for groupe in groupes])[1]

    # Vérification de la normalité des résidus
    p_value_shapiro = stats.shapiro(data_anova[col])[1]

    if p_value_levene > 0.05 and p_value_shapiro > 0.05:
        
        return True
    
    else:
        
        return False       

"""
Test ANOVA entre cpv et la distance
""" 
def anova(data, columns):
    
    dict_stats = {}
    
    for col in columns:
    
        dfx = data[['divisions', col]]
    
        # Remplacer les valeurs manquantes par la moyenne de la colonne
        dfx[col].fillna(dfx[col].median(), inplace=True)
    
        scaler = StandardScaler()
        norm = pd.DataFrame(scaler.fit_transform(dfx[[col]]), columns=[col])  # Correction ici
        dfx[col] = norm
    
        booleen = verif(dfx, col)
    
        if(booleen):
        
            result = f_oneway(*dfx.values.T)
            F, p = result.statistic, result.pvalue
    
        else:
        
            F, p = stats.kruskal(*[group[col].values for name, group in dfx.groupby('divisions')])  

        #corr, ps = stats.spearmanr(data['divisions'], data[col])

        dict_stats[col] = [F, p]
    
    return dict_stats

dict_stats = anova(data, columns)

"""
Fdd, Pdd = anova(data_with_cpv[["distance","divisions"]], "distance")
Fad, Pad = anova(data_with_cpv[["awardPrice","divisions"]], "awardPrice")
"""

###############################################################################

###############################################################################

# Corrélation entre deux variables quanti (float64)
def pearson(data):
    
    dict_pearson = {}
      
    for i in range(data.shape[1]-1):
            
        for j in range(i+1, data.shape[1]):
              
            copy_data = data.copy()
            col1 = copy_data.columns[i]
            col2 = copy_data.columns[j]
               
            copy_data.dropna(subset=[col1, col2], inplace=True)
            coeff_pearson,_ = pearsonr(copy_data[col1], copy_data[col2])
            dict_pearson[f"({col1}, {col2})"] = coeff_pearson
            if(coeff_pearson>0.5 or coeff_pearson<-0.5):
                print("col 1 = ", col1, " col 2 = ", col2)
                print("coefficient de Pearson = {}".format(coeff_pearson))
    
    return dict_pearson
"""
coef_pearson = pearson(data[['awardPrice','numberTenders','contractDuration','publicityDuration',
       'distance','DELAY','ENVIRONMENTAL','OTHER','PRICE','SOCIAL','TECHNICAL']])
"""
# Effectuer un clustering entre les diff stat afin de voir des potentiels regroupement entre les cpv (divisions/groupes...)