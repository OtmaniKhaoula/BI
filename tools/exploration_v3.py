# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:59:53 2024

@author: Utilisateur
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import f_oneway
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
plt.style.use('ggplot')

"""
Importation des tables
"""
"""
agents = pd.read_csv("../data/Agents.csv", sep=",")
criteria = pd.read_csv("../data/Criteria.csv", sep=",")
lotBuyers = pd.read_csv("../data/LotBuyers.csv", sep=",")
lots = pd.read_csv("../data/Lots.csv", sep=",")
lotSuppliers = pd.read_csv("../data/LotSuppliers.csv", sep=",")
name = pd.read_csv("../data/Names.csv", sep=",")
"""

# Stats, valeurs manquantes, etc sur les tables
# dataList: La liste des tables (dataFrame) qu'on veut explorer
# Liste des noms des tables (pour pouvoir retrouver quelles infos correspond à qu'elle table)
def statistiques(dataList, dataNames):
    
    describe = {}
    for names, data in zip(dataNames, dataList):
        
        describe[names] = {}
        # Types des variables
        describe[names]["types"] = data.dtypes
        # Taux de valeurs inconnues
        describe[names]["na"] = data.isna().sum()/data.shape[0]
        data_float64 = data.select_dtypes(include=['float64'])
        # Description pour objects de type object
        if data_float64.shape[1] > 0:
            describe[names]["describe"] = data_float64.describe()
        data_object = data.select_dtypes(include=['object', 'int64', 'bool'])
        # Description pour variables de type float64
        if data_object.shape[1] > 0:
            describe[names]["values"] = {}
            describe[names]["value_counts"] = {}
            for col in data_object.columns:
                describe[names]["values"][col] = data_object[col].unique()
                describe[names]["value_counts"][col] = data_object[col].value_counts()

    return describe

"""
dataList = [agents, criteria, lotBuyers, lots, lotSuppliers, name]
dataNames = ["agents", "criteria", "LotBuyers", "lots", "LotSuppliers", "name"]
describe = stats(dataList, dataNames)
"""

###############################################################################
# Graphiques univariés
###############################################################################

"""
Produisez un graphique montrant la distribution de la variable, et discutez-le. S’agit-il
d’une distribution standard, et si oui laquelle ?
https://jhub.cnam.fr/doc/notebooks/Representation_graphique_Python_Matplotlib.html
"""

# Graphiques pour les variables quali (de type object)
# Data: table qu'on veut explorer
# name: placer les graphiques dans le dossier BI/graphics/name
# columns: les colonnes des variables catégorielles
def graphics_qual(data, name, columns):
    
    for col in columns:
        
        print(col)
            
        # Calculer la fréquence des valeurs dans la colonne 'col'
        frequence_valeurs = data[col].value_counts()
        if(frequence_valeurs.shape[0]>20):
               
            # Récupération des valeurs de la série
            values = pd.Series(frequence_valeurs.values)

            title = f'Effectifs de la fréquence des modalités pour la variable {col}'
            #print(values)
                
            plt.figure(figsize=(10, 10))
            plt.hist(values, bins = 10, label="histogramme")
            plt.title(title)
            plt.xlabel('Variables')
            plt.ylabel('Effectifs')
            #plt.ylim(0, max(frequence_valeurs.value_counts())*1.2)
            plt.yscale('log')
            plt.savefig(f'../graphics/{name}/histogramme_{col}.png')
            plt.close()                
                
        else:
            # Sélectionner les 20 variables les plus fréquentes
            title = f'Barplot des {col}'
              
            x_pos = np.arange(frequence_valeurs.shape[0])
                
            plt.figure(figsize=(10, 10))
            plt.bar(x_pos, frequence_valeurs)
            plt.title(title, fontsize=15)
            # Étiqueter les barres avec les modalités
            plt.xticks(x_pos, frequence_valeurs.index, rotation=90, fontsize=9)
            plt.xlabel('Variables')
            plt.ylabel('Effectifs')
            plt.savefig(f'../graphics/{name}/barplot_{col}.png')
            plt.close()

"""
graphics_qual(lots, "lots", ['tedCanId','cpv', 'numberTenders', 'onBehalf',
'jointProcurement', 'fraAgreement', 'fraEstimated','accelerated', 'outOfDirectives','subContracted', 'gpa', 'multipleCae', 'typeOfContract', 'topType',
'renewal'])
graphics_qual(agents, "agents", ['name', 'siret', 'address', 'city', 'zipcode', 'country','department'])
graphics_qual(name, "name", ['name'])
graphics_qual(criteria, "criteria", ['lotId', 'name','type'])
graphics_qual(lotBuyers, "LotBuyers", ['lotId', 'agentId'])
graphics_qual(lotSuppliers, "LotSuppliers", ['lotId', 'agentId'])
"""

# Graphiques pour les variables quanti (de type float64)
# Data: table qu'on veut explorer
# name: placer les graphiques dans le dossier BI/graphics/name
# columns: les colonnes des variables numériques
def graphics_quan(data, name, columns):
    
    for col in columns:
        
        print("col = ", col)
                     
        sns.stripplot(x = col, data = data, jitter = True)
        plt.title(f'stripplot des {col}')
        plt.savefig(f'../graphics/{name}/stripplot_{col}.png')
        plt.close()
            
        plt.figure(figsize=(9, 9))
        plt.hist(data[col].dropna())
        plt.title(f'Histogramme des {col}')
        plt.xlabel('Variables')
        plt.ylabel('Effectifs')
        plt.savefig(f'../graphics/{name}/histogramme_{col}.png')
        plt.close()

"""
graphics_quan(lots, "lots", ['correctionsNb', 'awardEstimatedPrice', 'awardPrice','numberTenders','numberTendersSme','contractDuration', 'publicityDuration'])
graphics_quan(agents, "agents",['longitude', 'latitude'])
graphics_quan(criteria, "criteria", ['weight'])
"""

###############################################################################
# Graphiques bivariés
###############################################################################

# Barplot: graphe entre deux variables quali avec les effectifs de la fréquence des modalités dans l'axe des abscisses
# Data: table qu'on veut explorer
# name: placer les graphiques dans le dossier BI/graphics/name
# columns: les colonnes des variables catégorielles
def barplot(data, name, columns):
    
      data_copy = data[columns]
    
      for i in range(len(columns)-1):
            
            col1 = data_copy.columns[i]
            
            for j in range(i+1,len(columns)):
                
                col2 = data_copy.columns[j]
                
                # Conversion des données en séries pandas
                serie1 = data_copy[col1]
                serie2 = data_copy[col2]

                # Comptage des occurrences de chaque catégorie dans chaque série
                counts1 = serie1.value_counts()
                counts2 = serie2.value_counts()
                
                """
                Je pense que faire un graphique si on a trop de modalités sur les deux variables catégorielles n'est pas très pertinent
                """
                if(data_copy[col1].nunique() > 20 and data_copy[col2].nunique() > 20):
                    
                    # Récupération des valeurs de la série
                    df = pd.DataFrame({col1: serie1.map(counts1.to_dict()), col2: serie2.map(counts2.to_dict())})
                    df = df.dropna()
                                        
                    plt.figure()
                    sns.scatterplot(data=df, x=col1, y=col2, palette='viridis')

                    # Ajout des titres et des étiquettes
                    #plt.title(f'{col2} en fonction de la {col1}', fontsize=25)
                    plt.xlabel(f'{col1}', fontsize=18)
                    plt.ylabel(f'Log({col2})', fontsize=18)
                    plt.yscale('log')
                    plt.savefig(f'../graphics/{name}/scatter_{col1}-{col2}.png')
                    plt.close()
                
                elif(data_copy[col1].nunique() > 20):
                    
                    # Récupération des valeurs de la série
                    dict_values = counts1.to_dict()
                    
                    df = pd.DataFrame({col1: serie1.map(dict_values), col2: serie2})
                    df = df.dropna()
                    
                    plt.figure(figsize=(18, 18))
                    sns.violinplot(x=col2, y=col1, data=df)
                    # Ajouter des titres et des étiquettes
                    plt.xlabel(f'{col2}', fontsize=20)
                    plt.ylabel(f'Log({col1})', fontsize=20)
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=16)
                    plt.yscale('log')
                    plt.xticks(rotation=45,fontsize=18)
                    plt.savefig(f'../graphics/{name}/boxplot_{col1}-{col2}.png')
                    plt.close()
                    
                elif(data_copy[col2].nunique() > 20):
                    
                    # Récupération des valeurs de la série
                    dict_values = counts2.to_dict()
                    df = pd.DataFrame({col1: serie1, col2: serie2.map(dict_values)})
                    df = df.dropna()
                    
                    plt.figure(figsize=(18, 18))
                    sns.violinplot(x=col1, y=col2, data=df)
                    plt.xlabel(f'{col2}', fontsize=20)
                    plt.ylabel(f'Log({col1})', fontsize=20)
                    plt.yscale('log')
                    plt.xticks(rotation=45, fontsize=18)
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=16)
                    plt.savefig(f'../graphics/{name}/boxplot_{col1}-{col2}.png')
                    plt.close()

                else:
                    # Créer un tableau croisé des deux variables
                    cross_tab = pd.crosstab(data_copy[col1], data_copy[col2])

                    # Tracer le barplot à partir du tableau croisé
                    cross_tab.plot(kind='bar', stacked=True, figsize=(18, 18))

                    # Ajouter de titre et d'étiquettes
                    plt.xlabel(col1, fontsize=20)
                    plt.ylabel('Effectifs', fontsize=20)
                    plt.xticks(rotation=45, fontsize=18)  # Rotation des étiquettes sur l'axe x pour une meilleure lisibilité
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=16)
                    plt.legend(title=col2)

                    # Sauvegarde du plot
                    plt.savefig(f'../graphics/{name}/barplot_{col1}_{col2}.png')
                    plt.close()

#barplot(lots, "lots", ['onBehalf','jointProcurement', 'outOfDirectives','topType','typeOfContract','renewal'])

# Scatter plot entre une variable quali et une variable quanti
# Création du scatter plot avec Seaborn
# Data: table qu'on veut explorer
# name: placer les graphiques dans le dossier BI/graphics/name
# columns: les colonnes des variables numériques
def scatter(data, name, columns):
    
    data_copy = data[columns] 
          
    for i in range(len(columns)-1):
            
        col1 = data_copy.columns[i]
            
        for j in range(i+1,len(columns)):
                
            col2 = data_copy.columns[j]
        
            plt.figure(figsize=(9, 9))
            plt.scatter(data_copy[col1], data_copy[col2])

            # Ajout des titres et des étiquettes
            plt.xlabel(f'{col1}', fontsize = 18)
            plt.ylabel(f'{col2}', fontsize = 18)
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)
            plt.savefig(f'../graphics/{name}/scatter_{col1}-{col2}.png')

#scatter(lots, "lots", ["awardPrice", "awardEstimatedPrice", "numberTendersSme", "numberTenders"])

# Bloxplot: une variable quali et une variable quanti
# Data: table qu'on veut explorer
# name: placer les graphiques dans le dossier BI/graphics/name
# columns_cat: les colonnes des variables catégorielles
# columns_num: les colonnes des variables numériques
def boxplots(data, name, columns_num, columns_cat):
           
    for col1 in columns_cat:
            
        print("col1 = ", col1)
            
        for col2 in columns_num:
                
            print("col2 = ", col2)
                
            # Conversion des données en séries pandas
            serie1 = data[col1]
            serie2 = data[col2]

            # Comptage des occurrences de chaque catégorie dans chaque série
            counts1 = serie1.value_counts()
                
            if(data[col1].nunique() > 30):
                    
                # Récupération des valeurs de la série
                df = pd.DataFrame({col1: serie1.map(counts1.to_dict()), col2: serie2})
                df = df.dropna()
                                        
                plt.figure()
                sns.scatterplot(data=df, x=col1, y=col2, palette='viridis')

                # Ajout des titres et des étiquettes
                plt.xlabel(f'{col1}', fontsize=20)
                plt.ylabel(f'{col2}', fontsize=20)
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)
                plt.savefig(f'../graphics/{name}/scatter_{col1}-{col2}.png')
                plt.close()
                    
                continue
                              
            # Filtrer le DataFrame original en fonction des valeurs les plus fréquentes de 'col1'
            #data_filtre = data[data[col1].isin(counts1.index)]
                
            plt.figure(figsize=(10, 10))
            sns.violinplot(x=col1, y=col2, data=data)

            plt.xlabel('Catégorie', fontsize = 18)
            plt.ylabel('Log(Valeur)', fontsize = 18)
                
            plt.xticks(rotation=90, fontsize=16)
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)
            plt.yscale('log')
            plt.savefig(f'../graphics/{name}/boxplot_{col1}-{col2}.png')
            plt.close()

"""
boxplots(lots, "lots", ["correctionsNb", "awardPrice"], ["cancelled", "typeOfContract"])
boxplots(criteria, "criteria", ["weight"], ["type"])
"""

##############################################################################
# Graphique avec les dates
##############################################################################

# Graphique avec le nombre de lots par année
def date_number_lots(data, colDate):
    
    # S'assurer d'avoir le bon format
    data[colDate] = data[colDate].astype(str)

    awardDate_year = []

    for i in range(data.shape[0]):
        
        if isinstance(data.loc[i,colDate], str):
            
            awardDate_year.append(data.loc[i,colDate][0:4])
    
    data["awardDate_year"] =  awardDate_year
    #Agréger les données par année pour obtenir le prix moyen, minimum, médian et maximum
    number_lots = data.groupby(colDate)["lotId"].agg(['count'])

    plt.figure(figsize=(15, 8))
    plt.plot(number_lots, marker='+', linestyle='-')
    plt.xlabel('Année', fontsize=18)
    plt.ylabel('Nombre de lots', fontsize=18)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.xticks(rotation=90)
    plt.savefig(f'../graphics/lots/graphic_lotId-{colDate}.png')
    plt.close()
    
#date_number_lots(lots, 'awardDate_year')    

# Stat sur une valeur numérique au fil des années 
def date_stat(data, col, colDate):
    
    # S'assurer d'avoir le bon format
    data[colDate] = data[colDate].astype(str)

    awardDate_year = []

    for i in range(data.shape[0]):
        
        if isinstance(data.loc[i,colDate], str):
            
            awardDate_year.append(data.loc[i,colDate][0:4])
    
    data["awardDate_year"] =  awardDate_year
    
    #Agréger les données par année pour obtenir le prix moyen, minimum, médian et maximum
    prix_stats = data.groupby(colDate)[col].agg(['mean', 'median'])

    # Créer le graphique
    plt.figure(figsize=(10, 6))

    # Tracer les courbes pour le prix moyen, minimum, médian et maximum
    prix_stats['mean'].plot(label='Prix moyen', marker='+', linestyle='-')
    prix_stats['median'].plot(label='Médiane des prix', marker='+', linestyle='-')

    plt.xlabel('Année', fontsize=18)
    plt.ylabel('Prix', fontsize=18)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend()

    plt.savefig(f'../graphics/lots/graphic_{col}-{colDate}.png')
    plt.close()

#date_stat(lots, "awardPrice", "awardDate_year")
 
##############################################################################
# Test de khi-deux
##############################################################################

"""
α = 0,05 lors du test d'indépendance
Dans ce cas, vous avez décidé de prendre un risque de 5 % de conclure que les
deux variables sont indépendantes alors qu'en réalité elles ne le sont pas
"""
# Test de khi-2 sur les données catégorielles
# col1 et col2 sont les deux colonnes dont nous voulons évaluer les associations
def khi_2(data, name, col1, col2):
            
    contingency_table = data.groupby([col1, col2]).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    
    # Calcul du coefficient de V de Cramer
    n = contingency_table.sum().sum()
    k = contingency_table.shape[0]
    r = contingency_table.shape[1]
    v_cramer = np.sqrt(chi2 / (n * min(k-1, r-1)))
                    
    # Afficher le résultat
    print(f"Les résultats de la comparaison des variables '{col1}' et '{col2}' de la table {name} sont: \n")
    print("Valeur de chi2 :", chi2)
    print("p-valeur :", p)
    print("Corrélation de V_Cramer :", v_cramer)

# Corrélation entre deux variables quanti (float64)
# col1 et col2 sont les deux colonnes dont nous voulons évaluer les associations
# name : nom de la table qu'on veut citer
def pearson(data, name, col1, col2):
    
    copy_data = data.copy()
    
    copy_data.dropna(subset=[col1, col2], inplace=True)
    
    coeff_pearson, p_value = pearsonr(copy_data[col1], copy_data[col2])
        
    # Afficher le résultat
    print(f"Les résultats du lien linéaire entre les variables '{col1}' et '{col2}' de la table {name} sont: \n")
    print("p-valeur :", p_value)
    print("Corrélation de Pearson :", coeff_pearson)

"""
pearson(lots, "Lots", "awardPrice","awardEstimatedPrice")
pearson(lots, "Lots", "numberTendersSme", "numberTenders")
"""     
   
# Corrélation entre une variable quali et une quanti
# https://datascientest.com/correlation-entre-variables-comment-mesurer-la-dependance

# Vérification des conditions pour ANOVA
def verif(data, col_cat, col_num):
    
    # Sélection des données
    data_anova = data[[col_cat, col_num]].dropna()

    # Vérification de l'égalité des variances
    groupes = data_anova[col_cat].unique()
    variances = []
    for groupe in groupes:
        variances.append(data_anova[data_anova[col_cat] == groupe][col_num].var())

    p_value_levene = stats.levene(*[data_anova[data_anova[col_cat] == groupe][col_num] for groupe in groupes])[1]

    # Vérification de la normalité des résidus
    p_value_shapiro = stats.shapiro(data_anova[col_num])[1]

    if p_value_levene > 0.05 and p_value_shapiro > 0.05:
        
        return True
    
    else:
        
        return False       

# ANOVA ou Krustal Wallis selon les conditions 
def anova(data, name, col_num, col_cat):
    
    dfx = data[[col_cat, col_num]]
       
    booleen = verif(dfx, col_cat, col_num)
    
    if(booleen):
        
        result = f_oneway(*dfx.values.T)
        F, p = result.statistic, result.pvalue
        
        # Afficher le résultat
        print(f"Les résultats de l'ANOVA entre les variables '{col_num}' et '{col_cat}' de la table {name} sont: \n")
        print("F :", F)
        print("p-valeur :", p)
    
    else:
        
        F, p = stats.kruskal(*[group[col_num].values for name, group in dfx.groupby(col_cat)])  

        # Afficher le résultat
        print(f"Les résultats de Krustal-Wallis entre les variables '{col_num}' et '{col_cat}' de la table {name} sont: \n")
        print("F :", F)
        print("p-valeur :", p)    

"""
columns_cat = ["awardDate_year"]
dict_stats = anova(lots, "lots", "awardPrice", "awardDate_year")
"""
"""
anova(criteria, "criteria", "weight", "type")
"""






          


