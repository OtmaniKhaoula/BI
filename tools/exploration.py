# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:26:06 2024

@author: Amarat Rim & Otmani Khaoula & Elisa Martin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import f_oneway
import scipy.stats as stats
plt.style.use('ggplot')

"""
Importation des tables
"""
agents = pd.read_csv("../data/Agents_v2.csv", sep=";")
criteria = pd.read_csv("../data/Criteria.csv", sep=",")
lotBuyers = pd.read_csv("../data/LotBuyers.csv", sep=",")
lots = pd.read_csv("../data/Lots.csv", sep=",")
lotSuppliers = pd.read_csv("../data/LotSuppliers.csv", sep=",")
name = pd.read_csv("../data/Names.csv", sep=",")

"""
Identifiez la nature de la variable, ainsi que son codage. Discutez-les, notamment si
vous jugez que le codage n’est pas approprié.

Calculez les statistiques standard, en fonction de la nature de la variable : moyenne,
écart-type, quantiles, mode, min, max, etc. Discutez-les.
"""
dataList_float64 = []
dataList_object = []
dataList = [agents, criteria, lotBuyers, lots, lotSuppliers, name]
dataNames = ["agents", "criteria", "lotBuyers", "lots", "lotSuppliers", "name"]

describe = {}
for names, data in zip(dataNames, dataList):
    
    if(names == "agents"):      
        #data['siret'] = data['siret'].fillna(0)
        data['siret'] = data['siret'].astype('object')
        #data['zipcode'] = data['zipcode'].fillna(0)
        data['zipcode'] = data['zipcode'].astype('object')
     
    elif(names == "lots"):
        #data['cpv'] = data['cpv'].fillna(np.nan)
        data['cpv'] = data['cpv'].astype('object')
        data['divisions'] = data['divisions'].astype('object')
        data['groupes'] = data['groupes'].astype('object')
        data['classes'] = data['classes'].astype('object')
        data['categories'] = data['categories'].astype('object')
        data['correctionsNb'] = data['correctionsNb'].astype('float64')
        #print("bbb = ", data.shape)
        
    if('lotId' in data.columns):
        data['lotId'] = data['lotId'].astype('object')
    if('agentId' in data.columns):
        data['agentId'] = data['agentId'].astype('object')
    if('tedCanId' in data.columns):
        data['tedCanId'] = data['tedCanId'].astype('object')  
    if('criterionId' in data.columns):
        data['criterionId'] = data['criterionId'].astype('object')
     
    describe[names] = {}
    # Types des variables
    describe[names]["types"] = data.dtypes
    # Taux de valeurs inconnues
    describe[names]["na"] = data.isna().sum()/data.shape[0]
    data_float64 = data.select_dtypes(include=['float64'])
    # Description pour objects de type object
    if data_float64.shape[1] > 0:
        describe[names]["describe"] = data_float64.describe()
        dataList_float64.append(data_float64)
    data_object = data.select_dtypes(include=['object', 'int64', 'bool'])
    # Description pour variables de type float64
    if data_object.shape[1] > 0:
        if data_object.select_dtypes(include=['object', 'bool']).shape[1] > 0:
            dataList_object.append(data_object.select_dtypes(include=['object', 'bool']))
        describe[names]["values"] = {}
        describe[names]["value_counts"] = {}
        for col in data_object.columns:
            describe[names]["values"][col] = data_object[col].unique()
            describe[names]["value_counts"][col] = data_object[col].value_counts()
            
"""
Produisez un graphique montrant la distribution de la variable, et discutez-le. S’agit-il
d’une distribution standard, et si oui laquelle ?
https://jhub.cnam.fr/doc/notebooks/Representation_graphique_Python_Matplotlib.html
"""
# Graphiques pour les variables quali (de type object)
def graphics_qual(dataList_object, dataNames):
    
    for name, data in zip(dataNames, dataList_object):

        print("name = ", name)
        for col in data.columns:
            
            # Calculer la fréquence des valeurs dans la colonne 'col'
            frequence_valeurs = data[col].value_counts()
            if(frequence_valeurs.shape[0]>45):
               
                # Suppression des valeurs inconnues
                frequence_valeurs = data[col].dropna().value_counts()
                
                # Récupération des valeurs de la série
                values = pd.Series(frequence_valeurs.values)
                
                plt.figure(figsize=(9, 9))
                plt.hist(values, bins = 10, label="histogramme", edgecolor='black')         
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)
                plt.xlabel("Fréquence d'apparition", fontsize=18)
                plt.ylabel('Log(Effectifs)', fontsize=18)
                plt.yscale('log')
                plt.savefig(f'../graphics/{name}/histogramme_{col}.png')
                plt.close()                
               
            # Si le nombre de modalités n'est pas trop élevé
            else:
                
                x_pos = np.arange(frequence_valeurs.shape[0])
                
                plt.figure(figsize=(9, 9))
                plt.bar(x_pos, frequence_valeurs)
                plt.xticks(x_pos, frequence_valeurs.index, rotation=90, fontsize=16)
                plt.xlabel('Variables', fontsize=20)
                plt.ylabel('Effectifs', fontsize=20)
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)
                plt.savefig(f'../graphics/{name}/barplot_{col}.png')
                plt.close()
            
dataNames = ["agents", "criteria", "lotBuyers", "lots", "lotSuppliers", "name"]
graphics_qual(dataList_object, dataNames)

# Graphiques pour les variables quanti (de type float64)
def graphics_quan(dataList_float, dataNames):
    
    for name, data in zip(dataNames, dataList_float):

        for col in data.columns:
                        
            print("col = ", col)
            
            # Scatter plot - qqplot (voir la distribution par rapport à une distribution considérée comme normale)
            serie = data[col]

            # Suppression des valeurs NaN
            serie_sans_nan = serie.dropna()
            print(len(serie_sans_nan))
            
            sns.stripplot(x = col, data = data, jitter = True)
            plt.savefig(f'../graphics/{name}/stripplot_{col}.png')
            plt.close()
            
            plt.figure(figsize=(7, 7))
            plt.hist(data[col].dropna(), edgecolor='black')
            plt.xlabel(f"{col}", fontsize=18)
            plt.ylabel('Log(Effectifs)', fontsize=18)
            plt.yscale('log')
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)
            plt.savefig(f'../graphics/{name}/histogramme_{col}.png')
            plt.close()
            
dataNames = ['agents', "criteria", 'lots']
graphics_quan(dataList_float64, dataNames)

"""
# Analyse descriptive bi-variée   
"""

# Barplot: graphe entre deux variables quali avec les effectifs de la fréquence des modalités dans l'axe des abscisses
def barplot(dataList_object, dataNames):
    
    for data,names in zip(dataList_object, dataNames):
        
        if(len(list(data.columns))<2):
            continue
        
        for i in range(data.shape[1]-1):
            
            col1 = data.columns[i]
            
            #print("col1 = ", col1)
            
            for j in range(i+1,data.shape[1]):
                
                col2 = data.columns[j]
                
                #print("col2 = ", col2)
                
                # Conversion des données en séries pandas
                serie1 = data[col1]
                serie2 = data[col2]

                # Comptage des occurrences de chaque catégorie dans chaque série
                counts1 = serie1.value_counts()
                counts2 = serie2.value_counts()
                
                """
                Je pense que faire un graphique si on a trop de modalités sur les deux variables catégorielles n'est pas très pertinent
                """
                if(data[col1].nunique() > 20 and data[col2].nunique() > 20):
                    
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
                    plt.savefig(f'../graphics/{names}/scatter_{col1}-{col2}.png')
                    plt.close()
                
                elif(data[col1].nunique() > 20):
                    
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
                    plt.savefig(f'../graphics/{names}/boxplot_{col1}-{col2}.png')
                    plt.close()
                    
                elif(data[col2].nunique() > 20):
                    
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
                    plt.savefig(f'../graphics/{names}/boxplot_{col1}-{col2}.png')
                    plt.close()

                else:
                    # Créer un tableau croisé des deux variables
                    cross_tab = pd.crosstab(data[col1], data[col2])

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
                    plt.savefig(f'../graphics/{names}/barplot_{col1}_{col2}.png')
                    plt.close()
       
columns_object = ["criteria", 'lots']
barplot(dataList_object, columns_object)                              

# Scatter plot entre une variable quali et une variable quanti
# Création du scatter plot avec Seaborn
def scatter(dataList, dataNames):
    
    for data,names in zip(dataList, dataNames):   
        
        data_copy = data.copy()   
        #print(data.columns)
        
        if(len(list(data.columns))<2):
            continue
        
        for i in range(data_copy.shape[1]-1):
            
            col1 = data.columns[i]
            
            for j in range(i+1,data_copy.shape[1]):
                
                col2 = data_copy.columns[j]
        
                plt.figure(figsize=(9, 9))
                plt.scatter(data[col1], data[col2])

                # Ajout des titres et des étiquettes
                plt.xlabel(f'{col1}', fontsize = 18)
                plt.ylabel(f'{col2}', fontsize = 18)
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)
                plt.savefig(f'../graphics/{names}/scatter_{col1}-{col2}.png')

dataNames = ['lots']
dataList = [lots[["awardPrice", "awardEstimatedPrice", "numberTendersSme", "numberTenders"]]]

scatter(dataList, dataNames)

###############################################################################

# Bloxplot: une variable quali et une variable quanti
def boxplots(dataList, dataNames):
    
    for data, names in zip(dataList, dataNames):
        
        print("data = ", names)
        
        # Sélectionner les colonnes de type 'object'
        colonnes_object = data.select_dtypes(include=['object', 'bool']).columns.tolist()
        # Sélectionner les colonnes de type 'float64'
        colonnes_float64 = data.select_dtypes(include=['float64']).columns.tolist()
        
        if len(colonnes_object) < 1 or len(colonnes_float64) < 1:
            continue
        
        for col1 in colonnes_object:
            
            print("col1 = ", col1)
            
            for col2 in colonnes_float64:
                
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
                    plt.savefig(f'../graphics/{names}/scatter_{col1}-{col2}.png')
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
                plt.savefig(f'../graphics/{names}/boxplot_{col1}-{col2}.png')
                plt.close()
                
dataList = [lots[["correctionsNb", "cancelled", "typeOfContract", "awardPrice"]]]
dataNames = ["lots"]
boxplots(dataList, dataNames)

dataList = [criteria[["weight", "type"]]]
dataNames = ["criteria"]
boxplots(dataList, dataNames)

##############################################################################
# Graphique avec les dates
##############################################################################

def date_number_lots(data, colDate):
    
    data = data[data[colDate] != 'nan']
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

#lots['awardDate_year'] = [lots.loc[i,'awardDate_year'][0:4] for i in range(lots.shape[0])]
lots['awardDate_year'] = lots['awardDate_year'].astype(str)

awardDate_year = []

for i in range(lots.shape[0]):
    
    if isinstance(lots.loc[i,'awardDate_year'], str):
        
        awardDate_year.append(lots.loc[i,'awardDate_year'][0:4])
    
    else:
        
        awardDate_year.append(np.nan)
        
lots['awardDate_year'] = awardDate_year

#date_number_lots(lots, 'awardDate_year')    

# Stat sur une valeur numérique au fil des années 
def date_stat(data, col, colDate):
    
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

date_stat(lots, "awardPrice", "awardDate_year")
                
###############################################################################
"""
α = 0,05 lors du test d'indépendance
Dans ce cas, vous avez décidé de prendre un risque de 5 % de conclure que les
deux variables sont indépendantes alors qu'en réalité elles ne le sont pas
"""
# Test de khi-2 sur les données catégorielles
def khi_2(dataListObject, dataNames):
    
    dict_khi_2 = {}
    
    for data, names in zip(dataListObject, dataNames):
    
        dict_khi_2[names] = {}
    
        for i in range(data.shape[1]-1):
            for j in range(i+1, data.shape[1]):
                col1 = data.columns[i]
                col2 = data.columns[j]
            
                try:
                    contingency_table = data.groupby([col1, col2]).size().unstack(fill_value=0)
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    
                    # Calcul du coefficient de V de Cramer
                    n = contingency_table.sum().sum()
                    k = contingency_table.shape[0]
                    r = contingency_table.shape[1]
                    v_cramer = np.sqrt(chi2 / (n * min(k-1, r-1)))
                    
                    dict_khi_2[names][f"({col1}, {col2})"] = [chi2, p, v_cramer, contingency_table]
                
                    # Afficher le résultat
                    print("Valeur de chi2 :", chi2)
                    print("p-valeur :", p)
                
                except ValueError as ve:
                    print(f"Erreur de valeur : {ve}. Colonne '{col1}' ou '{col2}' peut contenir des valeurs non valides.")
                except Exception as e:
                    print(f"Une erreur s'est produite : {e}")
                
    return dict_khi_2

dataNames = ['agents', "criteria", 'lots', 'name']
#♣dict_khi_2 = khi_2(dataList_object, dataNames)

# Si V-Cramer = nan: ça vient d'une colonne qui a une seule modalité
###############################################################################

# Corrélation entre deux variables quanti (float64)
def pearson(dataList_float64, dataNames):
    
    dict_pearson = {}
    
    for data,names in zip(dataList_float64, dataNames):
        
        dict_pearson[names] = {}
        
        for i in range(data.shape[1]-1):
            
            for j in range(i+1, data.shape[1]):
                
                copy_data = data.copy()
                col1 = copy_data.columns[i]
                col2 = copy_data.columns[j]
                                                            
                print("col1 = ", col1)
                print("col2 = ", col2)
                
                copy_data.dropna(subset=[col1, col2], inplace=True)
                coeff_pearson, p_value = pearsonr(copy_data[col1], copy_data[col2])
                dict_pearson[names][f"({col1}, {col2})"] = [coeff_pearson, p_value]
                print("coefficient de Pearson = {}".format(coeff_pearson))
    
    return dict_pearson

dataNames = ['lots']
dataList = [lots[["awardPrice", "awardEstimatedPrice", "numberTendersSme", "numberTenders"]]]
dict_pearson = pearson(dataList, dataNames)

###############################################################################

# Corrélation entre une variable quali et une quanti
# ex: country et la latitude
# https://datascientest.com/correlation-entre-variables-comment-mesurer-la-dependance
# Peut petre utilisé une fois normalisé? 
# A revoir avec kruskal

"""
Vérification des conditions pour ANOVA
"""
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

"""
Test ANOVA
"""
def anova(data, columns_num, columns_cat):
    
    dict_stats = {}
    
    for col_num in columns_num:
        print(col_num)
    
        for col_cat in columns_cat:
            print(col_cat)
            
            dfx = data[[col_cat, col_num]]
       
            booleen = verif(dfx, col_cat, col_num)
    
            if(booleen):
        
                result = f_oneway(*dfx.values.T)
                F, p = result.statistic, result.pvalue
    
            else:
        
                F, p = stats.kruskal(*[group[col_num].values for name, group in dfx.groupby(col_cat)])  

            dict_stats[(col_num, col_cat)] = [F, p]
    
    return dict_stats

columns_cat = ["awardDate_year"]
columns_num = ["awardPrice"]
dict_stats = anova(lots, columns_num, columns_cat)

columns_cat = ["type"]
columns_num = ["weight"]
dict_stats = anova(criteria, columns_num, columns_cat)


