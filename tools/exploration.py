# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:26:06 2024

@author: Elisa Martin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import f_oneway
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest

"""
Importation des tables
"""
agents = pd.read_csv("../data/Agents.csv", sep=",")
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
dataList = [agents, criteria, lotBuyers, lots, lotSuppliers, name]
dataList_float64 = []
dataList_object = []
dataNames = ["agents", "criteria", "lotBuyers", "lots", "lotSuppliers", "name"]

describe = {}
for name, data in zip(dataNames, dataList):
    describe[name] = {}
    describe[name]["types"] = data.dtypes
    describe[name]["na"] = data.isna().sum()/data.shape[0]
    data_float64 = data.select_dtypes(include=['float64'])
    if data_float64.shape[1] > 0:
        describe[name]["describe"] = data_float64.describe()
        dataList_float64.append(data_float64)
    data_object = data.select_dtypes(include=['object', 'int64'])
    if data_object.shape[1] > 0:
        if data_object.select_dtypes(include=['object']).shape[1] > 0:
            dataList_object.append(data_object.select_dtypes(include=['object']))
        describe[name]["values"] = {}
        describe[name]["value_counts"] = {}
        for col in data_object.columns:
            describe[name]["values"][col] = data_object[col].unique()
            describe[name]["value_counts"][col] = data_object[col].value_counts()
             
# Agents: Le siret et le zipcode sont de type float64 alors que ce n'est pas nécessaire, nous aurions plutôt eu tendance à les mettre de type int64       
# Lots: idem pour cpv (plutôt à mettre en mode object)

"""
Produisez un graphique montrant la distribution de la variable, et discutez-le. S’agit-il
d’une distribution standard, et si oui laquelle ?
https://jhub.cnam.fr/doc/notebooks/Representation_graphique_Python_Matplotlib.html
"""

# Graphiques pour les variables quali (de type object)
def graphics_qual(dataList_object, dataNames):
    
    for name, data in zip(dataNames, dataList_object):

        for col in data.columns:
            # Calculer la fréquence des valeurs dans la colonne 'col'
            frequence_valeurs = data[col].value_counts()
            # Sélectionner les 20 variables les plus fréquentes
            top_10_series = frequence_valeurs.head(10)
            
            """
            if(len(frequence_valeurs)>10):
                # Regrouper les autres variables sous une catégorie "Autres"
                autres_effectifs = frequence_valeurs.iloc[10:].sum()
                # Créer une série contenant les 19 valeurs les plus fréquentes et la valeur "Autres"
                top_10_series = top_10_frequences.append(pd.Series(autres_effectifs, index=['Autres']))
            """
            #print("top_10_series = ", top_10_series)
            
            #print(col)
            
            x_pos = np.arange(len(top_10_series))
            
            plt.figure(figsize=(9, 9))
            plt.bar(x_pos, top_10_series)
            plt.title(f'Barplot des {col}', fontsize=14)
            # Étiqueter les barres avec les modalités
            plt.xticks(x_pos, top_10_series.index, rotation=90, fontsize=8)
            plt.xlabel('Variables')
            plt.ylabel('Effectifs')
            #plt.show()
            plt.savefig(f'../graphics/{name}/barplot_{col}.png')
            
            #print("cercle")
            #explode = (0, 0.2, 0, 0, 0) # on isole seulement la deuxième part (c.a.d 'solid')
            labels = top_10_series.index
            proportion_pattern = top_10_series
            fig, ax = plt.subplots(figsize=(14, 14))
            ax.pie(proportion_pattern, labels=labels, #explode=explode, 
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax.axis('equal')
            plt.title(f'Répartition des {col}', fontsize=14)
            #plt.show()
            plt.savefig(f'../graphics/{name}/cercle_{col}.png')
            print("terminé")

"""
columns_object = ['agents', "criteria", 'lots', 'name']
graphics_qual(dataList_object, columns_object)
"""

# Graphiques pour les variables quanti (de type float64)
def graphics_quan(dataList_float, dataNames):
    
    for name, data in zip(dataNames, dataList_float):

        for col in data.columns:
                        
            print("col = ", col)
            
            # Scatter plot - qqplot (voir la distribution par rapport à une distribution considérée comme normale)
            # Création d'une série avec des valeurs
            serie = data[col]

            # Suppression des valeurs NaN
            serie_sans_nan = serie.dropna()
            print(len(serie_sans_nan))

            # Trier les valeurs
            serie_sans_nan_sorted = np.sort(serie_sans_nan)

            # Tracé du QQ plot
            sm.qqplot(serie_sans_nan_sorted, line ='45') 
            plt.title(f'qqplot des {col}')
            plt.savefig(f'../graphics/{name}/qqplot_{col}.png')
            
            plt.figure(figsize=(9, 9))
            plt.hist(data[col].dropna(), bins=100, density=True, label="histogramme")
            plt.title(f'Histogramme des {col}')
            plt.xlabel('Variables')
            plt.ylabel('Effectifs')
            plt.savefig(f'../graphics/{name}/histogramme_{col}.png')
            
            print("terminer")
            

dataNames = ['agents', "criteria", 'lots']
#graphics_quan(dataList_float64, dataNames)

# Analyse descriptive bi-variée

"""
# Barplot: graphe entre deux variables quali
data = dataList_object[0]

# Création de la table de données pour la heatmap
heatmap_data = data.pivot_table(index=data.columns[0], columns=data.columns[3], aggfunc=len, fill_value=0)

# Affichage du heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='d')

# Ajout des titres et des étiquettes
plt.title('Heatmap de distribution croisée entre X et Y')
plt.xlabel('Y')
plt.ylabel('X')

# Affichage du heatmap
plt.show()
"""

#agents.dropna(subset=['longitude', 'latitude'], inplace=True)

# Scatter plot entre une variable quali et une variable quanti
# Création du scatter plot avec Seaborn
def scatter(dataList, dataNames):
    
    for data,names in zip(dataList, dataNames):   
        
        print("names = ", names)
        
        data_copy = data.copy()   
        print(data.columns)
        
        if(len(list(data.columns))<2):
            continue
        
        for i in range(data_copy.shape[1]-1):
            
            print("i = ", i)
            
            col1 = data.columns[i]
            data_copy.dropna(subset=[col1], inplace=True) 
            
            print("col1 = ", col1)
            
            for j in range(i+1,data_copy.shape[1]):
                
                print("j = ", j)
                
                col2 = data_copy.columns[j]
                data_copy.dropna(subset=[col2], inplace=True) 
                
                print("col2 = ", col2)
        
                plt.figure(figsize=(8, 20))
                sns.scatterplot(data=data_copy, x=col1, y=col2, palette='viridis')

                # Ajout des titres et des étiquettes
                plt.title(f'{col2} en fonction de la {col1}')
                plt.xlabel(f'{col1}')
                plt.ylabel(f'{col2}')
                plt.savefig(f'../graphics/{names}/scatter_{col1}-{col2}.png')

                # Affichage du scatter plot
                #plt.show()

dataList = [agents, lots]
dataNames = ['agents', 'criteria', 'lots']
scatter(dataList_float64, dataNames)

###############################################################################

# Bloxplot: une variable quali et une variable quanti
def boxplots(dataList, dataNames):
    
    for data,names in zip(dataList, dataNames):
        
        # Sélectionner les colonnes de type 'object'
        colonnes_object = data.select_dtypes(include=['object']).columns.tolist()
        # Sélectionner les colonnes de type 'float64'
        colonnes_float64 = data.select_dtypes(include=['float64']).columns.tolist()
        
        for col1 in colonnes_object:
            
            print("col1 = ", col1)
            
            for col2 in colonnes_float64:
                
                print("col2 = ", col2)
                
                # Calculer la fréquence des valeurs dans la colonne 'col'
                frequence_valeurs = data[col1].value_counts()
                # Sélectionner les 20 variables les plus fréquentes
                top_20_series = frequence_valeurs.head(20)

                plt.figure(figsize=(18, 18))
                sns.boxplot(x=col1, y=col2, data=top_20_series)

                # Ajouter des titres et des étiquettes
                plt.title(f'Dispersion de {col2} en fonction de {col1}')
                plt.xlabel('Catégorie')
                plt.ylabel('Valeur')
                plt.xticks(rotation=90)
                plt.savefig(f'../graphics/{names}/boxplot_{col1}-{col2}.png')

                # Afficher le boxplot
                #plt.show()

dataList = [agents, criteria, lots]
dataNames = ["agents", "criteria", "lots"]
boxplots(dataList, dataNames)
                
###############################################################################

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
                    
                    dict_khi_2[names][f"({col1}, {col2})"] = [chi2, p, v_cramer]
                
                    # Afficher le résultat
                    print("Valeur de chi2 :", chi2)
                    print("p-valeur :", p)
                
                except ValueError as ve:
                    print(f"Erreur de valeur : {ve}. Colonne '{col1}' ou '{col2}' peut contenir des valeurs non valides.")
                except Exception as e:
                    print(f"Une erreur s'est produite : {e}")
                
    return dict_khi_2

"""      
dataNames = ['agents', "criteria", 'lots', 'name']
dict_khi_2 = khi_2(dataList_object, dataNames)
"""

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
                coeff_pearson,_ = pearsonr(copy_data[col1], copy_data[col2])
                dict_pearson[names][f"({col1}, {col2})"] = coeff_pearson
                print("coefficient de Pearson = {}".format(coeff_pearson))
    
    return dict_pearson

"""
dataNames = ['agents', "criteria", 'lots']
dict_pearson = pearson(dataList_float64, dataNames)
"""

###############################################################################

# Corrélation entre une variable quali et une quanti
# ex: country et la latitude
# https://datascientest.com/correlation-entre-variables-comment-mesurer-la-dependance

def anova(dataList, dataNames):
    
    dict_anova = {}
    
    for data,names in zip(dataList, dataNames):
        
        dict_anova[names] = {}
        
        # Sélectionner les colonnes de type 'object'
        colonnes_object = data.select_dtypes(include=['object']).columns.tolist()
        # Sélectionner les colonnes de type 'float64'
        colonnes_float64 = data.select_dtypes(include=['float64']).columns.tolist()
        
        for col_object in colonnes_object:
            
            print("col_object = ", col_object)
            
            for col_float in colonnes_float64:
                
                print("col_float = ", col_float)

                data_copy = data.copy()
                data_copy.dropna(subset=[col_float], inplace=True)
                
                # Compter le nombre d'occurrences de chaque pays
                counts = data_copy[col_object].value_counts()
                # Définir une fréquence minimale
                frequence_minimale = 10  # Par exemple, vous pouvez ajuster cette valeur selon vos besoins
                # Obtenir les pays qui ont une fréquence suffisamment élevée
                frequents = counts[counts >= frequence_minimale].index
                # Filtrer les lignes du DataFrame pour ne garder que celles correspondant aux pays fréquents
                data_frequents = data_copy[data_copy[col_object].isin(frequents)]

                # Effectuer le test d'ANOVA
                donnees = [data_frequents[data_frequents[col_object] == groupe][col_float] for groupe in data_frequents[col_object].unique()]
                result = f_oneway(*donnees)

                # Afficher le résultat
                print("Statistique de test F :", result.statistic)
                print("p-valeur :", result.pvalue)
                dict_anova[names][f"({col_object}, {col_float})"] = [result.statistic, result.pvalue]

    return dict_anova
    
"""
dataList = [agents, criteria, lots]
dataNames = ["agents", "criteria", "lots"]
dict_anova = anova(dataList, dataNames)
"""

# Usage de IsolationForest pour détecter les valeurs aberrantes
def outliers(dataList_float, dataNames):
    
    for name, data in zip(dataNames, dataList_float):

        for col in data.columns:
                        
            print("col = ", col)

            # n_estimators: nombre d'arbre dans la forêt
            # max_samples: nombre d'échantillon à utiliser
            # contamination: prop de valeur aberrantes attendue
            model = IsolationForest(n_estimators=100,max_samples='auto',contamination=0.001)
            dfx = data[[col]]
            dfx = pd.DataFrame(dfx[col].dropna())
            model.fit(dfx[[col]])
            dfx["scores"] = model.decision_function(dfx[[col]])
            dfx["anomalies"] = model.predict(dfx[[col]])
            # si la variable "anomalies" donne -1
            # alors l'individu est anormal
            
            dfx = dfx[dfx["anomalies"]==1]
            
            plt.figure(figsize=(9, 9))
            plt.hist(dfx, bins=20)
            plt.title(f'Histogramme des {col}')
            plt.xlabel('Variables')
            plt.ylabel('Effectifs')
            #plt.show()

    return dfx

dataNames = ['agents', "criteria", 'lots']
outliers(dataList_float64, dataNames)



