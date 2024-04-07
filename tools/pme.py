# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 09:03:17 2024

@author: Khaoula, Rim, Elisa

Le même genre d’approche peut s’appliquer à d’autres variables, par exemple le fait
qu’un agent soit une PME. Quelle est la part des PME dans la commande publique ? Par
secteur ? Par type d’acheteur ? Etc.

Champs: numberTendersSme et numberTenders

A voir: Faire test khi deux et corrélation entre la division et les autres données

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency as chi2_contingency

"""
Fonctions des tests statistiques
"""
# Test de khi-2
def khi_2(data, col1, col2):
    
    contingency_table = pd.crosstab(data[col1], data[col2])
    #contingency_table = data.groupby([col1, col2]).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Calcul du coefficient de V de Cramer
    n = contingency_table.sum().sum()
    k = contingency_table.shape[0]
    r = contingency_table.shape[1]
    v_cramer = np.sqrt(chi2 / (n * min(k-1, r-1)))
    
    return p, v_cramer, chi2, contingency_table

# ANOVA pour voir l'indépendance entre la variable divisions et le taux de PME
def anova(data):
    
    dfx = data.copy()

    dfx["prop"] = dfx["numberTendersSme"]/dfx["numberTenders"]
    print(dfx["prop"])
    dfx = dfx[["divisions","prop"]]
    
    result = f_oneway(*dfx.values.T)

    # Afficher le résultat
    print("Statistique de test F :", result.statistic)
    print("p-valeur :", result.pvalue)
    F, p = result.statistic, result.pvalue

    return F, p

"""
Importation des tables
"""
agents = pd.read_csv("../data/Agents.csv", sep=",")
criteria = pd.read_csv("../data/Criteria.csv", sep=",")
lotBuyers = pd.read_csv("../data/LotBuyers.csv", sep=",")
lots = pd.read_csv("../data/lots_v2.csv", sep=";")
lotSuppliers = pd.read_csv("../data/LotSuppliers.csv", sep=",")
name = pd.read_csv("../data/Names.csv", sep=",")

"""
Valeurs manquantes
"""
lots.shape[0]
lots[["numberTendersSme", "numberTenders"]].isna().sum()

"""
Supprimer les lignes où une des deux colonnes est égales à NA
"""
df = lots.dropna(subset=["numberTendersSme", "numberTenders"], axis=0)
df = df.reset_index(drop=True)

"""
Part de PME dans les commande publique
"""
sme = sum(df["numberTendersSme"])
others = sum(df["numberTenders"]) - sum(df["numberTendersSme"])

y = np.array([sme, others])
mylabels = ["pme", "others"]
mycolors = ["orange", "green"]

plt.pie(y, labels = mylabels, colors = mycolors, autopct='%1.1f%%')
#plt.title("Part des PME dans la commande publique")
plt.savefig('../graphics/pme/part_pme.png')
plt.close()

"""
Part de PME par secteur et par type d'activité
"""
# Création des colonnes en groupant les cpv selon différentes granularités
def stat_cpv(data, liste):
    
    cpv = data['cpv']
    cpt = 2
    
    new_data = data.copy()
    
    for element in liste:
        
        new_data[element] = pd.Series([str(valeur)[:cpt] for valeur in cpv])
        print(pd.Series([str(valeur)[:cpt] for valeur in cpv]))
        cpt += 1
    
    return new_data

liste = ["divisions", "groupes", "classes", "categories"]
data_with_cpv = stat_cpv(df, liste)

# Grader seulement les colonnes qui nous intéressent
data = data_with_cpv[["numberTendersSme", "numberTenders", "divisions", "groupes", "classes", "categories"]]

# value_counts modalités 
"""
data["divisions"].value_counts()
data["groupes"].value_counts() 
data["classes"].value_counts() 
data["categories"].value_counts() 
"""

# value_counts nombre de modalités 
data["divisions"].nunique()
data["groupes"].nunique() 
data["classes"].nunique() 
data["categories"].nunique()

def cpv_desc(data, liste, columns):
    
    dict_stat_cpv = {}
    
    for element in liste:
        
        dict_stat_cpv[element] = {}
        
        for col in columns:
        
            groupes = data.groupby(element).agg({col: ['sum']})
            dict_stat_cpv[element][col] = groupes
    
    return dict_stat_cpv

columns = ["numberTendersSme", "numberTenders"]

dict_stat_cpv = cpv_desc(data, liste, columns)

data_with_sums_sme = pd.concat([dict_stat_cpv["divisions"]["numberTendersSme"], dict_stat_cpv["divisions"]["numberTenders"]], axis = 1)

sme_divisions = data_with_sums_sme["numberTendersSme"].values.tolist()
sme_divisions = [val[0] for val in sme_divisions]

other_divisions = data_with_sums_sme["numberTenders"] - data_with_sums_sme["numberTendersSme"]
other_divisions = other_divisions.values.tolist()
other_divisions = [val[0] for val in other_divisions]

# Part de PME selon la division
pos = np.arange(data_with_sums_sme.shape[0])
width = 0.35  # épaisseur de chaque bâton

# Création du diagramme en bâtons (bâtons côte à côte)
plt.figure(figsize=(24,12))
plt.bar(pos - width/2, sme_divisions, width, color='lightsteelblue')
plt.bar(pos + width/2, other_divisions, width, color='IndianRed')
plt.xticks(pos, list(data_with_sums_sme.index), fontsize=20, rotation=90)
plt.xlabel('Divisions', fontsize=20)
plt.ylabel('Effectifs', fontsize=20)
#plt.title('Part de la PME dans le marché de la fonction publique pour chaque division',fontsize=26)
plt.legend(["sme", "others"],loc=1, fontsize=20)
plt.savefig('../graphics/pme/divisions.png')
plt.close()

new_data = data_with_cpv[["numberTendersSme", "numberTenders", "divisions"]]
Fdd, Pdd = anova(new_data)

# Diff entre les groupes très significatives

###############################################################################

###############################################################################

# Part de la PME gagnant une offre

"""
Supprimer les lignes où une des deux colonnes est égales à NA
"""
df = lots.dropna(subset=["contractorSme"], axis=0)
df = df.reset_index(drop=True)

print("shape de df = ", df.shape)
#df = df[(df["contractorSme"] == "Y") | (df["contractorSme"] == "N")]
#df['contractorSme'] = df['contractorSme'].replace({'Y': '1', 'N': '0'}).astype(float)

data_with_cpv = stat_cpv(df, liste)

# Grader seulement les colonnes qui nous intéressent
data = data_with_cpv[["contractorSme", "divisions", "groupes", "classes", "categories"]]

columns = ["contractorSme"]

dict_stat_cpv = cpv_desc(data, liste, columns)

sme2 = sum(data["contractorSme"])
others2 = data.shape[0] - sum(data["contractorSme"])

y = np.array([sme2, others2])
mylabels = ["pme", "others"]
mycolors = ["orange", "green"]

plt.pie(y, labels = mylabels, colors = mycolors, autopct='%1.1f%%')
#plt.title("Part des PME gagnant une offre (dans la commande publique)")
plt.savefig('../graphics/pme/part_winner_pme.png')
plt.close()

# Faire un test de khi-deux pour voir s'il y a une dépendance entre la part et les gains 
matrice = np.array([[sme, others], [sme2, others2]])
khi2, pval , ddl , contingent_theorique = chi2_contingency(matrice)
print("p-value = ", pval)

# Test entre la division et le booléen indiquant si c'est un PME qui a gagné l'offre
p, v_cramer, chi2, contingency_table = khi_2(data_with_cpv, "contractorSme", "divisions")
print("p-value = ", pval, "  v_cramer = ",  v_cramer, "chi-2 = ", chi2)

# Avoir le nombre de gagnant PME par divisions
columns = ["contractorSme"]

contractorSme = dict_stat_cpv["divisions"]["contractorSme"]
counts = data.groupby("divisions").agg({"divisions": ['count']})

sme_divisions = contractorSme.values.tolist()
sme_divisions = [val[0] for val in sme_divisions]

all_divisions = counts.values.tolist()
all_divisions = [val[0] for val in all_divisions]
other_divisions = [all_divisions[i] - sme_divisions[i] for i in range(len(all_divisions))]

# Part de PME selon la division
pos = np.arange(contractorSme.shape[0])
width = 0.35

# Graphique
plt.figure(figsize=(24,12))
plt.bar(pos - width/2, sme_divisions, width, color='lightsteelblue')
plt.bar(pos + width/2, other_divisions, width, color='IndianRed')
plt.xticks(pos, list(contractorSme.index), fontsize=20, rotation=90)
plt.xlabel('Divisions', fontsize=20)
plt.ylabel('Effectifs', fontsize=20)
#plt.title('Part de la PME winner dans le marché de la fonction publique pour chaque division',fontsize=26)
plt.legend(["sme", "others"],loc=1, fontsize=20)
plt.savefig('../graphics/pme/divisions_winner_pme.png')
plt.close()

###############################################################################

###############################################################################

# Part des projets annulés

"""
Supprimer les lignes où une des deux colonnes est égales à NA
"""
df = lots.dropna(subset=["cancelled"], axis=0)
df = df.reset_index(drop=True)

df['cancelled'] = df['cancelled'].replace({'True': '1', 'False': '0'}).astype(float)

data_with_cpv = stat_cpv(df, liste)

# Grader seulement les colonnes qui nous intéressent
data = data_with_cpv[["cancelled", "divisions", "groupes", "classes", "categories"]]

columns = ["cancelled"]

dict_stat_cpv = cpv_desc(data, liste, columns)

# Test entre la division et le booléen indiquant si c'est un PME qui a gagné l'offre
p, v_cramer, chi2, contingency_table = khi_2(data_with_cpv, "contractorSme", "divisions")
print("p-value = ", pval, "  v_cramer = ",  v_cramer, " chi2 = ", chi2)


# Préparation des graphiques pour cancelled
noCancelled = sum(data["cancelled"])
cancelled = data.shape[0] - sum(data["cancelled"])

y = np.array([noCancelled, cancelled])
mylabels = ["No Cancelled", "Cancelled"]
mycolors = ["orange", "green"]

plt.pie(y, labels = mylabels, colors = mycolors, autopct='%1.1f%%')
#plt.title("Part des contrats annulés (dans la commande publique)")
plt.savefig('../graphics/pme/part_cancelled.png')
plt.close()

# Nombre de gagnant PME par divisions
columns = ["cancelled"]

cancelled = dict_stat_cpv["divisions"]["cancelled"]
counts = data.groupby("divisions").agg({"divisions": ['count']})

cancelled_divisions = cancelled.values.tolist()
cancelled_divisions = [val[0] for val in cancelled_divisions]

all_divisions = counts.values.tolist()
all_divisions = [val[0] for val in all_divisions]
noCancelled_divisions = [all_divisions[i] - cancelled_divisions[i] for i in range(len(all_divisions))]

# Part d'offres annulées selon la division
pos = np.arange(cancelled.shape[0])
width = 0.35

# Création du diagramme en bâtons (bâtons côte à côte)
plt.figure(figsize=(24,12))
plt.bar(pos - width/2, cancelled_divisions, width, color='lightsteelblue')
plt.bar(pos + width/2, noCancelled_divisions, width, color='IndianRed')
plt.xticks(pos, list(cancelled.index), fontsize=20, rotation=90)
plt.yscale('log')
plt.xlabel('Divisions', fontsize=20)
plt.ylabel('logScale(Effectifs)', fontsize=20)
#plt.title('Part des lots annulés dans le marché de la fonction publique pour chaque division',fontsize=26)
plt.legend(["Cancelled", "no Cancelled"],loc=1, fontsize=20)
plt.savefig('../graphics/pme/divisions_cancelled.png')
plt.close()




