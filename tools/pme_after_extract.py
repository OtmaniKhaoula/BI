# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:03:51 2024

@author: Utilisateur
"""

"""
Importation des librairies
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency as chi2_contingency
plt.style.use('ggplot')

"""
Importation des tables
"""
agents = pd.read_csv("../data/agents_v2.csv", sep=";")
agents_with_effectif = pd.read_csv("../data/export_v1.csv", sep=",")
lots = pd.read_csv("../data/lots_v2.csv", sep=";")
lotSuppliers = pd.read_csv("../data/LotSuppliers.csv", sep=",")

print("Nombre de tranche inconnues =", agents_with_effectif["trancheEffectifsEtablissement"].isna().sum() + (agents_with_effectif["trancheEffectifsEtablissement"] == "NN").sum())

"""
Jointure entre agents et la table extraite avec le SIRET de l'entreprise pour avoir l'effectif d'employés dans les entreprises 
"""
new_agents = agents.merge(agents_with_effectif, on='siret', how = "inner")

"""
Jointure entre new_agents et lotSuppliers consistant à lier les lots et les infos sur les fournisseurs
"""
suppliers = new_agents.merge(lotSuppliers, on='agentId', how = "inner")

"""
Discrétisation
"""
def discretize4(data):
    
    categories = []

    for i in range(data.shape[0]):
        
        if(data.loc[i, "trancheEffectifsEtablissement"] <= 3):
                
            categories.append("TPE")
            
        elif(data.loc[i, "trancheEffectifsEtablissement"] <= 31):
                    
            categories.append("PE")
            
        elif(data.loc[i, "trancheEffectifsEtablissement"] <= 51):
                        
            categories.append("ETI")

        elif(data.loc[i, "trancheEffectifsEtablissement"] > 51):
                        
            categories.append("GE") 
            
        else:
            
            categories.append(np.nan)
            
    data["categorie_entreprise"] = categories
    
    return data

def discretize2(data):
    
    categories = []

    for i in range(data.shape[0]):
        
        if(data.loc[i, "trancheEffectifsEtablissement"] <= 31):
                
            categories.append(True)

        elif(data.loc[i, "trancheEffectifsEtablissement"] > 31):
                        
            categories.append(False) 
            
        else:
            
            categories.append(np.nan)
            
    data["pme"] = categories
    
    return data

suppliers = suppliers.replace('NN', np.nan)

suppliers['trancheEffectifsEtablissement'] = suppliers['trancheEffectifsEtablissement'].astype(float)

# Discrétiser en 2 catégories
#suppliers = discretize2(suppliers)

# Discrétiser en 4 catégories
suppliers = discretize4(suppliers)

#suppliers = suppliers[["agentId", "lotId", "pme", "categorie_entreprise", "siret"]]

"""
Regarder s'il y a un lot où il y a un seul fournisseur  et regarder si se sont des pme
"""
"""
lots_one_suppliers = lots[(lots['numberTenders']==1)]
lots_one_suppliers_v2 = lots_one_suppliers.merge(lotSuppliers, on='lotId', how = "inner")

lots_one_suppliers_v2 = lots_one_suppliers_v2[["lotId", "contractorSme"]]

lots_one_suppliers_v2 = lots_one_suppliers_v2[lots_one_suppliers_v2["contractorSme"]!=np.nan]

# Jointure
suppliers_v2 = suppliers.merge(lots_one_suppliers_v2, on='lotId', how = "left")

for i in range(suppliers_v2.shape[0]):
    
    if(suppliers_v2.loc[i, "pme"] == np.nan and suppliers_v2.loc[i, "contractorSme"] != np.nan):
        
        suppliers_v2.loc[i, "pme"] = suppliers_v2.loc[i, "contractorSme"]
    
#print("nb valeur manquante avant = ", suppliers["trancheEffectifsEtablissement"].isna().sum())
"""

"""
Grouper avec des groupBy et voir les taux de gagnants pour chaque catégories d'entreprise
"""
def group(data, column):
    
    values = data[column].value_counts()
    
    return values

#pme = group(suppliers, "pme")
categorie_entreprise = group(suppliers, "categorie_entreprise")
categorie_entreprise_values = [categorie_entreprise[2],
                               categorie_entreprise.iloc[0],
                               categorie_entreprise.iloc[1],
                               categorie_entreprise.iloc[3]]

# Micro-entreprise, pmete entreprise, Entreprise moyenne, Grande entreprise
# https://www.insee.fr/fr/statistiques/3303564?sommaire=3353488
cpt_reels_categories_2015 = [3674141, 139941, 5753, 287]
# https://www.insee.fr/fr/statistiques/4277836?sommaire=4318291
cpt_reels_categories_2017 = [3701363, 147767, 5722, 257]
# https://www.insee.fr/fr/statistiques/6666955?sommaire=6667157
cpt_reels_categories_2020 = [4085606 , 146381, 5951, 273]
 
"""
Création d'un graphique pie chart pour rendre compte de la quantité des entreprises
"""
def pie_chart(liste, labels, colors, title):
    
    for i in range(len(liste)):
        
        y = np.array(liste[i])
        total = np.sum(y)  
        percentages = y / total * 100  
        
        if(title[i] != "Part_categorie_winner"):
            plt.figure(figsize=(8, 8))
            plt.pie(y, labels=None)
            plt.legend([f'{label}: {percentage:.1f}%' for label, percentage in zip(labels, percentages)], loc='upper right')
        
        else:
            plt.pie(y, labels = labels, colors = colors, autopct='%1.1f%%')
            
        plt.savefig(f'../graphics/pme/{titles[i]}.png')
        plt.close()

liste = [categorie_entreprise_values, cpt_reels_categories_2015, cpt_reels_categories_2017, cpt_reels_categories_2020]
labels = ["TPE", "PE", "ETI", "GE"]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
titles = ["Part_categorie_winner", "Part_catégories_2015", "Part_catégories_2017", "Part_catégories_2020"]
pie_chart(liste, labels, colors, titles)

def barplot(liste, labels, labels_insee, colors, title):
    
    barWidth = 0.2
    r1 = range(len(liste[0]))
        
    for i in range(len(liste)):
        
        r2 = [x + barWidth * i for x in r1]
        print("r2 = ", r2)
        print("liste = ", liste[i])
        plt.bar(r2, liste[i], width = barWidth, color = [colors[i] for j in range(len(liste[0]))],
           edgecolor = ['black' for i in liste[0]], linewidth = 2, label=labels_insee[i])

    plt.xticks([r + barWidth * 1.5 for r in range(len(liste[i]))], labels)
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'../graphics/pme/{title}.png')
    
title = "tranche_entreprise_marche_public"
labels_insee = ["Part de lots gagnés", "2015", "2017", "2020"]
barplot(liste, labels, labels_insee, colors, title)

# Test de Khi-deux 
def chi2(team1, team2):
    
    matrice = np.array([categorie_entreprise_values, cpt_reels_categories_2015])
    khi2, pval , ddl , contingency_table = chi2_contingency(matrice)

    n = contingency_table.sum().sum()
    k = contingency_table.shape[0]
    r = contingency_table.shape[1]
    v_craETIr = np.sqrt(khi2 / (n * min(k-1, r-1)))
    
    return  pval, v_craETIr

p_2015, v_craETIr_2015 = chi2(categorie_entreprise_values, cpt_reels_categories_2015)
p_2017, v_craETIr_2017 = chi2(categorie_entreprise_values, cpt_reels_categories_2017)
p_2020, v_craETIr_2020 = chi2(categorie_entreprise_values, cpt_reels_categories_2020)

# Travail avec les divisions d'activité
#suppliers.columns















