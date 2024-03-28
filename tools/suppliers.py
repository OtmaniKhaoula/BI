# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:42:26 2024

Objectif: Répondre à des questions sur le fonctionnement problématique des marchés publics en se basant sur les fournisseurs

1) Répartition des contrats entre différents fournisseurs
2) Etudier les fournisseurs sortant et entrant du marché public
3) Etudier l'écart des prix entre les fournisseurs 

#Valeur aberrantes dans les dates

"""
# Import de librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

###############################################################################

###############################################################################   

# Importer table avec les lots et le id des buyers et suppliers 
data = pd.read_csv("../data/data_with_suppliers_buyers.csv", sep=";")

# Nombre de lots par fournisseurs
frequence = data["agentId_suppliers"].value_counts()

"""
x_pos = np.arange(len(frequence))

plt.figure(figsize=(9, 9))
plt.bar(x_pos, frequence)
# Étiqueter les barres avec les modalités
plt.xticks(x_pos, frequence.value_counts, rotation=90, fontsize=8)
plt.xlabel('Nombre de lots')
plt.ylabel('Fréquence')
#plt.ylim(0, max(frequence_valeurs)*1.2)
plt.show()
#plt.savefig(f'../graphics/{name}/barplot_{col}.png')
plt.close()
"""

# Jointure afin de pouvoir analyser le nombre de lots par fournisseurs et le prix des lots 
frequence = pd.DataFrame(frequence)
frequence['nb_lots'] = frequence['agentId_suppliers']
frequence['agentId_suppliers'] = frequence.index
frequence = frequence.reset_index(drop=True)

###############################################################################

###############################################################################   

# 2) Etudier les fournisseurs sortant et entrant du marché public
# Regroupement par agentId_suppliers et regarder la date du premier contrat et celle du dernier 

data_without_na = data.dropna(subset=["awardDate"])
data_without_na['awardDate'] = pd.to_datetime(data_without_na['awardDate'])

# Supprimer les dates aberrantes
fin = pd.Timestamp('2023-12-31')
resultat_filtre = data_without_na[(data_without_na['awardDate'] <= fin)]

data_suppliers = data_without_na.groupby('agentId_suppliers')['awardDate'].apply(list).reset_index()

# Dates du premier et deuxième à laquelle la décision d’attribution a été prise pour un lot à l'avantage des fournisseurs
data_suppliers["debut"] = pd.Series([min(data_suppliers.loc[i,"awardDate"]) for i in range(data_suppliers.shape[0])])
data_suppliers["fin"] = pd.Series([max(data_suppliers.loc[i,"awardDate"]) for i in range(data_suppliers.shape[0])])

data_suppliers["durate"] = data_suppliers["fin"] - data_suppliers["debut"] 
data_suppliers["durate_days"] = data_suppliers["durate"].dt.days

# Mettre en avant la durée pendant laquelle les fournisseurs étaient dans le marché public
data_suppliers["durate_days"].describe()

plt.figure(figsize=(15, 9)) 
plt.hist(data_suppliers["durate_days"], bins = 30, label="histogramme", color="skyblue")
# Ajouter des lignes verticales pour les statistiques (moyenne, médiane, etc.)
plt.axvline(np.mean(data_suppliers["durate_days"]), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {np.mean(data_suppliers["durate_days"]):.2f}')
plt.axvline(data_suppliers["durate_days"].median(), color='orange', linestyle='dashed', linewidth=1.5, label=f'Median: {data_suppliers["durate_days"].median()}')
plt.axvline(data_suppliers["durate_days"].quantile(0.25), color='green', linestyle='dashed', linewidth=1.5, label=f'25th percentile: {data_suppliers["durate_days"].quantile(0.25)}')
plt.axvline(data_suppliers["durate_days"].quantile(0.75), color='purple', linestyle='dashed', linewidth=1.5, label=f'75th percentile: {data_suppliers["durate_days"].quantile(0.75)}')
plt.axvline(np.min(data_suppliers["durate_days"]), color='blue', linestyle='dashed', linewidth=1.5, label=f'Min: {np.min(data_suppliers["durate_days"])}')
plt.axvline(np.max(data_suppliers["durate_days"]), color='black', linestyle='dashed', linewidth=1.5, label=f'Max: {np.max(data_suppliers["durate_days"])}')

#plt.yscale('log')
# Ajouter une légende
plt.legend(loc='upper right')

plt.xlabel('Durée (en jours)')
plt.ylabel('Effectifs')

#plt.ylim(0, max(frequence_valeurs.value_counts())*1.2)
plt.savefig('../graphics/suppliers/durate.png')
plt.close()

# Graphique sur les fréquences d'entrées et de sorties des fournisseurs

# Nombre d'entrées et de sorties pour chaque date
entrees = data_suppliers['debut'].value_counts().sort_index()
sorties = data_suppliers['fin'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(entrees.index, entrees.values, label='Entrées', color='blue')
plt.plot(sorties.index, sorties.values, label='Sorties', color='red')

plt.xlabel('Date')
plt.ylabel('Fréquence')
plt.legend()
plt.xlim(min(data_suppliers['debut']), max(data_suppliers['fin']))
plt.xticks(rotation=45)  # Pour faire pivoter les étiquettes de l'axe des x
plt.tight_layout()       # Pour ajuster la disposition du graphique
plt.savefig('../graphics/suppliers/date_entree_sortie.png')
plt.close()

#Lien entre la durée et le nombre de fournisseurs
data_suppliers = data_suppliers.merge(frequence, on='agentId_suppliers', how='left')
# Voir si le prix moyen a un lien avec la fréquence d'un fournisseur
data_suppliers = data_suppliers.sort_values(by='nb_lots', ascending=False)

# Graphique 
# Création du graphique prix moyen de l'attribution en fonction du nombre d'attribution du fournisseur
plt.plot(data_suppliers["nb_lots"], data_suppliers["durate_days"], marker='o', linestyle='-')

# Ajout de titres et de labels
plt.xlabel('Nombre de lots')
plt.ylabel('Durée')
plt.grid(True)
plt.savefig('../graphics/suppliers/durate_nb_lots.png')
plt.close()

# Corrélation de Pearson et P-valeur
correlation, p_value = pearsonr(data_suppliers["nb_lots"], data_suppliers["durate_days"])
print("Corrélation de Pearson :", correlation)
print("Valeur de p :", p_value)
    
###############################################################################

###############################################################################         

# Prix effectif du lot indiqué dans l’avis d’attribution

data_without_na = data.dropna(subset=["awardPrice"]).reset_index(drop=True)

data_award_price = data_without_na.groupby('agentId_suppliers')["awardPrice"].apply(list).reset_index()
#data_award_price['agentId_suppliers'] = data_award_price['agentId_suppliers'].astype(int)

# Prix et importance d'un fournisseur dans le marché public à joindre pour analyse
data_award_price = pd.DataFrame(data_award_price).merge(frequence, on='agentId_suppliers', how='left')

# Voir si le prix moyen a un lien avec la fréquence d'un fournisseur
data_award_price = data_award_price.sort_values(by='nb_lots', ascending=False)
data_award_price = data_award_price.reset_index(drop=True)

# Faire la moyenne des prix pour chaque fournisseurs
data_award_price["mean_awardPrice"] = [np.mean(data_award_price.loc[i,"awardPrice"]) for i in range(data_award_price.shape[0])]

# Création du graphique prix moyen de l'attribution en fonction du nombre d'attribution du fournisseur
plt.plot(data_award_price["nb_lots"], data_award_price["mean_awardPrice"], marker='o', linestyle='-')

# Ajout de titres et de labels
plt.xlabel('Nombre de lots')
plt.ylabel('Prix')
# Affichage du graphique
plt.savefig('../graphics/suppliers/price_nb_lots.png')
plt.close()

# Corrélation de Pearson et P-valeur
correlation, p_value = pearsonr(data_award_price["nb_lots"], data_award_price["mean_awardPrice"])
print("Corrélation de Pearson :", correlation)
print("Valeur de p :", p_value)








