# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:20:34 2024

@author: Utilisateur
"""

import subprocess
from exploration_v3 import statistiques, graphics_qual, graphics_quan, barplot, scatter, boxplots, date_number_lots, date_stat, pearson, anova
import pandas as pd
from cleaning import clean_names, complete_address_info, clean_contractor_sme  


# Importation des tables avant nettoyage
agents = pd.read_csv("../data/Agents.csv", sep=",")
criteria = pd.read_csv("../data/Criteria.csv", sep=",")
lotBuyers = pd.read_csv("../data/LotBuyers.csv", sep=",")
lots = pd.read_csv("../data/Lots.csv", sep=",")
lotSuppliers = pd.read_csv("../data/LotSuppliers.csv", sep=",")
name = pd.read_csv("../data/Names.csv", sep=",")

# Exploration avant nettoyage

print("------------------------------------------\nExploration avant nettoyage\n------------------------------------------")

dataList = [agents, criteria, lotBuyers, lots, lotSuppliers, name]
dataNames = ["agents", "criteria", "LotBuyers", "lots", "LotSuppliers", "name"]

describe = statistiques(dataList, dataNames)

graphics_qual(lots, "lots", ['tedCanId','cpv', 'numberTenders', 'onBehalf',
'jointProcurement', 'fraAgreement', 'fraEstimated','accelerated', 'outOfDirectives','subContracted', 'gpa', 'multipleCae', 'typeOfContract', 'topType',
'renewal'])
graphics_qual(agents, "agents", ['name', 'siret', 'address', 'city', 'zipcode', 'country','department'])
graphics_qual(name, "name", ['name'])
graphics_qual(criteria, "criteria", ['lotId', 'name','type'])
graphics_qual(lotBuyers, "LotBuyers", ['lotId', 'agentId'])
graphics_qual(lotSuppliers, "LotSuppliers", ['lotId', 'agentId'])

graphics_quan(lots, "lots", ['correctionsNb', 'awardEstimatedPrice', 'awardPrice','numberTenders','numberTendersSme','contractDuration', 'publicityDuration'])
graphics_quan(agents, "agents",['longitude', 'latitude'])
graphics_quan(criteria, "criteria", ['weight'])

barplot(lots, "lots", ['onBehalf','jointProcurement', 'outOfDirectives','topType','typeOfContract','renewal'])

scatter(lots, "lots", ["awardPrice", "awardEstimatedPrice", "numberTendersSme", "numberTenders"])

boxplots(lots, "lots", ["correctionsNb", "awardPrice"], ["cancelled", "typeOfContract"])
boxplots(criteria, "criteria", ["weight"], ["type"])

pearson(lots, "Lots", "awardPrice","awardEstimatedPrice")
pearson(lots, "Lots", "numberTendersSme", "numberTenders")
 
anova(criteria, "criteria", "weight", "type")

# Nettoyage des données (Agents et Lots)
print("------------------------------------------\nNettoyage\n------------------------------------------")

df = clean_names()
df.to_csv("../data/Agents_v2.csv", index=False, sep=",")
print('DONE CLEANING')
complete_address_info()
clean_contractor_sme()

clean = [
    #"cleaning.py",
    "cleaning_lots.py",
    ]

# Boucle pour exécuter chaque script
for chemin in clean:
    result = subprocess.run(["python", chemin], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

# Les fichiers Agents_v2 et Lots_v2 ont été généré
agents = pd.read_csv("../data/Agents_v2.csv", sep=",")
lots = pd.read_csv("../data/Lots_v2.csv", sep=";")

# Exploration des données après le nettoyage
print("------------------------------------------\nExploration après nettoyage\n------------------------------------------")

dataList = [agents, lots]
dataNames = ["agents_clean", "lots_clean"]

describe = statistiques(dataList, dataNames)

graphics_qual(lots, "lots_clean", ['tedCanId','cpv', 'numberTenders', 'onBehalf',
'jointProcurement', 'fraAgreement', 'fraEstimated','accelerated', 'outOfDirectives','subContracted', 'gpa', 'multipleCae', 'typeOfContract', 'topType',
'renewal'])
graphics_qual(agents, "agents_clean", ['name', 'siret', 'address', 'city', 'zipcode', 'country','department'])

graphics_quan(lots, "lots_clean", ['correctionsNb', 'awardEstimatedPrice', 'awardPrice','numberTenders','numberTendersSme','contractDuration', 'publicityDuration'])
graphics_quan(agents, "agents_clean",['longitude', 'latitude'])

barplot(lots, "lots_clean", ['onBehalf','jointProcurement', 'outOfDirectives','topType','typeOfContract','renewal'])

scatter(lots, "lots_clean", ["awardPrice", "awardEstimatedPrice", "numberTendersSme", "numberTenders"])

boxplots(lots, "lots_clean", ["contractDuration", "publicityDuration", "awardPrice"], ["topType", "typeOfContract"])

pearson(lots, "Lots", "awardPrice","awardEstimatedPrice")
pearson(lots, "Lots", "numberTendersSme", "numberTenders")
 
date_number_lots(lots, 'awardDate_year')
date_stat(lots, "awardPrice", "awardDate_year")

# Liste des chemins vers les fichiers Python à exécuter
questions = [
    "distance.py",
    "cpv.py",
    "clustering.py",
    "regression.py",
    "pme.py",
    "suppliers.py",
    #"extract_dump_etablissement.py",
    #"extract_dump_unite_legales.py",
    #"analyse_dump_data.py",
    "pme_after_extract.py",
    #"graph_generator.py",
    #"graph_generator_networkx.py"    
]

# Boucle pour exécuter chaque script
for chemin in questions:
    result = subprocess.run(["python", chemin], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
