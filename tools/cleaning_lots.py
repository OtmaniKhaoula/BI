# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:56:59 2024

@author: Utilisateur
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

"""
Importation des tables
"""
agents = pd.read_csv("../data/Agents.csv", sep=",")
criteria = pd.read_csv("../data/Criteria.csv", sep=",")
lotBuyers = pd.read_csv("../data/LotBuyers.csv", sep=",")
lots = pd.read_csv("../data/Lots.csv", sep=",")
lotSuppliers = pd.read_csv("../data/LotSuppliers.csv", sep=",")
name = pd.read_csv("../data/Names.csv", sep=",")

print(lots["awardPrice"].isna().sum())

# Changer les types des colonnes insatisfaisants
"""
Identifiez la nature de la variable, ainsi que son codage. Discutez-les, notamment si
vous jugez que le codage n’est pas approprié.

Calculez les statistiques standard, en fonction de la nature de la variable : moyenne,
écart-type, quantiles, mode, min, max, etc. Discutez-les.
"""

dataList = [lots]
indexs = ["lotId"]
dataNames = ["lots"]

# A voir avec la normalisation et après garder que les valeurs entre -4 et 4

# Changement des types
for names, data in zip(dataNames, dataList):
    
    if(names == "agents"):      
        #data['siret'] = data['siret'].fillna(0)
        data['siret'] = data['siret'].astype('object')
        #data['zipcode'] = data['zipcode'].fillna(0)
        data['zipcode'] = data['zipcode'].astype('object')
     
    elif(names == "lots"):
        #data['cpv'] = data['cpv'].fillna(0)
        data['cpv'] = data['cpv'].astype('object')
        data['correctionsNb'] = data['correctionsNb'].astype('float64')
    #print(data.shape)
    #print(names, " = ", data.dtypes)


# Usage de IsolationForest pour détecter les valeurs aberrantes
def outliers(dataList_float, dataNames, indexs):
    dataList = []
    
    for names, data, index in zip(dataNames, dataList_float, indexs):
        for col in data.select_dtypes(include=['float64']).columns:
            
            if(col == "numberTenders"):
                
                continue

            model = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.0005)
            dfx = data[[index, col]].copy()  # Copier les données pour éviter les modifications indésirables
            dfx.dropna(subset=[col], inplace=True)  # Supprimer les lignes avec des valeurs NaN dans la colonne 'col'
            #dfx.reset_index(drop=True, inplace=True)  # Réinitialiser l'index après la suppression des lignes
            print(dfx.columns)
            model.fit(dfx[[col]])
            dfx["anomalies"] = model.predict(dfx[[col]])
            
            print("col = ", col)
            
            print(dfx["anomalies"].value_counts())
            
            print("avant = ",  data[col].isna().sum())
            # Remplacer les valeurs anormales par NaN dans la colonne originale
            data.loc[dfx.index, col] = np.where(dfx["anomalies"] == -1, np.nan, data.loc[dfx.index, col])
            print("après = ", data[col].isna().sum())
            
        dataList.append(data)

    return dataList

dataList = outliers(dataList, dataNames, indexs)

print(dataList[0]["awardPrice"].isna().sum())

new_lots = dataList[0]

# Supprimer les valeurs négative dans publicityDuration
for i in range(new_lots.shape[0]):
    if(new_lots.loc[i, "publicityDuration"]<0):
        new_lots.loc[i, "publicityDuration"] = np.nan
        
# Supprimer les dates allant au-delà de début 2023
new_lots['awardDate'] = pd.to_datetime(new_lots['awardDate'])
new_lots['awardDate_year'] = new_lots['awardDate'].dt.year
new_lots['awardDate_year'] = np.where(new_lots['awardDate_year'] > 2023, np.nan, new_lots['awardDate_year'])
new_lots['awardDate_year'] = new_lots['awardDate_year'].astype('str')

"""
awardDate_year = []

for i in range(new_lots.shape[0]):
    
    if isinstance(new_lots.loc[i,'awardDate'], str):
        
        awardDate_year.append(new_lots.loc[i,'awardDate'][0:4])
    
    else:
        
        awardDate_year.append(np.nan)
"""

#new_lots['awardDate_year'] = awardDate_year  
#new_lots['awardDate_year'] = [new_lots.loc[i,'awardDate'][0:4] for i in range(new_lots.shape[0]) if isinstance(new_lots.loc[i,'awardDate'], str)]


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
new_lots = stat_cpv(new_lots, liste)

"""
Regarder si numberTenders < numberTendersSme, les ramener à une égalité si c'est le cas
Remplacer par des valeurs inconnues celles qui sont erronées
"""
def verify_error(data):
    
    for i in range(data.shape[0]):
        
        if data.loc[i, "numberTenders"] == np.nan or data.loc[i, "numberTendersSme"] == np.nan:
            
            continue
        
        elif data.loc[i, "numberTendersSme"] > data.loc[i, "numberTenders"]:
        
            data.loc[i, "numberTenders"] = data.loc[i, "numberTendersSme"]
        
        if(data.loc[i,"contractorSme"]!="Y" and data.loc[i,"contractorSme"]!="N"):
            
            data.loc[i,"contractorSme"] = np.nan
    
    return data

new_lots = verify_error(new_lots)

"""
Remplacer les valeurs inconnues dans accelerated par N
"""
new_lots["accelerated"] = new_lots["accelerated"].fillna("N")

"""
Mettre les variables booléennes sous forme booléennes
"""
new_lots["cancelled"] = new_lots["cancelled"].replace({0: False, 1: True})
new_lots["outOfDirectives"] = new_lots["outOfDirectives"].replace({0: False, 1: True})
new_lots["accelerated"] = new_lots["accelerated"].replace({"N": False, "Y": True})
new_lots["onBehalf"] = new_lots["onBehalf"].replace({"N": False, "Y": True})
new_lots["jointProcurement"] = new_lots["jointProcurement"].replace({"N": False, "Y": True})
new_lots["fraAgreement"] = new_lots["fraAgreement"].replace({"N": False, "Y": True})
new_lots["contractorSme"] = new_lots["contractorSme"].replace({"N": False, "Y": True})
new_lots["subContracted"] = new_lots["subContracted"].replace({"N": False, "Y": True})                                                      
new_lots["gpa"] = new_lots["gpa"].replace({"N": False, "Y": True})
new_lots["multipleCae"] = new_lots["multipleCae"].replace({"N": False, "Y": True})
new_lots["renewal"] = new_lots["renewal"].replace({"N": False, "Y": True})                                                      

                                                          
# Exporter data
new_lots.to_csv("../data/lots_v2.csv", sep = ";", index = None, header=True)

"""
# Les valeurs manquantes sont ensuite remplacées par la médiane de la variable concernée
def replace_value_na(data, columns):
    
    print("Avant: ", data[columns].isna().sum())
    
    median = SimpleImputer(missing_values=np.nan, strategy='median')
    
    data_without_na = data.copy()
    data_without_na[columns] = median.fit_transform(data[columns])
    
    print("Après: ", data_without_na[columns].isna().sum())
    
    return data_without_na
        

# Pour la table Lots
columns = ['correctionsNb', 'awardEstimatedPrice', 'awardPrice', 'cpv',
       'numberTenders', 'contractDuration','publicityDuration']

#new_lots = replace_value_na(new_lots, columns)

# Pour la table agents
#columns = ['longitude', 'latitude']

#new_agents = replace_value_na(agents, columns)
"""









