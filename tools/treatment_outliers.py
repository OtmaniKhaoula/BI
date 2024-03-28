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
        data['siret'] = data['siret'].fillna(0)
        data['siret'] = data['siret'].astype('int64')
        data['zipcode'] = data['zipcode'].fillna(0)
        data['zipcode'] = data['zipcode'].astype('int64')
     
    elif(names == "lots"):
        data['cpv'] = data['cpv'].fillna(0)
        data['cpv'] = data['cpv'].astype('int64')
        data['correctionsNb'] = data['correctionsNb'].astype('float64')
        data["cancelled"] = data["cancelled"].replace({0: False, 1: True})
        data["outOfDirectives"] = data["outOfDirectives"].replace({0: False, 1: True})

    #print(data.shape)
    #print(names, " = ", data.dtypes)
    
# Usage de IsolationForest pour détecter les valeurs aberrantes
def outliers(dataList_float, dataNames, indexs):
    
    dataList = []
    
    for names, data, index in zip(dataNames, dataList_float, indexs):

        for col in data.columns:
            
            if(data[col].dtypes != 'float64'):
                
                continue
            
            print("col = ", col)

            # n_estimators: nombre d'arbre dans la forêt
            # max_samples: nombre d'échantillon à utiliser
            # contamination: prop de valeur aberrantes attendue
            model = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.0005)
            dfx = data[[index, col]].dropna()  # Supprimer les valeurs NaN pour l'entraînement
            model.fit(dfx[[col]])
            #dfx["scores"] = model.decision_function(dfx[[col]])
            dfx["anomalies"] = model.predict(dfx[[col]])
            
            dfx[col] = np.where(dfx["anomalies"] == -1, np.nan, dfx[col])
            dfx = dfx.drop(columns=["anomalies"])
            
            data = pd.merge(data, dfx, on=index, how='left', suffixes=("_x", "_y"))
            data[col] = np.where(data[col+"_y"] == -1, np.nan, data[col+"_x"])
            data = data.drop(columns=[col+"_x", col+"_y"])
            
            """
            # Sélectionner les valeurs aberrantes et les stocker dans un DataFrame
            outliers = dfx[dfx["anomalies"] == -1]
            outliers_df = pd.concat([outliers_df, outliers], ignore_index=True)
            """
            
        dataList.append(data)

    return dataList

dataList = outliers(dataList, dataNames, indexs)

print(dataList[0]["awardPrice"].isna().sum())

dataList[0].to_csv("../data/lots_v2.csv", sep = ";", index = None, header=True)

"""
dfx.loc[dfx['anomalie'] == 1, col] = np.nan
data[col] = dfx[col]
                        
plt.figure(figsize=(9, 9))
sns.distplot(dfx[col], kde=True)
#plt.hist(dfx, bins=20)
plt.title(f'Histogramme des {col}')
plt.xlabel('Variables')
plt.ylabel('Effectifs')
#plt.show()
plt.savefig(f'../graphics/{names}/histogramme_without_outliers_{col}.png')
plt.close()
"""
