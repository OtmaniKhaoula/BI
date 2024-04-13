# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:17:01 2024

@author: Utilisateur
"""

import pandas as pd
import haversine as hs   
from haversine import Unit
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""
Importation des tables
"""
agents = pd.read_csv("../data/Agents_v2.csv", sep=",")
criteria = pd.read_csv("../data/Criteria.csv", sep=",")
lotBuyers = pd.read_csv("../data/LotBuyers.csv", sep=",")
lots = pd.read_csv("../data/Lots_v2.csv", sep=";")
lotSuppliers = pd.read_csv("../data/LotSuppliers.csv", sep=",")
name = pd.read_csv("../data/Names.csv", sep=",")

"""
Garder seulement les variables numériques (NUMERIC et INTEGER)
"""
# 'correctionsNb': variance trop faible (plus de 75% des valeurs sont à 0)
# 'numberTendersSME': corrélation importante (>0.75) avec numberTenders et bcp de valeurs manquantes
# 'awardEstimatedPrice': 86% de valeurs manquantes

"""
Jointure pour lier la latitude et la longitude associé à chaque agent qu'il soit acheteur ou vendeur pour chaque lot
"""
lotBuyers = agents.merge(lotBuyers, on='agentId', how = "outer")
lotSuppliers = agents.merge(lotSuppliers, on='agentId', how = 'outer')

lots_v2 = lots.merge(lotBuyers, how = "left", on='lotId')
lots_v2 = lots_v2.merge(lotSuppliers, on='lotId', how = "left", suffixes=('_buyers', '_suppliers'))

"""
Concaténer l'ensemble des données
"""
"""
key = 'lotId'
dataList = [lots_numeric, agentsBuyers]
df = agentsSuppliers.copy()
for data in dataList:
    df = df.merge(data, on=key, suffixes=('_suppliers', '_buyers'))
"""


###############################################################################

###############################################################################

# Distance des latitude/longitude entre les acheteurs et les fournisseurs
import math

def distance_between_both_city(data, col_lat1, col_lon1, col_lat2, col_lon2):
        
    distance = []
    #print("nombre de longitude manquantes: ", data[col_lon1].isna().sum())
    
    for i in range(data.shape[0]):
        
        # Convertir les latitudes et longitudes de degrés à radians
        lat1 = data.loc[i,col_lat1]
        lon1 = data.loc[i,col_lon1]
        lat2 = data.loc[i,col_lat2]
        lon2 = data.loc[i,col_lon2]
        
        #if(lat1 == np.nan or lat2 == np.nan or lon1 == np.nan or lon2 == np.nan):
        
        if (math.isnan(lat1) or math.isnan(lon1) or math.isnan(lat2) or math.isnan(lon2)):
            #distance.append(np.nan)
            continue
        
        loc1=(lat1, lon1)
        loc2=(lat2, lon2)
        
        """
        print("loc 1 = ", loc1)
        print("loc 2 = ", loc2)
        print("i = ", i)
        """
        
        result=hs.haversine(loc1,loc2,unit=Unit.KILOMETERS)
    
        # Calculer la distance entre les deux points en kilomètres
        distance.append(result)
    
    print(data.shape)
    print(len(distance))

    data = pd.concat([data, pd.DataFrame(distance, columns = ["distance"])], axis = 1)
    
    dist = data.groupby("lotId").agg({"distance": ['mean']})
    # Fusionner avec le DataFrame original pour conserver les autres colonnes
    data = pd.merge(data.drop(columns=['distance']), dist, on='lotId', how='left')
    
    """
    Supprimer les lots dupliquées
    """
    data.drop_duplicates(subset=["lotId"],inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    #print("data = ", data.shape)
    """
    data =  (1474310, 13)
    1241703
    data =  (1474310, 13)
    """
        
    return data


df = distance_between_both_city(lots_v2, "latitude_buyers", "longitude_buyers", "latitude_suppliers", "longitude_suppliers")

col_to_drop = ["latitude_buyers", "longitude_buyers", "latitude_suppliers", "longitude_suppliers"]
df = df.drop(columns=col_to_drop)

df = df.rename(columns={df.columns[-1]: "distance"})

print(df["distance"].describe())

print(df["distance"].isna().sum())

# Graphique de la distance
plt.figure(figsize=(7, 7))
plt.hist(df["distance"].dropna(), edgecolor='black')
plt.xlabel("Distance (en km)", fontsize=18)
plt.ylabel('Log(Effectifs)', fontsize=18)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
#plt.grid(True)
plt.savefig("../graphics/distance/histogramme_distance.png")
plt.close()

#df.to_csv("../data/data_with_suppliers_buyers.csv", sep = ";", index = None, header=True)

# Séparer en plusieurs colonnes les weights dans criteria afin de savoir quels types de critères ont plus d'importance pour chaque lot
table = pd.pivot_table(criteria, index='lotId', values='weight',columns=['type'], fill_value=0)
criteria2 = table.reset_index()

# Concaténation de criteria et lots 
# df, table2
df = df.merge(criteria2, on='lotId', how='left')

df["SOCIAL"].describe()
df["TECHNICAL"].describe()
df["PRICE"].describe()
df["OTHER"].describe()
df["ENVIRONMENTAL"].describe()
df["DELAY"].describe()

#df.to_csv("../data/data_with_lots_agents_distance.csv", sep = ";", index = None, header=True)



