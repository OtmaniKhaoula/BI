# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:36:51 2024

@author: Rim, Khaoula, Elisa

Avec aide du code de Salima Mdhaffar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from R_square_clustering import r_square
from sklearn.decomposition import PCA

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
Garder seulement les variables numériques (NUMERIC et INTEGER)
"""
lots_numeric = lots[['lotId', 'cpv', 'correctionsNb', 'awardEstimatedPrice', 'awardPrice','numberTenders','numberTendersSme','contractDuration', 'publicityDuration']]
criteria_numeric = criteria[['lotId', 'weight']]
agents_numeric = agents[['agentId', 'longitude', 'latitude']]

"""
Jointure pour lier la latitude et la longitude associé à chaque agent qu'il soit acheteur ou vendeur pour chaque lot
"""
agentsBuyers = lotBuyers.merge(agents_numeric, on='agentId')
agentsSuppliers = lotSuppliers.merge(agents_numeric, on='agentId')

"""
Concaténer l'ensemble des données
"""
key = 'lotId'
dataList = [lots_numeric, criteria_numeric, agentsBuyers]
df = agentsSuppliers.copy()
for data in dataList:
    df = df.merge(data, on=key, suffixes=('_suppliers', '_buyers'))
    
"""
Supprimer les lignes dupliquées
"""
df = df.drop_duplicates()

"""
Remplacer les valeurs inconnues par la médiane
"""
columns = ['correctionsNb', 'awardEstimatedPrice', 'awardPrice','numberTenders','numberTendersSme','contractDuration', 'publicityDuration', 'weight', 'longitude_suppliers',
       'latitude_suppliers', 'longitude_buyers', 'latitude_buyers']
median = SimpleImputer(missing_values=np.nan, strategy='median')
median.fit(df[columns])
df_without_na = pd.DataFrame(median.transform(df[columns]), columns=columns)

y = df['cpv']

"""
Normalisation des données
"""
scaler = StandardScaler()
df_normalize = pd.DataFrame(scaler.fit_transform(df_without_na), columns = columns)

"""
Faire des stats sur les CPV
"""
# Création des colonnes en groupant les cpv selon différentes granularités
def stat_cpv(data, liste):
    
    cpv = data['cpv']
    print(cpv)
    cpt = 2
    
    new_data = data.copy()
    
    for element in liste:
        
        new_data[element] = pd.Series([str(valeur)[:cpt] for valeur in cpv])
        cpt += 1
    
    return new_data

liste = ["divisions", "groupes", "classes", "catégories"]
data_with_cpv = stat_cpv(df, liste)

# Utiliser ces nouvelles colonnes pour faire des stats par regroupement
def spv_desc(data, liste, columns):
    
    dict_stat_cpv = {}
    
    for element in liste:
        
        dict_stat_cpv[element] = {}
        
        for col in columns:
        
            groupes = data.groupby(element).agg({col: ['mean', 'median', 'min', 'max', 'count']})
            dict_stat_cpv[element][col] = groupes
    
    return dict_stat_cpv

dict_stat_cpv = spv_desc(data_with_cpv, liste, columns) 

"""
Utilisation de l'ACP pour réduire la dimension des données
"""
# Réalisation de l'ACP
"""
acp = PCA(svd_solver='full')
coord = acp.fit_transform(df_normalize)
"""

from sklearn.decomposition import IncrementalPCA

acp = IncrementalPCA()
coord = acp.fit_transform(df_normalize)

# nb of computed components
print(acp.n_components_) 

# explained variance scores
exp_var_pca = acp.explained_variance_ratio_
print(exp_var_pca)

# Cumulative sum of explained variance values; This will be used to
cum_sum_expl_var = np.cumsum(exp_var_pca)
print(cum_sum_expl_var)

# plot instances on the first plan (first 2 factors) or 2nd plan
def plot_instances_acp(coord,df_labels,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(10,10))
    axes.set_xlim(-7,9) # limits must be manually adjusted to the data
    axes.set_ylim(-7,8)
    for i in range(len(df_labels.index)):
        plt.annotate(df_labels.values[i],(coord[i,x_axis],coord[i,y_axis]), fontsize=15)
    plt.plot([-7,9],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-7,8],color='silver',linestyle='-',linewidth=1)
    plt.show()
    plt.close(fig)

# coord: results of the PCA 
plot_instances_acp(coord,y,0,1)
plot_instances_acp(coord,y,2,3)

# Plot sur l'accumulation des valeurs propres
fig = plt.figure()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_expl_var)), cum_sum_expl_var, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.close(fig)

# compute correlations between factors and original variables
loadings = acp.components_.T * np.sqrt(acp.explained_variance_)

# plot correlation_circles
def correlation_circle(components,var_names,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    minx = -1
    maxx = 1
    miny = -1
    maxy = 1
    axes.set_xlim(minx,maxx)
    axes.set_ylim(miny,maxy)
    # label with variable names
    # ignore first variable (instance name)
    for i in range(0, components.shape[1]):
        axes.arrow(0,
                   0,  # Start the arrow at the origin
                   components[i, x_axis],  #0 for PC1
                   components[i, y_axis],  #1 for PC2
             head_width=0.01,
             head_length=0.02)

        plt.text(components[i, x_axis] + 0.05,
                 components[i, y_axis] + 0.05,
                 var_names[i])
    # axes
    plt.plot([minx,maxx],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[miny,maxy],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.show()
    plt.close(fig)

# ignore 1st 2 columns: country and country_code
correlation_circle(loadings,df_normalize.columns,0,1)
correlation_circle(loadings,df_normalize.columns,2,3)

"""
Clustering avec kmeans
"""
"""
lst_k=range(2,20)
lst_rsq = []
for k in lst_k:
    print("k = ", k)
    est=KMeans(n_clusters=k,n_init=25)
    est.fit(df_normalize)
    lst_rsq.append(r_square(df_normalize, est.cluster_centers_,est.labels_,k))

print("liste de RSQ: ", lst_rsq)

fig = plt.figure()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.xlabel('k')
plt.ylabel('RSQ score')
plt.title('The Elbow Method showing the optimal k')
plt.show()
plt.close()
"""

"""
Notre label est le CPV (agents économiques)
Code CPV:  
    Les deux premiers chiffres identifient les divisions (XX000000-Y);
    Les trois premiers chiffres identifient les groupes (XXX00000-Y);
    Les quatre premiers chiffres identifient les classes (XXXX0000-Y);
    Les cinq premiers chiffres identifient les catégories (XXXXX000-Y);
"""

















