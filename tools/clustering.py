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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from R_square_clustering import r_square
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

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
# 'correctionsNb': variance trop faible (plus de 75% des valeurs sont à 0)
# 'numberTendersSME': corrélation importante (>0.75) avec numberTenders et bcp de valeurs manquantes
# 'awardEstimatedPrice': 86% de valeurs manquantes
lots_numeric = lots[['lotId', 'cpv', 'awardPrice','numberTenders','contractDuration', 'publicityDuration']]
criteria_numeric = criteria[['lotId', 'weight']]
agents_numeric = agents[['agentId', 'longitude', 'latitude']]

"""
Jointure pour lier la latitude et la longitude associé à chaque agent qu'il soit acheteur ou vendeur pour chaque lot
"""
lotBuyers = agents_numeric.merge(lotBuyers, on='agentId')
lotSuppliers = agents_numeric.merge(lotSuppliers, on='agentId')

lots_numeric2 = lots_numeric.merge(lotBuyers, how = "left", on='lotId')
lots_numeric2 = lots_numeric2.merge(lotSuppliers, on='lotId', how = "left", suffixes=('_buyers', '_suppliers'))

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

def distance_between_both_city(data, col_lat1, col_lat2, col_lon1, col_lon2):
    
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    
    distance = []
    
    for i in range(data.shape[0]):
        # Convertir les latitudes et longitudes de degrés à radians
        lat1 = math.radians(data.loc[i,col_lat1])
        lon1 = math.radians(data.loc[i,col_lon1])
        lat2 = math.radians(data.loc[i,col_lat2])
        lon2 = math.radians(data.loc[i,col_lon2])
    
        # Calculer la distance entre les deux points en kilomètres
        distance.append(math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * 6371)
    
    data = pd.concat([data, pd.DataFrame(distance, columns = ["distance"])], axis = 1)
    
    print("data = ", data.shape)
    print(data["lotId"].nunique())
    dist = data.groupby("lotId").agg({"distance": ['mean']})
    # Fusionner avec le DataFrame original pour conserver les autres colonnes
    data = pd.merge(data.drop(columns=['distance']), dist, on='lotId', how='left')
    
    """
    Supprimer les lots dupliquées
    """
    data.drop_duplicates(subset=["lotId"],inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    print("data = ", data.shape)
    """
    data =  (1474310, 13)
    1241703
    data =  (1474310, 13)
    """
        
    return data

df = distance_between_both_city(lots_numeric2, "latitude_buyers", "longitude_buyers", "latitude_suppliers", "longitude_suppliers")

col_to_drop = ["agentId_suppliers", "agentId_buyers", "latitude_buyers", "longitude_buyers", "latitude_suppliers", "longitude_suppliers"]
df = df.drop(columns=col_to_drop)

df = df.rename(columns={df.columns[-1]: "distance"})

#df2.to_csv("data_with_lots_agents_distance", sep = ";", index = None, header=True)
# Certains lots ne sont pas dans critères => ne contiennent pas de poids?
# Pour l'instant on ne les garde pas 
# Certains lotId dans criteria ne sont pas dans df2

# Certains lots n'ont pas de poids spécifiés

# A faire:
    
# Ou bien entre cette distance et le montant du contrat, voire une autre variable ?
# On regarde la distance en se basant sur la latitude et la longitude et on regarde le prix => Faire régression linéaire? 
"""
Régression linéaire entre la distance moyenne entre les fournisseurs et acheteurs et le prix
Seulement avec les données complétées
"""

data_without_na = df.dropna()

"""
from sklearn.ensemble import IsolationForest
# Détection des valeurs aberrantes dans awardPrice
model = IsolationForest(n_estimators=100,max_samples='auto',contamination=0.0002)
dfx = data_without_na[["awardPrice"]]
model.fit(dfx)
data_without_na["scores"] = model.decision_function(dfx)
data_without_na["anomalies"] = model.predict(dfx[["awardPrice"]])
# si la variable "anomalies" donne -1
# alors l'individu est anormal
dfx = data_without_na[data_without_na["anomalies"]==1]

# Graphique 
plt.figure(figsize=(10, 10))
plt.scatter(dfx["distance"],dfx["awardPrice"]) 
plt.ylabel("Prix du lot")
plt.xlabel("Distance moyenne entre les fournisseurs et les acheteurs pour chaque lots")
plt.grid(True)
plt.show()

# Régression linéaire
from sklearn.linear_model import LinearRegression

reg = LinearRegression(normalize=True)
reg.fit(dfx[["distance"]],dfx[["awardPrice"]])
a = reg.coef_
b = reg.intercept_
ordonne = dfx["distance"].min(), dfx["distance"].max(), 100
plt.scatter(dfx["distance"],dfx["awardPrice"])
plt.xlim(dfx["distance"].min(), dfx["distance"].max())
plt.plot(np.array(ordonne),(a*ordonne+b).flatten(),color='r')

# => Donne pas l'impression qu'il y est un vrai lien

# https://ledatascientist.com/regression-polynomiale-avec-python/

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import operator

# Avec la régression polynomiale
def degreeChoice (x,y,degree):
    polynomial_features= PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

    # Tri des données en fonction de la première colonne de x_poly
    sort_axis = np.argsort(x_poly[:, 1])
    x_p = x.iloc[sort_axis,:].values
    y_poly_pred_P = y_poly_pred[sort_axis]

    return rmse, x_p, y_poly_pred_P
 
rmselist = np.zeros(40)
x_p_list = [None]*40
y_poly_pred_P_list=[None]*40
for i in np.arange(1, 41):
     
    rmselist[i-1],x_p_list[i-1],y_poly_pred_P_list[i-1]= degreeChoice(dfx[["distance"]],dfx[["awardPrice"]],i)
     
plt.plot(np.arange(1, 41), rmselist, color='r')
plt.ylabel("erreur quadratique")
plt.xlabel("degré du polynôme")
plt.show()

# Voir les résultats pour différents degrès
fig, axs = plt.subplots(3, 2,figsize=(20,20))
 
axs[0, 0].scatter(dfx[["distance"]],dfx[["awardPrice"]])
axs[0, 0].plot(x_p_list[0],y_poly_pred_P_list[0],color='g')
axs[0, 0].set_title('Regression linéaire simple')
 
#degre 4
axs[0, 1].scatter(dfx[["distance"]],dfx[["awardPrice"]])
axs[0, 1].plot(x_p_list[3],y_poly_pred_P_list[3],color='g')
axs[0, 1].set_title('Regression polynomiale deg 2')
 
#degre 8
axs[1, 0].scatter(dfx[["distance"]],dfx[["awardPrice"]])
axs[1, 0].plot(x_p_list[7],y_poly_pred_P_list[7],color='g')
axs[1, 0].set_title('Regression polynomiale deg 4')
 
#degre 22
axs[1, 1].scatter(dfx[["distance"]],dfx[["awardPrice"]])
axs[1, 1].plot(x_p_list[21],y_poly_pred_P_list[21],color='g')
axs[1, 1].set_title('Regression polynomiale deg 16')
 
#degre 30
axs[2, 0].scatter(dfx[["distance"]],dfx[["awardPrice"]])
axs[2, 0].plot(x_p_list[29],y_poly_pred_P_list[29],color='g')
axs[2, 0].set_title('Regression polynomiale deg 32')
 
#degre 40
axs[2, 1].scatter(dfx[["distance"]],dfx[["awardPrice"]])
axs[2, 1].plot(x_p_list[39],y_poly_pred_P_list[39],color='g')
axs[2, 1].set_title('Regression polynomiale deg 64')
for ax in axs.flat:
    ax.set(xlabel='x', ylabel='y')
 
for ax in axs.flat:
    ax.label_outer()
"""
###############################################################################

###############################################################################

# Séparer en plusieurs colonnes les weights dans criteria afin de savoir quels types de critères ont plus d'importance pour chaque lot
table = pd.pivot_table(criteria, index='lotId', values='weight',columns=['type'], fill_value=0)
criteria2 = table.reset_index()

# Concaténation de criteria et lots 
# df, table2
data = df.merge(criteria2, on='lotId', how='left')

"""
Notre label est le CPV (agents économiques)
Code CPV:  
    Les deux premiers chiffres identifient les divisions (XX000000-Y);
    Les trois premiers chiffres identifient les groupes (XXX00000-Y);
    Les quatre premiers chiffres identifient les classes (XXXX0000-Y);
    Les cinq premiers chiffres identifient les catégories (XXXXX000-Y);
"""

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

liste = ["divisions", "groupes", "classes", "categories"]
data_with_cpv = stat_cpv(data, liste)

# Utiliser ces nouvelles colonnes pour faire des stats par regroupement
def spv_desc(data, liste, columns):
    
    dict_stat_cpv = {}
    
    for element in liste:
        
        dict_stat_cpv[element] = {}
        
        for col in columns:
        
            groupes = data.groupby(element).agg({col: ['mean', 'median', 'min', 'max', 'count']})
            dict_stat_cpv[element][col] = groupes
    
    return dict_stat_cpv

columns = ['awardPrice','numberTenders','contractDuration','publicityDuration','distance','DELAY','ENVIRONMENTAL','OTHER','PRICE','SOCIAL','TECHNICAL']

dict_stat_cpv = spv_desc(data_with_cpv, liste, columns) 

"""
Test ANOVA entre cpv et la distance
""" 
from scipy.stats import f_oneway

def anova(data, col):
    
    dfx = data.copy()
    
    scaler = StandardScaler()
    norm = pd.DataFrame(scaler.fit_transform(dfx[[col]]), columns=[col])  # Correction ici
    dfx["distance"] = norm
    
    result = f_oneway(*dfx.values.T)

    # Afficher le résultat
    print("Statistique de test F :", result.statistic)
    print("p-valeur :", result.pvalue)
    F, p = result.statistic, result.pvalue

    return F, p

Fdd, Pdd = anova(data_with_cpv[["distance","divisions"]], "distance")
Fad, Pad = anova(data_with_cpv[["awardPrice","divisions"]], "awardPrice")

###############################################################################

###############################################################################

# Corrélation entre deux variables quanti (float64)
def pearson(data):
    
    dict_pearson = {}
      
    for i in range(data.shape[1]-1):
            
        for j in range(i+1, data.shape[1]):
              
            copy_data = data.copy()
            col1 = copy_data.columns[i]
            col2 = copy_data.columns[j]
               
            copy_data.dropna(subset=[col1, col2], inplace=True)
            coeff_pearson,_ = pearsonr(copy_data[col1], copy_data[col2])
            dict_pearson[f"({col1}, {col2})"] = coeff_pearson
            if(coeff_pearson>0.5 or coeff_pearson<-0.5):
                print("col 1 = ", col1, " col 2 = ", col2)
                print("coefficient de Pearson = {}".format(coeff_pearson))
    
    return dict_pearson

coef_pearson = pearson(data[['awardPrice','numberTenders','contractDuration','publicityDuration',
       'distance','DELAY','ENVIRONMENTAL','OTHER','PRICE','SOCIAL','TECHNICAL']])

"""
Remplacer les valeurs inconnues par la médiane
"""
"""
columns = ['awardPrice','numberTenders','contractDuration', 'publicityDuration', 'weight', 'longitude_suppliers',
       'latitude_suppliers', 'longitude_buyers', 'latitude_buyers']

# Dans le cas où on supprime les valeurs aberrantes, il peut être interessant de regarder avec la moyenne
median = SimpleImputer(missing_values=np.nan, strategy='median')
median.fit(df[columns])
df_without_na = pd.DataFrame(median.transform(df[columns]), columns=columns)

print("colonnes = ", df_without_na.columns)
"""
# Corrélation >0.89 pour latitude_suppliers, latitude_buyers et longitude_suppliers, longitude_buyers >0.8
data_num = data_with_cpv[['divisions', 'awardPrice','numberTenders','contractDuration','publicityDuration',
       'distance','DELAY','ENVIRONMENTAL','OTHER','PRICE','SOCIAL','TECHNICAL']]

df_without_na = data_num.dropna()
df_without_na.reset_index(drop=True, inplace=True)

y = df_without_na['divisions']

df_without_na = df_without_na.drop(['divisions'], axis = 1)

"""
Normalisation des données
"""
scaler = StandardScaler()
df_normalize = pd.DataFrame(scaler.fit_transform(df_without_na), columns = df_without_na.columns)

coef_pearson_normalize = pearson(df_normalize)

"""
Utilisation de l'ACP pour réduire la dimension des données
"""
# Réalisation de l'ACP
"""
acp = PCA(svd_solver='full')
coord = acp.fit_transform(df_normalize)
"""
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

# Plot sur l'accumulation des valeurs propres
fig = plt.figure()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_expl_var)), cum_sum_expl_var, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('../graphics/acp/components.png')
plt.close(fig)

# plot instances on the first plan (first 2 factors) or 2nd plan
def plot_instances_acp(coord,df_labels,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(10,10))
    axes.set_xlim(-7,9) # limits must be manually adjusted to the data
    axes.set_ylim(-7,8)
    for i in range(len(df_labels.index)):
        plt.annotate(df_labels.values[i][:2],(coord[i,x_axis],coord[i,y_axis]), fontsize=15)
    plt.plot([-7,9],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-7,8],color='silver',linestyle='-',linewidth=1)
    plt.savefig(f'../graphics/acp/axis_{x_axis}_{y_axis}.png')
    plt.close(fig)

# coord: results of the PCA 
index = [0,1000,2000,3000,10000,20000,30000,40000,50000,60000,70000,100000,200000,300000,400000,500000,600000]
y = y.astype('str')
plot_instances_acp(coord[index,:],y[index],0,1)
plot_instances_acp(coord[index,:],y[index],2,3)

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
    plt.savefig(f'../graphics/acp/correlation_circle_{x_axis}_{y_axis}.png')
    plt.close(fig)

# ignore 1st 2 columns: country and country_code
correlation_circle(loadings,df_normalize.columns,0,1)
correlation_circle(loadings,df_normalize.columns,2,3)
"""
"""
Clustering avec kmeans
"""
lst_k=range(5,25)
lst_rsq = []
for k in lst_k:
    print("k = ", k)
    est=KMeans(n_clusters=k,n_init=25)
    est.fit(df_normalize.values)
    lst_rsq.append(r_square(df_normalize.values, est.cluster_centers_,est.labels_,k))

print("liste de RSQ: ", lst_rsq)

fig = plt.figure()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.xlabel('k')
plt.ylabel('RSQ score')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('../graphics/clustering/rsq_kmeans_with_distance.png')
plt.close()

"""
# k = 8 RSQ = 0.89
est=KMeans(n_clusters=8,n_init=25)
est.fit(coord[:,:3])

center = est.cluster_centers_
labels = est.labels_
r_square(coord[:,:3], center, labels, 8)

# Intégrer les labels aux données pour faire le lien avec les CVP
cpv_labels = pd.concat([y, pd.DataFrame(labels, columns = ["labels"])], axis = 1)

# Utiliser ces nouvelles colonnes pour faire des stats par regroupement
def cpv_desc_labels(data, liste, columns):
    
    dict_stat_cpv = {}
    
    for element in liste:
        
        dict_stat_cpv[element] = {}
              
        groupes = data.groupby(element)[columns].value_counts().unstack(fill_value=0)
        dict_stat_cpv[element] = groupes
    
    return dict_stat_cpv

liste = ["divisions", "groupes", "classes", "categories"]
columns = "labels"
cpv_labels = stat_cpv(cpv_labels, liste)
dict_labels_count_cpv = cpv_desc_labels(cpv_labels, liste, columns)

CT = pd.crosstab(cpv_labels["divisions"], cpv_labels["labels"])

# Axe 1: Corrélation positive avec la latitude et négative avec la longitude
# Axe 2: Corrélation positive avec la latitude et la longitude
# Axe 3: Corrélation positive avec publicityDuration et contractDuration (>0.5) et négative avec weight (<-0.5)

# k = 4 RSQ = 0.75
est=KMeans(n_clusters=4,n_init=25)
est.fit(coord[:,:3])

center = est.cluster_centers_
labels = est.labels_

# Intégrer les labels aux données pour faire le lien avec les CVP
cpv_labels = pd.concat([cpv_labels, pd.DataFrame(labels, columns = ["labels_4"])], axis = 1)
CT = pd.crosstab(cpv_labels["divisions"], cpv_labels["labels_4"])

# Calculer les sommes des lignes et des colonnes
contingency_table_with_totals = CT.copy()
contingency_table_with_totals.loc['Total'] = contingency_table_with_totals.sum()
contingency_table_with_totals['Total'] = contingency_table_with_totals.sum(axis=1)

from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(CT)
print("\nRésultats du test du chi-deux:")
print(f"Chi2: {chi2}")
print(f"P-valeur: {p}")
print(f"Degrés de liberté: {dof}")
# Pas de dépendance

# V-Cramer
n = CT.sum().sum()
k = CT.shape[0]
r = CT.shape[1]
v_cramer = np.sqrt(chi2 / (n * min(k-1, r-1)))
print(f"V-Cramer: {v_cramer}")

# sans ACP: k=12
est=KMeans(n_clusters=12,n_init=25)
est.fit(df_normalize)

center = est.cluster_centers_
labels = est.labels_

cpv_labels = pd.concat([cpv_labels, pd.DataFrame(labels, columns = ["labels_without_cpv"])], axis = 1)
CT = pd.crosstab(cpv_labels["divisions"], cpv_labels["labels_without_cpv"])

# Calculer les sommes des lignes et des colonnes
contingency_table_with_totals = CT.copy()
contingency_table_with_totals.loc['Total'] = contingency_table_with_totals.sum()
contingency_table_with_totals['Total'] = contingency_table_with_totals.sum(axis=1)

from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(CT)
print("\nRésultats du test du chi-deux:")
print(f"Chi2: {chi2}")
print(f"P-valeur: {p}")
print(f"Degrés de liberté: {dof}")
# Pas de dépendance

# V-Cramer
n = CT.sum().sum()
k = CT.shape[0]
r = CT.shape[1]
v_cramer = np.sqrt(chi2 / (n * min(k-1, r-1)))
print(f"V-Cramer: {v_cramer}")
"""

###############################################################################

###############################################################################

# Y a-t-il un lien entre l’attribution d’un contrat et la distance spatiale séparant l’acheteur et le fournisseur remportant le marché ? 
# Analyser corrélations entre les latitudes et longitudes des acheteurs et des vendeurs ??? 
# => Si c'est le cas, on a une corrélation assez forte entre les variables 

# A faire:
# Ou bien entre cette distance et le montant du contrat, voire une autre variable ?
# On regarde la distance en se basant sur la latitude et la longitude et on regarde le prix => Faire régression linéaire? 
# cpv 

"""
Peut-on identifier des classes de similarité d’agents économiques ? De lots ? Quelles sont
les variables et valeurs qui permettent de distinguer ces classes, et/ou qui sont caractéris-
tiques de certaines classes ? Comment interpréter ces classes, que signifient-elles ? Y a-t-il
des anomalies (lots ou agents) ? Si oui, lesquelles et comment les expliquer ?
Y a-t-il des différences entre agents et lots issus de secteurs d’activité différents ? Si
oui, lesquelles, et comment les interpréter ? Mêmes questions en considérant les divisions
administratives à la place du secteur d’activité.
"""
# distance entre chaque clients et fournisseurs (peut en avoir plusieurs par lots) 
# graphes départements collaborations avec poids (prix, aspect environnementaux, ...) => Type

# A voir: se baser sur les stats pour chaque divisions, activités etc pour faire un clustering au lieu de prendre toutes les instances 
# Voir aussi comment évolue les lots etc au fil des années 




