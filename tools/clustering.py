# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:49:45 2024

@author: Rim, Khaoula, Elisa

Faire des histogrammes et stats pour chaque critères (DELAY, PRICE ...)
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
plt.style.use('ggplot')

"""
Importation des tables
"""
data = pd.read_csv("../data/data_with_suppliers_buyers.csv", sep=";")
data = pd.read_csv("../data/data_with_lots_agents_distance.csv", sep=";")

print("dimension de data = ", data.shape)
print("description de data: \n", data.describe())

print("Nombre de valeur manquante par colonne: ", data.isna().sum())

# Calcul de la variance de plusieurs colonnes
#columns = ['awardPrice','numberTenders','contractDuration','publicityDuration','distance','numberTendersSme']
#columns = ['DELAY', 'ENVIRONMENTAL', 'OTHER', 'PRICE', 'SOCIAL', 'TECHNICAL', 'awardPrice','numberTenders','contractDuration','publicityDuration','distance']
columns = ['DELAY', 'ENVIRONMENTAL', 'OTHER','SOCIAL', 'awardPrice']

data = data[columns + ['lotId', 'divisions', 'topType', 'typeOfContract']]

#data = data.dropna().reset_index(drop=True)

# 'SOCIAL' quasi que des 0: 1014916
# 'OTHER' quasi que des 0:  813497
# 'ENVIRONMENTAL': quasi que des 0: 903367
# 'DELAY': quasi que des 0: 901244

variance_multiple = data[columns].var()

print("Variance des colonnes :\n", variance_multiple)
 
y = data['lotId']

print("Value counts divisions: \n", y.value_counts())

###############################################################################

###############################################################################
"""
Traitement des valeurs manquantes: voir si c'est mieux de directement supprimer les lignes
Remplacer les valeurs inconnues par la médiane
"""

# Dans le cas où on supprime les valeurs aberrantes, il peut être interessant de regarder avec la moyenne
median = SimpleImputer(missing_values=np.nan, strategy='median')
median.fit(data[columns])
df_without_na = pd.DataFrame(median.transform(data[columns]), columns=columns)

print("colonnes = ", df_without_na.columns)

"""
df_without_na_cat = data.dropna(subset=columns)
df_without_na = df_without_na_cat[columns]
"""

"""
Normalisation des données
"""
scaler = StandardScaler()
df_normalize = pd.DataFrame(scaler.fit_transform(df_without_na), columns = df_without_na.columns)

"""
Utilisation de l'ACP pour réduire la dimension des données
"""
acp = PCA(svd_solver='full')
coord = acp.fit_transform(df_normalize)

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
plt.ylabel('Part de variance expliquée')
plt.xlabel('Index des principaux composants')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('../graphics/acp/components.png')
plt.close(fig)

# plot instances on the first plan (first 2 factors) or 2nd plan
def plot_instances_acp(coord,df_labels,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(24,24))
    axes.set_xlim(-7,9) # limits must be manually adjusted to the data
    axes.set_ylim(-7,8)
    for i in range(len(df_labels.index)):
        plt.annotate(df_labels.values[i][:2],(coord[i,x_axis],coord[i,y_axis]), fontsize=20)
    plt.plot([-7,9],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-7,8],color='silver',linestyle='-',linewidth=1)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.savefig(f'../graphics/acp/axis_{x_axis}_{y_axis}.png')
    plt.close(fig)

# coord: results of the PCA 
"""
index = [0,1000,2000,3000,10000,20000,30000,40000,50000,60000,70000,100000,200000,300000]
y = y.astype('str')
plot_instances_acp(coord,y,0,1)
plot_instances_acp(coord,y,2,3)
"""

# compute correlations between factors and original variables
loadings = acp.components_.T * np.sqrt(acp.explained_variance_)

# plot correlation_circles
def correlation_circle(components,var_names,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(24,24))
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
             head_width=0.03,
             head_length=0.05)

        plt.text(components[i, x_axis] + 0.05,
                 components[i, y_axis] + 0.05,
                 var_names[i],
                 fontsize=30)
    # axes
    plt.plot([minx,maxx],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[miny,maxy],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.savefig(f'../graphics/acp/correlation_circle_{x_axis}_{y_axis}.png')
    plt.close(fig)

correlation_circle(loadings,df_normalize.columns,0,1)
correlation_circle(loadings,df_normalize.columns,2,3)

"""
#Clustering avec kmeans
"""
"""
# Avec ACP
lst_k=range(2,27,3)
lst_rsq = []
for k in lst_k:
    print("k = ", k)
    est=KMeans(n_clusters=k,n_init=25)
    est.fit(coord[:,:3])
    lst_rsq.append(r_square(coord[:,:3], est.cluster_centers_,est.labels_,k))

print("liste de RSQ: ", lst_rsq)

fig = plt.figure()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.xlabel('k')
plt.ylabel('RSQ score')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('../graphics/clustering/rsq_kmeans_with_acp.png')
plt.close()

# Sans ACP
lst_k=range(2,27,3)
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
plt.savefig('../graphics/clustering/rsq_kmeans.png')
plt.close()
"""

# Avec ACP k = 8 RSQ = 0.8
est_with_acp=KMeans(n_clusters=8,n_init=25)
est_with_acp.fit(coord[:,:3])

center_with_acp = est_with_acp.cluster_centers_
labels_with_acp = est_with_acp.labels_
r_square(coord[:,:3], center_with_acp, labels_with_acp, 8)

df_with_labels = pd.concat([data, pd.DataFrame(labels_with_acp, columns = ["labels_with_acp"])], axis = 1)

print(r_square(coord[:,:3], est_with_acp.cluster_centers_, est_with_acp.labels_,8))

CT = pd.crosstab(df_with_labels["divisions"],df_with_labels["labels_with_acp"])
CT_percentage = pd.crosstab(df_with_labels["divisions"], df_with_labels["labels_with_acp"], normalize='index')

CT_percentage = pd.crosstab(df_with_labels["divisions"], df_with_labels["labels_with_acp"], normalize='columns')

stats = df_with_labels.groupby("labels_with_acp").agg({"awardPrice": ['mean', 'median', 'min', 'max', 'count']})

#CT = pd.crosstab(cpv_labels["divisions"], cpv_labels["labels"])

# Axe 1: Corrélation positive avec la latitude et négative avec la longitude
# Axe 2: Corrélation positive avec la latitude et la longitude
# Axe 3: Corrélation positive avec publicityDuration et contractDuration (>0.5) et négative avec weight (<-0.5)

# Sans ACP k = 8 RSQ = 0.7
est=KMeans(n_clusters=8,n_init=25)
est.fit(df_normalize)

center = est.cluster_centers_
labels = est.labels_

print(r_square(df_normalize.values, est.cluster_centers_, est.labels_,8))

df_with_labels = pd.concat([df_with_labels, pd.DataFrame(labels, columns = ["labels"])], axis = 1)

# Intégrer les labels aux données pour faire le lien avec les CVP
#cpv_labels = pd.concat([cpv_labels, pd.DataFrame(labels, columns = ["labels_4"])], axis = 1)
#CT = pd.crosstab(cpv_labels["divisions"], cpv_labels["labels_4"])
"""

# Calculer les sommes des lignes et des colonnes
"""
"""
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