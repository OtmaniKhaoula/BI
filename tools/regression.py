# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:56:16 2024

@author: Rim, Khaoula, Elisa

Objectif: Régression linéaire entre la variable distance (distance moyenne entre les fournisseur et les acheteurs pour chaque lot) et le prix des lots
"""

# Importation des librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

# Inportation des tables
df = pd.read_csv("../data/data_with_lots_agents_distance.csv", sep=";")

# Ou bien entre cette distance et le montant du contrat, voire une autre variable ?
# On regarde la distance en se basant sur la latitude et la longitude et on regarde le prix => Faire régression linéaire? 
"""
Régression linéaire entre la distance moyenne entre les fournisseurs et acheteurs et le prix
Seulement avec les données complétées
"""
dfx = df.dropna(subset=['distance', 'awardPrice'])

# Graphique 
plt.figure(figsize=(10, 10))
plt.scatter(dfx["distance"],dfx["awardPrice"]) 
plt.ylabel("awardPrice")
plt.xlabel("Distance")
plt.grid(True)
plt.savefig('../graphics/distance/distance_price.png')

# Régression linéaire
from sklearn.linear_model import LinearRegression

reg = LinearRegression(normalize=True)
reg.fit(dfx[["distance"]],dfx[["awardPrice"]])
a = reg.coef_
b = reg.intercept_
ordonne = dfx["distance"].min(), dfx["distance"].max(), 100
plt.scatter(dfx["distance"],dfx["awardPrice"])
plt.yscale('log')
plt.ylabel("Prix")
plt.xlabel("Distance")
plt.plot(np.array(ordonne),(a*ordonne+b).flatten(),color='r')
plt.savefig('../graphics/distance/regression_linear.png')

print(f"{a} * {ordonne} + {b}")

# => Donne pas l'impression qu'il y est un vrai lien

# https://ledatascientist.com/regression-polynomiale-avec-python/

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Avec la régression polynomiale
def degreeChoice(x,y,degree):
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
plt.savefig('../graphics/distance/quadratic_error.png')

# Voir les résultats pour différents degrès
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.scatter(dfx["distance"], dfx["awardPrice"])
        ax.plot(x_p_list[(i)*2+j+1], y_poly_pred_P_list[(i)*2+j+1], color='black', linewidth=3)
        ax.set_title(f'Regression polynomiale deg {(i)*2+j+2}', fontsize=22)
        ax.set_xlabel('Distance', fontsize=20)
        ax.set_ylabel('AwardPrice', fontsize=20)
        ax.set_yscale('log')

plt.tight_layout()
plt.savefig('../graphics/distance/linear_regression_distance_price.png')

# Coefficient de Pearson 
from scipy.stats import pearsonr

coeff_pearson, p_value = pearsonr(dfx["distance"],dfx["awardPrice"])










