import sqlite3
import argparse
from tqdm import tqdm
import uuid
import re
import unidecode
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network
import community
import numpy as np
import random
import csv

number_to_region = {
    84: "Auvergne-Rhône-Alpes",
    27: "Bourgogne-Franche-Comté",
    53: "Brittany",
    24: "Centre-Val de Loire",
    94: "Corsica",
    44: "Grand Est",
    32: "Hauts-de-France",
    11: "Île-de-France",
    28: "Normandy",
    75: "Nouvelle-Aquitaine",
    76: "Occitanie",
    52: "Pays de la Loire",
    93: "Provence-Alpes-Côte d'Azur",
    1: "Guadeloupe",
    2: "Martinique",
    3: "Guyane",
    4: "Réunion",
    6: "Mayotte",
    7: "French Guiana",
    8: "Monaco",
    9: "Nouvelle-Calédonie",
    10: "Polynésie",
    12: "Saint-Pierre-et-Miquelon",
    13: "Saint-Barthélemy",
    14: "Wallis-et-Futuna"
}

"""
    PARTIE ANALYSE DU GRAPHE avec networkx

"""

# Read the CSV file into a DataFrame
edges = pd.read_csv('../data/graphs/edges/date-edges.csv', sep = ';')
nodes = pd.read_csv('../data/graphs/nodes/date-nodes.csv', sep = ';')

# Display the DataFrame
# Create a directed graph
G = nx.DiGraph()

# Add edges from DataFrame
for index, row in edges.iterrows():
    source = row['Source']
    target = row['Target']
    weight = row['Weight']
    G.add_edge(source, target, weight=weight)



"""# Set layout
pos = nx.spring_layout(G, k=1, iterations=20, scale=10)
# Draw the graph with adjusted spacing
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300, arrowsize=20)

# Draw edge labels
edge_labels = {(source, target): G[source][target]['weight'] for source, target in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)"""

#UNCOMMENT FOR DEGREEES
out_degrees ={}
in_degrees = {}
degrees = {}
difference = {}
total_degree = 0

for n in G.nodes:
    try:
        region = n #number_to_region[n]
    except:
        region = n

    out_degrees[region] = G.out_degree(n, weight='weight')
    in_degrees[region] = G.in_degree(n, weight='weight')
    degrees[region] = G.degree(n, weight='weight')
    difference[region] = in_degrees[region] - out_degrees[region]
    total_degree += degrees[region]

print("in", in_degrees)
print("out", out_degrees)
print("degree", degrees)
print("in - out", difference)


degree_percent = {}
for d in degrees:
    degree_percent[d] = round((degrees[d]/total_degree)*100, 2)

degree_percent = {k: v for k, v in sorted(degree_percent.items(), key=lambda item: item[1], reverse=True)}

print("pourcentage du marché publique", degree_percent)

#UNCOMMENT FOR CENTRALITY

#CENTRALITY
"""centrality_dict = nx.betweenness_centrality(G)
#centrality_dict = nx.closeness_centrality(G)
#centrality_dict = nx.degree_centrality(G)
centrality_dict = {k: v for k, v in sorted(centrality_dict.items(), key=lambda item: item[1], reverse=True)}

max_items = 9
labels = []
values = []

for key in centrality_dict:
    labels.append(nodes.iloc[nodes.index[nodes['Id'] == key][0]]['Label'])
    values.append(centrality_dict[key])
    max_items -= 1

    if max_items==0:
        break


plt.figure(figsize=(15, 5))
plt.bar(labels, values)
plt.xlabel('Agents')
plt.ylabel('Betweeness centrality')
plt.title('Bar Plot betweeness centrality')
#plt.xticks(rotation=20) 
plt.subplots_adjust(bottom=0.4)
xlabels_new = [label.replace(" ", '\n') for label in labels]
plt.xticks(range(9), xlabels_new)
plt.savefig("date-betweenness.png")"""


#UNCOMMENT FOR COMMUNITY 
#communities = community.best_partition(G)
communitie = nx.community.greedy_modularity_communities(G)
#communitie = list(nx.community.louvain_communities(G))
print(communitie)



communityColor = {}

colors = [(17,120,127), (255,247,164), (238,136,34), (156,64,0), (142,212,210), (140,158,68), (78,95,57), (250,234,182), (97,93,90), (64,61,54), (107,99,26), (20,24,81), (251,213,236), (0,74,255)]
i = 0

communities = {}
for j in range(len(communitie)):
    for com in communitie[j]:
        communities[com] = j

"""
data = ""
i = 0
for comm in communitie:
    for c in comm:
        data += f"{c};{i}\n"
    i+=1

with open('mycsvfile.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, communities.keys())
    w.writeheader()
    w.writerow(communities)
exit()

#print(communities)
print(communities)
print(len(communities))"""

r = lambda: random.randint(0,255)
for com in communities:
    if communities[com] not in communityColor:
        communityColor[communities[com]] = '#%02X%02X%02X' % (r(),r(),r())
        i += 1


"""
    PARTIE VISUALISATION DU GRAPHE avec pyvis

"""

G = Network(directed = True)
# Add nodes

id_to_label = {}
for _, row in nodes.iterrows():
    G.add_node(row['Id'], label=f"{row['Label']}", color=communityColor[communities[row['Id']]])
    id_to_label[row['Label']] = row['Id']

for n in G.nodes:
    #n["size"] = 50
    n["font"]={"size": 40}

# Add edges from DataFrame
for _, row in edges.iterrows():
    source = row['Source']
    target = row['Target']
    weight = row['Weight']
    G.add_edge(source, target, value=weight, title=weight) #, label=str(weight))

# Visualize the graph
G.force_atlas_2based()
G.show('Date-graph.html', notebook=False)


#THIS SCRIPT READS EDGES AND NODES FILE UNDER DATA/GRAPHS/ TO GENERATE NETWORKS


"""

comm = nx.community.girvan_newman(G)

communitie = []
for com in comm:
    for c in com:
        communitie.append(c)
print(communitie)

communityColor = {}

colors = [(17,120,127), (255,247,164), (238,136,34), (156,64,0), (142,212,210), (140,158,68), (78,95,57), (250,234,182), (97,93,90), (64,61,54), (107,99,26), (20,24,81), (251,213,236), (0,74,255)]
i = 0

communities = {}
for j in range(len(communitie)):
    for com in communitie[j]:
        communities[com] = j 

print(communities)
print(len(communities))

for com in communities:
    if communities[com] not in communityColor:
        communityColor[communities[com]] = '#%02X%02X%02X' % colors[i]
        i += 1

print(communities)

print(communityColor)
"""



