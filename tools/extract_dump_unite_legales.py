import csv
import tqdm
import sqlite3

#THIS SCRIPT OPENS THE SIRENE DUMP FILES, KEEPS ONLY THE INFORMATION WE SELECTED
#AND WRITES THOSES INFORMATION INTO A SMALLER CSV FILE
#THAT CAN BE USED IN OUR EXPERIMENTATIONS



f = open("sirets.txt", "r")
sirets = f.read().split('\n')
for i in range(len(sirets)):
    sirets[i] = sirets[i][:9]
    


header = ['siren', 'statutDiffusionUniteLegale', 'unitePurgeeUniteLegale', 'dateCreationUniteLegale', 'sigleUniteLegale', 'sexeUniteLegale', 'prenom1UniteLegale', 'prenom2UniteLegale', 'prenom3UniteLegale', 'prenom4UniteLegale', 'prenomUsuelUniteLegale', 'pseudonymeUniteLegale', 'identifiantAssociationUniteLegale', 'trancheEffectifsUniteLegale', 'anneeEffectifsUniteLegale', 'dateDernierTraitementUniteLegale', 'nombrePeriodesUniteLegale', 'categorieEntreprise', 'anneeCategorieEntreprise', 'dateDebut', 'etatAdministratifUniteLegale', 'nomUniteLegale', 'nomUsageUniteLegale', 'denominationUniteLegale', 'denominationUsuelle1UniteLegale', 'denominationUsuelle2UniteLegale', 'denominationUsuelle3UniteLegale', 'categorieJuridiqueUniteLegale', 'activitePrincipaleUniteLegale', 'nomenclatureActivitePrincipaleUniteLegale', 'nicSiegeUniteLegale', 'economieSocialeSolidaireUniteLegale', 'societeMissionUniteLegale', 'caractereEmployeurUniteLegale']
map_header_index = {}

for i in range(len(header)):
    map_header_index[header[i]] = i


new_header = ['siren','sexeUniteLegale', 'economieSocialeSolidaireUniteLegale']

data = []

with open('../data/sirene-dump/StockUniteLegaleS4F3_utf8.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    for row in tqdm.tqdm(datareader):
        
        if (row[0] in sirets):
            data.append([row[map_header_index[n]] for n in new_header])

print(data)

with open('UniteLegaleData.txt', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(new_header)
    write.writerows(data)