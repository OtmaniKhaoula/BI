import csv
import tqdm
import sqlite3

#THIS SCRIPT OPENS THE SIRENE DUMP FILES, KEEPS ONLY THE INFORMATION WE SELECTED
#AND WRITES THOSES INFORMATION INTO A SMALLER CSV FILE
#THAT CAN BE USED IN OUR EXPERIMENTATIONS



"""conn = sqlite3.connect('./../Foppa1.1.2.db')
c = conn.cursor()

siret_qry = "SELECT siret FROM Agents WHERE siret != 'NULL'"
   


sirets = c.execute(siret_qry).fetchall()

siret = []

for sir in tqdm.tqdm(sirets):
    siret.append(sir[0])"""

f = open("sirets.txt", "r")
sirets = f.read().split('\n')
print(sirets)

header = ['siren', 'nic', 'siret', 'statutDiffusionEtablissement', 'dateCreationEtablissement', 'trancheEffectifsEtablissement', 'anneeEffectifsEtablissement', 'activitePrincipaleRegistreMetiersEtablissement', 'dateDernierTraitementEtablissement', 'etablissementSiege', 'nombrePeriodesEtablissement', 'complementAdresseEtablissement', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement', 'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement', 'typeVoieEtablissement', 'libelleVoieEtablissement', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'libelleCommuneEtrangerEtablissement', 'distributionSpecialeEtablissement', 'codeCommuneEtablissement', 'codeCedexEtablissement', 'libelleCedexEtablissement', 'codePaysEtrangerEtablissement', 'libellePaysEtrangerEtablissement', 'identifiantAdresseEtablissement', 'coordonneeLambertAbscisseEtablissement', 'coordonneeLambertOrdonneeEtablissement', 'complementAdresse2Etablissement', 'numeroVoie2Etablissement', 'indiceRepetition2Etablissement', 'typeVoie2Etablissement', 'libelleVoie2Etablissement', 'codePostal2Etablissement', 'libelleCommune2Etablissement', 'libelleCommuneEtranger2Etablissement', 'distributionSpeciale2Etablissement', 'codeCommune2Etablissement', 'codeCedex2Etablissement', 'libelleCedex2Etablissement', 'codePaysEtranger2Etablissement', 'libellePaysEtranger2Etablissement', 'dateDebut', 'etatAdministratifEtablissement', 'enseigne1Etablissement', 'enseigne2Etablissement', 'enseigne3Etablissement', 'denominationUsuelleEtablissement', 'activitePrincipaleEtablissement', 'nomenclatureActivitePrincipaleEtablissement', 'caractereEmployeurEtablissement']
map_header_index = {}

for i in range(len(header)):
    map_header_index[header[i]] = i


new_header = ['caractereEmployeurEtablissement', 'dateCreationEtablissement', 'etablissementSiege', 'etatAdministratifEtablissement', 'siren', 'siret', 'statutDiffusionEtablissement', 'trancheEffectifsEtablissement', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'libelleCommuneEtrangerEtablissement', 'coordonneeLambertAbscisseEtablissement', 'coordonneeLambertOrdonneeEtablissement', 'activitePrincipaleEtablissement']

data = []
i = 0
with open('../data/sirene-dump/StockEtablissementS4F3_utf8.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    for row in tqdm.tqdm(datareader):
        print(row[2])
        if (row[2] in sirets):
            d = []
            for n in new_header:
                d.append(row[map_header_index[n]])
            
            data.append(d)
        i += 1
        if i == 100:
            break


with open('NEW-etablissementData.txt', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(new_header)
    write.writerows(data)