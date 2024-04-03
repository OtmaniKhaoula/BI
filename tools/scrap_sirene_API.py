import requests
import sqlite3
import csv
import tqdm


#  FINALEMENT NOOUS N'AVONS PAS UTILISER CE CODE CAR IL Y A DES LIMITATIONS
#  DU NOMBRE DE REQUÊTES QUI PEUVENT ÊTRE FAIT
#  VOIR extract_dump_data.py


headers = {
  "Accept": "application/json",
  "Authorization" : "Bearer 6ccc3cfe-bad3-3b8e-acf7-f0b019b2a6fd"
}



conn = sqlite3.connect('./../Foppa1.1.2.db')
c = conn.cursor()

siret_qry = "SELECT siret FROM Agents WHERE siret != 'NULL'"
   


sirets = c.execute(siret_qry).fetchall()
s = []
file = open('sirets.txt','w')

for sir in tqdm.tqdm(sirets):
    file.write(sir[0]+'\n')

file.close()

exit()

fields = [
    "siret",
    "trancheEffectifsEtablissement",
    "activitePrincipaleRegistreMetiersEtablissement",
    "etablissementSiege",
    "etatAdministratifUniteLegale",
    "dateCreationUniteLegale",
    "denominationUniteLegale",
    "sexeUniteLegale",
    "activitePrincipaleUniteLegale",
    "economieSocialeSolidaireUniteLegale",
    "trancheEffectifsUniteLegale",
    "categorieEntreprise"
]

i = 0
data = []
for siret in tqdm.tqdm(sirets):
    try:
        data.append([])

        endpoint = f"https://api.insee.fr/entreprises/sirene/V3.11/siret/{siret[0]}"

        response = requests.get(endpoint, headers=headers)
        resp = response.json()['etablissement']

        data[-1].append(siret[0])
        data[-1].append(resp['trancheEffectifsEtablissement'])
        data[-1].append(resp['activitePrincipaleRegistreMetiersEtablissement'])
        data[-1].append(resp['etablissementSiege'])
        data[-1].append(resp['periodesEtablissement'][0]['etatAdministratifEtablissement'])
        data[-1].append(resp['dateCreationEtablissement'])
        data[-1].append(resp['uniteLegale']['sexeUniteLegale'])
        data[-1].append(resp['uniteLegale']['activitePrincipaleUniteLegale'])
        data[-1].append(resp['periodesEtablissement'][0]['changementActivitePrincipaleEtablissement'])
        data[-1].append(resp['uniteLegale']['economieSocialeSolidaireUniteLegale'])
        data[-1].append(resp['uniteLegale']['categorieEntreprise'])
        
    except KeyError:
        print("KeyError")

with open('sirene.txt', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(fields)
    write.writerows(data)

print("done")


"""
# documentation variables selectionnées:
https://www.sirene.fr/static-resources/htm/siret_unitaire_variables_reponse_311.html

00 : 0 salarié (n'ayant pas d'effectif au 31/12 mais ayant employé des salariés au cours de l'année de référence)
01 : 1 ou 2 salariés
02 : 3 à 5 salariés
03 : 6 à 9 salariés
11 : 10 à 19 salariés
12 : 20 à 49 salariés
21 : 50 à 99 salariés
22 : 100 à 199 salariés
31 : 200 à 249 salariés
32 : 250 à 499 salariés
41 : 500 à 999 salariés
42 : 1 000 à 1 999 salariés 51 : 2 000 à 4 999 salariés 52 : 5 000 à 9 999 salariés 53 : 10 000 salariés et plus
"""

