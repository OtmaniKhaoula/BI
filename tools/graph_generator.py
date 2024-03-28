import sqlite3
import argparse
from tqdm import tqdm
import uuid
import re
import unidecode

french_departments = {
    '01': 'Auvergne-Rhône-Alpes',
    '02': 'Hauts-de-France',
    '03': 'Auvergne-Rhône-Alpes',
    '04': 'Provence-Alpes-Côte d\'Azur',
    '05': 'Provence-Alpes-Côte d\'Azur',
    '06': 'Provence-Alpes-Côte d\'Azur',
    '07': 'Auvergne-Rhône-Alpes',
    '08': 'Grand Est',
    '09': 'Occitanie',
    '10': 'Grand Est',
    '11': 'Occitanie',
    '12': 'Occitanie',
    '13': 'Provence-Alpes-Côte d\'Azur',
    '14': 'Normandy',
    '15': 'Auvergne-Rhône-Alpes',
    '16': 'Nouvelle-Aquitaine',
    '17': 'Nouvelle-Aquitaine',
    '18': 'Centre-Val de Loire',
    '19': 'Nouvelle-Aquitaine',
    '21': 'Bourgogne-Franche-Comté',
    '22': 'Brittany',
    '23': 'Nouvelle-Aquitaine',
    '24': 'Nouvelle-Aquitaine',
    '25': 'Bourgogne-Franche-Comté',
    '26': 'Auvergne-Rhône-Alpes',
    '27': 'Normandy',
    '28': 'Centre-Val de Loire',
    '29': 'Brittany',
    '20': 'Corsica',
    '2A': 'Corsica',
    '2B': 'Corsica',
    '30': 'Occitanie',
    '31': 'Occitanie',
    '32': 'Occitanie',
    '33': 'Nouvelle-Aquitaine',
    '34': 'Occitanie',
    '35': 'Brittany',
    '36': 'Centre-Val de Loire',
    '37': 'Centre-Val de Loire',
    '38': 'Auvergne-Rhône-Alpes',
    '39': 'Bourgogne-Franche-Comté',
    '40': 'Nouvelle-Aquitaine',
    '41': 'Centre-Val de Loire',
    '42': 'Auvergne-Rhône-Alpes',
    '43': 'Auvergne-Rhône-Alpes',
    '44': 'Pays de la Loire',
    '45': 'Centre-Val de Loire',
    '46': 'Occitanie',
    '47': 'Nouvelle-Aquitaine',
    '48': 'Occitanie',
    '49': 'Pays de la Loire',
    '50': 'Normandy',
    '51': 'Grand Est',
    '52': 'Grand Est',
    '53': 'Pays de la Loire',
    '54': 'Grand Est',
    '55': 'Grand Est',
    '56': 'Brittany',
    '57': 'Grand Est',
    '58': 'Bourgogne-Franche-Comté',
    '59': 'Hauts-de-France',
    '60': 'Hauts-de-France',
    '61': 'Normandy',
    '62': 'Hauts-de-France',
    '63': 'Auvergne-Rhône-Alpes',
    '64': 'Nouvelle-Aquitaine',
    '65': 'Occitanie',
    '66': 'Occitanie',
    '67': 'Grand Est',
    '68': 'Grand Est',
    '69': 'Auvergne-Rhône-Alpes',
    '70': 'Bourgogne-Franche-Comté',
    '71': 'Bourgogne-Franche-Comté',
    '72': 'Pays de la Loire',
    '73': 'Auvergne-Rhône-Alpes',
    '74': 'Auvergne-Rhône-Alpes',
    '75': 'Île-de-France',
    '76': 'Normandy',
    '77': 'Île-de-France',
    '78': 'Île-de-France',
    '79': 'Nouvelle-Aquitaine',
    '80': 'Hauts-de-France',
    '81': 'Occitanie',
    '82': 'Occitanie',
    '83': 'Provence-Alpes-Côte d\'Azur',
    '84': 'Provence-Alpes-Côte d\'Azur',
    '85': 'Pays de la Loire',
    '86': 'Nouvelle-Aquitaine',
    '87': 'Nouvelle-Aquitaine',
    '88': 'Grand Est',
    '89': 'Bourgogne-Franche-Comté',
    '90': 'Bourgogne-Franche-Comté',
    '91': 'Île-de-France',
    '92': 'Île-de-France',
    '93': 'Île-de-France',
    '94': 'Île-de-France',
    '95': 'Île-de-France',
    '98': 'Monaco',
    '971': 'Guadeloupe',
    '972': 'Martinique',
    '973': 'French Guiana',
    '974': 'Réunion',
    '975': 'Saint-Pierre-et-Miquelon',
    '976': 'Mayotte',
    '988': 'Nouvelle-Calédonie',
    '987' : 'Polynésie',
    '977' : 'Saint-Barthélemy',
    '986' : "Wallis-et-Futuna"
}


region_to_number = {
    "Auvergne-Rhône-Alpes": 84,
    "Bourgogne-Franche-Comté": 27,
    "Brittany": 53,
    "Centre-Val de Loire": 24,
    "Corsica": 94,
    "Grand Est": 44,
    "Hauts-de-France": 32,
    "Île-de-France": 11,
    "Normandy": 28,
    "Nouvelle-Aquitaine": 75,
    "Occitanie": 76,
    "Pays de la Loire": 52,
    "Provence-Alpes-Côte d'Azur": 93,
    "Guadeloupe": 1,
    "Martinique": 2,
    "Guyane": 3,
    "Réunion": 4,
    "Mayotte": 6,
    "French Guiana": 7,
    "Monaco": 8,
    "Nouvelle-Calédonie": 9,
    "Polynésie": 10,
    "Saint-Pierre-et-Miquelon": 12,
    "Saint-Barthélemy" : 13,
    "Wallis-et-Futuna": 14
}


def node_grouped_in_regions(nodes):
    """
        Generates the CSV file containing all nodes info.\n

        :params: None\n
        :return: None
    """

    output = ["Id;Label"]
    #1 node = 1 author
    # wirte id and label columns
    for x in tqdm(range(len(nodes))):
        print(nodes[x])
        print(len(nodes[x]))
        if len(nodes[x]) > 2 and nodes[x][2] != '97':
            i = region_to_number[french_departments[nodes[x][2]]] #id
            n = french_departments[nodes[x][2]] #label
            output.append(str(i)+';'+str(n))
        # write i in id and n in name

    with open('../data/graphs/nodes/nodes_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))


def edges_grouped_in_region(edges):
    """
        Generates the CSV file containing all edges info.\n

        :params: None\n
        :return: None
    """    

    output = ["Source;Target;Type;Id;Weight"]
    #1 node = 1 author
    # wirte id and label columns
    index = 0
    for x in tqdm(range(len(edges))):
        a1 = region_to_number[french_departments[edges[x][1]]]
        a2 = region_to_number[french_departments[edges[x][0]]]
        output.append(str(a1)+";"+str(a2)+";"+"Directed;"+str(index)+';'+str(edges[x][2]))
        index += 1

    with open('../data/graphs/edges/paca-edges_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))


def nodes_csv(nodes):
    """
        Generates the CSV file containing all nodes info.\n

        :params: None\n
        :return: None
    """

    output = ["Id;Label"]
    #1 node = 1 author
    # wirte id and label columns
    for x in tqdm(range(len(nodes))):
        i = nodes[x][0]
        n = nodes[x][1]
        output.append(str(i)+';'+str(n))
        # write i in id and n in name

    with open('../data/graphs/nodes/nodes_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))


def edges_csv(edges):
    """
        Generates the CSV file containing all edges info.\n

        :params: None\n
        :return: None
    """    

    output = ["Source;Target;Type;Id;Weight"]
    #1 node = 1 author
    # wirte id and label columns
    index = 0
    for x in tqdm(range(len(edges))):
        i = edges[x][0]
        a1 = edges[x][1]
        a2 = edges[x][2]
        output.append(str(a1)+";"+str(a2)+";"+"Directed;"+str(index)+';'+str(1))
        index += 1

    with open('../data/graphs/edges/paca-edges_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))


def main():
    global uid
    global c


    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dept', type=int, default=0, metavar='N',
                        help='numéro de département à observé, 0 sinon')
    parser.add_argument('--region', type=str, default="PACA", metavar='N',
                        help='PACA ou Île de France')
    parser.add_argument('--country', action='store_true', default=False,
                        help="crée un graphique sur l'ensemble des échanges de lots dans le pays")

    args = parser.parse_args()

    uid = uuid.uuid4()


    conn = sqlite3.connect('./../Foppa1.1.2.db')
    c = conn.cursor()


    print("#########################################################################################")
    print("#                                                                                       #")
    print("#    Welcom to our Collaboration Network generator !                                    #")
    print("#    Type !help to get more informations.                                               #")
    print("#    Type !start to directly generate the network with default values.                  #")
    print("#                                                                                       #")
    print("#                                                                                       #")
    print("#########################################################################################")


    #ADD PARAMETERS TO ENTER THE DEPARTMENT ID 
    #ADD PARAMETER TO SPECIFY THE TYPE OF GRAPH WE WANT TO GENERATE

    if (args.dept != 0):
        nodes_qry = "SELECT agentId, name FROM Agents WHERE department = " + str(args.dept)
        edges_qry = "SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2 FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId WHERE a1 in (SELECT agentId FROM Agents WHERE department = '"+ str(args.dept) +"') and a2 in (SELECT agentId FROM Agents WHERE department = '"+ str(args.dept) +"')" 
   
        print(nodes_qry)
        nodes = c.execute(nodes_qry).fetchall()

        print(edges_qry) 
        edges = c.execute(edges_qry).fetchall()

        nodes_csv(nodes)
        edges_csv(edges)

    elif (args.country):
        print("COUNTRY")
        nodes_qry = "SELECT agentId, name, department FROM Agents WHERE department != 'NULL'"
        edges_qry = "SELECT a.department as buyerDept, b.department as supplierDept , count(*) FROM (SELECT Agents.agentId,  Agents.department,  LotBuyers.lotId FROM Agents JOIN LotBuyers ON Agents.agentId = LotBuyers.agentId) as a JOIN  (SELECT Agents.agentId, Agents.department, LotSuppliers.lotId FROM Agents JOIN LotSuppliers ON Agents.agentId = LotSuppliers.agentId) as b ON a.lotId = b.lotId WHERE NOT (a.department == 'NULL' OR b.department == 'NULL' OR a.department == '97' OR b.department == '97') GROUP BY a.department, b.department"
        
        print(nodes_qry)
        nodes = c.execute(nodes_qry).fetchall()

        print(edges_qry) 
        edges = c.execute(edges_qry).fetchall()

        node_grouped_in_regions(nodes)
        edges_grouped_in_region(edges)

   
    elif (args.region == "PACA"):
        nodes_qry = "SELECT agentId, name FROM Agents WHERE department = '84' or department = '83' or department = '13' or department = '06' or department = '05' or department = '04'"
        edges_qry = "SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2 FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId WHERE a1 in (SELECT agentId FROM Agents WHERE department = '84' or department = '83' or department = '13' or department = '06' or department = '05' or department = '04') and a2 in (SELECT agentId FROM Agents WHERE department = '84' or department = '83' or department = '13' or department = '06' or department = '05' or department = '04')" 
    else: #region avec Ile de France
        nodes_qry = "SELECT agentId, name FROM Agents WHERE department = '75' or department = '77' or department = '78' or department = '91' or department = '92' or department = '93' or department = '94' or department = '95'"
        edges_qry = "SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2 FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId WHERE a1 in (SELECT agentId FROM Agents WHERE department = '75' or department = '77' or department = '78' or department = '91' or department = '92' or department = '93' or department = '94' or department = '95') and a2 in (SELECT agentId FROM Agents WHERE department = '75' or department = '77' or department = '78' or department = '91' or department = '92' or department = '93' or department = '94' or department = '95')" 

    #"SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2 FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId LIMIT 5000"
    
    #nodes_csv(nodes)
    #edges_csv(edges)

    print("done")




main()