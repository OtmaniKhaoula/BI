import sqlite3
import argparse
from tqdm import tqdm
import uuid
import re
import unidecode
import pandas as pd
import pandasql as ps

french_departments = {
    "01": "Ain",
    "02": "Aisne",
    "03": "Allier",
    "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes",
    "07": "Ardèche",
    "08": "Ardennes",
    "09": "Ariège",
    "10": "Aube",
    "11": "Aude",
    "12": "Aveyron",
    "13": "Bouches-du-Rhône",
    "14": "Calvados",
    "15": "Cantal",
    "16": "Charente",
    "17": "Charente-Maritime",
    "18": "Cher",
    "19": "Corrèze",
    "20": "Corse",
    "2A": "Corse",
    "2B": "Corse",
    "21": "Côte-d'Or",
    "22": "Côtes-d'Armor",
    "23": "Creuse",
    "24": "Dordogne",
    "25": "Doubs",
    "26": "Drôme",
    "27": "Eure",
    "28": "Eure-et-Loir",
    "29": "Finistère",
    "30": "Gard",
    "31": "Haute-Garonne",
    "32": "Gers",
    "33": "Gironde",
    "34": "Hérault",
    "35": "Ille-et-Vilaine",
    "36": "Indre",
    "37": "Indre-et-Loire",
    "38": "Isère",
    "39": "Jura",
    "40": "Landes",
    "41": "Loir-et-Cher",
    "42": "Loire",
    "43": "Haute-Loire",
    "44": "Loire-Atlantique",
    "45": "Loiret",
    "46": "Lot",
    "47": "Lot-et-Garonne",
    "48": "Lozère",
    "49": "Maine-et-Loire",
    "50": "Manche",
    "51": "Marne",
    "52": "Haute-Marne",
    "53": "Mayenne",
    "54": "Meurthe-et-Moselle",
    "55": "Meuse",
    "56": "Morbihan",
    "57": "Moselle",
    "58": "Nièvre",
    "59": "Nord",
    "60": "Oise",
    "61": "Orne",
    "62": "Pas-de-Calais",
    "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin",
    "68": "Haut-Rhin",
    "69": "Rhône",
    "70": "Haute-Saône",
    "71": "Saône-et-Loire",
    "72": "Sarthe",
    "73": "Savoie",
    "74": "Haute-Savoie",
    "75": "Paris",
    "76": "Seine-Maritime",
    "77": "Seine-et-Marne",
    "78": "Yvelines",
    "79": "Deux-Sèvres",
    "80": "Somme",
    "81": "Tarn",
    "82": "Tarn-et-Garonne",
    "83": "Var",
    "84": "Vaucluse",
    "85": "Vendée",
    "86": "Vienne",
    "87": "Haute-Vienne",
    "88": "Vosges",
    "89": "Yonne",
    "90": "Territoire de Belfort",
    "91": "Essonne",
    "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne",
    "95": "Val-d'Oise",
    "971": "Guadeloupe",
    "972": "Martinique",
    "973": "Guyane",
    "974": "La Réunion",
    "976": "Mayotte",
    "975": 'Saint-Pierre-et-Miquelon',
    "988": 'Nouvelle-Calédonie',
    "987": 'Polynésie',
    "977": 'Saint-Barthélemy',
    "986": "Wallis-et-Futuna"
}

dept_to_regions = {
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
    "Guyane": 7,
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

year_to_node_label = {
    year: (
        'Avant 1985' if year < 1985 else
        '1985-1990' if 1985 <= year < 1990 else
        '1990-1995' if 1990 <= year < 1995 else
        '1995-2000' if 1995 <= year < 2000 else
        '2000-2005' if 2000 <= year < 2005 else
        '2005-2010' if 2005 <= year < 2010 else
        '2010-2015' if 2010 <= year < 2015 else
        '2015-2020' if 2015 <= year < 2020 else
        '2020-2025' if 2020 <= year < 2025 else None
    )
    for year in range(1900, 2025)
}

def edges_grouped_by_year(node_label_to_index, edges):
    """
        Generates the CSV file containing all edges info.\n

        :params: None\n
        :return: None
    """    
    print("EDGES GROUPED BY YEAR")

    output = ["Source;Target;Type;Id;Weight"]
    #1 node = 1 author
    # wirte id and label columns
    map_count_weight = {}

    index = 0
    for index, row in edges.iterrows():
        buyerYear = year_to_node_label[row['buyerYear']]
        supplierYear = year_to_node_label[row['supplierYear']]
        value = row['count(*)']

        node_str = f"{node_label_to_index[buyerYear]};{node_label_to_index[supplierYear]}"
        if node_str not in map_count_weight:
            map_count_weight[node_str] = int(value)
        else:
            map_count_weight[node_str] += int(value)


    for key in map_count_weight:
        output.append(key+";"+"Directed;"+str(index)+';'+str(map_count_weight[key]))

        index += 1

    with open('../data/graphs/edges/date-edges_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))


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
            i = region_to_number[dept_to_regions[nodes[x][2]]] #id
            n = dept_to_regions[nodes[x][2]] #label
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
    map_count_weight = {}

    index = 0
    for x in tqdm(range(len(edges))):
        #print(edges[x])
        
        nodes_str = str(region_to_number[dept_to_regions[edges[x][1]]]) + ";" + str(region_to_number[dept_to_regions[edges[x][0]]])
        if nodes_str not in map_count_weight:
            map_count_weight[nodes_str] = int(edges[x][2])
        else:
            map_count_weight[nodes_str] += int(edges[x][2])

    for key in map_count_weight:
        output.append(key+";"+"Directed;"+str(index)+';'+str(map_count_weight[key]))

        index += 1

    with open('../data/graphs/edges/paca-edges_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))


def dept_nodes_csv(nodes):
    """
        Generates the CSV file containing all nodes info.\n

        :params: None\n
        :return: None
    """

    output = ["Id;Label"]
    #1 node = 1 author
    # wirte id and label columns

    for x in tqdm(range(len(nodes))):
        n = french_departments[nodes[x][0]] #name
        i = nodes[x][0] #dept id
        output.append(str(i)+';'+str(n))
        # write i in id and n in name

    with open('../data/graphs/nodes/971-nodes_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))

def dept_edges_csv(edges):
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
        a1 = edges[x][0] #Buyer dept
        a2 = edges[x][1] #Supplier dept
        count = edges[x][2] #count
        output.append(str(a1)+";"+str(a2)+";"+"Directed;"+str(index)+';'+str(count))
        index += 1

    with open('../data/graphs/edges/dept-edges_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))

def nodes_csv(nodes):
    """
        Generates the CSV file containing all nodes info.\n

        :params: None\n
        :return: None
    """
    node_label_to_index = {}

    output = ["Id;Label"]
    #1 node = 1 author
    # wirte id and label columns

    for x in tqdm(range(len(nodes))):
        i = nodes[x][0]
        n = nodes[x][1]
        output.append(str(i)+';'+str(n))
        # write i in id and n in name
        node_label_to_index[n] = i

    with open('../data/graphs/nodes/date-nodes_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))

    return node_label_to_index

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
        count = edges[x][3]
        output.append(str(a1)+";"+str(a2)+";"+"Directed;"+str(index)+';'+str(count))
        index += 1

    with open('../data/graphs/edges/dept-edges_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))

def create_dataframe():
    sirene = pd.read_csv('etablissementData.txt')
    sirene['siret'] = sirene['siret'].astype(str)


    qry_agents = "SELECT name, siret, agentId FROM Agents WHERE siret != 'NULL' AND agentId != 'NULL'"
    qry_res = c.execute(qry_agents)

    agents = pd.DataFrame(qry_res.fetchall())
    agents.columns = ['name', 'siret', 'agentId']


    agents = pd.merge(sirene, agents, on='siret')
    agent['year'] = pd.DatetimeIndex(agent['dateCreationEtablissement']).year

    qry_buyers = "SELECT lotId, agentId FROM LotBuyers WHERE agentId != 'NULL'"
    qry_res = c.execute(qry_buyers)

    buyers = pd.DataFrame(qry_res.fetchall())
    buyers.columns = ['lotBought', 'agentId']

    data = pd.merge(agents, buyers, on='agentId')

    
    qry_suppliers = "SELECT lotId, agentId FROM LotSuppliers WHERE agentId != 'NULL'"
    qry_res = c.execute(qry_suppliers)

    suppliers = pd.DataFrame(qry_res.fetchall())
    suppliers.columns = ['lotSupplied', 'agentId']

    data = pd.merge(data, suppliers, on='agentId')

    return data, agents, buyers, suppliers

def main():
    global uid
    global c


    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dept', type=int, default=0, metavar='N',
                        help='numéro de département à observer, 0 sinon')
    parser.add_argument('--countryDept', action='store_true', default=False,
                        help="crée un graphique sur l'ensemble des échanges de lots dans le pays par département")
    parser.add_argument('--country', action='store_true', default=False,
                        help="crée un graphique sur l'ensemble des échanges de lots dans le pays par region")
    parser.add_argument('--date', action='store_true', default=False,
                        help="crée un graphique sur l'ensemble des échanges de lots dans le pays par region")

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
        edges_qry = "SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2, count(*) FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId WHERE a1 in (SELECT agentId FROM Agents WHERE department = '"+ str(args.dept) +"') and a2 in (SELECT agentId FROM Agents WHERE department = '"+ str(args.dept) +"') GROUP BY a1, a2" 
   
        print(nodes_qry)
        nodes = c.execute(nodes_qry).fetchall()

        print(edges_qry) 
        edges = c.execute(edges_qry).fetchall()
        print(edges)

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
   
    elif (args.countryDept):
        nodes_qry = "SELECT DISTINCT department FROM Agents WHERE department != 'NULL' AND department < 95"
        edges_qry = "SELECT a.department as buyerDept, b.department as supplierDept , count(*) FROM (SELECT Agents.agentId,  Agents.department,  LotBuyers.lotId FROM Agents JOIN LotBuyers ON Agents.agentId = LotBuyers.agentId) as a JOIN (SELECT Agents.agentId, Agents.department, LotSuppliers.lotId FROM Agents JOIN LotSuppliers ON Agents.agentId = LotSuppliers.agentId) as b ON a.lotId = b.lotId WHERE NOT (a.department == 'NULL' OR b.department == 'NULL' OR a.department > 95 OR b.department > 95 OR a.department == '98' OR b.department == '98') GROUP BY a.department, b.department"

        print(nodes_qry)
        nodes = c.execute(nodes_qry).fetchall()

        print(edges_qry) 
        edges = c.execute(edges_qry).fetchall()

        dept_nodes_csv(nodes)
        dept_edges_csv(edges)

    elif (args.date):
        data, agents, suppliers, buyers = create_dataframe()

        ###### SCRIPT TO GENERATE THE NODES AND EDGES OF OUR GRAPHS
        i = 0
        nodes = [[i, 'Avant 1985']]
        i+=1
        for year in range(1985, 2020, 5):
            nodes.append([i, f'{year}-{year+5}'])
            i += 1

        label_to_index = nodes_csv(nodes)
        
        edges = """SELECT a.year as buyerYear, b.year as supplierYear, count(*) FROM (SELECT lotSupplied, agentId, year FROM data) as a JOIN (SELECT lotBought, agentId, year FROM data) as b ON a.lotSupplied = b.lotBought GROUP BY a.year, b.year"""
        edges_grouped_by_year(label_to_index, ps.sqldf(edges, locals()))
        
    #"SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2 FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId LIMIT 5000"
    
    #nodes_csv(nodes)
    #edges_csv(edges)

    print("done")




main()