import sqlite3
from tqdm import tqdm
import uuid
import re
import unidecode

""" Script to generate the csv files for giphy.\n
    Type: "python .\CSV_gen.py" to launch the script.\n
    /!\ Once the program started type !h for more info.\n
    Output: under Edges/ Nodes/ and Infos/\n\n"""

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

    with open('../data/graphs/edges/edges_'+str(uid)+'.csv', 'w') as f:
        for i in range(len(output)):
            f.write("%s\n" % unidecode.unidecode(output[i]))


def main():
    global uid
    global c

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


    

    qry = "SELECT agentId, name FROM Agents WHERE department = 84"
    print(qry)
    nodes = c.execute(qry).fetchall()

    qry = "SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2 FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId WHERE a1 in (SELECT agentId FROM Agents WHERE department = 84) and a2 in (SELECT agentId FROM Agents WHERE department = 84)" 
    #"SELECT LotBuyers.lotId, LotBuyers.agentId as a1, LotSuppliers.agentId  as a2 FROM LotBuyers JOIN LotSuppliers ON LotBuyers.lotId = LotSuppliers.lotId LIMIT 5000"
    print(qry) 
    edges = c.execute(qry).fetchall()

    nodes_csv(nodes)
    edges_csv(edges)

    print("done")

main()