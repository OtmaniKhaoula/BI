import pandas as pd
import pandasql as ps
from ydata_profiling import ProfileReport
import sqlite3


sirene = pd.read_csv('etablissementData.txt')
sirene['siret'] = sirene['siret'].astype(str)


conn = sqlite3.connect('./../Foppa1.1.2.db')
c = conn.cursor()


columns_qry = c.execute('SELECT name FROM PRAGMA_TABLE_INFO("Agents")').fetchall()
columns = []
for col in columns_qry:
    columns.append(col[0])

def gen_table():
    qry_agents = "SELECT Agents.name, Agents.agentId, a.lotId, siret FROM (Agents JOIN LotSuppliers ON Agents.agentId = LotSuppliers.agentId) as a JOIN LotBuyers ON LotBuyers.agentId = Agents.agentId WHERE siret != 'NULL' AND Agents.agentId != 'NULL' AND LotSuppliers.agentId != 'NULL' AND LotBuyers.agentId != 'NULL'"

    qry_res = c.execute(qry_agents)


    agents = pd.DataFrame(qry_res.fetchall())
    agents.columns = ['name', 'agentId', 'lotId', 'siret']#columns

        

    data = pd.merge(sirene, agents, on='siret')
    data['dateCreationEtablissement'] = pd.DatetimeIndex(data['dateCreationEtablissement']).year

    qry_lots = "SELECT lotId, contractDuration, typeOfContract, accelerated, cpv, awardPrice, awardEstimatedPrice, cancelled FROM lots WHERE lotId != 'NULL'"
    qry_res = c.execute(qry_lots)


    lots = pd.DataFrame(qry_res.fetchall())

    lots.columns = ['lotId', 'contractDuration', 'typeOfContract', 'accelerated', 'cpv', 'awardPrice', 'awardEstimatedPrice', 'cancelled'] #columns

    data = pd.merge(data, lots, on='lotId')
    data = data.drop_duplicates()

    return data

def general_profiling(data):
    #UNCOMMENT FOR EXPLORATIVE ANALYSIS
    data = data.drop(columns=['siret', 'lotId', 'siren', 'agentId', 'name']) #'longitude', 'latitude', 'addresse', 'city', 'zipcode', 'country', 'departement'])

    report=ProfileReport(data, title="Exploration", 
        
        correlations={
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
        explorative=True)
    report.to_file('lots-siege-res.html')

def get_public_part_for_var(var):
    qry_agents = "SELECT name, agentId, siret FROM Agents WHERE siret != 'NULL' AND agentId != 'NULL'"
    qry_res = c.execute(qry_agents)
    agents = pd.DataFrame(qry_res.fetchall())
    agents.columns = ['name', 'agentId', 'siret']#columns
    data = pd.merge(sirene, agents, on='siret')

    count = f"""SELECT {var}, count(name) as nbPublic FROM data WHERE LOWER(name) LIKE '%mairie%' OR LOWER(name) LIKE '%hospital%' OR LOWER(name) LIKE '%commune%'  OR LOWER(name) LIKE '%departement%'  OR LOWER(name) LIKE '%université%' OR LOWER(name) LIKE '%région%' OR LOWER(name) LIKE '%caisse%' OR LOWER(name) LIKE '%gendarmerie%' OR LOWER(name) LIKE '%public%' OR LOWER(name) LIKE '%collecti%' OR LOWER(name) LIKE '%tribunal%' OR LOWER(name) LIKE '%centre%' OR LOWER(name) LIKE '%greffe%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%police%' OR LOWER(name) LIKE '%emploi%' OR LOWER(name) LIKE '%greta%' OR LOWER(name) LIKE '%impôt%' OR LOWER(name) LIKE '%national%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%armée%' GROUP BY {var}"""
    
    res = ps.sqldf(count, locals())
    print(res)

def stats_public_prive(data):
    count_prive = f"""SELECT count(lotId) as nbLotsPrive FROM data WHERE NOT (LOWER(name) LIKE '%mairie%' OR LOWER(name) LIKE '%hospital%' OR LOWER(name) LIKE '%commune%'  OR LOWER(name) LIKE '%departement%'  OR LOWER(name) LIKE '%université%' OR LOWER(name) LIKE '%région%' OR LOWER(name) LIKE '%caisse%' OR LOWER(name) LIKE '%gendarmerie%' OR LOWER(name) LIKE '%public%' OR LOWER(name) LIKE '%collecti%' OR LOWER(name) LIKE '%tribunal%' OR LOWER(name) LIKE '%centre%' OR LOWER(name) LIKE '%greffe%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%police%' OR LOWER(name) LIKE '%emploi%' OR LOWER(name) LIKE '%greta%' OR LOWER(name) LIKE '%impôt%' OR LOWER(name) LIKE '%national%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%armée%')"""
    count_public = f"""SELECT count(lotId) as nbLotsPublic FROM data WHERE LOWER(name) LIKE '%mairie%' OR LOWER(name) LIKE '%hospital%' OR LOWER(name) LIKE '%commune%'  OR LOWER(name) LIKE '%departement%'  OR LOWER(name) LIKE '%université%' OR LOWER(name) LIKE '%région%' OR LOWER(name) LIKE '%caisse%' OR LOWER(name) LIKE '%gendarmerie%' OR LOWER(name) LIKE '%public%' OR LOWER(name) LIKE '%collecti%' OR LOWER(name) LIKE '%tribunal%' OR LOWER(name) LIKE '%centre%' OR LOWER(name) LIKE '%greffe%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%police%' OR LOWER(name) LIKE '%emploi%' OR LOWER(name) LIKE '%greta%' OR LOWER(name) LIKE '%impôt%' OR LOWER(name) LIKE '%national%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%armée%'"""
    count_total = f"""SELECT count(lotId) as nbLotsTotal FROM data"""

    res = ps.sqldf(count_prive, locals())
    print(res)

    res = ps.sqldf(count_public, locals())
    print(res)
    
    res = ps.sqldf(count_total, locals())
    print(res)

def get_avg_lots_for_var(data, var):
    
    #SANS LES ETABLISSEMENTS PUBLIQUES

    #count = f"""SELECT {var}, AVG(nbLots) FROM (SELECT etablissementSiege, agentId, count(lotId) as nbLots FROM data WHERE NOT (LOWER(name) LIKE '%mairie%' OR LOWER(name) LIKE '%hospital%' OR LOWER(name) LIKE '%commune%'  OR LOWER(name) LIKE '%departement%'  OR LOWER(name) LIKE '%université%' OR LOWER(name) LIKE '%région%' OR LOWER(name) LIKE '%caisse%' OR LOWER(name) LIKE '%gendarmerie%' OR LOWER(name) LIKE '%public%' OR LOWER(name) LIKE '%collecti%' OR LOWER(name) LIKE '%tribunal%' OR LOWER(name) LIKE '%centre%' OR LOWER(name) LIKE '%greffe%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%police%' OR LOWER(name) LIKE '%emploi%' OR LOWER(name) LIKE '%greta%' OR LOWER(name) LIKE '%impôt%' OR LOWER(name) LIKE '%national%' OR LOWER(name) LIKE '%chambre%' OR LOWER(name) LIKE '%armée%') GROUP BY agentId) GROUP BY {var}"""

    count = f"""SELECT {var}, AVG(nbLots) FROM (SELECT {var}, agentId, count(lotId) as nbLots FROM data GROUP BY agentId) GROUP BY {var}"""
    res = ps.sqldf(count, locals())
    print(res)

def get_avg_price_for_var(data, var):

    count = f"""SELECT {var}, AVG(price) FROM (SELECT {var}, agentId, avg(awardPrice) as price FROM data GROUP BY agentId) GROUP BY {var}"""
    res = ps.sqldf(count, locals())
    print(res)

data = gen_table()
#var = 'etablissementSiege'
var = 'etatAdministratifEtablissement'

#general_profiling(data)

get_public_part_for_var(var)
get_avg_lots_for_var(data, var)
get_avg_price_for_var(data, var)