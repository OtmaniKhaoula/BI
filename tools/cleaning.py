import numpy as np
import pandas as pd
import re
from geopy.geocoders import Photon
# Read CSV file
patterns = {
    "address": r"(\d{1,5})\s(.*?)\s(\d{5})\s(.*?)(,| |;)\s(.*?)",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "zipcode": r"\d{2}( )?\d{3}",
    "phone": r"(\d{1,3})",
    "price": "\b\d+(?:(\.|\,)\d{1,4})?\s*(€|EURO|EUR)\b"
}

bad_words = ['15 (CRITERE', '15) NOTE:']
def clean_names():
    csv_file = '../data/Agents.csv'
    df = pd.read_csv(csv_file, index_col=False)
    for index, row in df.iterrows():
        row['name'] = row['name'].strip()
        row['name'] = row['name'].replace("CNRS", "CENTRE NATIONAL DE LA RECHERCHE SCIENTIFIQUE")
        row['name'] = row['name'].replace("MAIRIE D ", "MAIRIE DE ")
        row['name'] = row['name'].replace("OPERATEURS", "")
        row['name'] = row['name'].replace("TITULAIRES", "")
        row['name'] = row['name'].replace("SOCIETES", "")
        row['name'] = row['name'].replace("ATTRIBUTAIRES", "")
        row['name'] = row['name'].replace("CO-TITULAIRES", "")

        re.sub(r'[^\w\s]+$', '', row['name'])
        re.sub(patterns['price'], '', row['name'])
        if re.search(patterns['email'], row['name']):
            re.sub(re.search(patterns['email'], row['name'])[0], '', row['name'])
        if re.search(patterns['zipcode'], row['name']):
            row['zipcode'] = int(re.search(patterns['zipcode'], row['name'])[0].replace(' ', ''))
            re.sub(re.search(patterns['zipcode'], row['name'])[0], '', row['name'])
        if re.search(patterns['address'], row['name']):
            row['address'] ="" + str(row['address']) + str(re.search(patterns['address'], row['name'])[0])
            re.sub(re.search(patterns['zipcode'], row['name'])[0], '', row['name'])
        if re.search(r'\d+', re.sub(r'[^\w\s]+$', '', row['name'].strip())) or any(word in row['name'] for word in bad_words):
            df = df.drop(index)
    return df

# Get longitude and latitude from address using geopy
def get_lat_long(address):
    geolocator = Photon(user_agent="geoapiExercises")
    location = geolocator.geocode(address, timeout=None)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

def complete_address_info():
    csv_file = '../data/Agents.csv'
    df = pd.read_csv(csv_file, index_col=False)
    i = 0
    # Iterate over each row
    for index, row in df.iterrows():
        address = str(row['address']).strip() + " " + str(int(row['zipcode'])) if not pd.isnull(row['zipcode']) else str(row['address']).strip()
        latitude = row['latitude']
        longitude = row['longitude']

        # Check if latitude and longitude are missing and address is not empty
        if pd.isnull(latitude) or pd.isnull(longitude):
            i+=1
            if address:
                # Check if address pattern exists
                if ' ' in address:  # Replace 'address_pattern' with your address pattern
                    # Get latitude and longitude from address
                    lat, long = get_lat_long(address)
                    if lat is not None and long is not None:
                        df.at[index, 'latitude'] = lat
                        df.at[index, 'longitude'] = long
    print("There was {} addresses with missing long/lat values".format(i))
    # Rewrite the whole table in the same CSV file
    df.to_csv(csv_file, index=False)
    return df



def clean_contractor_sme():
    bad_lot = [r'\d\W\d', 'UNIQUE', 'SEUL', r'\d(.*)ET(.*)\d', r'\W', r'\d{7,}']
    csv_file = "../data/Lots.csv"
    df = pd.read_csv(csv_file, index_col=False)
    for index, row in df.iterrows():
        if (index%100000==0):
            print(index)
        if 'Y' in str(row['contractorSme']).upper().strip():
            df.at[index, 'contractorSme'] = 'Y'
        elif 'N' in str(row['contractorSme']).upper().strip():
            df.at[index, 'contractorSme'] = 'N'
        else:
            df.at[index, 'contractorSme'] = np.NAN
        if any(re.search(lot, str(row['lotsNumber']).strip().upper()) for lot in bad_lot):
            if re.search('\d\W\d\W', str(row['lotsNumber']).strip().upper()):
                df.at[index, 'lotsNumber'] = len(str(row['lotsNumber']).split(' '))
            elif re.search('UNIQUE', str(row['lotsNumber']).upper()) or re.search('SEUL', str(row['lotsNumber']).upper()):
                df.at[index, 'lotsNumber'] = 1
            elif re.search('\W', str(row['lotsNumber']).upper()) or re.search('SANS SUITE', str(row['lotsNumber']).strip().upper()) or re.search('', str(row['lotsNumber']).strip().upper()):
                df.drop(index)
    df.to_csv(csv_file, index=False)
    return df

df_new = clean_contractor_sme()
