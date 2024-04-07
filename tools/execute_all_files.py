# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:20:34 2024

@author: Utilisateur
"""

import subprocess

# Liste des chemins vers les fichiers Python à exécuter
chemins_vers_scripts = [
    "exploration_v2.py",
    "cleaning.py",
    "cleaning_lots.py",
    "exploration.py",
    "distance.py",
    "cpv.py",
    "clustering.py",
    "regression.py",
    "pme.py",
    "suppliers.py",
    "pme_after_extract.py",
    
]

# Boucle pour exécuter chaque script
for chemin in chemins_vers_scripts:
    result = subprocess.run(["python", chemin], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
