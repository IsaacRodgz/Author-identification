import json
from pprint import pprint

with open('data/dict.json') as data_file:    
    data = json.load(data_file)

pprint(data["authors"][1])