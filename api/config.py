import json


def load(file='cfg.json'):
    with open(file, 'r') as filereader:
        return json.loads(filereader.read())


config = load()