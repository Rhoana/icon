import os
import zlib
import StringIO
import base64
import numpy as np;
import json


path = 'database.json'
with open(path) as dbfile:
     database = json.loads(dbfile.read())

index = 0
for label in database:
    print label['name'], label['index']
    label['index'] = index
    index += 1

# save the database
with open(path, 'w') as outfile:
     json.dump(database, outfile)
