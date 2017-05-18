import os
import sqlite3 as lite
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(2,os.path.join(base_path, '../common'))
DATABASE_NAME = os.path.join(base_path, '../../data/database/icon.db')


#---------------------------------------------------------------------------
# Label datum
#---------------------------------------------------------------------------
class Label (object):

        def __init__(	self, 
			index,     # unique index of the label
			name,      # human readable name of the label
			r,         # red component of color
			g,         # green component of color
			b          # blue component of color
			):
                self.index     = index
		self.name      = name
		self.r         = r
		self.g         = g
		self.b         = b

	def toJson(self):
		data = {}
		data['index']      = self.index
		data['name']       = self.name
		data['r']          = self.r
		data['g']          = self.g
		data['b']          = self.b
		return data
