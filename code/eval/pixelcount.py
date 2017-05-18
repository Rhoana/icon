import os
import sys
import numpy as np

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
sys.path.insert(2,os.path.join(base_path, '../database'))
sys.path.insert(3,os.path.join(base_path, '../model'))

from db import DB
from project import Project

from data import Data

def compute_pixel_counts(projectId):
    imagepixels = 1024*1024*10
    project = DB.getProject(projectId)
    data = Data( project )
    pixelcounts = data.get_pixel_count( project )
    total = np.sum( pixelcounts )
    print ''
    print 'project:', projectId
    print 'pixel counts:', pixelcounts
    print 'total:', total
    print 'all:', imagepixels
    print 'percent:', (float(total)/imagepixels)*100.0

if __name__ == '__main__':

    compute_pixel_counts('testmlpv2')
    compute_pixel_counts('testcnnv2')
    compute_pixel_counts('testmlp')
    compute_pixel_counts('testcnn')
