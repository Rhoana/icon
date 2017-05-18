import os
import sys
import theano
import theano.tensor as T
import numpy
import numpy as np
import mahotas
import partition_comparison
import StringIO
import glob

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../../common'))
sys.path.insert(2,os.path.join(base_path, '../../database'))
sys.path.insert(3,os.path.join(base_path, '../'))

from db import DB
from project import Project
from performance import Performance
from paths import Paths

from mlp import MLP
from data import Data

print 'base_path:', base_path

if __name__ == '__main__':

    # load the model to use for performance evaluation
    x = T.matrix('x')

    rng = numpy.random.RandomState(1234)

    project = DB.getProject('mlpnew') #evalmlp')

    model = MLP(
            id=project.id,
            rng=rng,
            input=x,
            momentum=0.0,
            offline=True,
            n_in=project.patchSize**2,
            n_hidden=project.hiddenUnits,
            n_out=len(project.labels),
            train_time=project.trainTime,
            #batch_size=project.batchSize,
            batch_size=50,
            patch_size=project.patchSize,
            path=project.path_offline)


    data = Data( project, offline=True, n_train_samples=700000, n_valid_samples=5000)
    #model.train(offline=True, data=data, mean=project.mean, std=project.std)
    #data.load(project )

    #print data.get_pixel_count(project)
    #exit(1)

    n_iterations = 5000
    for iteration in xrange(n_iterations):
        print 'iteration:', iteration
        model.train(data=data, offline=True, mean=project.mean, std=project.std)
