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
from db import DB
from project import Project
from performance import Performance
from paths import Paths

from mlp import MLP


print 'base_path:', base_path

if __name__ == '__main__':

    # load the model to use for performance evaluation
    x = T.matrix('x')

    rng = numpy.random.RandomState(1234)

    #project = DB.getProject('evalmlp')
    project = DB.getProject('mlpnew')

    model = MLP(
            id=project.id,
            rng=rng,
            input=x,
            offline=True,
            n_in=project.patchSize**2,
            n_hidden=project.hiddenUnits,
            n_out=len(project.labels),
            train_time=project.trainTime,
            batch_size=project.batchSize,
            patch_size=project.patchSize,
            path=project.path_offline)


    nTests = 1
    print 'measuring offline performance...'
    #Performance.measureOffline(model, project.id, mean=project.mean, std=project.std,maxNumTests=nTests)


    x = T.matrix('x')
    model = MLP(
            id=project.id,
            rng=rng,
            input=x,
            n_in=project.patchSize**2,
            n_hidden=project.hiddenUnits,
            n_out=len(project.labels),
            train_time=project.trainTime,
            batch_size=project.batchSize,
            patch_size=project.patchSize,
            path=project.path)


    print 'measuring online performance...'
    Performance.measureOnline(model, project.id, mean=project.mean, std=project.std,maxNumTests=nTests)
    #Performance.measureBaseline(model, project.id,maxNumTests=nTests)
    #Performance.measureGroundTruth(model, project.id,maxNumTests=nTests)
    

