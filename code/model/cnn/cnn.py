import cPickle
import gzip

import os
import sys
import time

import numpy
import numpy as np

import multiprocessing

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../mlp'))
sys.path.insert(2,os.path.join(base_path, '../../common'))
sys.path.insert(3,os.path.join(base_path, '../../database'))

from db import DB
from logistic_sgd import LogisticRegression
from mlp import MLP
from activation_functions import rectified_linear
from paths import Paths

from generateTrainValTestData import gen_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
from classifyImage import generate_patch_data_rows
from vsk_utils import shared_single_dataset
from fast_segment import *

import getpass
from convlayer import LeNetConvPoolLayer

class CNN(object):
    def __init__(
        self,
        id,
        input, 
        batch_size, 
        patch_size, 
        rng, 
        nkerns, 
        kernel_sizes, 
        hidden_sizes, 
        offline=False,
        path=None,
        train_time=5.0,
        learning_rate=0.1,
        momentum=0.9,
        activation=rectified_linear):

        self.id = id
        self.type = 'CNN'
        self.offline = offline
        self.done = False
        self.path = path    
        self.rng = rng
        self.activation = activation
        self.input = input
        self.nkerns = nkerns
        self.kernelSizes = kernel_sizes
        self.hiddenSizes = hidden_sizes
        self.batchSize = batch_size
        self.patchSize = patch_size 
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.x           =  input

        self.best_validation_loss = numpy.inf
        self.trainTime = train_time
        self.resample = False       
        self.error = np.inf
        self.error_threshold = 0.06
        self.initialize() 

    def get_path(self):
        if self.offline:
            return self.path

        rev  = DB.getRevision( self.id )
        path = '%s/best_%s.%s.%d.pkl'%(Paths.Models, self.id, self.type, rev )
        return path.lower()

    
    def initialize(self):
        input = self.input
        input = self.input.reshape((self.batchSize, 1, self.patchSize, self.patchSize))

        self.layer0_input = input
        self.params = []
        self.convLayers = []

        input_next = input
        numberOfFeatureMaps = 1
        featureMapSize = self.patchSize

        for i in range(len(self.nkerns)):
            layer = LeNetConvPoolLayer(
                self.rng,
                input=input_next,
                image_shape=(self.batchSize, numberOfFeatureMaps, featureMapSize, featureMapSize),
                filter_shape=(self.nkerns[i], numberOfFeatureMaps, self.kernelSizes[i], self.kernelSizes[i]),
                poolsize=(2, 2)
            )
            input_next = layer.output
            numberOfFeatureMaps = self.nkerns[i]
            featureMapSize = np.int16(np.floor((featureMapSize - self.kernelSizes[i]+1) / 2))

            self.params += layer.params
            self.convLayers.append(layer)

        # the 2 is there to preserve the batchSize
        mlp_input = self.convLayers[-1].output.flatten(2)

        self.mlp = MLP(
                    rng=self.rng, 
                    input=mlp_input, 
                    n_in=self.nkerns[-1] * (featureMapSize ** 2), 
                    n_hidden=self.hiddenSizes,
                    n_out=2, 
                    patch_size=self.patchSize,
                    batch_size=self.batchSize,
                    activation=self.activation)
        self.params += self.mlp.params

        self.cost = self.mlp.negative_log_likelihood
        self.errors = self.mlp.errors
        self.p_y_given_x = self.mlp.p_y_given_x
        self.debug_x = self.p_y_given_x


        path = self.get_path()

        if not path is None and os.path.exists(path):
            with open(path, 'r') as file:
                print 'loading cnn model from file...', path
                data = cPickle.load(file)
                saved_convLayers         = data[0]
                saved_hiddenLayers       = data[1]
                saved_logRegressionLayer = data[2]
                saved_nkerns             = data[3]
                saved_kernelSizes        = data[4]
                saved_batchSize         = data[5]
                saved_patchSize          = data[6]
                saved_hiddenSizes        = data[7]

            for s_cl, cl in zip(saved_convLayers, self.convLayers):
                cl.W.set_value(s_cl.W.get_value())
                cl.b.set_value(s_cl.b.get_value())

            for s_hl, hl in zip(saved_hiddenLayers, self.mlp.hiddenLayers):
                hl.W.set_value(np.float32(s_hl.W.eval()))
                hl.b.set_value(s_hl.b.get_value())

            self.mlp.logRegressionLayer.W.set_value(np.float32(saved_logRegressionLayer.W.eval()))
            self.mlp.logRegressionLayer.b.set_value(saved_logRegressionLayer.b.get_value())

    def save_t(self, version):
        revision = DB.getRevision( self.id )
        path = '%s/best_%s.%s.%d.%s.pkl'%(Paths.Models, self.id, self.type, revision, version)
        print 'saving - ', path
        with open(path, 'wb') as file:
            cPickle.dump((self.convLayers,
                self.mlp.hiddenLayers,
                self.mlp.logRegressionLayer,
                self.nkerns,
                self.kernelSizes,
                self.batchSize,
                self.patchSize,
                self.hiddenSizes),
                file)


    def save(self):
        path = self.path
        revision = 0
        if not self.offline:
            revision = DB.getRevision( self.id )
            revision = (revision+1)%10
            path = '%s/best_%s.%s.%d.pkl'%(Paths.Models, self.id, self.type, revision)
            path = path.lower()

        print 'saving...', path
        with open(path, 'wb') as file:
            cPickle.dump((self.convLayers,
                self.mlp.hiddenLayers,
                self.mlp.logRegressionLayer,
                self.nkerns,
                self.kernelSizes,
                self.batchSize,
                self.patchSize,
                self.hiddenSizes),
                file)

        if not self.offline:
            DB.finishSaveModel( self.id, revision )

    def classify(self, image, mean=None, std=None):
        if mean != None:
            image = image - mean
        
        if std != None:
            image = image/np.tile(std, (np.shape(image)[0],1))

        return classify_image( image, self )

    def predict(self, image, mean=None, std=None, threshold=0.5):

        prob = self.classify( image=image, mean=mean, std=std)
        prob = self.threshold( prob, factor=threshold )
        prob = prob.astype(dtype=int)
        prob = prob.flatten()
        return prob

    def threshold(self, prob, factor=0.5):
        prob[ prob >= factor ] = 9
        prob[ prob <  factor ] = 1
        prob[ prob == 9      ] = 0
        return prob

    def train(self, offline=False, data=None, mean=None, std=None):
        if offline:
            self.train_offline(data=data, mean=mean, std=std)
        else:
            self.train_online(data)

    def train_online(self, data):
        print 'train online...'
        def gradient_updates_momentum(cost, params, learning_rate, momentum):
            updates = []
            for param in params:
                param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                updates.append((param, param - learning_rate*param_update))
                updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
            return updates

        # DATA INITIALIZATION
        d       = data.sample()
        train_x = d[0]
        train_y = d[1]
        valid_x = d[2]
        valid_y = d[3]
        reset   = d[4]


        if reset:
            self.best_validation_loss = numpy.inf 

        print 'best_validation:', self.best_validation_loss
        train_samples = len(train_y)
        valid_samples = len(valid_y)

        if self.resample:
            self.lr_shared.set_value( np.float32(self.learning_rate) )
            self.m_shared.set_value( np.float32(self.momentum) )
        else:
            self.resample  = True
            self.y         = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
            self.lr        = T.scalar('learning_rate')
            self.m         = T.scalar('momentum')

            self.lr_shared = theano.shared(np.float32(self.learning_rate))
            self.m_shared  = theano.shared(np.float32(self.momentum))
        
        index          =  T.lscalar()  # index to a [mini]batch
        x              = self.x
        y              = self.y
        lr             = self.lr
        m              = self.m
        lr_shared      = self.lr_shared
        m_shared       = self.m_shared
        patchSize      = self.patchSize
        batchSize      = self.batchSize
        train_set_x, train_set_y = shared_dataset((train_x, train_y), doCastLabels=True)

        if valid_samples > 0:
            valid_set_x, valid_set_y = shared_dataset((valid_x, valid_y), doCastLabels=True)

        # compute number of minibatches for training, validation 
        n_train_batches = train_samples / batchSize
        n_valid_batches = valid_samples / batchSize


        #BUILD THE MODEL
        cost = self.cost(y)

        if valid_samples > 0:
            validate_model = theano.function(
                [index],
                self.errors(y),
                givens={
                    x: valid_set_x[index * batchSize: (index + 1) * batchSize],
                    y: valid_set_y[index * batchSize: (index + 1) * batchSize]
                }
            )

        predict_samples = theano.function(
                inputs=[index],
                outputs=T.neq(self.mlp.y_pred, self.y),
                givens={
                        x: train_set_x[index * batchSize: (index + 1) * batchSize],
                        y: train_set_y[index * batchSize: (index + 1) * batchSize]   
                } 
        )

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = gradient_updates_momentum(cost, self.params, lr, m)

        train_model = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batchSize:(index + 1) * batchSize],
                    y: train_set_y[index * batchSize:(index + 1) * batchSize],
                    lr: lr_shared,
                    m: m_shared})


        # TRAIN THE MODEL
        print '... training'
        best_iter = 0
        validation_frequency = 1

        start_time = time.clock()

        elapsed_time = 0
        iter = 0

        minibatch_avg_costs = []
        minibatch_index = 0

        count1 = 0
        count2 = 0


        while (elapsed_time < self.trainTime)\
            and (minibatch_index<n_train_batches)\
            and (not self.done):

            train_cost = train_model(minibatch_index)
            #print '----->traincost:', type(train_cost), train_cost

            minibatch_avg_costs.append(train_cost)
       
            #print 'minibatch_index:', minibatch_index, 'n_train_batches:',n_train_batches, self.batchSize,
 
            probs = predict_samples(minibatch_index)

            indices = data.i_train[minibatch_index * batchSize:(minibatch_index + 1) * batchSize]
            data.p[ indices ] = probs
            #print 'probs:', probs
        
            iter += 1
            if (iter + 1) % validation_frequency == 0 and n_valid_batches > 0:

                validation_losses = np.array([validate_model(i) for i in xrange(n_valid_batches)])
                this_validation_loss = numpy.sum(validation_losses) * 100.0 / valid_samples

                elapsed_time = time.clock() - start_time

                data.report_stats(
                    self.id,
                    elapsed_time,
                    minibatch_index,
                    this_validation_loss,
                    minibatch_avg_costs[-1].item(0))

                # if we got the best validation score until now
                count1 += len(np.where(probs==0)[0])
                count2 += len(np.where(probs==1)[0])
            
                data.add_validation_loss( this_validation_loss )
    
                if this_validation_loss < self.best_validation_loss:
                    self.best_validation_loss = this_validation_loss
                    best_iter = iter

                    print '===>saving....'
                    self.save()
                    print "New best score!"

            # advance to next mini batch
            minibatch_index += 1

            # update elapsed time
            elapsed_time = time.clock() - start_time

        data.save_stats()
 
        p = data.p[ data.i_train ]
        n_bad = len( np.where( p == 1 )[0] )
        error = float(n_bad)/len(p)
        print '----------'
        print 'accuracy:', data.accuracy
        print 'error:', error
        print 'lerror:', self.error
        print 'probi:', np.bincount( np.int64( p ) )

        if n_valid_batches == 0:
            self.save()

        elapsed_time = time.clock() - start_time
        msg = 'The code ran for'
        status = '%f seconds' % (elapsed_time)
        Utility.report_status( msg, status )

    def reportTrainingStats(self, elapsedTime, batchIndex, valLoss, trainCost, mode=0):
        DB.storeTrainingStats( self.id, valLoss, trainCost, mode=mode)
        msg = '(%0.1f)     %i     %f%%'%\
        (
           elapsedTime,
           batchIndex,
           valLoss
        )
        status = '[%f]'%(trainCost)
        Utility.report_status( msg, status )

    def train_offline(self, data, mean=None, std=None):

        print 'training....'
        train_samples=300000
        val_samples=1000
        test_samples=1000
        batchSize = self.batchSize
        learning_rate  = self.learning_rate
        momentum = self.momentum

        def gradient_updates_momentum(cost, params, learning_rate, momentum):
            updates = []
            for param in params:
                param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                updates.append((param, param - learning_rate*param_update))
                updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
            return updates

        rng = numpy.random.RandomState(23455)

        # training data
        d = data.gen_samples_offline(
            nsamples=train_samples,
            purpose='train',
            patchSize=patchSize,
            mean=mean,
            std=std)

        data_mean = d[2]
        data_std = d[3]

        train_set_x, train_set_y = shared_dataset((d[0],d[1]), doCastLabels=True)

        d = data.gen_samples_offline(
            nsamples=val_samples,
            purpose='validate',
            patchSize=patchSize,
            mean=data_mean,
            std=data_std)
        valid_set_x, valid_set_y = shared_dataset((d[0],d[1]), doCastLabels=True)


        d = data.gen_samples_offline(
            nsamples=test_samples,
            purpose='test',
            patchSize=patchSize,
            mean=data_mean,
            std=data_std)
        test_set_x, test_set_y = shared_dataset((d[0],d[1]), doCastLabels=True)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_samples / batchSize
        n_valid_batches = val_samples / batchSize
        n_test_batches = test_samples / batchSize

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        x = self.x #T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        cost = self.cost(y)

        lr = T.scalar('learning_rate')
        m = T.scalar('momentum')

        learning_rate_shared = theano.shared(np.float32(learning_rate))
        momentum_shared = theano.shared(np.float32(momentum))

        print 'training data....'
        print 'min: ', np.min( train_set_x.eval() )
        print 'max: ', np.max( train_set_x.eval() )
        print 'n_train_batches:',n_train_batches
        print 'n_valid_batches:',n_valid_batches
        print 'n_test_batches:',n_test_batches

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            self.errors(y),
            givens={
                x: test_set_x[index * batchSize: (index + 1) * batchSize],
                y: test_set_y[index * batchSize: (index + 1) * batchSize]
            }
        )

        validate_model = theano.function(
            [index],
            self.errors(y),
            givens={
                x: valid_set_x[index * batchSize: (index + 1) * batchSize],
                y: valid_set_y[index * batchSize: (index + 1) * batchSize]
            }
        )


        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = gradient_updates_momentum(cost, self.params, lr, m)


        train_model = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batchSize:(index + 1) * batchSize],
                    y: train_set_y[index * batchSize:(index + 1) * batchSize],
                    lr: learning_rate_shared,
                    m: momentum_shared})

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        best_validation_loss = numpy.inf
        best_iter = 0
        decrease_epoch = 1
        decrease_patience = 1
        test_score = 0.
        doResample = True

        validation_frequency = 1

        start_time = time.clock()

        epoch = 0
        done_looping = False

        last_avg_validation_loss = 0
        avg_validation_losses = []

        while (epoch < n_epochs) and (not self.done):
            minibatch_avg_costs = []
            epoch = epoch + 1


            if doResample and epoch>1: # and len(avg_validation_losses) > 0:
                epoch=0
                avg = np.mean(avg_validation_losses)
                diff = abs(avg-last_avg_validation_loss)
                last_avg_validation_loss = avg
                avg_validation_losses = []


                d = data.gen_samples_offline(
                    nsamples=train_samples,
                    purpose='train',
                    patchSize=patchSize,
                    mean=mean,
                    std=std)
                dx = d[0]
                dy = d[1]
                train_set_x.set_value(np.float32(dx))
                train_set_y.set_value(np.int32(dy))


            for minibatch_index in xrange(n_train_batches):
                if self.done:
                    break

                train_cost = train_model(minibatch_index)
                minibatch_avg_costs.append( train_cost )
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    #self.save()
                    # compute zero-one loss on validation set
                    validation_losses = np.array([validate_model(i) for i
                                         in xrange(n_valid_batches)])
                    this_validation_loss = numpy.sum(validation_losses) * 100.0 / val_samples

                    msg = 'epoch %i, minibatch %i/%i, training error %.3f, validation error %.2f %%' % (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_costs[-1], this_validation_loss)

                    print(msg)

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        self.save()
                        print "New best score!"


        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))



