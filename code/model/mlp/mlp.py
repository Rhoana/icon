import cPickle
import gzip
import os
import sys
import time

import numpy
import numpy as np

import theano
import theano.tensor as T

from scipy.ndimage.interpolation import shift

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../'))
sys.path.insert(2,os.path.join(base_path, '../../common'))
sys.path.insert(3,os.path.join(base_path, '../../database'))

print 'mlp path:', base_path

from db import DB
from paths import Paths
from utility import Utility
from hiddenlayer import HiddenLayer
from logistic_sgd import LogisticRegression
from generateTrainValTestData import gen_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
import multiprocessing
from vsk_utils import shared_single_dataset
from activation_functions import rectified_linear
import smtplib
import getpass

class MLP(object):
    def __init__(
        self, 
        rng, 
        input, 
        n_in=None, 
        n_hidden=None, 
        n_out=2, 
        path=None, 
        id=None,
        offline=False,
        batch_size=None, 
        patch_size=None, 
        train_time=5.0,
        learning_rate=0.1,
        momentum=0.9,
        activation=rectified_linear):

        self.n_out = n_out
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.input = input
        self.x = input

        self.id = id
        self.type = 'MLP'
        self.offline = offline
        self.rng = rng
        self.done = False
        self.path = path    
        self.activation = activation
        self.hiddenSizes = n_hidden
        self.batchSize = batch_size
        self.patchSize = patch_size 
        self.hiddenSizes = n_hidden
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.trainTime = train_time
        self.resample = False
        self.best_validation_loss = numpy.inf
        self.best_train_error = np.inf
	self.revision = DB.getRevision( self.id ) 
        self.initialize()

    def get_path(self):
        if self.offline:
            return self.path

        rev  = DB.getRevision( self.id )
        path = '%s/best_%s.%s.%d.pkl'%(Paths.Models, self.id, self.type, rev )
        return path.lower()

    def initialize(self):

        self.hiddenLayers = []
        self.params = []
        input = self.input
        rng = self.rng
        n_out = self.n_out

        path = self.get_path()

        fromFile = (path is not None) and os.path.exists( path )

        if fromFile: 
            with open(path, 'r') as file:
                print 'loading mlp file from file...', path
                d = cPickle.load(file)
                savedhiddenLayers        = d[0]
                saved_logRegressionLayer = d[1]
                self.n_in                = d[2]
                self.n_hidden            = d[3]

        next_input = input
        next_n_in = self.n_in
           

        print 'self.n_hidden:', self.n_hidden
         
        for n_h in self.n_hidden:
            hl = HiddenLayer(rng=rng, input=next_input,
                             n_in=next_n_in, n_out=n_h,
                             activation=self.activation)
            next_input = hl.output
            next_n_in = n_h
            self.hiddenLayers.append(hl)
            self.params += hl.params
            
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=self.n_hidden[-1],
            n_out=n_out)
        
        self.params += self.logRegressionLayer.params
        
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        self.y_pred = self.logRegressionLayer.y_pred

        if fromFile:
            for hl, shl in zip(self.hiddenLayers, savedhiddenLayers):
                hl.W.set_value(shl.W.get_value())
                hl.b.set_value(shl.b.get_value())

            self.logRegressionLayer.W.set_value(saved_logRegressionLayer.W.get_value())
            self.logRegressionLayer.b.set_value(saved_logRegressionLayer.b.get_value())

        self.cost = self.negative_log_likelihood 


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
            cPickle.dump((
                self.hiddenLayers, 
                self.logRegressionLayer, 
                self.n_in, 
                self.n_hidden), file)

        if not self.offline:
            DB.finishSaveModel( self.id, revision )


    def get_patch_size(self):
        return np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

    def predict_image(self, x, img, normMean=None, norm_std=None):
        start_time = time.clock()

        row_range = 1
        img = normalizeImage(img)
        imSize = np.shape(img)
        membraneProbabilities = np.zeros(1024*1024, dtype=int )
        patchSize = np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

        data_shared = shared_single_dataset(np.zeros((imSize[0]*row_range,patchSize**2)), borrow=True)

        classify = theano.function(
            [],
            self.logRegressionLayer.y_pred,
            givens={x: data_shared}
        )
        for row in xrange(0,1024,row_range):
            if row%100 == 0:
                print row
            data = generate_patch_data_rows(img, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize, data_mean=normMean, data_std=norm_std)
            data_shared.set_value(np.float32(data))
            membraneProbabilities[row*1024:row*1024+row_range*1024] = classify()

        end_time = time.clock()
        total_time = (end_time - start_time)
        print >> sys.stderr, ('Running time: ' +
                              '%.2fm' % (total_time / 60.))

        return np.array(membraneProbabilities)
 
    def classify(self, image, mean=None, std=None):

        #imSize = (1024,1024)
        imSize = np.shape(image)
        print 'imSize:', imSize
        image = Utility.pad_image( image, self.patchSize )
        imSizePadded = np.shape(image)

        print 'mlp.classify mean:', mean, 'std:', std
        start_time = time.clock()

        row_range = 1
        #imSize = np.shape(image)
        #membraneProbabilities = np.zeros(np.shape(image))
        membraneProbabilities = np.zeros(imSize)
        patchSize = np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

        print 'imSize:', imSize
        print 'imSizePadded:', imSizePadded
        print 'mp:', np.shape(membraneProbabilities)

        data_shared = shared_single_dataset(np.zeros((imSize[0]*row_range,patchSize**2)), borrow=True)

        classify = theano.function(
            [],
            self.p_y_given_x,
            givens={self.x: data_shared}
        )

        for row in xrange(0,1024,row_range):
            if row%100 == 0:
                print row

            #data = Utility.get_patch(image, row, row_range, patchSize ) 

            #print 'data shape:', data.shape  
            '''
            data = Utility.generate_patch_data_rows(
                    image,
                    rowOffset=row,
                    rowRange=row_range,
                    patchSize=patchSize,
                    imSize=imSizePadded,
                    data_mean=mean,
                    data_std=std)
            '''
            data = Utility.get_patch(
                    image, 
                    row, 
                    row_range, 
                    patchSize,
                    data_mean=mean,
                    data_std=std)

            #print 'data:', np.shape(data)
            data_shared.set_value(np.float32(data))
            result = classify()
            #print 'results:', np.shape(result)
            membraneProbabilities[row,:] = result[:,0]

        end_time = time.clock()
        total_time = (end_time - start_time)
        print >> sys.stderr, ('Running time: ' +
                              '%.2fm' % (total_time / 60.))

        return np.array(membraneProbabilities)
 
   
    def predict(self, image, mean=None, std=None, threshold=0.5):
        prob = self.classify( image, mean=mean, std=std)
        prob = self.threshold( prob, factor=threshold )
        prob = prob.astype(dtype=int)
        prob = prob.flatten()
        return prob
 
    def threshold(self, prob, factor=0.5):
        prob[ prob > factor ] = 9
        prob[ prob <=  factor ] = 1
        prob[ prob == 9      ] = 0
        return prob

    def reportTrainingStats(self, elapsedTime, batchIndex, valLoss, trainCost, mode=0):
        if not self.offline:
            DB.storeTrainingStats( self.id, valLoss, trainCost, mode=mode)

        msg = '(%0.1f)     %i     %f%%'%\
        (
           elapsedTime,
           batchIndex,
           valLoss
        )
        status = '[%f]'%(trainCost)
        Utility.report_status( msg, status )


    def oldtrain(self, offline=False, data=None, mean=None,std=None):
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

        train_samples = len(train_y)
        valid_samples = len(valid_y)

        print 'valid_samples:',valid_samples
        print 'train_samples:', train_samples

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

        '''
        predict_samples = theano.function(
                inputs=[index],
                outputs=T.neq(self.y_pred, self.y),
                givens={
                        x: train_set_x[index * batchSize: (index + 1) * batchSize],
                        y: train_set_y[index * batchSize: (index + 1) * batchSize]
                }
        )
        '''
        predict_samples = theano.function(
                [],
                outputs=T.neq(self.y_pred, self.y),
                givens={
                        x: train_set_x,
                        y: train_set_y,
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
        print 'self.best_validation_loss:', self.best_validation_loss
        best_iter = 0
        validation_frequency = 1

        start_time = time.clock()

        elapsed_time = 0
        iter = 0

        minibatch_avg_costs = []
        minibatch_index = 0


        #while (elapsed_time < self.trainTime)\
        #    and (minibatch_index<n_train_batches)\
        #    and (not self.done):
        while (minibatch_index<n_train_batches) and (not self.done):
            if (elapsed_time >= self.trainTime):
                break

            train_cost = train_model(minibatch_index)

            # test the trained samples against the target
            # values to measure the training performance
            i = minibatch_index

            '''
            probs = predict_samples(minibatch_index)
            #print 'probs:', probs.shape
            i_batch = data.i_train[ i * batchSize:(i+1)*batchSize ]
            data.p[ i_batch ] = probs
            '''

            '''
            good = np.where( probs == 0)[0]
            bad  = np.where( probs == 1)[0]
            print 'bad:', len(bad)
            print 'good:', len(good)
            #print probs
            '''
            #print '----->traincost:', type(train_cost), train_cost

            minibatch_avg_costs.append(train_cost)

            iter += 1
            #iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0 and valid_samples > 0:

                validation_losses = np.array([validate_model(i) for i in xrange(n_valid_batches)])
                this_validation_loss = numpy.sum(validation_losses) * 100.0 / valid_samples
                elapsed_time = time.clock() - start_time

                '''
                self.reportTrainingStats(elapsed_time,
                        minibatch_index,
                        this_validation_loss,
                        minibatch_avg_costs[-1].item(0))
                '''
                print this_validation_loss, '/', self.best_validation_loss
                data.add_validation_loss( this_validation_loss )

                # if we got the best validation score until now
                if this_validation_loss < self.best_validation_loss:
                    self.best_validation_loss = this_validation_loss
                    best_iter = iter

                    self.save()
                    print "New best score!"

            # advance to next mini batch
            minibatch_index += 1

            # update elapsed time
            elapsed_time = time.clock() - start_time

        if valid_samples == 0:
            self.save()

        probs = predict_samples()
        data.p[ data.i_train ] = probs

        elapsed_time = time.clock() - start_time
        msg = 'The code an for'
        status = '%f seconds' % (elapsed_time)
        Utility.report_status( msg, status )
        print 'done...'


    def train(self, 
        offline=False, 
        data=None, 
        mean=None,
        std=None
        ):
        print 'mlp.train'

        def gradient_updates_momentum(cost, params, learning_rate, momentum):
            updates = []
            for param in params:
                param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                updates.append((param, param - learning_rate*param_update))
                updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
            return updates

        patchSize = self.patchSize
        batchSize = self.batchSize
        learning_rate  = self.learning_rate
        momentum = self.momentum

        rng = numpy.random.RandomState(1234)

        tx, ty, vx, vy, reset = data.sample()
        train_samples  = len(ty)
        val_samples    = len(vy)
        train_set_x, train_set_y = shared_dataset((tx, ty), doCastLabels=True)

        if val_samples > 0:
            valid_set_x, valid_set_y = shared_dataset((vx, vy), doCastLabels=True)

        if reset:
            self.best_validation_loss = numpy.inf

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_samples / batchSize
        n_valid_batches = val_samples / 1000 #batchSize

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
        print 'n_train_batches:',n_train_batches
        print 'n_valid_batches:',n_valid_batches
        print 'train_samples:', train_samples
        print 'val_samples:', val_samples
        print 'best_validation:', self.best_validation_loss

        if val_samples > 0:
            validate_model = theano.function(
                [index],
                self.errors(y),
                givens={
                    x: valid_set_x[index * batchSize: (index + 1) * batchSize],
                    y: valid_set_y[index * batchSize: (index + 1) * batchSize]
                }
            )

        predict_samples = theano.function(
                [],
                outputs=T.neq(self.y_pred, y),
                givens={
                        x: train_set_x,
                        y: train_set_y,
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
        validation_frequency = 1
        start_time = time.clock()

        minibatch_avg_costs = []
        iter = 0
        epoch = 0
        self.best_train_error = np.inf
        last_train_error = numpy.inf
        for minibatch_index in xrange(n_train_batches):
            if self.done:
                break

            train_cost = train_model(minibatch_index)
            minibatch_avg_costs.append( train_cost )

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if n_valid_batches == 0:
                train_error = minibatch_avg_costs[-1].item(0)

                print minibatch_index, '-', train_error
                if train_error < self.best_train_error:
                    self.best_train_error = train_error
                    self.save()
                      

            if n_valid_batches > 0 and (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = np.array([validate_model(i) for i
                                     in xrange(n_valid_batches)])
                #this_validation_loss = numpy.sum(validation_losses) * 100.0 / val_samples
                this_validation_loss = numpy.mean(validation_losses*100.0)

                elapsed_time = time.clock() - start_time
 
                data.report_stats(
                    self.id,
                    elapsed_time, 
                    minibatch_index, 
                    this_validation_loss, 
                    minibatch_avg_costs[-1].item(0))

                # if we got the best validation score until now
                if this_validation_loss < self.best_validation_loss:
                    self.best_validation_loss = this_validation_loss
                    self.save()
                    print "New best score!"

        #if n_valid_batches == 0:
        #    self.save()

        if not self.offline:
            probs = predict_samples()
            data.p[ data.i_train ] = probs
            data.save_stats()

        



    def train_offline(self, data, mean=None, std=None):
        print 'training....'
        train_samples = 700000
        val_samples=5000
        test_samples=1000
        n_epochs=5000
        patchSize = self.patchSize
        batchSize = 50 #self.batchSize
        learning_rate  = self.learning_rate
        momentum = 0.9  #self.momentum

        def gradient_updates_momentum(cost, params, learning_rate, momentum):
            updates = []
            for param in params:
                param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                updates.append((param, param - learning_rate*param_update))
                updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
            return updates

        rng = numpy.random.RandomState(1234)

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

        '''

        d = gen_data_supervised(
            purpose='train',
            nsamples=train_samples,
            patchSize=patchSize,
            balanceRate=0.5,
            data_mean=mean,
            data_std=std)
        data = d[0]
        train_set_x, train_set_y = shared_dataset(data, doCastLabels=True)

        #print 'data:', np.shape(data)
        #print 'train:', np.shape(train_set_x), np.shape(train_set_y)
        #print 'valid:', np.shape(valid_set_x), np.shape(valid_set_y)
        #print 'test :', np.shape(test_set_x), np.shape(test_set_y)

        norm_mean = d[1]
        norm_std  = d[2]
        grayImages = d[3]
        labelImages = d[4]
        maskImages = d[5]

        print 'norm_std:', norm_std
        print 'norm_mean:',norm_mean
    
        # validation data
        d = gen_data_supervised(
            purpose='validate',
            nsamples=val_samples,
            patchSize=patchSize,
            balanceRate=0.5,
            data_mean=norm_mean,
            data_std=norm_std)[0]
        valid_set_x, valid_set_y = shared_dataset(d, doCastLabels=True)

        # test data
        d = gen_data_supervised(
            purpose='test',
            nsamples=test_samples,
            patchSize=patchSize,
            balanceRate=0.5,
            data_mean=norm_mean,
            data_std=norm_std)[0]
        test_set_x, test_set_y = shared_dataset(d, doCastLabels=True)

        '''

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_samples / batchSize
        n_valid_batches = val_samples / 1000 #batchSize
        n_test_batches = test_samples / 1000 #batchSize

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

        print 'lr:', learning_rate
        print 'patchsizwe:', patchSize
        print 'm:', momentum

        print 'n_train_batches:', n_train_batches
        print 'n_valid_batches:', n_valid_batches
        print 'n_test_batches:', n_test_batches


        # start pool for data
        print "Starting worker."
        '''
        pool = multiprocessing.Pool(processes=1)
        futureData = pool.apply_async(
                        stupid_map_wrapper,
                        [[gen_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])
        '''

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

                #if diff < 0.025:
                print 'resampling...'
                print 'diff:', diff
                print 'last_avg_validation_loss:', last_avg_validation_loss
                '''
                d = gen_data_supervised(
                    purpose='train',
                    nsamples=train_samples,
                    patchSize=patchSize,
                    balanceRate=0.5,
                    data_mean=mean,
                    data_std=std)
                data = d[0]
                train_set_x.set_value(np.float32(data[0]))
                train_set_y.set_value(np.int32(data[1]))

                '''
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
                    #this_validation_loss = numpy.sum(validation_losses) * 100.0 / val_samples
                    this_validation_loss = numpy.mean(validation_losses*100.0)
                    avg_validation_losses.append(this_validation_loss*100)

                    msg = 'epoch %i, minibatch %i/%i, training error %.3f, validation error %.2f %%' % (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_costs[-1], this_validation_loss)

                    print(msg)

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        self.save()
                        print "New best score!"

        #pool.close()
        #pool.join()
        print "Pool closed."

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

