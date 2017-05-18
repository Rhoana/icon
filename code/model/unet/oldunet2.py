import cPickle
import gzip

import os
import sys
import time

import numpy
import numpy as np

import multiprocessing

from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, merge, ZeroPadding2D, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import SGD
from keras.regularizers import l2
from generate_data import *
#import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt
# loosing independence of backend for 
# custom loss function
import theano
import theano.tensor as T
from evaluation import Rand_membrane_prob
from theano.tensor.shared_randomstreams import RandomStreams


base_path = os.path.dirname(__file__)
#sys.path.insert(1,os.path.join(base_path, '..'))
sys.path.insert(1,os.path.join(base_path, '../mlp'))
sys.path.insert(2,os.path.join(base_path, '../../common'))
sys.path.insert(3,os.path.join(base_path, '../../database'))
sys.path.insert(4,os.path.join(base_path, '../'))

from db import DB
from paths import Paths
from utility import Utility
from data import Data

train_samples = 30
val_samples = 20

#patchSize = 572 #140
#patchSize_out = 388 #132

#learning_rate = 0.01
#momentum = 0.95

srng = RandomStreams(1234)

# need to define a custom loss, because all pre-implementations
# seem to assume that scores over patch add up to one which
# they clearly don't and shouldn't
def unet_crossentropy_loss(y_true, y_pred):
    weight_class_1 = 1.
    epsilon = 1.0e-4
    y_pred_clipped = T.clip(y_pred, epsilon, 1.0-epsilon)
    loss_vector = -T.mean(weight_class_1*y_true * T.log(y_pred_clipped) + (1-y_true) * T.log(1-y_pred_clipped), axis=1)
    average_loss = T.mean(loss_vector)
    return average_loss

def unet_crossentropy_loss_sampled(y_true, y_pred):
    print 'unet_crossentropy_loss_sampled'
    epsilon = 1.0e-4
    y_pred_clipped = T.flatten(T.clip(y_pred, epsilon, 1.0-epsilon))
    y_true = T.flatten(y_true)
    # this seems to work
    # it is super ugly though and I am sure there is a better way to do it
    # but I am struggling with theano to cooperate
    # filter the right indices
    indPos = T.nonzero(y_true)[0] # no idea why this is a tuple
    indNeg = T.nonzero(1-y_true)[0]
    # shuffle
    n = indPos.shape[0]
    indPos = indPos[srng.permutation(n=n)]
    n = indNeg.shape[0]
    indNeg = indNeg[srng.permutation(n=n)]
    # take equal number of samples depending on which class has less
    n_samples = T.cast(T.min([T.sum(y_true), T.sum(1-y_true)]), dtype='int64')

    indPos = indPos[:n_samples]
    indNeg = indNeg[:n_samples]
    loss_vector = -T.mean(T.log(y_pred_clipped[indPos])) - T.mean(T.log(1-y_pred_clipped[indNeg]))
    average_loss = T.mean(loss_vector)
    print 'average_loss:', average_loss
    return average_loss

def unet_block_down(input, nb_filter, doPooling=True, doDropout=False, doBatchNorm=False, initialization = 'glorot_uniform', weight_decay = 0.):
    # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
    # All are valid area, not same
    act1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(input)
    if doBatchNorm:
        act1 = BatchNormalization(mode=0, axis=1)(act1)

    act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
    if doBatchNorm:
        act2 = BatchNormalization(mode=0, axis=1)(act2)

    if doDropout:
        act2 = Dropout(0.5)(act2)

    if doPooling:
        # now downsamplig with maxpool
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid")(act2)
    else:
        pool1 = act2

    return (act2, pool1)


# need to define lambda layer to implement cropping
# input is a tensor of size (batchsize, channels, width, height)
def crop_layer(x, cs):
    cropSize = cs
    return x[:,:,cropSize:-cropSize, cropSize:-cropSize]


def unet_block_up(input, nb_filter, down_block_out, doBatchNorm=False, initialization = 'glorot_uniform', weight_decay = 0.):
    print "This is unet_block_up"
    print "input ", input._keras_shape
    # upsampling
    up_sampled = UpSampling2D(size=(2,2))(input)
    print "upsampled ", up_sampled._keras_shape
    # up-convolution
    conv_up = Convolution2D(nb_filter=nb_filter, nb_row=2, nb_col=2, subsample=(1,1),
                             init=initialization, activation='relu', border_mode="same", W_regularizer=l2(weight_decay))(up_sampled)
    print "up-convolution ", conv_up._keras_shape
    # concatenation with cropped high res output
    # this is too large and needs to be cropped
    print "to be merged with ", down_block_out._keras_shape

    #padding_1 = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
    #padding_2 = int((down_block_out._keras_shape[3] - conv_up._keras_shape[3])/2)
    #print "padding: ", (padding_1, padding_2)
    #conv_up_padded = ZeroPadding2D(padding=(padding_1, padding_2))(conv_up)
    #merged = merge([conv_up_padded, down_block_out], mode='concat', concat_axis=1)

    cropSize = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
    down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:], arguments={"cs":cropSize})(down_block_out)
    print "cropped layer size: ", down_block_out_cropped._keras_shape
    merged = merge([conv_up, down_block_out_cropped], mode='concat', concat_axis=1)

    print "merged ", merged._keras_shape
    # two 3x3 convolutions with ReLU
    # first one halves the feature channels
    act1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(merged)

    if doBatchNorm:
        act1 = BatchNormalization(mode=0, axis=1)(act1)

    print "conv1 ", act1._keras_shape
    act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
    if doBatchNorm:
        act2 = BatchNormalization(mode=0, axis=1)(act2)


    print "conv2 ", act2._keras_shape

    return act2


class UNET(object):

    srng = RandomStreams(1234)

    def __init__(
        self,
        id,
        rng,
        batch_size, 
        patch_size=572, 
        patch_size_out=388,
        offline=False,
        path=None,
        train_time=5.0,
        learning_rate=0.01,
        momentum=0.95):
        self.id = id
        self.type = 'UNET'
        self.offline = offline
        self.done = False
        self.path = path    
        self.batchSize = batch_size
        self.patchSize = patch_size 
        self.patchSize_out = patch_size_out
        self.learning_rate = learning_rate
        self.momentum      = momentum

        self.best_validation_loss = numpy.inf
        self.trainTime = train_time
        self.resample = False       
        self.error = np.inf
        self.error_threshold = 0.06
        self.best_val_loss_so_far = 0
        self.patience_counter = 0
        self.patience = 100
        self.patience_reset = 100

        self.doBatchNormAll = False
        self.doFineTune = False

        self.weight_decay = 0.
        self.weight_class_1 = 1.
        self.initialization = 'glorot_uniform'

        self.model = None

        self.initialize() 


    def initialize(self):
        print 'Unet.initialize'


    def train(self, offline=False, data=None, mean=None, std=None):

        print 'Unet.train'

        patchSize = self.patchSize
        patchSize_out = self.patchSize_out

        # input data should be large patches as prediction is also over large patches
        print
        print "=== building network ==="

        print "== BLOCK 1 =="
        input = Input(shape=(1, self.patchSize, self.patchSize))
        print "input ", input._keras_shape
        block1_act, block1_pool = UNET.unet_block_down(input=input, nb_filter=64, doBatchNorm=self.doBatchNormAll)
        print "block1 act ", block1_act._keras_shape
        print "block1 ", block1_pool._keras_shape
        #sys.stdout.flush()

        print "== BLOCK 2 =="
        block2_act, block2_pool = UNET.unet_block_down(input=block1_pool, nb_filter=128, doBatchNorm=self.doBatchNormAll)
        print "block2 ", block2_pool._keras_shape
        #sys.stdout.flush()

        print "== BLOCK 3 =="
        block3_act, block3_pool = UNET.unet_block_down(input=block2_pool, nb_filter=256, doBatchNorm=self.doBatchNormAll)
        print "block3 ", block3_pool._keras_shape
        #sys.stdout.flush()

        print "== BLOCK 4 =="
        block4_act, block4_pool = UNET.unet_block_down(input=block3_pool, nb_filter=512, doDropout=True, doBatchNorm=self.doBatchNormAll)
        print "block4 ", block4_pool._keras_shape
        #sys.stdout.flush()

        print "== BLOCK 5 =="
        print "no pooling"
        block5_act, block5_pool = UNET.unet_block_down(input=block4_pool, nb_filter=1024, doDropout=True, doPooling=False, doBatchNorm=self.doBatchNormAll)
        print "block5 ", block5_pool._keras_shape
        #sys.stdout.flush()

        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = UNET.unet_block_up(input=block5_act, nb_filter=512, down_block_out=block4_act, doBatchNorm=self.doBatchNormAll)
        print "block4 up", block4_up._keras_shape
        print
        #sys.stdout.flush()

        print "== BLOCK 3 UP =="
        block3_up = UNET.unet_block_up(input=block4_up, nb_filter=256, down_block_out=block3_act, doBatchNorm=self.doBatchNormAll)
        print "block3 up", block3_up._keras_shape
        print
        #sys.stdout.flush()

        print "== BLOCK 2 UP =="
        block2_up = UNET.unet_block_up(input=block3_up, nb_filter=128, down_block_out=block2_act, doBatchNorm=self.doBatchNormAll)
        print "block2 up", block2_up._keras_shape
        #sys.stdout.flush()

        print
        print "== BLOCK 1 UP =="
        block1_up = UNET.unet_block_up(input=block2_up, nb_filter=64, down_block_out=block1_act, doBatchNorm=self.doBatchNormAll)
        print "block1 up", block1_up._keras_shape
        sys.stdout.flush()

        print "== 1x1 convolution =="
        output = Convolution2D(nb_filter=1, nb_row=1, nb_col=1, subsample=(1,1),
                                 init=self.initialization, activation='sigmoid', border_mode="valid")(block1_up)
        print "output ", output._keras_shape
        output_flat = Flatten()(output)
        print "output flat ", output_flat._keras_shape

        print 'Unet.train'

        #self.load()

        project = DB.getProject( self.id )
        project.outPatchSize = patchSize_out
 
        if self.model == None: 

            dataset = Data( project )
            dataset.load( project )
            j_path, w_path = self.get_model_paths( )

            if os.path.exists( j_path ) and os.path.exists( w_path ):
                print 'loading from: ', j_path
                self.model = model_from_json(open( j_path ).read())
                self.model.load_weights( w_path )
            else:
                print 'creating....'
                self.model = Model(input=input, output=output_flat)

        sgd = SGD(lr=self.learning_rate, decay=0, momentum=self.momentum, nesterov=False)
        self.model.compile(loss=UNET.unet_crossentropy_loss_sampled, optimizer=sgd)
        #self.model.compile(loss=UNET.unet_crossentropy_loss, optimizer=sgd)
 
        print 'sampling data...' 

        # start pool for data
        print "Starting worker."
        pool = multiprocessing.Pool(processes=1)
        purpose = 'train'
        #futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction,purpose, train_samples, patchSize, patchSize_out, project]])

        best_val_loss_so_far = 0

        patience_counter = 0

            
        for epoch in xrange(10000000):
            if self.done:
                print 'stopping training...'
                break

            print "Waiting for data."
            #data = futureData.get()
            data   = dataset.sample()
            data_x = data[0].astype(np.float32)
            data_x = np.reshape(data_x, [-1, 1, patchSize, patchSize])
            data_y = data[1].astype(np.float32)

            data_x_val = np.reshape(data_x_val, [-1, 1, patchSize, patchSize])
            data_x = np.reshape(data_x, [-1, 1, patchSize, patchSize])

            print "got new data"
            print data_x.shape, data_y.shape

            #futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction,purpose, train_samples, patchSize, patchSize_out, project]])

            print "current learning rate: ", self.model.optimizer.lr.get_value()
            self.model.fit(data_x, data_y, batch_size=1, nb_epoch=1)
            #im_pred = 1-model.predict(x=data_x_val, batch_size = 1)

            self.save_current()

            mean_val_rand = 0.0
            for val_ind in xrange(val_samples):
                im_pred_single = np.reshape(im_pred[val_ind,:], (patchSize_out,patchSize_out))
                im_gt = np.reshape(data_label_val[val_ind], (patchSize_out,patchSize_out))
                validation_rand = Rand_membrane_prob(im_pred_single, im_gt)
                mean_val_rand += validation_rand
                print 'val:', val_ind, 'rand:', validation_rand, 'mrand:', mean_val_rand
            mean_val_rand /= np.double(val_samples)
            print "validation RAND ", mean_val_rand


    def train_offline(self, data, mean=None, std=None):
        pass

    def classify(self, image, mean=None, std=None):
        print 'Unet.classify'

    def predict(self, image, mean=None, std=None, threshold=0.5):
        print 'Unet.predict'

        patchSize = self.patchSize
        patchSize_out = self.patchSize_out


        start_time = time.clock()

        j_path, w_path = self.get_model_paths( )

        print 'loading model from:', j_path

        model = model_from_json(open( j_path ).read())
        model.load_weights( w_path )
        sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
        #model.compile(loss='categorical_crossentropy', optimizer=sgd)
        model.compile(loss=UNET.unet_crossentropy_loss_sampled, optimizer=sgd)

        image = image - 0.5

        probImage = np.zeros(image.shape)
        # count compilation time to init
        row = 0
        col = 0
        patch = image[row:row+patchSize,col:col+patchSize]
        data = np.reshape(patch, (1,1,patchSize,patchSize))
        probs = model.predict(x=data, batch_size=1)

        init_time = time.clock()
        #print "Initialization took: ", init_time - start_time

        patchSize = self.patchSize
        patchSize_out = self.patchSize_out
        image_orig = image.copy()
        for rotation in range(1):
            image = np.rot90(image_orig, rotation)
            # pad the image
            padding_ul = int(np.ceil((patchSize - patchSize_out)/2.0))
            # need large padding for lower right corner
            paddedImage = np.pad(image, patchSize, mode='reflect')
            needed_ul_padding = patchSize - padding_ul
            paddedImage = paddedImage[needed_ul_padding:, needed_ul_padding:]

            probImage_tmp = np.zeros(image.shape)
            for row in xrange(0,image.shape[0],patchSize_out):
                for col in xrange(0,image.shape[1],patchSize_out):
                    patch = paddedImage[row:row+patchSize,col:col+patchSize]
                    data = np.reshape(patch, (1,1,patchSize,patchSize))
                    probs = 1-model.predict(x=data, batch_size = 1)
                    probs = np.reshape(probs, (patchSize_out,patchSize_out))

                    row_end = patchSize_out
                    if row+patchSize_out > probImage.shape[0]:
                        row_end = probImage.shape[0]-row
                    col_end = patchSize_out
                    if col+patchSize_out > probImage.shape[1]:
                        col_end = probImage.shape[1]-col

                    probImage_tmp[row:row+row_end,col:col+col_end] = probs[:row_end,:col_end]
            probImage += np.rot90(probImage_tmp, 4-rotation)

        probImage = probImage / 1.0

        prob = self.threshold( probImage, factor=threshold )
        prob = prob.astype(dtype=int)
        prob = prob.flatten()

        end_time = time.clock()

        print "Prediction took: ", end_time - init_time
        print "Speed: ", 1./(end_time - init_time)
        print "Time total: ", end_time-start_time

        print 'results :', np.bincount( prob )
        print prob.shape
        print prob
        return prob


    def threshold(self, prob, factor=0.5):
        prob[ prob >= factor ] = 9
        prob[ prob <  factor ] = 1
        prob[ prob == 9      ] = 0
        return prob


    def get_model_paths(self):
        path = self.get_path()
        j_path = '%s_best.json'%(path)
        w_path = '%s_best_weights.h5'%(path)

        # first, attempt the best model, otherwise default to the latest
        if not os.path.exists( j_path ) and not os.path.exists( w_path ):
            path = Utility.get_dir(self.path)
            j_path = '%s/%s_%s.json'%(Paths.Models, self.id, self.type)
            w_path = '%s/%s_%s_weights.h5'%(Paths.Models, self.id, self.type)

        return j_path.lower(), w_path.lower()

    def load(self):
        j_path, w_path = self.get_model_paths( )

        if os.path.exists( j_path ) and os.path.exists( w_path ):
            print 'loading from: ', j_path
            self.model = model_from_json(open( j_path ).read())
            self.model.load_weights( w_path )
        else:
            print 'creating....'
            inp, out = self.gen_input_output()
            print inp.shape, out.shape
            self.model = Model(input=inp, output=out)

        sgd = SGD(lr=self.learning_rate, decay=0, momentum=self.momentum, nesterov=False)
        self.model.compile(loss=UNET.unet_crossentropy_loss_sampled, optimizer=sgd)


    def save_current(self):
        path = Utility.get_dir(self.path)
        j_path = '%s/%s_%s.json'%(Paths.Models, self.id, self.type)
        w_path = '%s/%s_%s_weights.h5'%(Paths.Models, self.id, self.type)
        j_path = j_path.lower()
        w_path = w_path.lower()

        json_string = self.model.to_json()
        open(j_path, 'w').write(json_string)
        self.model.save_weights(w_path, overwrite=True)

    def save_best(self):
        print 'Unet.save'
        path = Utility.get_dir(self.path)
        revision = 0
        if not self.offline:
            revision = DB.getRevision( self.id )
            revision = (revision+1)%10
            path = '%s/%s_%s_%d'%(Paths.Models, self.id, self.type, revision)
            path = path.lower()

        j_path = '%s_best.json'%(path)
        w_path = '%s_best_weights.h5'%(path)
        j_path = j_path.lower()
        w_path = w_path.lower()

        print 'saving...', path
        # saving code here...
        json_string = self.model.to_json()
        open(j_path, 'w').write(json_string)
        self.model.save_weights(w_path, overwrite=True)

        if not self.offline:
            DB.finishSaveModel( self.id, revision )

    def get_path(self):
        if self.offline:
            return self.path

        rev  = DB.getRevision( self.id )
        path = '%s/%s.%s.%d'%(Paths.Models, self.id, self.type, rev )
        return path.lower()

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


    # need to define a custom loss, because all pre-implementations
    # seem to assume that scores over patch add up to one which
    # they clearly don't and shouldn't
    @staticmethod
    def unet_crossentropy_loss(y_true, y_pred):
        weight_class_1 = 1.
        epsilon = 1.0e-4
        y_pred_clipped = T.clip(y_pred, epsilon, 1.0-epsilon)
        loss_vector = -T.mean(weight_class_1*y_true * T.log(y_pred_clipped) + (1-y_true) * T.log(1-y_pred_clipped), axis=1)
        average_loss = T.mean(loss_vector)
        return average_loss

    @staticmethod
    def unet_crossentropy_loss_sampled(y_true, y_pred):
        epsilon = 1.0e-4
        y_pred_clipped = T.flatten(T.clip(y_pred, epsilon, 1.0-epsilon))
        y_true = T.flatten(y_true)
        # this seems to work
        # it is super ugly though and I am sure there is a better way to do it
        # but I am struggling with theano to cooperate
        # filter the right indices
        indPos = T.nonzero(y_true)[0] # no idea why this is a tuple
        indNeg = T.nonzero(1-y_true)[0]
        # shuffle
        n = indPos.shape[0]
        indPos = indPos[UNET.srng.permutation(n=n)]
        n = indNeg.shape[0]
        indNeg = indNeg[UNET.srng.permutation(n=n)]
        # take equal number of samples depending on which class has less
        n_samples = T.cast(T.min([T.sum(y_true), T.sum(1-y_true)]), dtype='int64')

        indPos = indPos[:n_samples]
        indNeg = indNeg[:n_samples]
        loss_vector = -T.mean(T.log(y_pred_clipped[indPos])) - T.mean(T.log(1-y_pred_clipped[indNeg]))
        average_loss = T.mean(loss_vector)
        return average_loss

    @staticmethod
    def unet_block_down(input, nb_filter, doPooling=True, doDropout=False, doBatchNorm=False, initialization = 'glorot_uniform', weight_decay = 0.):
        # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
        # All are valid area, not same
        act1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(input)
        if doBatchNorm:
            act1 = BatchNormalization(mode=0, axis=1)(act1)

        act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
        if doBatchNorm:
            act2 = BatchNormalization(mode=0, axis=1)(act2)

        if doDropout:
            act2 = Dropout(0.5)(act2)

        if doPooling:
            # now downsamplig with maxpool
            pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid")(act2)
        else:
            pool1 = act2

        return (act2, pool1)

    # need to define lambda layer to implement cropping
    # input is a tensor of size (batchsize, channels, width, height)
    @staticmethod
    def crop_layer( x, cs):
        cropSize = cs
        return x[:,:,cropSize:-cropSize, cropSize:-cropSize]

    @staticmethod
    def unet_block_up(input, nb_filter, down_block_out, doBatchNorm=False, initialization = 'glorot_uniform', weight_decay = 0.):
        print "This is unet_block_up"
        print "input ", input._keras_shape
        # upsampling
        up_sampled = UpSampling2D(size=(2,2))(input)
        print "upsampled ", up_sampled._keras_shape
        # up-convolution
        conv_up = Convolution2D(nb_filter=nb_filter, nb_row=2, nb_col=2, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="same", W_regularizer=l2(weight_decay))(up_sampled)
        print "up-convolution ", conv_up._keras_shape
        # concatenation with cropped high res output
        # this is too large and needs to be cropped
        print "to be merged with ", down_block_out._keras_shape

        #padding_1 = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
        #padding_2 = int((down_block_out._keras_shape[3] - conv_up._keras_shape[3])/2)
        #print "padding: ", (padding_1, padding_2)
        #conv_up_padded = ZeroPadding2D(padding=(padding_1, padding_2))(conv_up)
        #merged = merge([conv_up_padded, down_block_out], mode='concat', concat_axis=1)

        cropSize = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
        down_block_out_cropped = Lambda(UNET.crop_layer, output_shape=conv_up._keras_shape[1:], arguments={"cs":cropSize})(down_block_out)
        print "cropped layer size: ", down_block_out_cropped._keras_shape
        merged = merge([conv_up, down_block_out_cropped], mode='concat', concat_axis=1)

        print "merged ", merged._keras_shape
        # two 3x3 convolutions with ReLU
        # first one halves the feature channels
        act1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(merged)

        if doBatchNorm:
            act1 = BatchNormalization(mode=0, axis=1)(act1)

        print "conv1 ", act1._keras_shape
        act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
        if doBatchNorm:
            act2 = BatchNormalization(mode=0, axis=1)(act2)


        print "conv2 ", act2._keras_shape

        return act2

