# yet another version of the IDSIA network
# based on code from keras tutorial 
# http://keras.io/getting-started/sequential-model-guide/
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, merge, ZeroPadding2D, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import SGD
from keras.regularizers import l2
from generate_data import *
import multiprocessing
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
from project import Project
from paths import Paths
from utility import Utility
from data import Data

# run as python unet.py 1 (for training) 0 (prediction)

rng = np.random.RandomState(7)

train_samples = 30 
val_samples = 20
learning_rate = 0.01
momentum = 0.95
doTrain = int(sys.argv[1])

# if you set mode to same then these values for input and output has to be the same
# currently it is set to valid
patchSize = 572 #140
patchSize_out = 388 #132

weight_decay = 0.
weight_class_1 = 1. 


patience = 100
patience_reset = 100

doBatchNormAll = False
doFineTune = False

purpose = 'train'
initialization = 'glorot_uniform'
filename = 'unet_Cerebellum_clahe'
print "filename: ", filename

srng = RandomStreams(1234)

# need to define a custom loss, because all pre-implementations
# seem to assume that scores over patch add up to one which
# they clearly don't and shouldn't
def unet_crossentropy_loss(y_true, y_pred):
    epsilon = 1.0e-4
    y_pred_clipped = T.clip(y_pred, epsilon, 1.0-epsilon)
    loss_vector = -T.mean(weight_class_1*y_true * T.log(y_pred_clipped) + (1-y_true) * T.log(1-y_pred_clipped), axis=1)
    average_loss = T.mean(loss_vector)
    return average_loss

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
    indPos = indPos[srng.permutation(n=n)]
    n = indNeg.shape[0]
    indNeg = indNeg[srng.permutation(n=n)]
    # take equal number of samples depending on which class has less
    n_samples = T.cast(T.min([T.sum(y_true), T.sum(1-y_true)]), dtype='int64')

    indPos = indPos[:n_samples]
    indNeg = indNeg[:n_samples]
    loss_vector = -T.mean(T.log(y_pred_clipped[indPos])) - T.mean(T.log(1-y_pred_clipped[indNeg]))
    average_loss = T.mean(loss_vector)
    return average_loss

def unet_block_down(input, nb_filter, doPooling=True, doDropout=False, doBatchNorm=False):
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


def unet_block_up(input, nb_filter, down_block_out, doBatchNorm=False):
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
    

if doTrain:
    # input data should be large patches as prediction is also over large patches
    print 
    print "=== building network ==="

    print "== BLOCK 1 =="
    input = Input(shape=(1, patchSize, patchSize))
    print "input ", input._keras_shape
    block1_act, block1_pool = unet_block_down(input=input, nb_filter=64, doBatchNorm=doBatchNormAll)
    print "block1 act ", block1_act._keras_shape
    print "block1 ", block1_pool._keras_shape
    #sys.stdout.flush()

    print "== BLOCK 2 =="
    block2_act, block2_pool = unet_block_down(input=block1_pool, nb_filter=128, doBatchNorm=doBatchNormAll)
    print "block2 ", block2_pool._keras_shape
    #sys.stdout.flush()

    print "== BLOCK 3 =="
    block3_act, block3_pool = unet_block_down(input=block2_pool, nb_filter=256, doBatchNorm=doBatchNormAll)
    print "block3 ", block3_pool._keras_shape
    #sys.stdout.flush()

    print "== BLOCK 4 =="
    block4_act, block4_pool = unet_block_down(input=block3_pool, nb_filter=512, doDropout=True, doBatchNorm=doBatchNormAll)
    print "block4 ", block4_pool._keras_shape
    #sys.stdout.flush()

    print "== BLOCK 5 =="
    print "no pooling"
    block5_act, block5_pool = unet_block_down(input=block4_pool, nb_filter=1024, doDropout=True, doPooling=False, doBatchNorm=doBatchNormAll)
    print "block5 ", block5_pool._keras_shape
    #sys.stdout.flush()

    print "=============="
    print

    print "== BLOCK 4 UP =="
    block4_up = unet_block_up(input=block5_act, nb_filter=512, down_block_out=block4_act, doBatchNorm=doBatchNormAll)
    print "block4 up", block4_up._keras_shape
    print
    #sys.stdout.flush()

    print "== BLOCK 3 UP =="
    block3_up = unet_block_up(input=block4_up, nb_filter=256, down_block_out=block3_act, doBatchNorm=doBatchNormAll)
    print "block3 up", block3_up._keras_shape
    print
    #sys.stdout.flush()

    print "== BLOCK 2 UP =="
    block2_up = unet_block_up(input=block3_up, nb_filter=128, down_block_out=block2_act, doBatchNorm=doBatchNormAll)
    print "block2 up", block2_up._keras_shape
    #sys.stdout.flush()

    print
    print "== BLOCK 1 UP =="
    block1_up = unet_block_up(input=block2_up, nb_filter=64, down_block_out=block1_act, doBatchNorm=doBatchNormAll)
    print "block1 up", block1_up._keras_shape
    sys.stdout.flush()

    print "== 1x1 convolution =="
    output = Convolution2D(nb_filter=1, nb_row=1, nb_col=1, subsample=(1,1),
                             init=initialization, activation='sigmoid', border_mode="valid")(block1_up)
    print "output ", output._keras_shape
    output_flat = Flatten()(output)
    print "output flat ", output_flat._keras_shape
    model = Model(input=input, output=output_flat)
    #model = Model(input=input, output=block1_act)
    #sys.stdout.flush()

    if doFineTune:
        model = model_from_json(open('unet_sampling_best.json').read())
        model.load_weights('unet_sampling_best_weights.h5')

    sgd = SGD(lr=learning_rate, decay=0, momentum=momentum, nesterov=False)
    #model.compile(loss='mse', optimizer=sgd)
    model.compile(loss=unet_crossentropy_loss_sampled, optimizer=sgd)


    project = DB.getProject('testunet')
    '''
    project.outPatchSize = patchSize_out
    data = Data( project )
    data.load( project )
    
    for epoch in xrange(10000000):
        d          = data.sample()
        data_x     = d[0]
        data_y     = d[1]
        data_x_val = d[2]
        data_y_val = d[3]
        reset      = d[4]

        print '----'
        print 'train_x:', data_x.shape
        print 'valid_x:', data_x_val.shape

        data_x_val = data_x_val.astype(np.float32)
        data_x_val = np.reshape(data_x_val, [-1, 1, patchSize, patchSize])
        data_y_val = data_y_val.astype(np.float32)
        data_label_val = data_y_val #data_val[2]

        data_x = data_x.astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, patchSize, patchSize])
        data_y = data_y.astype(np.float32)

        print 'train_x:', data_x.shape
        print 'train_y:', data_y.shape
        print 'valid_x:', data_x_val.shape
        print 'valid_y:', data_y_val.shape

        print "epoch:", epoch, "current learning rate: ", model.optimizer.lr.get_value()
        model.fit(data_x, data_y, batch_size=1, nb_epoch=1)

        exit(1)

        im_pred = 1-model.predict(x=data_x_val, batch_size = 1)

        print im_pred.shape
        print im_pred
        mean_val_rand = 0
        for val_ind in xrange(val_samples):
            im_pred_single = np.reshape(im_pred[val_ind,:], (patchSize_out,patchSize_out))
            im_gt = np.reshape(data_label_val[val_ind], (patchSize_out,patchSize_out))
            validation_rand = Rand_membrane_prob(im_pred_single, im_gt)
            mean_val_rand += validation_rand
            print 'val:', val_ind, 'rand:', validation_rand, 'mrand:', mean_val_rand
        mean_val_rand /= np.double(val_samples)
        print "validation RAND ", mean_val_rand

    exit(1)
    '''

    data_val = generate_experiment_data_patch_prediction(purpose='validate', nsamples=val_samples, patchSize=patchSize, outPatchSize=patchSize_out, project=project)

    data_x_val = data_val[0].astype(np.float32)
    data_x_val = np.reshape(data_x_val, [-1, 1, patchSize, patchSize])
    data_y_val = data_val[1].astype(np.float32)
    data_label_val = data_y_val #data_val[2]

    # start pool for data
    print "Starting worker."
    pool = multiprocessing.Pool(processes=1)
    futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction,purpose, train_samples, patchSize, patchSize_out, project]])

    #project = DB.getProject('testunet') 
    #data = Data( project )
    #data.load( project )

    best_val_loss_so_far = 0
    
    patience_counter = 0
    for epoch in xrange(10000000):
        print "Waiting for data."
        data = futureData.get()

        data_x = data[0].astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, patchSize, patchSize])
        data_y = data[1].astype(np.float32)

        print "got new data"
        futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction,purpose, train_samples, patchSize, patchSize_out, project]])

        print "current learning rate: ", model.optimizer.lr.get_value()
        model.fit(data_x, data_y, batch_size=1, nb_epoch=1)
        print data_x.shape, data_y.shape

        im_pred = 1-model.predict(x=data_x_val, batch_size = 1)

        if True:
            continue

        mean_val_rand = 0
        for val_ind in xrange(val_samples):
            im_pred_single = np.reshape(im_pred[val_ind,:], (patchSize_out,patchSize_out))
            im_gt = np.reshape(data_label_val[val_ind], (patchSize_out,patchSize_out))
            validation_rand = Rand_membrane_prob(im_pred_single, im_gt)
            mean_val_rand += validation_rand
            print 'val:', val_ind, 'rand:', validation_rand, 'mrand:', mean_val_rand
        mean_val_rand /= np.double(val_samples)
        print "validation RAND ", mean_val_rand
       
        json_string = model.to_json()
        open(filename+'.json', 'w').write(json_string)
        model.save_weights(filename+'_weights.h5', overwrite=True) 
        
        print mean_val_rand, " > ",  best_val_loss_so_far
        print mean_val_rand - best_val_loss_so_far
        if mean_val_rand > best_val_loss_so_far:
            best_val_loss_so_far = mean_val_rand
            print "NEW BEST MODEL"
            json_string = model.to_json()
            open(filename+'_best.json', 'w').write(json_string)
            model.save_weights(filename+'_best_weights.h5', overwrite=True) 
            patience_counter=0
        else:
            patience_counter +=1

        # no progress anymore, need to decrease learning rate
        if patience_counter == patience:
            print "DECREASING LEARNING RATE"
            print "before: ", learning_rate
            learning_rate *= 0.1
            print "now: ", learning_rate
            model.optimizer.lr.set_value(learning_rate)
            patience = patience_reset
            patience_counter = 0

            # reload best state seen so far
            model = model_from_json(open(filename+'_best.json').read())
            model.load_weights(filename+'_best_weights.h5')
            model.compile(loss=unet_crossentropy_loss_sampled, optimizer=sgd)

        
        # stop if not learning anymore
        if learning_rate < 1e-7:
            break

else:
    start_time = time.clock()

    network_file_path = './'
    #network_file_path = 'to_evaluate/'
    file_search_string = network_file_path + '*.json'
    files = sorted( glob.glob( file_search_string ) )
    #pathPrefix = '/media/vkaynig/Data1/all_data/testing/AC4_small/'
    #pathPrefix = '/media/vkaynig/Data1/all_data/testing/left_cylinder_test/'
    #pathPrefix = '/media/vkaynig/Data1/all_data/testing/AC3/'
    #pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/Thalamus-LGN/Data/25-175_train/'
    #pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/Cerebellum-P7/Dense/'

    # this is the folder where the images to be predicted resides.
    # the folder is called 'gray_images' and it should contain tiff
    # images.
    # the results of prediction are written to a folder called
    # 'boundaryProbabilities' inside this parent folder.
    pathPrefix = '/n/home00/fgonda/felix_data/' 

    for file_index in xrange(np.shape(files)[0]):
        print files[file_index]
        model = model_from_json(open(files[file_index]).read())
        weight_file = ('.').join(files[file_index].split('.')[:-1])
        model.load_weights(weight_file+'_weights.h5')
        model_name = os.path.splitext(os.path.basename(files[file_index]))[0]
        
        # create directory
        if not os.path.exists(pathPrefix+'boundaryProbabilities/'+model_name):
            os.makedirs(pathPrefix+'boundaryProbabilities/'+model_name)

        sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        
        img_search_string = pathPrefix + 'gray_images/*.tif'
        img_files = sorted( glob.glob( img_search_string ) )
        
        for img_index in xrange(np.shape(img_files)[0]):
            print img_files[img_index]
            image = mahotas.imread(img_files[img_index])
            image = normalizeImage(image, doClahe=True) 
            
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
            
            print pathPrefix+'boundaryProbabilities/'+model_name+'/'+str(img_index).zfill(4)+'.tif'
            mahotas.imsave(pathPrefix+'boundaryProbabilities/'+model_name+'/'+str(img_index).zfill(4)+'.tif', np.uint8(probImage*255))
            
            end_time = time.clock()
            print "Prediction took: ", end_time - init_time
            print "Speed: ", 1./(end_time - init_time)
            print "Time total: ", end_time-start_time
            
            
            print "min max output ", np.min(probImage), np.max(probImage)

