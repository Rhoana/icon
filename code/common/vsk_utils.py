import theano
import theano.tensor as T
import numpy
import gzip
import cPickle
#from partition_comparison import *

from classifyImage import *

#from dropout_MLP import *
import mahotas

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def shared_single_dataset(data_x, borrow=True):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x

def region_growing(labelImg):
    distances = mahotas.stretch(mahotas.distance(labelImg>0))
    surface = numpy.int32(distances.max() - distances)
    areas = mahotas.cwatershed(surface, labelImg)
    return areas

def remove_small_regions(labelImg, minRegionSize):
    regionSizes = mahotas.labeled.labeled_size(labelImg)
    deleteIds = numpy.where(regionSizes<minRegionSize)
    labelImg = mahotas.labeled.remove_regions(labelImg, deleteIds)
    return labelImg

def draw_frame(image, width=1, value=0):
    image[:width,:] = value
    image[-width:,:] = value
    image[:,:width] = value
    image[:,-width:] = value
    return image


def evaluate_classifier_vi_old(imName, imTrueName, thresholds=[0.5], classifier=None, classification=None):
    if classification==None:
        classification = classifyImage_MLP(imName, classifier, None, None, doThresh=False)

    #read the ground truth image
    imTrue = mahotas.imread(imTrueName)

    #compute variation of information
    imTrue = numpy.int32(imTrue.ravel())

    vi = []
    for thresh in thresholds:
        membraneLabels = classification>thresh

    #draw frame to to connect around image border
        membraneLabels = draw_frame(membraneLabels, 11, 1)

    #delete false postive detections inside cells
        membraneLabels, _ = mahotas.label(membraneLabels)
        membraneLabels = remove_small_regions(membraneLabels, 5000)

    #shrink membrane label
        labelImg, nrObjects = mahotas.label(membraneLabels==0)
        labelImg = region_growing(labelImg)
    
    #delete regions that are too small
        labelImg = remove_small_regions(labelImg, 30)
        labelImg = region_growing(labelImg)
        labelImg = draw_frame(labelImg,11,0)

        if numpy.median(labelImg) == 0:
            print 'Too bad to be evaluated!'
            return numpy.inf

        labelImg = labelImg.ravel()
        vi.append(variation_of_information(imTrue, labelImg))

    return numpy.min(vi)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        print "creating directory"
        print d
        os.makedirs(d)
        
if __name__ == '__main__':      
    imName = '/data/Verena/I00000_image.tif'
    imTrueName = '/data/Verena/I00000_label.tif'

    f = gzip.open('params.pkl.gz','rb', compresslevel=1)
    params,_,_ = cPickle.load(f)
    f.close()

    rng = numpy.random.RandomState(1234)
    x = T.matrix('x')
    '''
    classifier = MLP_dropout(rng=rng, input=x, n_in=21**2, hidden_layer_sizes=[1500,1000,500], n_out=2, dropout_input=0., dropout_hiddens=[0.,0.,0.], params = params)

    print evaluate_classifier_vi(imName, imTrueName, np.arange(0.1,0.9,0.01), classifier)
    '''
