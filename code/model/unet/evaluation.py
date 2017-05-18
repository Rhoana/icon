import numpy as np
from scipy.ndimage.filters import maximum_filter
import fast64counter
import mahotas
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import glob
import os
import cPickle

def thin_boundaries(im, mask):
    im = im.copy()
    assert (np.all(im >= 0)), "Label images must be non-negative"

    # make sure image is not all zero
    if np.sum(im) == 0:
       im[:] = 1.0
       im[0,:] = 2.0

    # repeatedly expand regions by one pixel until the background is gone
    while (im[mask] == 0).sum() > 0:
        zeros = (im == 0)
        im[zeros] = maximum_filter(im, 3)[zeros]

    # make sure image is not constant to avoid zero division
    if len(np.unique(im))==1:
        im[0,:] = 5
    return im

def Rand(pair, gt, pred, alpha):
    '''Parameterized Rand score

    Arguments are pairwise fractions, ground truth fractions, and prediction
    fractions.

    Equation 3 from Arganda-Carreras et al., 2015
    alpha = 0 is Rand-Split, alpha = 1 is Rand-Merge

    '''

    return np.sum(pair ** 2) / (alpha * np.sum(gt ** 2) +
                                (1.0 - alpha) * np.sum(pred ** 2))

def VI(pair, gt, pred, alpha):
    ''' Parameterized VI score

    Arguments are pairwise fractions, ground truth fractions, and prediction
    fractions.

    Equation 6 from Arganda-Carreras et al., 2015
    alpha = 0 is VI-Split, alpha = 1 is VI-Merge
    '''

    pair_entropy = - np.sum(pair * np.log(pair))
    gt_entropy = - np.sum(gt * np.log(gt))
    pred_entropy = - np.sum(pred * np.log(pred))
    mutual_information = gt_entropy + pred_entropy - pair_entropy

    return mutual_information / ((1.0 - alpha) * gt_entropy + alpha * pred_entropy)

def segmentation_metrics(ground_truth, prediction, seq=False):
    '''Computes adjusted FRand and VI between ground_truth and prediction.

    Metrics from: Crowdsourcing the creation of image segmentation algorithms
    for connectomics, Arganda-Carreras, et al., 2015, Frontiers in Neuroanatomy

    ground_truth - correct labels
    prediction - predicted labels

    Boundaries (label == 0) in prediction are thinned until gone, then are
    masked to foreground (label > 0) in ground_truth.

    Return value is ((FRand, FRand_split, FRand_merge), (VI, VI_split, VI_merge)).

    If seq is True, then it is assumed that the ground_truth and prediction are
    sequences that should be processed elementwise.

    '''

    # make non-sequences into sequences to simplify the code below
    if not seq:
        ground_truth = [ground_truth]
        prediction = [prediction]

    counter_pairwise = fast64counter.ValueCountInt64()
    counter_gt = fast64counter.ValueCountInt64()
    counter_pred = fast64counter.ValueCountInt64()

    for gt, pred in zip(ground_truth, prediction):
        mask = (gt > 0)
        pred = thin_boundaries(pred, mask)
        gt = gt[mask].astype(np.int32)
        pred = pred[mask].astype(np.int32)
        counter_pairwise.add_values_pair32(gt, pred)
        counter_gt.add_values_32(gt)
        counter_pred.add_values_32(pred)

    # fetch counts
    frac_pairwise = counter_pairwise.get_counts()[1]
    frac_gt = counter_gt.get_counts()[1]
    frac_pred = counter_pred.get_counts()[1]

    # normalize to probabilities
    frac_pairwise = frac_pairwise.astype(np.double) / frac_pairwise.sum()
    frac_gt = frac_gt.astype(np.double) / frac_gt.sum()
    frac_pred = frac_pred.astype(np.double) / frac_pred.sum()

    alphas = {'F-score': 0.5, 'split': 0.0, 'merge': 1.0}

    Rand_scores = {k: Rand(frac_pairwise, frac_gt, frac_pred, v) for k, v in alphas.items()}
    VI_scores = {k: VI(frac_pairwise, frac_gt, frac_pred, v) for k, v in alphas.items()}

    return {'Rand': Rand_scores, 'VI': VI_scores}


# Just doing one, so the interface is easier for the network training
# And yes that means I should refactor the function above... when I have time
def quick_Rand(gt, pred, seq=False):
    counter_pairwise = fast64counter.ValueCountInt64()
    counter_gt = fast64counter.ValueCountInt64()
    counter_pred = fast64counter.ValueCountInt64()

    mask = (gt > 0)
    pred = thin_boundaries(pred, mask)
    gt = gt[mask].astype(np.int32)
    pred = pred[mask].astype(np.int32)
    counter_pairwise.add_values_pair32(gt, pred)
    counter_gt.add_values_32(gt)
    counter_pred.add_values_32(pred)

    # fetch counts
    frac_pairwise = counter_pairwise.get_counts()[1]
    frac_gt = counter_gt.get_counts()[1]
    frac_pred = counter_pred.get_counts()[1]

    #print 'frac_pairwise:', frac_pairwise
    #print 'frac_gt:', frac_gt
    #print 'frac_pred:', frac_pred

    # normalize to probabilities
    frac_pairwise = frac_pairwise.astype(np.double) / frac_pairwise.sum()
    frac_gt = frac_gt.astype(np.double) / frac_gt.sum()
    frac_pred = frac_pred.astype(np.double) / frac_pred.sum()

    return Rand(frac_pairwise, frac_gt, frac_pred, 0.5)

def Rand_membrane_prob(im_pred, im_gt):
    Rand_score = []
    for thresh in np.arange(0,1,0.05):
        # white regions, black boundaries
        im_seg = im_pred>thresh
        # connected components
        seeds, nr_regions = mahotas.label(im_seg)
        result = quick_Rand(im_gt, seeds)        
        Rand_score.append(result)

    return np.max(Rand_score)

def run_evaluation_boundary_predictions(network_name):
    pathPrefix = './AC4_small/'
    img_gt_search_string = pathPrefix + 'labels/*.tif'
    img_pred_search_string = pathPrefix + 'boundaryProbabilities/'+network_name+'/*.tif'

    img_files_gt = sorted( glob.glob( img_gt_search_string ) )
    img_files_pred = sorted( glob.glob( img_pred_search_string ) )

    allVI = []
    allVI_split = []
    allVI_merge = []

    allRand = []
    allRand_split = []
    allRand_merge = []

    for i in xrange(np.shape(img_files_pred)[0]):
        print img_files_pred[i]
        im_gt = mahotas.imread(img_files_gt[i])
        im_pred = mahotas.imread(img_files_pred[i])
        im_pred = im_pred / 255.0

        VI_score = []
        VI_score_split = []
        VI_score_merge = []

        Rand_score = []
        Rand_score_split = []
        Rand_score_merge = []
    
        start_time = time.clock()

        for thresh in np.arange(0,1,0.05):
            # white regions, black boundaries
            im_seg = im_pred>thresh
            # connected components
            seeds, nr_regions = mahotas.label(im_seg)
            
            result = segmentation_metrics(im_gt, seeds, seq=False)   
            
            VI_score.append(result['VI']['F-score'])
            VI_score_split.append(result['VI']['split'])
            VI_score_merge.append(result['VI']['merge'])

            Rand_score.append(result['Rand']['F-score'])
            Rand_score_split.append(result['Rand']['split'])
            Rand_score_merge.append(result['Rand']['merge'])

        print "This took in seconds: ", time.clock() - start_time

        allVI.append(VI_score)
        allVI_split.append(VI_score_split)
        allVI_merge.append(VI_score_merge)

        allRand.append(Rand_score)
        allRand_split.append(Rand_score_split)
        allRand_merge.append(Rand_score_merge)
        
    with open(pathPrefix+network_name+'.pkl', 'wb') as file:
        cPickle.dump((allVI, allVI_split, allVI_merge, allRand, allRand_split, allRand_merge), file)
    

    # for i in xrange(len(allVI)):
    #     plt.plot(np.arange(0,1,0.05), allVI[i], 'g', alpha=0.5)
    # plt.plot(np.arange(0,1,0.05), np.mean(allVI, axis=0), 'r')
    # plt.show()
    
def run_evaluation_segmentations3D():
    # first test how to convert a great boundary segmentation quickly into 3d objects
    pathPrefix = './AC4/'
    img_gt_search_string = pathPrefix + 'labels/*.tif'
    img_pred_search_string = pathPrefix + 'boundaryProbabilities/IDSIA/*.tif'

    img_files_gt = sorted( glob.glob( img_gt_search_string ) )
    img_files_pred = sorted( glob.glob( img_pred_search_string ) )
    
    s = 100
    img_gt_volume = np.zeros((1024,1024,s))
    img_pred_volume = np.zeros((1024,1024,s))

    for i in xrange(s):
        print img_files_gt[i]
        # read image
        img_gt = mahotas.imread(img_files_gt[i])
        img_gt_volume[:,:,i] = img_gt
        # compute gradient to get perfect segmentation
        img_gt = np.gradient(img_gt)
        img_gt = np.sqrt(img_gt[0]**2 + img_gt[1]**2)
        #img_gt = mahotas.morph.erode(img_gt == 0)
        img_pred_volume[:,:,i] = img_gt == 0


    all_VI = []
    for i in xrange(20):
        print i
        if i>0:
            for j in xrange(s):
                img_pred_volume[:,:,j] = mahotas.morph.erode(img_pred_volume[:,:,j]>0)

    # connected component labeling
    print "labeling"
    seeds, nr_objects = mahotas.label(img_pred_volume)
    # compute scores
    print "computing metric"
    result = segmentation_metrics(img_gt_volume, seeds, seq=False)   
    print result
    all_VI.append(result['VI']['F-score'])
    return seeds

def plot_evaluations():
    pathPrefix = './AC4_small/'
    search_string = pathPrefix + '*.pkl'
    files = sorted( glob.glob( search_string ) )

    for i in xrange(np.shape(files)[0]):
        with open(files[i], 'r') as file:
            allVI, allVI_split, allVI_merge, allRand, allRand_split, allRand_merge = cPickle.load(file)
            # for ii in xrange(len(allVI)):
            #     plt.plot(np.arange(0,1,0.05), allVI[ii], colors[i]+'--', alpha=0.5)
            plt.plot(np.arange(0,1,0.05), np.mean(allRand, axis=0), label=files[i])
            #print "VI: ", files[i], np.max(np.mean(allVI, axis=0))
            print "Rand:", files[i], np.max(np.mean(allRand, axis=0))
    plt.title("Rand_info comparison - higher is better, bounded by 1")
    plt.xlabel("Threshold")
    plt.ylabel("Rand_info")
    plt.legend(loc="upper left")
    plt.show()


    # for i in xrange(np.shape(files)[0]):
    #     with open(files[i], 'r') as file:
    #         allVI, allVI_split, allVI_merge, allRand, allRand_split, allRand_merge = cPickle.load(file)
    #         # for ii in xrange(len(allVI)):
    #         #     plt.plot(allVI_split[ii], allVI_merge[ii], colors[i]+'--', alpha=0.5)
    #         plt.plot(np.mean(allVI_split, axis=0), np.mean(allVI_merge, axis=0), colors[i], label=files[i])
    # plt.xlabel("VI_split")
    # plt.ylabel("VI_merge")
    # #plt.legend()
    # plt.show()

    
    

if __name__=="__main__":
#   seeds = run_evaluation_segmentations3D()
    network_names = [os.path.basename(p[:-1]) for p in glob.glob('AC4_small/boundaryProbabilities/*/')]

    for name in network_names:
        if not os.path.exists('AC4_small/'+name+'.pkl'):
            print name, "is new"
            seeds = run_evaluation_boundary_predictions(name)
        else:
            print name, "is already done"
    
    plot_evaluations()

