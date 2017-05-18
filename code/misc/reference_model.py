import cPickle
import gzip
import os
import sys
import time

import numpy
import numpy as np

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from generateTrainValTestData import generate_experiment_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
from utils import tile_raster_images
from classifyImage import generate_patch_data_rows
from mlp import MLP_deep
import multiprocessing

import smtplib
import getpass

def rectified_linear(p):
    #return T.switch(p > 0., p, 0.0)
    return T.switch(p > 0., p, 0.0)
    #return p * (p > 0.)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=rectified_linear):
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
        self.activation = activation


    def get_output(self, data):
        return self.activation(T.dot(data, self.W) + self.b)
        #return rectified_linear(T.dot(data, self.W) + self.b)

    def visualize_filters(self):
        W = self.W.eval()
        patchSize = np.sqrt(W.shape[0])

        return tile_raster_images(X=W.T, img_shape=(patchSize, patchSize), tile_shape=(20,20), tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True)

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,dropout_rate, W=None, b=None, activation=rectified_linear):
        super(DropoutHiddenLayer, self).__init__(rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, activation=activation)
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP_dropout(object):
    def __init__(self, rng, input, n_in=None, n_hidden=None, n_out=None, dropout_rate=None, fileName=None, activation=rectified_linear):
        self.hiddenLayers = []
        self.dropoutLayers = []
        self.params = []
        self.trainingCost = []
        self.validationError = []
        self.dropout_rate = dropout_rate
        self.n_in = n_in
        self.n_hidden = n_hidden

        if fileName is None:
            next_input = input
            next_dropout_input = _dropout_from_layer(rng, input, p=self.dropout_rate[0])
            next_n_in = n_in

            for n_h, drop_r in zip(n_hidden, dropout_rate[:-1]):
                hl_dropout = DropoutHiddenLayer(rng=rng, input=next_dropout_input,
                                                n_in = next_n_in, n_out=n_h, dropout_rate=drop_r,
                                                activation=activation)
                hl = HiddenLayer(rng=rng, input=next_input,
                                 n_in=next_n_in, n_out=n_h,
                                 activation=activation, W=hl_dropout.W * (1-drop_r),
                                 b=hl_dropout.b)

                next_input = hl.output
                next_dropout_input = hl_dropout.output
                next_n_in = n_h
                self.dropoutLayers.append(hl_dropout)
                self.hiddenLayers.append(hl)
                self.params += hl_dropout.params


            self.logRegressionLayer_dropout = LogisticRegression(
                input=self.dropoutLayers[-1].output,
                n_in=n_hidden[-1],
                n_out=n_out)


            self.params += self.logRegressionLayer_dropout.params

        else:
            with open(fileName, 'r') as file:
                self.dropoutLayers, self.logRegressionLayer_dropout, self.trainingCost, self.validationError, self.dropout_rate, self.n_in, self.n_hidden = cPickle.load(file)

            next_input = input
            next_n_in = self.n_in

            counter=0
            for hl_dropout in self.dropoutLayers:
                hl = HiddenLayer(rng=rng, input=next_input,
                                 n_in=next_n_in, n_out=self.n_hidden[counter],
                                 activation=activation, W=hl_dropout.W * (1-self.dropout_rate[counter]),
                                 b=hl_dropout.b)

                next_input = hl.output
                next_n_in = self.n_hidden[counter]
                counter += 1
                self.hiddenLayers.append(hl)

            for hl in self.dropoutLayers:
                self.params += hl.params
            self.params += self.logRegressionLayer_dropout.params

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            W = self.logRegressionLayer_dropout.W * (1-self.dropout_rate[-1]),
            b = self.logRegressionLayer_dropout.b,
            n_in=self.n_hidden[-1],
            n_out=n_out)

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.dropout_negative_log_likelihood = self.logRegressionLayer_dropout.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors
        self.get_p_y_given_x = self.logRegressionLayer.get_p_y_given_x

    def save_MLP(self, filename):
        with open(filename, 'wb') as file:
            cPickle.dump((self.dropoutLayers, self.logRegressionLayer_dropout, self.trainingCost, self.validationError, self.dropout_rate, self.n_in, self.n_hidden), file)

    def classify_image(self, img, normMean=None, norm_std=None):
        data_type = T.matrix('data')
        start_time = time.clock()

        row_range = 1
        img = normalizeImage(img)
        imSize = np.shape(img)
        membraneProbabilities = np.zeros(np.shape(img))
        patchSize = np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

        def applyNetwork(data):
            for hl in self.hiddenLayers:
                data = hl.get_output(data)
            return self.get_p_y_given_x(data)

        data_shared = theano.shared(np.float32(np.zeros((imSize[0]*row_range,patchSize**2))))

        #classify = theano.function(inputs=[data_type], outputs=applyNetwork(data_type))
        classify = theano.function(inputs=[], outputs=applyNetwork(data_type),
                                   givens={data_type: data_shared}, on_unused_input='ignore')

        for row in xrange(0,1024,row_range):
            if row%100 == 0:
                print row
            data = generate_patch_data_rows(img, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize, data_mean=normMean, data_std=norm_std)
            data_shared.set_value(np.float32(data))
            result = classify() #classify(data)
            membraneProbabilities[row,:] = result[:,1]

        end_time = time.clock()
        total_time = (end_time - start_time)
        print >> sys.stderr, ('Running time: ' +
                              '%.2fm' % (total_time / 60.))

        return np.array(membraneProbabilities)


def send_email(gmail_pwd, msg):
    to = 'vkaynig@seas.harvard.edu'
    gmail_user = 'vkaynig@gmail.com'
    smtpserver = smtplib.SMTP("smtp.gmail.com",587)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    header = 'To:' + to + '\n' + 'From: ' + gmail_user + '\n' + 'Subject:DNN update \n'
    msg = header + '\n' + msg + '\n\n'
    smtpserver.sendmail(gmail_user, to, msg)
    smtpserver.close()


def train_mlp(learning_rate=0.01, n_epochs=10, batch_size=500, n_hidden=[500], patchSize=39, train_samples=10000, val_samples=10000, test_samples=10000, doResample=False, validation_frequency = 50, dropout_rate=0.0, activation=rectified_linear, doEmailUpdate=False, momentum=0.9):

    def adadelta_updates(parameters,gradients,rho,eps):
        # create variables to store intermediate updates
        gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters ]
        deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters ]
        # calculates the new "average" delta for the next iteration
        gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in zip(gradients_sq,gradients) ]

        # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
        deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in zip(deltas_sq,gradients_sq_new,gradients) ]

        # calculates the new "average" deltas for the next step.
        deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in zip(deltas_sq,deltas) ]

        # Prepare it as a list f
        gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
        deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
        parameters_updates = [ (p,p - d) for p,d in zip(parameters,deltas) ]
        return gradient_sq_updates + deltas_sq_updates + parameters_updates

    def gradient_updates_momentum(cost, params, learning_rate, momentum):
        updates = []
        for param in params:
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate*param_update))
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        return updates

    if doEmailUpdate:
        gmail_pwd = getpass.getpass()

    rng = numpy.random.RandomState(1234)

    data, norm_mean, norm_std, grayImages, labelImages, maskImages = generate_experiment_data_supervised(purpose='train', nsamples=train_samples, patchSize=patchSize, balanceRate=0.5, data_mean=0.5, data_std=1.0)
    train_set_x, train_set_y = shared_dataset(data, doCastLabels=True)

    data = generate_experiment_data_supervised(purpose='validate', nsamples=val_samples, patchSize=patchSize, balanceRate=0.5, data_mean=norm_mean, data_std=norm_std)[0]
    valid_set_x, valid_set_y = shared_dataset(data, doCastLabels=True)

    data = generate_experiment_data_supervised(purpose='test', nsamples=test_samples, patchSize=patchSize, balanceRate=0.5, data_mean=norm_mean, data_std=norm_std)[0]
    test_set_x, test_set_y = shared_dataset(data, doCastLabels=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_samples / batch_size
    n_valid_batches = val_samples / 1000
    n_test_batches = test_samples / 1000

    learning_rate_shared = theano.shared(np.float32(learning_rate))
    momentum_shared = theano.shared(np.float32(momentum))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    lr = T.scalar('learning_rate')
    m = T.scalar('momentum')

    # construct the MLP class
    classifier = MLP_dropout(rng=rng, input=x, n_in=patchSize**2,
                             n_hidden=n_hidden, n_out=2, dropout_rate=dropout_rate, activation=activation)


    cost = classifier.dropout_negative_log_likelihood(y)

    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                     y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    #SGD
    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - lr * gparam))

    #updates = adadelta_updates(classifier.params, gparams, lr, 0.000001)
#    updates = gradient_updates_momentum(cost, classifier.params, lr, m)


    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                lr: learning_rate_shared})#,
                #m: momentum_shared})

    print '... training'

    best_validation_loss = numpy.inf
    best_iter = 0
    decrease_epoch = 1
    decrease_patience = 1
    test_score = 0.

    start_time = time.clock()

    epoch = 0
    done_looping = False

    # start pool for data
    print "Starting worker."
    pool = multiprocessing.Pool(processes=1)
    futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])

    while (epoch < n_epochs) and (not done_looping):
        minibatch_avg_costs = []
        epoch = epoch + 1

        if epoch % 10 == 0:
            classifier.save_MLP('train_progress.pkl')

        if doResample and epoch>1:
            print "Waiting for data."
            data = futureData.get()
            print "GOT NEW DATA"
            train_set_x.set_value(np.float32(data[0]))
            train_set_y.set_value(np.int32(data[1]))
            futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])
#            try:
#                data = futureData.get(timeout=1)
#                print "GOT NEW DATA"
#                train_set_x.set_value(np.float32(data[0]))
#                train_set_y.set_value(np.int32(data[1]))
#                futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,True, 'train', train_samples, patchSize, 0.5, norm_mean, 1.0]])
#            except multiprocessing.TimeoutError:
#                print "TIMEOUT, TRAINING ANOTHER ROUND WITH CURRENT DATA"
#                pass
#


        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_costs.append(train_model(minibatch_index))
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                msg = 'epoch %i, minibatch %i/%i, training error %f, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_costs[-1], this_validation_loss * 100.)

                print(msg)

                classifier.trainingCost.append(minibatch_avg_costs[-1])
                classifier.validationError.append(this_validation_loss*100)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    classifier.save_MLP('best_so_far.pkl')
                    print "New best score!"
                    if doEmailUpdate:
                        send_email(gmail_pwd, msg)
                    # test it on the test set
                    #test_losses = [test_model(i) for i
                    #               in xrange(n_test_batches)]
                    #test_score = numpy.mean(test_losses)
                    #
                    #print(('epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))

    pool.close()
    pool.join()
    print "Pool closed."

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return classifier



if __name__ == '__main__':
    import mahotas
    import matplotlib.pyplot as plt
    image = mahotas.imread('train-input0099.tif')

    x = T.matrix('x')

#    test2 = MLP_dropout(rng=numpy.random.RandomState(1), input=x, n_in = 49**2, n_hidden=[4000]*5, n_out=2,
#                        dropout_rate=0.0, fileName = 'tmp.pkl')

    test2 = MLP_dropout(rng=numpy.random.RandomState(1), input=x, n_out=2, fileName = 'tmp.pkl')

#    prob = test2.classify_image(img=image, normMean=0.5, norm_std=1.0)
#    plt.imshow(1-prob)
#    plt.show()
#
#    mahotas.imsave('tmp_output_8.png', np.uint8((1-prob)*255))
#
    hl = test2.hiddenLayers[0]
    plt.imshow(hl.visualize_filters())
    plt.show()

    plt.plot(np.array(test2.trainingCost), label='training')
    plt.plot(np.array(test2.validationError), label='validation')
    plt.legend()
    plt.show()

    print "best validation score: ", test2.validationError[-1]
