import time

n_epochs = 20
n_train_batches = 6366
validation_frequency = n_train_batches
max_train_time = 30
elapsed_time = 0
epoch = 0
start_time = time.clock()
while (epoch < n_epochs) and (elapsed_time < max_train_time):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
	elapsed_time = time.clock() - start_time
	iteration = (epoch - 1) * n_train_batches + minibatch_index
	if (iteration+1)%validation_frequency == 0:
           print iteration, epoch
