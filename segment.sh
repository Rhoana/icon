cd code/model
#cd code/model/cnn
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python segment.py -m online
