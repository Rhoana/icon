
import theano
import theano.tensor as T

def rectified_linear(p):
        return T.maximum(0.0, p)

