## Neural network layers ##
# ==============================================================================
import tensorflow as tf

DTYPE = tf.float32

#tf.set_random_seed(666)

class FullLayer():
    """
    """
    def __init__(self):
        """
        """
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'sigmoid' : tf.nn.sigmoid, 'tanh': tf.nn.tanh}
        
    def __call__(self, Input, nodes, nl='softplus', scope=None, name='out',
                 initializer=None,
                 b_initializer=None, isconst=False):
        """
        """
        nonlinearity = self.nl_dict[nl]
        input_dim = Input.get_shape()[-1]
        
        if b_initializer is None:
            b_initializer = tf.zeros([nodes]);
        if initializer is None:
            initializer = tf.orthogonal_initializer();
            
        with tf.variable_scope(scope):
            if isconst:
                weights = tf.get_variable('weights', dtype=DTYPE, initializer=initializer);
            else:
                weights = tf.get_variable('weights', [input_dim, nodes], dtype=DTYPE, initializer=initializer);
                
            biases = tf.get_variable('biases', dtype=DTYPE, initializer=b_initializer)
            full = nonlinearity(tf.matmul(Input, weights) + biases,
                                name=name)
                    
        return full
