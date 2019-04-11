import tensorflow as tf
import numpy as np
import scipy.stats as ss
from layers import FullLayer
DTYPE = tf.float32

def poissonEncoding(Y, X, yDim, xDim, learning_rate):
    DTYPE = tf.float32
    #yDim = tf.shape(Y)[1];
    #xDim = tf.shape(X)[1];

    fullyConnectedLayer = FullLayer();

    rangeRate1 = 1/tf.sqrt(tf.cast(xDim, DTYPE));
    
    with tf.variable_scope("poi_rate_nn", reuse=tf.AUTO_REUSE):
        full = fullyConnectedLayer(X, yDim, nl='linear', scope='output',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));

    rate = tf.exp(full);
    entropy_loss = tf.reduce_sum(Y*tf.log(rate) - rate); #-tf.reduce_sum((Y-rate)**2)#

    with tf.variable_scope("poi_adam", reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-entropy_loss);
    
    return optimizer, entropy_loss, rate

def bernoulliEncoding(Y, X, yDim, xDim, learning_rate):
    DTYPE = tf.float32
    #yDim = tf.shape(Y)[1];
    #xDim = tf.shape(X)[1];

    fullyConnectedLayer = FullLayer();

    rangeRate1 = 1/tf.sqrt(tf.cast(xDim, DTYPE));

    with tf.variable_scope("ber_rate_nn", reuse=tf.AUTO_REUSE):
        full = fullyConnectedLayer(X, yDim, nl='linear', scope='output',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));

    temp = tf.exp(full);
    rate = temp/(1+temp);
    entropy_loss = tf.reduce_sum(Y*full - tf.log(1+temp)); #-tf.reduce_sum((Y-rate)**2)#

    with tf.variable_scope("ber_adam", reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-entropy_loss);
    
    return optimizer, entropy_loss, rate

def sngEncoding(Y, X, yDim, xDim, learning_rate, factor, gamma_np):
    DTYPE = tf.float32
    #yDim = tf.shape(Y)[1];
    #xDim = tf.shape(X)[1];

    fullyConnectedLayer = FullLayer();

    rangeRate1 = 1/tf.sqrt(tf.cast(xDim, DTYPE));

    with tf.variable_scope("sng_rate_nn", reuse=tf.AUTO_REUSE):
        full = fullyConnectedLayer(X, yDim, nl='linear', scope='output',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));
    
    #mean_temp, var_temp = tf.nn.moments(Y, axes=0);
    #gamma_temp = mean_temp/var_temp;
    #gamma_temp = tf.reduce_mean(Y, axis=0)/#Y.mean(axis=0)/Y.var(axis=0)
    gamma_temp = tf.cast(tf.log(gamma_np), DTYPE)
    
    params = dict([ ("loggamma", gamma_temp),
                ("loc", tf.convert_to_tensor(factor, DTYPE))
                ])
    with tf.variable_scope("sng_obsmodel", reuse=tf.AUTO_REUSE):
        if "loggamma" in params:
            loggamma = tf.get_variable('loggamma', initializer=tf.cast(params["loggamma"], DTYPE), dtype=DTYPE)
        else:
            loggamma = tf.get_variable('loggamma', initializer=tf.cast(tf.zeros(yDim), DTYPE), dtype=DTYPE)
        gamma = tf.exp(loggamma) + 1e-7;

        if "logk" in params:
            logk = tf.get_variable('logk', initializer=tf.cast(params["logk"], DTYPE), dtype=DTYPE)
        else:
            logk = tf.get_variable('logk', initializer=tf.cast(tf.zeros(yDim), DTYPE), dtype=DTYPE)
        k = tf.exp(logk) + 1e-7;

        if "loc" in params:
            loc = tf.cast(params["loc"], DTYPE);
        else:
            loc = tf.cast(tf.zeros(yDim), DTYPE);
        #self.loc = tf.minimum(self.loc, tf.cast(params["min_y"], DTYPE)-1e-6);
    rate = tf.exp(full);
    
    # now compute the entropy loss
    Nsamps = tf.shape(Y)[0];
    mask = tf.not_equal(Y, tf.zeros_like(Y));
    k_NTxD = tf.reshape(tf.tile(k, [Nsamps]), [Nsamps, yDim]);
    loc_NTxD = tf.reshape(tf.tile(loc, [Nsamps]), [Nsamps, yDim]);
    gamma_rate = rate*gamma;

    y_temp = tf.boolean_mask(Y, mask);
    r_temp = tf.boolean_mask(rate, mask);
    gr_temp = tf.boolean_mask(gamma_rate, mask);
    p_temp = 1-tf.exp(-gr_temp);
    k_NTxD = tf.boolean_mask(k_NTxD, mask);
    loc_NTxD = tf.boolean_mask(loc_NTxD, mask);
    r_temp = r_temp - loc_NTxD*p_temp;

    p_temp = p_temp*(1-2e-6) + 1e-6; 
    r_temp = r_temp + 1e-6; 

    LY1 = tf.reduce_sum((k_NTxD+1)*tf.log(p_temp) - k_NTxD*tf.log(r_temp) - (y_temp-loc_NTxD)*k_NTxD*p_temp/r_temp);
    LY2 = tf.reduce_sum((k_NTxD*tf.log(k_NTxD)-tf.lgamma(k_NTxD))+(k_NTxD-1)*tf.log(y_temp-loc_NTxD));

    gr_temp = tf.boolean_mask(gamma_rate, ~mask);
    LY3 = -tf.reduce_sum(gr_temp);
    
    entropy_loss = LY1+LY2+LY3;
    with tf.variable_scope("sng_adam", reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-entropy_loss);
        
    return optimizer, entropy_loss, rate, k, gamma, loc

def sngRlxEncoding(Y, X, yDim, xDim, learning_rate, factor):
    DTYPE = tf.float32
    #yDim = tf.shape(Y)[1];
    #xDim = tf.shape(X)[1];

    fullyConnectedLayer = FullLayer();

    rangeRate1 = 1/tf.sqrt(tf.cast(xDim, DTYPE));

    with tf.variable_scope("sngrlx_rate_nn", reuse=tf.AUTO_REUSE):
        
        full_theta = fullyConnectedLayer(X, yDim, nl='linear', scope='output_theta',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));
        full_p = fullyConnectedLayer(X, yDim, nl='linear', scope='output_p',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));
    
    params = dict([("loc", tf.convert_to_tensor(factor, DTYPE))]);
    
    with tf.variable_scope("sngrlx_obsmodel", reuse=tf.AUTO_REUSE):
        if "logk" in params:
            logk = tf.get_variable('logk', initializer=tf.cast(params["logk"], DTYPE), dtype=DTYPE)
        else:
            logk = tf.get_variable('logk', initializer=tf.cast(tf.zeros(yDim), DTYPE), dtype=DTYPE)
        k = tf.exp(logk) + 1e-7;

        if "loc" in params:
            loc = tf.cast(params["loc"], DTYPE);
        else:
            loc = tf.cast(tf.zeros(yDim), DTYPE);
        #self.loc = tf.minimum(self.loc, tf.cast(params["min_y"], DTYPE)-1e-6);
    #theta = tf.nn.relu(full_theta);
    theta = tf.exp(full_theta);
    p = tf.exp(full_p)/(1+tf.exp(full_p));
    
    # now compute the entropy loss
    Nsamps = tf.shape(Y)[0];
    mask = tf.not_equal(Y, tf.zeros_like(Y));
    k_NTxD = tf.reshape(tf.tile(k, [Nsamps]), [Nsamps, yDim]);
    loc_NTxD = tf.reshape(tf.tile(loc, [Nsamps]), [Nsamps, yDim]);

    y_temp = tf.boolean_mask(Y, mask);
    r_temp = tf.boolean_mask(theta, mask);
    p_temp = tf.boolean_mask(p, mask);
    k_NTxD = tf.boolean_mask(k_NTxD, mask);
    loc_NTxD = tf.boolean_mask(loc_NTxD, mask);

    p_temp = p_temp*(1-2e-6) + 1e-6; 
    r_temp = r_temp + 1e-6; 

    LY1 = tf.reduce_sum(tf.log(p_temp) - k_NTxD*tf.log(r_temp) - (y_temp-loc_NTxD)/r_temp);
    LY2 = tf.reduce_sum(-tf.lgamma(k_NTxD)+(k_NTxD-1)*tf.log(y_temp-loc_NTxD));
    gr_temp = tf.boolean_mask(p, ~mask);
    LY3 = tf.reduce_sum(tf.log(1-gr_temp));

    entropy_loss = LY1+LY2+LY3;
    with tf.variable_scope("sngrlx_adam", reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-entropy_loss);
    
    rate = (theta*k + loc)*p;
    return optimizer, entropy_loss, theta, k, p, loc, rate

def sngRREncoding(Y, X, yDim, xDim, learning_rate, factor):
    DTYPE = tf.float32
    #yDim = tf.shape(Y)[1];
    #xDim = tf.shape(X)[1];

    fullyConnectedLayer = FullLayer();

    rangeRate1 = 1/tf.sqrt(tf.cast(xDim, DTYPE));

    with tf.variable_scope("sngrr_rate_nn", reuse=tf.AUTO_REUSE):
        
        full_theta = fullyConnectedLayer(X, yDim, nl='linear', scope='output_theta',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));
        full_p = fullyConnectedLayer(X, yDim, nl='linear', scope='output_p',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));
        full_k = fullyConnectedLayer(X, yDim, nl='linear', scope='output_k',
                initializer=tf.random_uniform_initializer(minval=-rangeRate1,maxval=rangeRate1));
    
    params = dict([("loc", tf.convert_to_tensor(factor, DTYPE))]);
    
    with tf.variable_scope("sngrr_obsmodel", reuse=tf.AUTO_REUSE):
        #if "logk" in params:
        #    logk = tf.get_variable('logk', initializer=tf.cast(params["logk"], DTYPE), dtype=DTYPE)
        #else:
        #    logk = tf.get_variable('logk', initializer=tf.cast(tf.zeros(yDim), DTYPE), dtype=DTYPE)
        #k = tf.exp(logk) + 1e-7;

        if "loc" in params:
            loc = tf.cast(params["loc"], DTYPE);
        else:
            loc = tf.cast(tf.zeros(yDim), DTYPE);
        #self.loc = tf.minimum(self.loc, tf.cast(params["min_y"], DTYPE)-1e-6);
    k = tf.exp(full_k) + 1e-7;
    theta = tf.exp(full_theta);
    p = tf.exp(full_p)/(1+tf.exp(full_p));
    
    # now compute the entropy loss
    Nsamps = tf.shape(Y)[0];
    mask = tf.not_equal(Y, tf.zeros_like(Y));
    #k_NTxD = tf.reshape(tf.tile(k, [Nsamps]), [Nsamps, yDim]);
    loc_NTxD = tf.reshape(tf.tile(loc, [Nsamps]), [Nsamps, yDim]);

    y_temp = tf.boolean_mask(Y, mask);
    r_temp = tf.boolean_mask(theta, mask);
    p_temp = tf.boolean_mask(p, mask);
    k_NTxD = tf.boolean_mask(k, mask);
    loc_NTxD = tf.boolean_mask(loc_NTxD, mask);

    p_temp = p_temp*(1-2e-6) + 1e-6; 
    r_temp = r_temp + 1e-6; 

    LY1 = tf.reduce_sum(tf.log(p_temp) - k_NTxD*tf.log(r_temp) - (y_temp-loc_NTxD)/r_temp);
    LY2 = tf.reduce_sum(-tf.lgamma(k_NTxD)+(k_NTxD-1)*tf.log(y_temp-loc_NTxD));
    gr_temp = tf.boolean_mask(p, ~mask);
    LY3 = tf.reduce_sum(tf.log(1-gr_temp));

    entropy_loss = LY1+LY2+LY3;
    with tf.variable_scope("sngrr_adam", reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-entropy_loss);
    
    rate = (theta*k + loc)*p;
    return optimizer, entropy_loss, theta, k, p, loc, rate

## MLE encoding

def poissonMleEncoding(Y_train, hd_train, bin_len=181):

    hd_bins = np.linspace(-180,180,bin_len);
    tuning_curve = np.zeros((len(hd_bins)-1, Y_train.shape[1]));
    for ii in range(len(hd_bins)-1):
        data_pos = ((hd_train>=hd_bins[ii])*(hd_train<=hd_bins[ii+1]));
        tuning_curve[ii,:] = Y_train[data_pos,:].mean(axis=0);
    
    spl_values = np.maximum(tuning_curve,1e-6);
    
    return spl_values

def bernoulliMleEncoding(Y_train, hd_train, thresh=0, bin_len=181):
    
    hd_bins = np.linspace(-180,180,bin_len);
    empirical_p = np.zeros((len(hd_bins)-1, Y_train.shape[1]));
    for ii in range(len(hd_bins)-1):
        data_pos = ((hd_train>=hd_bins[ii])*(hd_train<=hd_bins[ii+1]));
        if data_pos.sum()==0:
            empirical_p[ii,:]=0;
        else:
            empirical_p[ii,:] = ((Y_train[data_pos,:]>thresh).sum(axis=0))/data_pos.sum();
            
    spl_values = np.clip(empirical_p, 1e-6, 1-1e-6);#empirical_p*(1-2e-6)+1e-6;
            
    return spl_values

def sngMleEncoding(Y_train, hd_train, smin=0, bin_len=181):
    
    hd_bins = np.linspace(-180,180,bin_len);
    empirical_p = np.zeros((len(hd_bins)-1, Y_train.shape[1]));
    sng_k = np.zeros((len(hd_bins)-1, Y_train.shape[1]));
    sng_loc = np.zeros((len(hd_bins)-1, Y_train.shape[1]));
    spl_theta_values = np.zeros((len(hd_bins)-1, Y_train.shape[1]));
    
    for ii in range(len(hd_bins)-1):
        data_pos = ((hd_train>=hd_bins[ii])*(hd_train<=hd_bins[ii+1]));
        if data_pos.sum()>0:
            empirical_p[ii,:] = ((Y_train[data_pos,:]>0).sum(axis=0))/data_pos.sum();
        
        # fit gamma part
        for jj in range(Y_train.shape[1]):
            rlt = Y_train[data_pos,jj];
            if (rlt>0).sum()>2:
                sng_k[ii,jj], sng_loc[ii,jj], spl_theta_values[ii,jj] = ss.gamma.fit(rlt[rlt>0], floc=smin);
    
    spl_p_values = np.clip(empirical_p, 1e-6, 1-1e-6);#empirical_p*(1-2e-6)+1e-6;
    spl_theta_values = np.maximum(spl_theta_values, 1e-6);
    spl_values = spl_p_values*(sng_k*spl_theta_values + sng_loc);
    
    return spl_values, spl_p_values, spl_theta_values, sng_k, sng_loc

def computeReduMSE(x,y):
    return ((x-y)**2).mean()/((np.var(y, axis=0)).mean())

def samplePoisson(yfit, seed=888):
    np.random.seed(seed);
    p_poi = (1-np.exp(-yfit));
    y_poisson = np.random.poisson(yfit, size=yfit.shape)
    return y_poisson, p_poi

def sampleBernoulli(yfit, seed=888):
    np.random.seed(seed);
    y_bernoulli = np.random.binomial(1, yfit, size=yfit.shape)
    return y_bernoulli

def sampleSng(yfit, sng_k, sng_gamma, sng_loc, seed=888):
    np.random.seed(seed);
    p_sng = (1-np.exp(-sng_gamma*yfit));
    theta_sng = (yfit/p_sng - sng_loc)/sng_k;
    y_gamma = np.zeros(yfit.shape);
    for ii in range(yfit.shape[0]):
        y_gamma[ii] = np.random.binomial(1,p_sng[ii])*(sng_loc+np.random.gamma(scale=theta_sng[ii], 
                                                                               shape=sng_k));
    return y_gamma, p_sng

def sampleSngRlx(pfit, sng_k, sng_theta, sng_loc, seed=888):
    np.random.seed(seed);
    y_gamma = np.zeros(pfit.shape);
    for ii in range(pfit.shape[0]):
        y_gamma[ii] = np.random.binomial(1,pfit[ii])*(sng_loc+np.random.gamma(scale=sng_theta[ii], 
                                                                               shape=sng_k));
    return y_gamma

def sampleSngRR(pfit, sng_k, sng_theta, sng_loc, seed=888):
    np.random.seed(seed);
    y_gamma = np.zeros(pfit.shape);
    for ii in range(pfit.shape[0]):
        y_gamma[ii] = np.random.binomial(1,pfit[ii])*(sng_loc+np.random.gamma(scale=sng_theta[ii], 
                                                                               shape=sng_k[ii]));
    return y_gamma

def computePreferDg(Y_real, hd):
    prefer_dg = np.zeros((Y_real.shape[1]));
    for ii in range(Y_real.shape[1]):
        prefer_dg[ii] = hd[np.where(Y_real[:,ii] == Y_real[:,ii].max())[0]]        
    order = np.argsort(prefer_dg);
    return prefer_dg, order

def fourierBasis(hd, gen_nodes):
    hd_angle = hd*np.pi/180;
    return np.array([[np.cos(hd_angle*(ii+1)),np.sin(hd_angle*(ii+1))] for ii in range(gen_nodes)]).reshape(gen_nodes*2, hd_angle.shape[0]).T;
    
## examples

def poissonRunner(Y_train, Y_valid, hd_train, hd_valid, hd_test, n_epochs=800, learning_rate=1e-2, gen_nodes=10, bin_len=181):
    #nTrain = 14600;
    #nTest = Y_real.shape[0] - nTrain;
    #Y_train = Y_real[:nTrain,:];
    #Y_test = Y_real[nTrain:,:];
    #hd_train = hd[:nTrain];
    #hd_test = hd[nTrain:];
    nTrain = Y_train.shape[0];
    nValid = Y_valid.shape[0];
    hd_angle_train = fourierBasis(hd_train, gen_nodes);
    hd_angle_valid = fourierBasis(hd_valid, gen_nodes);
    hd_angle_test = fourierBasis(hd_test, gen_nodes);

    #n_epochs = 800;
    cost = [];
    test_cost = [];
    yDim = Y_train.shape[1];
    xDim = gen_nodes*2;
    #gen_nodes = 60;
    #learning_rate = 1e-2;
    
    Y = tf.placeholder(DTYPE, [None, yDim], name='Y');
    X = tf.placeholder(DTYPE, [None, xDim], name='X');

    optimizer, entropy_loss, rate = poissonEncoding(Y, X, yDim, xDim, learning_rate);

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver();
    
    for ie in range(n_epochs):
        #cost_ib = sess.run([sgvb.cost], feed_dict={sgvb.Y: batch_y})
        _, cost_ib = sess.run((optimizer, entropy_loss), feed_dict={Y: Y_train, X: hd_angle_train})
        avgcost_ib = cost_ib/Y_train.shape[0];
        print("Epoch:", '%d' % (ie+1), "cost=", "{:.9f}".format(avgcost_ib))
        cost.append(avgcost_ib);

        avgcost_ib = sess.run(entropy_loss, feed_dict={Y: Y_valid, X: hd_angle_valid});
        print("Epoch:", '%d' % (ie+1), "test cost=", "{:.9f}".format(avgcost_ib/nValid));
        test_cost.append(avgcost_ib/nValid)
        
        # save results
        if ie>0 and test_cost[-1]==max(test_cost):
            saver.save(sess, "./rlt/poi.ckpt");
    
    saver.restore(sess, "./rlt/poi.ckpt");
    
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    hd_bins_angle = fourierBasis(hd_bins_use, gen_nodes);
    spl_values = sess.run(rate, feed_dict={X: hd_bins_angle})
    yfit = sess.run(rate, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))})
    return spl_values, yfit, cost, test_cost

def bernoulliRunner(Y_train_use, Y_valid_use, hd_train, hd_valid, hd_test, thresh, n_epochs=800, learning_rate=1e-2, gen_nodes=10, bin_len=181):
    #nTrain = 14600;
    Y_train = (Y_train_use>thresh);
    Y_valid = (Y_valid_use>thresh);
    nTrain = Y_train.shape[0];
    nValid = Y_valid.shape[0];
    
    hd_angle_train = fourierBasis(hd_train, gen_nodes);
    hd_angle_valid = fourierBasis(hd_valid, gen_nodes);
    hd_angle_test = fourierBasis(hd_test, gen_nodes);

    #n_epochs = 800;
    cost = [];
    test_cost = [];
    yDim = Y_train.shape[1];
    xDim = gen_nodes*2;
    #gen_nodes = 60;
    #learning_rate = 1e-2;
    
    Y = tf.placeholder(DTYPE, [None, yDim], name='Y');
    X = tf.placeholder(DTYPE, [None, xDim], name='X');

    optimizer, entropy_loss, rate = bernoulliEncoding(Y, X, yDim, xDim, learning_rate);

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver();

    for ie in range(n_epochs):
        #cost_ib = sess.run([sgvb.cost], feed_dict={sgvb.Y: batch_y})
        _, cost_ib = sess.run((optimizer, entropy_loss), feed_dict={Y: Y_train, X: hd_angle_train})
        avgcost_ib = cost_ib/Y_train.shape[0];
        print("Epoch:", '%d' % (ie+1), "cost=", "{:.9f}".format(avgcost_ib))
        cost.append(avgcost_ib);

        avgcost_ib = sess.run(entropy_loss, feed_dict={Y: Y_valid, X: hd_angle_valid});
        print("Epoch:", '%d' % (ie+1), "test cost=", "{:.9f}".format(avgcost_ib/nValid));
        test_cost.append(avgcost_ib/nValid)
        
        # save results
        if ie>0 and test_cost[-1]==max(test_cost):
            saver.save(sess, "./rlt/ber.ckpt");
    
    saver.restore(sess, "./rlt/ber.ckpt");

    
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    hd_bins_angle = fourierBasis(hd_bins_use, gen_nodes);
    spl_values = sess.run(rate, feed_dict={X: hd_bins_angle})
    yfit = sess.run(rate, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))})
    return spl_values, yfit, cost, test_cost

def sngRunner(Y_train, Y_valid, hd_train, hd_valid, hd_test, factor, n_epochs=800, learning_rate=1e-2, gen_nodes=10, bin_len=181):
    #nTrain = 14600;
    nTrain = Y_train.shape[0];
    nValid = Y_valid.shape[0];
    
    hd_angle_train = fourierBasis(hd_train, gen_nodes);
    hd_angle_valid = fourierBasis(hd_valid, gen_nodes);
    hd_angle_test = fourierBasis(hd_test, gen_nodes);

    #n_epochs = 800;
    cost = [];
    test_cost = [];
    yDim = Y_train.shape[1];
    xDim = gen_nodes*2;
    #gen_nodes = 60;
    #learning_rate = 1e-2;
    #factor = np.ones(yDim)*0.875;
    gamma = Y_train.mean(axis=0)/Y_train.var(axis=0);
    
    Y = tf.placeholder(DTYPE, [None, yDim], name='Y');
    X = tf.placeholder(DTYPE, [None, xDim], name='X');

    optimizer, entropy_loss, rate, k, gamma, loc = sngEncoding(Y, X, yDim, xDim, learning_rate, factor, gamma);
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver();

    for ie in range(n_epochs):
        #cost_ib = sess.run([sgvb.cost], feed_dict={sgvb.Y: batch_y})
        _, cost_ib = sess.run((optimizer, entropy_loss), feed_dict={Y: Y_train, X: hd_angle_train})
        avgcost_ib = cost_ib/Y_train.shape[0];
        print("Epoch:", '%d' % (ie+1), "cost=", "{:.9f}".format(avgcost_ib))
        cost.append(avgcost_ib);

        avgcost_ib = sess.run(entropy_loss, feed_dict={Y: Y_valid, X: hd_angle_valid});
        print("Epoch:", '%d' % (ie+1), "test cost=", "{:.9f}".format(avgcost_ib/nValid));
        test_cost.append(avgcost_ib/nValid)
        
        # save results
        if ie>0 and test_cost[-1]==max(test_cost):
            saver.save(sess, "./rlt/sng.ckpt");
    
    saver.restore(sess, "./rlt/sng.ckpt");

    sng_k = sess.run(k);
    sng_gamma = sess.run(gamma);
    sng_loc = sess.run(loc);
    
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    hd_bins_angle = fourierBasis(hd_bins_use, gen_nodes);
    
    spl_values = sess.run(rate, feed_dict={X: hd_bins_angle});
    spl_p_values = 1-np.exp(-sng_gamma*spl_values);
    spl_theta_values = ((spl_values/spl_p_values)-sng_loc)/sng_k;
    
    yfit = sess.run(rate, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    pfit = 1-np.exp(-sng_gamma*yfit);
    thetafit = ((yfit/pfit)-sng_loc)/sng_k;
    
    return spl_values, spl_p_values, spl_theta_values, sng_k, sng_gamma, sng_loc, yfit, pfit, thetafit, cost, test_cost

def sngRlxRunner(Y_train, Y_valid, hd_train, hd_valid, hd_test, factor, n_epochs=800, learning_rate=1e-2, gen_nodes=10, bin_len=181):
    #nTrain = 14600;
    nTrain = Y_train.shape[0];
    nValid = Y_valid.shape[0];
    
    hd_angle_train = fourierBasis(hd_train, gen_nodes);
    hd_angle_valid = fourierBasis(hd_valid, gen_nodes);
    hd_angle_test = fourierBasis(hd_test, gen_nodes);

    #n_epochs = 800;
    cost = [];
    test_cost = [];
    yDim = Y_train.shape[1];
    xDim = gen_nodes*2;
    #gen_nodes = 60;
    #learning_rate = 1e-2;
    #factor = np.ones(yDim)*0.875;
    
    Y = tf.placeholder(DTYPE, [None, yDim], name='Y');
    X = tf.placeholder(DTYPE, [None, xDim], name='X');

    optimizer, entropy_loss, theta, k, p, loc, rate = sngRlxEncoding(Y, X, yDim, xDim, learning_rate, factor);
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver();

    for ie in range(n_epochs):
        #cost_ib = sess.run([sgvb.cost], feed_dict={sgvb.Y: batch_y})
        _, cost_ib = sess.run((optimizer, entropy_loss), feed_dict={Y: Y_train, X: hd_angle_train})
        avgcost_ib = cost_ib/Y_train.shape[0];
        print("Epoch:", '%d' % (ie+1), "cost=", "{:.9f}".format(avgcost_ib))
        cost.append(avgcost_ib);

        avgcost_ib = sess.run(entropy_loss, feed_dict={Y: Y_valid, X: hd_angle_valid});
        print("Epoch:", '%d' % (ie+1), "test cost=", "{:.9f}".format(avgcost_ib/nValid));
        test_cost.append(avgcost_ib/nValid)
        
        # save results
        if ie>0 and test_cost[-1]==max(test_cost):
            saver.save(sess, "./rlt/sng_rlx.ckpt");
    
    saver.restore(sess, "./rlt/sng_rlx.ckpt");
    
    sng_k = sess.run(k);
    sng_loc = sess.run(loc);
    
    hd_bins = np.linspace(-180,180,bin_len);
    hd_bins_use = hd_bins[1:];
    hd_bins_angle = fourierBasis(hd_bins_use, gen_nodes);
    
    spl_values = sess.run(rate, feed_dict={X: hd_bins_angle});
    spl_p_values = sess.run(p, feed_dict={X: hd_bins_angle});
    spl_theta_values = sess.run(theta, feed_dict={X: hd_bins_angle});
    
    yfit = sess.run(rate, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    pfit = sess.run(p, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    thetafit = sess.run(theta, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    
    return spl_values, spl_p_values, spl_theta_values, sng_k, sng_loc, yfit, pfit, thetafit, cost, test_cost

def sngRRRunner(Y_train, Y_valid, hd_train, hd_valid, hd_test, factor, n_epochs=800, learning_rate=1e-2, gen_nodes=10, bin_len=181):
    #nTrain = 14600;
    nTrain = Y_train.shape[0];
    nValid = Y_valid.shape[0];
    
    hd_angle_train = fourierBasis(hd_train, gen_nodes);
    hd_angle_valid = fourierBasis(hd_valid, gen_nodes);
    hd_angle_test = fourierBasis(hd_test, gen_nodes);

    #n_epochs = 800;
    cost = [];
    test_cost = [];
    yDim = Y_train.shape[1];
    xDim = gen_nodes*2;
    #gen_nodes = 60;
    #learning_rate = 1e-2;
    #factor = np.ones(yDim)*0.875;
    
    Y = tf.placeholder(DTYPE, [None, yDim], name='Y');
    X = tf.placeholder(DTYPE, [None, xDim], name='X');

    optimizer, entropy_loss, theta, k, p, loc, rate = sngRREncoding(Y, X, yDim, xDim, learning_rate, factor);
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver();

    for ie in range(n_epochs):
        #cost_ib = sess.run([sgvb.cost], feed_dict={sgvb.Y: batch_y})
        _, cost_ib = sess.run((optimizer, entropy_loss), feed_dict={Y: Y_train, X: hd_angle_train})
        avgcost_ib = cost_ib/Y_train.shape[0];
        print("Epoch:", '%d' % (ie+1), "cost=", "{:.9f}".format(avgcost_ib))
        cost.append(avgcost_ib);

        avgcost_ib = sess.run(entropy_loss, feed_dict={Y: Y_valid, X: hd_angle_valid});
        print("Epoch:", '%d' % (ie+1), "test cost=", "{:.9f}".format(avgcost_ib/nValid));
        test_cost.append(avgcost_ib/nValid)
        
        # save results
        if ie>0 and test_cost[-1]==max(test_cost):
            saver.save(sess, "./rlt/sng_rr.ckpt");
    
    saver.restore(sess, "./rlt/sng_rr.ckpt");
    
    #sng_k = sess.run(k);
    sng_loc = sess.run(loc);
    
    hd_bins = np.linspace(-180,180,bin_len);
    hd_bins_use = hd_bins[1:];
    hd_bins_angle = fourierBasis(hd_bins_use, gen_nodes);
    
    spl_values = sess.run(rate, feed_dict={X: hd_bins_angle});
    spl_p_values = sess.run(p, feed_dict={X: hd_bins_angle});
    spl_theta_values = sess.run(theta, feed_dict={X: hd_bins_angle});
    spl_k_values = sess.run(k, feed_dict={X: hd_bins_angle});
    
    yfit = sess.run(rate, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    pfit = sess.run(p, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    kfit = sess.run(k, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    thetafit = sess.run(theta, feed_dict={X: np.vstack((hd_angle_train, hd_angle_valid, hd_angle_test))});
    
    return spl_values, spl_p_values, spl_theta_values, spl_k_values, sng_loc, yfit, pfit, thetafit, kfit, cost, test_cost

def fitmle(p, rlt):
    # fit spike mle
    p_fit = sum(rlt>0)/len(rlt);
    # fit gamma mle
    k_fit, loc_fit, theta_fit = ss.gamma.fit(rlt[rlt>0], loc=0)
    print("p fit" + str(p_fit))
    print("k fit" + str(k_fit))
    print("loc fit" + str(loc_fit))
    print("theta fit" + str(theta_fit))
    # plot
    fig = plt.figure(figsize=(4,4));
    plt.hist(rlt[rlt>0],bins=50,density=True);
    x = np.linspace(0, max(np.ceil(max(rlt)),0.5), 100);
    y1 = ss.gamma.pdf(x, scale=theta_fit, a=k_fit, loc=loc_fit);
    plt.plot(x, y1);
    plt.ylabel(str(p));
    return p_fit, k_fit, loc_fit, theta_fit, fig
