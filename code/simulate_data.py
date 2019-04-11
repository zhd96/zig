import numpy as np

def simfp(nid, tuning_curve, sns, stims):
### simulate from poisson distribution    

    np.random.seed(888+nid);
    noise_var = 1;
    time_interval = 1; # unit milisecond
    
    tau_d = 400;
    tau_r = 1;
    #A_c = np.array([[-(1/tau_d+1/tau_r), -1/(tau_d*tau_r)], [1, 0]]);
    e_ac = np.array([-1/tau_d, -1/tau_r]); #np.linalg.eig(A_c)[0]; # if AR(1) then tau_r=0
    lambda_1 = np.exp(e_ac[0]*time_interval) # decay how much in one time bin
    lambda_2 = np.exp(e_ac[1]*time_interval)
    gamma_1 = lambda_1 + lambda_2;
    gamma_2 = -1*lambda_1*lambda_2;
    g1 = 1/(np.power(2, time_interval*33.34/tau_d));
    #print(g1);
    
    # simulate bernoulli
    series_length = stims.shape[0];
    hd_bins_use = np.linspace(-180,180,181)[1:];
    temp = np.asarray(np.ceil((stims-hd_bins_use[0])/2), dtype=int)    
    temp = tuning_curve[temp,[nid]]/33;
    
    sigma = sns[nid];
    
    p = np.minimum(temp,1);
    spike_s = np.random.binomial(1, p, series_length);
    
    # simulate calcium decay
    cal = np.zeros(series_length);
    for ii in range(2, series_length):
        cal[ii] = gamma_1*cal[ii-1]+gamma_2*cal[ii-2]+spike_s[ii];
        
    # simulate noisy calcium decay
    noisy_cal = cal + np.random.normal(0, sigma*noise_var, series_length);
    
    # subsample step
    spike_s = spike_s.reshape(int(series_length/33),33).sum(axis=1);
    cal = cal[np.arange(0,series_length,33)];
    noisy_cal = noisy_cal[np.arange(0,series_length,33)];
    
    return spike_s, cal, noisy_cal

def simfnb(nid, tuning_curve, sns, stims, factors=2):
### simulate from negative binomial distribution    

    np.random.seed(888+nid);
    noise_var = 1;
    time_interval = 33; # unit milisecond
    
    tau_d = 400;
    tau_r = 1;
    #A_c = np.array([[-(1/tau_d+1/tau_r), -1/(tau_d*tau_r)], [1, 0]]);
    e_ac = np.array([-1/tau_d, -1/tau_r]); #np.linalg.eig(A_c)[0]; # if AR(1) then tau_r=0
    lambda_1 = np.exp(e_ac[0]*time_interval) # decay how much in one time bin
    lambda_2 = np.exp(e_ac[1]*time_interval)
    gamma_1 = lambda_1 + lambda_2;
    gamma_2 = -1*lambda_1*lambda_2;
    g1 = 1/(np.power(2, 33.34/tau_d));
    #print(g1);
    
    # simulate bernoulli
    series_length = stims.shape[0];
    hd_bins_use = np.linspace(-180,180,181)[1:];
    temp = np.asarray(np.ceil((stims-hd_bins_use[0])/2), dtype=int);
    temp = tuning_curve[temp,[nid]]#/33;
    temp+=1e-6;
    p = temp;
    
    sigma = sns[nid];
    
    if len(factors)>1:
        factor = factors[nid];
    else:
        factor = factors;
    spike_s = np.random.negative_binomial(temp/(factor-1), 1/factor, series_length);
    
    # simulate calcium decay
    cal = np.zeros(series_length);
    for ii in range(2, series_length):
        cal[ii] = gamma_1*cal[ii-1]+gamma_2*cal[ii-2]+spike_s[ii];
        
    # simulate noisy calcium decay
    noisy_cal = cal + np.random.normal(0, sigma*noise_var, series_length);
        
    return spike_s, cal, noisy_cal

