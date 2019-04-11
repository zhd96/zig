import tensorflow as tf
import numpy as np
import scipy.stats as ss
import scipy.special as ssp
#from layers import FullLayer

def poissonDecoding(Y_real, spl_values, bin_len=181):
    spl_values = np.maximum(spl_values, 1e-6);
    poi_decode = np.zeros(Y_real.shape[0]);
    poi_mean_decode = np.zeros(Y_real.shape[0]);
    poi_lik = np.zeros(Y_real.shape[0]);
    #bin_len = np.shape(spl_values)[0]+1;
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    poi_lik_mat = np.zeros((Y_real.shape[0], len(hd_bins_use)));
    for ii in range(Y_real.shape[0]):
        loglik = (Y_real[ii]*np.log(spl_values) - spl_values).sum(axis=1);
        poi_decode[ii] = hd_bins_use[loglik.argmax()];
        lik = np.exp(loglik - loglik.max());
        lik /= sum(lik);
        poi_lik[ii] = lik.max();
        poi_lik_mat[ii] = lik;
        poi_mean_decode[ii] = np.arctan2((np.sin(hd_bins_use*np.pi/180)*lik).sum(), 
                   (np.cos(hd_bins_use*np.pi/180)*lik).sum())*180/np.pi;
    
    return poi_mean_decode, poi_decode, poi_lik_mat

def bernoulliDecoding(Y_real_use, thresh, spl_values, bin_len=181):
    #spl_values = np.maximum(spl_values, 1e-6);
    #Y_real = (Y_real_use>thresh);
    Y_real = (Y_real_use!=0);
    ber_decode = np.zeros(Y_real.shape[0]);
    ber_mean_decode = np.zeros(Y_real.shape[0]);
    ber_lik = np.zeros(Y_real.shape[0]);
    #bin_len = np.shape(spl_values)[0]+1;
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    ber_lik_mat = np.zeros((Y_real.shape[0], len(hd_bins_use)));
    for ii in range(Y_real.shape[0]):
        loglik = (Y_real[ii]*np.log(spl_values)+(1-Y_real[ii])*np.log(1-spl_values)).sum(axis=1);
        ber_decode[ii] = hd_bins_use[loglik.argmax()];
        lik = np.exp(loglik - loglik.max());
        lik /= sum(lik);
        ber_lik[ii] = lik.max();
        ber_lik_mat[ii] = lik;
        ber_mean_decode[ii] = np.arctan2((np.sin(hd_bins_use*np.pi/180)*lik).sum(), 
                   (np.cos(hd_bins_use*np.pi/180)*lik).sum())*180/np.pi;
    
    return ber_mean_decode, ber_decode, ber_lik_mat

def sngDecoding(Y_real, spl_values, sng_gamma, sng_k, sng_loc, bin_len=181):
    spl_p = (1-np.exp(-sng_gamma*spl_values))*(1-2e-6)+1e-6;
    spl_theta = (spl_values/spl_p - sng_loc)/sng_k;
    sng_decode = np.zeros(Y_real.shape[0]);
    sng_mean_decode = np.zeros(Y_real.shape[0]);
    sng_lik = np.zeros(Y_real.shape[0]);
    #bin_len = np.shape(spl_values)[0]+1;
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    sng_lik_mat = np.zeros((Y_real.shape[0], len(hd_bins_use)));
    for ii in range(Y_real.shape[0]):
        eq0_loglik =  - sng_gamma*spl_values;
        neq0_loglik = -(Y_real[ii]-sng_loc)/spl_theta - sng_k*np.log(spl_theta) + np.log(spl_p);
        loglik = (eq0_loglik*(Y_real[ii]==0) + neq0_loglik*(Y_real[ii]!=0)).sum(axis=1)
        sng_decode[ii] = hd_bins_use[loglik.argmax()]
        lik = np.exp(loglik-loglik.max());
        lik /= sum(lik);
        sng_lik_mat[ii] = lik;
        sng_lik[ii] = lik.max();
        sng_mean_decode[ii] = np.arctan2((np.sin(hd_bins_use*np.pi/180)*lik).sum(), 
                   (np.cos(hd_bins_use*np.pi/180)*lik).sum())*180/np.pi;      
    
    return sng_mean_decode, sng_decode, sng_lik_mat

def gammaDecoding(Y_real_use, spl_k, spl_theta, factor=1e-7, bin_len=181):
    Y_real = (Y_real_use+factor).copy();
    sng_decode = np.zeros(Y_real.shape[0]);
    sng_mean_decode = np.zeros(Y_real.shape[0]);
    sng_lik = np.zeros(Y_real.shape[0]);
    #bin_len = np.shape(spl_values)[0]+1;
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    sng_lik_mat = np.zeros((Y_real.shape[0], len(hd_bins_use)));
    for ii in range(Y_real.shape[0]):
        loglik = -spl_k*np.log(spl_theta)-(1/spl_theta)*Y_real[ii]+(spl_k-1)*np.log(Y_real[ii])-ssp.loggamma(spl_k);
        loglik = loglik.sum(axis=1)
        sng_decode[ii] = hd_bins_use[loglik.argmax()]
        lik = np.exp(loglik-loglik.max());
        lik /= sum(lik);
        sng_lik_mat[ii] = lik;
        sng_lik[ii] = lik.max();
        sng_mean_decode[ii] = np.arctan2((np.sin(hd_bins_use*np.pi/180)*lik).sum(), 
                   (np.cos(hd_bins_use*np.pi/180)*lik).sum())*180/np.pi;      
    
    return sng_mean_decode, sng_decode, sng_lik_mat

def sngRlxDecoding(Y_real, spl_theta, spl_p, sng_k, sng_loc, bin_len=181):
    sng_decode = np.zeros(Y_real.shape[0]);
    sng_mean_decode = np.zeros(Y_real.shape[0]);
    sng_lik = np.zeros(Y_real.shape[0]);
    #bin_len = np.shape(spl_values)[0]+1;
    hd_bins = np.linspace(-180,180,bin_len)
    hd_bins_use = hd_bins[1:]
    sng_lik_mat = np.zeros((Y_real.shape[0], len(hd_bins_use)));
    for ii in range(Y_real.shape[0]):
        eq0_loglik =  np.log(1-spl_p);
        neq0_loglik = -(Y_real[ii]-sng_loc)/spl_theta - sng_k*np.log(spl_theta) + np.log(spl_p);
        loglik = (eq0_loglik*(Y_real[ii]==0) + neq0_loglik*(Y_real[ii]!=0)).sum(axis=1)
        sng_decode[ii] = hd_bins_use[loglik.argmax()]
        lik = np.exp(loglik-loglik.max());
        lik /= sum(lik);
        sng_lik_mat[ii] = lik;
        sng_lik[ii] = lik.max();
        sng_mean_decode[ii] = np.arctan2((np.sin(hd_bins_use*np.pi/180)*lik).sum(), 
                   (np.cos(hd_bins_use*np.pi/180)*lik).sum())*180/np.pi;      
    
    return sng_mean_decode, sng_decode, sng_lik_mat
    
## CI analysis
def getHdr(lik_mat_param, hd_bins_use, conf_level, ii):
    lik = lik_mat_param[ii];
    lik_sort = -np.sort(-lik);
    thresh = lik_sort[sum(np.cumsum(lik_sort)<conf_level)];
    return hd_bins_use[lik>=thresh]

def getHdrCoverage(Y_real, hd, hd_bins_use, poi_lik_mat, conf_level= 1-1e-2):
    hd_left = np.zeros(Y_real.shape[0]);
    bin_size = hd_bins_use[1] - hd_bins_use[0];
    for ii in range(Y_real.shape[0]):
        hd_left[ii] = hd_bins_use[int(np.floor((hd[ii]-hd_bins_use[0])/bin_size))];

    poi_hdr = [];
    for ii in range(Y_real.shape[0]):
        poi_hdr.append(getHdr(poi_lik_mat[:,:],hd_bins_use,conf_level,ii));

    poi_hdr_cover = np.zeros(Y_real.shape[0])
    for ii in range(Y_real.shape[0]):
        poi_hdr_cover[ii] = (np.sum(hd_left[ii]+bin_size == poi_hdr[ii])>0)
        
    return poi_hdr, poi_hdr_cover

def getCICoverage(Y_real, hd, hd_bins_use, poi_lik_mat, poi_mean_decode, conf_level= 1-1e-2):
    poi_conf_interval = np.zeros((Y_real.shape[0], 2));
    for ii in range(Y_real.shape[0]):
        #poi_conf_interval[ii] = getCIOld(poi_lik_mat[:,:],hd_bins_use,poi_mean_decode[:],1-1e-2, ii);
        poi_conf_interval[ii] = getCI(poi_lik_mat[:,:],hd_bins_use,poi_mean_decode[:],conf_level, ii);
    
    def cover_rate(mean_decode_param, gt, ii):
        if ((gt[ii]>mean_decode_param[ii,0]) and (gt[ii]<=mean_decode_param[ii,1])):
            return True
        elif mean_decode_param[ii,1]>180:
            if gt[ii]<=mean_decode_param[ii,1]-360:
                return True
        elif mean_decode_param[ii,0]<=-180:
            if gt[ii]>mean_decode_param[ii,0]+360:
                return True            
        return False

    poi_conf_cover = np.zeros((Y_real.shape[0]));
    for ii in range(Y_real.shape[0]):
        poi_conf_cover[ii] = cover_rate(poi_conf_interval,hd,ii);
    
    return poi_conf_interval, poi_conf_cover

#def getCI(lik_mat_param, hd_bins_use, mean_decode_param, conf_level, ii):
#    lik = lik_mat_param[ii];
#    #lik /= sum(lik);
#    conf_level = conf_level*sum(lik);#np.minimum(conf_level, sum(lik));
#    lik3 = np.tile(lik, 3);
#    hd3 = np.tile(hd_bins_use, 3);
#    bin_size = hd_bins_use[1] - hd_bins_use[0];
#    conf_interval = [hd_bins_use[0],hd_bins_use[-1]];
#    interval = [len(hd_bins_use), 2*len(hd_bins_use)-1];
#    left = int(np.ceil((mean_decode_param[ii]-hd_bins_use[0])/bin_size))+len(hd_bins_use)-1;
#    right = left+1;
#    k = 0;
#    while k < len(hd_bins_use)/2:
#        if lik3[(left-k+1):(right+k+1)].sum()>=conf_level:
#            conf_interval = [hd3[left-k],hd3[right+k]];
#            interval = [left-k, right+k];
#            break;
#        else:
#            k+=1;
#    if interval[0]<len(hd_bins_use):
#        conf_interval[0] -= 360;
#    elif interval[1]>2*len(hd_bins_use)-1:
#        conf_interval[1] += 360;
#    return conf_interval

def getCI(lik_mat_param, hd_bins_use, mean_decode_param, conf_level, ii):
    lik = lik_mat_param[ii];
    #lik /= sum(lik);
    conf_level = conf_level*sum(lik);#np.minimum(conf_level, sum(lik));
    lik3 = np.tile(lik, 3);
    hd3 = np.tile(hd_bins_use, 3);
    bin_size = hd_bins_use[1] - hd_bins_use[0];
    conf_interval = [hd_bins_use[0]-bin_size,hd_bins_use[-1]];
    interval = [len(hd_bins_use), 2*len(hd_bins_use)-1];
    left = int(np.ceil((mean_decode_param[ii]-hd_bins_use[0])/bin_size))+len(hd_bins_use)-1;
    right = left+1;
    k = 0;

    while k < len(hd_bins_use)/2:
        if lik3[(right-k):(right+k+1)].sum()>=conf_level:
            conf_interval = [hd3[left-k],hd3[right+k]];
            interval = [left-k, right+k];
            break;
        else:
            k+=1;
    if interval[0]<len(hd_bins_use):
        conf_interval[0] -= 360;
    elif interval[1]>2*len(hd_bins_use)-1:
        conf_interval[1] += 360;
    return conf_interval

def getCIOld(lik_mat_param, hd_bins_use, mean_decode_param, conf_level, ii):
    lik = lik_mat_param[ii];
    #lik /= sum(lik);
    conf_level = conf_level*sum(lik);#np.minimum(conf_level, sum(lik));
    bin_size = hd_bins_use[1] - hd_bins_use[0];
    conf_interval = [hd_bins_use[0],hd_bins_use[-1]];
    
    #print(conf_level)
    left = int(np.floor((mean_decode_param[ii]-hd_bins_use[0])/bin_size));
    right = left+1;
    if left < len(hd_bins_use)/2:
        if lik[:(right+left+1)].sum()==conf_level:
            conf_interval = [hd_bins_use[0], hd_bins_use[right+left]];
        elif lik[:(right+left+1)].sum()>conf_level:
            k=0;
            while k<=left:
                if lik[(left-k):(right+k+1)].sum()>=conf_level:
                    conf_interval = [hd_bins_use[left-k], hd_bins_use[right+k]];
                    break #return conf_interval
                else:
                    k+=1;
        else:
            temp = lik[:(right+left+1)].sum();
            #print(temp)
            k = 1;
            while k<=np.minimum(len(lik)-right-left-1, (len(lik)-right-left)/2):
                if temp+lik[(right+left+1):(right+left+k+1)].sum()+lik[(-k):].sum()>=conf_level:
                    conf_interval = [hd_bins_use[-k]-360, hd_bins_use[right+left+k]];
                    break #return conf_interval
                else:
                    k+=1;
            #print(left)
            #print(right)
            #print(k)
    if left>=90:
        if lik[(left-len(lik)+right+1):].sum()==conf_level:
            conf_interval = [hd_bins_use[left-len(lik)+right+1], hd_bins_use[-1]];
        elif lik[(left-len(lik)+right+1):].sum()>conf_level:
            k=0;
            while k<=len(lik)-1-right:
                if lik[(left-k):(right+k+1)].sum()>=conf_level:
                    conf_interval = [hd_bins_use[left-k], hd_bins_use[right+k]];
                    break #return conf_interval
                else:
                    k+=1;
        else:
            temp = lik[(left-len(lik)+right+1):].sum();
            k = 1;
            while k<=np.minimum(right+left-len(lik)+1, (right+left-len(lik)+2)/2):
                if temp+lik[(left-len(lik)+right-k+1):(left-len(lik)+right+1)].sum()+lik[:k].sum()>=conf_level:
                    conf_interval = [hd_bins_use[left-len(lik)+right-k+1], hd_bins_use[k-1]+360];
                    break #return conf_interval
                else:
                    k+=1;
    conf_interval[0] -=bin_size;
    #conf_interval[1] +=bin_size;
    return conf_interval

def getVaryCI(Y_real, hd, hd_bins_use, sngr_lik_mat, poi_lik_mat, ber_lik_mat, gam_lik_mat, poi_mean_decode, ber_mean_decode, sngr_mean_decode, gam_mean_decode, nTrain, nValid, nTest, conf_level_list=list(np.linspace(0.9,0.999,10))[:-1] + [1-1e-2, 1-5e-3, 1-1e-3]):
    width_ber_med = [];
    width_poi_med = [];
    width_sngr_med = [];
    width_gam_med = [];
    
    width_ber_mean = [];
    width_poi_mean = [];
    width_sngr_mean = [];
    width_gam_mean = [];
    
    conf_rate_ber = [];
    conf_rate_sngr = [];
    conf_rate_poi = [];
    conf_rate_gam = [];
    
    for conf_level in conf_level_list:
        poi_conf_interval, poi_conf_cover = getCICoverage(Y_real, hd, hd_bins_use, poi_lik_mat, poi_mean_decode, conf_level=conf_level);
        ber_conf_interval, ber_conf_cover = getCICoverage(Y_real, hd, hd_bins_use, ber_lik_mat, ber_mean_decode, conf_level=conf_level);
        sngr_conf_interval, sngr_conf_cover = getCICoverage(Y_real, hd, hd_bins_use, sngr_lik_mat, sngr_mean_decode, conf_level=conf_level);
        gam_conf_interval, gam_conf_cover = getCICoverage(Y_real, hd, hd_bins_use, gam_lik_mat, gam_mean_decode, conf_level=conf_level);
        
        ber_conf_len = ber_conf_interval[:,1] - ber_conf_interval[:,0]
        sngr_conf_len = sngr_conf_interval[:,1] - sngr_conf_interval[:,0]
        poi_conf_len = poi_conf_interval[:,1] - poi_conf_interval[:,0]
        gam_conf_len = gam_conf_interval[:,1] - gam_conf_interval[:,0]
        
        width_ber_med.append([np.median(ber_conf_len[:nTrain]), np.median(ber_conf_len[(nTrain+nValid):])]);
        width_poi_med.append([np.median(poi_conf_len[:nTrain]), np.median(poi_conf_len[(nTrain+nValid):])]);
        width_sngr_med.append([np.median(sngr_conf_len[:nTrain]), np.median(sngr_conf_len[(nTrain+nValid):])]);
        width_gam_med.append([np.median(gam_conf_len[:nTrain]), np.median(gam_conf_len[(nTrain+nValid):])]);
        
        width_ber_mean.append([np.mean(ber_conf_len[:nTrain]), np.mean(ber_conf_len[(nTrain+nValid):])]);
        width_poi_mean.append([np.mean(poi_conf_len[:nTrain]), np.mean(poi_conf_len[(nTrain+nValid):])]);
        width_sngr_mean.append([np.mean(sngr_conf_len[:nTrain]), np.mean(sngr_conf_len[(nTrain+nValid):])]);
        width_gam_mean.append([np.mean(gam_conf_len[:nTrain]), np.mean(gam_conf_len[(nTrain+nValid):])]);
        
        conf_rate_ber.append([ber_conf_cover[:nTrain].sum()/nTrain, ber_conf_cover[(nTrain+nValid):].sum()/nTest])
        conf_rate_sngr.append([sngr_conf_cover[:nTrain].sum()/nTrain, sngr_conf_cover[(nTrain+nValid):].sum()/nTest])
        conf_rate_poi.append([poi_conf_cover[:nTrain].sum()/nTrain, poi_conf_cover[(nTrain+nValid):].sum()/nTest])
        conf_rate_gam.append([gam_conf_cover[:nTrain].sum()/nTrain, gam_conf_cover[(nTrain+nValid):].sum()/nTest])
    
    width_med = [width_sngr_med, width_poi_med, width_ber_med,width_gam_med];
    width_mean = [width_sngr_mean, width_poi_mean, width_ber_mean,width_gam_mean];
    conf_rate = [conf_rate_sngr, conf_rate_poi, conf_rate_ber,conf_rate_gam];
    
    conf_rate = np.array(conf_rate);
    width_mean = np.array(width_mean);
    width_med = np.array(width_med);
    
    return width_med, width_mean, conf_rate

def getVaryHdr(Y_real, hd, hd_bins_use, sngr_lik_mat, poi_lik_mat, ber_lik_mat, gam_lik_mat, nTrain, nValid, nTest, conf_level_list=list(np.linspace(0.9,0.999,10))[:-1] + [1-1e-2, 1-5e-3, 1-1e-3]):
    width_ber_med = [];
    width_poi_med = [];
    width_sngr_med = [];
    width_gam_med = [];
    
    width_ber_mean = [];
    width_poi_mean = [];
    width_sngr_mean = [];
    width_gam_mean = [];
    
    conf_rate_ber = [];
    conf_rate_sngr = [];
    conf_rate_poi = [];
    conf_rate_gam = [];
    
    for conf_level in conf_level_list:
        poi_hdr, poi_hdr_cover = getHdrCoverage(Y_real, hd, hd_bins_use, poi_lik_mat, conf_level=conf_level);
        ber_hdr, ber_hdr_cover = getHdrCoverage(Y_real, hd, hd_bins_use, ber_lik_mat, conf_level=conf_level);
        sngr_hdr, sngr_hdr_cover = getHdrCoverage(Y_real, hd, hd_bins_use, sngr_lik_mat, conf_level=conf_level);
        gam_hdr, gam_hdr_cover = getHdrCoverage(Y_real, hd, hd_bins_use, gam_lik_mat, conf_level=conf_level);

        ber_conf_len = [];
        sngr_conf_len = [];
        poi_conf_len = [];
        gam_conf_len = [];
        for m in poi_hdr:
            poi_conf_len.append(len(m));
        for m in ber_hdr:
            ber_conf_len.append(len(m));
        for m in sngr_hdr:
            sngr_conf_len.append(len(m));
        for m in gam_hdr:
            gam_conf_len.append(len(m));
            
        poi_conf_len = np.array(poi_conf_len);
        ber_conf_len = np.array(ber_conf_len);
        sngr_conf_len = np.array(sngr_conf_len);
        gam_conf_len = np.array(gam_conf_len);
        
        width_ber_med.append([np.median(ber_conf_len[:nTrain]), np.median(ber_conf_len[(nTrain+nValid):])]);
        width_poi_med.append([np.median(poi_conf_len[:nTrain]), np.median(poi_conf_len[(nTrain+nValid):])]);
        width_sngr_med.append([np.median(sngr_conf_len[:nTrain]), np.median(sngr_conf_len[(nTrain+nValid):])]);
        width_gam_med.append([np.median(gam_conf_len[:nTrain]), np.median(gam_conf_len[(nTrain+nValid):])]);
        
        width_ber_mean.append([np.mean(ber_conf_len[:nTrain]), np.mean(ber_conf_len[(nTrain+nValid):])]);
        width_poi_mean.append([np.mean(poi_conf_len[:nTrain]), np.mean(poi_conf_len[(nTrain+nValid):])]);
        width_sngr_mean.append([np.mean(sngr_conf_len[:nTrain]), np.mean(sngr_conf_len[(nTrain+nValid):])]);
        width_gam_mean.append([np.mean(gam_conf_len[:nTrain]), np.mean(gam_conf_len[(nTrain+nValid):])]);
        
        conf_rate_ber.append([ber_hdr_cover[:nTrain].sum()/nTrain, ber_hdr_cover[(nTrain+nValid):].sum()/nTest])
        conf_rate_sngr.append([sngr_hdr_cover[:nTrain].sum()/nTrain, sngr_hdr_cover[(nTrain+nValid):].sum()/nTest])
        conf_rate_poi.append([poi_hdr_cover[:nTrain].sum()/nTrain, poi_hdr_cover[(nTrain+nValid):].sum()/nTest])
        conf_rate_gam.append([gam_hdr_cover[:nTrain].sum()/nTrain, gam_hdr_cover[(nTrain+nValid):].sum()/nTest])
    
    width_med = [width_sngr_med, width_poi_med, width_ber_med,width_gam_med];
    width_mean = [width_sngr_mean, width_poi_mean, width_ber_mean,width_gam_mean];
    conf_rate = [conf_rate_sngr, conf_rate_poi, conf_rate_ber,conf_rate_gam];

    conf_rate = np.array(conf_rate);
    width_mean = np.array(width_mean);
    width_med = np.array(width_med);
    
    return width_med, width_mean, conf_rate

def error(decode_rlt, hd):
    decode_rlt_pri = decode_rlt.copy();
    decode_rlt_error = decode_rlt_pri - hd;
    decode_rlt_pri[decode_rlt_error<-180]+=360;
    decode_rlt_error[decode_rlt_error<-180] += 360;
    decode_rlt_pri[decode_rlt_error>180]-=360;
    decode_rlt_error[decode_rlt_error>180] -= 360;
    return decode_rlt_pri, decode_rlt_error

def pair_t(x, y):
    return np.mean(x-y)*np.sqrt(x.shape[0])/(x-y).std()

def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)
