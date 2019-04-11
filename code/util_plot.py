import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

## general plot functions
def plotTuningCurves(y, hd, yfit = None):
    hd_bins = np.linspace(-180,180,121);
    tuning_curve = np.zeros((len(hd_bins)-1, y.shape[1]));
    for ii in range(len(hd_bins)-1):
        data_pos = ((hd>=hd_bins[ii])*(hd<=hd_bins[ii+1]));
        tuning_curve[ii,:] = y[data_pos,:].mean(axis=0);
    
    if yfit is not None:
        tuning_curve2 = np.zeros((len(hd_bins)-1, y.shape[1]));
        for ii in range(len(hd_bins)-1):
            data_pos = ((hd>=hd_bins[ii])*(hd<=hd_bins[ii+1]));
            tuning_curve2[ii,:] = yfit[data_pos,:].mean(axis=0);        
    
    fig = plt.figure(figsize=(4,2*y.shape[-1]));
    for ii in range(y.shape[-1]):
        plt.subplot(y.shape[-1],1,ii+1);
        plt.plot(tuning_curve[:,ii],color='r');
        if yfit is not None:
            plt.plot(tuning_curve2[:,ii]);
        plt.ylabel(str(ii))
    plt.tight_layout();
    return fig

## encoding plot functions
def plotParams(params, titles, size):
    order = np.argsort(params[0].max(axis=0));
    prefer_dg = np.zeros(params[0].shape[1]);
    for ii in range(params[0].shape[1]):
        prefer_dg[ii] = np.where(params[0][:,ii] == params[0].max(axis=0)[ii])[0][0];
    order = np.argsort(prefer_dg)
    cmap='jet';
    if size-1==120:
        fig = plt.figure(figsize=(9,2*(params[0].shape[1])/(params[0].shape[0])));
    else:
        #fig = plt.figure(figsize=(12,3.6*(params[0].shape[1])/(params[0].shape[0])));
        fig = plt.figure(figsize=(12,4));
    ax = fig.add_subplot(111);
    ax.set_xlabel('Stimulus (degree $^{\circ}$)',fontsize=14,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax0 = fig.add_subplot(131);
    img0=ax0.imshow(params[0][:,order].T, cmap=cmap,aspect='auto');
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    plt.setp(ax0.get_xticklabels(), fontsize=14)
    plt.setp(ax0.get_yticklabels(), fontsize=14)
    ax0.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3,min_n_ticks=3,prune=None))
    ax0.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    if size-1==180:
        ax0.set_xticks((0,44,89,134,179))
        ax0.set_xticklabels((-178,-90,0,90,180))
        if ii ==0:
            ax0.legend(fontsize=14)
        
    else:
        ax0.set_xticks((29,59,89,119))
        ax0.set_xticklabels((-90,0,90,180))
    
    ax0.set_ylabel('Neurons', fontsize=14);
    ax0.set_title(titles[0], fontsize=14, fontweight='bold');
    divider = make_axes_locatable(ax0)
    #cax = divider.append_axes("right", size="1%", pad=0.1)
    cax = divider.new_horizontal(size="5%",pad=0.1);
    fig.add_axes(cax)
    if params[0].max()<1:
        cbar=fig.colorbar(img0, cax=cax,orientation='vertical',spacing='uniform',format="%.1f")
    else:
        cbar=fig.colorbar(img0, cax=cax,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=14) 
    tick_locator = ticker.MaxNLocator(nbins=5,prune="both")
    cbar.locator = tick_locator
    cbar.update_ticks()

    ax1 = fig.add_subplot(132,sharex=ax0, sharey=ax0);
    img1=ax1.imshow(params[1][:,order].T, cmap=cmap,aspect='auto');
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3,min_n_ticks=3,prune=None))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    if size-1==180:
        ax1.set_xticks((0,44,89,134,179))
        ax1.set_xticklabels((-178,-90,0,90,180))
        if ii ==0:
            ax1.legend(fontsize=14)
        
    else:
        ax1.set_xticks((29,59,89,119))
        ax1.set_xticklabels((-90,0,90,180))
    
    #ax1.set_ylabel('Neurons', fontsize=14);
    ax1.set_title(titles[1], fontsize=14, fontweight='bold');
    divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="1%", pad=0.1)
    cax = divider.new_horizontal(size="5%",pad=0.1);
    fig.add_axes(cax)
    if params[1].max()<1:
        cbar=fig.colorbar(img1, cax=cax,orientation='vertical',spacing='uniform',format="%.1f")
    else:
        cbar=fig.colorbar(img1, cax=cax,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=14) 
    tick_locator = ticker.MaxNLocator(nbins=5,prune="both")
    cbar.locator = tick_locator
    cbar.update_ticks()

    ax2 = fig.add_subplot(133,sharex=ax0,sharey=ax0);
    img2=ax2.imshow(params[2][:,order].T, cmap=cmap,aspect='auto');
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3,min_n_ticks=3,prune=None))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax2.set_title(titles[2], fontsize=14, fontweight='bold');
    divider = make_axes_locatable(ax2)
    #cax = divider.append_axes("right", size="1%", pad=0.1)
    cax = divider.new_horizontal(size="5%",pad=0.1);
    fig.add_axes(cax)
    if params[2].max()<1:
        cbar=fig.colorbar(img2, cax=cax,orientation='vertical',spacing='uniform',format="%.1f")
    else:
        cbar=fig.colorbar(img2, cax=cax,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=14) 
    cbar.set_ticks([0.1,0.3,0.5,0.7]);
    #tick_locator = ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune="both")
    #cbar.locator = tick_locator
    #cbar.update_ticks()

    plt.tight_layout();
    return fig

def plotSampleP(Y_real, p_sng):
    rg = (1-p_sng).sum(axis=0).max()
    fig = plt.figure();
    plt.plot((1-p_sng).sum(axis=0),(Y_real==0).sum(axis=0),'x')
    plt.plot([0, rg], [0, rg], ls="--", c="red")
    plt.ylabel("real zero inflation")
    plt.xlabel("sample zero inflation")
    return fig

def plotSampleVar(Y_real, y_sample):
    fig = plt.figure();
    rg = y_sample.var(axis=0).max()
    plt.plot(y_sample.var(axis=0), Y_real.var(axis=0), "x")
    #plt.plot([0, (np.diag(cov_noi_est_sng)).max()], [0, (np.diag(cov_noi_est_sng)).max()], ls="--", c="red")
    plt.plot([0, rg], [0, rg], ls="--", c="red")

    plt.ylabel("real obs var")
    plt.xlabel("sample obs var")
    return fig

def plotSamplePm(Y_real, p_poi, p_ber, p_sng, hd, p_sngr=None,ss=0,bins=36,fsz=15):
    Tbins = Y_real.shape[0];
    font_name = '';
    def get_zero(y, hd):
        hd_bins = np.linspace(-180,180,bins+1);
        T = Y_real.shape[0];
        zero_est_short = [];
        data_use = np.zeros((len(hd_bins)-1,2));
        for ii in range(len(hd_bins)-1):
            data_pos = ((hd[:T]>=hd_bins[ii])*(hd[:T]<=hd_bins[ii+1]));
            zero_est_short.append(y[:T][data_pos,:].mean(axis=0));
            #rate_real_short.append(Y_real[:T][data_pos,:].mean(axis=0));
        zero_est_short = np.asarray(zero_est_short).ravel();
        #rate_real_short = np.asarray(rate_real_short).ravel();
        return zero_est_short
    zero_real = get_zero(Y_real==0, hd);
    zero_poi = get_zero(1-p_poi, hd);
    zero_ber = get_zero(1-p_ber, hd);
    #zero_gam = get_zero(1-y_gamma_real, hd);
    zero_sng = get_zero(1-p_sng, hd);
    wid=0.7;
    sz=7;
    if p_sngr is None:
        fig = plt.figure(figsize=(12,4));
        ax1 = plt.subplot(1,3,1);
        rg = zero_real.max();
        
        ax1.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        ax1.plot(zero_poi, zero_real,'bo',markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.setp(ax1.get_xticklabels(), fontsize=fsz)
        plt.setp(ax1.get_yticklabels(), fontsize=fsz)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
        ax1.set_title('Poisson',fontsize=fsz,fontweight='bold');
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax1.set_ylabel("Observed probability of zeros",fontsize=fsz,fontweight='normal')

        ax2 = plt.subplot(1,3,2,sharey=ax1);
        #rg = (1-p_ber).sum(axis=0).max()/Tbins;
        ax2.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        ax2.plot(zero_ber, zero_real,'yo',markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), fontsize=fsz)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax2.set_xlabel("Fitted probability of zeros",fontsize=fsz,fontweight='normal')
        ax2.set_title('Bernoulli',fontsize=fsz, fontweight='bold');

        ax3 = plt.subplot(1,3,3,sharey=ax1);
        #rg = (1-p_sng).sum(axis=0).max()/Tbins;
        ax3.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        ax3.plot(zero_sng, zero_real,'ro',markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), fontsize=fsz)
        ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax3.set_title('Spike and gamma',fontsize=fsz, fontweight='bold');
        plt.tight_layout();
    else:
        fig = plt.figure(figsize=(24,4));
        ax1 = plt.subplot(1,4,1);
        rg = (1-p_poi).sum(axis=0).max()
        ax1.plot((1-p_poi).sum(axis=0),(Y_real==0).sum(axis=0),'x')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.plot([0, rg], [0, rg], ls="--", c="red")
        plt.title('poisson');
        plt.ylabel("real zero inflation")
        plt.xlabel("sample zero inflation")
        
        ax2 = plt.subplot(1,4,2);
        rg = (1-p_ber).sum(axis=0).max()
        ax2.plot((1-p_ber).sum(axis=0),(Y_real==0).sum(axis=0),'x')
        ax2.plot([0, rg], [0, rg], ls="--", c="red")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.title('bernoulli');
        
        ax3 = plt.subplot(1,4,3);
        rg = (1-p_sng).sum(axis=0).max()
        ax3.plot((1-p_sng).sum(axis=0),(Y_real==0).sum(axis=0),'x')
        ax3.plot([0, rg], [0, rg], ls="--", c="red")
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)      
        plt.title('spike and gamma');

        ax4 = plt.subplot(1,4,4);
        rg = (1-p_sngr).sum(axis=0).max()
        ax4.plot((1-p_sngr).sum(axis=0),(Y_real==0).sum(axis=0),'x')
        ax4.plot([0, rg], [0, rg], ls="--", c="red")
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)   
        plt.title('spike and gamma relax');

        plt.tight_layout();        
    return fig

def plotSampleVarm(Y_real, y_poisson, y_bernoulli, y_gamma_real, y_gammar, hd, y_gamma=None,ss=0,bins=36,fsz=16):
    def get_var(y, hd):
        hd_bins = np.linspace(-180,180,bins+1);
        T = Y_real.shape[0];
        var_est_short = [];
        data_use = np.zeros((len(hd_bins)-1,2));
        for ii in range(len(hd_bins)-1):
            data_pos = ((hd[:T]>=hd_bins[ii])*(hd[:T]<=hd_bins[ii+1]));
            var_est_short.append(y[:T][data_pos,:].var(axis=0));
            #rate_real_short.append(Y_real[:T][data_pos,:].mean(axis=0));
        var_est_short = np.asarray(var_est_short).ravel();
        #rate_real_short = np.asarray(rate_real_short).ravel();
        return var_est_short
    var_real = get_var(Y_real, hd);
    var_poi = get_var(y_poisson, hd);
    var_ber = get_var(y_bernoulli, hd);
    var_gam = get_var(y_gamma_real, hd);
    var_sngr = get_var(y_gammar, hd);
    wid=0.7;
    sz=7;
    if ss>0:
        print("sub-sample!");
        np.random.seed(888);
        use = np.random.permutation(var_real.shape[0])[:ss];
        var_real = var_real[use];
        var_poi = var_poi[use];
        var_ber = var_ber[use];
        var_gam = var_gam[use];
        var_sngr = var_sngr[use];
    
    if y_gamma is None:
        fig = plt.figure(figsize=(16,4));
        ax = fig.add_subplot(111);
        ax.set_xlabel('Fitted variance',fontsize=fsz,fontweight='normal');
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

        ax1 = fig.add_subplot(141);
        rg = var_real.max()
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        plt.plot(var_poi, var_real, "bo",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.setp(ax1.get_xticklabels(), fontsize=fsz)
        plt.setp(ax1.get_yticklabels(), fontsize=fsz)
        #tick_locator = ticker.MaxNLocator(nbins=6,prune="both");
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax1.set_title('Poisson',fontsize=fsz,fontweight='bold');
        ax1.set_ylabel("Observed variance",fontsize=fsz,fontweight='normal')
        
        ax2 = fig.add_subplot(142,sharey=ax1);
        #rg = y_bernoulli.var(axis=0).max()
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        plt.plot(var_ber, var_real, "yo",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.setp(ax2.get_xticklabels(), fontsize=fsz)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax2.set_title('Bernoulli',fontsize=fsz,fontweight='bold');
        
        ax3 = fig.add_subplot(143,sharey=ax1);
        #rg = y_gamma_real.var(axis=0).max()
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        plt.plot(var_gam, var_real, "go",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.setp(ax3.get_xticklabels(), fontsize=fsz)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax3.set_title('Gamma',fontsize=fsz,fontweight='bold');
       
        ax4 = fig.add_subplot(144,sharey=ax1);
        #rg = y_gammar.var(axis=0).max()
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        plt.plot(var_sngr, var_real, "ro",markersize=sz,markeredgecolor='white',markeredgewidth=wid)

        
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        plt.setp(ax4.get_xticklabels(), fontsize=fsz)
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax4.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax4.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax4.set_title('Spike and gamma',fontsize=fsz,fontweight='bold');

        plt.tight_layout();
    else:
        fig = plt.figure(figsize=(30,4));
        plt.subplot(1,5,1);
        rg = y_poisson.var(axis=0).max()
        plt.plot(y_poisson.var(axis=0), Y_real.var(axis=0), "x")
        #plt.plot([0, (np.diag(cov_noi_est_sng)).max()], [0, (np.diag(cov_noi_est_sng)).max()], ls="--", c="red")
        plt.plot([0, rg], [0, rg], ls="--", c="red")

        plt.ylabel("real obs var")
        plt.xlabel("sample obs var")
        plt.title('poisson');
        plt.subplot(1,5,2);
        rg = y_bernoulli.var(axis=0).max()
        plt.plot(y_bernoulli.var(axis=0), Y_real.var(axis=0), "x")
        #plt.plot([0, (np.diag(cov_noi_est_sng)).max()], [0, (np.diag(cov_noi_est_sng)).max()], ls="--", c="red")
        plt.plot([0, rg], [0, rg], ls="--", c="red")

        plt.ylabel("real obs var")
        plt.xlabel("sample obs var")
        plt.title('bernoulli');
        plt.subplot(1,5,3);
        rg = y_gamma.var(axis=0).max()
        plt.plot(y_gamma.var(axis=0), Y_real.var(axis=0), "x")
        #plt.plot([0, (np.diag(cov_noi_est_sng)).max()], [0, (np.diag(cov_noi_est_sng)).max()], ls="--", c="red")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.ylabel("real obs var")
        plt.xlabel("sample obs var")
        plt.title('spike and gamma');
        plt.subplot(1,5,4);
        rg = y_gamma_real.var(axis=0).max()
        plt.plot(y_gamma_real.var(axis=0), Y_real.var(axis=0), "x")
        #plt.plot([0, (np.diag(cov_noi_est_sng)).max()], [0, (np.diag(cov_noi_est_sng)).max()], ls="--", c="red")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.ylabel("real obs var")
        plt.xlabel("sample obs var")
        plt.title('gamma');
        
        plt.subplot(1,5,5);
        rg = y_gammar.var(axis=0).max()
        plt.plot(y_gammar.var(axis=0), Y_real.var(axis=0), "x")
        #plt.plot([0, (np.diag(cov_noi_est_sng)).max()], [0, (np.diag(cov_noi_est_sng)).max()], ls="--", c="red")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.ylabel("real obs var")
        plt.xlabel("sample obs var")
        plt.title('spike and gamma relax');        
        plt.tight_layout();
    
    return fig

def plotSampleRatem(Y_real, hd, y_poisson, y_bernoulli, y_gamma_real, y_gammar, y_gamma=None, ss=0,bins=36,fsz=16):
    def get_rate(y, hd):
        hd_bins = np.linspace(-180,180,bins+1);
        T = Y_real.shape[0];
        rate_est_short = [];
        data_use = np.zeros((len(hd_bins)-1,2));
        for ii in range(len(hd_bins)-1):
            data_pos = ((hd[:T]>=hd_bins[ii])*(hd[:T]<=hd_bins[ii+1]));
            rate_est_short.append(y[:T][data_pos,:].mean(axis=0));
            #rate_real_short.append(Y_real[:T][data_pos,:].mean(axis=0));
        rate_est_short = np.asarray(rate_est_short).ravel();
        #rate_real_short = np.asarray(rate_real_short).ravel();
        return rate_est_short
    rate_real = get_rate(Y_real, hd);
    rate_poi = get_rate(y_poisson, hd);
    rate_ber = get_rate(y_bernoulli, hd);
    rate_gam = get_rate(y_gamma_real, hd);
    rate_sngr = get_rate(y_gammar, hd);
    wid=0.7;
    sz=7;
    if ss>0:
        print("sub-sample!");
        np.random.seed(888);
        use = np.random.permutation(rate_real.shape[0])[:ss];
        rate_real = rate_real[use];
        rate_poi = rate_poi[use];
        rate_ber = rate_ber[use];
        rate_gam = rate_gam[use];
        rate_sngr = rate_sngr[use];
        
    if y_gamma is None:
        fig = plt.figure(figsize=(16,4));
        ax = fig.add_subplot(111);
        ax.set_xlabel('Fitted mean firing rate',fontsize=fsz,fontweight='normal');
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
        
        ax1 = fig.add_subplot(141);
        rg = rate_real.max();
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        plt.plot(rate_poi[:], rate_real[:], "bo",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.setp(ax1.get_xticklabels(), fontsize=fsz)
        plt.setp(ax1.get_yticklabels(), fontsize=fsz)
        #tick_locator = ticker.MaxNLocator(nbins=6,prune="both");
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax1.set_title('Poisson',fontsize=fsz,fontweight='bold');
        ax1.set_ylabel("Observed mean firing rate",fontsize=fsz,fontweight='normal')

        ax2 = fig.add_subplot(142,sharey=ax1);
        #rg = rate_ber.max();
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        plt.axvline(1, ls="--", linewidth=3,c="black");
        plt.text(1.1,0.1,'x=1',fontsize=fsz);
        plt.plot(rate_ber[:], rate_real[:], "yo",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.setp(ax2.get_xticklabels(), fontsize=fsz)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax2.set_title('Bernoulli',fontsize=fsz,fontweight='bold');
        
        ax3 = fig.add_subplot(143,sharey=ax1);
        #rg = rate_gam.max();
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
        plt.plot(rate_gam[:], rate_real[:], "go",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.setp(ax3.get_xticklabels(), fontsize=fsz)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax3.set_title('Gamma',fontsize=fsz,fontweight='bold');
        
        ax4 = fig.add_subplot(144,sharey=ax1);
        #rg = rate_sngr.max();
        plt.plot([0, rg], [0, rg], ls="--", linewidth=3, c="black")
        plt.plot(rate_sngr[:], rate_real[:], "ro",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
        
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        plt.setp(ax4.get_xticklabels(), fontsize=fsz)
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax4.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
        ax4.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
        ax4.set_title('Spike and gamma',fontsize=fsz,fontweight='bold');

        plt.tight_layout();
        
    else:
        rate_sng = get_rate(y_gamma, hd);
        fig = plt.figure(figsize=(30,4));
        plt.subplot(1,5,1);
        rg = rate_poi.max();
        plt.plot(rate_poi[:], rate_real[:], "x")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.xlabel("sample obs rate")
        plt.ylabel("real obs rate")
        plt.title('poisson');
        plt.subplot(1,5,2);
        rg = rate_ber.max();
        plt.plot(rate_ber[:], rate_real[:], "x")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.xlabel("sample obs rate")
        plt.ylabel("real obs rate")
        plt.title('bernoulli');
        plt.subplot(1,5,3);
        rg = rate_sng.max();
        plt.plot(rate_sng[:], rate_real[:], "x")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.xlabel("sample obs rate")
        plt.ylabel("real obs rate")
        plt.title('spike and gamma');
        plt.subplot(1,5,4);
        rg = rate_gam.max();
        plt.plot(rate_gam[:], rate_real[:], "x")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.xlabel("sample obs rate")
        plt.ylabel("real obs rate")
        plt.title('gamma');        
        plt.subplot(1,5,5);
        rg = rate_sngr.max();
        plt.plot(rate_sngr[:], rate_real[:], "x")
        plt.plot([0, rg], [0, rg], ls="--", c="red")
        plt.xlabel("sample obs rate")
        plt.ylabel("real obs rate")
        plt.title('spike and gamma relax');        
        plt.tight_layout();        
    return fig
    
def plotSampleData(Y_real, y_poisson, y_bernoulli, y_gamma, hd, tb=[0,2000], lines=[221,443,665], size=[8,8]):
    prefer_dg = np.zeros((Y_real.shape[1]));
    for ii in range(Y_real.shape[1]):
        prefer_dg[ii] = hd[np.where(Y_real[:,ii] == Y_real[:,ii].max())[0]]        
    order = np.argsort(prefer_dg);
    
    #tb = 2000;
    start = tb[0];
    end = tb[1];
    y_mat = np.vstack([y_poisson[start:end,order].T, y_bernoulli[start:end,order].T,y_gamma[start:end,order].T, Y_real[start:end,order].T])
    #print(y_mat.shape);
    fig = plt.figure(figsize=(size[0],size[1]))
    ax1=plt.subplot(2,1,1)
    plt.imshow(y_mat,"binary",vmin=0,vmax=y_mat.max())
    #for ii in range(y_mat.shape[0]):
    #    plt.scatter(np.where(y_mat[ii,start:end]>0), np.tile(ii,(y_mat[ii,start:end]>0).sum()),s=0.001,c='k')
    plt.axhline(y=lines[0])
    plt.axhline(y=lines[1])
    plt.axhline(y=lines[2])
    ax1.xaxis.set_ticks([])
    #plt.colorbar()
    ax2=plt.subplot(2,1,2)
    plt.plot(hd[start:end])
    ax2.xaxis.set_ticks([])
    plt.tight_layout()
    
    return fig

def plotTC(num, spl_values_poi, spl_values_ber, spl_values_gam, spl_values_sngr,tuning_curve,tuning_curve2,size=180):
    fs=17;
    if size==180:
        fig = plt.figure(figsize=(6.1,(6.1*2/3)*num));
    else:
        fig = plt.figure(figsize=(6.6,4.4*num)); #7.2,4.8
    for ii in range(num):
        ax = plt.subplot(num,1,ii+1);
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
                
        plt.plot(tuning_curve[:,ii], 'x-', markersize=8, label='Observed train');
        plt.plot(tuning_curve2[:,ii], 'o-', label='Observed test');
        plt.plot(spl_values_poi[:,ii], 'b-',linewidth=3,label='Poisson');
        plt.plot(spl_values_ber[:,ii], 'y-',linewidth=3,label='Bernoulli');
        plt.plot(spl_values_gam[:,ii], 'g-',linewidth=3,label='Gamma')
        plt.plot(spl_values_sngr[:,ii], 'r-', linewidth=3,label='SNG');
        
        plt.setp(ax.get_xticklabels(), fontsize=fs)
        plt.setp(ax.get_yticklabels(), fontsize=fs)
        #tick_locator = ticker.MaxNLocator(nbins=6,prune="both");
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune='both'))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_ylabel("Tuning curves",fontsize=fs,fontweight='normal')
        ax.set_xlabel("Stimulus (degree $^{\circ}$)",fontsize=fs,fontweight='normal')
        if size==180:
            ax.set_xticks((0,44,89,134,179))
            ax.set_xticklabels((-178,-90,0,90,180))
            if ii ==0:
                ax.legend(fontsize=14)
            
        else:
            ax.set_xticks((0,29,59,89,120))
            ax.set_xticklabels((-178,-90,0,90,180))
            if ii ==0:
                ax.legend(fontsize=12.8)
            
        plt.tight_layout()
    return fig

# depreciated
def plotTC2(num, spl_values_poi, spl_values_ber, spl_values_gam, spl_values_sngr,tuning_curve,tuning_curve2):
    fig = plt.figure(figsize=(12,4*num));
    
    for ii in range(num):
        ax = plt.subplot(num,2,2*ii+1);
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.plot(spl_values_poi[:,ii], label='Poisson');
        plt.plot(spl_values_ber[:,ii], label='Bernoulli');
        plt.plot(spl_values_gam[:,ii], label='Gamma')
        plt.plot(spl_values_sngr[:,ii], label='SNG');
        
        #plt.plot(spl_values_sng[:,2*ii], label='sng');
        #plt.plot(spl_values_ber[:,2*ii]*(spl_values_poi[:,2*ii].max()/spl_values_ber[:,2*ii].max()), label='bernoulli');
        plt.plot(tuning_curve[:,ii], '--', label='real');
        plt.ylabel(str(2*ii+1))
        if ii ==0:
            plt.legend()
            plt.title('training data')

        ax=plt.subplot(num,2,2*ii+2);
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.plot(spl_values_poi[:,ii], label='Poisson');
        plt.plot(spl_values_ber[:,ii], label='Bernoulli');
        plt.plot(spl_values_gam[:,ii], label='Gamma')
        plt.plot(spl_values_sngr[:,ii], label='SNG');
        plt.plot(tuning_curve2[:,ii], '--', label='real');
        if ii==0:
            plt.title('test data')
        #plt.ylabel(str(2*ii+2))

        plt.tight_layout()
    return fig

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def getBinNum(hd, hd_bins_use):
    bin_size= hd_bins_use[1] - hd_bins_use[0];
    hd_temp = np.asarray(np.ceil((hd-hd_bins_use[0])/bin_size), dtype=int);
    return hd_temp

def getBins(hd, hd_bins_use):
    bin_size= hd_bins_use[1] - hd_bins_use[0];
    hd_temp = np.asarray(np.ceil((hd-hd_bins_use[0])/bin_size), dtype=int);
    hd_temp = hd_bins_use[hd_temp];
    return hd_temp

def plotCdfSlabm(Y_real, hd_temp, spl_values_poi, spl_values_ber, spl_values_theta_sng, sng_k, sng_loc, spl_values_p_sng, spl_values_theta_gam, spl_values_k_gam, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0,lsz=17):
    ii=0;
    fig = plt.figure(figsize=(size[1]*4,size[0]*4))
    sz=17;
    #ax = plt.subplot(size[0],size[1],1);
    for m in bin_list:
        if ((ii!=0) and (ii!=3)):
            ax = plt.subplot(size[0],size[1],ii+1,sharey=ax0);
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax = plt.subplot(size[0],size[1],ii+1);
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];
        
        x = np.linspace(0, max(dat_temp.max(),0.5), np.ceil(max(dat_temp.max(),0.5)*1000));
        x[x<=1e-4] = 1e-3;
        x[0]=0;
        y1 = ss.poisson.cdf(x, spl_values_poi[m,nid]);
        y2 = ss.gamma.cdf(x-sng_loc[nid], scale = spl_values_theta_sng[m, nid], a= sng_k[nid]);
        y2 = y2*(spl_values_p_sng[m,nid])+1-spl_values_p_sng[m,nid];
        y3 = ss.gamma.cdf(x, scale = spl_values_theta_gam[m, nid], a= spl_values_k_gam[m, nid]);

        x2, yy = ecdf(dat_temp[dat_temp>0]); ## empirical cdf
        yy = yy*(dat_temp>0).sum()/dat_temp.shape[0] + (dat_temp==0).sum()/dat_temp.shape[0];
        
        y4= ss.bernoulli.cdf(x, spl_values_ber[m,nid]);
        
        plt.plot(x2, yy, 'k-',linewidth=3, label='Observed data');
        ax.plot(x[1:], y1[1:], 'b-',linewidth=3,label='Poisson');
        ax.plot(x[1:], y4[1:], 'y-',linewidth=3,label='Bernoulli');
        ax.plot(x[1:], y3[1:], 'g-',linewidth=3,label='Gamma');
        ax.plot(x[1:], y2[1:], 'r-',linewidth=3,label='SNG');
        
        plt.setp(ax.get_xticklabels(), fontsize=sz)
        plt.setp(ax.get_yticklabels(), fontsize=sz)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3,min_n_ticks=3,prune=None))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_title(str(hd_bins_use[m])+"$^{\circ}$",fontsize=sz,fontweight='bold')
        plt.draw();
        labels = [l.get_text() for l in ax.get_xticklabels()]
        labels[1] = '1e-3';
        ax.set_xticklabels(labels);
        if ii==0:
            ax.legend(fontsize=lsz);
        if ((ii==0) or (ii==3)):
            ax.set_ylabel("Probability",fontsize=sz,fontweight='normal')
            ax0 = ax;
        ii+=1;
        plt.tight_layout()
    return fig

def plotCdfSSm(Y_real, hd_temp, spl_values_poi, spl_values_ber, spl_values_theta_sng, sng_k, sng_loc, spl_p_values_sng, spl_values_theta_gam, spl_values_k_gam, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0):
    ii=0;
    fig = plt.figure(figsize=(size[1]*4,size[0]*4))
    #ax = plt.subplot(size[0],size[1],1);
    for m in bin_list:
        if ((ii!=0) and (ii!=3)):
            ax = plt.subplot(size[0],size[1],ii+1,sharey=ax0);
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax = plt.subplot(size[0],size[1],ii+1);
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];
        
        
        x = np.linspace(0, max(dat_temp.max(),0.5), np.ceil(max(dat_temp.max(),0.5)*1000));
        x[x<1e-4] = 1e-3;
        x[0]=0;
        y1 = ss.poisson.cdf(x, spl_values_poi[m,nid]);
        y2 = ss.gamma.cdf(x-sng_loc[nid], scale = spl_values_theta_sng[m, nid], a= sng_k[nid]);
        y2 = y2*spl_p_values_sng[m,nid]+1-spl_p_values_sng[m,nid];
        y3 = ss.gamma.cdf(x, scale = spl_values_theta_gam[m, nid], a= spl_values_k_gam[m, nid]);

        x2, yy = ecdf(dat_temp[dat_temp>0]); ## empirical cdf
        yy = yy*(dat_temp>0).sum()/dat_temp.shape[0] + (dat_temp==0).sum()/dat_temp.shape[0];
        
        #x2, yy = ecdf(dat_temp); ## empirical cdf
        
        y4= ss.bernoulli.cdf(x, spl_values_ber[m,nid]);
        
        plt.plot(x2, yy, 'k-',linewidth=3, label='Observed data');
        ax.plot(x[:], y1[:], 'b-',linewidth=3,label='Poisson');
        ax.plot(x[:], y4[:], 'y-',linewidth=3,label='Bernoulli');
        ax.plot(x[:], y3[:], 'g-',linewidth=3,label='Gamma');
        ax.plot(x[:], y2[:], 'r-',linewidth=3,label='SNG');
        
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3,min_n_ticks=3,prune=None))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_title(str(hd_bins_use[m])+"$^{\circ}$",fontsize=14,fontweight='bold')
        if ii==0:
            ax.legend(fontsize=14);
        if ((ii==0) or (ii==3)):
            ax.set_ylabel("Probability",fontsize=14,fontweight='normal')
            ax0 = ax;
        ii+=1;
        plt.tight_layout()
    
    return fig

def plotPoiCdf(Y_real, hd_temp, spl_values_poi, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0):
    ii=0;
    fig = plt.figure(figsize=(size[1]*5,size[0]*4))
    for m in bin_list:
        plt.subplot(size[0],size[1],ii+1);
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];

        x = np.linspace(0, max(dat_temp.max(),0.5), 500);
        y1 = ss.poisson.cdf(x, spl_values_poi[m,nid]);
        plt.plot(x, y1, 'r-');

        x2, y2 = ecdf(dat_temp); ## empirical cdf
        plt.plot(x2, y2, 'b-');
        plt.ylabel(str(hd_bins_use[m]))
        ii+=1;
        plt.tight_layout()
    return fig

def plotSngCdf(Y_real, hd_temp, spl_values_theta_sng, sng_k, sng_loc, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0):
    ii=0;
    fig = plt.figure(figsize=(size[1]*5,size[0]*4))
    for m in bin_list:
        plt.subplot(size[0],size[1],ii+1);
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];

        x = np.linspace(0, max(dat_temp.max(),0.5), 100);
        y1 = ss.gamma.cdf(x-sng_loc[nid], scale = spl_values_theta_sng[m, nid], a= sng_k[nid]);
        plt.plot(x, y1, 'r-');

        x2, y2 = ecdf(dat_temp[dat_temp>0]); ## empirical cdf
        plt.plot(x2, y2, 'b-');
        plt.ylabel(str(hd_bins_use[m]))
        ii+=1;
        plt.tight_layout()
    return fig

def plotGamCdf(Y_real, hd_temp, spl_values_theta_gam, spl_values_k_gam, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0):
    ii=0;
    fig = plt.figure(figsize=(size[1]*5,size[0]*4))
    for m in bin_list:
        plt.subplot(size[0],size[1],ii+1);
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];

        x = np.linspace(0, max(dat_temp.max(),0.5), 10000);
        y1 = ss.gamma.cdf(x, scale = spl_values_theta_gam[m, nid], a= spl_values_k_gam[m, nid]);
        plt.plot(x, y1, 'r-');

        x2, y2 = ecdf(dat_temp); ## empirical cdf
        plt.plot(x2, y2, 'b-');
        plt.ylabel(str(hd_bins_use[m]))
        ii+=1;
        plt.tight_layout()
    return fig

def plotSngRRCdf(Y_real, hd_temp, spl_values_theta_sng, sng_k, sng_loc, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0):
    ii=0;
    fig = plt.figure(figsize=(size[1]*5,size[0]*4))
    for m in bin_list:
        plt.subplot(size[0],size[1],ii+1);
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];

        x = np.linspace(0, max(dat_temp.max(),0.5), 100);
        y1 = ss.gamma.cdf(x-sng_loc[nid], scale = spl_values_theta_sng[m, nid], a= sng_k[m,nid]);
        plt.plot(x, y1, 'r-');

        x2, y2 = ecdf(dat_temp[dat_temp>0]); ## empirical cdf
        plt.plot(x2, y2, 'b-');
        plt.ylabel(str(hd_bins_use[m]))
        ii+=1;
        plt.tight_layout()
    return fig

def plotSngPdf(Y_real, hd_temp, spl_values_sng, sng_k, sng_loc, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0):
    ii=0;
    fig = plt.figure(figsize=(size[1]*5,size[0]*4))
    for m in bin_list:
        plt.subplot(size[0],size[1],ii+1);
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];
        plt.hist(dat_temp[dat_temp>0], density=True, bins=30); ## empirical pdf
        
        x = np.linspace(0, max(dat_temp.max(),0.5), 100);
        y1 = ss.gamma.pdf(x-sng_loc[nid], scale = spl_values_sng[m, nid], a= sng_k[nid]);
        plt.plot(x, y1, 'r-');

        plt.ylabel(str(hd_bins_use[m]))
        ii+=1;
        plt.tight_layout()
    return fig

def plotGamPdf(Y_real, hd_temp, spl_values_theta_gam, spl_values_k_gam, hd_bins_use, bin_list=[5,10,20,40,80,120,140,160,179], size=[3,3], nid=0):
    ii=0;
    fig = plt.figure(figsize=(size[1]*5,size[0]*4))
    for m in bin_list:
        plt.subplot(size[0],size[1],ii+1);
        dat_temp = Y_real[(hd_temp==hd_bins_use[m]),nid];
        plt.hist(dat_temp, density=True, bins=1000); ## empirical pdf
        
        x = np.linspace(0, max(dat_temp.max(),0.5), 10000);
        y1 = ss.gamma.pdf(x, scale = spl_values_theta_gam[m, nid], a= spl_values_k_gam[m, nid]);
        plt.plot(x, y1, 'r-');

        plt.ylabel(str(hd_bins_use[m]))
        ii+=1;
        plt.tight_layout()
    return fig

## decoding plot functions
# CI
def plotPosteriorSuper(hd_train,hd_test,poi_md_tr,ber_md_tr,gam_md_tr,sngr_md_tr,poi_md_te,ber_md_te,gam_md_te,sngr_md_te,poi_ci_tr,ber_ci_tr,gam_ci_tr,sngr_ci_tr,poi_ci_te,ber_ci_te,gam_ci_te,sngr_ci_te,pos=False):
    fig = plt.figure(figsize=(15,12))
    alpha=0.1;
    size=5;
    ax = fig.add_subplot(111);
    ax.set_xlabel('Frames',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",labelpad=20, fontsize=14,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(4,2,1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax1.plot(poi_md_tr,'x',markersize=size,label='Estimate')
    ax1.plot(hd_train,'ro',markersize=size,alpha=alpha,label='True')
    ax1.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),poi_ci_tr[:,0],poi_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    if not pos:
        ax1.legend(fontsize=14);
    else:
        ax1.legend(fontsize=11.8, loc=pos);
    ax1.set_title("Poisson train", fontsize=14, fontweight='bold');

    ax3 = fig.add_subplot(4,2,3,sharex=ax1,sharey=ax1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax3.plot(ber_md_tr,'x',markersize=size,label='estimated')
    ax3.plot(hd_train,'ro',markersize=size,alpha=alpha,label='true')
    ax3.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),ber_ci_tr[:,0],ber_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    #ax3.legend(fontsize=14);
    ax3.set_title("Bernoulli train", fontsize=14, fontweight='bold');

    ax5 = fig.add_subplot(4,2,5,sharex=ax1,sharey=ax1)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax5.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax5.plot(gam_md_tr,'x',markersize=size,label='estimated')
    ax5.plot(hd_train,'ro',markersize=size,alpha=alpha,label='true')
    ax5.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),gam_ci_tr[:,0],gam_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    #ax5.legend(fontsize=14);
    ax5.set_title("Gamma train", fontsize=14, fontweight='bold');

    ax7 = fig.add_subplot(4,2,7,sharex=ax1,sharey=ax1)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), visible=False)
    plt.setp(ax7.get_yticklabels(), visible=False)
    ax7.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax7.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax7.plot(sngr_md_tr,'x',markersize=size,label='estimated')
    ax7.plot(hd_train,'ro',markersize=size,alpha=alpha,label='true')
    ax7.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),sngr_ci_tr[:,0],sngr_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    #ax5.legend(fontsize=14);
    ax7.set_title("SNG train", fontsize=14, fontweight='bold');

    ax2 = fig.add_subplot(4,2,2,sharex=ax1,sharey=ax1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax2.plot(poi_md_te,'x',markersize=size,label='estimated')
    ax2.plot(hd_test,'ro',markersize=size,alpha=alpha,label='true')
    ax2.fill_between(np.linspace(0,hd_test.shape[0]-1,hd_test.shape[0]),poi_ci_te[:,0],poi_ci_te[:,1],color="#b9cfe7", edgecolor="");
    #ax2.legend(fontsize=14);
    ax2.set_title("Poisson test", fontsize=14, fontweight='bold');

    ax4 = fig.add_subplot(4,2,4,sharex=ax2,sharey=ax2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax4.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax4.plot(ber_md_te,'x',markersize=size,label='estimated')
    ax4.plot(hd_test,'ro',markersize=size,alpha=alpha,label='true')
    ax4.fill_between(np.linspace(0,hd_test.shape[0]-1,hd_test.shape[0]),ber_ci_te[:,0],ber_ci_te[:,1],color="#b9cfe7", edgecolor="");
    #ax3.legend(fontsize=14);
    ax4.set_title("Bernoulli test", fontsize=14, fontweight='bold');

    ax6 = fig.add_subplot(4,2,6,sharex=ax2,sharey=ax2)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax6.get_yticklabels(), visible=False)
    ax6.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax6.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax6.plot(gam_md_te,'x',markersize=size,label='estimated')
    ax6.plot(hd_test,'ro',markersize=size,alpha=alpha,label='true')
    ax6.fill_between(np.linspace(0,hd_test.shape[0]-1,hd_test.shape[0]),gam_ci_te[:,0],gam_ci_te[:,1],color="#b9cfe7", edgecolor="");
    #a65.legend(fontsize=14);
    ax6.set_title("Gamma test", fontsize=14, fontweight='bold');

    ax8 = fig.add_subplot(4,2,8,sharex=ax2,sharey=ax2)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    plt.setp(ax8.get_xticklabels(), visible=False)
    plt.setp(ax8.get_yticklabels(), visible=False)
    ax8.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax8.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax8.plot(sngr_md_te,'x',markersize=size,label='estimated')
    ax8.plot(hd_test,'ro',markersize=size,alpha=alpha,label='true')
    ax8.fill_between(np.linspace(0,hd_test.shape[0]-1,hd_test.shape[0]),sngr_ci_te[:,0],sngr_ci_te[:,1],color="#b9cfe7", edgecolor="");
    #ax5.legend(fontsize=14);
    ax8.set_title("SNG test", fontsize=14, fontweight='bold');

    plt.tight_layout()
    return fig

# CI
def plotPosteriorSuper2(hd_train,poi_md_tr,ber_md_tr,gam_md_tr,sngr_md_tr,poi_ci_tr,ber_ci_tr,gam_ci_tr,sngr_ci_tr,pos=False):
    fig = plt.figure(figsize=(18,10.8))
    alpha=0.1;
    size=5;
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",labelpad=20, fontsize=14,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax7 = fig.add_subplot(4,1,4)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    #ax7.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax7.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax7.plot(sngr_md_tr,'x',markersize=size,label='estimated')
    ax7.plot(hd_train,'ro',markersize=size,alpha=alpha,label='true')
    ax7.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),sngr_ci_tr[:,0],sngr_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    #ax5.legend(fontsize=14);
    ax7.set_title("SNG", fontsize=14, fontweight='bold');
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,3000));
    
    ax1 = fig.add_subplot(4,1,1,sharex=ax7,sharey=ax7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax1.plot(poi_md_tr,'x',markersize=size,label='Estimate')
    ax1.plot(hd_train,'ro',markersize=size,alpha=alpha,label='True')
    ax1.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),poi_ci_tr[:,0],poi_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    if not pos:
        ax1.legend(fontsize=14,loc=3,bbox_to_anchor=(-0.06,-0.1));
    else:
        ax1.legend(fontsize=11.8, loc=pos);
    ax1.set_title("Poisson", fontsize=14, fontweight='bold');

    ax3 = fig.add_subplot(4,1,2,sharex=ax7,sharey=ax7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    #ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax3.plot(ber_md_tr,'x',markersize=size,label='estimated')
    ax3.plot(hd_train,'ro',markersize=size,alpha=alpha,label='true')
    ax3.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),ber_ci_tr[:,0],ber_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    #ax3.legend(fontsize=14);
    ax3.set_title("Bernoulli", fontsize=14, fontweight='bold');

    ax5 = fig.add_subplot(4,1,3,sharex=ax7,sharey=ax7)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    #ax5.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax5.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    ax5.plot(gam_md_tr,'x',markersize=size,label='estimated')
    ax5.plot(hd_train,'ro',markersize=size,alpha=alpha,label='true')
    ax5.fill_between(np.linspace(0,hd_train.shape[0]-1,hd_train.shape[0]),gam_ci_tr[:,0],gam_ci_tr[:,1],color="#b9cfe7", edgecolor="");
    #ax5.legend(fontsize=14);
    ax5.set_title("Gamma", fontsize=14, fontweight='bold');

    plt.tight_layout()
    return fig

# posterior 4 panels
def plotPosteriorSuper3(hd_train,poi_md_tr,ber_md_tr,gam_md_tr,sngr_md_tr,poi_lik_tr,ber_lik_tr,gam_lik_tr,sngr_lik_tr,ysize=180):
    fig = plt.figure(figsize=(10,7))
    alpha=0.5;
    size=1;
    cmap = 'Reds';
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal'); #labelpad=20, 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    vmin = min([sngr_lik_tr.min(),poi_lik_tr.min(),ber_lik_tr.min(),gam_lik_tr.min()]);
    vmax = max([sngr_lik_tr.max(),poi_lik_tr.max(),ber_lik_tr.max(),gam_lik_tr.max()]);

    ax7 = fig.add_subplot(4,1,4)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    
    im=ax7.imshow(sngr_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax7.plot(sngr_md_tr,'yo',markersize=size,alpha=alpha,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5);
    #ax5.legend(fontsize=14);
    ax7.set_title("SNG", fontsize=14, fontweight='bold');
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,3000));
    if ysize==180:
        ax7.set_yticks((44,89,134,179))
        ax7.set_yticklabels((-90,0,90,180))
        
    else:
        ax7.set_yticks((29,59,89,119))
        ax7.set_yticklabels((-90,0,90,180))
    
    ax1 = fig.add_subplot(4,1,1,sharex=ax7,sharey=ax7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    #ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    #ax1.plot(poi_md_tr,'yo',markersize=size,alpha=alpha,label='Estimate')
    im=ax1.imshow(poi_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    ax1.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax1.plot(hd_train,'b',linewidth=0.5);
    ax1.set_title("Poisson", fontsize=14, fontweight='bold');

    ax3 = fig.add_subplot(4,1,2,sharex=ax7,sharey=ax7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    #ax3.spines['left'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    
    #ax3.plot(ber_md_tr,'yo',markersize=size,alpha=alpha,label='estimated')
    im=ax3.imshow(ber_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    ax3.plot(hd_train,'bo', markersize=size,alpha=alpha,label='true')
    #ax3.plot(hd_train,'b',linewidth=0.5);
    ax3.set_title("Bernoulli", fontsize=14, fontweight='bold');

    ax5 = fig.add_subplot(4,1,3,sharex=ax7,sharey=ax7)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    #ax5.spines['left'].set_visible(False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    
    #ax5.plot(gam_md_tr,'yo',mfc='none',markersize=size,label='estimated')
    im=ax5.imshow(gam_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    ax5.plot(hd_train,'bo',markersize=size,alpha=alpha,label='true')
    #ax5.plot(hd_train,'b',linewidth=0.5);
    ax5.set_title("Gamma", fontsize=14, fontweight='bold');
    axins = fig.add_axes([0.99, 0.1, 0.01, 0.85]) 
    cbar = plt.colorbar(im, ax=[ax1,ax3,ax5,ax7],cax=axins,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=18) 
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    plt.tight_layout()
    return fig

# posterior 5 panels
def plotPosteriorSuper4(hd_train,poi_md_tr,ber_md_tr,gam_md_tr,sngr_md_tr,poi_md_tr2, poi_lik_tr,ber_lik_tr,gam_lik_tr,sngr_lik_tr,poi_lik_tr2, ysize=180):
    fig = plt.figure(figsize=(10,35/4))
    alpha=0.5;
    size=1;
    cmap='Reds';
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal'); #labelpad=20, 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    vmin = min([sngr_lik_tr.min(),poi_lik_tr2.min(),poi_lik_tr.min(),ber_lik_tr.min(),gam_lik_tr.min()]);
    vmax = max([sngr_lik_tr.max(),poi_lik_tr2.max(),poi_lik_tr.max(),ber_lik_tr.max(),gam_lik_tr.max()]);

    ax7 = fig.add_subplot(5,1,5)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    #ax7.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax7.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    #ax7.set_yticks([]);
    
    im=ax7.imshow(poi_lik_tr2, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax7.plot(poi_md_tr2,'g+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5, label='True')
    ax7.set_title("Unnormalized Poisson", fontsize=14, fontweight='bold');
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,3000));
    if ysize==180:
        ax7.set_yticks((44,89,134,179))
        ax7.set_yticklabels((-90,0,90,180))
        
    else:
        ax7.set_yticks((29,59,89,119))
        ax7.set_yticklabels((-90,0,90,180))
    
    ax8 = fig.add_subplot(5,1,4,sharex=ax7,sharey=ax7)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    #ax8.spines['left'].set_visible(False)
    plt.setp(ax8.get_xticklabels(), visible=False)
    plt.setp(ax8.get_yticklabels(), visible=False)
    
    im=ax8.imshow(sngr_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax8.plot(sngr_md_tr,'g+',markersize=size,label='Estimate')
    ax8.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax8.plot(hd_train,'b',linewidth=0.5, label='True')
    ax8.set_title("SNG", fontsize=14, fontweight='bold');
    
    ax1 = fig.add_subplot(5,1,1,sharex=ax7,sharey=ax7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #ax1.plot(poi_md_tr,'g+',markersize=size, label='Estimate')
    im=ax1.imshow(poi_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    ax1.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax1.plot(hd_train,'b',linewidth=0.5, label='True')
    ax1.set_title("Poisson", fontsize=14, fontweight='bold');

    ax3 = fig.add_subplot(5,1,2,sharex=ax7,sharey=ax7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    #ax3.spines['left'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    
    #ax3.plot(ber_md_tr,'g+',markersize=size,alpha=alpha,label='estimated')
    im=ax3.imshow(ber_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    ax3.plot(hd_train,'bo', markersize=size,alpha=alpha,label='true')
    #ax3.plot(hd_train,'b',linewidth=0.5, label='True')
    ax3.set_title("Bernoulli", fontsize=14, fontweight='bold');

    ax5 = fig.add_subplot(5,1,3,sharex=ax7,sharey=ax7)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    #ax5.spines['left'].set_visible(False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    
    #ax5.plot(gam_md_tr,'g+',markersize=size,label='estimated')
    im=ax5.imshow(gam_lik_tr,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax5.plot(hd_train,'b',linewidth=0.5, label='True')
    ax5.plot(hd_train,'bo',markersize=size,alpha=alpha,label='true')
    #ax5.legend(fontsize=14);
    ax5.set_title("Gamma", fontsize=14, fontweight='bold');
    axins = fig.add_axes([0.99, 0.07, 0.01, 0.89]) 
    cbar = plt.colorbar(im, ax=[ax1,ax3,ax5,ax7,ax8],cax=axins,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=18) 
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    #fig.colorbar(im, ax)

    plt.tight_layout()
    return fig

# decoding error 5 panels
def plotPosteriorSuper5(hd_train,poi_md_tr,ber_md_tr,gam_md_tr,sngr_md_tr,poi_md_tr2, poi_lik_tr,ber_lik_tr,gam_lik_tr,sngr_lik_tr,poi_lik_tr2, ysize=180):
    fig = plt.figure(figsize=(10,35/4))
    alpha=0.2;
    size=3;
    
    cmap='nipy_spectral_r';
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal'); #labelpad=20, 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax7 = fig.add_subplot(5,1,5)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    #ax7.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax7.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    #ax7.set_yticks([]);
    ax7.set_xlim((0,2999));
    ax7.plot(poi_md_tr2,'r+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5, label='True')
    ax7.set_title("Unnormalized Poisson", fontsize=14, fontweight='bold');
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,2999));
    if ysize==180:
        ax7.set_yticks((44,89,134,179))
        ax7.set_yticklabels((-90,0,90,180))
        
    else:
        ax7.set_yticks((29,59,89,119))
        ax7.set_yticklabels((-90,0,90,180))
    
    ax8 = fig.add_subplot(5,1,4,sharex=ax7,sharey=ax7)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    #ax8.spines['left'].set_visible(False)
    plt.setp(ax8.get_xticklabels(), visible=False)
    plt.setp(ax8.get_yticklabels(), visible=False)
    ax8.plot(sngr_md_tr,'r+',markersize=size,label='Estimate')
    ax8.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax8.plot(hd_train,'b',linewidth=0.5, label='True')
    ax8.set_title("SNG", fontsize=14, fontweight='bold');
    ax8.set_xlim((0,2999));
    
    ax1 = fig.add_subplot(5,1,1,sharex=ax7,sharey=ax7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.plot(poi_md_tr,'r+',markersize=size, label='Estimate')
    ax1.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    ax1.set_title("Poisson", fontsize=14, fontweight='bold');
    ax1.set_xlim((0,2999));
    
    ax3 = fig.add_subplot(5,1,2,sharex=ax7,sharey=ax7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    #ax3.spines['left'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.plot(ber_md_tr,'r+',markersize=size,label='estimated')
    ax3.plot(hd_train,'bo', markersize=size,alpha=alpha,label='true')
    ax3.set_title("Bernoulli", fontsize=14, fontweight='bold');
    ax3.set_xlim((0,2999));
    
    ax5 = fig.add_subplot(5,1,3,sharex=ax7,sharey=ax7)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    #ax5.spines['left'].set_visible(False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.set_xlim((0,2999));
    ax5.plot(gam_md_tr,'r+',markersize=size,label='estimated')
    ax5.plot(hd_train,'bo',markersize=size,alpha=alpha,label='true')
    ax5.set_title("Gamma", fontsize=14, fontweight='bold');
    plt.tight_layout()
    return fig

# decoding error 4 panels
def plotPosteriorSuper6(hd_train,poi_md_tr,ber_md_tr,gam_md_tr,sngr_md_tr,poi_lik_tr,ber_lik_tr,gam_lik_tr,sngr_lik_tr,ysize=120):
    fig = plt.figure(figsize=(10,7))
    alpha=0.1;
    size=3;
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal'); #labelpad=20, 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax7 = fig.add_subplot(4,1,4)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    ax7.set_xlim((0,2999));
    ax7.plot(sngr_md_tr,'r+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5);
    #ax5.legend(fontsize=14);
    ax7.set_title("SNG", fontsize=14, fontweight='bold');
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,2999));
    ax7.set_yticks((29,59,89,120))
    ax7.set_yticklabels((-90,0,90,180))
    
    ax1 = fig.add_subplot(4,1,1,sharex=ax7,sharey=ax7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    
    ax1.plot(poi_md_tr,'r+',markersize=size,label='Estimate')
    ax1.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    ax1.set_title("Poisson", fontsize=14, fontweight='bold');
    ax1.set_xlim((0,2999));
    ax3 = fig.add_subplot(4,1,2,sharex=ax7,sharey=ax7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    #ax3.spines['left'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.plot(ber_md_tr,'r+',markersize=size,label='estimated')
    ax3.plot(hd_train,'bo', markersize=size,alpha=alpha,label='true')
    #ax3.plot(hd_train,'b',linewidth=0.5);
    ax3.set_title("Bernoulli", fontsize=14, fontweight='bold');
    ax3.set_xlim((0,2999));
    ax5 = fig.add_subplot(4,1,3,sharex=ax7,sharey=ax7)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    #ax5.spines['left'].set_visible(False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.set_xlim((0,2999));
    ax5.plot(gam_md_tr,'r+',markersize=size,label='estimated')
    ax5.plot(hd_train,'bo',markersize=size,alpha=alpha,label='true')
    ax5.set_title("Gamma", fontsize=14, fontweight='bold');
    plt.tight_layout()
    return fig

def plotPosSuper(hd_train,sngr_md_tr,poi_lik_tr,sngr_lik_tr):
    fig = plt.figure(figsize=(10,3.5))
    alpha=0.5;
    size=1;
    cmap='Reds';
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal',labelpad=20); #labelpad=20, 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    #ax8 = fig.add_subplot(3,1,3)
    #ax8.spines['top'].set_visible(False)
    #ax8.spines['right'].set_visible(False)
    ##ax8.spines['left'].set_visible(False)
    #plt.setp(ax8.get_xticklabels(), fontsize=14)
    #plt.setp(ax8.get_yticklabels(), fontsize=14)
    #ax8.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
#
    #ax8.set_xlim((0,2999));
    #ax8.set_ylim((0,119));
    #ax8.plot(sngr_md_tr,'r+',markersize=size,label='Estimate')
    #ax8.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    ##ax8.plot(hd_train,'b',linewidth=0.5, label='True')
    #ax8.set_title("SNG", fontsize=14, fontweight='bold');
    #ax8.set_xticklabels((0,20,40,60,80,100));
    #ax8.set_xticks((0,600,1200,1800,2400,2999));
#
    #ax8.set_yticks((0,29,59,89,119))
    #ax8.set_yticklabels((-178,-90,0,90,180))    
    vmin = min([sngr_lik_tr.min(),poi_lik_tr.min()]);
    vmax = max([sngr_lik_tr.max(),poi_lik_tr.max()]);

    ax7 = fig.add_subplot(2,1,2)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    
    im=ax7.imshow(sngr_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax7.plot(poi_md_tr2,'g+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5, label='True')
    ax7.set_title("SNG", fontsize=14, fontweight='bold');
    #ax7.set_xlim((0,2999));
    #ax7.set_ylim((0,119));
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,2999));

    ax7.set_yticks((29,59,89,119))
    ax7.set_yticklabels((-90,0,90,180))    
    
    ax1 = fig.add_subplot(2,1,1,sharex=ax7,sharey=ax7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #ax1.plot(poi_md_tr,'g+',markersize=size, label='Estimate')
    im=ax1.imshow(poi_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    ax1.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax1.plot(hd_train,'b',linewidth=0.5, label='True')
    ax1.set_title("Poisson", fontsize=14, fontweight='bold');
    #ax1.set_xlim((0,2999));
    #ax1.set_ylim((0,119));
    
    axins = fig.add_axes([0.99, 0.18, 0.01, 0.72])  # left/right, up/down, width, height
    cbar = plt.colorbar(im, ax=[ax1,ax7],cax=axins,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=18) 
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    #fig.colorbar(im, ax)

    plt.tight_layout()
    
    
    return fig

def plotPosSuper2(hd_train,sngr_lik_tr):
    fig = plt.figure(figsize=(10,3.5/2))
    alpha=0.5;
    size=1;
    cmap='Reds';
    
    ax7 = fig.add_subplot(1,1,1)
    ax7.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax7.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal');     
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    
    im=ax7.imshow(sngr_lik_tr, cmap=cmap,origin='lower',aspect='auto');
    #ax7.plot(poi_md_tr2,'g+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5, label='True')
    ax7.set_title("SNG", fontsize=14, fontweight='bold');
    #ax7.set_xlim((0,2999));
    #ax7.set_ylim((0,119));
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,2999));

    ax7.set_yticks((29,59,89,119))
    ax7.set_yticklabels((-90,0,90,180))    
        
    axins = fig.add_axes([0.91, 0.12, 0.01, 0.76])  # left/right, up/down, width, height
    cbar = plt.colorbar(im, ax=[ax7],cax=axins,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=18) 
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()

    #plt.tight_layout()
       
    return fig

def plotHDSuper3(hd_train,sngr_lik_tr):
    fig = plt.figure(figsize=(10,3.5/2))
    alpha=0.5;
    size=1;
    cmap='Reds';
    
    ax7 = fig.add_subplot(1,1,1)
    ax7.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax7.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal');     
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), fontsize=14)
    plt.setp(ax7.get_yticklabels(), fontsize=14)
    
    im=ax7.imshow(sngr_lik_tr, cmap=cmap,origin='lower',aspect='auto');
    #ax7.plot(poi_md_tr2,'g+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5, label='True')
    ax7.set_title("SNG", fontsize=14, fontweight='bold');
    #ax7.set_xlim((0,2999));
    #ax7.set_ylim((0,119));
    ax7.set_xticklabels((0,20,40,60,80,100));
    ax7.set_xticks((0,600,1200,1800,2400,2999));

    ax7.set_yticks((44,89,134,179))
    ax7.set_yticklabels((-90,0,90,180))    
        
    axins = fig.add_axes([0.91, 0.12, 0.01, 0.76])  # left/right, up/down, width, height
    cbar = plt.colorbar(im, ax=[ax7],cax=axins,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=18) 
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()

    #plt.tight_layout()
       
    return fig

def plotHDSuper(hd_train,poi_lik_tr,sngr_lik_tr,poi_lik_tr2):
    fig = plt.figure(figsize=(10,3.5*3/2))
    alpha=0.5;
    size=1;
    cmap='Reds';
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,fontweight='normal'); #labelpad=20, 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax8 = fig.add_subplot(3,1,3)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    #ax8.spines['left'].set_visible(False)
    plt.setp(ax8.get_xticklabels(), fontsize=14)
    plt.setp(ax8.get_yticklabels(), fontsize=14)
    ax8.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    vmin = min([sngr_lik_tr.min(),poi_lik_tr2.min(),poi_lik_tr.min()]);
    vmax = max([sngr_lik_tr.max(),poi_lik_tr2.max(),poi_lik_tr.max()]);

    im=ax8.imshow(sngr_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax8.plot(sngr_md_tr,'g+',markersize=size,label='Estimate')
    ax8.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax8.plot(hd_train,'b',linewidth=0.5, label='True')
    ax8.set_title("SNG", fontsize=14, fontweight='bold');
    ax8.set_xticklabels((0,20,40,60,80,100));
    ax8.set_xticks((0,600,1200,1800,2400,3000));

    ax8.set_yticks((44,89,134,179))
    ax8.set_yticklabels((-90,0,90,180))    

    ax7 = fig.add_subplot(3,1,1,sharex=ax8,sharey=ax8)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), visible=False)
    plt.setp(ax7.get_yticklabels(), visible=False)
    
    im=ax7.imshow(poi_lik_tr2, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax7.plot(poi_md_tr2,'g+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5, label='True')
    ax7.set_title("Unnormalized Poisson", fontsize=14, fontweight='bold');
    
    ax1 = fig.add_subplot(3,1,2,sharex=ax8,sharey=ax8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #ax1.plot(poi_md_tr,'g+',markersize=size, label='Estimate')
    im=ax1.imshow(poi_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    ax1.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax1.plot(hd_train,'b',linewidth=0.5, label='True')
    ax1.set_title("Poisson", fontsize=14, fontweight='bold');

    axins = fig.add_axes([0.99, 0.12, 0.01, 0.82])  # left/right, up/down, width, height
    cbar = plt.colorbar(im, ax=[ax1,ax7,ax8],cax=axins,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=18) 
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    #fig.colorbar(im, ax)

    plt.tight_layout()
    
    return fig

def plotHDSuper2(hd_train,sngr_lik_tr,poi_lik_tr2):
    fig = plt.figure(figsize=(10,3.5*2/2))
    alpha=0.5;
    size=1;
    cmap='Reds';
    ax = fig.add_subplot(111);
    ax.set_xlabel('Time (s)',fontsize=14,fontweight='normal');
    ax.set_ylabel("Stimulus (degree $^{\circ}$)",fontsize=14,labelpad=18,fontweight='normal'); #labelpad=20, 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    vmin = min(sngr_lik_tr.min(),poi_lik_tr2.min());
    vmax = max(sngr_lik_tr.max(),poi_lik_tr2.max());
    ax8 = fig.add_subplot(2,1,2)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    #ax8.spines['left'].set_visible(False)
    plt.setp(ax8.get_xticklabels(), fontsize=14)
    plt.setp(ax8.get_yticklabels(), fontsize=14)
    ax8.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))

    im1=ax8.imshow(sngr_lik_tr, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax8.plot(sngr_md_tr,'g+',markersize=size,label='Estimate')
    ax8.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax8.plot(hd_train,'b',linewidth=0.5, label='True')
    ax8.set_title("SNG", fontsize=14, fontweight='bold');
    ax8.set_xticklabels((0,20,40,60,80,100));
    ax8.set_xticks((0,600,1200,1800,2400,3000));

    ax8.set_yticks((44,89,134,179))
    ax8.set_yticklabels((-90,0,90,180))    

    ax7 = fig.add_subplot(2,1,1,sharex=ax8,sharey=ax8)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    #ax7.spines['left'].set_visible(False)
    plt.setp(ax7.get_xticklabels(), visible=False)
    plt.setp(ax7.get_yticklabels(), visible=False)
    
    im=ax7.imshow(poi_lik_tr2, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto');
    #ax7.plot(poi_md_tr2,'g+',markersize=size,label='Estimate')
    ax7.plot(hd_train,'bo',markersize=size,alpha=alpha,label='True')
    #ax7.plot(hd_train,'b',linewidth=0.5, label='True')
    ax7.set_title("Unnormalized Poisson", fontsize=14, fontweight='bold');
    
    axins = fig.add_axes([0.99, 0.25, 0.01, 0.65])  # left/right, up/down, width, height
    cbar = plt.colorbar(im,ax=[ax7,ax8],cax=axins,orientation='vertical',spacing='uniform')
    cbar.ax.tick_params(width=2,labelsize=18) 
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    #fig.colorbar(im, ax)

    plt.tight_layout()
    
    return fig

# depreciated
def plotPosterior(nTrain, hd, poi_mean_decode, sng_mean_decode, ber_mean_decode, poi_conf_interval, sng_conf_interval, ber_conf_interval, tb=250):
    T_bins_left = nTrain-tb;
    T_bins_right = nTrain+tb;
    fig = plt.figure(figsize=(30,10))

    plt.subplot(3,1,1)
    plt.plot(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
             poi_mean_decode[T_bins_left:T_bins_right],'x',label='estimated')
    plt.plot(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
             hd[T_bins_left:T_bins_right],'rx',alpha=0.1,label='true')
    plt.fill_between(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
                     poi_conf_interval[T_bins_left:T_bins_right,0],
                     poi_conf_interval[T_bins_left:T_bins_right,1],color="#b9cfe7", edgecolor="");
    plt.legend()
    plt.title("poisson");

    plt.subplot(3,1,2)
    plt.plot(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
             ber_mean_decode[T_bins_left:T_bins_right],'x',label='estimated')
    plt.plot(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
             hd[T_bins_left:T_bins_right],'rx',alpha=0.1,label='true')
    plt.fill_between(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
                     ber_conf_interval[T_bins_left:T_bins_right,0],
                     ber_conf_interval[T_bins_left:T_bins_right,1],color="#b9cfe7", edgecolor="");
    plt.title("bernoulli");

    plt.subplot(3,1,3)
    plt.plot(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
             sng_mean_decode[T_bins_left:T_bins_right],'x')
    plt.plot(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
             hd[T_bins_left:T_bins_right],'rx',alpha=0.1,label='true')
    plt.fill_between(np.linspace(T_bins_left,T_bins_right-1,T_bins_right-T_bins_left),
                     sng_conf_interval[T_bins_left:T_bins_right,0],
                     sng_conf_interval[T_bins_left:T_bins_right,1],color="#b9cfe7", edgecolor="")
    plt.title("spike and gamma");

    plt.tight_layout()
    return fig

def plotPos(pos,poi_mean_pos_decode,sng_mean_pos_decode,sngr_mean_pos_decode,ber_mean_pos_decode,nTrain, Tbins=1000):
    fig = plt.figure(figsize=(10,12))
    start = nTrain-Tbins;
    end=nTrain+Tbins;
    plt.subplot(4,1,1)
    plt.plot(pos[start:end])
    plt.plot(poi_mean_pos_decode[start:end], 'rx', alpha=0.5)
    plt.title('poisson')
    plt.subplot(4,1,2)
    plt.plot(pos[start:end])
    plt.plot(sng_mean_pos_decode[start:end], 'rx', alpha=0.5)
    plt.title('spike and gamma')
    plt.subplot(4,1,3)
    plt.plot(pos[start:end])
    plt.plot(sngr_mean_pos_decode[start:end], 'rx', alpha=0.5)
    plt.title('spike and gamma relax')

    plt.subplot(4,1,4)
    plt.plot(pos[start:end])
    plt.plot(ber_mean_pos_decode[start:end], 'rx', alpha=0.5)
    plt.title('bernoulli')
    plt.tight_layout()
    return fig

def plotPosRes(pos,poi_mean_pos_decode,sng_mean_pos_decode,sngr_mean_pos_decode,ber_mean_pos_decode,nTrain, Tbins=1000):
    fig = plt.figure(figsize=(10,12))
    start = nTrain-Tbins;
    end=nTrain+Tbins;
    plt.subplot(4,1,1)
    #plt.plot(pos[tempp][start:end])
    plt.plot((poi_mean_pos_decode-pos)[start:end], 'rx', alpha=0.5)
    plt.title('poisson')
    plt.subplot(4,1,2)
    #plt.plot(pos[tempp][start:end])
    plt.plot((sng_mean_pos_decode-pos)[start:end], 'rx', alpha=0.5)
    plt.title('spike and gamma')
    plt.subplot(4,1,3)
    #plt.plot(pos[tempp][start:end])
    plt.plot((sngr_mean_pos_decode-pos)[start:end], 'rx', alpha=0.5)
    plt.title('spike and gamma relax')    
    plt.subplot(4,1,4)
    #plt.plot(pos[tempp][start:end])
    plt.plot((ber_mean_pos_decode-pos)[start:end], 'rx', alpha=0.5)
    plt.title('bernoulli')
    plt.tight_layout()
    return fig

def plotCIcov_med(conf_rate_ci, width_med_ci, hdr = False):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    plt.subplot(1,2,1)

    plt.plot(width_med_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    plt.plot(width_med_ci[1,:,0], conf_rate_ci[1,:,0],label='sng rlx')
    plt.plot(width_med_ci[2,:,0], conf_rate_ci[2,:,0],label='poisson')
    plt.plot(width_med_ci[3,:,0], conf_rate_ci[3,:,0],label='bernoulli')
    plt.plot(width_med_ci[4,:,0], conf_rate_ci[4,:,0],label='gamma')

    plt.legend()
    plt.title('training data');
    
    if hdr:
        plt.xlabel('HDR width median')
        plt.ylabel('HDR coverage rate')
    else:
        plt.xlabel('CI width median')
        plt.ylabel('CI coverage rate')

    plt.subplot(1,2,2)
    plt.plot(width_med_ci[0,:,1], conf_rate_ci[0,:,1],label='sng')
    plt.plot(width_med_ci[1,:,1], conf_rate_ci[1,:,1],label='sng rlx')
    plt.plot(width_med_ci[2,:,1], conf_rate_ci[2,:,1],label='poisson')
    plt.plot(width_med_ci[3,:,1], conf_rate_ci[3,:,1],label='bernoulli')
    plt.plot(width_med_ci[4,:,1], conf_rate_ci[4,:,1],label='gamma')

    plt.legend()
    plt.title('test data');
    if hdr:
        plt.xlabel('HDR width median')
        plt.ylabel('HDR coverage rate')
    else:
        plt.xlabel('CI width median')
        plt.ylabel('CI coverage rate')

    plt.tight_layout()
    return fig

def plotCIcov_mean(conf_rate_ci, width_mean_ci, hdr=False, drop=False):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    ax = fig.add_subplot(111);
    if hdr:
        ax.set_xlabel('HDR width mean',fontsize=sz,fontweight='normal');
    else:
        ax.set_xlabel('CI width mean',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax1.plot(width_mean_ci[1,:,0], conf_rate_ci[1,:,0],'b-',linewidth=3,label='Poisson')
    ax1.plot(width_mean_ci[2,:,0], conf_rate_ci[2,:,0],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax1.plot(width_mean_ci[3,:,0], conf_rate_ci[3,:,0],'g-',linewidth=3,label='Gamma')
    ax1.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],'r-',linewidth=3,label='SNG')

    ax1.legend(fontsize=sz);

    if hdr:
        #ax1.set_xlabel("HDR width mean",fontsize=14,fontweight='normal')
        ax1.set_ylabel("HDR coverage rate",fontsize=sz,fontweight='normal')
    else:
        #ax1.set_xlabel("CI width mean",fontsize=14,fontweight='normal')
        ax1.set_ylabel("CI coverage rate",fontsize=sz,fontweight='normal')
        
    ax2 = fig.add_subplot(122, sharey=ax1, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), visible = False)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax2.plot(width_mean_ci[1,:,1], conf_rate_ci[1,:,1],'b-',linewidth=3,label='Poisson')
    ax2.plot(width_mean_ci[2,:,1], conf_rate_ci[2,:,1],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax2.plot(width_mean_ci[3,:,1], conf_rate_ci[3,:,1],'g-',linewidth=3,label='Gamma')
    ax2.plot(width_mean_ci[0,:,1], conf_rate_ci[0,:,1],'r-',linewidth=3,label='SNG')

    plt.tight_layout()
    return fig

def plotCIcov_meanv(conf_rate_ci, width_mean_ci, hdr=False, drop=False):
    fig = plt.figure(figsize=(5,7))
    sz=15;
    ax = fig.add_subplot(111);
    if hdr:
        ax.set_xlabel('HDR width mean',fontsize=sz,fontweight='normal');
    else:
        ax.set_xlabel('CI width mean',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(211)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible = False)
    plt.setp(ax1.get_yticklabels(), visible = False)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    #ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax1.plot(width_mean_ci[1,:,0], conf_rate_ci[1,:,0],'b-',linewidth=3,label='Poisson')
    ax1.plot(width_mean_ci[2,:,0], conf_rate_ci[2,:,0],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax1.plot(width_mean_ci[3,:,0], conf_rate_ci[3,:,0],'g-',linewidth=3,label='Gamma')
    ax1.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],'r-',linewidth=3,label='SNG')

    #ax1.legend(fontsize=sz);

    if hdr:
        #ax1.set_xlabel("HDR width mean",fontsize=14,fontweight='normal')
        ax.set_ylabel("HDR coverage rate",fontsize=sz,fontweight='normal',labelpad=20)
    else:
        #ax1.set_xlabel("CI width mean",fontsize=14,fontweight='normal')
        ax.set_ylabel("CI coverage rate",fontsize=sz,fontweight='normal',labelpad=20)
        
    ax2 = fig.add_subplot(212, sharey=ax1, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.setp(ax2.get_xticklabels(), fontsize=sz)
    plt.setp(ax2.get_yticklabels(), fontsize=sz)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    #ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax2.plot(width_mean_ci[1,:,1], conf_rate_ci[1,:,1],'b-',linewidth=3,label='Poisson')
    ax2.plot(width_mean_ci[2,:,1], conf_rate_ci[2,:,1],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax2.plot(width_mean_ci[3,:,1], conf_rate_ci[3,:,1],'g-',linewidth=3,label='Gamma')
    ax2.plot(width_mean_ci[0,:,1], conf_rate_ci[0,:,1],'r-',linewidth=3,label='SNG')

    plt.tight_layout()
    return fig

def plotCIcov_lev(conf_rate_ci, conf_level_list, hdr=False, real=False, drop=False):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    ax = fig.add_subplot(111);
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    if not real:
        ax1.set_ylim((conf_rate_ci.min()-0.1,3.1));
        ax1.set_xlim((conf_rate_ci.min()-0.1,3.1));
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax1.plot(conf_level_list, conf_rate_ci[1,:,0],'b-',linewidth=3,label='Poisson')
    ax1.plot(conf_level_list, conf_rate_ci[2,:,0],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax1.plot(conf_level_list, conf_rate_ci[3,:,0],'g-',linewidth=3,label='Gamma')
    ax1.plot(conf_level_list, conf_rate_ci[0,:,0],'r-',linewidth=3,label='SNG')
    if hdr:
        ax1.set_ylabel('HDR coverage rate (log scale)',fontsize=sz,fontweight='normal');
    else:
        ax1.set_ylabel('CI coverage rate (log scale)',fontsize=sz,fontweight='normal');

    #ax1.legend(fontsize=14);

    if hdr:
        ax.set_xlabel("HDR confidence level (log scale)",fontsize=sz,fontweight='normal')
    else:
        ax.set_xlabel("CI confidence level (log scale)",fontsize=sz,fontweight='normal')

    ax1.set_xticks((1,2,3));
    ax1.set_xticklabels(('0.9', '0.99', '0.999'));   
    plt.draw();
    if real:
        ticks = ax1.get_yticks();
        temp = np.round(1-np.power(10, -ticks),2);
        labels = [str(ii) for ii in temp];
        ax1.set_yticks(ticks[1:-1]);    
        ax1.set_yticklabels(labels[1:-1]);
    else:
        temp = np.array([0.9,0.99,0.999]);
        ticks = -np.log10(1-temp);
        labels = [str(ii) for ii in temp];
        ax1.set_yticks(ticks[:]);    
        ax1.set_yticklabels(labels[:]);
        ax1.plot([1,3], [1,3], ls="--", c="black");

    #print(labels);
    
    ax2 = fig.add_subplot(122, sharey=ax1, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), visible = False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if not real:
        ax2.set_ylim((conf_rate_ci.min()-0.1,3.1));
        ax2.set_xlim((conf_rate_ci.min()-0.1,3.1));
        ax2.plot([1,3], [1,3], ls="--", c="black");

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax2.plot(conf_level_list, conf_rate_ci[1,:,1],'b-',linewidth=3,label='Poisson')
    ax2.plot(conf_level_list, conf_rate_ci[2,:,1],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax2.plot(conf_level_list, conf_rate_ci[3,:,1],'g-',linewidth=3,label='Gamma')
    ax2.plot(conf_level_list, conf_rate_ci[0,:,1],'r-',linewidth=3,label='SNG')

    plt.tight_layout()
    return fig

def plotCIcov_levv(conf_rate_ci, conf_level_list, hdr=False, real=False, drop=False):
    fig = plt.figure(figsize=(5,7))
    sz=15;
    ax = fig.add_subplot(111);
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(211)
    
    if not real:
        ax1.set_ylim((conf_rate_ci.min()-0.1,3.1));
        ax1.set_xlim((conf_rate_ci.min()-0.1,3.1));
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax1.set_title('',fontsize=sz,fontweight='bold');
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax1.plot(conf_level_list, conf_rate_ci[1,:,0],'b-',linewidth=3,label='Poisson')
    ax1.plot(conf_level_list, conf_rate_ci[2,:,0],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax1.plot(conf_level_list, conf_rate_ci[3,:,0],'g-',linewidth=3,label='Gamma')
    ax1.plot(conf_level_list, conf_rate_ci[0,:,0],'r-',linewidth=3,label='SNG')
    if hdr:
        ax.set_ylabel('HDR coverage rate (log scale)',fontsize=sz,fontweight='normal',labelpad=26);
    else:
        ax.set_ylabel('CI coverage rate (log scale)',fontsize=sz,fontweight='normal',labelpad=26);

    #ax1.legend(fontsize=14);

    if hdr:
        ax.set_xlabel("HDR confidence level (log scale)",fontsize=sz,fontweight='normal')
    else:
        ax.set_xlabel("CI confidence level (log scale)",fontsize=sz,fontweight='normal')


    #print(labels);
    
    ax2 = fig.add_subplot(212, sharey=ax1, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), fontsize=sz)
    plt.setp(ax2.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))               
    ax2.set_title('',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if not real:
        ax2.set_ylim((conf_rate_ci.min()-0.1,3.1));
        ax2.set_xlim((conf_rate_ci.min()-0.1,3.1));
        ax2.plot([1,3], [1,3], ls="--", c="black");
    ax2.set_xticks((1,2,3));
    ax2.set_xticklabels(('0.9', '0.99', '0.999'));   
    plt.draw();
    if real:
        ticks = ax2.get_yticks();
        temp = np.round(1-np.power(10, -ticks),2);
        labels = [str(ii) for ii in temp];
        ax2.set_yticks(ticks[1:-1]);    
        ax2.set_yticklabels(labels[1:-1]);
    else:
        temp = np.array([0.9,0.99,0.999]);
        ticks = -np.log10(1-temp);
        labels = [str(ii) for ii in temp];
        ax2.set_yticks(ticks[:]);    
        ax2.set_yticklabels(labels[:]);
        ax2.plot([1,3], [1,3], ls="--", c="black");

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax2.plot(conf_level_list, conf_rate_ci[1,:,1],'b-',linewidth=3,label='Poisson')
    ax2.plot(conf_level_list, conf_rate_ci[2,:,1],'y-',linewidth=3,label='Bernoulli')
    if not drop:
        ax2.plot(conf_level_list, conf_rate_ci[3,:,1],'g-',linewidth=3,label='Gamma')
    ax2.plot(conf_level_list, conf_rate_ci[0,:,1],'r-',linewidth=3,label='SNG')

    plt.tight_layout()
    return fig

def plotErrSmin(error_mean_mat, error_mean_bs,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    ax = fig.add_subplot(111);
    ax.set_xlabel('Minimum spike size $s_{min}$',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels((0,0.25,0.5,1,2));    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    #ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax1.plot(error_mean_mat[:,0,0],'b-',linewidth=3,label='Poisson')
    ax1.plot(error_mean_mat[:,0,1],'y-',linewidth=3,label='Bernoulli')
    ax1.plot(error_mean_mat[:,0,4],'g-',linewidth=3,label='Gamma')
    ax1.plot(error_mean_mat[:,0,3],'r-',linewidth=3,label='SNG')
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Pois on spikes');
        ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-1., max(error_mean_mat.max(), error_mean_bs.max())+1.));
    else:
        ax1.set_ylim((error_mean_mat.min()-1, error_mean_mat.max()+1));
    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Decoding error (degree $^{\circ}$)",fontsize=sz,fontweight='normal')
    plt.draw();
    labels = [l.get_text() for l in ax1.get_yticklabels()]
    #print(labels);
    ticks = ax1.get_yticks();
    if error_mean_bs is not None:
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks[0] = error_mean_bs[0];
        ax1.set_yticks(ticks[:-1]);
        ax1.set_yticklabels(labels[:-1]);
    else:
        ax1.set_yticks(ticks[1:-1]);
        ax1.set_yticklabels(labels[1:-1]);
    
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    #plt.setp(ax2.get_yticklabels(), visible = False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Pois on spikes');
        ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-1., max(error_mean_mat.max(), error_mean_bs.max())+1.));
    else:
        ax2.set_ylim((error_mean_mat.min()-1., error_mean_mat.max()+1.));

    #plt.plot(width_mean_ci[0,:,0], conf_rate_ci[0,:,0],label='sng')
    ax2.plot(error_mean_mat[:,1,0],'b-',linewidth=3,label='Poisson')
    ax2.plot(error_mean_mat[:,1,1],'y-',linewidth=3,label='Bernoulli')
    ax2.plot(error_mean_mat[:,1,4],'g-',linewidth=3,label='Gamma')
    ax2.plot(error_mean_mat[:,1,3],'r-',linewidth=3,label='SNG')
    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        plt.setp(ax2.get_yticklabels(), fontsize=sz);
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);
    else:
        #print(ticks);
        plt.setp(ax2.get_yticklabels(), visible = False)
        ax2.set_yticks(ticks[1:-1]);
        ax2.set_yticklabels(labels[1:-1]);        

    plt.tight_layout()
    return fig

def plotErrBarSmin(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, real=False,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Minimum spike size $s_{min}$',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels((0,0.25,0.5,1,2));    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    
    if not real:
        ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Pois on spikes');
        ax1.set_ylim((min(np.delete(error_mean_mat,2,axis=2).min(), error_mean_bs.min())-1., max(np.delete(error_mean_mat,2,axis=2).max(), error_mean_bs.max())+1.));
    else:
        ax1.set_ylim((np.delete(error_mean_mat,2,axis=2).min()-1, np.delete(error_mean_mat,2,axis=2).max()+1));

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Mean abs error (degree $^{\circ}$)",fontsize=sz,fontweight='normal')
    plt.draw();
    labels = [l.get_text() for l in ax1.get_yticklabels()]
    ticks = ax1.get_yticks();
    
    print(labels);
    if error_mean_bs is not None:
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks[0] = error_mean_bs[0];
        ax1.set_yticks(ticks[:-1]);
        ax1.set_yticklabels(labels[:-1]);
    else:
        ax1.set_yticks(ticks[1:-1]);
        ax1.set_yticklabels(labels[1:-1]);
        
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    #plt.setp(ax2.get_yticklabels(), visible = False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Pois on spikes');
        ax2.set_ylim((min(np.delete(error_mean_mat,2,axis=2).min(), error_mean_bs.min())-1., max(np.delete(error_mean_mat,2,axis=2).max(), error_mean_bs.max())+1.));
    else:
        ax2.set_ylim((np.delete(error_mean_mat,2,axis=2).min()-1., np.delete(error_mean_mat,2,axis=2).max()+1.));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    
    if not real:
        ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        plt.setp(ax2.get_yticklabels(), fontsize=sz);
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);
    else:
        plt.setp(ax2.get_yticklabels(), visible = False)
        ax2.set_yticks(ticks[1:-1]);
        ax2.set_yticklabels(labels[1:-1]);

    plt.tight_layout()
    return fig

def plotErrBarSminv(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, real=False,fsz=15):
    fig = plt.figure(figsize=(5,7))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Minimum spike size $s_{min}$',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(211)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels((0,0.25,0.5,1,2));    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    
    if not real:
        ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Pois on spikes');
        ax1.set_ylim((min(np.delete(error_mean_mat,2,axis=2).min(), error_mean_bs.min())-1., max(np.delete(error_mean_mat,2,axis=2).max(), error_mean_bs.max())+1.));
    else:
        ax1.set_ylim((np.delete(error_mean_mat,2,axis=2).min()-1, np.delete(error_mean_mat,2,axis=2).max()+1));

    ax1.legend(fontsize=fsz);
    if error_mean_bs is not None:
        ax.set_ylabel("Mean abs error (degree $^{\circ}$)",fontsize=sz,fontweight='normal',labelpad=18)
    else:
        ax.set_ylabel("Mean abs error (degree $^{\circ}$)",fontsize=sz,fontweight='normal')
    plt.draw();
    labels = [l.get_text() for l in ax1.get_yticklabels()]
    ticks = ax1.get_yticks();
    
    print(labels);
    if error_mean_bs is not None:
        plt.setp(ax1.get_yticklabels(), fontsize=sz)
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[0],2));
        ticks2[0] = error_mean_bs[0];
        for ii in range(len(labels)-1):
            labels2[ii+1]='';
        ax1.set_yticks(ticks2[:-1]);
        ax1.set_yticklabels(labels2[:-1]);
    else:
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.set_yticks(ticks[1:-1]);
        ax1.set_yticklabels(labels[1:-1]);
        
    ax2 = fig.add_subplot(212, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), fontsize=sz)
    #plt.setp(ax2.get_yticklabels(), visible = False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Pois on spikes');
        ax2.set_ylim((min(np.delete(error_mean_mat,2,axis=2).min(), error_mean_bs.min())-1., max(np.delete(error_mean_mat,2,axis=2).max(), error_mean_bs.max())+1.));
    else:
        ax2.set_ylim((np.delete(error_mean_mat,2,axis=2).min()-1., np.delete(error_mean_mat,2,axis=2).max()+1.));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    
    if not real:
        ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        plt.setp(ax2.get_yticklabels(), fontsize=sz);
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        #for ii in range(len(labels2)-1):
        #    labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);
    else:
        plt.setp(ax2.get_yticklabels(), fontsize=sz)
        ax2.set_yticks(ticks[1:-1]);
        ax2.set_yticklabels(labels[1:-1]);

    plt.tight_layout()
    return fig

def plotErrBarSminL1(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, labels,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Penalty parameter',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels(labels);    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    #ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='SNG smin=0');
        ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-1., max(error_mean_mat.max(), error_mean_bs.max())+1.));
    else:
        ax1.set_ylim((error_mean_mat.min()-1, error_mean_mat.max()+1));

    #ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Mean abs error (degree $^{\circ}$)",fontsize=sz,fontweight='normal')
    plt.draw();
    
    #print(labels);
    if error_mean_bs is not None:
        labels = [l.get_text() for l in ax1.get_yticklabels()]
        #print(labels);
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks = ax1.get_yticks();
        ticks[0] = error_mean_bs[0];
        if (ticks[1] - ticks[0] < 0.05):
            ticks[1] = ticks[0];
            labels[1] = labels[0];
            ax1.set_yticks(ticks[1:-1]);
            ax1.set_yticklabels(labels[1:-1]);
            #ax1.set_yticks(ticks[0] + ticks[2:-1]);
            #ax1.set_yticklabels(labels[0] + labels[2:-1]);        
        else:
            ax1.set_yticks(ticks[1:-1]);
            ax1.set_yticklabels(labels[1:-1]);
        
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    #plt.setp(ax2.get_yticklabels(), fontsize=sz)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='SNG smin=0');
        ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-1., max(error_mean_mat.max(), error_mean_bs.max())+1.));
    else:
        ax2.set_ylim((error_mean_mat.min()-1., error_mean_mat.max()+1.));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    #ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        plt.setp(ax2.get_yticklabels(), fontsize=sz);
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);
        
    #else:
    #    #print(ticks);
    #    ticks2 = ticks.copy();
    #    labels2 = labels.copy();
    #    #labels2[0] = str(np.round(error_mean_bs[1],2));
    #    for ii in range(len(labels2)):
    #        labels2[ii]='';
    #    #ticks = ax2.get_yticks();
    #    #ticks2[0] = error_mean_bs[1];
    #    ax2.set_yticks(ticks2[1:-1]);
    #    ax2.set_yticklabels(labels2[1:-1]);        
    ax2.legend(fontsize=fsz);
    plt.tight_layout()
    return fig

def plotPostErrBarSmin(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Minimum spike size $s_{min}$',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels((0,0.25,0.5,1,2));    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Pois on spikes');
        ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.8, max(error_mean_mat.max(), error_mean_bs.max())+0.8));
    else:
        ax1.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Log posterior prob mean",fontsize=sz,fontweight='normal')
    plt.draw();
    labels = [l.get_text() for l in ax1.get_yticklabels()]
    ticks = ax1.get_yticks();
    
    if error_mean_bs is not None:
        labels = [l.get_text() for l in ax1.get_yticklabels()]
        #print(labels);
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks = ax1.get_yticks();
        ticks[0] = error_mean_bs[0];
        ax1.set_yticks(ticks[:-1]);
        ax1.set_yticklabels(labels[:-1]);
    else:
        ax1.set_yticks(ticks[1:-1]);
        ax1.set_yticklabels(labels[1:-1]);
        
    
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    #plt.setp(ax2.get_yticklabels(), visible = False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Pois on spikes');
        ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.8, max(error_mean_mat.max(), error_mean_bs.max())+0.8));
    else:
        ax2.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        plt.setp(ax2.get_yticklabels(), fontsize=sz);
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);
    else:
        plt.setp(ax2.get_yticklabels(), visible = False)
        ax2.set_yticks(ticks[1:-1]);
        ax2.set_yticklabels(labels[1:-1]);
    
    plt.tight_layout()
    return fig


def plotPostErrBarSminReal(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, drop=False,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Minimum spike size $s_{min}$',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels((0,0.25,0.5,1,2));    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    
    if not drop:
        if error_mean_bs is not None:
            ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Random guess');
            ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
        else:
            ax1.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));
    else:
        if error_mean_bs is not None:
            ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Random guess');
            ax1.set_ylim((min(error_mean_mat[:,:,:-1].min(), error_mean_bs.min())-1.3, max(error_mean_mat[:,:,:-1].max(), error_mean_bs.max())+0.3));
        else:
            ax1.set_ylim((error_mean_mat[:,:,:-1].min()-1.3, error_mean_mat[:,:,:-1].max()+0.3));
        

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Log posterior prob mean",fontsize=sz,fontweight='normal')
    plt.draw();
    if error_mean_bs is not None:
        labels = [l.get_text() for l in ax1.get_yticklabels()]
        #print(labels);
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks = ax1.get_yticks();
        ticks[0] = error_mean_bs[0];
        if (np.abs(ticks[1] - ticks[0]) < 0.1):
            ticks[1] = ticks[0];
            labels[1] = labels[0];
            ax1.set_yticks(ticks[1:-1]);
            ax1.set_yticklabels(labels[1:-1]);
            #ax1.set_yticks(ticks[0] + ticks[2:-1]);
            #ax1.set_yticklabels(labels[0] + labels[2:-1]);        
        else:
            ax1.set_yticks(ticks[:-1]);
            ax1.set_yticklabels(labels[:-1]);
    
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), visible = False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    if not drop:
        if error_mean_bs is not None:
            ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Random guess');
            ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
        else:
            ax2.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));
    else:
        if error_mean_bs is not None:
            ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Random guess');
            ax2.set_ylim((min(error_mean_mat[:,:,:-1].min(), error_mean_bs.min())-1.3, max(error_mean_mat[:,:,:-1].max(), error_mean_bs.max())+0.3));
        else:
            ax2.set_ylim((error_mean_mat[:,:,:-1].min()-1.3, error_mean_mat[:,:,:-1].max()+0.3));
    
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);
    else:
        ax2.set_yticks(ticks[:-1]);
        ax2.set_yticklabels(labels[:-1]);

    plt.tight_layout()
    return fig

def plotPostErrBarSminRealMed(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, drop=True,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Minimum spike size $s_{min}$',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels((0,0.25,0.5,1,2));    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if not drop:
        if error_mean_bs is not None:
            ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Random guess');
            ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
        else:
            ax1.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));
    else:
        if error_mean_bs is not None:
            ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Random guess');
            ax1.set_ylim((min(error_mean_mat[:,:,:-1].min(), error_mean_bs.min())-0.3, max(error_mean_mat[:,:,:-1].max(), error_mean_bs.max())+0.3));
        else:
            ax1.set_ylim((error_mean_mat[:,:,:-1].min()-0.3, error_mean_mat[:,:,:-1].max()+0.3));

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Log posterior prob median",fontsize=sz,fontweight='normal')
    plt.draw();
    if error_mean_bs is not None:
        labels = [l.get_text() for l in ax1.get_yticklabels()]
        #print(labels);
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks = ax1.get_yticks();
        ticks[0] = error_mean_bs[0];
        if (ticks[1] - ticks[0] < 0.05):
            ticks[1] = ticks[0];
            labels[1] = labels[0];
            ax1.set_yticks(ticks[1:-1]);
            ax1.set_yticklabels(labels[1:-1]);
            #ax1.set_yticks(ticks[0] + ticks[2:-1]);
            #ax1.set_yticklabels(labels[0] + labels[2:-1]);        
        else:
            ax1.set_yticks(ticks[:-1]);
            ax1.set_yticklabels(labels[:-1]);
    
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), visible = False)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if not drop:
        if error_mean_bs is not None:
            ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Random guess');
            ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
        else:
            ax2.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));
    else:
        if error_mean_bs is not None:
            ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Random guess');
            ax2.set_ylim((min(error_mean_mat[:,:,:-1].min(), error_mean_bs.min())-0.3, max(error_mean_mat[:,:,:-1].max(), error_mean_bs.max())+0.3));
        else:
            ax2.set_ylim((error_mean_mat[:,:,:-1].min()-0.3, error_mean_mat[:,:,:-1].max()+0.3));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);

    plt.tight_layout()
    return fig

def plotPostErrBarSminRealMedL1(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, labels,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Penalty parameter',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    ax1.set_xticks((0,1,2,3,4));
    ax1.set_xticklabels(labels);    
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    #ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='SNG smin=0');
        ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
    else:
        ax1.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Log posterior prob median",fontsize=14,fontweight='normal')
    plt.draw();
    if error_mean_bs is not None:
        labels = [l.get_text() for l in ax1.get_yticklabels()]
        #print(labels);
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks = ax1.get_yticks();
        ticks[0] = error_mean_bs[0];
        if (ticks[1] - ticks[0] < 0.1):
            ticks[1] = ticks[0];
            labels[1] = labels[0];
            ax1.set_yticks(ticks[1:-1]);
            ax1.set_yticklabels(labels[1:-1]);
            #ax1.set_yticks(ticks[0] + ticks[2:-1]);
            #ax1.set_yticklabels(labels[0] + labels[2:-1]);        
        else:
            ax1.set_yticks(ticks[:-1]);
            ax1.set_yticklabels(labels[:-1]);    
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), fontsize=sz)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='SNG smin=0');
        ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
    else:
        ax2.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    #ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);

    plt.tight_layout()
    return fig

def plotErrBarSS(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, num=3,fsz=15):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Number of neurons',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    if num==3:
        ax1.set_xticks((0,1,2));
        ax1.set_xticklabels((50, 100, 222));    
    
    if num==4:
        ax1.set_xticks((0,1,2,3));
        ax1.set_xticklabels((50, 100, 200, 311));    
        
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Random guess');
        ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-1., max(error_mean_mat.max(), error_mean_bs.max())+1.));
    else:
        ax1.set_ylim((error_mean_mat.min()-1, error_mean_mat.max()+1));

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Mean abs error (degree $^{\circ}$)",fontsize=14,fontweight='normal')
    plt.draw();
    labels = [l.get_text() for l in ax1.get_yticklabels()]
    ticks = ax1.get_yticks();
    
    #print(labels);
    if error_mean_bs is not None:
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks[0] = error_mean_bs[0];
        ax1.set_yticks(ticks[:-1]);
        ax1.set_yticklabels(labels[:-1]);
    else:
        ax1.set_yticks(ticks[1:-1]);
        ax1.set_yticklabels(labels[1:-1]);
        
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), fontsize=sz)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Random guess');
        ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-1., max(error_mean_mat.max(), error_mean_bs.max())+1.));
    else:
        ax2.set_ylim((error_mean_mat.min()-1., error_mean_mat.max()+1.));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);
    else:
        #print(ticks);
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        #labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)):
            labels2[ii]='';
        #ticks = ax2.get_yticks();
        #ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[1:-1]);
        ax2.set_yticklabels(labels2[1:-1]);        

    plt.tight_layout()
    return fig

def plotPostErrBarSSReal(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, num=3,fsz=15,drop=False):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Number of neurons',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    
    if num==3:
        ax1.set_xticks((0,1,2));
        ax1.set_xticklabels((50, 100, 222));
        
    if num == 4:
        ax1.set_xticks((0,1,2,3));
        ax1.set_xticklabels((50, 100, 200, 311));        
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Random guess');
        ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
    else:
        ax1.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Log posterior prob mean",fontsize=sz,fontweight='normal')
    plt.draw();
    if error_mean_bs is not None:
        labels = [l.get_text() for l in ax1.get_yticklabels()]
        #print(labels);
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks = ax1.get_yticks();
        ticks[0] = error_mean_bs[0];
        ax1.set_yticks(ticks[:-1]);
        ax1.set_yticklabels(labels[:-1]);
    
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), fontsize=sz)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Random guess');
        ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
    else:
        ax2.set_ylim((error_mean_mat.min()-0.3, error_mean_mat.max()+0.3));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma');
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);

    plt.tight_layout()
    return fig

def plotPostErrBarSSRealMed(error_mean_mat, error_mean_bs, error_var_mat, nTrain, nTest, num=3,fsz=15,drop=False):
    fig = plt.figure(figsize=(8,4))
    sz=15;
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    ax = fig.add_subplot(111);
    ax.set_xlabel('Number of neurons',fontsize=sz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax1 = fig.add_subplot(121)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=sz)
    plt.setp(ax1.get_yticklabels(), fontsize=sz)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=5,prune=None,interger=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    if num==3:
        ax1.set_xticks((0,1,2));
        ax1.set_xticklabels((50, 100, 222));
        
    if num == 4:
        ax1.set_xticks((0,1,2,3));
        ax1.set_xticklabels((50, 100, 200, 311));        
    
    ax1.set_title('Training data',fontsize=sz,fontweight='bold');
    ax1.errorbar(xx,error_mean_mat[:,0,0],yerr=error_var_mat[:,0,0]/np.sqrt(nTrain),c='b',linewidth=3, label='Poisson');
    ax1.errorbar(xx,error_mean_mat[:,0,1],yerr=error_var_mat[:,0,1]/np.sqrt(nTrain),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax1.errorbar(xx,error_mean_mat[:,0,4],yerr=error_var_mat[:,0,4]/np.sqrt(nTrain),c='g',linewidth=3,label='Gamma');
    ax1.errorbar(xx,error_mean_mat[:,0,3],yerr=error_var_mat[:,0,3]/np.sqrt(nTrain),c='r',linewidth=3,label='SNG');
    if error_mean_bs is not None:
        ax1.axhline(y=error_mean_bs[0],linestyle='--',linewidth=3,label='Random guess');
        ax1.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
    else:
        ax1.set_ylim((error_mean_mat.min()-0., error_mean_mat.max()+0.3));

    ax1.legend(fontsize=fsz);
    ax1.set_ylabel("Log posterior prob median",fontsize=sz,fontweight='normal')
    plt.draw();
    if error_mean_bs is not None:
        labels = [l.get_text() for l in ax1.get_yticklabels()]
        #print(labels);
        labels[0] = str(np.round(error_mean_bs[0],2));
        ticks = ax1.get_yticks();
        ticks[0] = error_mean_bs[0];
        if (ticks[1] - ticks[0] < 0.05):
            ticks[1] = ticks[0];
            labels[1] = labels[0];
            ax1.set_yticks(ticks[1:-1]);
            ax1.set_yticklabels(labels[1:-1]);
            #ax1.set_yticks(ticks[0] + ticks[2:-1]);
            #ax1.set_yticklabels(labels[0] + labels[2:-1]);        
        else:
            ax1.set_yticks(ticks[:-1]);
            ax1.set_yticklabels(labels[:-1]);
    
    ax2 = fig.add_subplot(122, sharex=ax1);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), fontsize=sz)
    #ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    #ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.set_title('Test data',fontsize=sz,fontweight='bold');
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if error_mean_bs is not None:
        ax2.axhline(y=error_mean_bs[1],linestyle='--',linewidth=3,label='Random guess');
        ax2.set_ylim((min(error_mean_mat.min(), error_mean_bs.min())-0.3, max(error_mean_mat.max(), error_mean_bs.max())+0.3));
    else:
        ax2.set_ylim((error_mean_mat.min()-0., error_mean_mat.max()+0.3));
    ax2.errorbar(xx,error_mean_mat[:,1,0],yerr=error_var_mat[:,1,0]/np.sqrt(nTest),c='b',linewidth=3,label='Poisson');
    ax2.errorbar(xx,error_mean_mat[:,1,1],yerr=error_var_mat[:,1,1]/np.sqrt(nTest),c='y',linewidth=3,label='Bernoulli');
    if not drop:
        ax2.errorbar(xx,error_mean_mat[:,1,4],yerr=error_var_mat[:,1,4]/np.sqrt(nTest),c='g',linewidth=3,label='Gamma'); 
    ax2.errorbar(xx,error_mean_mat[:,1,3],yerr=error_var_mat[:,1,3]/np.sqrt(nTest),c='r',linewidth=3,label='SNG');

    plt.draw();
    #labels = [l.get_text() for l in ax2.get_yticklabels()]
    #print(labels);
    if error_mean_bs is not None:
        ticks2 = ticks.copy();
        labels2 = labels.copy();
        labels2[0] = str(np.round(error_mean_bs[1],2));
        for ii in range(len(labels2)-1):
            labels2[ii+1]='';
        #ticks = ax2.get_yticks();
        ticks2[0] = error_mean_bs[1];
        ax2.set_yticks(ticks2[:-1]);
        ax2.set_yticklabels(labels2[:-1]);

    plt.tight_layout()
    return fig

def barComp(bars1,bars2,bars3,yer1,yer2,yer3,nTest,ylabel):
    # width of the bars
    barWidth = 0.15

    sz=15;
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    fig = plt.figure(figsize=(8,4));
    ax = fig.add_subplot(111);
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Create blue bars
    ax.bar(r1, bars1, width = barWidth, color = 'red', edgecolor = '', yerr=yer1/np.sqrt(nTest), capsize=7, label='unnormalized')

    # Create cyan bars
    ax.bar(r2, bars2, width = barWidth, color = '#b9cfe7', edgecolor = '', yerr=yer2/np.sqrt(nTest), capsize=7, label='noise level')

    # Create cyan bars
    ax.bar(r3, bars3, width = barWidth, color = 'blue', edgecolor = '', yerr=yer3/np.sqrt(nTest), capsize=7, label='fano factor')

    # general layout
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Poisson', 'Bernoulli', 'Gamma','SNG'],fontsize=sz)
    plt.ylabel(ylabel,fontsize=sz)
    #plt.legend(fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=sz)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    
    return fig

def barCompBA(bars1,bars2,bars3,yer1,yer2,yer3,nTest,ylabel,pos):
    # width of the bars
    barWidth = 0.15

    sz=15;
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    fig = plt.figure(figsize=(8,4));
    ax0 = fig.add_subplot(111);
    ax0.set_ylabel(ylabel,fontsize=sz,fontweight='normal',labelpad=15);
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);
    
    ax = fig.add_subplot(211);
    ax2 = fig.add_subplot(212,sharex=ax);
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)    
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(top=False, bottom=False, right=False);
    
    # Create blue bars
    ax.bar(r1, bars1, width = barWidth, color = 'red', edgecolor = '', yerr=yer1/np.sqrt(nTest), capsize=7, label='unnormalized')
    # Create cyan bars
    ax.bar(r2, bars2, width = barWidth, color = '#b9cfe7', edgecolor = '', yerr=yer2/np.sqrt(nTest), capsize=7, label='noise level')
    # Create cyan bars
    ax.bar(r3, bars3, width = barWidth, color = 'blue', edgecolor = '', yerr=yer3/np.sqrt(nTest), capsize=7, label='fano factor')

    ax2.bar(r1, bars1, width = barWidth, color = 'red', edgecolor = '', yerr=yer1/np.sqrt(nTest), capsize=7, label='unnormalized')
    # Create cyan bars
    ax2.bar(r2, bars2, width = barWidth, color = '#b9cfe7', edgecolor = '', yerr=yer2/np.sqrt(nTest), capsize=7, label='noise level')
    # Create cyan bars
    ax2.bar(r3, bars3, width = barWidth, color = 'blue', edgecolor = '', yerr=yer3/np.sqrt(nTest), capsize=7, label='fano factor')

    ax2.set_ylim(bars1.min()-2,bars1.min()+2);
    ax.set_ylim(pos, 0)
    
    # general layout
    plt.setp(ax2.get_xticklabels(), fontsize=14);
    plt.setp(ax.get_xticklabels(), visible=False);
    ax2.set_xticks([r + barWidth for r in range(len(bars1))])
    ax2.set_xticklabels(['Poisson', 'Bernoulli', 'Gamma','SNG']);
    ax2.legend(fontsize=sz)
    plt.setp(ax.get_yticklabels(), fontsize=sz)
    plt.setp(ax2.get_yticklabels(), fontsize=sz)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))     
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,interger=True))
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
 
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    
    plt.tight_layout()
    return fig
## aevb plot functions
def plotRMSEVAE(error_mean_mat, error_mean_bs, xticks=[0,1],lsz=12):
    fig = plt.figure(figsize=(4,4))
    fsz = 14;
    ax = fig.add_subplot(111);
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #plt.plot((1-error_mean_mat[:,1,:].min(axis=-1))*100,linewidth=3,label='Normalized Pois')
    #plt.plot((1-error_mean_mat[:,0,:].min(axis=-1))*100,linewidth=3,label='Constrained SNG')
    #plt.plot((1-error_mean_mat[:,2,:].min(axis=-1))*100,linewidth=3,label='SNG')
    #plt.plot((1-error_mean_mat[:,1,:].mean(axis=-1))*100,linewidth=3,label='Normalized Pois')
    #plt.plot((1-error_mean_mat[:,0,:].mean(axis=-1))*100,linewidth=3,label='Constrained SNG')
    #plt.plot((1-error_mean_mat[:,2,:].mean(axis=-1))*100,linewidth=3,label='SNG')
    xx = np.linspace(0,error_mean_mat.shape[0]-1,error_mean_mat.shape[0]);
    kk=error_mean_mat.shape[-1];
    plt.errorbar(xx,(1-error_mean_mat[:,1,:].mean(axis=-1))*100,yerr=np.sqrt(error_mean_mat[:,1,:].var(axis=-1)/kk)*100,linewidth=3,c='b',label='Normalized Pois');
    plt.errorbar(xx,(1-error_mean_mat[:,0,:].mean(axis=-1))*100,yerr=np.sqrt(error_mean_mat[:,0,:].var(axis=-1)/kk)*100,linewidth=3,c='g',label='Constrained SNG');    
    plt.errorbar(xx,(1-error_mean_mat[:,2,:].mean(axis=-1))*100,yerr=np.sqrt(error_mean_mat[:,2,:].var(axis=-1)/kk)*100,linewidth=3,c='r',label='SNG');
    plt.axhline(y=(1-error_mean_bs.mean())*100,linestyle='--',color='orange',linewidth=3,label='Pois');
    plt.legend(fontsize=lsz)
    plt.ylabel('% Reduced MSE', fontsize=fsz);
    plt.xlabel('Minimum spike size', fontsize=fsz)
    plt.setp(ax.get_xticklabels(), fontsize=fsz);
    plt.setp(ax.get_yticklabels(), fontsize=fsz);
    ax.set_xticks(np.arange(error_mean_mat.shape[0]))
    ax.set_xticklabels(xticks);
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    return fig

def plotSampleRatemVAE(Y_real, y_poisson, y_gammar, y_sngr, hd, ss=0,bins=36,fsz=16):
    def get_var(y, hd):
        hd_bins = np.linspace(-180,180,bins+1);
        T = Y_real.shape[0];
        var_est_short = [];
        data_use = np.zeros((len(hd_bins)-1,2));
        for ii in range(len(hd_bins)-1):
            data_pos = ((hd[:T]>=hd_bins[ii])*(hd[:T]<=hd_bins[ii+1]));
            var_est_short.append(y[:T][data_pos,:].mean(axis=0));
            #rate_real_short.append(Y_real[:T][data_pos,:].mean(axis=0));
        var_est_short = np.asarray(var_est_short).ravel();
        #rate_real_short = np.asarray(rate_real_short).ravel();
        return var_est_short
    var_real = get_var(Y_real, hd);
    var_poi = get_var(y_poisson, hd);
    var_sng = get_var(y_gammar, hd);
    var_sngr = get_var(y_sngr, hd);
    wid=0.7;
    sz=7;
    if ss>0:
        print("sub-sample!");
        np.random.seed(888);
        use = np.random.permutation(var_real.shape[0])[:ss];
        var_real = var_real[use];
        var_poi = var_poi[use];
        var_sngr = var_sngr[use];
    
    fig = plt.figure(figsize=(12,4));
    ax = fig.add_subplot(111);
    ax.set_xlabel('Fitted firing rate',fontsize=fsz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax1 = fig.add_subplot(131);
    rg = var_real.max()
    plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
    plt.plot(var_poi, var_real, "bo",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.setp(ax1.get_xticklabels(), fontsize=fsz)
    plt.setp(ax1.get_yticklabels(), fontsize=fsz)
    #tick_locator = ticker.MaxNLocator(nbins=6,prune="both");
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1.set_title('Normalized Poisson',fontsize=fsz,fontweight='bold');
    ax1.set_ylabel("Observed firing rate",fontsize=fsz,fontweight='normal')
    
    ax2 = fig.add_subplot(132,sharey=ax1);
    #rg = y_bernoulli.var(axis=0).max()
    plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
    plt.plot(var_sng, var_real, "go",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
    

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), fontsize=fsz)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.set_title('Constrained SNG',fontsize=fsz,fontweight='bold');

    ax3 = fig.add_subplot(133,sharey=ax1);
    #rg = y_bernoulli.var(axis=0).max()
    plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
    plt.plot(var_sngr, var_real, "ro",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
    

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), fontsize=fsz)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax3.set_title('SNG',fontsize=fsz,fontweight='bold');
    plt.tight_layout();    
    return fig

def plotSampleVarmVAE(Y_real, y_poisson, y_gammar, y_sngr, hd, ss=0,bins=36,fsz=16):
    def get_var(y, hd):
        hd_bins = np.linspace(-180,180,bins+1);
        T = Y_real.shape[0];
        var_est_short = [];
        data_use = np.zeros((len(hd_bins)-1,2));
        for ii in range(len(hd_bins)-1):
            data_pos = ((hd[:T]>=hd_bins[ii])*(hd[:T]<=hd_bins[ii+1]));
            var_est_short.append(y[:T][data_pos,:].var(axis=0));
            #rate_real_short.append(Y_real[:T][data_pos,:].mean(axis=0));
        var_est_short = np.asarray(var_est_short).ravel();
        #rate_real_short = np.asarray(rate_real_short).ravel();
        return var_est_short
    var_real = get_var(Y_real, hd);
    var_poi = get_var(y_poisson, hd);
    var_sng = get_var(y_gammar, hd);
    var_sngr = get_var(y_sngr, hd);
    wid=0.7;
    sz=7;
    if ss>0:
        print("sub-sample!");
        np.random.seed(888);
        use = np.random.permutation(var_real.shape[0])[:ss];
        var_real = var_real[use];
        var_poi = var_poi[use];
        var_sngr = var_sngr[use];
    
    fig = plt.figure(figsize=(12,4));
    ax = fig.add_subplot(111);
    ax.set_xlabel('Fitted variance',fontsize=fsz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax1 = fig.add_subplot(131);
    rg = var_real.max()
    plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
    plt.plot(var_poi, var_real, "bo",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.setp(ax1.get_xticklabels(), fontsize=fsz)
    plt.setp(ax1.get_yticklabels(), fontsize=fsz)
    #tick_locator = ticker.MaxNLocator(nbins=6,prune="both");
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1.set_title('Normalized Poisson',fontsize=fsz,fontweight='bold');
    ax1.set_ylabel("Observed variance",fontsize=fsz,fontweight='normal')
    
    ax2 = fig.add_subplot(132,sharey=ax1);
    #rg = y_bernoulli.var(axis=0).max()
    plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
    plt.plot(var_sng, var_real, "go",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
    

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), fontsize=fsz)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.set_title('Constrained SNG',fontsize=fsz,fontweight='bold');

    ax3 = fig.add_subplot(133,sharey=ax1);
    #rg = y_bernoulli.var(axis=0).max()
    plt.plot([0, rg], [0, rg], ls="--", linewidth=3,c="black")
    plt.plot(var_sngr, var_real, "ro",markersize=sz,markeredgecolor='white',markeredgewidth=wid)
    

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), fontsize=fsz)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax3.set_title('SNG',fontsize=fsz,fontweight='bold');
    plt.tight_layout();    
    return fig

def plotLatVAE(X_proj_poi, X_proj_sng, X_proj_sngr, hd, fsz=16):
    fig = plt.figure(figsize=(14,4));
    ax = fig.add_subplot(111);
    #ax.set_xlabel('dim 1',fontsize=fsz,fontweight='normal');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False);

    ax1 = fig.add_subplot(131);
    ax1.scatter(X_proj_poi[:,0],X_proj_poi[:,1],c=(hd[:]+180)/360)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.setp(ax1.get_xticklabels(), fontsize=fsz)
    plt.setp(ax1.get_yticklabels(), fontsize=fsz)
    #tick_locator = ticker.MaxNLocator(nbins=6,prune="both");
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None,integer=True))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1.set_title('Normalized Poisson',fontsize=fsz,fontweight='bold');
    #ax1.set_ylabel("Observed variance",fontsize=fsz,fontweight='normal')
    
    ax2 = fig.add_subplot(132,sharey=ax1);
    ax2.scatter(X_proj_sng[:,0],X_proj_sng[:,1],c=(hd[:]+180)/360)    

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax2.get_xticklabels(), fontsize=fsz)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.set_title('Constrained SNG',fontsize=fsz,fontweight='bold');

    ax3 = fig.add_subplot(133,sharey=ax1);
    im = ax3.scatter(X_proj_sngr[:,0],X_proj_sngr[:,1],c=(hd[:]+180)/360)    

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.setp(ax3.get_xticklabels(), fontsize=fsz)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,min_n_ticks=4,prune=None))  
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax3.set_title('SNG',fontsize=fsz,fontweight='bold');
    
    axins = fig.add_axes([0.99, 0.16, 0.01, 0.78]); # left/right, up/down, width, height
    cbar = fig.colorbar(im,ticks=[((hd[:]+180)/360).min(),0.25,0.5,0.75,1], cax=axins);
    cbar.ax.tick_params(width=1,labelsize=fsz) 
    #tick_locator = ticker.MaxNLocator(nbins=6)
    #cbar.locator = tick_locator
    #cbar.update_ticks()
    #cbar.ax.set_yticks([0.25,0.5,0.75,1]);
    cbar.ax.set_yticklabels(['-178','-90','0','90','180'])
    plt.tight_layout();    
    return fig

