import numpy as np
from scipy import interpolate
import scipy.stats as st

import glob, json, os
import logging
from tqdm import tqdm

import astropy_healpix as ahp
from astropy_healpix import healpy as hp
import healpy as hp_old
from astropy import constants as c
from astropy import units as u
# import astropy.coordinates as cc

import matplotlib.pyplot as plt
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.environ['JAX_PLATFORMS'] = 'cpu'
import hasasia.sensitivity as hsen
# import hasasia.sim as hsim
import hasasia.skymap as hsky
import os; os.environ['TEMPO2'] = '/home/awc/.local/share/mamba/envs/gwenv-1/share/tempo2'
from enterprise.pulsar import Pulsar as ePulsar

# import PhenomA as pa
import utils

# constants
fm     = 3.168753575e-8   # LISA modulation frequency
YEAR   = 3.15581497632e7  # year in seconds
AU     = 1.49597870660e11 # Astronomical unit (meters)
Clight = 299792458.       # speed of light (m/s)

def plot_multi_obs_sensitivity(obs_list,xlim=None, ylim=None, colors=None,labels=None,
                               show=True,save=False,saveto=None):
    """ 
    Plot the characteristic strain sensitivity curve of multiple Observatory objects.
    
    If figure_file is provided, the figure will be saved

    Arguments
    ------------------
    obs_list (list)
        List of Observatory objects for which to plot sensitivity curves.
    xlim (tuple. optional)
        Plot x limits. Default None (matplotlib auto-lims).
    ylim (tuple. optional)
        Plot y limits. Default None (matplotlib auto-lims).
    colors (list of str)
        Color strings fo each observatory to pass to plt.plot(), should be same length as obs_list.
        Default None (current matplotlib color cycle).
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    """
    
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.tight_layout()

    ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
    ax.set_ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    for i, obs_i in enumerate(obs_list):
        if colors is not None:
            col_i = colors[i]
        else:
            col_i = None
        if labels is not None:
            lab_i = labels[i]
        else:
            if obs_i.name == 'astrometry':
                lab_i = obs_i.survey
            else:
                lab_i = obs_i.name
        
        obs_i.add_sensitivity_curve(ax,color=col_i,label=lab_i)

    plt.legend()
    
    ## save
    if save:
        utils.savefig('multi_senitivity_curve',saveto=saveto)
    
    if show:
        plt.show()

    return ax


   
class Observatory():
    '''
    Generic base observatory class
    '''

    def __init__(self,name,Tobs,sens_curve=None,interpolate=True):

        self.name = name
        self.Tobs = Tobs
        self.interpolate = interpolate

        if self.interpolate:
            if sens_curve is None:
                raise ValueError("Must provide sensitivity curve if interp_Sn is True.")
            self.base_fs, self.base_Sn = self.setup_interpolator(sens_curve)
            self.fmin = self.base_fs[0]
            self.fmax = self.base_fs[-1]
            self._Sn = self._interp_Sn


    def setup_interpolator(self,sens_curve,**kwargs):
        '''
        Sets up a basic interpolator for cases where we don't have a functional form of Sn(f)

        Arguments
        ---------------
        sens_curve (array or str)
            The observatory sensitivity curve. Can be provided directly as an array, or as a filepath to load a csv.
            If array, must be Nf x 2 array of fs, Sn(fs).
        '''
        if type(sens_curve) is str:
            sens_curve = np.loadtxt(sens_curve)
        
        ## enforce shape
        sens_curve = sens_curve.squeeze()
        if sens_curve.ndim != 2:
            raise("If providing the sensitivity curve as an array, it must be a Nf x 2 array of fs, Sn(fs).")
        if sens_curve.shape[0] != 2:
            sens_curve = sens_curve.T
        fs = sens_curve[0,:]
        Sn = sens_curve[1,:]

        ## initiate the interpolator
        self._interp_Sn = interpolate.make_splrep(fs, Sn, s=0)

        return fs, Sn

    def Sn(self,f):
        '''
        Wrapper for the sensitivity curve; overwrite if providing calculations for this Observatory.
        '''
        filt = (f > self.fmin)*(f < self.fmax)
        Sn = np.atleast_1d(self._Sn(f)*filt)
        Sn[Sn==0] = np.inf
        return Sn
    
    def PlotSensitivityCurve(self,fs=None,Sn=None,xlim=None, ylim=None, figure_file=None):
        """ 
        Plot the characteristic strain sensitivity curve 
        
        If figure_file is provided, the figure will be saved
        """
        
        fig, ax = plt.subplots(1, figsize=(8,6))
        plt.tight_layout()
    
        ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
        ax.set_ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=20)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        
        if Sn is None:
            if fs is None:
                fs = self.base_fs
            Sn = self.Sn(fs)

        ax.loglog(fs, np.sqrt(fs*Sn)) # plot the characteristic strain
        
        plt.show()
        
        if (figure_file != None):
            plt.savefig(figure_file)
            
        return

    def add_sensitivity_curve(self,ax,fs=None,**kwargs):
        '''
        Adds the observatory's sensitivity curve to an existing plot

        Arguments
        --------------
        ax (matplotlib.axes)
            Matplotlib axes on which to plot the sensitivity curve.
        fs (array, optional)
            Frequencies, if desired. Default None (uses the observatory's default frequencies)
        **kwargs (kwargs, optional)
            Keyword arguments to pass to plt.loglog()
        '''
        if fs is None:
            fs = self.base_fs
        Sn = self.Sn(fs)

        ax.loglog(fs, np.sqrt(fs*Sn), **kwargs) # plot the characteristic strain

        return

class Astrometry(Observatory):
    '''
    Astrometry class
    ------------------------
    Handles the basic astrometric sensitivity model

    '''

    def __init__(self,survey='Kepler',**kwargs):
        '''
        Default values are for the Kepler field. Assumes regularly-sampled cadence.

        Roman values are from https://arxiv.org/abs/1712.05420 (+papers by Scott Gaudi's group, ask Kris Pard)
        
        Arguments
        ---------------------
        survey (str):
            Either a known survey (supported: Kepler, Roman), or a new survey name with the below provided as keyword arguments:
        
        
        sigma_ss (astropy Quantity, optional):
            Single-star astrometric precision, in angular units (default 0.7 mas).
        T_survey (astropy Quantity, optional):
            Survey duration (default 3.5 yr).
        T_cadence (astropy Quantity, optional):
            Survey cadence (default 30 min).
        N_stars (int, optional):
            Number of stars observed (default 160,000).
        '''

        self.name = 'astrometry'
        self.survey = survey
        self.interpolate = False
        
        if self.survey == 'Kepler':
            self.sigma_ss = 0.7*u.mas
            self.T_survey = 3.5*u.yr
            self.T_cadence = 30*u.min
            self.N_stars = 160000
        elif self.survey == 'Roman':
            self.sigma_ss = 1.1*u.mas
            self.T_survey = 5*u.yr
            self.T_cadence = 12*u.min
            self.N_stars = 1e8
        else:
            ## these need to be provided as kwargs
            self.sigma_ss = sigma_ss
            self.T_survey = T_survey
            self.T_cadence = T_cadence
            self.N_stars = N_stars

        self.Tobs = self.T_survey
        
        self.h_sens = self.get_h_sens(self.sigma_ss,self.T_survey,self.T_cadence,self.N_stars)
        self.psd = 0.5*self.get_psd_simple(self.sigma_ss,self.T_survey,self.T_cadence,self.N_stars).value
        
        ## minimum frequency determined by survey duration
        self.fmin = (1/self.T_survey).to(u.Hz).value
        ## maximum is Nyquist
        self.fmax = (1/(2*self.T_cadence)).to(u.Hz).value

        ## set default frequency range
        self.base_fs = np.linspace(self.fmin,self.fmax,500)

        
        
    def Sn(self,f):
        '''
        Sky-averaged one-sided astrometric sensitivity curve.

        Arguments
        ------------------
        f (float or array):
            Frequencies at which to compute the sensitivity curve.

        Returns
        ------------------
        Sn (float or array):
            The sky-averaged one-sided astrometric sensitivity curve at f.
        '''
        ## astrometric Sn is flat, so:
        ## mask the sensitivity "curve" to the allowed frequencies 
        ## and set the sensitivity to +np.inf outside of that range
        filt = (f > self.fmin)*(f < self.fmax)
        Sn = np.atleast_1d(self.psd*filt)
        Sn[Sn==0] = np.inf

        return Sn
        
    
    ## simple h sens. for astrometry
    def get_h_sens(self,sig_ss, Tsurvey, tcad, Nstars):
        '''
        all inputs should be astropy quantities with attached units, except Nstars.
        '''
        sig_ss = sig_ss.to(u.rad)
        Tsurvey = Tsurvey.to(u.s)
        tcad = tcad.to(u.s)
        Nmeas = (Tsurvey/tcad).to('')
        return (2. * sig_ss/np.sqrt(Nmeas * Nstars)).value

    ## simple psd for astrometry
    def get_psd_simple(self,sig_ss, Tsurvey, tcad, Nstars):
        '''
        2-sided PSD
        all inputs should be astropy quantities with attached units, except Nstars.
        '''
        h = self.get_h_sens(sig_ss, Tsurvey, tcad, Nstars)
        return (h**2 * tcad).to(1./u.Hz)

class LISA(Observatory):
    """ 
    LISA class
    -----------------------
    Handles LISA's orbit and detector noise quantities
    
    Methods:
        LoadTranfer - read in, and store, transfer function data file
        SC_Orbit    - return calculate spacecraft (S/C) positions
        SC_Seps     - return unit-separation vectors between LISA S/C
        Pn          - return LISA's strain power spectral density
        Pn_WC       - return LISA's strain power spectral density with confusion noise estimate
        SnC         - return confusion noise estimate
        Sn          - return LISA's sensitivity curve
    """
    
    def __init__(self, Tobs=4*YEAR, Larm=2.5e9, NC=2, transfer_file='R.txt'):
        """
        Tobs - LISA observation period (4 years is nominal mission lifetime)
        Larm = 2.5e9 LISA's arm length, current design arm length, 
                            constant to 1st order in eccentricity
        NC - Number of data channels
        """

        super().__init__("LISA",Tobs,interpolate=False)
        
        self.Larm = Larm 
        self.NC   = NC 
        
        self.ecc   = self.Larm/(2*np.sqrt(3.)*AU)  # to maintain quasi-equilateral triangle configuration
        self.fstar = Clight/(2*np.pi*self.Larm) # transfer frequency, design value ~ 19.1 mHz
        
        self.LoadTransfer(transfer_file) # load the transfer function


    ##########################################################
    ################# Noise Curve Methods ####################
    ##########################################################
    
    def LoadTransfer(self, file_name):
        """ 
        Load the data file containing the numerically calculate transfer function
        (sky and polarization averaged)
        """
        
        try:    # try to read in the data file
            transfer_data = np.genfromtxt(file_name) # read in the data
            
        except: # If file isn't successfully read in, use approximate transfer function
            print("Warning: Could not find transfer function file!")
            print("         \tApproximation will be used...")
            self.FLAG_R_APPROX = True
            return
            
        f = transfer_data[:,0]*self.fstar        # convert to frequency
        R = transfer_data[:,1]*self.NC           # response gets improved by more data channels
    
        # create an interpolation function; attach to LISA object
        self.R_INTERP = interpolate.splrep(f, R, s=0)
        self.FLAG_R_APPROX = False
        self.base_fs = f

        ## setting these manually
        self.fmin = self.base_fs.min()
        self.fmax = self.base_fs.max()
        
        return
        
    def Pn(self, f):
        """
        Caclulate the Strain Power Spectral Density
        """
        
        # single-link optical metrology noise (Hz^{-1}), Equation (10)
        P_oms = (1.5e-11)**2*(1. + (2.0e-3/f)**4) 
        
        # single test mass acceleration noise, Equation (11)
        P_acc = (3.0e-15)**2*(1. + (0.4e-3/f)**2)*(1. + (f/(8.0e-3))**4) 
        
        # total noise in Michelson-style LISA data channel, Equation (12)
        Pn = (P_oms + 2.*(1. + np.cos(f/self.fstar)**2)*P_acc/(2.*np.pi*f)**4)/self.Larm**2
        
        return Pn
        
    def SnC(self, f):
        """
        Get an estimation of the galactic binary confusion noise are available for
            Tobs = {0.5 yr, 1 yr, 2 yr, 4yr}
        Enter Tobs as a year or fraction of a year
        """
        Tobs = self.Tobs 
        NC   = self.NC
    
        # Fix the parameters of the confusion noise fit
        if (Tobs < .75*YEAR):
            est = 1
        elif (0.75*YEAR < Tobs and Tobs < 1.5*YEAR):
            est = 2
        elif (1.5*YEAR < Tobs and Tobs < 3.0*YEAR):   
            est = 3
        else:
            est = 4
                
        if (est==1):
            alpha  = 0.133
            beta   = 243.
            kappa  = 482.
            gamma  = 917.
            f_knee = 2.58e-3  
        elif (est==2):
            alpha  = 0.171
            beta   = 292.
            kappa  = 1020.
            gamma  = 1680.
            f_knee = 2.15e-3 
        elif (est==3):
            alpha  = 0.165
            beta   = 299.
            kappa  = 611.
            gamma  = 1340.
            f_knee = 1.73e-3  
        else:
            alpha  = 0.138
            beta   = -221.
            kappa  = 521.
            gamma  = 1680.
            f_knee = 1.13e-3 
        
        A = 1.8e-44/NC
        
        Sc  = 1. + np.tanh(gamma*(f_knee - f))
        Sc *= np.exp(-f**alpha + beta*f*np.sin(kappa*f))
        Sc *= A*f**(-7./3.)
        
        return Sc
        
    def Sn(self, f):
        """ Calculate the sensitivity curve """
    
        if (self.FLAG_R_APPROX == False): # if sensitivity curve file is provided use it
            R = interpolate.splev(f, self.R_INTERP, der=0)
        else:
            R = 3./20./(1. + 6./10.*(f/self.fstar)**2)*self.NC
            
        Sn = self.Pn(f)/R + self.SnC(f)

        filt = (f > self.fmin)*(f < self.fmax)
        Sn = np.atleast_1d(Sn*filt)
        Sn[Sn==0] = np.inf
    
        return Sn
    	
    def Pn_WC(self, f):
        """ Calculate Power Spectral Density with confusion (WC) noise estimate """
    
        if (self.FLAG_R_APPROX == False):
            R = interpolate.splev(f, self.R_INTERP, der=0)
        else:
            R = 3./20./(1. + 6./10.*(f/self.fstar)**2)*self.NC
            
        PnC = self.Pn(f) + self.SnC(f)*R
    
        return PnC

    ##########################################################
    ################# LISA's Orbit Methods ###################
    ##########################################################
     
    def SC_Orbits(self, t):
        """ Calculate the analytic (leading order in eccentricity) LISA orbits """
    
        N = len(t)
        kappa  = 0.0 # initial phase of LISA orbits
        Lambda = 0.0 # initial phase of spacecraft in their quasi-triangle configuration
    
        alpha = (2.*np.pi*fm*t + kappa).reshape((1,N)) 
        sa = np.sin(alpha) 
        ca = np.cos(alpha)
    
        beta = (np.array([0.0, 2.*np.pi/3., 4.*np.pi/3.]) + Lambda).reshape((3,1))
        sb = np.sin(beta)
        cb = np.cos(beta) # (S/C, len(t))
        
        x = np.zeros((3, 3, N)) # dim, S/C, time
    
        x[0] = AU*ca + AU*self.ecc*(sa*ca*sb - (1. + sa*sa)*cb)
        x[1] = AU*sa + AU*self.ecc*(sa*ca*cb - (1. + ca*ca)*sb)
        x[2] = -np.sqrt(3.)*AU*self.ecc*(ca*cb + sa*sb)
    
        return x
        
    def SC_Seps(self, t, x):
        """ Calculate S/C unit-separation vectors """
    
        N = len(t)
    
        rij = np.zeros((3,3,3,N))
    
        rij[:,0,1,:] = x[:,1,:] - x[:,0,:]
        rij[:,1,0,:] = -rij[:,0,1,:]
    
        rij[:,0,2,:] = x[:,2,:] - x[:,0,:]
        rij[:,2,0,:] = -rij[:,0,2,:]
    
        rij[:,1,2,:] = x[:,2,:] - x[:,1,:]
        rij[:,2,1,:] = -rij[:,1,2,:]
    
        return rij/self.Larm
    



    
class EchoArray():
    '''
    Class to wrap all the PTA echo calculations.
    '''

    def __init__(self,datadir='/datadisk/data/NANOGrav/ipta_dr3_like/',
                 pardir=None,timdir=None,noise_dir=None,seed=170817,
                 pulsar_distances='default',pdist_extent=5*u.kpc,nside=32):
        '''

        Arguments
        ----------------
        datadir (str, optional):
            Data directory. All files (par, tim, and noise) will be asumed to be in this directory
            unless specified via pardir, timdir, or noise_dir, respectively.
        pardir, timedir, noisedir (str, optional):
            Default None (all files in datadir). If specified, will point the code to these directories for
            .par, .tim, and noise files, respectively.
        seed (int, optional):
            Numpy rng seed. Default 170817.
        pulsar_distances (str or array, optional):
            Pulsar distances. Several options:
            - If 'default', will be drawn from a volumetrically uniform distribution in the Galactic plane (p(r)~r^2).
            - If 'data', will follow the distances of the pulsars in datadir. 
                WARNING: if your pulsars do not have distances, this will result in all pulsar distances being set to 1kpc by Enterprise.
            - If pulsar_distances is an array, it must be of length(pulsars) and give the distance for each in kpc.
        pdist_extent (float or astropy.Quantity)
            Maximum allowed pulsar distance; only needed if pulsar_distances is 'default'. Default is 5 kpc.
        '''

        ## get rng from seed
        self.rng = np.random.default_rng(seed)

        ## set nside, initialize astropy.HEALPix
        self.nside = nside
        self.HEALPix = ahp.HEALPix(self.nside,frame='barycentricmeanecliptic')

        ## set data directories
        if pardir is not None:
            self.pardir = pardir
        else:
            self.pardir = datadir
        if timdir is not None:
            self.timdir = timdir
        else:
            self.timdir = datadir
        if noise_dir is not None:
            self.noise_dir = noise_dir
        else:
            self.noise_dir = datadir

        ## need to run setup_pta here and get Npulsars
        ## TODO -- mute warnings about distances here
        self.psrs, self.psr_specs = self.setup_pta(self.pardir,self.timdir,self.noise_dir,pdists=pulsar_distances)
        self.Npulsars = len(self.psrs)

        ## compute/assign pulsar distances
        if type(pulsar_distances) is str:
            if pulsar_distances == 'default':
                self.pdist_extent = pdist_extent
                self.pulsar_distances = self.draw_pulsar_dists(self.Npulsars,self.pdist_extent,self.rng)
            elif pulsar_distances == 'data':
                print("Continuing with original pulsar distances...")
                self.pulsar_distances = None
            else:
                raise ValueError("Unknown setting for pulsar_distances. Must be 'default', 'data', or a numpy array.")
        elif type(pulsar_distances) is np.ndarray:
            assert len(pulsar_distances) == self.Npulsars
            self.pulsar_distances = pulsar_distances
        else:
            raise TypeError("pulsar_distances must be 'default', 'data', or a numpy array.")

        ## attach the distances to the spectra objects
        self.assign_pulsar_distances()



        
    def setup_pta(self,pardir,timdir,noise_dir,pdists=None):
        '''
        Function to wrap all the setup from pulsar files -> the PTA.
        '''


        ## load datafiles
        pars = sorted(glob.glob(pardir+'*.par'))
        tims = sorted(glob.glob(timdir+'*.tim'))
        noise_files = [noise_dir+'all_pulsar_IRN.json'] #sorted(glob.glob(noise_dir+'*.json'))

        ## hasasia data preliminaries
        rnoise = {}

        for nf in noise_files:
            with open(nf,'r') as fin:
                rnoise.update(json.load(fin))
        
        ## load pulsar instances
        print('Loading Enterprise pulsars...')
        ePsrs = []
        ## catch the warnings about distances, as we handle those later
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)
        for par,tim in zip(pars,tims):
            ePsr = ePulsar(par, tim,  ephem='DE436')
            ePsrs.append(ePsr)
            print('\rPSR {0} complete '.format(ePsr.name),end=' ',flush=True)
        ## switch logging level back to normal so we get our own status updates
        logger.setLevel(logging.INFO)
        ## get timespan
        Tspan = hsen.get_Tspan(ePsrs)
        
        print("\nTimespan: {:0.1f} yrs".format(Tspan/(3600*24*365))) ## in yrs
        
        ## frequencies
        fyr = 1/(365.25*24*3600)
        freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(1e-6),800)
        
        ## hasasia Spectrum instances
        psrs = []
        thin = 10 
        os.environ['JAX_PLATFORMS'] = 'cpu'
        print("\nComputing correlations...")
        for ePsr in ePsrs:
            corr = self.make_corr(ePsr)[::thin,::thin]
            plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)
            if ePsr.name in rnoise.keys():
                logAmp, gam = rnoise[ePsr.name]['log10_A'], rnoise[ePsr.name]['gamma']
                plaw += hsen.red_noise_powerlaw(A=10**logAmp, gamma=gam, freqs=freqs)
                
            corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                       toas=ePsr.toas[::thin])
            psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                              toaerrs=ePsr.toaerrs[::thin],
                              phi=ePsr.phi,theta=ePsr.theta, 
                              N=corr, designmatrix=ePsr.Mmat[::thin,:])
            psr.name = ePsr.name
            psrs.append(psr)
            # del ePsr
            print('\rPSR {0} complete '.format(psr.name),end=' ',flush=True)

        ## make the spectra objects
        print("\nCreating hasasia spectra...")
        specs = []
        for p in psrs:
            sp = hsen.Spectrum(p, freqs=freqs)
            _ = sp.NcalInv
            specs.append(sp)
            print('\rPSR {0} complete '.format(p.name),end=' ',flush=True)

        return psrs, specs

    ## adapting this from hasasia. With efac=1.0, and equad/ecorr set to 0, this correlation matrix should just be diagonal in efac**2
    ## because this dataset has no detailed treatment of individual observatories, we ignore all the backend-specificity of the original code
    def make_corr(self,psr):
        N = psr.toaerrs.size
        corr = np.zeros((N,N))
    
        sigma_sqr = np.zeros(N)
    
        efac = 1.0
        sigma_sqr = efac**2 * psr.toaerrs**2
        
        corr = np.diag(sigma_sqr)
        return corr
    
    def draw_pulsar_dists(self,N,extent,rng):
        '''
        Draw pulsar distances from a volumetrically uniform distribution in the Galactic plane (p(r) ~ r^2).

        Arguments
        --------------
        N (int):
            Number of pulsars.
        extent (float or astropy.Quantity):
            Maximum allowed pulsar distance in kpc.
        rng (numpy.random.default_rng):
            rng

        Returns
        ------------
        pdists (array)
            Pulsar distances in kpc.
        '''
        extent = utils.assert_units(extent,u.kpc).value
        pdists = st.beta.rvs(3,1,scale=0.99*extent,size=N,random_state=rng)*u.kpc ## this is a p(r) ~ r**2 dist
        return pdists

    def assign_pulsar_distances(self):
        '''
        Wrapper function to attach distances to the hasasia spectrum objects.
        '''
        if self.pulsar_distances is not None:
            for pdist, sp in zip(self.pulsar_distances,self.psr_specs):
                sp.pdist = pdist
        return
    
    def compute_echo(self,binary):
        '''
        Compute the per-pulsar SNR, GW frequencies, and GW amplitudes for a given binary.

        Arguments
        ------------
        binary (WaveformTools.Binary object):
            The binary object, carrying all intrinsic information about the binary.

        Returns
        ------------
        psr_snrs_squared (list of float):
            The per-pulsar squared SNRs.
        psr_gw_freqs (list of float):
            The per-pulsar GW echo frequencies.
        psr_gw_amps (list of float):
            The per-pulsar GW echo amplitudes.
        
        '''
        ## hasasia skysens takes (theta_gw,phi_gw), following the Healpy pix2ang convention
        ## but we have a SkyCoord that doesn't use the silly Healpy convention
        ## so convert from longitude and latitude into colatitude and longitude
        theta_gw = np.atleast_1d(np.abs(binary.sky_loc.barycentricmeanecliptic.lat.rad - np.pi/2))
        phi_gw = np.atleast_1d(binary.sky_loc.barycentricmeanecliptic.lon.rad)
        
        psr_dists = [psr_spec.pdist for psr_spec in self.psr_specs]
    
        ## deal with units
        psr_dists = [utils.assert_units(psr_dist,u.kpc) for psr_dist in psr_dists]
        
        psr_snrs_squared = []
        psr_gw_freqs = []
        psr_gw_amps = []
        for psr_spec, psr_dist in zip(self.psr_specs, psr_dists):
            ## make the directional responses
            sky_spec = hsky.SkySensitivity([psr_spec],theta_gw, phi_gw, pulsar_term='only')
            ## get Tspan
            T = sky_spec.Tspan
    
            ## Use pulsar attributes to determine GW frequency in pulsar
            time_delay = (1+np.einsum('ij,il->jl',sky_spec.pos,sky_spec.K)) * psr_dist/c.c
            pulsar_freq = binary.f_of_t(time_delay)
            psr_gw_freqs.append(pulsar_freq)
            ## get frequency array index of GW frequency in this pulsar
            pf_idx = np.argmin(np.abs(psr_spec.freqs-pulsar_freq.value))
    
            ## deal with high frequencies
            last_bin_width = psr_spec.freqs[-1] - psr_spec.freqs[-2]
            if (pf_idx == len(psr_spec.freqs)-1) and (pulsar_freq.value > psr_spec.freqs[-1]+last_bin_width):
                S_eff = np.inf
            else:
                ## effective noise curve
                S_eff = sky_spec.S_eff[pf_idx]
    
            ## get monochromatic strain at t = -time_delay
            psr_gw_amp = binary.h_echo_of_t(time_delay)
            psr_gw_amps.append(psr_gw_amp)
    
            ## compute snr squared
            snr_squared = psr_gw_amp**2  * (T/S_eff)
            
            psr_snrs_squared.append(snr_squared)
    
        return psr_snrs_squared, psr_gw_freqs, psr_gw_amps

    def calc_echo_snr(self,binary):
        '''
        Get just the summed array SNR for a given binary.

        Arguments
        ------------
        binary (WaveformTools.Binary object):
            The binary object, carrying all intrinsic information about the binary.

        Returns
        ------------
        snr_tot (float):
            The total SNR, summed across the echo array.
        '''

        psr_snrs_squared, _, _ = self.compute_echo(binary)

        snr_tot = np.sum(np.array(psr_snrs_squared))

        return snr_tot

        
    
    def calc_allsky_snr(self,binary,plot=False):
        '''
        Calculate the per-pulsar echo array SNR across the sky.

        Arguments
        ------------
        binary (WaveformTools.Binary object):
            The binary object, carrying all intrinsic information about the binary.
        plot (bool, optional):
            Whether to display a plot of the SNR skymap.

        Returns
        ------------
        rho2_by_loc (list of arrays):
            The per-pulsar squared SNR for each sky location.
        fs_by_loc (list of arrays):
            The per-pulsar echo frequency for each sky location.
        amps_by_loc (list of arrays):
            The per-pulsar echo amplitude for each sky location.
        
        '''
        ## save original binary location
        sky_loc_temp = binary.sky_loc

        ## initialize per-location lists
        rho2_by_loc = []
        fs_by_loc = []
        amps_by_loc = []
        
        for idx in tqdm(range(hp.nside2npix(self.nside))):
            sky_loc = self.HEALPix.healpix_to_skycoord(idx)
            binary.sky_loc = sky_loc
            rho2_i, fs_i, amps_i = self.compute_echo(binary)
            rho2_by_loc.append(rho2_i)
            fs_by_loc.append(fs_i)
            amps_by_loc.append(amps_i)

        ## reset binary sky location
        binary.sky_loc = sky_loc_temp
        
        if plot:
            ## call the skymap plotter
            pass

        return rho2_by_loc, fs_by_loc, amps_by_loc

    def get_summed_rho2(self,rho2_by_loc):
        '''
        Wrapper function to unpack rho2_by_loc into pulsar-summed SNR squared by sky location.
        '''
        return np.array([np.sum(np.array(rho2_i)) for rho2_i in rho2_by_loc])

    def plot_snr_skymap(self,rho2_by_loc,summed=False,plot_psrs=True,
                        save=False,saveto=None,show=True):
        '''
        Plot the array-summed SNR as a function of binary location.
        
        Arguments
        rho2_by_loc (list of lists or array):
            The per-pulsar squared SNR for each sky location.
            (or, if summed==True, the array-summd SNR for each sky location.)
        summed (bool, optional):
            Whether the provided rho2_by_loc has already been summed over the array. Default False.
        plot_psrs (bool, optional):
            Whether to also plot the array pulsar positions. Default True.
        show : bool, optional
            Whether to show the plot at runtime. The default is True.
        save : bool, optional
            Whether to save the created figures to disk. The default is False.
        saveto : str, optional
            If save, the desired output directory. The default is None (saves in current directory).

        '''

        ## sum across the array if this has not already been done
        if not summed:
            rho2_by_loc = self.get_summed_rho2(rho2_by_loc)
        
        hp_old.mollview(np.sqrt(rho2_by_loc),#max=3,
                        unit="SNR",title="Echo Array SNR by GW Origin")
        if plot_psrs:
            hp_old.projscatter(theta=[sp.theta for sp in self.psr_specs],phi=[sp.phi for sp in self.psr_specs],
                               s=3,marker='*',c='cyan')
        
        ## save
        if save:
            utils.savefig('Echo_SNR_Skymap',saveto=saveto)
        
        if show:
            plt.show()

        return plt.gca()














        
    
    