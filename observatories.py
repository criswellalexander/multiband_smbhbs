import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import astropy.units as u

import PhenomA as pa

# constants
fm     = 3.168753575e-8   # LISA modulation frequency
YEAR   = 3.15581497632e7  # year in seconds
AU     = 1.49597870660e11 # Astronomical unit (meters)
Clight = 299792458.       # speed of light (m/s)

   
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
        return self._Sn(f)
    
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
        Sn = self.psd*filt
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
    



    
    





    
    
    
    
    