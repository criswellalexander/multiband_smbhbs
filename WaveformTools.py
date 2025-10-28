import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import interpolate
import astropy.units as u
import astropy.constants as c

import PhenomA as pa
import LISA as li
import utils

""" Constants """
C       = 299792458.         # m/s
YEAR    = 3.15581497632e7    # sec
TSUN    = 4.92549232189886339689643862e-6 # mass of sun in seconds (G=C=1)
MPC     = 3.08568025e22/C    # mega-Parsec in seconds

TOBS_MAX = 4*YEAR # Maximum observation period (LISA's nominal mission lifetime)

""" Cosmological values """
H0      = 69.6 * u.km/u.s/u.Mpc     # Hubble parameter today
Omega_m = 0.286     # density parameter of matter


def get_Dl(z, Omega_m, H0,unit_out=u.Mpc):
    """ calculate luminosity distance"""
    # see http://arxiv.org/pdf/1111.6396v1.pdf
    x0 = (1. - Omega_m)/Omega_m
    xZ = x0/(1. + z)**3

    Phi0  = (1. + 1.320*x0 + 0.4415*x0**2  + 0.02656*x0**3)
    Phi0 /= (1. + 1.392*x0 + 0.5121*x0**2  + 0.03944*x0**3)
    PhiZ  = (1. + 1.320*xZ + 0.4415*xZ**2  + 0.02656*xZ**3)
    PhiZ /= (1. + 1.392*xZ + 0.5121*xZ**2  + 0.03944*xZ**3)
    
    return (2.*(c.c/H0)*(1. + z)/np.sqrt(Omega_m)*(Phi0 - PhiZ/np.sqrt(1. + z))).to(unit_out)


def get_z(z, Dl, Omega_m, H0):
    """ calculate redishift uisng root finder """
    
    return (get_Dl(z, Omega_m, H0) - Dl).to(u.Mpc).value

def get_omega_isco(Mtot):
    Mtot = utils.assert_units(Mtot,u.Msun)
    return 6**(-3/2) / (Mtot*c.G/c.c**3).decompose()

def get_fgw_isco(Mtot):
    omega_isco = get_omega_isco(Mtot).to(u.Hz)
    return omega_isco/np.pi

## definitions for f(t) and t(f)
def dt_by_f(f_gw,mc,units_out=u.yr):
    '''
    Returns delta_t = t_c - t (time to coalescence t_c)
    
    Assumes f_gw in Hz, mc in Msun.
    
    units_out (astropy.unit) must be a time unit; result will be returned in these units (default yrs)
    '''
    ## unit checking
    if not hasattr(f_gw,'unit'):
        f_gw = f_gw*u.Hz
    elif f_gw.unit is not u.Hz:
        f_gw = f_gw.to(u.Hz)
    
    if not hasattr(mc,'unit'):
        mc = mc*u.M_sun
        mc = mc.to(u.kg)
    elif mc.unit is not u.kg:
        mc = mc.to(u.kg)
    
    return ((5/(8*np.pi)**(8/3)) * (c.G*mc/c.c**3)**(-5/3) * f_gw**(-8/3)).decompose().to(units_out)

def f_by_dt(dt,mc,units_in=u.yr):
    '''
    Inverse of above. dt = t_c - t (i.e., time to coalescence)
    '''
    if not hasattr(dt,'unit'):
        dt = dt*units_in.to(u.s)
    elif dt.unit is not u.s:
        dt = dt.to(u.s)
    
    if not hasattr(mc,'unit'):
        mc = mc*u.M_sun
        mc = mc.to(u.kg)
    elif mc.unit is not u.kg:
        mc = mc.to(u.kg)
    
    return ((5**(3/8) / (8*np.pi)) * (c.G*mc/c.c**3)**(-5/8) * dt**(-3/8)).decompose().to(u.Hz)

## strain amplitude for a monochromatic source h_echo
def calc_h_echo(mc,f_gw,d_L):

    ## deal with units
    if not hasattr(mc,'unit'):
        mc = mc*u.M_sun
        mc = mc.to(u.kg)
    elif mc.unit is not u.kg:
        mc = mc.to(u.kg)

    if not hasattr(f_gw,'unit'):
        f_gw = f_gw*u.Hz
    elif f_gw.unit is not u.Hz:
        f_gw = f_gw.to(u.Hz)

    if not hasattr(d_L,'unit'):
        d_L = d_L*u.Mpc
        d_L = d_L.to(u.m)
    elif f_gw.unit is not u.m:
        d_L = d_L.to(u.m)
    
    h0 = (4*c.c/d_L) * (c.G*mc/c.c**3)**(5/3) * (np.pi*f_gw)**(2/3)

    return h0.decompose()
    
def calc_k(theta, phi):
    """ Calculate the unit-direction vector pointing towards the source """
    
    sth = np.sin(theta)

    k = -np.array([sth*np.cos(phi), sth*np.sin(phi), np.cos(theta)])

    return k
    
def calc_k_dot_r(k, rij):
    """ Dot product between unit-direction vector and the S/C unit-separation vectors """

    k_dot_r = k[0]*rij[0,:,:,:] + k[1]*rij[1,:,:,:] + k[2]*rij[2,:,:,:]

    return k_dot_r
    
def get_XX_TDI(OBJ, lisa, f, Aeff, theta, phi, iota):
    """ Construct cos(\\iota) and \\psi averaged Michelson-equivalent TDI response """
     
    N = len(f)       
            
    # stationary time of SPA
    tStar = pa.dPsieff_df(f, OBJ.geom_M, OBJ.geom_eta, 0.0)/(2*np.pi) 
    
    
    # Direction vectors
    k   = calc_k(theta, phi)
    x   = lisa.SC_Orbits(tStar)
    rij = lisa.SC_Seps(tStar, x)
    
    rij_OUTER_rij = rij.reshape((3,1, 3,3, N))*rij.reshape((1,3, 3,3, N))
    
    # GW basis tesnrors
    u = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    v = np.array([np.sin(phi), -np.cos(phi), 0.0])
    ep = np.outer(u,u) - np.outer(v,v)
    ec = np.outer(u,v) + np.outer(v,u)
    
    # construct the detector tensor for LISA
    dp12 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,0,1,:], ep)
    dc12 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,0,1,:], ec)
    dp21 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,1,0,:], ep)
    dc21 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,1,0,:], ec)
    dp13 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,0,2,:], ep)
    dc13 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,0,2,:], ec)
    dp31 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,2,0,:], ep)
    dc31 = np.einsum('nmk,nm->k', rij_OUTER_rij[:,:,2,0,:], ec)
    
    # Piece together the transfer function
    kDOTrij = calc_k_dot_r(k, rij)
    
    kDOTr12 = kDOTrij[0,1,:] 
    kDOTr21 = kDOTrij[1,0,:]
    kDOTr13 = kDOTrij[0,2,:]
    kDOTr31 = kDOTrij[2,0,:]
    
    TransArg12 = f/(2*lisa.fstar)*(1. - kDOTr12)
    TransArg21 = f/(2*lisa.fstar)*(1. - kDOTr21)
    TransArg13 = f/(2*lisa.fstar)*(1. - kDOTr13)
    TransArg31 = f/(2*lisa.fstar)*(1. - kDOTr31)
    
    Trans12 = 0.5*np.sinc(TransArg12/np.pi)*np.exp(1j*TransArg12)*np.exp(1j*2*np.pi*f*np.dot(k,x[:,0,:])/C)
    Trans21 = 0.5*np.sinc(TransArg21/np.pi)*np.exp(1j*TransArg21)*np.exp(1j*2*np.pi*f*np.dot(k,x[:,1,:])/C)
    Trans13 = 0.5*np.sinc(TransArg13/np.pi)*np.exp(1j*TransArg13)*np.exp(1j*2*np.pi*f*np.dot(k,x[:,0,:])/C)
    Trans31 = 0.5*np.sinc(TransArg31/np.pi)*np.exp(1j*TransArg31)*np.exp(1j*2*np.pi*f*np.dot(k,x[:,2,:])/C)

    if (iota == None):
        # iota = pi/2, psi = 0
        y12_a = Trans12*Aeff*0.5*dp12/2
        y21_a = Trans21*Aeff*0.5*dp21/2
        y13_a = Trans13*Aeff*0.5*dp13/2
        y31_a = Trans31*Aeff*0.5*dp31/2
        
    else:
        y12_a = 0.5*Trans12*Aeff*(0.5*(1 + np.cos(iota)**2)*dp12 + 1j*np.cos(iota)*dc12)
        y21_a = 0.5*Trans21*Aeff*(0.5*(1 + np.cos(iota)**2)*dp21 + 1j*np.cos(iota)*dc21)
        y13_a = 0.5*Trans13*Aeff*(0.5*(1 + np.cos(iota)**2)*dp13 + 1j*np.cos(iota)*dc13)
        y31_a = 0.5*Trans31*Aeff*(0.5*(1 + np.cos(iota)**2)*dp31 + 1j*np.cos(iota)*dc31)
        
    X_TDI = (y12_a - y13_a)*np.exp(-1j*f/lisa.fstar) + (y12_a - y13_a)
    
    if (iota == None):
        XX_TDI = 8./5*np.abs(X_TDI)**2
    else:
        XX_TDI = 1./2*np.abs(X_TDI)**2


    if (iota == None):
        # iota = pi/2, psi = pi/4
        y12_a = Trans12*Aeff*0.5*dc12/2
        y21_a = Trans21*Aeff*0.5*dc21/2
        y13_a = Trans13*Aeff*0.5*dc13/2
        y31_a = Trans31*Aeff*0.5*dc31/2
    else:
        y12_a = 0.5*Trans12*Aeff*(-0.5*(1 + np.cos(iota)**2)*dc12 + 1j*np.cos(iota)*dp12)
        y21_a = 0.5*Trans21*Aeff*(-0.5*(1 + np.cos(iota)**2)*dc21 + 1j*np.cos(iota)*dp21)
        y13_a = 0.5*Trans13*Aeff*(-0.5*(1 + np.cos(iota)**2)*dc13 + 1j*np.cos(iota)*dp13)
        y31_a = 0.5*Trans31*Aeff*(-0.5*(1 + np.cos(iota)**2)*dc31 + 1j*np.cos(iota)*dp31)

    X_TDI = (y12_a - y13_a)*np.exp(-1j*f/lisa.fstar) + (y12_a - y13_a)
    
    if (iota == None):
        XX_TDI += 8./5*np.abs(X_TDI)**2
    else:
        XX_TDI += 1./2*np.abs(X_TDI)**2
    
    return XX_TDI    
    

## let's make a class for the binaries
class Binary():

    def __init__(self,m1,m2,d_L=None,z=None,sky_loc=None,circular=True):
        '''
        Class to house binary attributes and related methods.

        Arguments
        --------------------
        m1, m2 (float or astropy.Quantity):
            Source-frame component masses. Assumed to be in solar masses if no unit is provided
        d_L (float or astropy.Quantity, optional):
            Luminosity distance. Assumed to be in Mpc if no unit is provided. 
            One of d_L or z must be provided.
        z (float, optional):
            Redshift. One of d_L or z must be provided.
        sky_loc (astropy.coordinates.SkyCoord object):
            Sky location of the merger as seen from Earth, as an astropy SkyCoord object.
        circular (bool):
            Whether to treat the binary as circular (Default True; eccentric binaries not yet implemented)
        
        '''
        ## we may eventually wish to have eccentric systems
        if not circular:
            raise NotImplementedError

        ## deal with units
        self.m1_source = utils.assert_units(m1,u.Msun)
        self.m2_source = utils.assert_units(m2,u.Msun)

        ## handle sky location
        if sky_loc is not None:
            if type(sky_loc) is not astropy.coordinates.sky_coordinate.SkyCoord:
                raise TypeError("sky_loc must be an astropy SkyCoord object.")
        self.sky_loc = sky_loc
        
        ## process or calculate luminosity distance
        if not ((d_L is None) ^ (z is None)):
            raise ValueError("Either d_L or z must be specified, but not both.")
        elif d_L is None:
            self.z = z # TODO: check that one of these is provided
            self.d_L = get_Dl(self.z, Omega_m, H0,unit_out=u.Mpc) # d_L returned in Mpc
            print("Redshift provided. \n\tLuminosity Distance........... {} Mpc".format(self.d_L))
        elif z is None:
            self.d_L = utils.assert_units(d_L,u.Mpc)
            self.z = optimize.root(get_z, 1., args=(self.d_L, Omega_m, H0)).x[0]

        ## redshift to detector-frame masses
        self.m1 = self.m1_source * (1+self.z)
        self.m2 = self.m2_source * (1+self.z)

        ## get chirp mass and other values
        self.mc_source = utils.get_mc(self.m1_source,self.m2_source)
        self.mc =utils.get_mc(self.m1,self.m2)
        # calculate relevant mass parameters
        self.M   = self.m1 + self.m2 # total mass
        self.eta = (self.m1*self.m2)/self.M**2 # symmetric mass ratio

        ## useful for Steve's Phenom to have some of these in geometric (G=c=1) units
        self.geom_M = self.M.value*pa.TSUN
        self.geom_eta = self.eta.value
        self.geom_Dl = (self.d_L/c.c).decompose().value ## in seconds

        # Obtain the frequency limits of the signal
        self.f_cut = pa.get_freq(self.geom_M, self.geom_eta, "cut")*u.Hz # PhenomA cut-off frequency i.e. frequency upper bound

        self.f_start = None
        self.f_end   = None
        self.T_merge = None
        
        ## call the merger frequency the frequency at t_c - t = 1 s
        ## NOTE: we diverge from the Hazboun+19 convention here; their f0 is the monochromatic PTA-band frequency,
        ##       whereas ours is the frequency at merger in the LISA band
        self.f0 = f_by_dt(1*u.s,self.mc)

        

    ## wrappers for f(t), t(f), h_echo(f), h_echo(t)
    def f_of_t(self,t):
        return f_by_dt(t,self.mc)
    def t_of_f(self,f):
        return(f,self.mc)
    def h_echo_of_f(self,f):
        return calc_h0(self.mc,f,self.d_L)
    def h_echo_of_t(self,t):
        foft = self.f_of_t(t)
        return calc_h_echo(self.mc,foft,self.d_L)

    ##################################
    ## Observatory-generic Wrappers ##
    ##################################

    def CalcStrain(self,observatory,**kwargs):

        if observatory.name == 'LISA':
            f, hc = self.CalcStrain_LISA(observatory, **kwargs) ## returns TDI X
        elif observatory.name == 'muAres' or observatory.name == 'astrometry':
            f, hc = self.CalcStrain_generic(observatory, **kwargs)
        elif observatory.name == 'astrometry':
            f, hc = self.CalcStrain_astrometry(observatory, **kwargs)
        else:
            raise ValueError("Unknown observatory. Can be 'LISA', 'muAres', or 'astrometry'.")

        return f, hc

    def CalcSNR(self,f,hc,observatory):

        if observatory.name == 'LISA':
            snr = self.CalcSNR_LISA(f, hc, observatory) ## assumed TDI X
        elif observatory.name == 'muAres' or observatory.name == 'astrometry':
            snr = self.CalcSNR_generic(f, hc, observatory)
        elif observatory.name == 'astrometry':
            snr = self.CalcSNR_astrometry(f, hc, observatory)
        else:
            raise ValueError("Unknown observatory. Can be 'LISA', 'muAres', or 'astrometry'.")

        return snr

    def run_full_SNR_calc(self,observatory,T_merge,**strain_kwargs):
        '''
        Does the full SNR calculation from start to finish. 

        Arguments
        ---------------
        observatory
            Observatory object
        '''
        self.T_merge = utils.assert_units(T_merge,u.s)
        
        self.SetFreqBounds(observatory)
    
        freqs, hc = self.CalcStrain(observatory,**strain_kwargs)
        snr = self.CalcSNR(freqs, hc, observatory)
        
        ## clean up
        self.T_merge_prev = self.T_merge
        self.f_start_prev = self.f_start
        self.f_end_prev = self.f_end
        self.T_merge = None
        self.f_start = None
        self.f_end = None

        return snr


    ##################################
    ## observatory-agnostic methods ##
    ##################################
    def CalcStrain_generic(self, observatory):
        '''
        Calculates the characteristic GW strain in a generic observatory.
        '''

        Delta_logf = np.log(self.f_end.to(u.Hz).value) - np.log(self.f_start.to(u.Hz).value)
        
        if (Delta_logf > 0.00005): # Generate a track
            N = 500 # number of points
            f = np.logspace(np.log10(self.f_start.to(u.Hz).value), np.log10(self.f_end.to(u.Hz).value), N)
            
            Aeff = pa.Aeff(f, self.geom_M, self.geom_eta, self.geom_Dl)

            self.Figure_Type = 'track'
            hc = 2*f*np.abs(Aeff) ## htilde -> hc
                
        else:
        
            N = 1
            f = np.array([self.f_start.to(u.Hz).value])
            Aeff = pa.Aeff(f, self.geom_M, self.geom_eta, self.geom_Dl)
            
            self.Figure_Type = 'point'

            Ncycles = observatory.Tobs*f

            hc = np.sqrt(Ncycles)*Aeff ## monochromatic hc
        
        return f, hc


    def CalcSNR_generic(self, f, hc, observatory):
        """ Calculate the signal to noise ratio for the source """

        if hasattr(f,"unit"):
            f = f.to(u.Hz).value

        if (self.Figure_Type == 'track'):
            N = len(f) # number of frequency samples
    
            d_logf = np.log(f[1:]) - np.log(f[:N-1])
            
            term_i   = hc[1:]**2/observatory.Sn(f[1:])
            term_im1 = hc[:N-1]**2/observatory.Sn(f[:N-1])

            ## average over the space between points
            snrSQ = np.sum(0.5*(term_i + term_im1)*d_logf) 
            
            
        elif (self.Figure_Type == 'point'):
            f_start = utils.assert_units(self.f_start,u.Hz).value
            f_end = utils.assert_units(self.f_end,u.Hz).value
            snrSQ = (hc**2/(f*observatory.Sn(f)))[0] * (np.log(f_end) - np.log(self.f_start))
            

        return np.sqrt(snrSQ)
    
    #####################
    ## Steve's methods ##
    #####################

    def SetFreqBounds(self, observatory):
        """ 
        Generalized method to set the minimum and maximum frequency for the SNR calculation.

        Arguments
        -----------------
        observatory (Observatory object)
            Observatory object with (at minimum) Tobs attribute.
        """
        
        Mc = self.geom_M*self.geom_eta**(3./5) # chirp mass

        Tobs = utils.assert_units(observatory.Tobs,u.s).value
        
        # Determine start frequency of binary
        if (self.f_start == None): # T_merge was specified
            self.f_start = (5.*Mc/self.T_merge.to(u.s).value)**(3./8.)/(8.*np.pi*Mc) * u.Hz
        else:
            f_start = utils.assert_units(self.f_start,u.Hz).value
            self.T_merge = 5.*Mc/(8.*np.pi*f_start*Mc)**(8./3.)
    
        # Determine the end frequency
        if (self.T_merge.to(u.s).value > Tobs):
            self.f_end = (5.*Mc/(np.abs(Tobs-self.T_merge.to(u.s).value)))**(3./8.)/(8.*np.pi*Mc) * u.Hz
        else:
            self.f_end = pa.get_freq(self.geom_M, self.geom_eta, "cut") * u.Hz # PhenomA cut-off frequency i.e. frequency upper bound    
    
        return

    def CalcStrain_LISA(self, lisa, theta=None, phi=None, iota=None):
        '''
        Calculates the characteristic GW strain in the LISA XX TDI channel.
        '''


        
        Delta_logf = np.log(self.f_end.to(u.Hz).value) - np.log(self.f_start.to(u.Hz).value)
        
        if (Delta_logf > 0.00005): # Generate a track
            N = 500 # number of points
            f = np.logspace(np.log10(self.f_start.to(u.Hz).value), np.log10(self.f_end.to(u.Hz).value), N)
            
            Aeff = pa.Aeff(f, self.geom_M, self.geom_eta, self.geom_Dl)
            
            if (theta == None and phi == None): # generate sky averaged response
                self.Figure_Type = 'track'
                X_char = np.sqrt(16./5*f)*Aeff
                
            else: # Generate X Michelson channel
                self.Figure_Type = 'track_sky_dependent'
    
                XX_TDI  = get_XX_TDI(self, lisa, f, Aeff, theta, phi, iota)
                
                X_char = np.sqrt(4*f*XX_TDI)
                
        else:
        
            N = 1
            f = np.array([self.f_start.to(u.Hz).value])
            Aeff = pa.Aeff(f, self.geom_M, self.geom_eta, self.geom_Dl)
        
            if (theta == None and phi == None): # generate sky averaged response
                       
                self.Figure_Type = 'point'
                
                X_char = np.sqrt(16./5*Aeff**2*np.sqrt(f)*(self.f_end.to(u.Hz).value - f))
                
            else:   
    
                self.Figure_Type = 'point_sky_dependent'
                
                XX_TDI  = get_XX_TDI(self, lisa, f, Aeff, theta, phi, iota)
            
                X_char = np.sqrt(4*XX_TDI*np.sqrt(f)*(self.f_end.to(u.Hz).value - f))
        
        return f, X_char
        
    def CalcSNR_LISA(self, f, X_char, lisa):
        """ Calculate the signal to noise ratio for the source """

        if hasattr(f,"unit"):
            f = f.to(u.Hz).value

        if (self.Figure_Type == 'track'):
            N = len(f) # number of frequency samples
    
            d_logf       = np.log(f[1:]) - np.log(f[:N-1])
            
            term_i   = X_char[1:]**2/lisa.Sn(f[1:])
            term_im1 = X_char[:N-1]**2/lisa.Sn(f[:N-1])
    
            snrSQ = np.sum(0.5*(term_i + term_im1)*d_logf) 
            
        elif (self.Figure_Type == 'track_sky_dependent'):
            N = len(f) # number of frequency samples
    
            d_logf       = np.log(f[1:]) - np.log(f[:N-1])
            
            term_i   = X_char[1:]**2/lisa.Pn_WC(f[1:])*lisa.NC
            term_im1 = X_char[:N-1]**2/lisa.Pn_WC(f[:N-1])*lisa.NC
            
            snrSQ = np.sum(0.5*(term_i + term_im1)*d_logf)
            
        elif (self.Figure_Type == 'point'):
    
            snrSQ = (X_char**2/np.sqrt(f)/lisa.Sn(f))[0]
            
        elif (self.Figure_Type == 'point_sky_dependent'):
    
            snrSQ = (X_char**2/np.sqrt(f)/lisa.Pn_WC(f)*lisa.NC)[0]

        return np.sqrt(snrSQ)
        
    def PlotStrain(self, freqs, X_char, observatory, xlim=None, ylim=None):
        """ Plot the characteristic strain curves """

        if hasattr(freqs,"unit"):
            freqs = freqs.to(u.Hz).value
        
        fig, ax = plt.subplots(1, figsize=(8,6))
        plt.tight_layout()
        
        ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
        ax.set_ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=20)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        
        f = np.logspace(np.log10(1.0e-6), np.log10(1.0e0), 1000)
        
        if (self.Figure_Type == 'track'):
            if observatory.name == 'LISA':
                ax.loglog(freqs, np.sqrt(freqs)*X_char)
            else:
                ax.loglog(freqs, X_char)
            ax.loglog(f, np.sqrt(f*observatory.Sn(f)))
            
        elif (self.Figure_Type == 'track_sky_dependent'):
    
            ax.loglog(freqs, np.sqrt(freqs)*X_char) 
            ax.loglog(f, np.sqrt(f*observatory.Pn_WC(f)))    
            
        elif (self.Figure_Type == 'point_sky_dependent'):
    
            ax.loglog(freqs, np.sqrt(freqs)*X_char, 'r.') 
            ax.loglog(f, np.sqrt(f*observatory.Pn_WC(f)))   
    
        elif (self.Figure_Type == 'point'):
    
            if observatory.name == 'LISA':
                ax.loglog(freqs, np.sqrt(freqs)*X_char, 'r.')
            else:
                ax.loglog(freqs, X_char, 'r.')
            ax.loglog(f, np.sqrt(f*observatory.Sn(f)))    
    
        return


# class Binary():
#     """ 
#     Binary Class
#     -------------------------------------------
#     Inputs:
#         Specify source-frame masses: m1, m2
#         Specify a distance parameter: z, Dl (redshift, luminosity distance IN SECONDS)
#         Specify an initial condition parameter: T_merge, f_start
#                     (note that an upper limit of 4 years will be set on the 
#                      observation period)
    
#     Methods:
#         CalcStrain: Calculate the characteristic strain of the binary. If (the optional
#                     arguments) sky angles are provided use the stataionary phase approximation
#                     signal generator, else use PhenomA amplitude exclusively
                    
#         CalcSNR: Calculate the SNR averaged over polarization, inclination,
#                   and sky angles. Theta, phi (spherical polar) are optional arguments
#                   allowing the user to calculate the SNR at a specific sky location
#                   averaged over only polarization and inclination angles

#         PlotStrain: Plot the characteristic strain
    
#     """
    
#     def __init__(self, m1, m2, z=None, Dl=None):
#         # source-frame component masses
#         self.m1 = m1
#         self.m2 = m2
        
#         # Store distance parameters
#         if (Dl == None): # convert redshift into luminosity distance
#             self.z = z # TODO: check that one of these is provided
#             self.Dl = get_Dl(self.z, Omega_m, H0) # Dl returned in seconds (i.e. G=c=1, geometric units)
#             print(r"Redshift provided. \n\tLuminosity Distance........... {} Mpc".format(self.Dl/MPC))

#         else: # convert luminosity distance to redshift
#             self.Dl = Dl # TODO: check that one of these is provided
#             self.z = optimize.root(get_z, 1., args=(self.Dl, Omega_m, H0)).x[0]
#             print(r"Luminosity Distance provided. \n\tredshift........... {}".format(self.z))
            
#         # adjust source-frame masses to detector-frame masses
#         self.m1 *= 1. + self.z 
#         self.m2 *= 1. + self.z
        
#         # calculate relevant mass parameters
#         self.M   = self.m1 + self.m2 # total mass
#         self.eta = self.m1*self.m2/self.M**2 # symmetric mass ratio
        
#         self.f_start = None
#         self.f_end   = None
        
        
#     # Methods
#     SetFreqBounds = SetFreqBounds
#     CalcStrain    = CalcStrain
#     CalcSNR       = CalcSNR
#     PlotStrain    = PlotStrain








