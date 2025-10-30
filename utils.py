import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

def assert_units(var,unit):
    '''
    Simple function to make sure a variable has a particular astropy.units unit.

    Arguments
    -----------
    var (float, array, astropy.Quantity)
        generic variable
    unit (astropy.units.unit)
    '''
    if not hasattr(var,'unit'):
        var = var*unit
    elif var.unit is not unit:
        var = var.to(unit)
    return var

def get_mc(m_1,m_2):
    '''
    Calc chirp mass from m_1, m_2.
    '''
    return (m_1*m_2)**(3/5) / (m_1+m_2)**(1/5)

def savefig_png_pdf(filepath,extensions=['.png','.pdf'],**savefig_kwargs):
    """
    Utility function to save a figure with multple extensions.

    Parameters
    ----------
    filepath : str
        '/path/to/file/save/location/filename.'
    extensions : list of str, optional
        Filetype extensions to save as, given as a list of strings. The default is ['.png','.pdf'].
    **savefig_kwargs : kwargs
        Keyword arguments for matplotlib.pyplot.savefig.
    
    Returns
    -------
    None.

    """
    
    for ext in extensions:
        ## catch filetype extensions without leading '.'
        if ext[0] != '.':
            ext = '.'+ext
        ## save
        plt.savefig(filepath+ext,**savefig_kwargs)
    
    return

def savefig(filename,saveto=None):
    """
    Utility function to save a figure of name [filename] to path [saveto] as both png and pdf.

    Parameters
    ----------
    filename : str
        Desired filename, sans extensions.
    saveto : str
        '/path/to/file/save/location/'. The default is None (save to current directory).

    Returns
    -------
    None.

    """
    
    if saveto is not None:
        fig_path_base = (saveto + '/{}'.format(filename)).replace('//','/')
    else:
        fig_path_base = filename
    savefig_png_pdf(fig_path_base, dpi=200)
    
    return

def set_style():
    plt.style.use('default')
    default_cycler=plt.cycler(color=['mediumorchid','teal','navy','firebrick','goldenrod','slategrey'])
    plt.rc('axes', prop_cycle=default_cycler)
    
    plt.rcParams['font.family'] = 'STIXGeneral'  # Closely matches Computer Modern
    plt.rcParams['mathtext.fontset'] = 'stix'    # Use STIX for math
    
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    return