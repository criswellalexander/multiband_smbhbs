import numpy as np
import astropy.units as u

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