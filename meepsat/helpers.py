import inspect
import json
from typing import Callable

import numpy as np

# Used to remove the elements of a dictionary (dict_to_filter) that
# don't correspond to the keyword arguments of a particular
# function (func_with_kwargs.)
# Adapted from https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function/44052550
def filter_dict(dict_to_filter: dict, func_with_kwargs: Callable) -> dict:
    """
    Filters a dictionary to only include keys that are parameters of a given function.
    Args:
        dict_to_filter (dict): The dictionary to filter.
        func_with_kwargs (Callable): The function whose parameter names will be used to filter the dictionary.
    Returns:
        dict: A dictionary containing only the key-value pairs from dict_to_filter where the keys are parameters of func_with_kwargs.
    Raises:
        TypeError: If func_with_kwargs is not a callable.
    """
    if not callable(func_with_kwargs):
        raise TypeError("func_with_kwargs must be callable")

    sig = inspect.signature(func_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values()]

    filtered_dict = {
        filter_key: dict_to_filter[filter_key]
        for filter_key in filter_keys
        if filter_key in dict_to_filter
    }
    return filtered_dict

# funnction to extract the xticks and yticks 
def extract_ticks(data, num_ticks, sim_box):
    """
    Generate tick positions and labels for a given simulation box.
    Parameters:
    data (array-like): The data to be plotted (currently unused).
    num_ticks (int): Number of ticks to generate on each axis.
    sim_box (list of tuples): A list containing two tuples, each representing the 
                              start and end points of the simulation box in the x 
                              and y directions, respectively. 
                              Example: [(x_start, x_end), (y_start, y_end)]
    Returns:
    tuple: A tuple containing four elements:
        - xticks (numpy.ndarray): Positions of ticks along the x-axis.
        - yticks (numpy.ndarray): Positions of ticks along the y-axis.
        - xticklabels (list of str): Labels for the ticks along the x-axis.
        - yticklabels (list of str): Labels for the ticks along the y-axis.
    """
    xticks = np.linspace(sim_box[0][0], sim_box[0][1], num_ticks)
    yticks = np.linspace(sim_box[1][0], sim_box[1][1], num_ticks)
    xticklabels = [f'{xtick:.1f}' for xtick in xticks
                    ]
    yticklabels = [f'{ytick:.1f}' for ytick in yticks
                    ] 
    
    return xticks, yticks, xticklabels, yticklabels


def sys_info(self, dist_unit, wvl=None, meep_freq=None, real_freq=None, points_per_wavelength=20):
    '''
    Writes the file that will then be 
    read within the MEEP simulation

    Arguments
    ---------
    dist_unit : float
        Chosen ratio between MEEP distances and real distance 
    wvl : float, optional
        Wavelength in MEEP units (default : None)
    meep_freq : float, optional
        Frequency in MEEP units (default : None)
    real_freq : float, optional
        Frequency in Hz (default : None)
    points_per_wavelength : int, optional
        Minimum number of grid points per wavelength (default: 20)
    '''

    c = 299792458.0
    if wvl is not None :
        meep_freq = 1/wvl
        real_freq = c*meep_freq/dist_unit

    if real_freq is not None:
        wvl = c/real_freq/dist_unit
        meep_freq = 1/wvl

    if meep_freq is not None: 
        wvl= 1/meep_freq
        real_freq = c*meep_freq/dist_unit

    # Calculate minimum resolution
    min_resolution = points_per_wavelength / wvl

    print('--- System Info ---')
    print('Real Wavelength = {:.1e}m'.format(wvl*dist_unit))
    print('MEEP Wavelength = {:.1e}'.format(wvl))
    print('System size = {:.0f} x {:.0f} wavelengths'.format(self.size_x/wvl, 
                                                    self.size_y/wvl))
    print('System size = {:.2e} x {:.2e} m'.format(self.size_x*dist_unit, 
                                                    self.size_y*dist_unit))
    print('Real frequency = {:.2e} Hz'.format(real_freq))
    print('MEEP frequency = {:.2e}'.format(meep_freq))
    if min_resolution:
        print('Minimum resolution (grid points/unit length): {:.2f}'.format(min_resolution))
        print('  (for {} points per wavelength)'.format(points_per_wavelength))
    print('------------------')


def read_json(json_file):
    """
    Reads a JSON file and returns the data as a dictionary.
    Args:
        json_file (str): Path to the JSON file.
    Returns:
        dict: The data from the JSON file.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def rotation_matrix(tx, ty, tz):
    """
    Zemax order: Rx -> Ry -> Rz (applied in that order)
    Equivalent to: Rz @ Ry @ Rx
    """
    return rot_z(tz) @ rot_y(ty) @ rot_x(tx)
