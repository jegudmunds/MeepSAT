from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import logging
import sys
import os
import site
from pathlib import Path
#import meep_testings as mp
import meep as mp
import numpy as np
import warnings
import math

import matplotlib.pyplot as plt
from matplotlib import rc

# MeepSAT functions
import meepsat.helpers as exf
import meepsat.meshing as mesh

# * ############################################################################################################
# * ############################################################################################################

# Defining some global functions that will decrease the length of the code
def set_sims_obj(self, mpsat_sim):
    """
    Set the MEEPSAT simulation object in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    sim : meep.Simulation
        MEEP simulation object

    mpsat_sim : MEEPSAT
        MEEPSAT simulation object
    """
    if mpsat_sim is None:
        raise ValueError("MEEPSAT simulation object is missing!")
    else:
        self.mpsat_sim = mpsat_sim

    return self.mpsat_sim

def set_center(self, center= None, default_center = None):
    """
    Set the material center in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    center : mp.Vector3
        Center of the material in the x, y and z directions
        Format : mp.Vector3(x, y, z)
    """
    if center is not None:
        self.center = center
    else:
        # There should always be a default center present in the class
        if default_center is None:
            raise ValueError(f"No center given to the material object in the {self.__class__.__name__} class!")
        else:
            warnings.warn(f"No center given to the material object: Taking the default center as {default_center}")
            self.center = default_center
    
    return self.center
        
def set_size(self, size= None, default_size = None):
    """
    Set the material size in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    size : mp.Vector3
        Size of the material in the x, y and z directions
        Format : mp.Vector3(sx, sy, sz)
    """
    if size is not None:
        self.size = size
    else:
        # There should always be a default size present in the class
        if default_size is None:
            raise ValueError(f"No size given to the material object in the {self.__class__.__name__} class!")
        else:
            warnings.warn(f"No size given to the material object: Taking the default size as {default_size}")
            self.size = default_size
    
    return self.size

def set_prop_component(self, component= None, default_component = None):
    """
    Set the component in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    component : str or meep component
        Propagating component of the source, can be a string ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz')
        or the actual MEEP component (mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz)
    """
    avail_components = ['Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz']
    meep_components = [mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz]
    
    # If component is None, use default_component
    if component is None:
        if default_component is None:
            raise ValueError(f"No component given to the object in the {self.__class__.__name__} class!")
        else:
            warnings.warn(f"No component given: Taking the default component as {default_component}")
            self.component = default_component
            return self.component
    
    # If component is already a MEEP component, use it directly
    if component in meep_components:
        self.component = component
        return self.component
    
    # If component is a string, convert to the corresponding MEEP component
    if isinstance(component, str):
        if component in avail_components:
            idx = avail_components.index(component)
            self.component = meep_components[idx]
            return self.component
    
    # If we reach here, the component is invalid
    raise ValueError(f"Invalid component given to the source object in the {self.__class__.__name__} class! "
                     f"Please choose from {avail_components} or use MEEP components directly (mp.Ez, mp.Ex, etc.)")

def set_freq_wvl(self, freq= None, wvl= None):
    """
    Set the frequency and wavelength in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    freq : float
        Frequency of the source in MEEP units (default : None)
    
    wvl : float
        Wavelength of the source in MEEP units (default : None)
        If freq is given, wvl is calculated as 1/freq
    """
    if freq is not None:
        self.freq = freq
        self.wvl = 1/freq
    elif wvl is not None:
        self.wvl = wvl
        self.freq = 1/wvl
    else:
        raise ValueError("Frequency or wavelength is missing! Please provide either the frequency or the wavelength")
    
    return self.freq, self.wvl

def set_source_angle(self, angle= None):
    """
    Set the angle of the source in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    angle : float
        Angle by which the plane wave is rotated w.r.t vertical (default : None)

    Returns
    -------
    self.rot_angle : float
        Angle of the source in radians
    """
    if angle is not None:
        self.rot_angle = angle
        self.rot_angle *= np.pi/180
        print(f"Angle of the source:{self.rot_angle} rad = {angle} degrees")
    else:
        self.rot_angle = 0

    return self.rot_angle

def set_material_obj(self, epsilon_real, epsilon_imag, freq):
    """
    Set the material object in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    epsilon_real : float
        Real part of the permittivity of the material

    epsilon_imag : float
        Imaginary part, i.e. conductivity, of the material

    freq : float
        Frequency at which the sim will be run.
        Needed to set the material property accordingly
    """
    self.epsilon_real = epsilon_real
    self.epsilon_imag = epsilon_imag
    self.freq = freq
    self.conductivity = epsilon_imag*2*np.pi*freq/epsilon_real

### -- RELEVANT FUNCTIONS FOR BROADBAND SOURCE -- ###
def set_broadband_freq_wvl(self, central_wvl=None, wvl_min=None, wvl_max=None):
    """
    Set the frequency and wavelength for the broadband source

    Parameters
    ----------
    self : object
        Class object

    central_wvl : float
        Central wavelength of the source in MEEP units (default : None)

    wvl_min : float
        Minimum wavelength of the source in MEEP units (default : None)

    wvl_max : float
        Maximum wavelength of the source in MEEP units (default : None)

    Returns
    -------
    self.freq : float
        Frequency of the source in MEEP units

    self.wvl : float
        Wavelength of the source in MEEP units
    """
    self.wvl_min = wvl_min
    self.wvl_max = wvl_max
    
    if central_wvl is None:
        self.central_wvl = (self.wvl_min + self.wvl_max) / 2

    self.center_freq = 1/self.central_wvl
    self.freq_width = 1/self.wvl_min - 1/self.wvl_max

    return self.freq_width, self.center_freq

# * ############################################################################################################

###& SOURCE
# ~ CONTINUOUS PLANE WAVES
class ContinuousPlaneWaves():
    """
    Class defining the continuous plane waves source
    """
    def __init__(self,
                 mpsat_sim,
                 center = None,
                 size = None,
                 component = None,
                 freq = None,
                 wvl = None,
                 angle = 0,
                 rot_axis= 'x',
                 kwargs= None):
        """
        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object

        center : mp.Vector3
            Center of the source in the x, y and z directions (default : mp.Vector3(0, 0, 0))
            Format : mp.Vector3(x, y, z)
        
        size : mp.Vector3
            Size of the source in the x, y and z directions (default : None)
            Format : mp.Vector3(sx, sy, sz)

        component : str or meep component
            Propagating component of the source, can be a string ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz')
            or the actual MEEP component (mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz)
        
        freq : float
            Frequency of the source in MEEP units (default : mpsat_sim.freq)
        
        wvl : float
            Wavelength of the source in MEEP units (default : mpsat_sim.wvl)
            If freq is given, wvl is calculated as 1/freq

        angle : float (optional)
            Angle by which the plane wave is rotated w.r.t vertical (default : 0)

        rot_axis : str
            Axis around which the source is rotated (default : 'x')

        **kwargs : dict
            Additional arguments for the meep.Source()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#source
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#continuoussource
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        # Centre
        self.center = set_center(self, center, default_center = mp.Vector3(0, 0, 0))
        # Size
        self.size = set_size(self, size, default_size = mp.Vector3(0, self.mpsat_sim.cell_size[1], 0))
        # Propagating component
        self.component = set_prop_component(self, component, default_component = mp.Ez)
        # Frequency and wavelength
        self.freq, self.wvl = set_freq_wvl(self, freq, wvl)
        # Angle of the source
        self.rot_angle = set_source_angle(self, angle)       
        # Rotation axis of the source
        self.rot_axis = rot_axis
        # Additional arguments for both mp.Source() and mp.ContinuousSource()
        self.additional_args = kwargs

        print("Source object created with the following parameters:")
        print("Center: ", self.center)
        print("Size: ", self.size)
        print("Component: ", self.component)
        print("Frequency: ", self.freq)
        print("Wavelength: ", self.wvl)
        print("Angle: ", self.rot_angle)
        print("Rotation axis: ", self.rot_axis)
        print("Additional arguments: ", self.additional_args)

    def amp_func(self, P):
        '''
        Adopted from MEEPART package
        ---
        Returns amplitude of source with added phase to 
        emulate source rotation

        Parameters
        ---------
        P : mp.Vector3
            Meep position object at which the source is evaluated.

        Returns
        -------
        amp : complex
            Complex amplitude of source at P.
        '''
        
        if self.rot_axis=='x':
            k = mp.Vector3(2*np.pi*np.cos(self.rot_angle)/self.wvl,
                        2*np.pi*np.sin(self.rot_angle)/self.wvl,
                        0)
        elif self.rot_axis=='y':
            k = mp.Vector3(2*np.pi*np.sin(self.rot_angle)/self.wvl,
                        2*np.pi*np.cos(self.rot_angle)/self.wvl,
                        0)
        else:
            raise ValueError("Invalid Rotation axis. Choose either 'x' OR 'y'")
        
        return np.exp(1j* k.dot(P))
    
    def assemble(self):
        """
        Return continuous planewaves source
        """
        if self.additional_args is not None:
            source_filtered_kwrg = exf.filter_dict(self.additional_args, mp.Source)
            print("Additional arguments for the Source: ", source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.additional_args, mp.ContinuousSource)
            print("Additional arguments for the ContinuousSource: ", source_type_filtered_kwrg)

            source = mp.Source(mp.ContinuousSource(frequency=self.freq, 
                                                   **source_type_filtered_kwrg),
                               center= self.center,
                               size= self.size,
                               component=self.component,
                               amp_func=self.amp_func,
                               **source_filtered_kwrg)
            
        else:
            source = mp.Source(mp.ContinuousSource(frequency=self.freq),
                               center= self.center,
                               size= self.size,
                               component=self.component,
                               amp_func=self.amp_func)
        
        print("Continuous plane waves source assembled!")
        return source



# ~ BROADBAND PLANE WAVES
class BroadbandPlaneWaveSource():
    """
    Class defining the broadband plane waves source
    """
    def __init__(self,
                 mpsat_sim,
                 center = None,
                 size = None,
                 component = None,
                 central_wvl = None,
                 wvl_min= None,
                 wvl_max= None,
                 angle = 0,
                 rot_axis= 'x',
                 kwargs= None):
        """
        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object

        center : mp.Vector3
            Center of the source in the x, y and z directions (default : mp.Vector3(0, 0, 0))
            Format : mp.Vector3(x, y, z)
        
        size : mp.Vector3
            Size of the source in the x, y and z directions (default : None)
            Format : mp.Vector3(sx, sy, sz)

        component : str or meep component
            Propagating component of the source, can be a string ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz')
            or the actual MEEP component (mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz)
        
        freq : float
            Frequency of the source in MEEP units (default : mpsat_sim.freq)
        
        wvl : float
            Wavelength of the source in MEEP units (default : mpsat_sim.wvl)
            If freq is given, wvl is calculated as 1/freq

        angle : float (optional)
            Angle by which the plane wave is rotated w.r.t vertical (default : 0)

        rot_axis : str
            Axis around which the source is rotated (default : 'x')

        **kwargs : dict
            Additional arguments for the meep.Source()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#source
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#continuoussource
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        
        # Centre
        self.center = set_center(self, center, default_center = mp.Vector3(0, 0, 0))
        # Size
        self.size = set_size(self, size, default_size = mp.Vector3(0, self.mpsat_sim.cell_size[1], 0))
        # Propagating component
        self.component = set_prop_component(self, component, default_component = mp.Ez)
        # Frequency and wavelength
        self.wvl_min, self.wvl_max, self.central_wvl = wvl_min, wvl_max, central_wvl
        self.freq_width, self.center_freq = set_broadband_freq_wvl(self, central_wvl, wvl_min, wvl_max)
        # Angle of the source
        self.rot_angle = set_source_angle(self, angle)       
        # Rotation axis of the source
        self.rot_axis = rot_axis
        # Additional arguments for both mp.Source() and mp.ContinuousSource()
        self.additional_args = kwargs

        print("Source object created with the following parameters:")
        print("Center: ", self.center)
        print("Size: ", self.size)
        print("Component: ", self.component)
        print("Central Frequency and Freq Width: ", self.center_freq, self.freq_width)
        print("Wavelength Range and Central Wavelength:", self.wvl_min, self.wvl_max, self.central_wvl)
        print("Angle: ", self.rot_angle)
        print("Rotation axis: ", self.rot_axis)
        print("Additional arguments: ", self.additional_args)

    def amp_func(self, P):
        '''
        Adopted from MEEPART package
        ---
        Returns amplitude of source with added phase to 
        emulate source rotation

        Parameters
        ---------
        P : mp.Vector3
            Meep position object at which the source is evaluated.

        Returns
        -------
        amp : complex
            Complex amplitude of source at P.
        '''
        
        if self.rot_axis=='x':
            k = mp.Vector3(2*np.pi*np.cos(self.rot_angle)/(1/self.wvl_max),
                        2*np.pi*np.sin(self.rot_angle)/(1/self.wvl_max),
                        0)
        elif self.rot_axis=='y':
            k = mp.Vector3(2*np.pi*np.sin(self.rot_angle)/(1/self.wvl_max),
                        2*np.pi*np.cos(self.rot_angle)/(1/self.wvl_max),
                        0)
        else:
            raise ValueError("Invalid Rotation axis. Choose either 'x' OR 'y'")
        
        return np.exp(1j* k.dot(P))

    
    def assemble(self):
        """
        Return Broadband planewave Pulse
        """
        if self.additional_args is not None:
            source_filtered_kwrg = exf.filter_dict(self.additional_args, mp.Source)
            print("Additional arguments for the Source: ", source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.additional_args, mp.GaussianSource)
            print("Additional arguments for the GaussianSource: ", source_type_filtered_kwrg)

            source = mp.Source(mp.GaussianSource(frequency= self.center_freq,
                                                fwidth= 2*self.freq_width,
                                                **source_type_filtered_kwrg),
                            center= self.center,
                            size= self.size,
                            component=self.component,
                            # amp_func=self.amp_func,
                            **source_filtered_kwrg)
            
        else:
            source = mp.Source(mp.GaussianSource(frequency= self.center_freq,
                                                fwidth= 2*self.freq_width),
                            center= self.center,
                            size= self.size,
                            component=self.component)#,
                            # amp_func=self.amp_func)
        
        print("Broadband plane waves source assembled!")
        return source



class GaussianBeam():

    def __init__(self,
                 mpsat_sim,
                 center = None,
                 size = None,
                 component = None,
                 freq = None,
                 wvl = None,
                 angle = 0,
                 width = 10,
                 cutoff = 0,
                 kwargs= None):

        """
        Parameters
        ----------

        center : mp.Vector3
            Center of the source in the x, y and z directions (default : None)
            Format : mp.Vector3(x, y, z)

        size : mp.Vector3
            Size of the source in the x, y and z directions (default : None)
            Format : mp.Vector3(sx, sy, sz)

        component : mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz
            Propagating component of the source (default : None)

        freq : float
            Frequency of the source in MEEP units (default : mpsat_sim.freq)
        
        wvl : float
            Wavelength of the source in MEEP units (default : mpsat_sim.wvl)
            If freq is given, wvl is calculated as 1/freq

        angle : float (optional)
            Angle by which the plane wave is rotated w.r.t vertical (default : 0)

        width : float
            Width of the Gaussian pulse (default : 10)

        cutoff : float
            Cutoff of the Gaussian pulse (default : 5)

        **kwargs : dict
            Additional arguments for the meep.Source() and meep.GaussianSource()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#source
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#gaussiansource  

            Note: width and cutoff (arguments for mp.GaussianSource) are already defined
            in the function definition because they are specific to the Gaussian beam                    
        """

        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        # Centre
        self.center = set_center(self, center, default_center = mp.Vector3(0, 0, 0))
        # Size
        self.size = set_size(self, size, default_size = mp.Vector3(0, self.mpsat_sim.cell_size[1], 0))
        # Propagating component
        self.component = set_prop_component(self, component, default_component = mp.Ez)
        # Frequency and wavelength
        self.freq, self.wvl = set_freq_wvl(self, freq, wvl)
        # Angle of the source (in radians)
        self.rot_angle = set_source_angle(self, angle)
        # k-vector
        self.k_vector = self.calculate_wave_vector(self.rot_angle, self.wvl)
        # Width of the Gaussian pulse
        self.width = width
        # Cutoff of the Gaussian pulse
        self.cutoff = cutoff
        # Additional arguments for both mp.Source() and mp.GaussianSource()
        self.additional_args = kwargs

    def help_gaussian_beam(self, taper_angle, wvl,
                                beam_waist = None,
                                taper = None):
        '''
        For a gaussian beam source
        Provides taper when given beam waist and provides beam waist
        when given taper, at a given taper angle and wavelength in meep units.
        Arguments
        ---------
        taper_angle : float
            Angle in degrees at which the taper is given
        wvl : float
            Wavelength of the source
        beam_waist : float, optional
            Size of the beam waist in MEEP units
        taper : float, optional
            Taper in dB
        '''                        

        a = 20*np.log10((1 + np.cos(np.radians(taper_angle)))/2)
        b = 10*(2*np.pi)**2 * (1-np.cos(np.radians(taper_angle)))*np.log10(np.exp(1)) 

        if beam_waist is None :
            w0 = np.sqrt(- wvl**2 * (taper - a)/b)
            print('The beam waist is {:.2e} MEEP units'.format(w0))

        if taper is None :
            A = a - b* beam_waist**2 / wvl**2
            print('The taper at angle {:.1f} deg is {:.2f} dB'.format(taper_angle, A))

    def calculate_wave_vector(self, angle_rad, wavelength):
        """
        Calculate the wave vector based on the angle and wavelength.

        Parameters
        ----------
        angle : float
            Angle of the source in radians.
        wavelength : float
            Wavelength of the source.

        Returns
        -------
        k_vector : mp.Vector3
            Wave vector for the given angle and wavelength.
            The length of the wave vector is ignored in Gaussian beam sources.
        """
        kx = 2 * np.pi * np.cos(angle_rad) / wavelength
        ky = 2 * np.pi * np.sin(angle_rad) / wavelength
        return mp.Vector3(kx, ky, 0)

    """def gaussianProfile(self,
                        vec):
        w0 = self.width
        return np.exp(-np.square((vec.x-(self.center.x))/w0))"""

    def assemble(self):
        """
        Return Gaussian beam source
        """

        if self.additional_args is not None:
            continuous_source_filtered_kwrg = exf.filter_dict(self.additional_args, mp.ContinuousSource)
            print("Additional arguments for the ContinuousSource: ", continuous_source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.additional_args, mp.GaussianBeam2DSource)
            print("Additional arguments for GaussianBeamSource: ", source_type_filtered_kwrg)

            source = mp.GaussianBeam2DSource(mp.ContinuousSource(frequency=self.freq,
                                                               **continuous_source_filtered_kwrg), 
                                            beam_w0=self.width,
                                            beam_kdir=self.k_vector,
                                            center= self.center,
                                            size= self.size,
                                            component=self.component,
                                            **source_type_filtered_kwrg)
            
            
        else:
            source = mp.GaussianBeam2DSource(mp.ContinuousSource(frequency=self.freq),
                                            beam_w0=self.width,
                                            beam_kdir=self.k_vector,
                                            center= self.center,
                                            size= self.size,
                                            component=self.component)
        print("Gaussian beam source assembled!")
        return source

#* Defining some global functions for meep block object
def meep_block(size, 
               center, 
               material,
               angle=0,
               rot_axis='z',
               e1=mp.Vector3(1, 0, 0),
               e2=mp.Vector3(0, 1, 0),
               e3=mp.Vector3(0, 0, 1),
               **kwargs):
    """
    Returns the block object for the source.

    Parameters
    ----------
    size : mp.Vector3
        Size of the block in the x, y, and z directions.
    center : mp.Vector3
        Center of the block in the x, y, and z directions.
    material : mp.Medium
        Material of the block.
    angle : float (optional) 
        Angle by which the block is rotated w.r.t the rot_axis anticlockwise (default: 0).
        Units are in degrees. 
    rot_axis : str (optional)
        Axis about which the block is rotated (default: 'x').
    e1, e2, e3 : mp.Vector3 (optional)
        Vectors defining the x, y, and z axes of the block (default: standard unit vectors).
    kwargs : dict
        Additional arguments for the meep.Block().
    """
    
    # Check for valid rotation axis
    if rot_axis not in ['x', 'y', 'z']:
        raise ValueError("Invalid rotation axis. Choose from 'x', 'y', or 'z'.")

    # Set the rotation axis
    if rot_axis == 'x':
        axis = mp.Vector3(1, 0, 0)
    elif rot_axis == 'y':
        axis = mp.Vector3(0, 1, 0)
    else:  # Default: z-axis
        axis = mp.Vector3(0, 0, 1)

    # Apply rotation if angle is non-zero
    if angle != 0:
        e1 = e1.rotate(axis, math.radians(angle))
        e2 = e2.rotate(axis, math.radians(angle))
        e3 = e3.rotate(axis, math.radians(angle))
        print(f"Rotating block by {angle}° around {rot_axis}-axis")


    # Return the block with the given parameters
    return mp.Block(size=size,
                    center=center,
                    material=material,
                    e1=e1,
                    e2=e2,
                    e3=e3,
                    **kwargs)


def ensure_ccw_order(vertices):
    """
    Returns the vertices in counter-clockwise order, as mp.Prism expects

    Parameters
    ----------
    vertices : list of mp.Vector3
        Vertices of a planar polygon in the xy-plane

    Returns
    -------
    list of mp.Vector3
        The same vertices, reversed if they were clockwise
    """
    # Signed area from the shoelace formula: negative means clockwise
    n = len(vertices)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].x * vertices[j].y
        area -= vertices[j].x * vertices[i].y

    if area < 0:
        return list(reversed(vertices))
    return vertices



#& APERTURE STOP


@dataclass
class ApertureAbsorberLayer:
    """
    Configuration for one absorber layer on a side of an aperture stop.

    Layers are stacked outwards from the face they sit on, in the order they
    are given, so a graded stack is just a list of layers on the same side.

    Attributes
    ----------
    side : {'front', 'back', 'inner'}
        Which face of the aperture stop blocks the layer sits on, in the
        stop's own frame:
        'front'  - the face the light arrives on (-x for a vertical stop,
                   -y for a horizontal one)
        'back'   - the opposite face
        'inner'  - the aperture lip, i.e. the faces bordering the opening
    thickness : float
        Thickness of the layer along the face normal. For a pyramidal layer
        this is the pyramid height, unless `height` is given separately.
    material : mp.Medium, optional
        Material of the layer. If None, it is built from epsilon_real and
        epsilon_imag (default: None)
    epsilon_real, epsilon_imag : float, optional
        Complex permittivity used when `material` is None. The imaginary part
        becomes a conductivity, D_conductivity = eps_imag*2*pi*freq/eps_real
        (defaults: 1.0 and 0.0)
    freq : float, optional
        Frequency at which epsilon_imag is converted to a conductivity
        (default: 1/3)
    shape : {'flat', 'pyramidal'}, optional
        'flat' places a slab covering the whole face; 'pyramidal' places a
        row of stepped pyramids on it, via PyramidalAbsorbers
        (default: 'flat')
    blocks : {'both', 'up', 'down'}, optional
        Which of the two aperture stop blocks the layer is applied to
        (default: 'both')
    height : float, optional
        Pyramid height, for shape='pyramidal' only (default: `thickness`)
    num_pyramids : int, optional
        Number of pyramids along the face, for shape='pyramidal'
        (default: 10)
    n_layers : int, optional
        Number of steps per pyramid, for shape='pyramidal' (default: 10)
    top_width : float, optional
        Width of the pyramid tip, for shape='pyramidal' (default: 0)
    pyramid_kwargs : dict, optional
        Extra keyword arguments passed straight to PyramidalAbsorbers
    """
    side: Literal['front', 'back', 'inner']
    thickness: float
    material: mp.Medium = None
    epsilon_real: float = 1.0
    epsilon_imag: float = 0.0
    freq: float = 1/3
    shape: Literal['flat', 'pyramidal'] = 'flat'
    blocks: Literal['both', 'up', 'down'] = 'both'
    # Pyramidal-only parameters
    height: float = None
    num_pyramids: int = 10
    n_layers: int = 10
    top_width: float = 0
    pyramid_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.side not in ('front', 'back', 'inner'):
            raise ValueError(f"Invalid absorber side: {self.side}. "
                             "Choose 'front', 'back' or 'inner'.")
        if self.shape not in ('flat', 'pyramidal'):
            raise ValueError(f"Invalid absorber shape: {self.shape}. "
                             "Choose 'flat' or 'pyramidal'.")
        if self.blocks not in ('both', 'up', 'down'):
            raise ValueError(f"Invalid absorber blocks: {self.blocks}. "
                             "Choose 'both', 'up' or 'down'.")
        if self.extent <= 0:
            raise ValueError(f"Absorber layer on the {self.side} side needs a "
                             "positive thickness (or height, if pyramidal).")

    @property
    def extent(self):
        """How far the layer reaches out from the face it sits on"""
        if self.shape == 'pyramidal' and self.height is not None:
            return self.height
        return self.thickness

    def get_material(self):
        """The mp.Medium of the layer, built from epsilon if not given directly"""
        if self.material is not None:
            return self.material

        if self.epsilon_imag:
            # Narrow-band absorption, as in meep.readthedocs.io/en/latest/Materials
            D_conductivity = self.epsilon_imag * 2 * np.pi * self.freq / self.epsilon_real
            return mp.Medium(epsilon=self.epsilon_real, D_conductivity=D_conductivity)

        return mp.Medium(epsilon=self.epsilon_real)


class ApertureStop(object):
    '''
    Class defining an aperture stop
    '''

    def __init__(self,
                 mpsat_sim,
                 type,
                 diameter, 
                 thickness,
                 pos_x=None,
                 pos_y=None,
                 orientation = None,
                 n_refr = 1,
                 material= None,
                 conductivity = mp.inf,
                 rot_axis = 'x',
                 rot_angle = 0,
                 y_centre_offset = [0,0],
                 y_size_offset = [0,0],
                 print_vertices = False,
                 absorber_layers = None,
                 preserve_aperture = False,
                 absorber_height = 1.0):
        '''
        Defines the attributes of the aperture stop object

        Arguments
        ---------
        mpsat_sim : object
            MEEPSAT simulation object
        type: str
            Type of the aperture stop: 
                circular, square, arrow, etc.
        diameter : float 
            Diameter of the aperture stop opening
        pos_x : float, optional
            For a 'vertical' aperture stop: position of the left surface of
            the slab along the x-axis.
            For a 'horizontal' aperture stop: position of the centre of the
            opening along the x-axis (defaults to 0 if not given).
        pos_y : float, optional
            For a 'horizontal' aperture stop: position of the bottom surface
            of the slab along the y-axis.
            For a 'vertical' aperture stop: position of the centre of the
            opening along the y-axis (defaults to 0 if not given).
            At least one of pos_x/pos_y must be given; both may be given
            together to place an off-axis (decentred) opening.
        orientation : str, optional
            'vertical' (slab normal along x) or 'horizontal' (slab normal
            along y). If not given, it is inferred from which position was
            supplied: pos_x only -> 'vertical', pos_y only -> 'horizontal'.
            It is required when both pos_x and pos_y are given.
        thickness : float
            Thickness of aperture stop slab
        n_refr : float, optional 
            Index of refraction of the material 
            if the stop is dielectric
            (default = 1)
        conductivity : float, optional
            Conductivity of the material (default = mp.inf)
        rot_axis : str, optional
            Axis about which the aperture stop is rotated 
            (default : 'x')
        rot_angle : float, optional
            Angle by which the aperture stop is rotated w.r.t rot_axis (default : 0 degrees)
        print_vertices : bool, optional
            If True, the corner vertices of both blocks are printed when the
            aperture stop is assembled (default : False). They can also be
            printed at any time with print_vertices(), or retrieved with
            get_vertices().
        absorber_layers : list, optional
            Absorber layers to put on the faces of the aperture stop, as
            ApertureAbsorberLayer objects or as dicts of their fields
            (default : None). Layers on the same side are stacked outwards in
            the order given. They are NOT returned by assemble(); collect them
            with get_absorbers().
        preserve_aperture : bool, optional
            If True, blocks carrying absorber layers on their 'inner' side are
            pulled back by the thickness of that stack, so the clear opening
            left between the absorbers is exactly `diameter` (default : False,
            i.e. the absorbers eat into the opening).
        absorber_height : float, optional
            Height along z of the absorber prisms; irrelevant in a 2D
            simulation (default : 1.0)
        '''
        self.mpsat_sim = mpsat_sim
        self.type = type                
        self.thick = thickness
        
        # Check that at least one position is specified
        if pos_x is None and pos_y is None:
            raise ValueError("Must specify either pos_x or pos_y for the aperture stop position.")

        # Work out the orientation of the slab
        if orientation is None:
            if pos_x is not None and pos_y is not None:
                raise ValueError("When both pos_x and pos_y are given, the orientation "
                                 "('vertical' or 'horizontal') must be specified.")
            # Blocks oriented vertically (along y-axis) if only pos_x is given,
            # horizontally (along x-axis) if only pos_y is given
            orientation = 'vertical' if pos_x is not None else 'horizontal'

        if orientation not in ('vertical', 'horizontal'):
            raise ValueError(f"Invalid orientation: {orientation}. Choose 'vertical' or 'horizontal'.")

        self.orientation = orientation

        # The position along the slab normal is mandatory, the position along the
        # opening is optional and defaults to a centred (on-axis) opening
        if self.orientation == 'vertical':
            if pos_x is None:
                raise ValueError("A 'vertical' aperture stop needs pos_x (position of its left surface).")
            # Convert the pos_x in (0,x) coordinate system to (-x/2, x/2)
            self.pos_x = pos_x #- self.mpsat_sim.cell.x/2
            self.pos_y = 0 if pos_y is None else pos_y
        else:
            if pos_y is None:
                raise ValueError("A 'horizontal' aperture stop needs pos_y (position of its bottom surface).")
            # Convert the pos_y in (0,y) coordinate system to (-y/2, y/2)
            self.pos_y = pos_y #- self.mpsat_sim.cell.y/2
            self.pos_x = 0 if pos_x is None else pos_x

        self.diameter = diameter        
        self.permittivity = n_refr**2   
        self.conductivity = conductivity
        self.material = material 
        self.object_type = 'AP_stop'
        self.rot_axis = rot_axis
        self.rot_angle = rot_angle
        self.y_centre_offset = y_centre_offset
        self.y_size_offset = y_size_offset
        self._print_vertices = print_vertices
        self.blocks = None      # Filled in once the aperture stop is assembled

        # Absorber layers on the faces of the stop
        self.absorber_layers = self._normalise_absorber_layers(absorber_layers)
        self.preserve_aperture = preserve_aperture
        self.absorber_height = absorber_height
        self.absorbers = None   # Filled in by get_absorbers()

        print(f"Aperture stop created with orientation: {self.orientation} "
              f"at (pos_x, pos_y) = ({self.pos_x}, {self.pos_y})")
        print("type material: ", self.material)

        if self.absorber_layers:
            sides = ', '.join(f'{layer.side}/{layer.shape}' for layer in self.absorber_layers)
            print(f"Absorber layers requested on: {sides} "
                  f"(preserve_aperture = {self.preserve_aperture})")


    @staticmethod
    def _normalise_absorber_layers(absorber_layers):
        '''
        Turns the absorber_layers argument into a list of ApertureAbsorberLayer

        Accepts ApertureAbsorberLayer objects, dicts of their fields, or a
        single one of either instead of a list.
        '''
        if not absorber_layers:
            return []

        if isinstance(absorber_layers, (ApertureAbsorberLayer, dict)):
            absorber_layers = [absorber_layers]

        layers = []
        for layer in absorber_layers:
            if isinstance(layer, dict):
                layers.append(ApertureAbsorberLayer(**layer))
            elif isinstance(layer, ApertureAbsorberLayer):
                layers.append(layer)
            else:
                raise TypeError("absorber_layers must hold ApertureAbsorberLayer "
                                f"objects or dicts, not {type(layer).__name__}")

        return layers

    def _layers_on(self, side, block):
        '''
        Returns the absorber layers on one side of one block ('up' or 'down'),
        in the order they are stacked outwards from the face
        '''
        return [layer for layer in self.absorber_layers
                if layer.side == side and layer.blocks in ('both', block)]

    def _aperture_inset(self, block):
        '''
        How far the given block is pulled back from the opening so that the
        inner absorber stack does not eat into the clear aperture
        '''
        if not self.preserve_aperture:
            return 0

        return sum(layer.extent for layer in self._layers_on('inner', block))

    def _check_block_sizes(self, size_up, size_down):
        '''
        Clips the two block sizes to zero and warns if the opening has been
        shifted so far off-axis that a block falls outside the cell
        '''
        if size_up < 0 or size_down < 0:
            warnings.warn("The aperture stop opening extends beyond the cell: "
                          "the block sizes have been clipped to 0. Check the "
                          "position/diameter of the aperture stop.")
        return max(size_up, 0), max(size_down, 0)

    def square_aperture(self):
        '''
        Returns the block object for the aperture stop
        '''
        if self.material is not None:
            material = self.material
        else:
            material = mp.Medium(epsilon=self.permittivity, 
                                D_conductivity = self.conductivity)
        
        # Half-opening seen by each block: with preserve_aperture the block is
        # pulled back so its inner absorber stack ends on the clear aperture
        half_open_up = self.diameter/2 + self._aperture_inset('up')
        half_open_down = self.diameter/2 + self._aperture_inset('down')

        if self.orientation == 'vertical':
            # Original implementation: blocks along y-axis, positioned at pos_x
            # pos_y (0 if not given) is the centre of the opening along y
            open_centre = self.pos_y
            block_size_y_up = self.mpsat_sim.cell.y/2 - open_centre - half_open_up
            block_size_y_down = self.mpsat_sim.cell.y/2 + open_centre - half_open_down
            block_size_y_up, block_size_y_down = self._check_block_sizes(block_size_y_up,
                                                                        block_size_y_down)

            size_up = mp.Vector3(self.thick, block_size_y_up + self.y_size_offset[0], 0)
            centre_up = mp.Vector3(self.pos_x + (self.thick/2),
                                open_centre + half_open_up + (block_size_y_up + self.y_size_offset[0])/2 + self.y_centre_offset[0],
                                0)

            size_down = mp.Vector3(self.thick, block_size_y_down + self.y_size_offset[1], 0)
            centre_down = mp.Vector3(self.pos_x + (self.thick/2),
                                    open_centre - half_open_down - (block_size_y_down + self.y_size_offset[1])/2 + self.y_centre_offset[1],
                                    0)

        else:  # orientation == 'horizontal'
            # New implementation: blocks along x-axis, positioned at pos_y
            # pos_x (0 if not given) is the centre of the opening along x
            open_centre = self.pos_x
            block_size_x_right = self.mpsat_sim.cell.x/2 - open_centre - half_open_up
            block_size_x_left = self.mpsat_sim.cell.x/2 + open_centre - half_open_down
            block_size_x_right, block_size_x_left = self._check_block_sizes(block_size_x_right,
                                                                           block_size_x_left)

            size_up = mp.Vector3(block_size_x_right + self.y_size_offset[0], self.thick, 0)
            centre_up = mp.Vector3(open_centre + half_open_up + (block_size_x_right + self.y_size_offset[0])/2 + self.y_centre_offset[0],
                                 self.pos_y + (self.thick/2),
                                 0)

            size_down = mp.Vector3(block_size_x_left + self.y_size_offset[1], self.thick, 0)
            centre_down = mp.Vector3(open_centre - half_open_down - (block_size_x_left + self.y_size_offset[1])/2 + self.y_centre_offset[1],
                                   self.pos_y + (self.thick/2),
                                   0)

        aperture_stop_up = meep_block(size = size_up,
                                        center = centre_up,
                                        material = material,
                                        angle = self.rot_angle,
                                        rot_axis = self.rot_axis)
        
        aperture_stop_down = meep_block(size = size_down,
                                        center = centre_down,
                                        material = material,
                                        angle = self.rot_angle,
                                        rot_axis = self.rot_axis)
        
        print(f'Aperture stop created ({self.orientation}): Up size={size_up}, Down size={size_down}')
        print(f'Centers: Up={centre_up}, Down={centre_down}')

        self.blocks = (aperture_stop_up, aperture_stop_down)

        return aperture_stop_up, aperture_stop_down

    @staticmethod
    def _block_vertices(block):
        '''
        Returns the four corner vertices of a 2D meep block as mp.Vector3,
        anticlockwise in the block's own (possibly rotated) frame:
        (-e1,-e2), (+e1,-e2), (+e1,+e2), (-e1,+e2)
        '''
        half_e1 = block.e1.unit().scale(block.size.x / 2)
        half_e2 = block.e2.unit().scale(block.size.y / 2)

        return [block.center - half_e1 - half_e2,
                block.center + half_e1 - half_e2,
                block.center + half_e1 + half_e2,
                block.center - half_e1 + half_e2]

    def get_vertices(self):
        '''
        Returns the corner vertices of the two aperture stop blocks

        Assembles the aperture stop first if it has not been assembled yet.

        Returns
        -------
        dict
            {'up': [v1, v2, v3, v4], 'down': [v1, v2, v3, v4]} where each
            vertex is an mp.Vector3. For a 'horizontal' aperture stop 'up'
            is the +x (right) block and 'down' the -x (left) one.
        '''
        if self.blocks is None:
            self._build()

        block_up, block_down = self.blocks

        return {'up': self._block_vertices(block_up),
                'down': self._block_vertices(block_down)}

    def print_vertices(self):
        '''
        Prints the corner vertices of the two aperture stop blocks

        Returns
        -------
        dict
            The same dictionary as get_vertices()
        '''
        vertices = self.get_vertices()

        # 'up'/'down' are along y for a vertical stop and along x for a horizontal one
        if self.orientation == 'vertical':
            block_names = {'up': 'UP (+y) block', 'down': 'DOWN (-y) block'}
        else:
            block_names = {'up': 'UP (+x, right) block', 'down': 'DOWN (-x, left) block'}

        corner_names = ['(-e1,-e2)', '(+e1,-e2)', '(+e1,+e2)', '(-e1,+e2)']

        print(f'Aperture stop vertices ({self.orientation}, '
              f'rotation: {self.rot_angle}° about {self.rot_axis}):')
        for key in ('up', 'down'):
            print(f'  {block_names[key]}:')
            for name, vertex in zip(corner_names, vertices[key]):
                print(f'    {name} : x = {vertex.x:.6g}, y = {vertex.y:.6g}, z = {vertex.z:.6g}')

        return vertices

    # Which face of a block each absorber side sits on, in the block's own frame.
    # For a vertical stop e1 is the slab normal (x) and e2 runs along the blade (y);
    # for a horizontal stop the two swap over.
    _ABSORBER_FACES = {
        'vertical':   {'front': '-e1', 'back': '+e1', 'inner_up': '-e2', 'inner_down': '+e2'},
        'horizontal': {'front': '-e2', 'back': '+e2', 'inner_up': '-e1', 'inner_down': '+e1'},
    }

    def _face_geometry(self, block, side, which):
        '''
        Returns the two corner vertices of a face and its outward unit normal

        Parameters
        ----------
        block : mp.Block
            One of the two aperture stop blocks
        side : str
            'front', 'back' or 'inner'
        which : str
            'up' or 'down', i.e. which of the two blocks this is

        Returns
        -------
        tuple
            (vertex_a, vertex_b, outward_normal) as mp.Vector3
        '''
        faces = self._ABSORBER_FACES[self.orientation]
        face = faces[side] if side in ('front', 'back') else faces[f'inner_{which}']

        v = self._block_vertices(block)
        # Edges of the block, keyed by the face they belong to
        edges = {'-e1': (v[0], v[3]), '+e1': (v[1], v[2]),
                 '-e2': (v[0], v[1]), '+e2': (v[3], v[2])}

        axis = block.e1.unit() if face.endswith('e1') else block.e2.unit()
        normal = axis.scale(-1) if face.startswith('-') else axis

        vertex_a, vertex_b = edges[face]

        return vertex_a, vertex_b, normal

    def _flat_absorber(self, vertex_a, vertex_b, normal, offset, layer):
        '''
        Returns a flat absorber slab covering a face, as an mp.Prism

        The slab spans the whole face and sits between `offset` and
        `offset + layer.extent` along the outward normal.
        '''
        inner_a = vertex_a + normal.scale(offset)
        inner_b = vertex_b + normal.scale(offset)
        outer_b = vertex_b + normal.scale(offset + layer.extent)
        outer_a = vertex_a + normal.scale(offset + layer.extent)

        vertices = ensure_ccw_order([inner_a, inner_b, outer_b, outer_a])

        return mp.Prism(vertices=vertices,
                        height=self.absorber_height,
                        axis=mp.Vector3(0, 0, 1),
                        material=layer.get_material())

    def _pyramidal_absorber(self, vertex_a, vertex_b, normal, offset, layer):
        '''
        Returns a row of pyramids standing on a face, as a list of blocks

        Built with PyramidalAbsorbers, which anchors its pyramids to a cell
        edge, so the corresponding edge offset is worked out from the position
        of the face. Substrate and PEC backing are switched off: the aperture
        stop itself is what backs these absorbers.
        '''
        if self.rot_angle != 0:
            raise ValueError("Pyramidal absorber layers need an unrotated aperture stop "
                             f"(rot_angle = {self.rot_angle}). Use shape='flat' instead.")

        # PyramidalAbsorbers measures its pyramids from the cell edge, past the PML
        base_offset = self.mpsat_sim.factor_dpml * self.mpsat_sim.dpml

        for key in ('add_substrate', 'add_pec_backing'):
            if layer.pyramid_kwargs.get(key):
                warnings.warn(f"{key} is not supported for aperture stop absorbers "
                              "and has been switched off: the stop backs the pyramids.")
        kwargs = {key: value for key, value in layer.pyramid_kwargs.items()
                  if key not in ('add_substrate', 'add_pec_backing')}

        cell = self.mpsat_sim.cell

        # Anchor the base of the pyramids on the face, pointing outwards, and
        # spread them over the extent of the face
        if abs(normal.x) > abs(normal.y):
            face_x = vertex_a.x + normal.x * offset
            span = sorted([vertex_a.y, vertex_b.y])
            coverage = {'y_coverage_start': span[0], 'y_coverage_end': span[1]}
            if normal.x > 0:
                edge = 'left'    # Pyramids grow towards +x
                kwargs['x_left_offset'] = face_x + cell.x/2 - base_offset
            else:
                edge = 'right'   # Pyramids grow towards -x
                kwargs['x_right_offset'] = face_x - cell.x/2 + base_offset
        else:
            face_y = vertex_a.y + normal.y * offset
            span = sorted([vertex_a.x, vertex_b.x])
            coverage = {'x_coverage_start': span[0], 'x_coverage_end': span[1]}
            if normal.y > 0:
                edge = 'bottom'  # Pyramids grow towards +y
                kwargs['y_bottom_offset'] = face_y + cell.y/2 - base_offset
            else:
                edge = 'top'     # Pyramids grow towards -y
                kwargs['y_top_offset'] = face_y - cell.y/2 + base_offset

        face_width = span[1] - span[0]
        if face_width <= 0:
            warnings.warn(f"Skipping the pyramidal absorber on the {layer.side} side: "
                          "the face it sits on has zero width.")
            return []

        pyramids = PyramidalAbsorbers(mpsat_sim=self.mpsat_sim,
                                      num_pyramids=layer.num_pyramids,
                                      base_width=face_width / layer.num_pyramids,
                                      top_width=layer.top_width,
                                      height=layer.extent,
                                      n_layers=layer.n_layers,
                                      material=layer.get_material(),
                                      freq=layer.freq,
                                      edges=[edge],
                                      add_substrate=False,
                                      add_pec_backing=False,
                                      name=f'{self.object_type} {layer.side} pyramids',
                                      **coverage,
                                      **kwargs)

        return pyramids.assemble()

    def get_absorbers(self):
        '''
        Returns the absorber layers requested for the aperture stop

        Assembles the aperture stop first if it has not been assembled yet.
        The objects are NOT part of what assemble() returns, so add them to
        the MEEP geometry separately, e.g.
        mpsat_sim.meep_geometry.extend(stop.get_absorbers())

        Returns
        -------
        list
            mp.Prism objects for the flat layers and mp.Block objects for the
            pyramidal ones, in the order the layers were given
        '''
        if not self.absorber_layers:
            return []

        if self.blocks is None:
            self._build()

        if self.rot_angle != 0 and self.rot_axis != 'z':
            warnings.warn(f"The aperture stop is rotated about {self.rot_axis}, so its faces "
                          "leave the xy-plane: the absorber layers will be built from the "
                          "out-of-plane vertices and are unlikely to be what you want.")

        blocks = {'up': self.blocks[0], 'down': self.blocks[1]}
        absorbers = []

        for which, block in blocks.items():
            for side in ('front', 'back', 'inner'):
                layers = self._layers_on(side, which)
                if not layers:
                    continue

                vertex_a, vertex_b, normal = self._face_geometry(block, side, which)

                # Layers stack outwards from the face, in the order given
                offset = 0
                for layer in layers:
                    if layer.shape == 'pyramidal':
                        absorbers.extend(self._pyramidal_absorber(vertex_a, vertex_b,
                                                                  normal, offset, layer))
                    else:
                        absorbers.append(self._flat_absorber(vertex_a, vertex_b,
                                                             normal, offset, layer))
                    offset += layer.extent

        self.absorbers = absorbers
        print(f'Aperture stop absorbers assembled: {len(absorbers)} objects from '
              f'{len(self.absorber_layers)} layer definitions')

        return absorbers

    def _build(self):
        '''
        Builds the block objects for the aperture stop according to the type
        '''
        if self.type == 'square':
            return self.square_aperture()
        else:
            raise ValueError(f'Invalid aperture stop type: {self.type}. Currently only "square" is supported.')

    def assemble(self):
        '''
        Returns the block objects for the aperture stop according to the type

        Returns
        -------
        tuple
            Two block objects (aperture_stop_up, aperture_stop_down)
        '''
        blocks = self._build()

        if self._print_vertices:
            self.print_vertices()

        return blocks

    def assemble_all(self):
        '''
        Returns the aperture stop blocks together with their absorber layers

        Returns
        -------
        tuple
            (aperture_stop_up, aperture_stop_down, absorbers), where absorbers
            is the (possibly empty) list returned by get_absorbers()
        '''
        block_up, block_down = self.assemble()

        return block_up, block_down, self.get_absorbers()


###& DETECTOR CLASS
class Detector():
    '''
    Class defining an image plane/ the location of the detectors
    '''

    def __init__(self,
                 type= None,
                 diameter= None, 
                 pos_x= None, 
                 thickness= None,
                 n_refr = None, 
                 conductivity = None):
        '''
        Defines the attributes of the aperture stop object

        Arguments
        ---------
        type : str, optional
            Type of Detector- block, circular etc (default : None)       
        diameter : float 
            Diameter of the image plane slab
        pos_x : float
            Position of the image plane along x-axis
        thickness : float
            Thickness of image plane slab
        n_refr : float, optional
            Index of refraction of the material 
            if the stop is dielectric
            (default = 1)
        conductivity : float, optional
            Conductivity of the material (default = np.inf)
        '''
        self.object_type = 'detector'

        self.name = type              
        self.diameter = diameter
        self.x = pos_x
        self.thickness = thickness
        self.n_refr = n_refr
        self.conductivity = conductivity

        self.center = [self.x, 0, 0]
        self.size = [self.thickness, self.diameter, 0]

    def center(self):
        return self.center
    
    def size(self):
        return self.size

    def position(self):
        if self.name is not None :
            return self.object_type + 'of type' + self.name + 'at position' + str(self.x)
        else : 
            return 'Image Plane/Detector at position ' + str(self.x)
        

    def block_detector(self):
        '''
        Returns the block object for the image plane/ detector! 
        '''
        
        meep_block_detector = mp.Block(size= mp.Vector3(self.thickness, self.diameter, 0),
                                       center= mp.Vector3(self.x, 0, 0),
                                       material= self.material)
        
        return meep_block_detector
    
    ### ^ Similarly we can add more types of detectors here
    # ^ def circular_detector(self):

    def assemble(self):
        '''
        Returns the block object for the image plane/ detector! 
        '''
        if self.conductivity != np.inf :
            #Defines the material with given properties
            self.material = mp.Medium(epsilon=self.n_refr**2, 
                                      D_conductivity = self.conductivity)
        
        else :
            #If the conductivity is infinite, Meep can define a perfect conductor
            self.material = mp.perfect_electric_conductor
        
        if self.name == 'meep_block':
            detector = self.block_detector()

        ### ^ Similarly we can add more types of detectors here
        # ^ elif self.name == 'circular':

        else:
            raise ValueError('Invalid detector type name')
        
        return detector
    
###& BOUNDARY CLASS CLASS

class Boundary():
    """
    Class defining the boundary conditions of the 2D simulation box
    """
    def __init__(self,
                 type = None,
                 thickness = None,
                 **kwargs):
        """
        Defines the attributes of the boundary object

        Arguments
        ---------
        type : str
            Type of the boundary conditions; e.g., PML (default : None)

        thickness : float
            Thickness of the boundary conditions (default : None)

        **kwargs : dict
            Additional arguments for the meep.Boundary()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#boundary
        """
        self.object_type = 'boundary_layer'

        if type is None:
            warnings.warn("No name given to the boundary object: Taking the default boundary as PML")
            self.name = 'PML'
        else:
            self.name = type

        if thickness is None:
            warnings.warn("No thickness given to the boundary object: Taking the default thickness as 2.0")
            self.thickness = 2.0
        else:
            self.thickness = thickness

        self.additional_args = kwargs

    def description(self):
        return self.object_type + ': ' + self.name + ' with thickness ' + str(self.thickness)
        
    def pml_boundary(self):
        """
        Return PML boundary conditions
        """
        if self.additional_args:
            filtered_kwrg = exf.filter_dict(self.additional_args, mp.Boundary)
            boundary = mp.PML(self.thickness, **filtered_kwrg)
        else:
            boundary = mp.PML(self.thickness)
        
        return boundary
    
    ### ^ Similarly we can add more types of boundaries here
    # ^ def periodic_boundary(self):

    def assemble(self):
        """
        Returns the boundary object according to the user input
        """
        if self.name == 'PML':
            boundary = self.pml_boundary()

        ### ^ Similarly we can add more types of boundaries here
        # ^ elif self.name == 'periodic':
        else:
            raise ValueError('Invalid boundary type name')

        return boundary
        
# * ############################################################################################################
# * ############################################################################################################
# * ############################################################################################################
# * Extracting some classes from the MEEPART package

class Filter(object):
    """
    Class defining the filter object
    """
    def __init__(self,
                 mpsat_sim,
                 name="block",
                 center=None,
                 size=None,
                 material=None,
                 angle=0,
                 rot_axis='x',
                 **kwargs):
        """
        Defines the attributes of the filter object

        Arguments
        ---------
        mpsat_sim : object
            MEEPSAT simulation object
        name : str, optional
            Type of filter (default: 'block')
        center : mp.Vector3
            Center of the filter
        size : mp.Vector3
            Size of the filter
        material : mp.Medium
            Material of the filter
        angle : float, optional
            Angle of rotation in degrees (default: 0)
        rot_axis : str, optional
            Axis of rotation ('x', 'y', or 'z') (default: 'x')
        **kwargs : dict
            Additional arguments for the meep functions
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'Filter'
        
        # Set center and size using your helper functions
        self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
        self.size = set_size(self, size, default_size=mp.Vector3(0, 0, 0))
        
        # Material
        if material is None:
            raise ValueError("Material must be specified for Filter")
        self.material = material
        
        # Rotation parameters
        self.angle = angle
        self.rot_axis = rot_axis
        self.kwargs = kwargs

    def block_filter(self):
        """
        Return the block filter object
        """
        # Use your existing meep_block utility function
        filter_block = meep_block(
            size=self.size,
            center=self.center,
            material=self.material,
            angle=self.angle,
            rot_axis=self.rot_axis,
            **self.kwargs
        )
        return filter_block
    
    def assemble(self):
        """
        Returns the filter object according to the user input
        """
        if self.name == 'block':
            filter_obj = self.block_filter()
        else:
            raise ValueError(f'Invalid filter type: {self.name}')

        return filter_obj
    

class Slab(object):
    """
    Class defining a slab object - a simple geometrical shape used for various optical elements
    """
    def __init__(self,
                 mpsat_sim,
                 name="block",
                 center=None,
                 size=None,
                 material=None,
                 angle=0,
                 rot_axis='x',
                 **kwargs):
        """
        Defines the attributes of the slab object

        Arguments
        ---------
        mpsat_sim : object
            MEEPSAT simulation object
        name : str, optional
            Type of slab (default: 'block')
        center : mp.Vector3
            Center of the slab
        size : mp.Vector3
            Size of the slab
        material : mp.Medium
            Material of the slab
        angle : float, optional
            Angle of rotation in degrees (default: 0)
        rot_axis : str, optional
            Axis of rotation ('x', 'y', or 'z') (default: 'x')
        **kwargs : dict
            Additional arguments for the meep functions
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'Slab'
        
        # Set center and size using helper functions
        self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
        self.size = set_size(self, size, default_size=mp.Vector3(0, 0, 0))
        
        # Material
        if material is None:
            raise ValueError("Material must be specified for Slab")
        self.material = material
        
        # Rotation parameters
        self.angle = angle
        self.rot_axis = rot_axis
        self.kwargs = kwargs

    def block_slab(self):
        """
        Return the block slab object
        """
        # Use the existing meep_block utility function
        slab_block = meep_block(
            size=self.size,
            center=self.center,
            material=self.material,
            angle=self.angle,
            rot_axis=self.rot_axis,
            **self.kwargs
        )
        return slab_block
    
    def assemble(self):
        """
        Returns the slab object according to the user input
        """
        if self.name == 'block':
            slab_obj = self.block_slab()
        else:
            raise ValueError(f'Invalid slab type: {self.name}')

        return slab_obj
    
#!= Modules for MEEP monitor


class VolumeMonitor():
    """
    Class defining a volume monitor for collecting data from a specific region
    """
    def __init__(self,
                 mpsat_sim,
                 name="volume_monitor",
                 center=None,
                 size=None,
                 components=None,
                 data_required=None,
                 **kwargs):
        """
        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object
            
        name : str, optional
            Name of the monitor (default: 'volume_monitor')
            
        center : list or mp.Vector3
            Center of the monitor volume [x, y, z]
            
        size : list or mp.Vector3
            Size of the monitor volume [x, y, z]
            
        components : list, optional
            Field components to monitor (default: None, monitors all components)
            
        data_required : dict, optional
            Dictionary specifying what data to collect and when:
            {
                'at_every_timestep': int,  # Collect data every N timesteps
                'at_every': list,          # List of data types to collect at each sampling
                'at_end': list             # List of data types to collect at simulation end
            }
            
        **kwargs : dict
            Additional arguments for customization
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'VolumeMonitor'
        
        # Set center - handle both list and Vector3
        if isinstance(center, list):
            self.center = mp.Vector3(center[0], center[1], center[2])
        else:
            self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
            
        # Set size - handle both list and Vector3
        if isinstance(size, list):
            self.size = mp.Vector3(size[0], size[1], size[2])
        else:
            self.size = set_size(self, size, default_size=mp.Vector3(1, 1, 0))
            
        # Components to monitor
        self.components = components
        
        # Data collection requirements
        self.data_required = data_required if data_required else {
            'at_every_timestep': 10,
            'at_every': [],
            'at_end': []
        }
        
        # Additional args
        self.kwargs = kwargs
        
        print(f"Volume monitor '{self.name}' created at {self.center} with size {self.size}")
        print(f"Data collection: every {self.data_required.get('at_every_timestep')} timesteps")
        print(f"Collecting: {self.data_required.get('at_every')} during simulation")
        print(f"Collecting: {self.data_required.get('at_end')} at end")

    def assemble(self):
        """
        Return the assembled monitor object
        """
        # Create a volume object
        volume = mp.Volume(center=self.center, size=self.size)
        print(f"Volume monitor assembled: {volume}")
        return volume
    

# Add after the VolumeMonitor class

class FluxMonitor():
    """
    Class defining a flux monitor for calculating power transmission and reflection
    """
    def __init__(self,
                 mpsat_sim,
                 name="flux_monitor",
                 center=None,
                 size=None,
                 direction=mp.X,
                 freq_min=None,
                 freq_max=None,
                 nfreq=100,
                 monitor_type="transmission",  # "incident", "reflection", "transmission"
                 use_flux_file=None,
                 **kwargs):
        """
        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object
            
        name : str, optional
            Name of the monitor (default: 'flux_monitor')
            
        center : list or mp.Vector3
            Center of the flux monitor plane
            
        size : list or mp.Vector3
            Size of the flux monitor plane
            
        direction : mp.direction constant
            Direction normal to the flux plane (mp.X, mp.Y, or mp.Z)
            
        freq_min : float
            Minimum frequency to monitor
            
        freq_max : float
            Maximum frequency to monitor
            
        nfreq : int
            Number of frequency points
            
        monitor_type : str
            Type of flux monitor: "incident", "reflection", or "transmission"

        use_flux_file : str, optional
            Path to saved flux data for normalization
        
        **kwargs : dict
            Additional arguments for mp.FluxRegion
        """
        self.use_flux_file = use_flux_file 
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'FluxMonitor'
        self.monitor_type = monitor_type
        
        # Set center - handle both list and Vector3
        if isinstance(center, list):
            self.center = mp.Vector3(center[0], center[1], center[2])
        else:
            self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
            
        # Set size - handle both list and Vector3
        if isinstance(size, list):
            self.size = mp.Vector3(size[0], size[1], size[2])
        else:
            self.size = set_size(self, size, default_size=mp.Vector3(0, self.mpsat_sim.cell_size[1], 0))
            
        # Frequency parameters
        self.freq_min = freq_min if freq_min is not None else self.mpsat_sim.freq * 0.8
        self.freq_max = freq_max if freq_max is not None else self.mpsat_sim.freq * 1.2
        self.nfreq = nfreq
        
        # Direction
        self.direction = direction
        
        # Additional args
        self.kwargs = kwargs
        
        print(f"Flux monitor '{self.name}' created at {self.center} with size {self.size}")
        print(f"Frequency range: {self.freq_min} to {self.freq_max} with {self.nfreq} points")
        print(f"Monitor type: {self.monitor_type}")

    def assemble(self):
        """
        Return the assembled flux monitor object
        """
        # Create a flux region
        flux_region = mp.FluxRegion(
            center=self.center,
            size=self.size,
            direction=self.direction,
            **self.kwargs
        )
        print(f"Flux monitor {self.monitor_type} assembled: {flux_region}")
        return flux_region
    

#~ NEW ABSORBER CLASS SUPPORTING DIFFERENT TYPES OF ABSORBERS

class Absorbers:
    def __init__(self,
                 p,
                 taper_type,
                 grid_size_sx,
                 grid_size_sy,
                 resolution,
                 center_x_mm,
                 center_y_mm,
                 eps_array,
                 geometry_objects,
                 z0,
                 z1,
                 orientation,
                 angle_axis,
                 h= None,
                 p_h_ratio= None,
                 # Substrate parameters
                 substrate_thickness=None,
                 substrate_material=None,
                 add_substrate = False,
                 mesh_filter_option='min',
                 # Absorber Material Properties
                 epsilon_r=None,
                 epsilon_i=None,
                 material=None,
                 freq= None,
                 material_type='narrow_bandwidth_absorption',
                 # Two points for placing N-number of absorbers between those two points in a straight line
                 start_point = None,
                 end_point = None,
                 overall_factor= 0.95,
                 # Plotting parameters
                 plot_alpha=False,
                 plot_profile=False,
                 plot_mesh= False,
                 savepath=None
                 ):
        
    
        # Dimensions of the absorbers
        self.p = p # base
        
        if p_h_ratio:
            self.p_h_ratio = p_h_ratio # p/h ratio
            h = p*p_h_ratio # height
            self.h = h
        else:
            self.h = h
        
        self.p_h_ratio = p_h_ratio # p/h ratio
        self.l_array = np.linspace(0,h, 1000) # l goes from 0 to h
        self.theta = np.arctan(self.p/(2*self.h)) # angle w.r.t base
        self.orientation = orientation
        self.angle_axis = angle_axis

        # Absorber material and impedance properties
        self.epsilon_r = epsilon_r
        self.epsilon_i = epsilon_i
        
        if material:
            self.material = material
            # Update epsilon_r for create_absorber_from_profile function later
            self.epsilon_r = material.epsilon
            
        if epsilon_r and epsilon_i:
            if material_type == 'narrow_bandwidth_absorption':
                if freq is None:
                    raise ValueError("Please give the frequency value for the narrow_bandwidth_absorption material.")
                print("Creating narrow bandwidth absorption material using the following instructions: https://meep.readthedocs.io/en/latest/Materials/#conductivity-and-complex")
                D_conductivity = (epsilon_i * 2 * math.pi * freq) / epsilon_r
                self.material = mp.Medium(epsilon=epsilon_r, D_conductivity=D_conductivity)    
        elif epsilon_r:
            self.material = mp.Medium(epsilon=epsilon_r)
        else:
            raise ValueError("Invalid material properties")
                
        if z0 is None:
            self.z0 = 1
        else:
            self.z0 = z0

        if z1 is None:
            self.z1 = 1/math.sqrt(epsilon_r)
        else:
            self.z1 = z1

        # Type of the taper
        self.taper_type = taper_type # ['Pyramidal', 'linear', 'exponential']
        
        # Simulation box parameters
        self.grid_size_sx = grid_size_sx
        self.grid_size_sy = grid_size_sy
        self.resolution = resolution
        self.center_x_mm = center_x_mm
        self.center_y_mm = center_y_mm
        
        # Epsilon Map and Geometry object array
        self.eps_array = eps_array
        self.geometry_objects = geometry_objects
        
        # Substrate parameters
        self.add_substrate = add_substrate
        self.substrate_thickness = substrate_thickness
        self.substrate_material = substrate_material
        # Default substrate material
        self.default_substrate_material = self.material

        # Triangular mesh option
        self.mesh_filter_option = mesh_filter_option

        # Placing N number of absorbers between two points
        # if start_point is not None and end_point is not None:
        self.start_point = start_point
        self.end_point = end_point
        self.overall_factor = overall_factor
        
        # Plotting options
        self.plot_alpha = plot_alpha
        self.plot_profile = plot_profile
        self.plot_mesh = plot_mesh
        
        
        # Save path
        if savepath:
            self.savepath = savepath
            os.makedirs(self.savepath, exist_ok=True)
        else:
            # Save in the current director
            self.savepath = './'

    def assemble_single_absorber(self):
        # Calculate the filling factor (alpha_array) 
        if self.taper_type == "Pyramidal":
            self.alpha_array = self.alpha_pyramidal(self.p, self.theta, self.l_array)
        elif self.taper_type == "Exponential":
            self.alpha_array = self.alpha_exponential(self.z0, self.z1, self.h, self.epsilon_r, self.l_array)
        elif self.taper_type == "Linear":
            self.alpha_array = self.alpha_linear(self.z0, self.z1, self.h, self.epsilon_r, self.l_array)
        else:
            raise ValueError("Invalid taper type")
        
        self.w_array = self.p * np.sqrt(self.alpha_array)

        if self.plot_alpha:
            self._plot_alpha()
        if self.plot_profile:
            self._plot_profile()

        # Create the absorber filled with triangular mesh
        self.absorber, self.tri = self.create_absorber_from_profile(
            grid_size_sx=self.grid_size_sx,
            grid_size_sy=self.grid_size_sy,
            eps_array=self.eps_array,
            center_x_mm=self.center_x_mm,
            center_y_mm=self.center_y_mm,
            pyramid_height=self.h,
            base_width=self.p,
            alpha_profile=self.alpha_array,
            orientation=self.orientation,
            angle_axis=self.angle_axis,
            add_substrate=self.add_substrate,
            substrate_thickness=self.substrate_thickness,
            substrate_material=self.substrate_material,
            material_value=self.epsilon_r,
            resolution=self.resolution
        )

        self.absorber_prisms = mesh.convert_triangles_to_prisms(gridx_size_mm=self.grid_size_sx,
                                                                gridy_size_mm=self.grid_size_sy,
                                                                tri=self.tri,
                                                                material=self.material,
                                                                # resolution=self.resolution,
                                                                thickness=1.0 # It won't affect in 2D
        )

        # Check if absorber_prisms is a list and extend, otherwise append
        if isinstance(self.absorber_prisms, list):
            self.geometry_objects.extend(self.absorber_prisms)
        else:
            self.geometry_objects.append(self.absorber_prisms)

        
        return self.geometry_objects
    
    def assemble(self):
        if self.start_point is None and self.end_point is None:
            return self.assemble_single_absorber()
        else:
            return self.place_absorbers_between(self.start_point, self.end_point)

    def place_absorbers_between(self, start_point, end_point):
        print("Placing absorbers between points:", start_point, "and", end_point)

        # Calculate the direction vector
        direction = np.array(end_point) - np.array(start_point)
        direction_normalized = direction / np.linalg.norm(direction)
        distance = np.linalg.norm(direction)

        # Adjust start/end points to account for absorber width
        start_point = np.array(start_point) + (self.p / 2) * direction_normalized
        end_point = np.array(end_point) - (self.p / 2) * direction_normalized
        distance = np.linalg.norm(end_point - start_point)

        # Calculate number of absorbers with overlap factor to eliminate gaps
        # Use slightly smaller spacing to ensure overlap
        overlap_factor = self.overall_factor  # 95% of p spacing = 5% overlap
        num_absorbers = int(np.ceil(distance / (self.p * overlap_factor)))

        # Calculate perpendicular direction
        perp_direction = np.array([-direction_normalized[1], direction_normalized[0]])
        angle_rad = np.arctan2(perp_direction[1], perp_direction[0])
        angle_deg = np.rad2deg(angle_rad)

        # Store original orientation and angle_axis
        original_orientation = self.orientation
        original_angle_axis = self.angle_axis

        # Only autocalculate and override if a string like '+y' or '-y' was not provided
        if not isinstance(original_orientation, str):
            self.orientation = angle_deg
            self.angle_axis = "x"

        print(f"Line angle: {np.rad2deg(np.arctan2(direction_normalized[1], direction_normalized[0])):.2f}°")
        print(f"Calculated perpendicular: {angle_deg:.2f}°")
        print(f"Active Absorber orientation: {self.orientation}")
        print(f"Direction: {direction_normalized}")
        print(f"Perpendicular: {perp_direction}")

        # Place absorbers with overlap to eliminate gaps
        absorber_centers = []
        spacing = distance / num_absorbers  # Evenly distribute absorbers
        
        for i in range(num_absorbers):
            center = start_point + (i + 0.5) * spacing * direction_normalized
            # Check if absorber center is within grid bounds
            if (-self.grid_size_sx/2 <= center[0] <= self.grid_size_sx/2 and 
                -self.grid_size_sy/2 <= center[1] <= self.grid_size_sy/2):
                absorber_centers.append(center)

        print(f"Calculated {len(absorber_centers)} absorber centers with spacing: {spacing:.2f}mm")
        print("Absorber centers:", absorber_centers)

        # Create absorbers at each center position
        for center in absorber_centers:
            self.center_x_mm = center[0]
            self.center_y_mm = center[1]
            self.assemble_single_absorber()

        # Restore original orientation
        self.orientation = original_orientation
        self.angle_axis = original_angle_axis

        return self.geometry_objects

    def _plot_alpha(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.l_array/self.h, self.alpha_array, label = self.taper_type, color = 'blue')
        plt.title('Alpha Values for Different Absorber profiles')
        plt.xlabel('Normalized Height (l/h)')
        plt.ylabel('Alpha (Filling factor)')
        plt.grid()
        plt.legend()
        plt.savefig(self.savepath + 'alpha_values_linear_absorber.png')
        plt.close()
            

    def _plot_profile(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.w_array, self.l_array)
        plt.xlabel("Width (w)")
        plt.ylabel("Height (l)")
        plt.title("Linear profile")
        plt.savefig(self.savepath + 'linear_profile.png')
        plt.close()
        

    # Pyramid Taper
    def alpha_pyramidal(self, p, theta, l):
        return ((p - 2*np.tan(theta)*l)**2)/p**2

    # Exponential Taper
    def alpha_exponential(self, z0, z1, h, e_r, l):
        """Exponential impedance profile from z0 to z1 over height h"""
        b = z0 / z1
        a = (1/h) * np.log(b)
        epsilon_l = b * np.exp(-2 * a * l)
        alpha = ((epsilon_l - 1) / (e_r - 1))
        
        # Normalize alpha to [0, 1] range
        # Benefits:
            # The bottom (highest alpha) reaches 1.0
            # The top (lowest alpha) reaches ~0
            # Values are spread across the full [0, 1] range
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)
        if alpha_max > alpha_min:
            alpha = (alpha - alpha_min) / (alpha_max - alpha_min)
        
        alpha = np.clip(alpha, 1e-6, 1.0)
        
        return alpha

    # Linear Taper
    def alpha_linear(self, z0, z1, h, e_r, l):
        m = (z1-z0)/h
        epsilon_l = (z0/(m*l + z0))**2
        alpha = (epsilon_l - 1) / (e_r - 1)
        
        # Reverse the profile so it decreases with height
        # This helps in inverting the pyramids
        alpha = alpha[::-1]

        return alpha



    def create_absorber_from_profile(self,
                                     grid_size_sx, 
                                    grid_size_sy, 
                                    eps_array,
                                    center_x_mm,
                                    center_y_mm,
                                    pyramid_height, 
                                    base_width,
                                    alpha_profile,
                                    orientation="+y",
                                    angle_axis = "x",
                                    add_substrate=True,
                                    substrate_thickness=1.0,
                                    substrate_material = None,
                                    material_value=1.0,
                                    resolution=1.0):
        """
        Create a pyramidal absorber using an impedance profile.
        
        Parameters:
        -----------
        alpha_profile : ndarray
            Filling factor profile (0 to 1) as function of height
        """
        absorber_array = eps_array.copy()
        scaled_pyramid_height = int(pyramid_height * resolution)
        scaled_base_width = int(base_width * resolution)
        scaled_grid_size_sx = int(grid_size_sx * resolution)
        scaled_grid_size_sy = int(grid_size_sy * resolution)

        # Convert to pixel coordinates from centered coordinates system to origin coordinate system
        center_x = int((center_x_mm + grid_size_sx/2) * resolution)
        center_y = int((center_y_mm + grid_size_sy/2) * resolution)

        if add_substrate:
            # Convert single values to lists for uniform handling
            if substrate_thickness is not None and not isinstance(substrate_thickness, (list, tuple)):
                substrate_thickness = [substrate_thickness]
            if substrate_material is not None and not isinstance(substrate_material, (list, tuple)):
                substrate_material = [substrate_material]
            
            # Validate that lists have same length
            if substrate_thickness and substrate_material:
                if len(substrate_thickness) != len(substrate_material):
                    raise ValueError("substrate_thickness and substrate_material lists must have same length")
            
            # Create empty lists to store the values of the centre, size, angle, material of the different substrates
            centre_x_substrate = []
            centre_y_substrate = []
            size_x_substrate = []
            size_y_substrate = []
            angle_substrate = []

        for layer in range(scaled_pyramid_height):
            # Get alpha value from profile for this layer
            # Calculating the index in the `alpha_profile` array that
            # corresponds to the current layer of the pyramidal absorber being created.
            profile_idx = min(int(layer / scaled_pyramid_height * len(alpha_profile)), len(alpha_profile) - 1)
            alpha = alpha_profile[profile_idx]
            
            # Width varies based on filling factor
            current_width = int(scaled_base_width * np.sqrt(alpha))
            
            # Layer count variable
            layer_count = 0

            
            if orientation == "+y":
                y_pos = center_y + layer
                if 0 <= y_pos < scaled_grid_size_sy:
                    for x in range(max(0, center_x - current_width//2), 
                                min(scaled_grid_size_sx, center_x + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        # absorber_array[y_pos, x] = material_value 
                        absorber_array[x, y_pos] = material_value 
                        
                # Add substrate
                if layer_count == 0:   
                    if add_substrate:
                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,                                                                                                                                                 center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)
                
                layer_count += 1
                
            elif orientation == "-y":
                y_pos = center_y - layer
                if 0 <= y_pos < scaled_grid_size_sy:
                    for x in range(max(0, center_x - current_width//2), 
                                min(scaled_grid_size_sx, center_x + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        # absorber_array[y_pos, x] = material_value
                        absorber_array[x, y_pos] = material_value
                        
                # Add substrate
                if layer_count == 0:   
                    if add_substrate:
                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                    center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)
                
                layer_count += 1
                

            elif orientation == "+x":
                x_pos = center_x + layer
                if 0 <= x_pos < scaled_grid_size_sx:
                    for y in range(max(0, center_y - current_width//2), 
                                min(scaled_grid_size_sy, center_y + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        # absorber_array[y, x_pos] = material_value 
                        absorber_array[x_pos, y] = material_value 

                # Add substrate                           
                if layer_count == 0:   
                    if add_substrate:
                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                    center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)                    

                layer_count += 1
                
            elif orientation == "-x":
                x_pos = center_x - layer
                if 0 <= x_pos < scaled_grid_size_sx:
                    for y in range(max(0, center_y - current_width//2), 
                                min(scaled_grid_size_sy, center_y + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        # absorber_array[y, x_pos] = material_value
                        absorber_array[x_pos, y] = material_value

                # Add substrate       
                if layer_count == 0:   
                    if add_substrate:

                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                    center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)

                layer_count += 1
                
            elif isinstance(orientation, float):
                orientation_rad = np.radians(orientation)
                if angle_axis == "x":
                    # Center position along the angled axis
                    x_pos = int(center_x + layer * np.cos(orientation_rad))
                    y_pos = int(center_y + layer * np.sin(orientation_rad))
                    
                    # Direction perpendicular to the angle (for width)
                    perp_x = -np.sin(orientation_rad)
                    perp_y = np.cos(orientation_rad)
                    
                    # Draw the width perpendicular to the angle direction
                    for w in range(-current_width//2, current_width//2 + 1):
                        x_line = int(x_pos + w * perp_x)
                        y_line = int(y_pos + w * perp_y)
                        if 0 <= x_line < scaled_grid_size_sx and 0 <= y_line < scaled_grid_size_sy:
                            # absorber_array[y_line, x_line] = material_value
                            absorber_array[x_line, y_line] = material_value

                    # Add substrate
                    if layer_count == 0:   
                        if add_substrate:
                            centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                        center_x=center_x,
                                                                                                                                                        center_y=center_y,
                                                                                                                                                        substrate_thickness=substrate_thickness,
                                                                                                                                                        substrate_material=substrate_material,
                                                                                                                                                        scaled_base_width=scaled_base_width,
                                                                                                                                                        resolution=resolution,
                                                                                                                                                        angle_axis=angle_axis)
                                
                elif angle_axis == "y":
                    # TODO: FIX THE BUG HERE!!
                    # # Center position along the angled axis
                    # x_pos = int(center_x + layer * np.sin(orientation_rad))
                    # y_pos = int(center_y + layer * np.cos(orientation_rad))
                    
                    # # Direction perpendicular to the angle (for width)
                    # perp_x = np.cos(orientation_rad)
                    # perp_y = -np.sin(orientation_rad)
                    
                    # # Draw the width perpendicular to the angle direction
                    # for w in range(-current_width//2, current_width//2 + 1):
                    #     x_line = int(x_pos + w * perp_x)
                    #     y_line = int(y_pos + w * perp_y)
                    #     if 0 <= x_line < scaled_grid_size_sx and 0 <= y_line < scaled_grid_size_sy:
                    #         absorber_array[y_line, x_line] = material_value 

                    # if layer_count == 0:   
                    #     if add_substrate:
                    #         centre_x_substrate = center_x - (substrate_thickness/2)*resolution*np.cos(orientation_rad)
                    #         centre_y_substrate = center_y - (substrate_thickness/2)*resolution*np.sin(orientation_rad)
                    #         size_x_substrate = substrate_thickness * resolution
                    #         size_y_substrate = scaled_base_width
                    #         angle_substrate = orientation  
                    raise Warning("Y-axis orientation is not yet implemented yet!!")

                layer_count += 1
                
            else:
                raise ValueError("Invalid orientation type")
            
        if add_substrate:
            import meepsat.meep_geometry as comp_meep
            for i in range(len(substrate_thickness)):
                # Convert substrate dimensions from pixels to mm
                size_x_substrate_mm = size_x_substrate[i] / resolution
                size_y_substrate_mm = size_y_substrate[i] / resolution
                substrate_thickness_mm = substrate_thickness[i]
                
                # Convert center from pixels to mm
                centre_x_substrate_mm = centre_x_substrate[i] / resolution - grid_size_sx/2
                centre_y_substrate_mm = centre_y_substrate[i] / resolution - grid_size_sy/2
                
                print(f"Substrate size: ({size_x_substrate_mm:.2f}, {size_y_substrate_mm:.2f}) mm")
                print(f"Substrate center: ({centre_x_substrate_mm:.2f}, {centre_y_substrate_mm:.2f}) mm")
                print(f"Substrate angle: {angle_substrate[i]}")
                print(f"Substrate thickness: {substrate_thickness_mm:.2f} mm")
                
                substrate = comp_meep.meep_block(
                    size=mp.Vector3(size_x_substrate_mm, 
                                size_y_substrate_mm, 
                                0),
                    center=mp.Vector3(centre_x_substrate_mm,
                                    centre_y_substrate_mm,
                                    0),
                    angle=angle_substrate[i],
                    material=substrate_material[i] if substrate_material[i] is not None else self.default_substrate_material
                )
                self.geometry_objects.append(substrate)
        
        # TRIANGULAR MESHGRID inside the absorber
        tri = mesh._create_triangular_mesh(epsilon_array= absorber_array.T,
                                    epsilon_val= material_value,
                                    grid_size_sx= grid_size_sx,
                                    grid_size_sy= grid_size_sy,
                                    resolution= resolution,
                                    filter_option="min",
                                    plot= self.plot_mesh,
                                    figname= self.savepath + 'absorber_triangular_mesh.png')

        return absorber_array, tri


    def _calculate_substrate_positions(self, orientation, substrate_thickness, 
                                        substrate_material, scaled_base_width,
                                        center_x, center_y, resolution, angle_axis='x'):

        """
        Calculate the positions and dimensions of substrate layers.
        This function computes the center coordinates, sizes, and rotation angles
        for substrate layers based on the specified orientation and layer configuration.
        Parameters
        ----------
        orientation : str or float
            The orientation of the substrate layers. Can be one of "+x", "-x", "+y", "-y"
            for axis-aligned orientations, or a float representing the angle in degrees
            for rotated orientations.
        substrate_thickness : list of float
            Thickness values for each substrate layer in simulation units.
        substrate_material : str
            The material type of the substrate (used for material properties).
        scaled_base_width : float
            The width of the substrate base in simulation units.
        center_x : float
            The x-coordinate of the substrate center in simulation units.
        center_y : float
            The y-coordinate of the substrate center in simulation units.
        resolution : float
            The resolution factor for converting thickness values to spatial dimensions.
        angle_axis : str, optional
            The axis around which rotation is applied (default: 'x').
        Returns
        -------
        centre_x_substrate : list of float
            X-coordinates of the center of each substrate layer.
        centre_y_substrate : list of float
            Y-coordinates of the center of each substrate layer.
        size_x_substrate : list of float
            X-dimension sizes of each substrate layer.
        size_y_substrate : list of float
            Y-dimension sizes of each substrate layer.
        angle_substrate : list of float
            Rotation angles (in degrees) of each substrate layer.
        Notes
        -----
        The function iteratively stacks substrate layers, calculating positions based
        on cumulative thicknesses. For the first layer (i=0), positioning is relative
        to the initial center point. For subsequent layers (i>0), positioning is
        relative to the previous layer's center position.
        """

        centre_x_substrate = []
        centre_y_substrate = []
        size_x_substrate = []
        size_y_substrate = []
        angle_substrate = []
        
        for i in range(len(substrate_thickness)):
            if i == 0:
                if orientation == "+y":
                    cx, cy = center_x, center_y - (substrate_thickness[i]/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "-y":
                    cx, cy = center_x, center_y + (substrate_thickness[i]/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "-x":
                    cx, cy = center_x + (substrate_thickness[i]/2)*resolution, center_y
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                elif orientation == "+x":
                    cx, cy = center_x - (substrate_thickness[i]/2)*resolution, center_y
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                    
                elif isinstance(orientation, float):
                    orientation_rad = np.radians(orientation)
                    cx = center_x - (substrate_thickness[0]/2)*resolution*np.cos(orientation_rad)
                    cy = center_y - (substrate_thickness[0]/2)*resolution*np.sin(orientation_rad)
                    sx = substrate_thickness[i] * resolution
                    sy = scaled_base_width
                    angle0 = orientation
            else:
                if orientation == "-y":
                    cx, cy = centre_x_substrate[i-1], centre_y_substrate[i-1] + ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "+y":
                    cx, cy = centre_x_substrate[i-1], centre_y_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "-x":
                    cx, cy = centre_x_substrate[i-1] + ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution, centre_y_substrate[i-1]
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                elif orientation == "+x":
                    cx, cy = centre_x_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution, centre_y_substrate[i-1]
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                elif isinstance(orientation, float):
                    orientation_rad = np.radians(orientation)
                    cx = centre_x_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution*np.cos(orientation_rad)
                    cy = centre_y_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution*np.sin(orientation_rad)
                    sx = substrate_thickness[i] * resolution
                    sy = scaled_base_width
                    angle0 = orientation

            centre_x_substrate.append(cx)
            centre_y_substrate.append(cy)
            size_x_substrate.append(sx)
            size_y_substrate.append(sy)
            angle_substrate.append(angle0)

        return centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate

# # ~ FOREBAFFLE CLASS

logger = logging.getLogger(__name__)
# * ############################################################################################################
# * BASE CLASSES - Define abstract base class first
# * ############################################################################################################

class ForebaffleComponent(ABC):
    """Abstract base class for forebaffle components."""
    
    @abstractmethod
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Return MEEP geometric objects for this component."""
        pass
    
    # @abstractmethod
    # def get_eps_map(self, parent_forebaffle) -> np.ndarray:
    #     """Return the epsilon map for this component."""
    #     pass


# * ############################################################################################################
# * DATACLASSES - Configuration classes
# * ############################################################################################################

# Absorbers over Forebaffle sides - base, height, hypotenuse 
@dataclass
class AbsorberLayer:
    """Configuration for an absorber layer on a forebaffle side."""
    side: Literal['base', 'height', 'hypotenuse']
    thickness: float
    material: mp.Medium = None
    epsilon_real: float = 1.0
    epsilon_imag: float = 0.0
    
    def get_material(self, freq: float = 1/3) -> mp.Medium:
        """Get the material with absorption properties."""
        if self.material is not None:
            return self.material
        return mp.Medium(epsilon=self.epsilon_real, 
                         D_conductivity=self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real)


class AbsorberComponent(ForebaffleComponent):
    """Component that adds absorber layers to forebaffle sides."""
    
    def __init__(self, absorber_layers: List[AbsorberLayer]):
        self.absorber_layers = absorber_layers
    
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Generate absorber layer geometries based on parent forebaffle."""
        geometries = []
        v1, v2, v3 = parent_forebaffle.calculate_vertices()
        
        for layer in self.absorber_layers:
            if layer.side == 'base':
                geom = self._create_base_absorber(v1, v2, layer, parent_forebaffle)
            elif layer.side == 'height':
                geom = self._create_height_absorber(v2, v3, layer, parent_forebaffle)
            elif layer.side == 'hypotenuse':
                geom = self._create_hypotenuse_absorber(v1, v3, v2, layer, parent_forebaffle)
            
            if geom:
                geometries.append(geom)
        
        return geometries
    
    def _ensure_ccw_order(self, vertices):
        """
        Ensure vertices are in counter-clockwise order using the shoelace formula.
        
        Parameters
        ----------
        vertices : list of mp.Vector3
            List of vertices to check
            
        Returns
        -------
        list of mp.Vector3
            Vertices in counter-clockwise order
        """
        return ensure_ccw_order(vertices)
    
    def _create_base_absorber(self, v1, v2, layer, parent):
        """Create absorber along the base edge (v1-v2)."""
        # Base is horizontal in most cases
        # Offset perpendicular to the base, outward from the triangle
        
        # Determine outward direction
        # For quadrant 1 (0-90°) and 4 (270-360°), offset downward
        # For quadrant 2 (90-180°) and 3 (180-270°), offset downward still works
        angle = parent.angle_degrees
        
        if 0 <= angle < 180:
            # Triangle is above base, offset downward
            offset_y = -layer.thickness
            offset_x = 0
        else:
            # Triangle is below base, offset upward
            offset_y = layer.thickness
            offset_x = 0
        
        # Create vertices - order matters for MEEP!
        vertices = [
            v1,  # Original edge start
            v2,  # Original edge end
            mp.Vector3(v2.x + offset_x, v2.y + offset_y),  # Offset edge end
            mp.Vector3(v1.x + offset_x, v1.y + offset_y),  # Offset edge start
        ]
        
        # Ensure counter-clockwise ordering
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )
    
    def _create_height_absorber(self, v2, v3, layer, parent):
        """Create absorber along the height edge (v2-v3)."""
        # Height is vertical in most cases
        angle = parent.angle_degrees
        
        if 0 <= angle < 90 or 270 <= angle < 360:
            # Triangle is to the left of height, offset to the right
            offset_x = layer.thickness
            offset_y = 0
        else:
            # Triangle is to the right of height, offset to the left
            offset_x = -layer.thickness
            offset_y = 0
        
        vertices = [
            v2,  # Original edge start
            v3,  # Original edge end
            mp.Vector3(v3.x + offset_x, v3.y + offset_y),  # Offset edge end
            mp.Vector3(v2.x + offset_x, v2.y + offset_y),  # Offset edge start
        ]
        
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )
    
    def _create_hypotenuse_absorber(self, v1, v3, v2, layer, parent):
        """
        Create absorber along the hypotenuse edge (v1-v3).
        
        Parameters
        ----------
        v1, v3 : mp.Vector3
            Endpoints of the hypotenuse
        v2 : mp.Vector3
            The right-angle vertex (used to determine outward direction)
        """
        # Calculate perpendicular direction
        dx = v3.x - v1.x
        dy = v3.y - v1.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            logger.warning("Zero-length hypotenuse detected")
            return None
        
        # Two possible perpendicular directions (90° rotation)
        perp_x1 = -dy / length
        perp_y1 = dx / length
        
        # Check which direction points away from v2
        # Vector from midpoint of hypotenuse to v2
        mid_x = (v1.x + v3.x) / 2
        mid_y = (v1.y + v3.y) / 2
        to_v2_x = v2.x - mid_x
        to_v2_y = v2.y - mid_y
        
        # Dot product tells us if perpendicular points toward or away from v2
        dot = perp_x1 * to_v2_x + perp_y1 * to_v2_y
        
        # If dot product is positive, flip the direction
        if dot > 0:
            perp_x1 = -perp_x1
            perp_y1 = -perp_y1
        
        # Apply thickness
        offset_x = perp_x1 * layer.thickness
        offset_y = perp_y1 * layer.thickness
        
        vertices = [
            v1,  # Original edge start
            v3,  # Original edge end
            mp.Vector3(v3.x + offset_x, v3.y + offset_y),  # Offset edge end
            mp.Vector3(v1.x + offset_x, v1.y + offset_y),  # Offset edge start
        ]
        
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )

#--- FLAIRING ON THE FOREBAFFLE TIP ---#
@dataclass
class FlareConfig:
    """Configuration for a flare extending from a vertex."""
    # Type of flaring: 
    # 1. 'linear' - straight line by using mp.Block 
    # 2. 'spline' - spline function pointing outwards
    flaring_type: str # Which type of flaring material
    
    # The below describes the parameters for each flaring type
    # 1. linear
    linear: Dict[str, Any] = field(default_factory=lambda: {
        "length": 1.0,
        "thickness": 1.0,
        "theta2_axis": 'x'
    })
    # 2. spline
    # spline: Dict[str, Any] = field(default_factory=lambda: {
    #     ""
    # })

    material: mp.Medium = None # Meep Material
    epsilon_real: float = 1.0 # Real permittivity
    epsilon_imag: float = 0.0 # Imaginary permittivity
    which_vertex: str = 'v3' # From which vertex of the baffle, the user wants to extend the flaring structure (default from the v3)
        

    def get_material(self, freq: float = 1/3) -> mp.Medium:
        """Get the material with absorption properties."""
        if self.material is not None:
            return self.material
        return mp.Medium(epsilon=self.epsilon_real, 
                         D_conductivity=self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real)

class FlairComponent(ForebaffleComponent):
    """Component representing the flaring structure on the forebaffle tip."""
    def __init__(self, flairs: List[FlareConfig]):
        self.flairs = flairs
        
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Generate flair geometries based on parent forebaffle."""
        geometries = self.create_flairs(parent_forebaffle)
        return geometries
    
    def get_eps_map(self, parent_forebaffle) -> np.ndarray:
        """Return the epsilon map (flairs don't modify it directly)."""
        return parent_forebaffle.epsilon_map

    def find_vertex_in_epsilon(self, vertex, parent_forebaffle):
        """
        Find the epsilon value at a vertex location in the epsilon map.
        
        Parameters
        ----------
        vertex : mp.Vector3 or tuple/list
            Vertex coordinates in real units
        parent_forebaffle : Forebaffle
            Parent forebaffle object containing simulation parameters
            
        Returns
        -------
        float
            Epsilon value at the vertex location
            
        Raises
        ------
        ValueError
            If vertex is outside the epsilon map bounds
        """
        # Extract coordinates - handle both mp.Vector3 and tuple/list
        if isinstance(vertex, mp.Vector3):
            x, y = vertex.x, vertex.y
        else:
            x, y = vertex[0], vertex[1]
        
        # Get simulation parameters
        resolution = parent_forebaffle.mpsat_sim.resolution
        epsilon_map = parent_forebaffle.epsilon_map
        
        # Calculate effective cell size (excluding PML on both sides)
        pml_thickness = parent_forebaffle.mpsat_sim.factor_dpml * parent_forebaffle.mpsat_sim.dpml
        xsize = parent_forebaffle.mpsat_sim.cell_size[0] - 4 * pml_thickness
        ysize = parent_forebaffle.mpsat_sim.cell_size[1] - 4 * pml_thickness
        
        # Transform from real coordinates to pixel indices
        # Real coords: origin at cell center, range [-size/2, +size/2]
        # Pixel coords: origin at array corner, range [0, size*resolution]
        x_idx = int((x + xsize / 2) * resolution)
        y_idx = int((y + ysize / 2) * resolution)
        
        # Validate bounds - note: epsilon_map.shape = (rows, cols) = (y_size, x_size)
        if not (0 <= y_idx < epsilon_map.shape[0] and 
                0 <= x_idx < epsilon_map.shape[1]):
            raise ValueError(
                f"Vertex at ({x:.3f}, {y:.3f}) maps to pixel indices ({x_idx}, {y_idx}), "
                f"which is outside epsilon map bounds {epsilon_map.shape} (y, x). "
                f"Valid ranges: y=[0, {epsilon_map.shape[0]-1}], x=[0, {epsilon_map.shape[1]-1}]"
            )
        
        # Access array with [row, col] = [y, x] indexing
        return epsilon_map[y_idx, x_idx]

    def _get_vertex(self, parent_forebaffle, which_vertex: str):
        """Get the specified vertex from the parent forebaffle."""
        v1, v2, v3 = parent_forebaffle.calculate_vertices()
        vertex_map = {'v1': v1, 'v2': v2, 'v3': v3}
        
        if which_vertex not in vertex_map:
            raise ValueError(f"Invalid vertex '{which_vertex}'. Must be 'v1', 'v2', or 'v3'")
        
        return vertex_map[which_vertex]

    def create_flairs(self, parent_forebaffle):
        """Create all flaring components."""
        meep_geometry = []
        
        # Iterate through all flair configurations
        for flair_config in self.flairs:
            vertex = self._get_vertex(parent_forebaffle, flair_config.which_vertex)
            eps_pixel_at_vertex = self.find_vertex_in_epsilon(vertex, parent_forebaffle)
            
            # Check flaring type from the config
            if flair_config.flaring_type == 'linear':
                linear_flair = self._create_linear_flair(vertex, eps_pixel_at_vertex, flair_config, parent_forebaffle)
                meep_geometry.append(linear_flair)

            elif flair_config.flaring_type == 'spline':
                # self._create_spline_flair(vertex, parent_forebaffle)
                pass
            else: 
                raise ValueError("Please give a valid flairing type")
        
        return meep_geometry  # Return only the geometry list, not a tuple
    
    def _create_linear_flair(self, vertex, eps_pixel_at_vertex, flair_config, parent_fb):
        """Create a linear flair extending from the specified vertex."""
        res = parent_fb.mpsat_sim.resolution
        characteristic_length = 1 #mm
        unit_pixel_length = characteristic_length / res
        linear_params = flair_config.linear
        length = linear_params["length"]
        thickness = linear_params["thickness"]
        theta2 = linear_params["theta2"]
        theta2_axis = linear_params["theta2_axis"]
        flair_material = flair_config.get_material()

        # # Calculate the center of the MEEP block using sin-cos approach depending on the rotation axis
        # if theta2_axis == 'x':
        #     x_center = vertex.x + thickness * math.cos(theta2)
        #     y_center = vertex.y + thickness * math.sin(theta2) 
        # elif theta2_axis == 'y':
        #     x_center = vertex.x + thickness * math.sin(theta2)
        #     y_center = vertex.y + thickness * math.cos(theta2)
        # else:
        #     raise ValueError("Invalid rotation axis. Must be 'x' or 'y'.")

        x_center, y_center = exf.linear_flair_center(
            vertex_x=vertex.x,
            vertex_y=vertex.y,
            length=length,
            thickness=thickness,
            angle_degrees=theta2,
            angle_axis=theta2_axis,
            unit_pixel_length=unit_pixel_length,
        )

        # x_center = vertex.x + offset_x +unit_pixel_length 
        # y_center = vertex.y + offset_y +unit_pixel_length
        # x_center, y_center = x_center - thickness, y_center #- thickness
        flair_block_meep = meep_block(size = mp.Vector3(length, thickness, 0),
                                      center = mp.Vector3(x_center, y_center, 0),
                                      material = flair_material,
                                      angle= theta2,
                                      rot_axis= 'z') # This will be always Z

        return flair_block_meep
    
# * ############################################################################################################
# * MAIN FOREBAFFLE CLASS
# * ############################################################################################################


class Forebaffle(object):
    '''
    Class defining a triangular forebaffle structure.
    '''
    _VALID_ABSORBER_SIDES = {'above', 'below', 'start_cap', 'end_cap'}

    def __init__(self,
                 mpsat_sim,
                 epsilon_map,
                 freq,
                 shape= 'linear',
                 angle_degrees=30,
                 x_vertex=None,
                 y_vertex=None,
                 material=None,
                 epsilon_real=5.4,
                 epsilon_imag=0,
                 name=None,
                 components: Optional[List[ForebaffleComponent]] = None,
                 # Linear Forebaffle parameters :
                 hypotenuse=70,
                 base=None,
                 height=None,
                 # Spline Forebaffle parameters:
                 num_periods=1,
                 amplitude=5,
                 no_of_points=300,
                 scaling_factor=3,
                 spline_degree=3,
                 spline_smoothing=1,
                 fb_thickness=10,
                 add_absorber=True,
                 absorber_side= 'above',
                 absorber_epsilon_real=5.4,
                 absorber_epsilon_imag=0,
                 absorber_thickness=5
):
        '''
        Defines a right angled triangular forebaffle structure with a specific opening angle

        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object
        epsilon_map: np.ndarray
            Epsilon map for the whole system
            This is useful for adding random geometries which are not trivial using MEEPs objects
        shape: str
            Shape of the forebaffle (default: 'linear')
            Available Shapes: ['linear', 'spline']
        angle_degrees : float, optional
            Angle of the forebaffle in degrees (default: 30)
        x_vertex : float, optional
            X-coordinate of the vertex (default: -300)
            v1 in the code
        y_vertex : float, optional
            Y-coordinate of the vertex (default: bottom of simulation cell)
            v1 in the code
        base : float, optional
            Length of the base of the right angled trianglular forebaffle (default: 40)
        hypotenuse : float, optional
            Length of the hypotenuse of the right angled trianglular forebaffle (default: 70)
        height : float, optional
            Length of the height of the right angled trianglular forebaffle (default: 30)
        material : mp.Medium, optional
            Material for the forebaffle (overrides epsilon if provided)
        epsilon_real : float, optional
            Permittivity of the material (default: 5.4)
        epsilon_imag : float, optional
            Imaginary part of permittivity (default: 0)
        freq : float, optional
            Frequency for material properties (default: 1/3)
        name : str, optional
            Name of the object (default: None)
            
        'THE BELOW PARAMETERS ARE FOR SPLINE FOREBAFFLE DESIGN'
        num_periods: int, optional (only needed if shape = 'spline')
            Number of oscillations between start and end (default: 1)
        amplitude: float, optional (only needed if shape = 'spline')
            Amplitude of oscillation in mm (default: 5). Positive amplitude
            bulges toward the baffle's v1 (base-vertex) side, so top/bottom
            baffles built from mirrored vertices produce mirrored shapes.
        no_of_points: int, optional (only needed if shape = 'spline')
            Number of points between start and end (default: 300)
        scaling_factor: float, optional (only needed if shape = 'spline')
            Frequency factor for the oscillation (default: 3)
        spline_degree: int, optional (only needed if shape = 'spline')
            Degree of the spline (default: 3)
        spline_smoothing: float, optional (only needed if shape = 'spline')
            Smoothing factor for the spline (default: 1)
        fb_thickness: float, optional
            Thickness of the forebaffle wall, measured perpendicular to the
            wall surface, so it is independent of the baffle angle (default: 10)
        add_absorber: bool, optional
            Whether to add an absorber layer (default: True)
        absorber_side: str or list of str, optional
            Which side(s) of the spline to add the absorber (default: 'above').
            Accepts a single side, or a list to compose multiple, e.g.
            `['start_cap', 'above']`.
            available options: ['above', 'below', 'start_cap', 'end_cap']
            plus the shorthands 'both' (-> above+below) and 'all' (-> every side).
            'start_cap'/'end_cap' cover the short exposed edge at the start/end
            of the wall (the sharpest diffracting feature, left bare by
            'above'/'below' alone).
        absorber_epsilon_real: float, optional
            Real part of the permittivity for the absorber (default: 5.4)
        absorber_epsilon_imag: float, optional
            Imaginary part of the permittivity for the absorber (default: 0)
        absorber_thickness: float, optional
            Thickness of the absorber layer, measured perpendicular to the wall
            surface, so it is independent of the baffle angle (default: 2.0)
        '''
        self.mpsat_sim = mpsat_sim
        self.epsilon_map = epsilon_map
        
        # Freq
        self.freq = freq
        
        # Basic parameters
        self.name = name if name else "Forebaffle"
        self.object_type = 'Forebaffle'
        self.fb_shape = shape
        
        # Geometry parameters
        self.angle_degrees = angle_degrees
        self.angle_radians = np.radians(angle_degrees)
        self.x_vertex = x_vertex if x_vertex is not None else -300
        self.y_vertex = y_vertex if y_vertex is not None else -self.mpsat_sim.cell_size[1]/2
        self.hypotenuse = hypotenuse

        
        if base is None and height is None:
            self.base, self.height = self._calculate_base_height_from_angle_hypotenuse(
                angle_degrees, hypotenuse
            )
            print("base",self.base,"\t","height",self.height)
        elif base is None or height is None:
            raise ValueError("Either provide hypotenuse + angle OR provide all the sides of the forebaffle.")
        else:
            self.base = base
            self.height = height

        # Material properties to be used in simulation using meep geometries
        if material is not None:
            self.material = material
        else:
            # Check if epsilon_real is -inf (perfect conductor)
            if np.isinf(epsilon_real) and epsilon_real < 0:
                self.material = mp.perfect_electric_conductor
            # Check if imaginary part is provided
            elif epsilon_imag != 0:
                self.epsilon_real = epsilon_real
                self.epsilon_imag = epsilon_imag
                self.conductivity = epsilon_imag * 2 * np.pi * freq / epsilon_real
                self.material = mp.Medium(epsilon=self.epsilon_real, D_conductivity=self.conductivity)
            else:
                self.material = mp.Medium(epsilon=epsilon_real)

        # Component system for additional features
        self.components = components if components else []
        
        # Spline parameters
        if shape == 'spline':
            self.spline_num_periods = num_periods
            self.spline_amplitude = amplitude
            self.spline_no_of_points = no_of_points
            self.spline_scaling_factor = scaling_factor
            self.spline_fb_thickness = fb_thickness
            self.spline_degree = spline_degree
            self.spline_smoothing = spline_smoothing
            self.spline_add_absorbers = add_absorber
            self.spline_absorber_side = self._normalize_absorber_sides(absorber_side)
            self.spline_abs_thickness =absorber_thickness
            self.spline_abs_epsilon_real = absorber_epsilon_real
            self.spline_abs_epsilon_imag = absorber_epsilon_imag

    def __str__(self):
        return f"{self.name}: angle={self.angle_degrees}°, height={self.height}"

    def _normalize_absorber_sides(self, absorber_side):
        """
        Normalize `absorber_side` into a set of concrete sides.

        Accepts a single string (e.g. 'above') or a list/tuple/set of strings
        (e.g. ['start_cap', 'above']). The shorthands 'both' (-> above+below)
        and 'all' (-> every side) expand to multiple sides each.
        """
        if isinstance(absorber_side, str):
            absorber_side = [absorber_side]

        sides = set()
        for side in absorber_side:
            if side == 'both':
                sides.update({'above', 'below'})
            elif side == 'all':
                sides.update(self._VALID_ABSORBER_SIDES)
            elif side in self._VALID_ABSORBER_SIDES:
                sides.add(side)
            else:
                raise ValueError(
                    f"Invalid absorber_side entry '{side}'. Must be one of "
                    f"{sorted(self._VALID_ABSORBER_SIDES)} or the shorthands "
                    f"'both'/'all'.")
        return sides
    
    def _calculate_base_height_from_angle_hypotenuse(self, angle_degrees, hypotenuse):
        angle_radians = np.radians(angle_degrees)
        # Using ASTC rule from Trigonometry
        if 0 <= angle_degrees <= 90:
            base = float(hypotenuse * np.cos(angle_radians))
            height = float(hypotenuse * np.sin(angle_radians))
        elif 90 < angle_degrees <= 180:
            base = float(hypotenuse * -1* np.cos(angle_radians))
            height = float(hypotenuse * np.sin(angle_radians))  
        elif 180 < angle_degrees <= 270:
            base = float(hypotenuse * -1* np.cos(angle_radians))
            height = float(hypotenuse * -1* np.sin(angle_radians))
        else:
            base = float(hypotenuse * np.cos(angle_radians))
            height = float(hypotenuse * -1* np.sin(angle_radians))
        return base, height

    def calculate_vertices(self):
        """
        Calculate the vertices of the triangular forebaffle based on the provided parameters
        
        Returns
        """
        # Calculate the coordinates of the three vertices
        v1 = mp.Vector3(self.x_vertex, self.y_vertex)  # Vertex where angle is measured
        
        # The right angle is at v2, so we calculate v2 based on the base and angle

        perp_height = self.height
        if 0 <= self.angle_degrees < 90:
            # Quadrant 1: base right, perpendicular up
            v2 = mp.Vector3(self.x_vertex + self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex + self.base, self.y_vertex + perp_height)
            
        elif 90 <= self.angle_degrees < 180:
            # Quadrant 2: base left, perpendicular up
            v2 = mp.Vector3(self.x_vertex - self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex - self.base, self.y_vertex + perp_height)
            
        elif 180 <= self.angle_degrees < 270:
            # Quadrant 3: base left, perpendicular down
            v2 = mp.Vector3(self.x_vertex - self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex - self.base, self.y_vertex - perp_height)
            
        else:  # 270 <= self.angle_degrees < 360
            # Quadrant 4: base right, perpendicular down
            v2 = mp.Vector3(self.x_vertex + self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex + self.base, self.y_vertex - perp_height)
                
        print(f"Calculated vertices: v1={v1}, v2={v2}, v3={v3}")
        print(f"Quadrant: {int(self.angle_degrees // 90) + 1}")
        
        # Consider adding the boundary layer thickness to the vertex positions
        if self.name == 'Right Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For right forebaffle, we need to consider the boundary layer on the right side
                v1 = mp.Vector3(v1.x - boundary_layer_size, v1.y)
                v2 = mp.Vector3(v2.x - boundary_layer_size, v2.y)
                v3 = mp.Vector3(v3.x - boundary_layer_size, v3.y)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")

        elif self.name == 'Left Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For left forebaffle, we need to consider the boundary layer on the left side
                v1 = mp.Vector3(v1.x + boundary_layer_size, v1.y)
                v2 = mp.Vector3(v2.x + boundary_layer_size, v2.y)
                v3 = mp.Vector3(v3.x + boundary_layer_size, v3.y)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")
                
        elif self.name == 'Top Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For top forebaffle, we need to consider the boundary layer on the top side
                v1 = mp.Vector3(v1.x, v1.y - boundary_layer_size)
                v2 = mp.Vector3(v2.x, v2.y - boundary_layer_size)
                v3 = mp.Vector3(v3.x, v3.y - boundary_layer_size)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")
                
        elif self.name == 'Bottom Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For bottom forebaffle, we need to consider the boundary layer on the bottom side
                v1 = mp.Vector3(v1.x, v1.y + boundary_layer_size)
                v2 = mp.Vector3(v2.x, v2.y + boundary_layer_size)
                v3 = mp.Vector3(v3.x, v3.y + boundary_layer_size)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")        
        else:
            logger.warning(f"Unknown forebaffle name '{self.name}' - no boundary layer adjustment applied")
            
        return v1, v2, v3
    

        
    def _create_spline_forebaffle_with_prisms(self, start_vertex, end_vertex):
        """
        Create a spline forebaffle using multiple MEEP prism objects.
        
        This creates a stepped approximation of the spline curve using rectangular
        prism elements, similar to step file export in CAD software.
        
        Parameters
        ----------
        start_vertex, end_vertex : mp.Vector3
            Start and end points of the spline
            
        Returns
        -------
        List[mp.GeometricObject]
            List of prism objects forming the spline structure
        """
        from scipy.interpolate import UnivariateSpline
        
        # Get simulation parameters
        x_start, y_start = start_vertex.x, start_vertex.y
        x_end, y_end = end_vertex.x, end_vertex.y
        
        # Spline parameters
        num_periods = self.spline_num_periods
        factor = self.spline_scaling_factor
        no_of_points = self.spline_no_of_points

        # Orient the sinusoidal modulation with the baffle's opening direction so
        # that top/bottom baffles built from mirrored vertices are mirror images:
        # positive amplitude always bulges toward the v1 (optical-axis) side.
        y_orientation = 1.0 if self.v1.y >= self.v3.y else -1.0
        amplitude = self.spline_amplitude * y_orientation
        
        # Generate spline curve
        x_points = np.linspace(x_start, x_end, num=no_of_points)
        y_base = np.linspace(y_start, y_end, len(x_points))
        y_periodic = y_base + amplitude * np.sin(factor * np.pi * num_periods * 
                                                (x_points - x_start) / (x_end - x_start))
        # Calculate offset correction before spline
        y_start_uncorrected = y_periodic[0]
        y_end_uncorrected = y_periodic[-1]
        offset_start = y_start - y_start_uncorrected
        offset_end = y_end - y_end_uncorrected
        
        # Apply linear interpolation of offset across the entire curve
        # This ensures endpoints match while preserving spline shape
        offset_correction = np.linspace(offset_start, offset_end, len(y_periodic))
        y_periodic = y_periodic + offset_correction

        # Create smooth spline
        spline = UnivariateSpline(x_points, y_periodic, k=self.spline_degree,
                                s=self.spline_smoothing)

        # Wall/absorber offsets follow the local surface normal, not +/-y. A
        # vertical offset would shrink the true perpendicular thickness by
        # cos(wall angle), so a steep baffle would end up a fraction of its
        # nominal fb_thickness (and its absorber a fraction of a pixel).
        spline_derivative = spline.derivative()

        def normal_at(x):
            """Unit normal to the spline at x, oriented toward +y."""
            slope = float(spline_derivative(x))
            norm = np.hypot(1.0, slope)
            return -slope / norm, 1.0 / norm

        def tangent_at(x):
            """Unit tangent to the spline at x, oriented toward +x."""
            slope = float(spline_derivative(x))
            norm = np.hypot(1.0, slope)
            return 1.0 / norm, slope / norm

        # Create prism objects
        geometries = []
        thickness = self.spline_fb_thickness
        
        # # Create material with proper absorption handling
        # if self.epsilon_imag != 0:
        #     # Use conductivity for absorption
        #     freq = 1/3  # Default frequency
        #     conductivity = self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real
        #     material = mp.Medium(epsilon=self.epsilon_real, D_conductivity=conductivity)
        # else:
        #     material = mp.Medium(epsilon=self.epsilon_real)
        material = self.material
        
        # Use fewer segments for prism creation (e.g., 1/10 of original points)
        prism_segments = max(10, no_of_points // 10)  # At least 10 segments
        x_prism_points = np.linspace(x_start, x_end, prism_segments)
        
        for i in range(len(x_prism_points) - 1):
            x1 = x_prism_points[i]
            x2 = x_prism_points[i + 1]
            
            # Evaluate spline at segment endpoints
            y1 = float(spline(x1))
            y2 = float(spline(x2))

            # Offset each endpoint along its own normal, so neighbouring
            # segments share an edge exactly and no gaps open up along the wall
            nx1, ny1 = normal_at(x1)
            nx2, ny2 = normal_at(x2)
            half = thickness / 2

            # Create a quadrilateral prism for this segment
            # Vertices at top and bottom of the segment
            v1_bottom = mp.Vector3(x1 - nx1*half, y1 - ny1*half)
            v2_bottom = mp.Vector3(x2 - nx2*half, y2 - ny2*half)
            v2_top = mp.Vector3(x2 + nx2*half, y2 + ny2*half)
            v1_top = mp.Vector3(x1 + nx1*half, y1 + ny1*half)
            
            # Create prism (quadrilateral)
            prism = mp.Prism(
                vertices=[v1_bottom, v2_bottom, v2_top, v1_top],
                height=self.height,
                axis=mp.Vector3(0, 0, 1),
                material=material
            )
            geometries.append(prism)
        
        # Add absorber layers if needed
        if self.spline_add_absorbers:
            # Create absorber material properly
            if self.spline_abs_epsilon_imag != 0:
                freq = self.freq
                abs_conductivity = self.spline_abs_epsilon_imag * 2 * np.pi * freq / self.spline_abs_epsilon_real
                absorber_material = mp.Medium(epsilon=self.spline_abs_epsilon_real, 
                                             D_conductivity=abs_conductivity)
            else:
                absorber_material = mp.Medium(epsilon=self.spline_abs_epsilon_real)
            
            absorber_thickness = self.spline_abs_thickness
            
            x_prism_points = np.linspace(x_start, x_end, prism_segments)
            
            for i in range(len(x_prism_points) - 1):
                x1 = x_prism_points[i]
                x2 = x_prism_points[i + 1]
                
                y1 = float(spline(x1))
                y2 = float(spline(x2))

                nx1, ny1 = normal_at(x1)
                nx2, ny2 = normal_at(x2)
                half = thickness / 2
                outer = half + absorber_thickness

                if 'above' in self.spline_absorber_side:
                    # Absorber above
                    v1_inner = mp.Vector3(x1 + nx1*half, y1 + ny1*half)
                    v2_inner = mp.Vector3(x2 + nx2*half, y2 + ny2*half)
                    v2_outer = mp.Vector3(x2 + nx2*outer, y2 + ny2*outer)
                    v1_outer = mp.Vector3(x1 + nx1*outer, y1 + ny1*outer)

                    absorber_prism = mp.Prism(
                        vertices=[v1_inner, v2_inner, v2_outer, v1_outer],
                        height=self.height,
                        axis=mp.Vector3(0, 0, 1),
                        material=absorber_material
                    )
                    geometries.append(absorber_prism)
                
                if 'below' in self.spline_absorber_side:
                    # Absorber below
                    v1_outer = mp.Vector3(x1 - nx1*outer, y1 - ny1*outer)
                    v2_outer = mp.Vector3(x2 - nx2*outer, y2 - ny2*outer)
                    v2_inner = mp.Vector3(x2 - nx2*half, y2 - ny2*half)
                    v1_inner = mp.Vector3(x1 - nx1*half, y1 - ny1*half)

                    absorber_prism = mp.Prism(
                        vertices=[v1_outer, v2_outer, v2_inner, v1_inner],
                        height=self.height,
                        axis=mp.Vector3(0, 0, 1),
                        material=absorber_material
                    )
                    geometries.append(absorber_prism)

            # Cap the exposed short edges at the ends of the wall - 'above'/'below'
            # only coat the long faces, leaving the wall's terminal edges (the
            # sharpest diffracting feature) bare even when both faces are covered.
            if 'start_cap' in self.spline_absorber_side:
                geometries.append(self._create_end_cap_absorber(
                    x_edge=x_start, y_edge=float(spline(x_start)),
                    thickness=thickness, absorber_thickness=absorber_thickness,
                    absorber_material=absorber_material, direction=-1,
                    normal=normal_at(x_start), tangent=tangent_at(x_start)))

            if 'end_cap' in self.spline_absorber_side:
                geometries.append(self._create_end_cap_absorber(
                    x_edge=x_end, y_edge=float(spline(x_end)),
                    thickness=thickness, absorber_thickness=absorber_thickness,
                    absorber_material=absorber_material, direction=1,
                    normal=normal_at(x_end), tangent=tangent_at(x_end)))

        return geometries

    def _create_end_cap_absorber(self, x_edge, y_edge, thickness, absorber_thickness,
                                  absorber_material, direction, normal, tangent):
        """
        Cap the exposed short edge at one end of the spline wall with an absorber
        block, extending outward from (x_edge, y_edge) by `absorber_thickness`
        along the wall's tangent (`direction` = -1 for the start edge, +1 for the
        end edge). The cap spans the wall cross-section along `normal`, so it
        stays flush with the face absorbers whatever the wall angle.

        Spans the full above/below stack when those sides are also active, so
        there's no bare gap at the corner where a cap meets a face absorber.
        """
        half_above = thickness / 2 + (
            absorber_thickness if 'above' in self.spline_absorber_side else 0)
        half_below = thickness / 2 + (
            absorber_thickness if 'below' in self.spline_absorber_side else 0)

        nx, ny = normal
        tx, ty = tangent
        off_x = direction * absorber_thickness * tx
        off_y = direction * absorber_thickness * ty

        low_x, low_y = x_edge - nx*half_below, y_edge - ny*half_below
        high_x, high_y = x_edge + nx*half_above, y_edge + ny*half_above

        vertices = [
            mp.Vector3(low_x, low_y),
            mp.Vector3(low_x + off_x, low_y + off_y),
            mp.Vector3(high_x + off_x, high_y + off_y),
            mp.Vector3(high_x, high_y),
        ]

        return mp.Prism(
            vertices=vertices,
            height=self.height,
            axis=mp.Vector3(0, 0, 1),
            material=absorber_material
        )

    def assemble(self):
        """
        Assemble the forebaffle with all components.
        
        Returns
        -------
        List[mp.GeometricObject]
            List of MEEP geometric objects (main structure + components)
        """
        self.v1, self.v2, self.v3 = self.calculate_vertices()
        geometries = []
                
        if self.fb_shape == 'linear':

            # Main forebaffle structure
            main_forebaffle = mp.Prism(
                vertices=[self.v1, self.v2, self.v3],
                height=self.height,
                axis=mp.Vector3(0, 0, 1),
                material=self.material
            )
            
            geometries.append(main_forebaffle)
            
            # Add component geometries
            for component in self.components:
                geometries.extend(component.get_geometry(self))
                    
            logger.info(f"Forebaffle assembled with {len(geometries)} geometric objects")
            
            return geometries
        
        elif self.fb_shape == 'spline':
            
            # Ensure start_vertex has smaller x-coordinate for monotonic spline
            start_v = self.v3
            end_v = self.v1
            if start_v.x > end_v.x:
                start_v, end_v = end_v, start_v
            
            geometries = self._create_spline_forebaffle_with_prisms(
                start_vertex=start_v,
                end_vertex=end_v
            )
            logger.info(f"Forebaffle assembled with {len(geometries)} spline prism segments")
            return geometries

        else:
            raise ValueError(f"Unknown forebaffle shape '{self.fb_shape}'")
        
