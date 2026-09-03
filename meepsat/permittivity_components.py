import sys
import os
import site
from pathlib import Path
#import meep_testings as mp
import meep as mp
import numpy as np
import warnings
import scipy.optimize as sc
import h5py
import math
from memory_profiler import profile
import matplotlib.pyplot as plt
from matplotlib import rc

import meepsat.helpers as exf

# ----------------------------------- Aspheric Lens ----------------------------------- #
class AsphericLens(object):
    
    '''
    Class defining an aspheric lens of arbitrary shape and 
    position, and creating the function of sag (curvature) 
    used to create the permitttivity map
    
    Some segments of this class was adopted from the open source code MEEPART:
    https://github.com/MolinAlexei/MEEPART/blob/main/meep_optics.py#L732
    '''
    
    def __init__(self,
                 diameter, 
                 r1, 
                 r2, 
                 thick,
                 c1 = 0, c2 = 0, 
                 # lens type
                 lens_type = 'aspheric',
                 # Add higher-order aspheric coefficients
                 a1_coeffs = None, a2_coeffs = None,
                 name = None, 
                 x = 0., y = 0., 
                 n_refr = 1.52, 
                 #! Single ARC parameters (useless- but not touching for now to avoid bugs)       
                 AR_left = None, 
                 AR_right = None,
                 AR_material = 1.52/4,
                 #! Parameters for multi-layer ARCs (single + multi-layer can be used simultaneously)
                 AR_left_layers = None,
                 AR_left_materials = None,
                 AR_right_layers = None,
                 AR_right_materials = None,
                 #! Adding stepped pyramid ARC here
                 ARC_type = None,
                 step_ARC_nlayers = None,
                 step_ARC_pitch = None,
                 step_ARC_kerf = None,
                 step_ARC_depth = None,
                 step_ARC_width = None,
                 step_ARC_material = None,
                 step_ARC_angle = 'perpendicular_to_surface',
                 step_ARC_rot_axis = 'z',
                 step_ARC_offset = [0,0],
                 delam_thick = 0,
                 delam_width = 10,
                 radial_slope = 0,
                 axial_slope = 0,
                 surf_err_width = 1,
                 surf_err_scale = 0,
                 custom_def = False,
                 eps = None,
                 mpsat_sim = None):
        '''
        Defines the attributes of the Lens object

        Arguments
        ---------       
        diameter : float 
            Diameter of the lens
        r1 : float
            Left surface curvature radius
        r2 : float  
            Right surface cruvature radius
        thick : float
            Thickness of lens on the optical axis
        name : str, optional
            Name of object (default : None)
        c1 : float, optional
            Left surface aspheric parameter (default : 0)
        c2 : float, optional
            Right surface aspheric parameter (default : 0)
        lens_type : str, optional
            Type of lens, either 'aspheric' or 'extended_aspheric'
            (default : 'extended_aspheric')
        a1_coeffs : list, optional
            Higher-order aspheric coefficients for left surface (Currently supported for the first 3 coefficients)
            (default : None)
        a2_coeffs : list, optional
            Higher-order aspheric coefficients for right surface (Currently supported for the first 3 coefficients)
            (default : None)
        x : float, optional
            Position of center of left surface along x axis (default : 0)
        y : float, optional
            Position of center of left surface along y axis (default : 0)
        n_refr : float, optional
            Index of refraction of the lens. 
            Set to HDPE by default.
            (default : 1.52) 

    
        #! Single ARC parameters (useless- but not touching for now to avoid bugs)       
        AR_left : float, optional
            Anti Reflection coating thickness of left surface of the lens
            (default : None) 
        AR_right : float, optional
            Anti Reflection coating thickness of right surface of the lens
            (default : None) 
        AR_material : float, optional
            Refractive index of the AR coating material

        #! Parameters for multi-layer ARCs (single + multi-layer can be used simultaneously)
        AR_left_layers : list, optional
            List of thicknesses of each layer in the multi-layer ARC on the left surface
            (default : None)
        AR_left_materials : list, optional
            List of refractive indices of each layer in the multi-layer ARC on the left surface
            (default : None)
        AR_right_layers : list, optional
            List of thicknesses of each layer in the multi-layer ARC on the right surface
            (default : None)
        AR_right_materials : list, optional
            List of refractive indices of each layer in the multi-layer ARC on the right surface
            (default : None)


        #! stepped pyramid ARC parameters
        ARC_type : str, optional
            Type of ARC, either 'stepped_pyramid' or 'default' in the current version
        
        step_ARC_nlayers : int, optional
            Number of layers in the stepped pyramid ARC (default : None)
        
        step_ARC_pitch : float, optional
            Pitch of the stepped pyramid ARC (default : None)

        step_ARC_kerf : float, optional
            Kerf of the stepped pyramid ARC (default : None)

        step_ARC_depth : float, optional
            Depth of the stepped pyramid ARC (default : None)

        step_ARC_width : float, optional
            Width of the stepped pyramid ARC (default : None)

        step_ARC_material : float, optional
            Refractive index of the stepped pyramid ARC material (default : None)

        step_ARC_angle : str, optional
            Angle of the stepped pyramid ARC, either 'perpendicular_to_surface' or None
            If 'perpendicular_to_surface', the ARC is perpendicular to the surface of the lens.
            (default : 'perpendicular_to_surface')

        step_ARC_rot_axis : str, optional
            Axis of rotation for the stepped pyramid ARC, either 'x', 'y', or 'z'
            (default : 'z' i.e. rotation in the xy-plane)

        step_ARC_offset : float, optional
            Offset of the stepped pyramid ARCs base layer's edge center from the lens surface (default : [0, 0])
            [0, 0] means no offset, i.e. the base layer's edge center is at the lens surface.
            
        delam_thick : float, optional
            Thickness of delaminated lumps at their center
            (default : 0)
        delam_width : float, optional
            Width of delaminated lumps along y-axis
            Used in a division, hence default is not 0.
            (default : 10)
        radial_slope : float, optional
            Derivative of the index of refraction w.r.t y-axis (default : 0)
        axial_slope : float, optional
            Derivative of the index of refraction w.r.t x-axis (default : 0)
        surf_err_scale : float, optional
            Width of the gaussian of the distribution of surface errors
            (default : 0)
        surf_err_width : float, optional
            Size of the bins of same surface error (default : 1)
        custom_def : bool, optional 
            Enables custom deformation function (default : False)

        # ^ ######### ^ #
        eps: np.array, optional
           Dielectric map of the other components in the system 
        
        mpsat_sim: object
            MEEPSAT object produced from sim_init() in simulation_2D.py
        '''
        self.name = name                  
        self.diameter = diameter        
        self.r1 = r1                    
        self.r2 = r2                    
        self.c1 = c1                    
        self.c2 = c2

        # Define the lens type
        self.lens_type = lens_type
        if lens_type not in ['aspheric', 'extended_aspheric']:
            raise ValueError("lens_type must be either 'aspheric' or 'extended aspheric'")
        
        # Add higher-order coefficients
        self.a1_coeffs = a1_coeffs if a1_coeffs is not None else None
        self.a2_coeffs = a2_coeffs if a2_coeffs is not None else None
        
        
        self.thick = thick              
        self.x = x                      
        self.y = y                      
        self.eps = n_refr**2            
        self.object_type = 'Lens'

        #! Single ARC parameters (useless- but not touching for now to avoid bugs)       
        self.AR_left = AR_left          
        self.AR_right = AR_right        
        self.AR_material = AR_material #!material of the AR coating  

        #! Parameters for multi-layer ARCs (single + multi-layer can be used simultaneously)
        self.left_layers=AR_left_layers 
        self.left_materials=AR_left_materials
        self.right_layers=AR_right_layers 
        self.right_materials=AR_right_materials
        
        #! stepped pyramid ARC parameters
        if ARC_type == 'stepped_pyramid':
            self.ARC_type = ARC_type
            self.step_ARC_nlayers = step_ARC_nlayers
            self.step_ARC_pitch = step_ARC_pitch
            self.step_ARC_kerf = step_ARC_kerf
            self.step_ARC_depth = step_ARC_depth
            self.step_ARC_width = step_ARC_width
            self.step_ARC_material = step_ARC_material
            self.step_ARC_angle = step_ARC_angle
            self.step_ARC_rot_axis = step_ARC_rot_axis
            self.step_ARC_offset = step_ARC_offset

        self.delam_thick = delam_thick  
        self.delam_width = delam_width  
        self.radial_slope = radial_slope
        self.axial_slope = axial_slope  
        self.surf_err_width = surf_err_width    
        self.surf_err_scale = surf_err_scale    
        self.custom_def = custom_def
        self.permittivity_map = eps # ~ THIS IS THE EPSILON MAP

        # Extracting the required parameters from the MEEPSAT object
        self.res = mpsat_sim.resolution 
        self.dpml = mpsat_sim.factor_dpml*mpsat_sim.dpml # mpsat_sim.dpml 
        # 2 times because there's pml on both sides
        self.size_x, self.size_y, self.size_z = mpsat_sim.cell_size[0] - 2*self.dpml, mpsat_sim.cell_size[1] - 2*self.dpml, mpsat_sim.cell_size[2]
        self.mpsat_sim = mpsat_sim # ~ MEEPSAT object

        #TESTING IMPORTED DEFORMED PROFILE AS CSV
        #deform = []
        #with open('deformedsurface.csv') as csvfile:
        #    reader = csv.reader(csvfile, delimiter=',')
        #    k = 0
        #    for row in reader:
        #        k+= 1 
        #        if k>=11 :
        #            deform.append(np.float(row[2]))
        #deform0 = 2*deform[0]-deform[1]
        #deform.insert(0, deform0)
        #self.deform = deform

    def position(self):
        if self.name is not None : 
            return self.name + ' at position ' + str(self.x)
        else :
            return 'Lens at position ' + str(self.x)

    def even_asphere_lens_eqn(self, y, r, k, higher_order_coeffs):#=[0, 0, 0]):
        '''
        Aspheric lens equation for even aspheric coefficients
        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed
        r : float
            Curvature radius of the lens
        k : float
            Aspheric coefficient
        higher_order_coeffs [A2, A3, A4] : list, optional
            Higher-order aspheric coefficients (default : [0, 0, 0])
        '''
        A2, A3, A4 = higher_order_coeffs
        sag_value = (y**2/r) / (1 + np.sqrt(1 - (1 + k)*y**2/r**2)) + A2 * y**2 + A3 * y**4 + A4 * y**6
        return sag_value

    def extended_asphere_lens_eqn(self, y, r, k, higher_order_coeffs):#=[0, 0, 0]):
        '''
        Aspheric lens equation for extended aspheric coefficients
        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed
        r : float
            Curvature radius of the lens
        a : float
            Aspheric coefficient
        higher_order_coeffs [A1, A2, A3] : list, optional
            Higher-order aspheric coefficients (default : [0, 0, 0])
        '''
        A1, A2, A3 = higher_order_coeffs
        sag_value = (y**2/r) / (1 + np.sqrt(1 - (1 + k)*y**2/r**2)) + A1 * y + A2 * y**2 + A3 * y**3
        return sag_value
    
    def left_surface(self, y):
        '''
        Aspheric lens equation for left surface

        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed

        Returns
        -------
        sag : float
            Sag at at distance y from optical axis.
        '''
        higher_order_coeffs_left = self.a1_coeffs if self.a1_coeffs is not None else [0, 0, 0]

        if self.lens_type == 'aspheric':
            left_surface = self.even_asphere_lens_eqn(y= y, 
                                                       r= self.r1, 
                                                       k = self.c1, 
                                                       higher_order_coeffs= higher_order_coeffs_left)
        elif self.lens_type == 'extended_aspheric':
            left_surface = self.extended_asphere_lens_eqn(y= y, 
                                                       r= self.r1, 
                                                       k = self.c1, 
                                                       higher_order_coeffs= higher_order_coeffs_left)
        else:
            raise ValueError("Invalid lens type. Use 'aspheric' or 'extended_aspheric'.")
        
        if self.r1 != np.inf:
            return left_surface
        else:
            # If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0
        

    def right_surface(self, y):
        '''
        Aspheric lens equation for right surface

        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed

        Returns
        -------
        sag : float
            Sag at at distance y from optical axis.
        '''
        higher_order_coeffs_right = self.a2_coeffs if self.a1_coeffs is not None else [0, 0, 0]
        
        if self.lens_type == 'aspheric':
            right_surface = self.even_asphere_lens_eqn(y= y, 
                                                       r= self.r2, 
                                                       k = self.c2, 
                                                       higher_order_coeffs= higher_order_coeffs_right)
        elif self.lens_type == 'extended_aspheric':
            right_surface = self.extended_asphere_lens_eqn(y= y, 
                                                            r= self.r2, 
                                                            k = self.c2, 
                                                            higher_order_coeffs= higher_order_coeffs_right)
        else:
            raise ValueError("Invalid lens type. Use 'aspheric' or 'extended_aspheric'.")
        
        if self.r2 != np.inf:
            return right_surface
        else:
            # If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0

    def delamination(self, y, y0):       
        '''
        Returns the air layer thickness that makes delamination, it is 
        zero everywhere excpet where there's the lump, centered on y0, defined by
        its width and thickness

        Arguments
        ---------
        y : float
            Distance from optical axis at which 
            the delamination is evaluated
        y0 : float
            Center of the delaminated lump
        Returns
        -------
        delam : float
            Delamination layer thickness along x-axis at y
        '''

        thick = self.delam_thick
        width = self.delam_width
        return np.abs(min((((y-y0)/width)**2-1)*thick, 0))

    def cust_def(self, y):
        '''
        Returns custom deformation function

        Arguments
        ---------
        y : float
            Distance from optical axis at which 
            the deformation is evaluated
        Returns
        -------
        deform : float
            Deformation of surface along x-axis at y
        '''
        if self.custom_def :
            # ~ Insert here the custom function
            return 0

        else :
            return 0
        
    def make_lens_bubbles(self, radius, nb_clusters, nb_per_cluster):
            '''
            Introduces clusters of air bubbles inside the lenses of the system, 
            each cluster has a central bubble and a number of smaller bubble gathered
            around this central bubble
            
            Arguments
            -----------------
            radius : float
                Radius of the central bubble
            nb_clusters : float
                Number of clusters per lens
            nb_per_cluster : 
                Number of bubbles surrounding the main one in each
                cluster
            Notes
            -----
            This function alters the permittivity map. 
            '''

            res = self.res

            #Function which, given a radius, that 
            #returns the indices of the points within 
            #the circle centered on (0,0)
            def bubble(rad):
                '''
                Introduces clusters of air bubbles inside the lenses of the system, 
                each cluster has a central bubble and a number of smaller bubble gathered
                around this central bubble
            
                Arguments
                -----------------
                rad : float
                    Radius of the bubble

                Returns
                -------
                bubble : array
                    Array of indexes within radius
                '''
                bubble = []
                for k in range(-rad, rad+1):
                    for j in range(-rad, rad+1):
                        if k**2 + j**2 <= rad**2 :
                            bubble.append([k,j])
                return np.array(bubble)

            #List of centers of bubbles
            list_centers = []

            #List of radii of bubbles
            list_radii = []

            #Iterate for all lenses
            for component in self.components:

                if component.object_type == 'Lens':

                    #Lens thickness
                    thick = component.thick*res

                    #So that the bubbles aren't generated 
                    #on the very edge of the lenses
                    low = np.int64(np.around(self.size_y*res*0.1))
                    high = np.int64(np.around(self.size_y*res*0.9))

                    #Iterate over cluster numbers
                    for i in range(nb_clusters):

                        #The center of the lens can be anywhere on the y axis
                        y0 = np.random.randint(low = low, high = high)
                    
                        #Left surface sag
                        x_left = np.int64(np.around((
                            component.left_surface(y0/res - self.size_y/2) + 
                            component.x)*res))
                        #Right surface sag       
                        x_right = np.int64(np.around((
                            component.right_surface(y0/res - self.size_y/2) + 
                            component.x)*res + 
                            thick))

                        #The center of the cluster has to be inside the lens
                        x0 = np.random.randint(low = x_left, high = x_right+1)

                        #Radius of the main can vary by 10 percent
                        radius_0 = radius*(0.9 + np.random.random()*0.2)
                    
                        #Update lists
                        list_centers.append([x0,y0])
                        list_radii.append(radius_0)

                        #Iterate over the number of surrounding bubbles
                        for k in range(nb_per_cluster):

                            #The center of each surrounding bubble is random, within
                            #a certain distance of the central bubble
                            phi = np.random.random()*2*np.pi
                            r = radius_0*(1 + np.random.random()*3)

                            #change of variables
                            x_k = np.int64(np.around(r*np.cos(phi)*res))
                            y_k = np.int64(np.around(r*np.sin(phi)*res))

                            #The radius is a function of distance, the farther the 
                            #smaller
                            radius_k = radius_0*np.exp(-r/(3*radius_0))*np.random.random()

                            #Update lists
                            list_centers.append([x0+x_k, y0+y_k])
                            list_radii.append(radius_k)

            list_centers = np.array(list_centers)
            list_radii = np.array(list_radii)
            list_all = []

            #Making bubbles for all centers and radii
            for k in range(len(list_centers)):
                radius_k = np.int64(np.around(list_radii[k]*res))
                bubble_k = bubble(radius_k)
                for u in bubble_k : 
                    list_all.append(list_centers[k] + u)

            #Update the map
            for index in list_all : 
                self.permittivity_map[index[0], index[1]] = 1
        

    def write_lens(self, comp, eps_map, res):
        '''
        The lens equation returns a sag (distance from plane orth. to
        optical axis) as a function of distance from optical axis y,
        so the code cycles through the different y to change the 
        dielectric map between left surface and right surface
        ---------
        comp : component
            Lens component object
        eps_map : 2D or 3D array
            Dielectric map on which the lens will be written
        res : float
            Resolution of map
        '''

        # The y axis has its zero in the middle of the cell, the offset
        # is mid_y
        mid_y = np.int64(self.size_y*res/2)

        #Thickness of the lens on optical axis
        thick = comp.thick*res

        #Generate the center of the lumps made by delamination, 
        #different for the left and right surface
        high = np.int64(np.around(self.size_y/2))
        y0_left = np.random.randint(low = -high, high = high)
        y0_right = np.random.randint(low = -high, high = high)

        radius = np.int64(np.float64(comp.diameter*res/2))

        #Generates the bins of random surface errors.
        if comp.surf_err_scale!=0 :
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.around(np.random.normal(scale = comp.surf_err_scale*res,
                                                  size = nb_bins))
            err_right = np.around(np.random.normal(scale = comp.surf_err_scale*res, 
                                                   size = nb_bins))

        if comp.surf_err_scale == 0:
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)
        
        #Iterates y over the radius, as the lenses are symmetric
        #above and below the optical axis
        for y_res in range(radius) :           

            #Left surface sag
            x_left = np.int64(np.around((
                        comp.left_surface(y_res/res) + self.dpml + 
                        comp.x - comp.cust_def((y_res+mid_y)/res))*res))
            #Right surface sag       
            x_right = np.int64(np.around((
                        comp.right_surface(y_res/res) + 
                        comp.x + self.dpml -
                        comp.cust_def((y_res+mid_y)/res))*res + 
                        thick))
            
            #Above and below the optical axis :
            y_positive = int(self.dpml*res + mid_y + y_res)
            y_negative = int(self.dpml*res + mid_y - y_res)

            #Get the delamination as a function of y on left surface
            delam_pos_L = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_left)))
            delam_neg_L = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_left)))

            #Get the delamination as a function of y on right surface
            delam_pos_R = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_right)))
            delam_neg_R = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_right)))
            
            #Gradient in the index
            #ONLY WORKS WHEN NO SURFACE DEFECT
            radial_slope = comp.radial_slope/res
            axial_slope = comp.axial_slope/res
            if radial_slope != 0 or axial_slope != 0 : 
            
                eps0 = comp.eps
                x0 = np.int64(np.around(comp.x*res))
                x_range = range(x_left, x_right+1) 
                #The value is squared as the permittivity is index squared
                eps_line = [eps0 + 
                            (y_res*radial_slope)**2 + 
                            ((k-x0)*axial_slope)**2 for k in x_range]
            if radial_slope ==0 and axial_slope == 0 :
                eps_line = comp.eps


            #Surface error
            err_bin_idx = int(np.around(y_res/res/comp.surf_err_width))
            err_left_pos = int(err_left[err_bin_idx])
            err_left_neg = int(err_left[- err_bin_idx])

            err_right_pos = int(err_left[err_bin_idx]) 
            err_right_neg = int(err_left[- err_bin_idx])

            x_left_neg = int(x_left + err_left_neg)
            x_left_pos = int(x_left + err_left_pos)

            x_right_neg = int(x_right + err_right_neg)
            x_right_pos = int(x_right + err_right_pos)


            #Write lens between left and right surface below optical axis
            eps_map[x_left_neg: x_right_neg+1, y_negative] *= eps_line
            
            #So that the center line is not affected twice :
            if y_res != 0 :
                #Write lens between left and right surface above optical axis
                eps_map[x_left_pos: x_right_pos+1, y_positive] *= eps_line
            
            #Write AR coating on left surface
            if comp.AR_left is not None :

                AR_thick = np.int64(np.around(comp.AR_left*res))

                eps_map[x_left_neg - AR_thick - delam_neg_L: x_left_neg - 
                        delam_neg_L, y_negative] *= comp.AR_material

                if y_res != 0 :
                    eps_map[x_left_pos - AR_thick - delam_pos_L: x_left_pos - 
                            delam_pos_L, y_positive] *= comp.AR_material
            
            #Write AR coating on right surface                    
            if comp.AR_right is not None :
                
                AR_thick = np.int64(np.around(comp.AR_right*res))

                eps_map[x_right_neg + 1 + delam_neg_R: AR_thick + x_right_neg + 
                        1 + delam_neg_R, y_negative] *= comp.AR_material

                if y_res != 0 :
                    eps_map[x_right_pos + 1 + delam_pos_R: AR_thick + 
                            x_right_pos + 1 + delam_pos_R, 
                            y_positive] *= comp.AR_material
                    
                    
    
    def assemble(self):
        """
        Assembling the lens object by callling the write_lens method
        """
        self.write_lens(self, self.permittivity_map, self.res)
        return self.permittivity_map
        

    ### ^ FOR PLOTTING AND SAVING THE EPSILON MAP ^ ###

    def plot_lenses(self, save = False):
        '''
        Plots the permittivity map, where we can see only the lenses,
        allows to check their dispostion and shape
        '''
        extent = (0, 
                  len(self.permittivity_map[:])/self.res,
                  0,
                  len(self.permittivity_map[:][0])/self.res)
        plt.figure(dpi = 150)
        plt.title('Permittivity map')
        plt.imshow(self.permittivity_map.transpose(), extent = extent)
        if save:
            plt.savefig('Lenses.png')
        plt.show()
        plt.close()
    

    def write_h5file(self, parallel=False, filename='epsilon_map'):
        '''
        Writes the file that will then be 
        read within the MEEP simulation

        Arguments
        ---------
        parallel : bool, optional
            If the computation is run in parallel (default : False)
        filename : str, optional
            Name of the permittivity map file written. 
            Needs to be the same name given to the MEEP simulation
            (default : 'epsilon_map')
        '''
        import os
        
        # Store the full path
        self.mapname = filename
        full_path = filename + '.h5'
        
        if parallel:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            
            # Only rank 0 writes the file
            if rank == 0:
                # Ensure directory exists
                file_dir = os.path.dirname(full_path) if os.path.dirname(full_path) else '.'
                os.makedirs(file_dir, exist_ok=True)
                
                # Remove old file if it exists
                if os.path.exists(full_path):
                    os.remove(full_path)
                
                with h5py.File(full_path, 'w') as h:
                    size_x = len(self.permittivity_map[:, 0])
                    size_y = len(self.permittivity_map[0, :])
                    dset = h.create_dataset('eps', (size_x, size_y), 
                                            dtype='float32')
                    dset[:, :] = self.permittivity_map.astype('float32')
                    h.flush()  # Ensure data is written to disk
                
                print(f"Rank 0: HDF5 file written to {os.path.abspath(full_path)}")
            
            # Wait for rank 0 to finish writing
            comm.barrier()
            
            # All ranks verify the file exists
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"HDF5 file not found at {os.path.abspath(full_path)}")
            
            # Additional barrier to ensure file system sync
            comm.barrier()            

        else:
            # Ensure directory exists
            file_dir = os.path.dirname(full_path) if os.path.dirname(full_path) else '.'
            os.makedirs(file_dir, exist_ok=True)
            
            # Remove old file if it exists
            if os.path.exists(full_path):
                os.remove(full_path)
            
            with h5py.File(full_path, 'w') as h:
                size_x = len(self.permittivity_map[:, 0])
                size_y = len(self.permittivity_map[0, :])
                dset = h.create_dataset('eps', (size_x, size_y), 
                                        dtype='float32', 
                                        compression='gzip')
                dset[:, :] = self.permittivity_map
                h.flush()  # Ensure data is written to disk
            
            print(f"HDF5 file written to {os.path.abspath(full_path)}")
    

    def write_lens_nARC(self, comp, eps_map, res, 
                    AR_left_layers=None, AR_left_materials=None,
                    AR_right_layers=None, AR_right_materials=None):
        '''
        Enhanced version of write_lens that supports multiple AR coating layers.
        The lens equation returns a sag (distance from plane orth. to
        optical axis) as a function of distance from optical axis y,
        so the code cycles through the different y to change the 
        dielectric map between left surface and right surface
        ---------
        Parameters:
        comp : component
            Lens component object
        eps_map : 2D or 3D array
            Dielectric map on which the lens will be written
        res : float
            Resolution of map
        AR_left_layers : list of float, optional
            List of thicknesses for each AR coating layer on the left surface
        AR_left_materials : list of float, optional
            List of permittivity values for each AR coating layer on the left surface
        AR_right_layers : list of float, optional
            List of thicknesses for each AR coating layer on the right surface
        AR_right_materials : list of float, optional
            List of permittivity values for each AR coating layer on the right surface
        '''
        # Validate AR coating parameters
        if AR_left_layers is not None and AR_left_materials is None:
            raise ValueError("AR_left_materials must be provided with AR_left_layers")
        if AR_right_layers is not None and AR_right_materials is None:
            raise ValueError("AR_right_materials must be provided with AR_right_layers")
        
        if AR_left_layers is not None and len(AR_left_layers) != len(AR_left_materials):
            raise ValueError("AR_left_layers and AR_left_materials must have the same length")
        if AR_right_layers is not None and len(AR_right_layers) != len(AR_right_materials):
            raise ValueError("AR_right_layers and AR_right_materials must have the same length")

        # The y axis has its zero in the middle of the cell, the offset is mid_y
        mid_y = np.int64(self.size_y*res/2)

        # Thickness of the lens on optical axis
        thick = comp.thick*res

        # Generate the center of the lumps made by delamination, 
        # different for the left and right surface
        high = np.int64(np.around(self.size_y*0.9/2))
        y0_left = np.random.randint(low=-high, high=high)
        y0_right = np.random.randint(low=-high, high=high)

        radius = np.int64(comp.diameter*res/2)

        # Generates the bins of random surface errors.
        if comp.surf_err_scale != 0:
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.around(np.random.normal(scale=comp.surf_err_scale*res,
                                                size=nb_bins))
            err_right = np.around(np.random.normal(scale=comp.surf_err_scale*res, 
                                                size=nb_bins))
        else:
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)

        # Iterate y over the radius, as the lenses are symmetric
        # above and below the optical axis
        for y_res in range(radius):           
            
            # Left surface sag
            x_left = np.int64(np.around((
                        comp.left_surface(y_res/res) + self.dpml + 
                        comp.x - comp.cust_def((y_res+mid_y)/res))*res))
            # Right surface sag       
            x_right = np.int64(np.around((
                        comp.right_surface(y_res/res) + 
                        comp.x + self.dpml -
                        comp.cust_def((y_res+mid_y)/res))*res + 
                        thick))
            
            # Above and below the optical axis:
            y_positive = int(self.dpml*res + mid_y + y_res)
            y_negative = int(self.dpml*res + mid_y - y_res)

            # Get the delamination as a function of y on left surface
            delam_pos_L = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_left)))
            delam_neg_L = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_left)))

            # Get the delamination as a function of y on right surface
            delam_pos_R = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_right)))
            delam_neg_R = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_right)))
            
            # Gradient in the index
            # ONLY WORKS WHEN NO SURFACE DEFECT
            radial_slope = comp.radial_slope/res
            axial_slope = comp.axial_slope/res
            if radial_slope != 0 or axial_slope != 0: 
                eps0 = comp.eps
                x0 = np.int64(np.around(comp.x*res))
                x_range = range(x_left, x_right+1) 
                # The value is squared as the permittivity is index squared
                eps_line = [eps0 + 
                            (y_res*radial_slope)**2 + 
                            ((k-x0)*axial_slope)**2 for k in x_range]
            else:
                eps_line = comp.eps

            # Surface error
            err_bin_idx = int(np.around(y_res/res/comp.surf_err_width))
            err_left_pos = int(err_left[err_bin_idx])
            err_left_neg = int(err_left[- err_bin_idx])

            err_right_pos = int(err_left[err_bin_idx]) 
            err_right_neg = int(err_left[- err_bin_idx])

            x_left_neg = int(x_left + err_left_neg)
            x_left_pos = int(x_left + err_left_pos)

            x_right_neg = int(x_right + err_right_neg)
            x_right_pos = int(x_right + err_right_pos)

            # Write lens between left and right surface below optical axis
            eps_map[x_left_neg:x_right_neg+1, y_negative] *= eps_line
            
            # So that the center line is not affected twice:
            if y_res != 0:
                # Write lens between left and right surface above optical axis
                eps_map[x_left_pos:x_right_pos+1, y_positive] *= eps_line
            
            # Write multi-layer AR coating on left surface
            if AR_left_layers is not None:
                # Start position is directly at the lens surface
                start_pos_neg = x_left_neg - delam_neg_L
                start_pos_pos = x_left_pos - delam_pos_L
                
                # Apply each layer, moving outward from the lens surface
                for i, (layer_thick, material) in enumerate(zip(AR_left_layers, AR_left_materials)):
                    AR_thick = np.int64(np.around(layer_thick*res))
                    
                    # Below optical axis
                    eps_map[start_pos_neg - AR_thick:start_pos_neg, y_negative] *= material
                    
                    # Above optical axis (if not on axis)
                    if y_res != 0:
                        eps_map[start_pos_pos - AR_thick:start_pos_pos, y_positive] *= material
                    
                    # Move starting position outward for next layer
                    start_pos_neg -= AR_thick
                    start_pos_pos -= AR_thick
            
            # Write multi-layer AR coating on right surface                    
            if AR_right_layers is not None:
                # Start position is directly at the lens surface
                start_pos_neg = x_right_neg + 1 + delam_neg_R
                start_pos_pos = x_right_pos + 1 + delam_pos_R
                
                # Apply each layer, moving outward from the lens surface
                for i, (layer_thick, material) in enumerate(zip(AR_right_layers, AR_right_materials)):
                    AR_thick = np.int64(np.around(layer_thick*res))
                    
                    # Below optical axis
                    eps_map[start_pos_neg:start_pos_neg + AR_thick, y_negative] *= material
                    
                    # Above optical axis (if not on axis)
                    if y_res != 0:
                        eps_map[start_pos_pos:start_pos_pos + AR_thick, y_positive] *= material
                    
                    # Move starting position outward for next layer
                    start_pos_neg += AR_thick
                    start_pos_pos += AR_thick

    # Example usage:
    def assemble_with_multi_arc(self):
        """
        Assembling the lens object with multi-layer AR coatings
        
        Parameters:
        left_layers: list of float
            Thicknesses of each AR coating layer on the left surface
        left_materials: list of float
            Permittivity values for each AR coating layer on the left surface
        right_layers: list of float
            Thicknesses of each AR coating layer on the right surface
        right_materials: list of float
            Permittivity values for each AR coating layer on the right surface
        """
        self.write_lens_nARC(self, self.permittivity_map, self.res,
                        AR_left_layers=self.left_layers,
                        AR_left_materials=self.left_materials,
                        AR_right_layers=self.right_layers,
                        AR_right_materials=self.right_materials)
        return self.permittivity_map
    
    # # Example usage:
    # # Create a lens with multi-layer AR coatings
    # lens = AsphericLens(diameter=10, r1=20, r2=-20, thick=5, mpsat_sim=sim_obj)

    # # Define the AR coating layers
    # left_layers = [0.25, 0.3, 0.15]  # Three layers with different thicknesses
    # left_materials = [1.4, 1.2, 1.3]  # Corresponding refractive indices squared

    # right_layers = [0.2, 0.25]  # Two layers on the right surface
    # right_materials = [1.3, 1.2]  # Corresponding refractive indices squared

    # # Apply the multi-layer AR coatings
    # lens.assemble_with_multi_arc(
    #     left_layers=left_layers, 
    #     left_materials=left_materials,
    #     right_layers=right_layers, 
    #     right_materials=right_materials
    # )

    #*The below functions are helpful for adding stepped pyramid ARC in the 
    #*lens permittivity map

    def meep_block(self,
                size, 
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

        # Return the block with the given parameters
        return mp.Block(size=size,
                        center=center,
                        material=material,
                        e1=e1,
                        e2=e2,
                        e3=e3,
                        **kwargs)

    def generate_discretized_points(self, center, angle, distance, num_points=1):
        """
        Generates discretized points along a line at a specified angle from the center.

        Parameters
        ----------
        center : tuple
            Center coordinates (x, y) from which the points are generated.
        angle : float
            Angle in degrees (w.r.t X axis) at which the points are generated.
        distance : float
            Distance from the center to the points.
        num_points : int (optional)
            Number of points to generate along the line (default: 1).

        Returns
        -------
        list of tuples
            List of generated points as (x, y) coordinates.
        """
        angle_rad = np.radians(angle)
        if num_points == 1:
            #! I did some debugging here, that I don't know :( --> But it works LoL
            x = center[0] + distance * np.sin(angle_rad)
            y = center[1] + distance * np.cos(angle_rad)
            points = [(x, y)]
            return points
        else:
            # Generate points from -distance/2 to distance/2
            distance_arr = np.linspace(-distance/2, distance/2, num_points)
            points = [(center[0] + d * np.cos(angle_rad), 
                    center[1] + d * np.sin(angle_rad)) for d in distance_arr]

            return points


    def stepped_pyramid_geometery(self,
                                nlayers,
                                base_bottom_edge_center,
                                pitch,
                                depth_arr,
                                kerf_arr,
                                width_arr=None,
                                material=mp.Medium(index=1.45),
                                rot_axis = 'z',
                                angle=0):
        
        """
        Function to generate the geometry of a stepped pyramid with specified parameters.

        Parameters:
        nlayers (int): Number of layers in the stepped pyramid.
        center (tuple): Base center coordinates (x,y) on which the pyramids will be attached.
        pitch (float): Pitch of the pyramid.
        depth_arr (list): List of depths for each layer, starting from the bottom layer.
        kerf_arr (list): List of kerfs for each layer, starting from the bottom layer.
        width_arr (list, optional): List of widths for each layer. If not provided, it will be calculated as pitch - depth for each layer.
        material (mp.Medium): Material of the blocks (default: mp.Medium(index=1.45)).
        rot_axis (str): Axis about which the blocks are rotated ('x', 'y', or 'z') (default: 'x').
        angle (float): Angle by which the blocks are rotated w.r.t the x-axis (default: 0).
                    Unit: degrees.

        Returns:
        A list of MEEP block objects representing the geometry of a single stepped pyramid geometry.
        """
        # Initialize the list to hold the blocks
        blocks = []

        if width_arr is None:
            print("Width array is not provided. It will be calculated as pitch - kerf for each layer.")
            width_arr = [pitch - kerf for kerf in kerf_arr]

        # Loop through each layer to create the blocks
        # Epty array to hold the centers of the base layer
        base_center = []

        # Initialise the depth to calculate d1 + d2 + ... + dn
        # This will be used to calculate the center of each layer after the base layer
        # The base layer is at depth = depth_arr[0]/2
        depth_from_previous_layers_centre = 0
        for i in range(nlayers):
            # Calculate the center of the base layer of the pyramid
            if i == 0:
                base_layer_center = self.generate_discretized_points(center= base_bottom_edge_center,
                                                                angle= angle,
                                                                distance= depth_arr[i]/2,
                                                                num_points=1)
                base_center.append(base_layer_center[0])
                # Create the block for the base layer
                base_layer_block = self.meep_block(size=mp.Vector3(width_arr[i], depth_arr[i], 0),
                                            center= mp.Vector3(base_layer_center[0][0], base_layer_center[0][1], 0),
                                            material=material,
                                            angle=angle,
                                            rot_axis=rot_axis)

                blocks.append(base_layer_block)


            # For rest of the layers, calculate the center based on the base layer's center
            # and add the depth of the current layer to the previous depth (basically cumulative depth from the base layer)
            else:
                
                depth_from_previous_layers_centre += depth_arr[i-1]/2 + depth_arr[i]/2
                other_layers_center = self.generate_discretized_points(center= base_center[0],
                                                                angle= angle,
                                                                distance= depth_from_previous_layers_centre,
                                                                num_points=1)
                
                # Create the block for the other layers
                other_layers_block = self.meep_block(size=mp.Vector3(width_arr[i], depth_arr[i], 0),
                                                center= mp.Vector3(other_layers_center[0][0], other_layers_center[0][1], 0),
                                                material=material,
                                                angle=angle,
                                                rot_axis=rot_axis)

                blocks.append(other_layers_block)

        return blocks

    
    
    """
    Instead of adding the stepped pyramid ARC to the  permittivity map of the lens,
    we will instead return the MEEP bllock objects representing the stepped pyramid ARC coating.
    These blocks can then be added into the geometry list of the MEEP simulation.
    This allows for more flexibility in how the ARC coating is applied and visualized. 
    ##! MOST IMPORTANT: CURRENT STEPPED PYRAMID ARC CANNOT BE APPLIED TO THE LENS SAGS 
    ##! WITH DELAMINATION AND SURFACE ERRORS ###!
    """

    def extract_lens_surface_coordinates(self, comp, res):
        """
        Extracts the x,y coordinates of the left and right lens surfaces,
        returns them in Meep units centered at (0,0).
        """
        mid_y = np.int64(self.size_y * res / 2)
        thick = comp.thick * res
        radius = np.int64(comp.diameter * res / 2)

        left_surface_coords = []
        right_surface_coords = []

        if comp.surf_err_scale != 0:
            nb_bins = int(comp.diameter / comp.surf_err_width)
            err_left = np.around(np.random.normal(scale=comp.surf_err_scale * res, size=nb_bins))
            err_right = np.around(np.random.normal(scale=comp.surf_err_scale * res, size=nb_bins))
        else:
            nb_bins = int(comp.diameter / comp.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)

        for y_res in range(radius):
            # print(f"Processing y_res: {y_res}")
            x_left = np.int64(np.around((
                comp.left_surface(y_res / res) + self.dpml +
                comp.x - comp.cust_def((y_res + mid_y) / res)) * res))
            x_right = np.int64(np.around((
                comp.right_surface(y_res / res) +
                comp.x + self.dpml -
                comp.cust_def((y_res + mid_y) / res)) * res + thick))

            y_positive = int(self.dpml * res + mid_y + y_res)
            y_negative = int(self.dpml * res + mid_y - y_res)

            err_bin_idx = int(np.around(y_res / res / comp.surf_err_width))
            err_left_pos = int(err_left[err_bin_idx])
            err_left_neg = int(err_left[-err_bin_idx])
            err_right_pos = int(err_right[err_bin_idx])
            err_right_neg = int(err_right[-err_bin_idx])

            x_left_neg = int(x_left + err_left_neg)
            x_left_pos = int(x_left + err_left_pos)
            x_right_neg = int(x_right + err_right_neg)
            x_right_pos = int(x_right + err_right_pos)

            # First convert from array indices to physical coordinates
            x_left_neg_phys = x_left_neg / res
            x_left_pos_phys = x_left_pos / res  
            x_right_neg_phys = x_right_neg / res
            x_right_pos_phys = x_right_pos / res
            y_negative_phys = y_negative / res
            y_positive_phys = y_positive / res

            # Then center at (0,0) by subtracting half the total size (including PML)
            total_size_x = self.size_x + 2 * self.dpml
            total_size_y = self.size_y + 2 * self.dpml

            x_left_neg_meep = x_left_neg_phys - (total_size_x / 2)
            x_left_pos_meep = x_left_pos_phys - (total_size_x / 2)
            x_right_neg_meep = x_right_neg_phys - (total_size_x / 2)
            x_right_pos_meep = x_right_pos_phys - (total_size_x / 2)
            y_negative_meep = y_negative_phys - (total_size_y / 2)
            y_positive_meep = y_positive_phys - (total_size_y / 2)

            # # Convert to Meep units and center at (0,0)
            # x_left_neg_meep = (x_left_neg / res) - ((self.size_x) / 2) - self.dpml/4 #! managed somehow
            # x_left_pos_meep = (x_left_pos / res) - ((self.size_x) / 2)  - self.dpml/4 #! managed somehow
            # x_right_neg_meep = (x_right_neg / res) - ((self.size_x) / 2)  - self.dpml/4 #! managed somehow
            # x_right_pos_meep = (x_right_pos / res) - ((self.size_x) / 2)  - self.dpml/4#! managed somehow
            # y_negative_meep = (y_negative / res) - ((self.size_y) / 2)  - self.dpml/2 #! working 
            # y_positive_meep = (y_positive / res) - ((self.size_y) / 2)  - self.dpml/2 #! working

            #if not np.isclose(np.around(y_negative_meep, 2), 0.0):
            left_surface_coords.append((x_left_neg_meep, y_negative_meep))
            right_surface_coords.append((x_right_neg_meep, y_negative_meep))
            #if not np.isclose(np.around(y_positive_meep, 2), 0.0):
            left_surface_coords.append((x_left_pos_meep, y_positive_meep))
            right_surface_coords.append((x_right_pos_meep, y_positive_meep))


        return {
            'left_surface': left_surface_coords,
            'right_surface': right_surface_coords
        }


    def create_arc_blocks_vectorized(self,
                                     x_left_steps, x_right_steps, angle_left, angle_right, y_steps_left, y_steps_right, 
                                    arc_layer_pitch, arc_layer_depth, arc_layer_kerf, arc_layer_width, arc_material):
        """
        Vectorized creation of ARC stepped pyramid blocks for better performance
        """
        all_blocks = []
        
        # Create arrays for all left and right centers
        left_centers = np.column_stack((x_left_steps, y_steps_left))
        right_centers = np.column_stack((x_right_steps, y_steps_right))
        
        # Batch create left stepped pyramids
        for i, (center, angle) in enumerate(zip(left_centers, angle_left)):
            left_pyramid = self.stepped_pyramid_geometery(
                nlayers=self.step_ARC_nlayers,
                base_bottom_edge_center=tuple(center),
                pitch=arc_layer_pitch,
                depth_arr=arc_layer_depth,
                kerf_arr=arc_layer_kerf,
                width_arr=arc_layer_width,
                material=arc_material,
                rot_axis=self.step_ARC_rot_axis,
                angle=angle
            )

            all_blocks.extend(left_pyramid)
        
        # Batch create right stepped pyramids
        for i, (center, angle) in enumerate(zip(right_centers, angle_right)):
            right_pyramid = self.stepped_pyramid_geometery(
                nlayers=self.step_ARC_nlayers,
                base_bottom_edge_center=tuple(center),
                pitch=arc_layer_pitch,
                depth_arr=arc_layer_depth,
                kerf_arr=arc_layer_kerf,
                width_arr=arc_layer_width,
                material=arc_material,
                rot_axis=self.step_ARC_rot_axis,
                angle=angle
            )


            all_blocks.extend(right_pyramid)
        
        return all_blocks

    # Sort the coordinates by y-values and remove duplicates
    def prepare_spline_data(self, x_coords, y_coords):
        """
        Prepare data for spline interpolation by sorting and removing duplicates
        """
        # Combine and sort by y-coordinates
        combined = list(zip(y_coords, x_coords))
        combined.sort(key=lambda item: item[0])  # Sort by y
        
        # Remove duplicates (keep first occurrence)
        unique_data = []
        prev_y = None
        for y, x in combined:
            if prev_y is None or not np.isclose(y, prev_y):
                unique_data.append((y, x))
                prev_y = y
        
        if len(unique_data) < 2:
            raise ValueError("Not enough unique points for interpolation")
        
        y_unique, x_unique = zip(*unique_data)
        return np.array(x_unique), np.array(y_unique)

    def write_lens_with_stepped_pyramid_ARC_v2(self, comp):
        """
        In this version, we will assume that the lens surfaces are:
        - Centered at (0,0), instead of at self.x and self.y
        - We will first generate the lens sags for left and right surfaces centered at (0,0)
        - Then self.x, self.y is given in the 0 to x,y coordinate system; convert this to -x/2 - x/2 and -y/2 - y/2 coordinates
        - Do a coordinate shift for the lens sags in the (-x/2, x/2) and (-y/2, y/2) coordinate system 
          (basically add the self.x and self.y coordinates in the (-x/2, x/2) and (-y/2, y/2) coordinate system)
        - Expectation: The lens surfaces will be centered at the required physcial coordinates of the system, 
          and the ARC coating will be applied on the lens surfaces.
        """
        # Check if all required parameters for stepped pyramid ARC are provided
        if self.step_ARC_nlayers is None or self.step_ARC_pitch is None or \
            self.step_ARC_kerf is None or self.step_ARC_depth is None or \
            self.step_ARC_material is None:
            raise ValueError("All stepped pyramid ARC parameters must be provided.")
        
        # Calculate the width if not provided
        if self.step_ARC_width is None:
            print("Width array is not provided. It will be calculated as pitch - kerf for each stepped pyramid ARC layer.")
            self.step_ARC_width = [self.step_ARC_pitch - kerf for kerf in self.step_ARC_kerf]

        def even_asphere_lens_eqn(y, r, k, A2=0, A3=0, A4=0):
            # y =y/10
            # r = r/10
            return (y**2/r) / (1 + np.sqrt(1 - (1 + k)*y**2/r**2)) + A2 * y**2 + A3 * y**4 + A4 * y**6

        # Defining the y array
        y_arc_steps = np.arange(-self.diameter/2 + self.dpml - self.step_ARC_pitch/2, self.diameter/2 + self.step_ARC_pitch/2, self.step_ARC_pitch)
        # start = -self.diameter / 2
        # stop = self.diameter / 2
        # step = self.step_ARC_pitch

        # num_points = int(np.floor((stop - start) / step)) + 1
        # y_arc_steps = np.linspace(start, stop, num_points)

        # Extracting the left and right surface coordinates using the lens sag equations
        x_left_arc_steps = even_asphere_lens_eqn(y_arc_steps, self.r1, self.c1, self.a1_coeffs[0], self.a1_coeffs[1], self.a1_coeffs[2]) + self.x - self.size_x/2 - comp.cust_def(y_arc_steps) + self.dpml + self.step_ARC_offset[0] #! Note: we need to check with cust_def
        x_right_arc_steps = even_asphere_lens_eqn(y_arc_steps, self.r2, self.c2, self.a2_coeffs[0], self.a2_coeffs[1], self.a2_coeffs[2])  + self.x + self.thick - self.size_x/2 - comp.cust_def(y_arc_steps) + self.dpml + self.step_ARC_offset[1] #! Note: we need to check with cust_def

        # Calculate the slope using scipy of each point by considering the adjacent points
        # from scipy.ndimage import gaussian_filter1d
        slope_left = np.gradient(x_left_arc_steps, y_arc_steps)
        slope_right = np.gradient(x_right_arc_steps, y_arc_steps)

        # # Calculate the perpendicular angle of the slope in radians
        angle_left = np.rad2deg(np.arctan(slope_left))
        angle_left = -angle_left - 90  # Adjusting the angle to be perpendicular on the left lens surface
        angle_right = np.rad2deg(np.arctan(slope_right))
        angle_right = -angle_right + 90  # Adjusting the angle to be perpendicular on the right lens surface

        # Create the ARC blocks for both left and right surfaces
        all_blocks = self.create_arc_blocks_vectorized(
            x_left_steps=x_left_arc_steps,
            x_right_steps=x_right_arc_steps,
            angle_left=angle_left,
            angle_right=angle_right,
            y_steps_left=y_arc_steps,
            y_steps_right=y_arc_steps,
            arc_layer_pitch=self.step_ARC_pitch,
            arc_layer_depth=self.step_ARC_depth,
            arc_layer_kerf=self.step_ARC_kerf,
            arc_layer_width=self.step_ARC_width,
            arc_material=self.step_ARC_material
        )

        return all_blocks
        



    def write_lens_with_stepped_pyramid_ARC(self):
        """
        Writes the lens surfaces with stepped pyramid ARC coating.
        This method generates the stepped pyramid ARC coating on the lens surfaces
        

        Returns:
        -------
        Adds the stepped pyramid ARC coating to the permittivity map of the lens.
        """
        from scipy.interpolate import UnivariateSpline
        if self.step_ARC_nlayers is None or self.step_ARC_pitch is None or \
            self.step_ARC_kerf is None or self.step_ARC_depth is None or \
            self.step_ARC_material is None:
            raise ValueError("All stepped pyramid ARC parameters must be provided.")
        
        # Calculating the separation between the base layers of the ARC by considering the kerf and width
        if self.step_ARC_width is None:
            # If width is not provided, calculate it as pitch - kerf
            print("Width array is not provided. It will be calculated as pitch - kerf for each stepped pyramid ARC layer.")
            self.step_ARC_width = [self.step_ARC_pitch - kerf for kerf in self.step_ARC_kerf]
        #!=====
        left_surface_coords = self.extract_lens_surface_coordinates(self, self.res)['left_surface']
        right_surface_coords = self.extract_lens_surface_coordinates(self, self.res)['right_surface']

        x_left = np.array([coord[0] for coord in left_surface_coords])
        y_left = np.array([coord[1] for coord in left_surface_coords])
        x_right = np.array([coord[0] for coord in right_surface_coords])
        y_right = np.array([coord[1] for coord in right_surface_coords])

        # Interpolate the left and right surface coordinates to get evenly spaced points
        # from scipy.interpolate import interp1d
        # interp_left = interp1d(y_left, x_left, bounds_error=False, fill_value="extrapolate")
        # interp_right = interp1d(y_right, x_right, bounds_error=False, fill_value="extrapolate")
        # Prepare data for spline interpolation
        try:
            x_left_clean, y_left_clean = self.prepare_spline_data(x_left, y_left)
            x_right_clean, y_right_clean = self.prepare_spline_data(x_right, y_right)
            
            # Create splines with cleaned data
            interp_left = UnivariateSpline(y_left_clean, x_left_clean, s=0)
            interp_right = UnivariateSpline(y_right_clean, x_right_clean, s=0)
            
        except ValueError as e:
            print(f"Spline interpolation failed: {e}")
            print("Falling back to linear interpolation...")
            
            # Fallback to linear interpolation
            from scipy.interpolate import interp1d
            interp_left = interp1d(y_left, x_left, bounds_error=False, fill_value="extrapolate")
            interp_right = interp1d(y_right, x_right, bounds_error=False, fill_value="extrapolate")

        # Extracting the N-1 top and bottom y coordinates of the left and right surfaces
        y_min_left = np.min(y_left)
        y_max_left = np.max(y_left)
        y_min_right = np.min(y_right)
        y_max_right = np.max(y_right)

        # Generate evenly spaced points along the y-axis for the left and right surfaces according to the pitch
        y_arc_steps_left = np.arange(y_min_left, y_max_left, self.step_ARC_pitch)
        y_arc_steps_right = np.arange(y_min_right, y_max_right, self.step_ARC_pitch)

        # Calculate the x-coordinates for the left edge of the ARC layers
        x_left_arc_steps = interp_left(y_arc_steps_left)
        # Calculate the x-coordinates for the right edge of the ARC layers
        x_right_arc_steps = interp_right(y_arc_steps_right)

        # # Calculate the slope using scipy of each point by considering the adjacent points
        # # from scipy.ndimage import gaussian_filter1d
        slope_left = np.gradient(x_left_arc_steps, y_arc_steps_left)
        slope_right = np.gradient(x_right_arc_steps, y_arc_steps_right)

        # # Calculate the perpendicular angle of the slope in radians
        angle_left = np.rad2deg(np.arctan(slope_left))
        angle_left = -angle_left - 90  # Adjusting the angle to be perpendicular on the left lens surface
        angle_right = np.rad2deg(np.arctan(slope_right))
        angle_right = -angle_right + 90  # Adjusting the angle to be perpendicular on the right lens surface

        # Create the ARC blocks for both left and right surfaces
        all_blocks = self.create_arc_blocks_vectorized(
            x_left_steps=x_left_arc_steps,
            x_right_steps=x_right_arc_steps,
            angle_left=angle_left,
            angle_right=angle_right,
            y_steps_left=y_arc_steps_left,
            y_steps_right=y_arc_steps_right,
            arc_layer_pitch=self.step_ARC_pitch,
            arc_layer_depth=self.step_ARC_depth,
            arc_layer_kerf=self.step_ARC_kerf,
            arc_layer_width=self.step_ARC_width,
            arc_material=self.step_ARC_material
        )

        return all_blocks

    
    def assemble_with_stepped_pyramid_ARC(self):
        """
        Assembling the lens object with stepped pyramid ARC coating.
        This method generates the stepped pyramid ARC coating on the lens surfaces
        """
        # First assemble the lens itself
        self.permitivitty_map = self.assemble()
        
        # Then generate the stepped pyramid blocks for ARC coating
        #! self.stepped_pyramid_blocks = self.write_lens_with_stepped_pyramid_ARC()
        self.stepped_pyramid_blocks = self.write_lens_with_stepped_pyramid_ARC_v2(self)

        # Return both the permittivity map and the stepped pyramid blocks
        return self.permittivity_map, self.stepped_pyramid_blocks
            

# ----------------------------------- Feedhorn ----------------------------------- #

class FeedHorn(object):
    """
    Class defining an FeedHorn from a txt file containing the geometry information about
    the Horn in r vs z.
    """
    
    def __init__(self,
                 mpsat_sim,
                 eps,
                 focal_plane_x,
                 focal_plane_y_range,
                 feedhorn_y_range,
                 # Feedhorn params
                 txt_file,
                 t_m,
                 t_f,
                 w2,
                 thick_x,
                 savepath,
                 central_metal_thickness = 0,
                 plot = False,
                 eps_pec = -1e-10,
                 eps_air = 1
                 ):
    
        """
        Arguments
        ---------
        mpsat_sim: object
            MEEPSAT object produced from sim_init() in simulation_2D.py
        
        eps: np.array
            Dielectric map of the other components in the system 
        
        focal_plane_x: float
            X-coordinate of the focal plane
        
        focal_plane_y_range: tuple
            Y-coordinate range of the focal plane

        feedhorn_y_range: tuple
            Y-coordinate range of the feedhorn distribution on the focal plane

        txt_file: str
            path to the text file containing the geometry information about
            the Horn in r vs z.

        t_m: float
            Thickness of the metal gap of the two consecutive apertures 

        t_f: float
            Gap between the centers of the two feedhorns

        w2: float
            Feedhorn's aperture width

        thick_x: float
            Total extent of the feedhorn in the X-axis

        savepath: str
            savepath for the generated plots and the data files 

        central_metal_thickness: float
            The thickness of the central metal layer separating feedhorn arrays of different wafers

        plot: bool
            Whether to generate plots 

        eps_pec: float
            Permittivity of the perfect electric conductor (PEC)

        eps_air: float
            Permittivity of air

        """

        self.mpsat_sim = mpsat_sim
        self.eps = eps
        self.focal_plane_y_range = focal_plane_y_range
        self.feedhorn_y_range = feedhorn_y_range

        # if self.focal_plane_y_range < self.feedhorn_y_range:
        #     raise(ValueError("focal_plane_y_range must be greater than or equal to feedhorn_y_range"))

        self.txt_file = txt_file
        self.t_m = t_m
        self.t_f = t_f
        self.w2 = w2
        self.thick_x = thick_x
        self.central_metal_thickness = central_metal_thickness
        self.savepath = savepath
        self.plot = plot
        self.eps_pec = eps_pec
        self.eps_air = eps_air

        self.focal_plane_x = focal_plane_x + thick_x #! Because we want the forebaffles opening aperture at the position of the focal plane

        # Extract some parameters from mpsat_sim
        self.sx = mpsat_sim.cell_size[0]
        self.sy = mpsat_sim.cell_size[1]
        self.res = mpsat_sim.resolution
        
    
    def load_txt_dat(self):
        import pandas as pd
        self.data = pd.read_csv(self.txt_file, sep=r'\s+')  # Fixed regex warning
        self.data['r_pos'] = self.data['r']*10
        self.data['r_neg'] = -self.data['r_pos']
        self.cumulative_z = np.cumsum(self.data['z']*10)

        if self.plot == True:
            plt.figure(figsize=(10, 6))
            plt.plot(self.cumulative_z, self.data['r_pos'])
            plt.plot(self.cumulative_z, self.data['r_neg'])
            plt.xlabel('z (mm)')
            plt.ylabel('r (mm)')
            plt.title('z vs r')
            plt.grid(True)
            plt.savefig(self.savepath + 'step1_z_column_plot.png')
            plt.close()

        return self.data
    
    
    
    def fit_spline_to_dat(self, s_factor=0, no_points=1000):
        from scipy.interpolate import UnivariateSpline
        r_pos_spline = UnivariateSpline(self.cumulative_z, self.data['r_pos'], s=s_factor) 
        r_neg_spline = UnivariateSpline(self.cumulative_z, self.data['r_neg'], s=s_factor)
        
        # Fit the spline
        z_new = np.linspace(self.cumulative_z.min(), self.cumulative_z.max(), no_points)
        r_pos_fitted = r_pos_spline(z_new)
        r_neg_fitted = r_neg_spline(z_new)

        if self.plot == True:
            plt.figure(figsize=(10, 6))
            plt.plot(z_new, r_pos_fitted, label='Fitted r_pos')
            plt.plot(z_new, r_neg_fitted, label='Fitted r_neg')
            plt.xlabel('z (mm)')
            plt.ylabel('r (mm)')
            plt.title('Fitted splines over original data')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.savepath + 'step2_fitted_splines_plot.png')
            plt.close()

        return r_pos_spline, r_neg_spline


    
    def create_coordinate_grids(self):
        """Create x, y coordinate arrays for the grid"""
        # Match the epsilon map dimensions which have +1
        self.x = np.linspace(-self.sx/2, self.sx/2, int(self.sx * self.res) + 1)
        self.y = np.linspace(-self.sy/2, self.sy/2, int(self.sy * self.res) + 1)
        return self.x, self.y

    
    def define_focal_plane_axis(self):
        """Create the focal plane axis array"""
        self.focal_plane_axis = np.linspace(
            self.focal_plane_y_range[0], 
            self.focal_plane_y_range[1], 
            int((self.focal_plane_y_range[1] - self.focal_plane_y_range[0]) * self.res)
        )
        return self.focal_plane_axis

    # 
    # def fill_pec_region(self):
    #     """Fill the focal plane region with PEC - VECTORIZED"""
    #     x, y = self.create_coordinate_grids()
        
    #     # Create meshgrid - use indexing='ij' to match epsilon array dimensions
    #     # epsilon array is (len(x), len(y)), so X varies along axis 0, Y along axis 1
    #     X, Y = np.meshgrid(x, y, indexing='ij')
        
    #     # Create boolean mask for PEC region
    #     mask_pec = ((X <= self.focal_plane_x) & 
    #                 (X >= (self.focal_plane_x - self.thick_x)) & 
    #                 (Y >= self.focal_plane_y_range[0]) & 
    #                 (Y <= self.focal_plane_y_range[1]))
        
    #     # Apply the mask
    #     self.eps[mask_pec] = self.eps_pec


    
    def fill_pec_region(self):
        """Fill the focal plane region with PEC - CHUNKED"""
        x, y = self.create_coordinate_grids()
        
        # Define chunk size (e.g., 1000 rows at a time)
        chunk_size = 1000
        nx = len(x)
        
        for i in range(0, nx, chunk_size):
            i_end = min(i + chunk_size, nx)
            
            # Create meshgrid only for this chunk
            X_chunk, Y_chunk = np.meshgrid(x[i:i_end], y, indexing='ij')
            
            # Create boolean mask for this chunk
            mask_pec_chunk = ((X_chunk <= self.focal_plane_x) & 
                            (X_chunk >= (self.focal_plane_x - self.thick_x)) & 
                            (Y_chunk >= self.focal_plane_y_range[0]) & 
                            (Y_chunk <= self.focal_plane_y_range[1]))
            
            # Apply the mask to this chunk
            self.eps[i:i_end][mask_pec_chunk] = self.eps_pec
            
            # Free memory
            del X_chunk, Y_chunk, mask_pec_chunk

    
    def calculate_feedhorn_centers(self):
        """Calculate feedhorn center positions"""
        n_feedhorns_positive = int(np.floor(self.feedhorn_y_range[1] / self.t_f)) + 1
        n_feedhorns_negative = int(np.floor(abs(self.feedhorn_y_range[0]) / self.t_f))
        
        if self.central_metal_thickness == 0:
            feedhorn_centers_positive = np.arange(0, n_feedhorns_positive) * self.t_f
            feedhorn_centers_negative = -np.arange(1, n_feedhorns_negative + 1) * self.t_f
        else:
            feedhorn_centers_positive = np.arange(self.central_metal_thickness/2, n_feedhorns_positive) * self.t_f
            feedhorn_centers_negative = -np.arange(self.central_metal_thickness/2, n_feedhorns_negative + 1) * self.t_f

        self.feedhorn_centers = np.sort(np.concatenate([feedhorn_centers_negative, feedhorn_centers_positive]))
        
        return self.feedhorn_centers

    
    def fill_feedhorn_profiles(self, r_pos_spline, r_neg_spline):
        """Fill air inside feedhorns using spline functions - CHUNKED"""
        x, y = self.create_coordinate_grids()
        
        # Define chunk size
        chunk_size = 1000
        nx = len(x)
        
        for i in range(0, nx, chunk_size):
            i_end = min(i + chunk_size, nx)
            
            # Create meshgrid only for this chunk
            X_chunk, Y_chunk = np.meshgrid(x[i:i_end], y, indexing='ij')
            
            # Mask for x within feedhorn extent (for this chunk)
            mask_x_chunk = ((X_chunk <= self.focal_plane_x) & 
                            (X_chunk >= (self.focal_plane_x - self.cumulative_z.max())))
            
            # Calculate z position for this chunk
            z_pos_chunk = self.focal_plane_x - X_chunk
            
            # Mask for valid z positions
            mask_z_chunk = (z_pos_chunk >= 0) & (z_pos_chunk <= self.cumulative_z.max())
            
            # Combine masks
            mask_region_chunk = mask_x_chunk & mask_z_chunk
            
            # Get the radial bounds at each z position (vectorized spline evaluation)
            z_pos_valid = z_pos_chunk[mask_region_chunk]
            r_upper_valid = r_pos_spline(z_pos_valid)
            r_lower_valid = r_neg_spline(z_pos_valid)
            
            # For each feedhorn center, check if points are inside
            for centre in self.feedhorn_centers:
                # Calculate y distance from feedhorn center for this chunk
                y_dist_chunk = Y_chunk - centre
                
                # Create temporary arrays for r_upper and r_lower for this chunk
                r_upper_grid = np.full_like(X_chunk, np.nan)
                r_lower_grid = np.full_like(X_chunk, np.nan)
                
                # Fill in the valid regions
                r_upper_grid[mask_region_chunk] = r_upper_valid
                r_lower_grid[mask_region_chunk] = r_lower_valid
                
                # Check if points are inside this feedhorn
                mask_feedhorn_chunk = (mask_region_chunk & 
                                    (y_dist_chunk >= r_lower_grid) & 
                                    (y_dist_chunk <= r_upper_grid))
                
                # Fill with air in the epsilon array
                self.eps[i:i_end][mask_feedhorn_chunk] = self.eps_air
                
                # Free memory inside the loop after processing each feedhorn
                del r_upper_grid, r_lower_grid, y_dist_chunk, mask_feedhorn_chunk
            
            # Free memory after each chunk (only variables created outside the feedhorn loop)
            del X_chunk, Y_chunk, z_pos_chunk, mask_x_chunk, mask_z_chunk
            del mask_region_chunk, z_pos_valid, r_upper_valid, r_lower_valid

    
    def plot_focal_plane(self):
        """Plot the simulation grid with focal plane axis"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        focal_plane_axis = self.define_focal_plane_axis()
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, Y, np.ones_like(X), cmap='gray', alpha=0.3, shading='auto')
        plt.plot(np.full_like(focal_plane_axis, self.focal_plane_x), focal_plane_axis, 
                'r-', linewidth=2, label='Focal plane axis')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with Focal Plane')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step3_focal_plane_plot.png')
        plt.close()

    
    def plot_pec_region(self):
        """Plot the simulation grid with PEC region filled"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        focal_plane_axis = self.define_focal_plane_axis()
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, y, self.eps.T, cmap='RdBu', shading='auto', vmin=-0.2, vmax=1)
        plt.plot(np.full_like(focal_plane_axis, self.focal_plane_x), focal_plane_axis, 
                'r-', linewidth=2, label='Focal plane axis')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with PEC Region')
        plt.colorbar(label='Epsilon')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step3b_focal_plane_with_PEC.png')
        plt.close()

    
    def plot_feedhorn_centers(self):
        """Plot feedhorn centers on the focal plane"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        focal_plane_axis = self.define_focal_plane_axis()
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, y, self.eps.T, cmap='RdBu', alpha=0.3, shading='auto', vmin=-0.2, vmax=1)
        plt.plot(np.full_like(focal_plane_axis, self.focal_plane_x), focal_plane_axis, 
                'r-', linewidth=2, label='Focal plane axis')
        
        # Draw each feedhorn as a circle
        for center in self.feedhorn_centers:
            circle = plt.Circle((self.focal_plane_x, center), self.w2/2, 
                            color='blue', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.plot(self.focal_plane_x, center, 'bo', markersize=5)
        
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with Focal Plane and Feedhorns')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step4_focal_plane_with_feedhorns_plot.png')
        plt.close()

    
    def plot_final_geometry(self):
        """Plot the final feedhorn geometry with all profiles filled"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        
        plt.figure(figsize=(12, 10))
        plt.pcolormesh(x, y, self.eps.T, cmap='RdBu', shading='auto', vmin=-0.2, vmax=1)
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with Feedhorns (Air-filled)')
        plt.colorbar(label='Epsilon')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step5_feedhorns_with_profiles.png')
        plt.close()

    
    def add_absorbers_to_extra_PEC(self):
        # # Calculate the remaining length remaining on the PEC layer in the focal plane
        # extra_layer_negative_side = self.focal_plane_y_range[0] - self.feedhorn_y_range[0]
        # extra_layer_positive_side = self.focal_plane_y_range[1] - self.feedhorn_y_range[1]

        # absorber_range_y_neg = [self.feedhorn_y_range[0], self.feedhorn_y_range[0]-extra_layer_negative_side]
        # absorber_range_y_pos = [self.feedhorn_y_range[1], self.feedhorn_y_range[1]+extra_layer_positive_side]

        # import meepsat.meep_geometry as comp_meep
        # absorbers_y_neg = comp_meep.PyramidalAbsorbers(self.mpsat_sim,
        #                                  base_width = 6,
        #                                  height = 9,
        #                                  n_layers = 70,
        #                                  top_width = 0.5,
        #                                  epsilon_real = 5.4,
        #                                  epsilon_imag = 0.8,
        #                                  freq = data["sources"]["source1"]["frequency"],
        #                                  add_substrate=True,
        #                                  substrate_thickness=7,#p,
        #                                  substrate_material=None, # If None, then it will be same as the absorber material
        #                                  substrate_extends_beyond_pyramids=False,
        #                                  substrate_extension=1,
        #                                  y_top_offset=-forebaffle_height +mpsat_sim.dpml*mpsat_sim.factor_dpml,# + 0.35,
        #                                  y_bottom_offset= +forebaffle_height-mpsat_sim.dpml*mpsat_sim.factor_dpml,# -0.35,
        #                                 #  num_pyramids = 150,
        #                                  x_coverage_start = -size_x/2 + cellx_sourcex_distance + sourcex_FB_vertex_distance + forebaffle_base,
        #                                  x_coverage_end = size_x/2 + 10,# - mpsat_sim.dpml*mpsat_sim.factor_dpml + 1,
        #                                  add_pec_backing = True,
        #                                  pec_thickness = forebaffle_height-7, # PEC thickness same as the forebaffle perpendicular height)
        #                                  pec_extends_beyond_substrate = False,
        #                                  pec_extension = 1, # pec extends beyond the substrate by 1 mm
        #                                  name = "absorbers"
        #                                 )
        pass

    
    def assemble(self):
        """Assemble the complete feedhorn geometry"""
        # Load and fit data
        self.load_txt_dat()
        
        # Plot step 3a: focal plane
        self.plot_focal_plane()
        
        # Get splines from fit_spline_to_dat
        r_pos_spline, r_neg_spline = self.fit_spline_to_dat()
        
        # Fill regions
        self.fill_pec_region()
        
        # Plot step 3b: PEC region
        self.plot_pec_region()
        
        self.calculate_feedhorn_centers()
        
        # Plot step 4: feedhorn centers
        self.plot_feedhorn_centers()
        
        self.fill_feedhorn_profiles(r_pos_spline, r_neg_spline)
        
        # if self.feedhorn_y_range != self.focal_plane_y_range:
        #     self.add_absorbers_to_extra_PEC()
        
        # Plot step 5: final geometry
        self.plot_final_geometry()
        
        return self.eps
    
# ----------------------------------- Polyexponential Mirror ----------------------------------- #
class Mirror(object):
    """
    One extended-polynomial mirror, from ZEMAX prescription into an epsilon map.

    The tilt is the CUMULATIVE rotation about the ZEMAX x-axis. Basically its 
    the sum of the tilt_x of every coordinate break preceding the surface, with the
    prescription's own sign. Use from_prescription() to have it taken from the
    surface table rather than typed in, so it cannot drift away from the centre
    computed from that same table.

    Attributes
    ----------
    permittivity_map : ndarray
        The epsilon map this mirror edits, indexed [x, y]. It is the same epsilon map
        shared with every other component in the simulation.
    res, dpml, size_x, size_y : float
        Cell geometry, from mpsat_sim when given. size_x/size_y exclude PML.
    origin_offset_mm : tuple
        Shared ZEMAX -> MEEP offset (see zemax_placement)
    centre_zemax_mm : tuple
        (y, z) surface centre in the global ZEMAX frame
    half_width_mm : float
        Half-width along the mirror's local y
    A_ij : ndarray
        Extended polynomial coefficients, A_ij[i][j] multiplying x^i y^j
    R_N, N : float, int
        Normalisation radius (mm) and polynomial degree
    ap_y : float
        Aperture decentre along local y (mm)
    tilt_deg : float
        Cumulative rotation about the ZEMAX x-axis (degrees)
    thickness_mm : float
        Body thickness, measured along the surface normal
    eps_value : float
        Permittivity written into the body
    optical_side_zemax_mm : tuple or None
        (y, z) of what this mirror reflects towards -- normally the next
        surface's centre. The material goes on the opposite side, so the
        reflecting face sits exactly on the prescription surface.
    """

    def __init__(self, eps, name, centre_zemax_mm, half_width_mm, A_ij, R_N,
                 N=7, ap_y=0.0, tilt_deg=0.0, thickness_mm=15.0, eps_value=12.0,
                 optical_side_zemax_mm=None, invert_sag=False,
                 samples_per_mm=10.0, origin_offset_mm=(0.0, 0.0),
                 mpsat_sim=None, size_x=None, size_y=None, resolution=None,
                 dpml=0.0):
        self.permittivity_map = eps   # ~ THIS IS THE EPSILON MAP (not ours)

        # Cell geometry: from the simulation object if there is one, exactly as
        # AsphericLens does it, otherwise from explicit arguments.
        if mpsat_sim is not None:
            self.res = mpsat_sim.resolution
            self.dpml = mpsat_sim.factor_dpml * mpsat_sim.dpml
            # 2 times because there's pml on both sides
            self.size_x = mpsat_sim.cell_size[0] - 2 * self.dpml
            self.size_y = mpsat_sim.cell_size[1] - 2 * self.dpml
        else:
            if size_x is None or size_y is None or resolution is None:
                raise ValueError('give either mpsat_sim, or all of size_x, '
                                 'size_y and resolution')
            self.res = float(resolution)
            self.dpml = float(dpml)
            self.size_x = float(size_x)
            self.size_y = float(size_y)
        self.mpsat_sim = mpsat_sim

        self.name = name
        self.centre_zemax_mm = (float(centre_zemax_mm[0]), float(centre_zemax_mm[1]))
        self.half_width_mm = float(half_width_mm)
        self.A_ij = np.asarray(A_ij, dtype=float)
        self.R_N = float(R_N)
        self.N = int(N)
        self.ap_y = float(ap_y)
        self.tilt_deg = float(tilt_deg)
        self.thickness_mm = float(thickness_mm)
        self.eps_value = float(eps_value)
        self.optical_side_zemax_mm = optical_side_zemax_mm
        self.invert_sag = bool(invert_sag)
        self.samples_per_mm = float(samples_per_mm)
        self.origin_offset_mm = (float(origin_offset_mm[0]),
                                 float(origin_offset_mm[1]))

        self._profile_zemax = None

    # --- helper ------------------------------------------------------
    # Defining a bunch of static methods (independent of the class methods) 
    # to help with the polynomial geometry of the mirror surface.
    
    @staticmethod
    def _ensure_dir(path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
                
    @staticmethod
    def extended_polynomial_terms(N=7):
        """
        ZEMAX Extended Polynomial parameter order, as term names.

        ZEMAX lists the coefficients by ascending total order, and within each
        total order s = i + j it goes x^s, x^(s-1)*y, ..., y^s:

            x1y0  x0y1  x2y0  x1y1  x0y2  x3y0  x2y1  x1y2  x0y3  ...

        Args:
            N: Degree of the polynomial (int)

        Returns:
            List of term names, in the order ZEMAX/Excel writes them
        """
        return [f'x{i}y{s - i}' for s in range(1, N + 1) for i in range(s, -1, -1)]

    @staticmethod
    def excel_to_aij(excel_values, N=7):
        """
        Convert a flat Excel/ZEMAX coefficient row to a 2D A_ij array.

        A_ij[i][j] multiplies x^i * y^j, and the input is read in the order given by
        extended_polynomial_terms(). Note the inner loop counts i *down* from s: the
        x^s term of each diagonal comes first, so counting i up would transpose the
        surface (the x-coefficients would be used as the y-coefficients).

        Args:
            excel_values: List of coefficients from Excel (35 values for N=7)
            N: Polynomial degree (default 7)

        Returns:
            (N+1) x (N+1) numpy array
        """
        A_ij = np.zeros((N + 1, N + 1))
        idx = 0

        # Iterate through diagonals (constant i+j sum)
        for s in range(1, N + 1):  # s = i + j, starting from 1 (skip x0y0)
            for i in range(s, -1, -1):
                j = s - i
                if j <= N:
                    A_ij[i][j] = excel_values[idx]
                    idx += 1

        return A_ij    

    @classmethod 
    def check_aij_ordering(cls, excel_values, A_ij, N=7, label=''):
        """
        Assert that A_ij holds every Excel coefficient at the index its name implies.

        Guards the one bug excel_to_aij can silently reintroduce: filling each
        i+j diagonal in the wrong direction, which swaps x and y.

        Args:
            excel_values: The flat coefficient row that produced A_ij
            A_ij: Output of excel_to_aij
            N: Polynomial degree
            label: Name used in the failure message (e.g. 'M1')
        """
        for k, name in enumerate(cls.extended_polynomial_terms(N)):
            i, j = int(name[1]), int(name[3])
            if A_ij[i][j] != excel_values[k]:
                raise AssertionError(
                    f'{label} A_ij ordering is wrong: Excel term {k} ({name}) = '
                    f'{excel_values[k]} but A_ij[{i}][{j}] = {A_ij[i][j]}')


    @staticmethod
    def compute_surface_centers(surfaces):
        """
        Compute the 3D centers of optical surfaces in a global coordinate system.
        
        This function processes a sequence of optical surfaces, tracking cumulative
        coordinate transformations (decenters and tilts) to compute the center position
        of each real surface in the global frame. It handles COORDBRK surfaces that
        define coordinate system changes and thickness offsets between surfaces.
        
        Parameters
        ----------
        surfaces : list of dict
            A list of surface dictionaries, where each dict can contain:
            - "type" (str): Surface type, either "COORDBRK" or a real surface type
            - "decenter_x" (float, optional): X-axis decentering in current frame (default: 0.0)
            - "decenter_y" (float, optional): Y-axis decentering in current frame (default: 0.0)
            - "tilt_x" (float, optional): Rotation about X-axis in degrees (default: 0.0)
            - "tilt_y" (float, optional): Rotation about Y-axis in degrees (default: 0.0)
            - "tilt_z" (float, optional): Rotation about Z-axis in degrees (default: 0.0)
            - "ap_x" (float, optional): Aperture X offset of surface center (default: 0.0)
            - "ap_y" (float, optional): Aperture Y offset of surface center (default: 0.0)
            - "thickness" (float, optional): Distance to next surface along Z-axis (default: 0.0)
        
        Returns
        -------
        list of tuple
            A list of tuples (index, center)
        """

        # Global state
        R_global = np.eye(3)
        p_global = np.zeros(3)

        centers = []

        for i, s in enumerate(surfaces):

            # --- 1. Apply COORDBRK ---
            if s["type"] == "COORDBRK":
                dx = s.get("decenter_x", 0.0)
                dy = s.get("decenter_y", 0.0)

                tx = np.deg2rad(s.get("tilt_x", 0.0))
                ty = np.deg2rad(s.get("tilt_y", 0.0))
                tz = np.deg2rad(s.get("tilt_z", 0.0))

                # Translation in current frame
                p_global = p_global + R_global @ np.array([dx, dy, 0.0])

                # Rotation update
                R_local = exf.rotation_matrix(tx, ty, tz)
                R_global = R_global @ R_local

            # --- 2. If real surface, compute center ---
            if s["type"] != "COORDBRK":
                apx = s.get("ap_x", 0.0)
                apy = s.get("ap_y", 0.0)

                center = p_global + R_global @ np.array([apx, apy, 0.0])
                centers.append((i, center))

            # --- 3. Apply thickness (always happens) ---
            t = s.get("thickness", 0.0)
            p_global = p_global + R_global @ np.array([0.0, 0.0, t])

        return centers


    @staticmethod
    def cumulative_tilts_x(surfaces):
        """
        Cumulative rotation about the X-axis at each real (non-COORDBRK) surface.

        This is the angle mirror_profile_mm() needs, taken straight from the
        prescription so its sign can never drift away from the surface centers
        computed by compute_surface_centers().

        Parameters
        ----------
        surfaces : list of dict
            Same surface list given to compute_surface_centers()

        Returns
        -------
        list of tuple
            A list of tuples (index, cumulative tilt_x in degrees)
        """
        tilt = 0.0
        tilts = []

        for i, s in enumerate(surfaces):
            if s["type"] == "COORDBRK":
                tilt += s.get("tilt_x", 0.0)
            else:
                tilts.append((i, tilt))

        return tilts

    @staticmethod
    def zemax_to_meep_mm(y_mm, z_mm):
        """
        The one and only ZEMAX -> MEEP axis convention. Still in mm.

        ZEMAX y becomes the MEEP x axis (the first array index, matching
        meepsat.components_2D_eps, which indexes its maps eps_map[x, y]).

        ZEMAX z is NEGATED to become the MEEP y axis. In the prescription light
        travels along +z, from the object surface at z = 0 up to the primary at
        z = 714, so without the flip the telescope is drawn upside down. Negating z
        puts the sky at the top of the map, the primary reflector at the bottom, the
        secondary at the left and the focal plane at the right -- the layout of
        Fig. 3 of the LFT MCD paper. Light then travels DOWN the map, off the
        primary, up and to the left onto the secondary, then right to the focal
        plane.

        Apply this once, to profiles and centres alike. Flipping z here while
        leaving the tilt signs in the unflipped frame (or vice versa) mirror-images
        each surface about the horizontal axis through its own centre, which is
        exactly how the mirrors ended up splayed apart and crossing.

        Args:
            y_mm, z_mm: Global ZEMAX coordinates (float or array)

        Returns:
            (x_meep_mm, y_meep_mm)
        """
        return np.asarray(y_mm, dtype=float), -np.asarray(z_mm, dtype=float)

    @staticmethod
    def centred_cell(x_mm, y_mm, margin_mm):
        """
        Smallest MEEP cell that holds the given points, centred on them.

        MEEP puts the origin at the middle of the cell, so once everything has been
        shifted by the returned centre the cell runs from -size/2 to +size/2 on both
        axes and origin_mm for meep_mm_to_pixel() is simply (-size_x/2, -size_y/2).

        Args:
            x_mm, y_mm: Every point the cell has to contain (MEEP mm)
            margin_mm: Clearance to leave around them, before any PML

        Returns:
            (size_x, size_y, centre_mm): Cell size in mm and the centre the points
            must be shifted by
        """
        x_mm = np.asarray(x_mm, dtype=float)
        y_mm = np.asarray(y_mm, dtype=float)

        centre_mm = (0.5 * (x_mm.min() + x_mm.max()),
                    0.5 * (y_mm.min() + y_mm.max()))
        size_x = float(np.ceil(x_mm.max() - x_mm.min() + 2 * margin_mm))
        size_y = float(np.ceil(y_mm.max() - y_mm.min() + 2 * margin_mm))

        return size_x, size_y, centre_mm

    @staticmethod
    def meep_mm_to_pixel(x_mm, y_mm, origin_mm, resolution):
        """
        The one and only mm -> pixel map for the epsilon array.

        Args:
            x_mm, y_mm: MEEP coordinates in mm (see zemax_to_meep_mm). With the
                optics recentred via centred_cell() these are MEEP's own centred
                coordinates, running from -size/2 to +size/2.
            origin_mm: (x, y) of pixel (0, 0), i.e. the lower-left corner of the
                map -- (-size_x/2, -size_y/2) for a centred cell
            resolution: Pixels per mm

        Returns:
            (ix, iy): Float pixel indices into eps_map[x, y]
        """
        ix = (np.asarray(x_mm, dtype=float) - origin_mm[0]) * resolution
        iy = (np.asarray(y_mm, dtype=float) - origin_mm[1]) * resolution
        return ix, iy

    @staticmethod
    def polyexp(x, y, A_ij, R_N, N):
        """
        Evaluate the extended polynomial surface equation.

        The result is in the same length unit as the A_ij coefficients (mm), whatever
        unit x, y and R_N are in -- x/R_N and y/R_N are dimensionless. Scaling x, y
        and R_N into pixels therefore does NOT give a sag in pixels.

        Args:
            x: X-coordinate (float or array)
            y: Y-coordinate (float or array)
            A_ij: Coefficients of the polynomial (2D array)
            R_N: Normalization radius (float)
            N: Degree of the polynomial (int)

        Returns:
            Z-coordinate (sag) on the surface, in mm
        """
        z = 0.0
        for i in range(N + 1):
            for j in range(N + 1 - i):
                z = z + A_ij[i][j] * (x / R_N) ** i * (y / R_N) ** j
        return z

    @staticmethod
    def profile_normals(u_mm, v_mm):
        """
        Unit normals along a sampled profile.

        Args:
            u_mm, v_mm: Profile coordinates, in whichever 2D frame the caller uses

        Returns:
            (n_u, n_v): Arrays of unit normal components
        """
        t_u = np.gradient(u_mm)
        t_v = np.gradient(v_mm)
        norm = np.hypot(t_u, t_v)
        norm[norm == 0] = 1.0
        return -t_v / norm, t_u / norm

    @classmethod
    def mirror_profile_mm(cls, center_mm, half_width_mm, A_ij, R_N, N,
                        ap_y=0.0, tilt_deg=0.0, invert_sag=False,
                        samples_per_mm=2.0):
        """
        2D (YZ) profile of an extended polynomial mirror, in the global ZEMAX frame.

        Everything is in mm. This is the only place a mirror profile is built --
        pixels are introduced later, by rasterize_profile().

        Args:
            center_mm: (y, z) surface centre in global ZEMAX coordinates
            half_width_mm: Half-width of the mirror along local y
            A_ij: Coefficients of the polynomial (2D array)
            R_N: Normalization radius (mm)
            N: Degree of the polynomial (int)
            ap_y: Aperture decentering along local y (mm)
            tilt_deg: CUMULATIVE rotation about the ZEMAX x-axis, i.e. the sum of
                the tilt_x values of every coordinate break preceding the surface.
                Use the prescription's own signs -- flipping them mirrors the
                surface about the horizontal axis through its own centre.
            invert_sag: Whether to invert the sag (concave vs convex)
            samples_per_mm: Sampling density along the local y axis. Use at least
                2 * resolution so consecutive samples land on adjacent pixels.

        Returns:
            (y_mm, z_mm): Arrays of global ZEMAX coordinates along the profile
        """
        n_points = max(int(np.ceil(2 * half_width_mm * samples_per_mm)), 2)
        y_local = np.linspace(-half_width_mm, half_width_mm, n_points)

        #! Evaluate polynomial at aperture-decentered position
        z_sag = cls.polyexp(0.0, y_local - ap_y, A_ij, R_N, N)

        if invert_sag:
            z_sag = -z_sag

        theta = np.radians(tilt_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotate about the ZEMAX x-axis, then translate to the global centre
        y_mm = y_local * cos_t - z_sag * sin_t + center_mm[0]
        z_mm = y_local * sin_t + z_sag * cos_t + center_mm[1]

        return y_mm, z_mm

    @classmethod
    def offset_profile(cls, x_mm, y_mm, thickness_mm, optical_side_mm):
        """
        Back face of a mirror body, offset along the surface normal.

        The material goes on the side facing AWAY from optical_side_mm, so the
        reflecting face stays exactly on the prescription surface. One global sign
        is taken at the middle of the profile, so curvature cannot flip the material
        from one side of the mirror to the other partway along.

        Args:
            x_mm, y_mm: Reflecting-face profile in MEEP coordinates (mm)
            thickness_mm: Body thickness along the surface normal
            optical_side_mm: (x, y) of what this mirror reflects towards

        Returns:
            (x_back, y_back, n_x, n_y): The back face, and the unit normal pointing
            into the material
        """
        n_x, n_y = cls.profile_normals(x_mm, y_mm)

        mid = len(x_mm) // 2
        to_optical_side = np.array([optical_side_mm[0] - x_mm[mid],
                                    optical_side_mm[1] - y_mm[mid]])
        if np.dot(to_optical_side, np.array([n_x[mid], n_y[mid]])) > 0:
            n_x, n_y = -n_x, -n_y  # normals now point into the material

        return x_mm + thickness_mm * n_x, y_mm + thickness_mm * n_y, n_x, n_y
    
    # --- For cell placement and sizing ------------------------------------------------------
    @classmethod
    def zemax_placement(cls, mirrors, extra_zemax_points=(), margin_mm=0.0):
        """
        The shared ZEMAX -> MEEP offset for one prescription, and the cell it needs.

        Apply the SAME offset to every Mirror of the prescription. The returned cell
        size is advisory: standalone it tells you what to allocate, but inside
        MeepSAT the cell comes from mpsat_sim.cell_size and only the offset is used.

        Args:
            mirrors: Iterable of Mirror objects (only their ZEMAX profiles are read,
                so the offset they currently carry does not matter)
            extra_zemax_points: Further (y, z) ZEMAX points that must fit, e.g. the
                focal plane and the object/sky surface
            margin_mm: Clearance to leave around the optics, before any PML

        Returns:
            (offset_mm, size_x, size_y)
        """
        xs, ys = [], []

        for mirror in mirrors:
            x_mm, y_mm = cls.zemax_to_meep_mm(*mirror.profile_zemax)
            xs.append(np.atleast_1d(x_mm))
            ys.append(np.atleast_1d(y_mm))

        for point in extra_zemax_points:
            x_mm, y_mm = cls.zemax_to_meep_mm(*point)
            xs.append(np.atleast_1d(x_mm))
            ys.append(np.atleast_1d(y_mm))

        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)

        size_x, size_y, offset_mm = cls.centred_cell(x_all, y_all, margin_mm)
        return offset_mm, size_x, size_y
    
    
    # --- For mirror construction ------------------------------------------------------
    @classmethod
    def from_excel(cls, eps, name, centre_zemax_mm, half_width_mm, excel_values,
                   R_N, N=7, **kwargs):
        """
        Build from a flat ZEMAX/Excel coefficient row.

        The row is checked against the Extended Polynomial term order, so a
        transposed A_ij (x-coefficients used as y-coefficients) raises here
        rather than quietly reshaping the mirror.
        """
        A_ij = cls.excel_to_aij(excel_values, N=N)
        cls.check_aij_ordering(excel_values, A_ij, N=N, label=name)
        return cls(eps, name, centre_zemax_mm, half_width_mm, A_ij, R_N, N=N,
                   **kwargs)
   
    @classmethod
    def from_prescription(cls, eps, name, surfaces, surface_index,
                          half_width_mm, excel_values, R_N, N=7, **kwargs):
        """
        Build from a ZEMAX surface table, taking centre and tilt from it.

        Args:
            eps: The epsilon map to edit
            surfaces: The surface list (see design_class.compute_surface_centers)
            surface_index: Index of this mirror's surface in that list
            half_width_mm, excel_values, R_N, N: As for from_excel()
            **kwargs: Passed on to __init__ (ap_y, thickness_mm, mpsat_sim, ...);
                tilt_deg is taken from the table and must not be given here
        """
        if 'tilt_deg' in kwargs:
            raise TypeError('tilt_deg comes from the prescription; drop it or '
                            'use Mirror.from_excel() instead')

        centres = dict(cls.compute_surface_centers(surfaces))
        tilts = dict(cls.cumulative_tilts_x(surfaces))

        if surface_index not in centres:
            raise KeyError(f'surface {surface_index} is not a real surface in '
                           f'this prescription (real surfaces: {sorted(centres)})')

        centre = centres[surface_index]
        return cls.from_excel(eps, name, (centre[1], centre[2]), half_width_mm,
                              excel_values, R_N, N=N,
                              tilt_deg=tilts[surface_index], **kwargs)
        
    # --- the cell this mirror lives in --------------------------------------------------
    @property
    def map_shape(self):
        """(nx, ny) the permittivity map should have for this cell."""
        return (int((self.size_x + 2 * self.dpml) * self.res) + 1,
                int((self.size_y + 2 * self.dpml) * self.res) + 1)

    @property
    def origin_mm(self):
        """(x, y) of pixel (0, 0): the lower-left corner, in centred coords."""
        return (-(self.size_x / 2 + self.dpml), -(self.size_y / 2 + self.dpml))

    @property
    def extent(self):
        """[xmin, xmax, ymin, ymax] in mm, for imshow(..., extent=...)."""
        half_x = self.size_x / 2 + self.dpml
        half_y = self.size_y / 2 + self.dpml
        return [-half_x, half_x, -half_y, half_y]

    # --- frame conversions (ZEMAX TO MEEPSAT and vice versa) -------------------------------------------------
    def from_zemax(self, y_mm, z_mm):
        """
        ZEMAX (y, z) mm -> MEEP centred (x, y) mm.

        The z-flip (sky at the top, primary at the bottom) and the shared
        origin_offset_mm, in that order.
        """
        x, y = self.zemax_to_meep_mm(y_mm, z_mm)
        return x - self.origin_offset_mm[0], y - self.origin_offset_mm[1]

    def to_zemax(self, x_mm, y_mm):
        """MEEP centred (x, y) mm -> ZEMAX (y, z) mm. The inverse of from_zemax."""
        x = np.asarray(x_mm, dtype=float) + self.origin_offset_mm[0]
        y = np.asarray(y_mm, dtype=float) + self.origin_offset_mm[1]
        return x, -y

    def to_pixel(self, x_mm, y_mm):
        """MEEP centred (x, y) mm -> float indices into permittivity_map[x, y]."""
        return self.meep_mm_to_pixel(x_mm, y_mm, self.origin_mm, self.res)

    # --- Now for creating the geometry ----------------------------------------------------------
    @property
    def profile_zemax(self):
        """(y_mm, z_mm) of the reflecting face in the global ZEMAX frame."""
        if self._profile_zemax is None:
            self._profile_zemax = self.mirror_profile_mm(
                center_mm=self.centre_zemax_mm,
                half_width_mm=self.half_width_mm,
                A_ij=self.A_ij, R_N=self.R_N, N=self.N,
                ap_y=self.ap_y, tilt_deg=self.tilt_deg,
                invert_sag=self.invert_sag,
                samples_per_mm=self.samples_per_mm)
        return self._profile_zemax
    
    @property
    def profile_meep(self):
        """(x_mm, y_mm) of the reflecting face in MEEP centred coordinates."""
        return self.from_zemax(*self.profile_zemax)

    @property
    def centre_meep(self):
        """The surface centre in MEEP centred coordinates."""
        return self.from_zemax(*self.centre_zemax_mm)

    @property
    def back_face_meep(self):
        """
        (x_mm, y_mm) of the mirror's back face in MEEP centred coordinates.

        Shares design_class.offset_profile() with the rasteriser, so what gets
        plotted is the same surface that gets written into the epsilon map.
        """
        self._require_optical_side()
        x_mm, y_mm = self.profile_meep
        optical_side = self.from_zemax(*self.optical_side_zemax_mm)
        x_back, y_back, _, _ = self.offset_profile(x_mm, y_mm, self.thickness_mm, 
                                                 optical_side)
        return x_back, y_back
    
    @property
    def arc_length_mm(self):
        """Length of the reflecting face along the surface."""
        y_mm, z_mm = self.profile_zemax
        return float(np.sum(np.hypot(np.diff(y_mm), np.diff(z_mm))))
    
    @property
    def sag_mm(self):
        """Peak-to-valley of the profile, measured perpendicular to its chord."""
        y_mm, z_mm = self.profile_zemax
        chord = np.array([y_mm[-1] - y_mm[0], z_mm[-1] - z_mm[0]])
        chord = chord / np.linalg.norm(chord)
        normal = np.array([-chord[1], chord[0]])
        dev = (y_mm - y_mm[0]) * normal[0] + (z_mm - z_mm[0]) * normal[1]
        return float(dev.max() - dev.min())
    
    def _require_optical_side(self):
        if self.optical_side_zemax_mm is None:
            raise ValueError(
                f'{self.name} has no optical_side_zemax_mm, so the material '
                f'side is undefined -- set it to whatever this mirror reflects '
                f'towards (normally the next surface centre)')
            
    # --- writing the mirror into the map ----------------------------------------------
    def rasterize_profile(self, eps_map, x_mm, y_mm, origin_mm, resolution,
                        thickness_mm, optical_side_mm, eps_value=12.0):
        """
        Write a solid mirror body into eps_map[x, y].

        Works entirely in the MEEP frame -- pass profiles that have already been
        through zemax_to_meep_mm(), so the z-flip is applied once, upstream, to
        profiles and centres alike.

        The reflecting face sits exactly on (x_mm, y_mm) and thickness_mm of
        material is added on the side facing AWAY from optical_side_mm, so the
        surface the beam sees is the one the prescription describes. Thickness is
        measured along the surface normal rather than down a grid axis, so a steeply
        tilted mirror gets the same body as a shallow one.

        The body is filled by stepping from the front face to the back face along
        the normal at every sample, on a lattice fine enough (half a pixel in both
        directions) that the rounded pixel set is contiguous -- no combing, however
        the profile is sampled. A single skimage polygon() over the whole outline
        would be correct too, but it costs O(bounding box x vertices), which for a
        mirror this size is billions of point-in-polygon tests.

        Args:
            eps_map: 2D array indexed [x, y], modified in place
            x_mm, y_mm: Reflecting-face profile in MEEP coordinates (mm)
            origin_mm: (x, y) of pixel (0, 0)
            resolution: Pixels per mm
            thickness_mm: Mirror thickness along the surface normal
            optical_side_mm: (x, y) of a point on the optical side of this mirror,
                i.e. what it has to reflect towards -- normally the other mirror's
                centre, in the same MEEP frame. The material goes on the opposite
                side.
            eps_value: Permittivity written into the body

        Returns:
            dict with 'rows'/'cols' (the pixels filled), 'x_back'/'y_back' (the
            back-face profile in mm) and 'normal' (the unit normal pointing into
            the material)
        """
        x_back, y_back, n_x, n_y = self.offset_profile(x_mm, y_mm, thickness_mm,
                                                optical_side_mm)

        # Half-pixel steps from the front face to the back face
        n_steps = int(np.ceil(2.0 * thickness_mm * resolution)) + 1
        depth = np.linspace(0.0, thickness_mm, n_steps)[:, None]

        x_body = x_mm[None, :] + depth * n_x[None, :]
        y_body = y_mm[None, :] + depth * n_y[None, :]

        ix, iy = self.meep_mm_to_pixel(x_body.ravel(), y_body.ravel(), origin_mm, resolution)
        rows = np.rint(ix).astype(np.intp)
        cols = np.rint(iy).astype(np.intp)

        inside = ((rows >= 0) & (rows < eps_map.shape[0]) &
                (cols >= 0) & (cols < eps_map.shape[1]))
        rows, cols = rows[inside], cols[inside]

        eps_map[rows, cols] = eps_value

        # Deduplicate so the caller counts pixels, not samples
        flat = np.unique(rows.astype(np.int64) * eps_map.shape[1] + cols)
        rows, cols = np.divmod(flat, eps_map.shape[1])

        return {'rows': rows.astype(np.intp), 'cols': cols.astype(np.intp),
                'x_back': x_back, 'y_back': y_back,
                'normal': (n_x, n_y),
                'clipped': int((~inside).sum())}

    def write_mirror(self, eps_map=None):
        """
        Write this mirror's body into the epsilon map, in place.

        Args:
            eps_map: The map to edit; defaults to the one this Mirror was given

        Returns:
            The body dict from design_class.rasterize_profile ('rows', 'cols',
            'x_back', 'y_back', 'normal', 'clipped')
        """
        self._require_optical_side()

        if eps_map is None:
            eps_map = self.permittivity_map
        if eps_map is None:
            raise ValueError(f'{self.name} has no epsilon map to write into')

        x_mm, y_mm = self.profile_meep
        optical_side = self.from_zemax(*self.optical_side_zemax_mm)

        return self.rasterize_profile(
            eps_map, x_mm, y_mm, self.origin_mm, self.res,
            thickness_mm=self.thickness_mm,
            optical_side_mm=optical_side,
            eps_value=self.eps_value)

    @staticmethod
    def assemble(mirrors, verbose=False):
        """
        Write all the mirror in the mirrors list into the epsilon map they share, and return it.

        Args:
            mirrors: Iterable of Mirror objects, all holding the same map
            verbose: Print a line per mirror as it is written

        Returns:
            (eps_map, bodies): The shared map and a dict of name -> body dict
        """
        mirrors = list(mirrors)
        if not mirrors:
            raise ValueError('no mirrors to assemble')

        eps_map = mirrors[0].permittivity_map
        for mirror in mirrors[1:]:
            if mirror.permittivity_map is not eps_map:
                raise ValueError(
                    f'{mirror.name} holds a different epsilon map from '
                    f'{mirrors[0].name} -- every component must share one map')

        bodies = {}
        for mirror in mirrors:
            body = mirror.write_mirror(eps_map)
            bodies[mirror.name] = body
            if verbose:
                print(f'  {mirror.name}: arc {mirror.arc_length_mm:.1f} mm, '
                      f'sag {mirror.sag_mm:.2f} mm, '
                      f'{len(body["rows"])} px written'
                      + (f', {body["clipped"]} samples clipped'
                         if body['clipped'] else ''))

        return eps_map, bodies

    # --- writing the mirror as MEEP geometry (triangular mesh) -------------------------
    def triangular_mesh(self, n_segments=200, plot=False, savepath=None):
        """
        Triangulate the mirror body, for meshing.convert_triangles_to_prisms().

        Deliberately NOT meshing._create_triangular_mesh(): that one takes the
        ConvexHull of the boundary, which is right for a pyramidal absorber but
        wrong for a mirror. A curved mirror is concave on one side, so its hull
        swallows the empty space between the chord and the reflecting face --
        for the LiteBIRD LFT primary that is 59% phantom material.

        Instead the mesh is built directly from the two faces this class already
        computes. offset_profile() returns the back face sample-for-sample against
        the front face, so front[i], front[i+1], back[i+1], back[i] is a quad for
        every profile segment, and each quad splits into two triangles. The result
        is an exact triangulation of the same body write_mirror() rasterises, with
        no hull, no Delaunay and no interior sampling.

        Coordinates come out in the corner-origin mm frame that
        meshing.convert_triangles_to_prisms() expects (it subtracts grid_size/2
        to recentre), NOT in the MEEP centred frame.

        Args:
            n_segments: Profile segments to mesh, i.e. 2 * n_segments triangles.
                The profile is decimated to this many samples, which is what
                controls the faceting error -- roughly L^2 / (8 R) for segment
                length L and local radius of curvature R. The default is far
                finer than one pixel for LFT-sized optics.
            plot: Save a triplot of the mesh, via meshing._visualize_triangular_mesh
            savepath: Where to write that plot; defaults to './<name>_triangular_mesh.png'

        Returns:
            matplotlib.tri.Triangulation
        """
        from matplotlib.tri import Triangulation
        import meepsat.meshing as mesh

        x_f, y_f = self.profile_meep
        x_b, y_b = self.back_face_meep

        # Decimate both faces with the SAME indices, so they stay paired
        n_segments = max(int(n_segments), 1)
        keep = np.unique(np.linspace(0, len(x_f) - 1,
                                     min(n_segments + 1, len(x_f))).astype(int))
        x_f, y_f = x_f[keep], y_f[keep]
        x_b, y_b = x_b[keep], y_b[keep]
        m = len(keep)

        # meshing.convert_triangles_to_prisms() recentres by subtracting
        # grid_size/2, so hand it corner-origin mm
        x = np.r_[x_f, x_b] - self.origin_mm[0]
        y = np.r_[y_f, y_b] - self.origin_mm[1]

        i = np.arange(m - 1)
        triangles = np.vstack([np.column_stack([i, i + 1, m + i]),
                               np.column_stack([i + 1, m + i + 1, m + i])])

        # Consistent counter-clockwise winding, as mp.Prism expects
        v0, v1, v2 = (np.column_stack([x, y])[triangles[:, k]] for k in range(3))
        cw = np.cross(v1 - v0, v2 - v0) < 0
        triangles[cw] = triangles[cw][:, ::-1]

        tri = Triangulation(x, y, triangles=triangles)

        if plot:
            mesh._visualize_triangular_mesh(
                tri, self.size_x + 2 * self.dpml, self.size_y + 2 * self.dpml,
                output_file=savepath or f'./{self.name}_triangular_mesh.png')

        return tri

    def to_prisms(self, material=None, n_segments=200, thickness=1.0,
                  plot_mesh=False, savepath=None):
        """
        This mirror as a list of mp.Prism objects, for the MEEP geometry list.

        The alternative to write_mirror(): instead of stamping pixels into the
        shared epsilon map, the body is handed to MEEP as geometry, exactly as
        meep_geometry.Absorbers does it. Use one or the other, not both -- MEEP
        resolves geometry on top of epsilon_input_file, so doing both just writes
        the same body twice.

        Worth it when the mirror should be a perfect conductor: mp.metal cannot be
        expressed as a permittivity, so PEC mirrors have to go down this route.

        Args:
            material: mp.Medium for the body; defaults to mp.Medium(epsilon=
                self.eps_value). Pass mp.metal for a PEC mirror.
            n_segments: Passed to triangular_mesh()
            thickness: Prism height along z. Irrelevant in 2D, as for Absorbers.
            plot_mesh, savepath: Passed to triangular_mesh()

        Returns:
            List of mp.Prism, one per triangle
        """
        import meepsat.meshing as mesh

        tri = self.triangular_mesh(n_segments=n_segments, plot=plot_mesh,
                                   savepath=savepath)

        return mesh.convert_triangles_to_prisms(
            tri=tri,
            gridx_size_mm=self.size_x + 2 * self.dpml,
            gridy_size_mm=self.size_y + 2 * self.dpml,
            material=material or mp.Medium(epsilon=self.eps_value),
            thickness=thickness)

    @staticmethod
    def assemble_prisms(mirrors, material=None, n_segments=200, verbose=False,
                        plot_mesh=False, savepath=None):
        """
        to_prisms() over several mirrors, flattened into one geometry list.

        Args:
            mirrors: Iterable of Mirror objects
            material: One mp.Medium for all of them, or a dict of name -> Medium
            n_segments, plot_mesh, savepath: Passed through to to_prisms()
            verbose: Print a line per mirror

        Returns:
            List of mp.Prism ready to extend mpsat_sim.meep_geometry
        """
        geometry = []
        for mirror in mirrors:
            mat = material.get(mirror.name) if isinstance(material, dict) else material
            prisms = mirror.to_prisms(
                material=mat, n_segments=n_segments, plot_mesh=plot_mesh,
                savepath=(savepath + f'{mirror.name}_triangular_mesh.png'
                          if savepath else None))
            geometry.extend(prisms)
            if verbose:
                print(f'  {mirror.name}: arc {mirror.arc_length_mm:.1f} mm, '
                      f'sag {mirror.sag_mm:.2f} mm, {len(prisms)} prisms')

        return geometry

    # --- Optional Plotting ----------------------------------------------------------
    @classmethod
    def plot_epsilon_map(cls, eps_map, mirrors, extra_zemax_points=None,
                         savepath=None, show=False, title=None,
                         mark_centre=True, return_points=False, show_centre_points=True):
        """
        Plot an epsilon map in MEEP centred coordinates.

        eps is stored [x, y], so it is transposed for imshow and given the cell's
        extent -- the axes really are mm, not pixels.

        Args:
            eps_map: The map to draw
            mirrors: Iterable of Mirror objects; the first sets the extent
            extra_zemax_points: Optional dict of label -> (y, z) ZEMAX points
            savepath, show, title, mark_centre: Figure handling
            return_points: If True, return the points used for plotting
        """
        mirrors = list(mirrors)
        ref = mirrors[0]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        # .real because MeepSAT maps are complex when a component is lossy
        im = ax.imshow(np.real(eps_map).T, extent=ref.extent, origin='lower',
                       cmap='viridis', interpolation='nearest')

        for mirror in mirrors:
            cu, cv = mirror.centre_meep
            ax.scatter(cu, cv, s=80, c='white', marker='x', zorder=5, label=mirror.name + f"({cu:.1f}, {cv:.1f})")
            ax.text(cu, cv, f'  {mirror.name}', fontsize=11, color='white')

        for label, point in (extra_zemax_points or {}).items():
            pu, pv = ref.from_zemax(*point)
            ax.scatter(pu, pv, s=80, c='white', marker='x', zorder=5, label=label + f"({pu:.1f}, {pv:.1f})")
            ax.text(pu, pv, f'  {label}', fontsize=11, color='white')

        if mark_centre:
            ax.axhline(0, color='white', linewidth=0.6, alpha=0.4)
            ax.axvline(0, color='white', linewidth=0.6, alpha=0.4)
            ax.scatter(0, 0, s=90, c='white', marker='+', zorder=6)
            

        plt.colorbar(im, ax=ax, label='Permittivity')
        ax.set_xlabel('X MEEP (mm, centred on the cell)', fontsize=12)
        ax.set_ylabel('Y MEEP (mm, centred on the cell)', fontsize=12)
        ax.set_title(title or (f'Epsilon Map  ({ref.size_x:.0f} x '
                               f'{ref.size_y:.0f} mm cell, {ref.res:g} px/mm)'),
                     fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        if show_centre_points:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.7, edgecolor='white')
        
        plt.tight_layout()
        if savepath:
            cls._ensure_dir(savepath)
            plt.savefig(savepath, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        if return_points:
            points = {}
            for mirror in mirrors:
                points[mirror.name] = mirror.centre_meep
            if extra_zemax_points:
                for label, point in extra_zemax_points.items():
                    points[label] = ref.from_zemax(*point)
            return points    
    
    
    def __repr__(self):
        return (f'Mirror({self.name!r}, centre_zemax_mm='
                f'({self.centre_zemax_mm[0]:.2f}, {self.centre_zemax_mm[1]:.2f}), '
                f'half_width_mm={self.half_width_mm:g}, '
                f'tilt_deg={self.tilt_deg:+.2f}, '
                f'map={"none" if self.permittivity_map is None else self.permittivity_map.shape})')

    
