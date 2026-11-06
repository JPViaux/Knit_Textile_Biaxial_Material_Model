import numpy as np
import subprocess as sb
import pathlib
import sys
import os
import ctypes
import copy
import shutil
import tempfile
from typing import Any, Literal, overload

from numpy.typing import NDArray

curr_file = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_file)
sys.path.insert(0, parent_dir)

from Python_Wrapper import mech
from Python_Wrapper.model_registry import (
    ModelSpec,
    ModelSpecificationError,
    load_model_spec,
)

def main() -> None:
    '''Main code for the file. Compiles the Umat and
    imports it'''
    raise NotImplementedError("This code cannot run by itself yet. Check the main function")

class Load_props:
    '''Class that makes it more convenient to load in a properties file and to handle
       the contents, requieres a path to a filename'''
    def __init__(
        self,
        fname: pathlib.Path,
        statev_initial: pathlib.Path | None = None,
        model_spec: ModelSpec | str | None = None,
    ) -> None:
        '''Initializer'''
        self.model_spec = (
            load_model_spec(model_spec) if isinstance(model_spec, str) else model_spec
        )
        self.data = np.loadtxt(fname, str, skiprows=1, ndmin=2)
        if self.data.shape[1] != 6:
            raise ValueError(
                f"Material-parameter file must contain 6 columns; "
                f"found {self.data.shape[1]} in {fname}"
            )
        self._reject_duplicate_names(fname)
        if self.model_spec is not None:
            self._order_for_model(self.model_spec, fname)
        self.set_prop_mapping()
        self.nprops = len(self.data)
        self.set_start_parameters()
        # If there is a path for the state variables initial value
        if statev_initial is not None:
            self.statev_ini_val = np.loadtxt(statev_initial, float)
            if self.statev_ini_val.shape == ():
                self.n_statev = 1
            else:
                self.n_statev = len(self.statev_ini_val)  
        else:
            self.statev_ini_val = None
            self.n_statev = None  

    def _reject_duplicate_names(self, fname: pathlib.Path) -> None:
        """Reject ambiguous parameter files before constructing dictionaries."""
        names = [str(row[0]) for row in self.data]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ModelSpecificationError(
                f"Duplicate parameters in {fname}: {', '.join(duplicates)}"
            )

    def _order_for_model(self, model_spec: ModelSpec, fname: pathlib.Path) -> None:
        """Validate names and reorder rows into the model's canonical PROPS order."""
        rows_by_name = {str(row[0]): row for row in self.data}
        expected = model_spec.parameter_names
        expected_set = set(expected)
        actual_set = set(rows_by_name)
        missing = [name for name in expected if name not in actual_set]
        extra = sorted(actual_set - expected_set)

        problems: list[str] = []
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected: {', '.join(extra)}")
        if problems:
            raise ModelSpecificationError(
                f"Parameter file {fname} is incompatible with "
                f"{model_spec.model_id}; " + "; ".join(problems)
            )

        self.data = np.asarray([rows_by_name[name] for name in expected])

    def __str__(self) -> str:
        '''When printing one instance of this class, return the properties dictionary'''
        return f"{self.prop_dic}"

    def set_prop_mapping(self) -> None:
        '''Props need to be mapped to entry in props_list for umat call'''
        self.props_map = {}
        for idx in range(len(self.data)):
            self.props_map[idx] = self.data[idx][0]
        
    def get_prop_mapping(self) -> dict:
        '''gets the mapping between properties entry and argument index for umat call'''
        return self.props_map

    def set_start_parameters(self) -> None:
        '''Saves the start values for each of the parameters, order as input'''
        self.prop_dic = {}
        for prop,startval,dtype,mini,maxi,fix_par in self.data:
            if dtype == 'int': #for int
                self.prop_dic[prop] = ( int(float(startval)), int, int(float(mini)), int(float(maxi)), 0, str_to_bool(fix_par))
            else:
                self.prop_dic[prop] = ( float(startval), float, float(mini), float(maxi), 0, str_to_bool(fix_par))
                    
    def get_start_sorted_parameters(self) -> list:
        '''returns the start parameters in the order that the umat want'''
        svals = []
        for idx in range(self.nprops):
            prop = self.props_map[idx]
            svals.append((prop, self.prop_dic[prop][0]))
        return svals

    def get_start_parameters(self) -> dict:
        '''returns the dicitonary of properties'''
        return self.prop_dic

class Material_model:
    '''Material model calls. It requieres and instance of the properties class
       to be initalized. It provides an interface to set and get properties'''
    def __init__(self, props_instance: Load_props) -> None:
        '''Define the parameters of the material model. Saves an instance
        of Load_props as loaded_data'''
        self.parameters = props_instance.get_start_parameters()
        self.mapping = props_instance.get_prop_mapping()
        self.nprops = len(self.parameters)
        self.loaded_data = props_instance
        self.model_spec = props_instance.model_spec
        self.statev_ini_val = props_instance.statev_ini_val
        self.n_statev = props_instance.n_statev

    def get_sorted_props(self) -> list:
        '''Returns the properties in the order provided in the input props file'''
        return self.loaded_data.get_start_sorted_parameters()
   
    def get_keys(self):
        '''Returns all keys in parameter dictionary'''
        return self.parameters.keys()
 
    def reset(self, start_props) -> None:
        '''Quick way to get back to certain properties'''
        self.__init__(start_props)

    def getv(self, key: str):
        '''Gets the value of a certain entry'''
        return self.parameters[key]  

    def getv_only(self, key: str):
        '''gets the value of a certain entry'''
        return self.parameters[key][0]

    def setv(self, key, value, ty, mini = 0, maxi = 1, nr = 0, fix_par = None):
        '''sets a certain value'''
        self.parameters[key] = (value, ty, mini, maxi, nr, fix_par)
    
    def setv_only(self, key, value) -> None:
        '''sets a certain value'''
        val, ty, mini, maxi, nr, fix_par = self.getv(key)
        self.parameters[key] = (value, ty, mini, maxi, nr, fix_par)
    
    def get_umat_input(self) -> list:
        '''just returns the sorted list of parameters for the umat call'''
        props = []
        for idx in range(self.nprops):
            props.append( float(self.parameters[ self.mapping[idx] ][0] ))
        return props
    
class Umat_wrapper:
    '''Class that handles the interfacing with the umat code. Is capable of calling
       fortran code and retrieving stress results from the umat calculation. It 
       requieres an instance of a material model to be initialized.'''
    def __init__(self, matmod: Material_model, umat_type: list[str]) -> None:
        '''Init function, creates the instance and defines the allowed strains 
        and stress. The strain and stress output type of the UMat must be specified'''
        self.load_model(matmod)
        self.valid_input_stress = {'Sigma','P','S'}
        self.valid_input_strain = {'C','E','F','H'}
        self.valid_cost_types = {'legacy', 'normalized'}
        self.current_stress = None
        self.cost_type = 'legacy'
        self.normalized_stress_floor_factor = 0.05
        
        umat_op_stress = umat_type[1]
        umat_in_strain = umat_type[0]

        # If there is a file with initial values for the state variables
        if matmod.n_statev == None:
            self.has_statev = False
        else:
            self.n_statev = matmod.n_statev
            self.statev_ini_val = matmod.statev_ini_val
            self.statev = self.statev_ini_val
            self.has_statev = True
        
        self.has_temp = False

        if umat_op_stress not in self.valid_input_stress:
            raise ValueError("The UMat stress output type is not valid")
        elif umat_in_strain not in self.valid_input_strain:
            raise ValueError("The UMat strain input type is not valid")
        else:
            self.umat_out_stress_type = umat_op_stress
            self.umat_in_strain_type = umat_in_strain

    def get_props_dic(self):
        '''Return the dictionary of properties from the underlying material model'''
        return self.m.parameters
    
    def get_valid_strain_stress(self):
        '''Returns the valid strain and stress types'''
        return self.valid_input_strain, self.valid_input_stress

    def set_cost_function(self, cost_type: str = 'legacy',
                          normalized_stress_floor_factor: float = 0.05,
                          normalized_stress_reference_maxima: dict[int, float] | None = None) -> None:
        '''Selects the cost function used for fitting.

        cost_type='legacy' uses the original sum of squared absolute stress
        errors. cost_type='normalized' uses a stabilized relative stress error,
        where normalized_stress_floor_factor defines the stress floor as a
        fraction of the maximum experimental stress magnitude for each Voigt
        component.
        '''
        cost_type = str(cost_type).lower()
        if cost_type not in self.valid_cost_types:
            valid = ', '.join(sorted(self.valid_cost_types))
            raise ValueError(f"Invalid cost_type '{cost_type}'. Use one of: {valid}")

        normalized_stress_floor_factor = float(normalized_stress_floor_factor)
        if normalized_stress_floor_factor < 0:
            raise ValueError("normalized_stress_floor_factor must be non-negative")

        self.cost_type = cost_type
        self.normalized_stress_floor_factor = normalized_stress_floor_factor
        # Fixed direction scales; example: {0: 0.05, 1: 0.08} in PK2 units.
        references = normalized_stress_reference_maxima or {}
        if any(not np.isfinite(value) or value < 0 for value in references.values()):
            raise ValueError('Normalization references must be finite and nonnegative')
        self.normalized_stress_reference_maxima = {int(axis): float(value) for axis, value in references.items()}

    def setv(self, key, value, ty, mini = 0, maxi = 1, nr = 0, fix_par = None) -> None:
        '''Set a value in the underlying material model'''
        self.m.setv(key, value, ty, mini, maxi, nr, fix_par)

    def getv(self, key):
        '''Get a value in the underlying material model'''
        return self.m.getv(key)

    def get_strstres(self):
        '''Get the strains and stresses from the loaded experiment'''
        return self.expstrains, self.expstresses

    def reset_history(self) -> None:
        '''Reset stress and state variables before a fresh prediction sweep.'''
        self.current_stress = None
        if self.has_statev:
            self.statev = np.array(self.statev_ini_val, copy=True)

    def load_model(self, matmod):
        '''loads in an instance of the material model parameters'''
        self.m = matmod

    def get_model(self) -> Material_model:
        '''Retrieves material model'''
        return self.m

    def call_umat(self, dfgrd1, props) -> np.array:
        '''Calls the umat with the current values in the material model
        Returns the stress in Voigt notation. F must come in matrix form'''
        
        global umatf
   
        F = mech.voigt_to_matrix(dfgrd1)
        stress = np.zeros(6, dtype=np.float64)
        # Transform to contiguous arrays for Fortran
        F = np.ascontiguousarray(F.flatten(), dtype=np.double)
        stress = np.ascontiguousarray(stress, dtype=np.double)
        props = np.ascontiguousarray(props, dtype=np.double)
        nprops = ctypes.c_int(len(props))
        # Call the myumat function. Use a _ when using meson
        # umatf.myumat_(stress,props,F,nprops)
        umatf.myumat_(
            stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            props.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            F.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(nprops)
        )

        return stress
    
    def call_umat_Abaqus(self, F, props, statev = None, temp = None, stran = None):
        '''Calls the UMAT using the Abaqus standard inputs. Needs at least a deformation
        gradient and material properties'''
        # Deformation gradient in 3x3 matrix form
        # Strain in voigt form

        global umatf    # Declare the umatf library as global to use it

        # Initialize all the UMAT variables
        ntens = 6   # Number of strain and stress components
        if statev is None:
            statev = np.asfortranarray(np.zeros(1, dtype=np.double))    # State variables
        if stran is None:
            stran = np.asfortranarray(np.zeros(ntens, dtype=np.double)) # Strain
        if temp is None:
            temp = 292.0    # Temperature
        
        if self.current_stress is None:
            stress = np.zeros(ntens, dtype=np.double)  # Pass stress as empty vector
        else:
            stress = self.current_stress
        
        #stress = np.zeros(ntens, dtype=np.double)  # Pass stress as empty vector
        nprops = len(props)  # Number of material properties
        nstatv = 1 if statev.shape == () else len(statev)  # Number of state variables
        ddsdde = np.zeros((ntens, ntens), dtype=np.double)  # Stiffness matrix initialized as 0
        time = np.zeros(2, dtype=np.double)  # Time
        dtime = 0.0  # Time variation
        sse = 0.0  # Specific elastic strain energy
        spd = 0.0  # Specific plastic dissipation
        scd = 0.0  # Specific creep dissipation
        rpl = 0.0  # Volumetric heat generation per unit time at the end of the increment
        ddsddt = np.zeros(ntens, dtype=np.double)  # Variation of the stress increments with respect to the temperature
        drplde = np.zeros(ntens, dtype=np.double)  # Variation of RPL with respect to the strain increments
        drpldt = 0.0  # Variation of RPL with respect to the temperature
        pnewdt = 1.0  # Ratio of suggested new time increment to time being used
        dstran = np.zeros(ntens, dtype=np.double)  # Array of strain increments
        dtemp = 0.0  # Increment of temperature
        predef = np.zeros(1, dtype=np.double)  # Array of interpolated values of predefined field variables
        dpred = np.zeros(1, dtype=np.double)  # Array of increments of predefined field variables
        cmname = "UMAT"  # User-defined material name, left justified
        ndi = 3  # Number of direct stress components at this point (3 in 3D)
        nshr = 3  # Number of engineering shear stress components at this point (3 in 3D)
        coords = np.zeros(3, dtype=np.double)  # An array containing the coordinates of this point
        drot = np.eye(3, dtype=np.double)  # Rotation increment matrix. Identity by default.
        celent = 1.0  # Characteristic element length
        dfgrd0 = mech.voigt_to_matrix(F)    # Array containing the deformation gradient at the beginning of the increment
        dfgrd1 = mech.voigt_to_matrix(F)    # Array containing the deformation gradient at the end of the increment
        noel = 1    # Element number
        npt = 1     # Integration point number
        layer = 1   # Layer number (for composite shells and layered solids)
        kspt = 1    # Section point number within the current layer
        kstep = 1   # Step number
        kinc = 1    # Increment number

        # Ensure the arrays are contiguous and of type double
        stress = np.ascontiguousarray(stress, dtype=np.double)
        statev = np.ascontiguousarray(statev, dtype=np.double)
        ddsdde = np.ascontiguousarray(ddsdde.flatten(), dtype=np.double)
        sse = ctypes.c_double(sse)
        spd = ctypes.c_double(spd)
        scd = ctypes.c_double(scd)
        rpl = ctypes.c_double(rpl)
        ddsddt = np.ascontiguousarray(ddsddt, dtype=np.double)
        drplde = np.ascontiguousarray(drplde, dtype=np.double)
        drpldt = ctypes.c_double(drpldt)
        stran = np.ascontiguousarray(stran, dtype=np.double)
        dstran = np.ascontiguousarray(dstran, dtype=np.double)
        time = np.ascontiguousarray(time, dtype=np.double)
        dtime = ctypes.c_double(dtime)
        temp = ctypes.c_double(temp)
        dtemp = ctypes.c_double(dtemp)
        predef = np.ascontiguousarray(predef, dtype=np.double)
        dpred = np.ascontiguousarray(dpred, dtype=np.double)
        cmname_buffer = ctypes.create_string_buffer(cmname.encode('utf-8'), 80)
        ndi = ctypes.c_int(ndi)
        nshr = ctypes.c_int(nshr)
        ntens = ctypes.c_int(ntens)
        nstatv = ctypes.c_int(nstatv)
        props = np.ascontiguousarray(props, dtype=np.double)
        nprops = ctypes.c_int(nprops)
        coords = np.ascontiguousarray(coords, dtype=np.double)
        drot = np.ascontiguousarray(drot.flatten(), dtype=np.double)
        pnewdt = ctypes.c_double(pnewdt)
        celent = ctypes.c_double(celent)
        dfgrd0 = np.ascontiguousarray(dfgrd0.flatten(), dtype=np.double)
        dfgrd1 = np.ascontiguousarray(dfgrd1.flatten(), dtype=np.double)
        noel = ctypes.c_int(noel)
        npt = ctypes.c_int(npt)
        layer = ctypes.c_int(layer)
        kspt = ctypes.c_int(kspt)
        kstep = ctypes.c_int(kstep)
        kinc = ctypes.c_int(kinc)
       
        # Call the UMat coded in Fortran
        umatf.umat_(
            stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            statev.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ddsdde.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(sse),
            ctypes.byref(spd),
            ctypes.byref(scd),
            ctypes.byref(rpl),
            ddsddt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            drplde.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(drpldt),
            stran.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dstran.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(dtime),
            ctypes.byref(temp),
            ctypes.byref(dtemp),
            predef.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dpred.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cmname_buffer,
            ctypes.byref(ndi),
            ctypes.byref(nshr),
            ctypes.byref(ntens),
            ctypes.byref(nstatv),
            props.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(nprops),
            coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            drot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(pnewdt),
            ctypes.byref(celent),
            dfgrd0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dfgrd1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(noel),
            ctypes.byref(npt),
            ctypes.byref(layer),
            ctypes.byref(kspt),
            ctypes.byref(kstep),
            ctypes.byref(kinc)
        )
        # Save state variables
        self.statev = np.array(statev)
        # Save stress
        self.current_stress = np.array(stress)
        # Return the stress
        return stress
    
    def load_inputs(self, exp_inp: pathlib.Path, typ: list, res_statev: pathlib.Path = None, exp_temp: pathlib.Path = None) -> None:
        '''Loads an input file of strains and stresses. Takes the type
        of stress and strain that the data expresses. The options are:
        type[0] -> Strain: E (Green Strain, right), C (Cauchy Strain), F (Deformation gradient)
        type[1] -> Stress: Sigma (Cauchy Stress), P (1st PK), S (2nd PK)
        This program assumes no rotations, so F = F^T and F = u = V (stretches). If there are
        rotations, F must be provided as the strain input.
        The optional res_statev path corresponds to a file with 1s and 0s indicating when to reset
        the state variables.
        The optional exp_temp path corresponds to a file with the temperature of each strain stress
        row.
        '''
        # Check type length
        if len(typ) != 2:
            raise ValueError("There must be only one type of strain and one of stress")
        # Save the stress and strain type
        self.define_exp_input_type(typ[0],typ[1])
        # Load the experimental strain and stress data
        a = np.loadtxt(exp_inp, float)
        self.expstrains  = a[:,:6]      # Strains
        self.expstresses = a[:,6:12]    # Stresses
        # Save the deformation gradient
        self.strains_to_F(self.expstrains)
        # Save the stresses in 2nd PK form
        self.expstressPK2 = self.stress_to_PK2(self.expstresses, typ[1])
        # Save the strains in the umat input type
        self.match_strain_type_umat(self.expstrains)

        # If there is a state variable path
        if res_statev is not None:
            stv = np.loadtxt(res_statev, int)
            if len(stv) != len(a):
                raise ValueError("The number of rows in the reset state variables file must equal the rows in the strain stress file")
            else:
                self.reset_statev = stv
        
        # If there is a temperature path
        if exp_temp is not None:
            ex_temp = np.loadtxt(exp_temp, float)
            if len(ex_temp) != len(a):
                raise ValueError("The number of rows in the temperature file must equal the rows in the strain stress file")
            else:
                self.exp_temperature = ex_temp
                self.has_temp = True            
    
    def define_exp_input_type(self, strain_type: str, stress_type: str) -> None:
        '''Define the input types for stress and strain'''
        self.check_valid_exp_type(strain_type, stress_type)
        self.strain_type = strain_type
        self.stress_type = stress_type

    def check_valid_exp_type(self, strain_type: str, stress_type: str) -> None:
        '''Checks if the strain and stress are of a valid type'''
        if stress_type not in self.valid_input_stress:
            print('Please enter a valid stress (Sigma, P or S)')
            raise ValueError
        if strain_type not in self.valid_input_strain:
            print('Please enter a valid strain (H,C,E or F)')
            raise ValueError

    def exp_inputs(self, strains, stresses) -> None:
        '''throw in strains and stresses directly'''
        self.expstrains  = np.array(strains)
        self.expstresses = np.array(stresses)
        self.strains_to_F(strains)

    def strains_to_F(self, strains) -> None:
        '''Converts the strains from an experiment to deformation gradients. The strain type must have
        been specified before'''
        self.dfgrds = []
        strain_type = self.strain_type
        for strain in strains:
            m_strain = mech.voigt_to_matrix(strain)
            if strain_type == 'C':
                F = mech.cauchystrain_to_dfgrd(m_strain)
            elif strain_type == 'E':
                F = mech.greenstrain_to_dfgrd(m_strain)
            elif strain_type == 'F':
                F = strain
                self.dfgrds.append(F)
                continue
            elif strain_type == 'H':
                F = mech.displgrd_to_dfgrd(m_strain)
            else:
                raise ValueError("The deformation gradient was not computed. Check the function \
                load_inputs in the Umat_wrapper")
            self.dfgrds.append(mech.matrix_to_voigt(F))
        self.dfgrds = np.array(self.dfgrds)

    def match_strain_type_umat(self, strains) -> None:
        '''Saves the strain in the umat input type'''
        self.expstrains_umat = []
        umat_strain_type = self.umat_in_strain_type
        exp_strain_type = self.strain_type
        #aux = np.array([0 for _ in range(3)] for _ in range(3))
        
        if umat_strain_type == exp_strain_type:
            self.expstrains_umat = self.expstrains
        else:
            for strain in strains:
                m_strain = mech.voigt_to_matrix(strain)
                if umat_strain_type == 'C' and exp_strain_type == 'E':
                    aux = mech.greenstrain_to_cauchystrain(m_strain)
                elif umat_strain_type == 'C' and exp_strain_type == 'F':
                    aux = mech.dfgrd_to_cauchystrain(m_strain)
                elif umat_strain_type == 'C' and exp_strain_type == 'H':
                    aux = mech.displgrd_to_cauchystrain(m_strain)
                elif umat_strain_type == 'E' and exp_strain_type == 'C':
                    aux = mech.cauchystrain_to_greenstrain(m_strain)
                elif umat_strain_type == 'E' and exp_strain_type == 'F':
                    aux = mech.dfgrd_to_greenstrain(m_strain)
                elif umat_strain_type == 'E' and exp_strain_type == 'H':
                    aux = mech.displgrd_to_greenstrain(m_strain)
                elif umat_strain_type == 'F' and exp_strain_type == 'E':
                    aux = mech.greenstrain_to_dfgrd(m_strain)
                elif umat_strain_type == 'F' and exp_strain_type == 'C':
                    aux = mech.cauchystrain_to_dfgrd(m_strain)
                elif umat_strain_type == 'F' and exp_strain_type == 'H':
                    aux = mech.displgrd_to_dfgrd(m_strain)
                elif umat_strain_type == 'H' and exp_strain_type == 'E':
                    aux = mech.greenstrain_to_displgrd(m_strain)
                elif umat_strain_type == 'H' and exp_strain_type == 'C':
                    aux = mech.cauchystrain_to_displgrd(m_strain)
                elif umat_strain_type == 'H' and exp_strain_type == 'F':
                    aux = mech.dfgrd_to_displgrd(m_strain)
                else:
                    raise ValueError("Check the function match_strain_type_umat")
                self.expstrains_umat.append(mech.matrix_to_voigt(aux))
            self.expstrains_umat = np.array(self.expstrains_umat)
                    
    def stress_to_PK2(self, stresses, stress_type: str) -> np.array:
        '''Converts stresses to 2nd PK depending on the input type'''
        # Initialize the variable where to store the 2nd PK
        stressPK2 = []
        if stress_type == 'S':
            return stresses
        else:
            for stress, df in zip(stresses, self.dfgrds):
                stress = mech.voigt_to_matrix(stress)
                df = mech.voigt_to_matrix(df)
                if stress_type == 'P':
                    stressPK2.append(mech.matrix_to_voigt(mech.pk1_to_pk2(stress, df)))
                elif stress_type == 'Sigma':
                    stressPK2.append(mech.matrix_to_voigt(mech.cauchystress_to_pk2(stress, df)))
                else:
                    raise ValueError("Something is wrong with the stress type. Check stress_to_PK2")
            return np.array(stressPK2)

    def _stress_component_cost(self, exp_component: np.ndarray,
                               calc_component: np.ndarray, axis: int | None = None) -> float:
        '''Calculates the cost for one Voigt stress component over all data points.'''
        difference = exp_component - calc_component

        if self.cost_type == 'legacy':
            return float(np.sum(difference**2))

        if self.cost_type == 'normalized':
            # Use the complete calibration scale when evaluating a subset.
            # Example: axis=0 selects the saved course/wale machine-axis scale.
            references = getattr(self, 'normalized_stress_reference_maxima', {})
            max_stress = references.get(axis, np.max(np.abs(exp_component)))
            stress_floor = self.normalized_stress_floor_factor * max_stress
            scale = np.abs(exp_component) + stress_floor

            if np.any(scale == 0):
                raise ValueError(
                    "Normalized cost has a zero scale for at least one stress value. "
                    "Use a positive normalized_stress_floor_factor or the legacy cost."
                )

            return float(np.sum((difference / scale)**2))

        raise ValueError(f"Invalid cost_type '{self.cost_type}'")

    def calc_stress_umat(self) -> np.array:
        '''Calculates stresses using the UMat'''
        props = self.m.get_umat_input()
        calcstresses =  []
        for df in self.dfgrds:
            stress = self.call_umat(df, props)
            calcstresses.append(stress)
        return np.array(calcstresses)
    
    def calc_stress_umat_Abaqus(self) -> np.array:
        '''Calculates stresses using the UMat in Abaqus format'''
        self.reset_history()
        props = self.m.get_umat_input()
        calcstresses =  []
        if not self.has_statev and not self.has_temp:   # If there are no state variables, nor temperature
            if self.umat_in_strain_type == 'F':
                for F in self.dfgrds:
                    stress = self.call_umat_Abaqus(F,props)
                    calcstresses.append(stress)
            else:
                for F,strn in zip(self.dfgrds,self.expstrains_umat):
                    stress = self.call_umat_Abaqus(F, props, stran = strn)
                    calcstresses.append(stress)
        
        elif self.has_statev and not self.has_temp:     # If there are state variables, but no temperature
            if self.umat_in_strain_type == 'F':
                for F,res_statv in zip(self.dfgrds,self.reset_statev):
                    if res_statv == 1:
                        stress = self.call_umat_Abaqus(F,props,statev = self.statev_ini_val.copy()) 
                        calcstresses.append(stress)
                    elif res_statv == 0:
                        stress = self.call_umat_Abaqus(F,props,statev = self.statev.copy())
                        calcstresses.append(stress)
                    else:
                        raise ValueError("The state variables reset file must contain only 1s and 0s")
            else:
                for F,strn,res_statv in zip(self.dfgrds,self.expstrains_umat,self.reset_statev):
                    if res_statv == 1:
                        stress = self.call_umat_Abaqus(F, props, stran = strn, statev = self.statev_ini_val.copy())
                        calcstresses.append(stress)
                    elif res_statv == 0:
                        stress = self.call_umat_Abaqus(F, props, stran = strn, statev = self.statev.copy())
                        calcstresses.append(stress)
                    else:
                        raise ValueError("The state variables reset file must contain only 1s and 0s")

        elif not self.has_statev and self.has_temp:     # If there is temperature, but no state variables
            if self.umat_in_strain_type == 'F':
                for F,tmp in zip(self.dfgrds,self.exp_temperature):
                    stress = self.call_umat_Abaqus(F,props,temp = tmp)
                    calcstresses.append(stress) 
            else:
                for F,strn,tmp in zip(self.dfgrds,self.expstrains_umat,self.exp_temperature):
                    stress = self.call_umat_Abaqus(F, props, stran = strn, temp = tmp)
                    calcstresses.append(stress)                
        
        elif self.has_statev and self.has_temp:         # If there are both state variables and temperature
            if self.umat_in_strain_type == 'F':
                for F,res_statv,tmp in zip(self.dfgrds,self.reset_statev,self.exp_temperature):
                    if res_statv == 1:
                        stress = self.call_umat_Abaqus(F,props,statev = self.statev_ini_val.copy(), temp = tmp) 
                        calcstresses.append(stress)
                    elif res_statv == 0:
                        stress = self.call_umat_Abaqus(F,props,statev = self.statev.copy(), temp = tmp)
                        calcstresses.append(stress)
                    else:
                        raise ValueError("The state variables reset file must contain only 1s and 0s")
            else:
                for F,strn,res_statv,tmp in zip(self.dfgrds,self.expstrains_umat,self.reset_statev,self.exp_temperature):
                    if res_statv == 1:
                        stress = self.call_umat_Abaqus(F, props, stran = strn, statev = self.statev_ini_val.copy(), temp = tmp)
                        calcstresses.append(stress)
                    elif res_statv == 0:
                        stress = self.call_umat_Abaqus(F, props, stran = strn, statev = self.statev.copy(), temp = tmp)
                        calcstresses.append(stress)
                    else:
                        raise ValueError("The state variables reset file must contain only 1s and 0s")
        
        else:
            raise ValueError("Check the calc_stress_umat_Abaqus function. This shouldn't be an option")
        
        return np.array(calcstresses)
    
    def calc_stress_umat_Abaqus_GPT(self) -> np.array:
        '''Calculates stresses using the UMat in Abaqus format'''
        props = self.m.get_umat_input()
        calcstresses = []

        def append_stress(F, strain=None, statev=None, temp=292):
            stress = self.call_umat_Abaqus(F, props, stran=strain, statev=statev, temp=temp)
            calcstresses.append(stress)

        if self.umat_in_strain_type == 'F':
            strains = self.expstrains
        else:
            strains = zip(self.dfgrds, self.expstrains_umat)
        
        for item in (strains if isinstance(strains, list) else list(strains)):
            if self.has_statev and self.has_temp:
                for (F, strn), res_statv, tmp in zip(item, self.reset_statev, self.exp_temperature):
                    statev = self.statev_ini_val if res_statv == 1 else self.statev if res_statv == 0 else None
                    if statev is None:
                        raise ValueError("The state variables reset file must contain only 1s and 0s")
                    append_stress(F, strn, statev, tmp)
            elif self.has_statev:
                for (F, strn), res_statv in zip(item, self.reset_statev):
                    statev = self.statev_ini_val if res_statv == 1 else self.statev if res_statv == 0 else None
                    if statev is None:
                        raise ValueError("The state variables reset file must contain only 1s and 0s")
                    append_stress(F, strn, statev)
            elif self.has_temp:
                for (F, strn), tmp in zip(item, self.exp_temperature):
                    append_stress(F, strn, temp=tmp)
            else:
                if self.umat_in_strain_type == 'F':
                    append_stress(item)
                else:
                    append_stress(*item)

        return np.array(calcstresses)
        
    def cost_calculations(self, get_plots = False, axis = []) -> float:
        '''will calculate the cost for a parameter set with a given input,
           if axis = (0) gets cost of 0 axis, can also be (0,1,2)'''
        # The cost is the difference between the calculated and the 
        # given stresses. The comparisons are made in terms of PK2
        try:
            calcstresses = self.calc_stress_umat_Abaqus()
        except TypeError as te:
            raise TypeError(f"There was a type error at the calc_stress_umat function: {te}")
        except Exception as e:
            raise Exception(f"There was an error with calc_stress_umat: {e}")
        #here I could specify a certain axis to get only the desired cost function of that axis 
        if len(axis) == 0:    
            print('WARNING! NO COMPONENT SPECIFIED!!!''')
            cost = 0
            if get_plots:
                return ([0],[0],0)
        
        else:
            if not self.umat_out_stress_type == 'S':
                if self.umat_out_stress_type == 'Sigma':
                    calcstresses = self.stress_to_PK2(calcstresses, 'Sigma')
                elif self.umat_out_stress_type == 'P':
                    calcstresses = self.stress_to_PK2(calcstresses, 'P')
                else:
                    raise ValueError("There is a problem with the UMat stress output type")
            
            cost = 0
            for i in axis:
                cost += self._stress_component_cost(self.expstressPK2[:, i], calcstresses[:, i], axis=i)
            if get_plots:
                data = []
                for ax in axis:
                    cost = self._stress_component_cost(self.expstressPK2[:, ax], calcstresses[:, ax], axis=ax)
                    data.append((self.expstrains[:,ax], calcstresses[:,ax], cost))
                return data

        return cost

def compile_umat(umat_path: pathlib.Path) -> None:
    '''Compile Umat'''
    try:
        sb.call(f'python -m numpy.f2py -c --fcompiler=gnu95 --compiler=mingw32 {umat_path} -m umat_new',shell=True)
        import umat_new as umatf
        print(f'The UMat {umat_path.name} was succesfully compiled and imported')
    except Exception as e:
        print("There was a problem compiling the UMat: ",e)

def compile_umat_meson(umat_path: pathlib.Path) -> None:
    '''Function to compile UMAT using meson'''
    global umatf

    if not umat_path.exists():
        raise FileNotFoundError(f"The file {umat_path} does not exist.")

    build_dir = pathlib.Path('builddir').absolute()
    shared_lib_path = build_dir / ('libumat_fort.dll' if os.name == 'nt' else 'libumat_fort.dylib' if sys.platform == 'darwin' else 'libumat_fort.so')
    python_dir = pathlib.Path(sys.executable).resolve().parent
    tool_dirs = [
        python_dir,
        python_dir / 'Scripts',
        python_dir / 'Library' / 'bin',
        python_dir / 'Library' / 'x86_64-w64-mingw32' / 'bin',
    ]
    build_env = os.environ.copy()
    build_env['PATH'] = os.pathsep.join(
        str(tool_dir) for tool_dir in tool_dirs if tool_dir.exists()
    ) + os.pathsep + build_env.get('PATH', '')

    # Generate the meson.build file with the correctly formatted path
    meson_content = f"""
project('UMAT_Project', 'fortran')

fortran_compiler = meson.get_compiler('fortran')

shared_library('umat_fort', '{umat_path.as_posix()}')
    """
    with open('meson.build', 'w') as meson_file:
        meson_file.write(meson_content)

    # Set the Fortran compiler environment variable
    compiler = (
        shutil.which('gfortran', path=build_env['PATH'])
        or shutil.which('x86_64-w64-mingw32-gfortran', path=build_env['PATH'])
    )
    if compiler is None:
        raise RuntimeError(
            "No Fortran compiler was found. Install one in the active conda "
            "environment, for example: conda install -n sv -c conda-forge "
            "fortran-compiler"
        )
    if shutil.which('ninja', path=build_env['PATH']) is None:
        raise RuntimeError(
            "Ninja was not found. Install it in the active conda environment, "
            "for example: conda install -n sv ninja"
        )
    archiver = (
        shutil.which('gcc-ar', path=build_env['PATH'])
        or shutil.which('x86_64-w64-mingw32-gcc-ar', path=build_env['PATH'])
        or shutil.which('ar', path=build_env['PATH'])
        or shutil.which('x86_64-w64-mingw32-ar', path=build_env['PATH'])
    )
    if archiver is not None:
        build_env['AR'] = archiver
    ranlib = (
        shutil.which('gcc-ranlib', path=build_env['PATH'])
        or shutil.which('x86_64-w64-mingw32-gcc-ranlib', path=build_env['PATH'])
        or shutil.which('ranlib', path=build_env['PATH'])
        or shutil.which('x86_64-w64-mingw32-ranlib', path=build_env['PATH'])
    )
    if ranlib is not None:
        build_env['RANLIB'] = ranlib
    build_env['FC'] = compiler
    meson_cmd = [sys.executable, '-m', 'mesonbuild.mesonmain']

    # Build the project using Meson and Ninja
    try:
        active_build_dir = build_dir
        result = sb.run(meson_cmd + ['setup', str(active_build_dir), '--reconfigure'], capture_output=True, text=True, env=build_env)
        if result.returncode != 0 and build_dir.exists():
            print(result.stdout)
            print(result.stderr)
            active_build_dir = pathlib.Path(tempfile.gettempdir()) / f"ssmccf_meson_{umat_path.stem}_{os.getpid()}"
            print(f"Meson reconfigure failed; retrying in {active_build_dir}.")
            result = sb.run(meson_cmd + ['setup', str(active_build_dir)], capture_output=True, text=True, env=build_env)
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()

        result = sb.run(meson_cmd + ['compile', '-C', str(active_build_dir)], capture_output=True, text=True, env=build_env)
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()

    except sb.CalledProcessError as e:
        print(f"An error occurred during the build process: {e}")
        sys.exit(1)

    # Check if the shared library file is created
    compiled_module_path = active_build_dir
    shared_lib = compiled_module_path / ('libumat_fort.dll' if os.name == 'nt' else 'libumat_fort.dylib' if sys.platform == 'darwin' else 'libumat_fort.so')
    if shared_lib.exists():
        print(f"Shared library {shared_lib} found.")
    else:
        print(f"Shared library {shared_lib} not found.")
        sys.exit(1)

    # Attempt to import the shared library as a Python module
    try:
        umatf = ctypes.CDLL(str(shared_lib))
        print(f"Successfully loaded umat {umat_path.name} shared library")
        #list_symbols(shared_lib)
        # Define the argument and return types for the Fortran function
        try:
            umatf.myumat_.argtypes = [
                ctypes.POINTER(ctypes.c_double), # stress
                ctypes.POINTER(ctypes.c_double), # props
                ctypes.POINTER(ctypes.c_double), # dfgrd1
                ctypes.POINTER(ctypes.c_int)     # nprops
            ]
            umatf.myumat_.restype = None
            print(f"The function myumat was found in {umat_path.name} and is ready to be used")
        except Exception as e:
            print(f"The function myumat was not found in {umat_path.name}")

        try:
            umatf.umat_.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # STRESS
                ctypes.POINTER(ctypes.c_double),  # STATEV
                ctypes.POINTER(ctypes.c_double),  # DDSDDE
                ctypes.POINTER(ctypes.c_double),  # SSE
                ctypes.POINTER(ctypes.c_double),  # SPD
                ctypes.POINTER(ctypes.c_double),  # SCD
                ctypes.POINTER(ctypes.c_double),  # RPL
                ctypes.POINTER(ctypes.c_double),  # DDSDDT
                ctypes.POINTER(ctypes.c_double),  # DRPLDE
                ctypes.POINTER(ctypes.c_double),  # DRPLDT
                ctypes.POINTER(ctypes.c_double),  # STRAN
                ctypes.POINTER(ctypes.c_double),  # DSTRAN
                ctypes.POINTER(ctypes.c_double),  # TIME
                ctypes.POINTER(ctypes.c_double),  # DTIME
                ctypes.POINTER(ctypes.c_double),  # TEMP
                ctypes.POINTER(ctypes.c_double),  # DTEMP
                ctypes.POINTER(ctypes.c_double),  # PREDEF
                ctypes.POINTER(ctypes.c_double),  # DPRED
                ctypes.c_char_p,                  # CMNAME
                ctypes.POINTER(ctypes.c_int),     # NDI
                ctypes.POINTER(ctypes.c_int),     # NSHR
                ctypes.POINTER(ctypes.c_int),     # NTENS
                ctypes.POINTER(ctypes.c_int),     # NSTATV
                ctypes.POINTER(ctypes.c_double),  # PROPS
                ctypes.POINTER(ctypes.c_int),     # NPROPS
                ctypes.POINTER(ctypes.c_double),  # COORDS
                ctypes.POINTER(ctypes.c_double),  # DROT
                ctypes.POINTER(ctypes.c_double),  # PNEWDT
                ctypes.POINTER(ctypes.c_double),  # CELENT
                ctypes.POINTER(ctypes.c_double),  # DFGRD0
                ctypes.POINTER(ctypes.c_double),  # DFGRD1
                ctypes.POINTER(ctypes.c_int),     # NOEL
                ctypes.POINTER(ctypes.c_int),     # NPT
                ctypes.POINTER(ctypes.c_int),     # LAYER
                ctypes.POINTER(ctypes.c_int),     # KSPT
                ctypes.POINTER(ctypes.c_int),     # KSTEP
                ctypes.POINTER(ctypes.c_int),     # KINC
            ]
            umatf.umat_.restype = None
            print(f"The function umat was found in {umat_path.name} and is ready to be used")
        
        except Exception as e:
            print(f"The function umat was not found in {umat_path.name}")
        
    except OSError as e:
        print(f"Failed to load shared library: {e}")
        sys.exit(1)

def list_symbols(shared_lib):
    '''List all symbols in the shared library'''
    print("Listing symbols in the shared library:")
    output = sb.check_output(['dumpbin', '/exports', str(shared_lib)], shell=True)
    print(output.decode())

def generate_exp_data(nPts: int, min_strain: float, max_strain: float, exp_type: list[str], parameters: np.array, 
                      umat_type: list[str], statev_init: np.array = None, min_temp: float = None, max_temp: float = None, 
                      strains: np.array = None) -> np.array:
    '''Uses the compiled UMAT to generate virtual experimental data. Returns the data in the experiment format (strain and stresses)'''
    # If the strains are provided, use them. Else, generate them with the min and max
    if strains is None:
        strans = np.linspace(min_strain,max_strain,nPts)
        strans = np.concatenate((strans[:, np.newaxis], np.zeros((nPts,5))),axis = 1)
    else:
        strans = strains
        nPts = len(strans)
    # Save state variables
    statev = statev_init

    strain_type = exp_type[0]
    umat_inp_strain_type = umat_type[0]

    if min_temp == None or max_temp == None:
        temps = [None] * nPts
    else:
        temps = np.linspace(min_temp, max_temp, nPts)
    stresses = np.zeros((nPts,6))

    # Convert to UMAT type strains
    strain_types = {
        'UMAT': umat_inp_strain_type,
        'Exp': strain_type
    }
    strans_umat_type = match_strains_type(strain_types, strans)
    # Now the strains are in umat type

    # Calculate deformation gradients
    dfgrd = strains_to_dfgrd(strans, strain_types['Exp'])

    # Loop calling the UMAT
    for index, (stran, temp, F) in enumerate(zip(strans_umat_type, temps, dfgrd)):
        stresses[index], statev = call_umat_Abqs_sta(F, parameters, statev, temp, stran)

    # Convert stresses to match the experiments type
    stress_types = {
        'UMAT': umat_type[1],
        'Exp': exp_type[1]
    }
    stresses_exp_type = match_stress_type(stress_types, stresses, dfgrd)

    exp_data = np.concatenate((strans,stresses_exp_type), axis = 1)
    return exp_data
    
@overload
def call_umat_Abqs_sta(
    F: Any,
    props: Any,
    statev: Any = None,
    temp: Any = None,
    stran: Any = None,
    *,
    return_tangent: Literal[False] = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def call_umat_Abqs_sta(
    F: Any,
    props: Any,
    statev: Any = None,
    temp: Any = None,
    stran: Any = None,
    *,
    return_tangent: Literal[True],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


def call_umat_Abqs_sta(
    F: Any,
    props: Any,
    statev: Any = None,
    temp: Any = None,
    stran: Any = None,
    *,
    return_tangent: bool = False,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    '''Calls the compiled UMAT to calculate a stress without the need of the class Umat_Wrapper'''
    global umatf    # Declare the umatf library as global to use it

    # Initialize all the UMAT variables
    ntens = 6   # Number of strain and stress components
    if statev is None:
        statev = np.asfortranarray(np.zeros(1, dtype=np.double))    # State variables
    if stran is None:
        stran = np.asfortranarray(np.zeros(ntens, dtype=np.double)) # Strain
    if temp is None:
        temp = 292.0    # Temperature
    nprops = len(props)  # Number of material properties
    stress = np.zeros(ntens, dtype=np.double)  # Pass stress as empty vector
    nstatv = len(statev)  # Number of state variables
    ddsdde = np.zeros((ntens, ntens), dtype=np.double)  # Stiffness matrix initialized as 0
    time = np.zeros(2, dtype=np.double)  # Time
    dtime = 0.0  # Time variation
    sse = 0.0  # Specific elastic strain energy
    spd = 0.0  # Specific plastic dissipation
    scd = 0.0  # Specific creep dissipation
    rpl = 0.0  # Volumetric heat generation per unit time at the end of the increment
    ddsddt = np.zeros(ntens, dtype=np.double)  # Variation of the stress increments with respect to the temperature
    drplde = np.zeros(ntens, dtype=np.double)  # Variation of RPL with respect to the strain increments
    drpldt = 0.0  # Variation of RPL with respect to the temperature
    pnewdt = 1.0  # Ratio of suggested new time increment to time being used
    dstran = np.zeros(ntens, dtype=np.double)  # Array of strain increments
    dtemp = 0.0  # Increment of temperature
    predef = np.zeros(1, dtype=np.double)  # Array of interpolated values of predefined field variables
    dpred = np.zeros(1, dtype=np.double)  # Array of increments of predefined field variables
    cmname = "UMAT"  # User-defined material name, left justified
    ndi = 3  # Number of direct stress components at this point (3 in 3D)
    nshr = 3  # Number of engineering shear stress components at this point (3 in 3D)
    coords = np.zeros(3, dtype=np.double)  # An array containing the coordinates of this point
    drot = np.eye(3, dtype=np.double)  # Rotation increment matrix. Identity by default.
    celent = 1.0  # Characteristic element length
    dfgrd0 = mech.voigt_to_matrix(F)    # Array containing the deformation gradient at the beginning of the increment
    dfgrd1 = mech.voigt_to_matrix(F)    # Array containing the deformation gradient at the end of the increment
    noel = 1    # Element number
    npt = 1     # Integration point number
    layer = 1   # Layer number (for composite shells and layered solids)
    kspt = 1    # Section point number within the current layer
    kstep = 1   # Step number
    kinc = 1    # Increment number

    # Ensure the arrays are contiguous and of type double
    stress = np.ascontiguousarray(stress, dtype=np.double)
    statev = np.ascontiguousarray(statev, dtype=np.double)
    ddsdde = np.ascontiguousarray(ddsdde.flatten(), dtype=np.double)
    sse = ctypes.c_double(sse)
    spd = ctypes.c_double(spd)
    scd = ctypes.c_double(scd)
    rpl = ctypes.c_double(rpl)
    ddsddt = np.ascontiguousarray(ddsddt, dtype=np.double)
    drplde = np.ascontiguousarray(drplde, dtype=np.double)
    drpldt = ctypes.c_double(drpldt)
    stran = np.ascontiguousarray(stran, dtype=np.double)
    dstran = np.ascontiguousarray(dstran, dtype=np.double)
    time = np.ascontiguousarray(time, dtype=np.double)
    dtime = ctypes.c_double(dtime)
    temp = ctypes.c_double(temp)
    dtemp = ctypes.c_double(dtemp)
    predef = np.ascontiguousarray(predef, dtype=np.double)
    dpred = np.ascontiguousarray(dpred, dtype=np.double)
    cmname_buffer = ctypes.create_string_buffer(cmname.encode('utf-8'), 80)
    ndi = ctypes.c_int(ndi)
    nshr = ctypes.c_int(nshr)
    ntens = ctypes.c_int(ntens)
    nstatv = ctypes.c_int(nstatv)
    props = np.ascontiguousarray(props, dtype=np.double)
    nprops = ctypes.c_int(nprops)
    coords = np.ascontiguousarray(coords, dtype=np.double)
    drot = np.ascontiguousarray(drot.flatten(), dtype=np.double)
    pnewdt = ctypes.c_double(pnewdt)
    celent = ctypes.c_double(celent)
    dfgrd0 = np.ascontiguousarray(dfgrd0.flatten(), dtype=np.double)
    dfgrd1 = np.ascontiguousarray(dfgrd1.flatten(), dtype=np.double)
    noel = ctypes.c_int(noel)
    npt = ctypes.c_int(npt)
    layer = ctypes.c_int(layer)
    kspt = ctypes.c_int(kspt)
    kstep = ctypes.c_int(kstep)
    kinc = ctypes.c_int(kinc)
    
    # Call the UMat coded in Fortran
    umatf.umat_(
        stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        statev.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ddsdde.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(sse),
        ctypes.byref(spd),
        ctypes.byref(scd),
        ctypes.byref(rpl),
        ddsddt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        drplde.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(drpldt),
        stran.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dstran.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(dtime),
        ctypes.byref(temp),
        ctypes.byref(dtemp),
        predef.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dpred.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cmname_buffer,
        ctypes.byref(ndi),
        ctypes.byref(nshr),
        ctypes.byref(ntens),
        ctypes.byref(nstatv),
        props.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(nprops),
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        drot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(pnewdt),
        ctypes.byref(celent),
        dfgrd0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dfgrd1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(noel),
        ctypes.byref(npt),
        ctypes.byref(layer),
        ctypes.byref(kspt),
        ctypes.byref(kstep),
        ctypes.byref(kinc)
    )
    
    if return_tangent:
        tangent = ddsdde.reshape((6, 6), order="F").copy()
        return stress, statev, tangent

    # Return the stress and state variables
    return stress, statev

def match_strains_type(strain_types: dict, strains: np.array) -> np.array:
    '''Turns the experiment strain into the UMAT input type of strain'''
    expstrains_umat = []
    umat_strain_type = strain_types['UMAT']
    exp_strain_type = strain_types["Exp"]
    
    if umat_strain_type == exp_strain_type:
        return strains
    else:
        for strain in strains:
            m_strain = mech.voigt_to_matrix(strain)
            if umat_strain_type == 'C' and exp_strain_type == 'E':
                aux = mech.greenstrain_to_cauchystrain(m_strain)
            elif umat_strain_type == 'C' and exp_strain_type == 'F':
                aux = mech.dfgrd_to_cauchystrain(m_strain)
            elif umat_strain_type == 'C' and exp_strain_type == 'H':
                aux = mech.displgrd_to_cauchystrain(m_strain)
            elif umat_strain_type == 'E' and exp_strain_type == 'C':
                aux = mech.cauchystrain_to_greenstrain(m_strain)
            elif umat_strain_type == 'E' and exp_strain_type == 'F':
                aux = mech.dfgrd_to_greenstrain(m_strain)
            elif umat_strain_type == 'E' and exp_strain_type == 'H':
                aux = mech.displgrd_to_greenstrain(m_strain)
            elif umat_strain_type == 'F' and exp_strain_type == 'E':
                aux = mech.greenstrain_to_dfgrd(m_strain)
            elif umat_strain_type == 'F' and exp_strain_type == 'C':
                aux = mech.cauchystrain_to_dfgrd(m_strain)
            elif umat_strain_type == 'F' and exp_strain_type == 'H':
                aux = mech.displgrd_to_dfgrd(m_strain)
            elif umat_strain_type == 'H' and exp_strain_type == 'E':
                aux = mech.greenstrain_to_displgrd(m_strain)
            elif umat_strain_type == 'H' and exp_strain_type == 'C':
                aux = mech.cauchystrain_to_displgrd(m_strain)
            elif umat_strain_type == 'H' and exp_strain_type == 'F':
                aux = mech.dfgrd_to_displgrd(m_strain)
            else:
                raise ValueError("Check the function match_strain_type_umat")
            expstrains_umat.append(mech.matrix_to_voigt(aux))
        return np.array(expstrains_umat)

def match_stress_type(stress_types: dict, stress_exp: np.array, defgrad: np.array) -> np.array:
    '''Converts the stress from UMAT type to experiment type to make them comparable. Uses the experimental strains
    to calculate the deformation gradients.'''
    umatstress_exp_f = []
    umat_stress_type = stress_types['UMAT']
    exp_stress_type = stress_types['Exp']

    if umat_stress_type == exp_stress_type:
        umatstress_exp_f = stress_exp
    else:
        for stre, F in zip(stress_exp, defgrad):
            m_stress = mech.voigt_to_matrix(stre)
            m_F = mech.voigt_to_matrix(F)
            if umat_stress_type == 'Sigma' and exp_stress_type == 'P':
                aux = mech.cauchystress_to_pk1(m_stress, m_F)
            elif umat_stress_type == 'Sigma' and exp_stress_type == 'S':
                aux = mech.cauchystress_to_pk2(m_stress, m_F)
            elif umat_stress_type == 'P' and exp_stress_type == 'S':
                aux = mech.pk1_to_pk2(m_stress, m_F)
            elif umat_stress_type == 'P' and exp_stress_type == 'Sigma':
                aux = mech.pk1_to_cauchystress(m_stress, m_F)
            elif umat_stress_type == 'S' and exp_stress_type == 'P':
                aux = mech.pk2_to_pk1(m_stress, m_F)
            elif umat_stress_type == 'S' and exp_stress_type == 'Sigma':
                aux = mech.pk2_to_cauchystress(m_stress, m_F)
            else:
                raise ValueError("The allowed stress types are 'Sigma', 'P', and 'S'")
            umatstress_exp_f.append(mech.matrix_to_voigt(aux))
    
    return np.array(umatstress_exp_f)
            
def strains_to_dfgrd(strains: np.array, strain_type: str) -> np.array:
    '''Converts the strains to deformation gradients'''
    dfgrds = []

    for strain in strains:
        m_strain = mech.voigt_to_matrix(strain)
        if strain_type == 'C':
            F = mech.cauchystrain_to_dfgrd(m_strain)
        elif strain_type == 'E':
            F = mech.greenstrain_to_dfgrd(m_strain)
        elif strain_type == 'F':
            F = strain
            dfgrds.append(F)
            continue
        elif strain_type == 'H':
            F = mech.displgrd_to_dfgrd(m_strain)
        else:
            raise ValueError(f"The deformation gradient was not computed. Check the function load_inputs in the Umat_wrapper")
        dfgrds.append(mech.matrix_to_voigt(F))
    
    return np.array(dfgrds)

def str_to_bool(text: str) -> bool:
    '''    
    Converts the string "True" or "False" to their respective boolean values.
    '''
    
    if text not in ('True', 'False'):
        raise ValueError("The fixed parameters value can be 'True' or 'False'")
    
    return text == 'True'

if __name__ == '__main__':
    main()
