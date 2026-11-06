import numpy as np
import random
import math
import copy

class Optimization():
    '''The meat of the code. A class that can perform simulated annealing of the
       Monte Carlo Method to find local minima of propertie combinations. The cost
       function is defined as the pointwise square function between input stresses and
       calculated stresses'''
    def __init__(self, umat, temp: float, stepsize: float):
        '''An instance of an Obptimization class, this class is able to
           perform changes on a given dictionary of properties and will
           yield the ideal collection of parameters for a given experi-
           mental curve'''
        # Exclude fixed parameters from optimization
        opt_props = []
        prop_dic = umat.get_props_dic()
        for prop in prop_dic:
            if not prop_dic[prop][5]:
                opt_props.append(prop)    
        self.to_optimize = opt_props

        self.mc_steps = 100
        self.temp = temp 
        self.u = umat
        self.stepsize = stepsize
        self.axis = [0,1,2,3,4,5]
        self.current_cost = None
        self.best_cost = float('inf')
        self.best_props = self._copy_props()

    def _copy_props(self) -> dict:
        '''Returns a detached copy of the current material parameters.'''
        return copy.deepcopy(self.u.get_props_dic())

    def _restore_props(self, props: dict) -> None:
        '''Restores a material parameter dictionary into the wrapped UMAT.'''
        for key, (value, ty, mini, maxi, nr, fix_par) in props.items():
            self.u.setv(key, value, ty, mini, maxi, nr, fix_par)

    def _initialize_cost_tracking(self) -> None:
        '''Initializes current and best cost from the current parameter state.'''
        if self.current_cost is None:
            self.current_cost = self.u.cost_calculations(get_plots=False, axis=self.axis)
            self.best_cost = self.current_cost
            self.best_props = self._copy_props()

    def _record_best_state(self, cost: float) -> None:
        '''Stores the current parameters when they improve the best cost.'''
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_props = self._copy_props()

    def restore_best_state(self) -> None:
        '''Restores the lowest-cost parameter set visited so far.'''
        self._restore_props(self.best_props)
        self.current_cost = self.best_cost

    def get_best_state(self) -> tuple[dict, float]:
        '''Returns a copy of the best parameter dictionary and its cost.'''
        return copy.deepcopy(self.best_props), self.best_cost
    
    def set_stepsize(self, stepsize: float) -> None:
        '''Define the stepsize for a property. (1 if int, 0.01 if float by default)'''
        self.stepsize = stepsize

    def barplot(self):
        '''returns a the names of parameters and the ctr value for a barplot'''
        props = self.u.get_props_dic()
        sort = self.u.m.get_sorted_props()
    
        xs = []
        ys = []
        for key, v in sort:
            xs.append(key)
            ys.append(props[key][-1])
        return xs, ys

    def simulated_annealing(self, howoften = 10, howmuch = 0.9, mc_steps = 1000, exclusion_threshold=False, change_stepsize = 0.9):
        '''use simulated annealing to go lower on the cost surface.
           will repeat monte carlo simulations'''
        self.mc_steps = mc_steps
        self._initialize_cost_tracking()
        for idx in range(howoften):
            self.monte_carlo(exclusion_threshold = exclusion_threshold)
            self.temp *= howmuch
            self.stepsize *= change_stepsize
            print('Finished Annealing Step', idx+1, 'of',howoften)
        self.restore_best_state()
        print('Best Cost:', self.best_cost)

    def set_comparison(self, strains, stresses):  
        '''Load in the strain - stress curve that will be used for 
           comparison. The strain will serve as basis for the defor-
           mation gradients that will be used in the evaluation of the
           umat''' 
        self.ref_strains = np.array(strains)
        self.ref_stresses = np.array(stresses)
        self.u.exp_inputs(strains, stresses)
        self.to_optimize = list(self.u.m.get_keys())        

    def get_props_dic(self):
        '''gets the underlying properties dictionary'''
        return self.u.get_props_dic()

    def set_temperature(self, temp) -> None:
        '''Set the starting temperature for simualted annealing'''
        try:
            self.temp = float(temp)
        except Exception as e:
            print('There is an error with the temperature: ', e)

    def set_duration(self, duration) -> None:
        '''Set the number of MC Steps that will be performed before decrease
           in temperature'''
        try:
            self.mc_steps = int(float(duration))
        except Exception as e:
            print('Invalid duration: ', e)

    def set_optimization_set(self, optimize: list) -> None:
        '''define the list of parameters that should be optimized, all otherwise'''
        self.to_optimize = optimize
    
    def set_axis(self, axis: list) -> None:
        '''set the axis you want to investigate'''
        self.axis = axis

    def exclude_props(self, exclude: list) -> None:
        '''takes a list of properties that shall be excluded from optimization'''
        print('EXCLUDE FROM SEARCH:', exclude)
        for item in exclude:
            self.to_optimize.remove(item)

    def exclude_negative(self, threshold = 50) -> None:
        '''search through properties, if one has a ctr > threshold, remove from
           optimization list'''
        exclude = []
        for key in self.get_props_dic():
            val, dtype, mini, maxi, ctr, fix_par = self.u.getv(key)
            if ctr > threshold:
                exclude.append(key)
        self.exclude_props(exclude)

    def monte_carlo(self, exclusion_threshold = False):
        '''Run a monte carlo simulation. Uses simulated annealing
           to find local minima of the property landscape, based on cost
           function evaluation for all interesting directions
           Stress can either be in 0, 1 or 0 and 1 direction
           If exclusion is an integer after each MC run parameters are 
           tested for impact. Only necesssary at the beginning'''
        kbtlike= self.temp
        epsilon = 0.1 
        ACCEPTED = 0
        COSTS = []
        self._initialize_cost_tracking()
        old_cost = self.current_cost

        for idx in range(self.mc_steps):

            change = random.choice(self.to_optimize)    # Randomly select one parameter
            val, dtype, mini, maxi, ctr, fix_par = self.u.getv(change)
            old_state = (val, dtype, mini, maxi, ctr, fix_par)

            if dtype == int:
                stepsize = 1
            else:
                stepsize = (maxi - mini) * float(self.stepsize) #change in 1% steps
 
            signum = random.choice([-1,1])
            ival = val + signum * stepsize 
           
            if ival < mini or ival > maxi: 
                #went to far, so turn back!
                ival = val + -1* signum * stepsize 
                
            self.u.setv(change,ival, dtype, mini, maxi, ctr, fix_par)            

            #make sure that the correct cost is calculated by specifing
            #the axis
            new_cost = self.u.cost_calculations(get_plots=False, axis = self.axis)
            
            #the monte carlo step
            dE = new_cost - old_cost
            if abs(dE) < epsilon:
                #count the properties that do not change Energy!
                ctr -= 1
            else:
                ctr += 1 
            self.u.setv(change,ival, dtype, mini, maxi, ctr, fix_par)            
            
            if dE < 0:
                #accepted step
                old_cost = new_cost
                self.current_cost = new_cost
                self._record_best_state(new_cost)
                ACCEPTED += 1
                COSTS.append(old_cost)            

            elif math.exp(- dE /float(kbtlike) ) > random.uniform(0,1):
                #accepted step
                old_cost = new_cost
                self.current_cost = new_cost
                self._record_best_state(new_cost)
                ACCEPTED += 1
                COSTS.append(old_cost)            
            else:
                #undo the changes
                self.u.setv(change, *old_state)            

            if idx % 100 == 0:
                print('Step ',idx, 'of', int(self.mc_steps), old_cost)
                

        #dict_tester(DDD.get_start_parameters(),self.get_props_dic())
        print('Acceptence Ratio:', ACCEPTED/float(self.mc_steps))
        if exclusion_threshold:
            self.exclude_negative(exclusion_threshold)
