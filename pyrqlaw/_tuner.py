import pygmo as pg
import numpy as np

from ._convergence import check_convergence_oe

class rqtof_udp:
    """User-Defined Problem returning the time of flight of the trajectory."""
    def __init__(self, obj_fcn, stage, tune_bounds):
        self._fitness = obj_fcn
        self._stage = stage
        self._tune_bounds = tune_bounds

    def fitness(self, x):
        return [self._fitness(weights=x, tune_bounds=self._tune_bounds)]

    def get_bounds(self):
        if self._stage == "Stage 1":
            lbs = np.full((5,), 0)
            ubs = np.full((5,), 1)
        elif self._stage == "Stage 2":
            lbs = np.full((7,), 0)
            ubs = np.full((7,), 1)
        return (lbs, ubs)
        

class RQLawTuner:
    def __init__(self, prob, stage="Stage 2", tune_bounds=[50]*5+[1,10]):
        """Object to tune RQ-Law algorithm on the 
            given problem instance and stage.
            
        Args:
            prob (RQLaw): RQLaw instance
            stage (str): stage of the maneuver over which to tune the algorithm
            tune_bounds (list): bounds on the weights to tune
        """
        self._prob = prob
        self._stage = stage
        self._tune_bounds = np.array(tune_bounds)
        self._obj_fcn = None
        self._x0_at_scale = None
        self._x0 = None # nondimensional
        self._x_at_scale = None
        self._x = None # nondimensional
        self._f = None

        # Check RQ-Law is worth tuning over the given stage
        if stage == "Stage 1":
            run_tuning = not check_convergence_oe(self._prob.oe0, 
                                                    self._prob.oeT, 
                                                    self._prob.woe1, 
                                                    self._prob.wl1, 
                                                    self._prob.tol_oe)
            assert run_tuning==True, "the chaser and target are already in the same orbit, within tolerance"
            self._obj_fcn = self._prob.solve_stage1
            self._x0_at_scale = self._prob.woe1
            self._x0 = self._x0_at_scale / self._tune_bounds
        elif stage == "Stage 2":
            deltaL = self._prob.eval_deltaL(self._prob.oe0[5], self._prob.oeT[5]) 
            run_tuning = not check_convergence_oe(self._prob.oe0, 
                                                    self._prob.oeT, 
                                                    self._prob.woe2, 
                                                    self._prob.wl2, 
                                                    self._prob.tol_oe,
                                                    deltaL=deltaL)
            assert run_tuning==True, "the chaser's and target's state vectors are equal, within tolerance"
            self._obj_fcn = self._prob.solve_stage2
            self._x0_at_scale = np.concatenate((self._prob.woe2, np.array([self._prob.wl2, self._prob.wscl2])))
            self._x0 = self._x0_at_scale / self._tune_bounds
        else:
            raise Exception("Please specify either 'Stage 1' or 'Stage 2' for tuning RQ-Law.")
        return

    def tune(self, pop_size=1, verbosity=1, solver="neldermead"):
        """Tune the RQ-Law algo using the Nelder-Mead algorithm.
        
        Args:
            pop_size (int): size of the population of chromosomes to evaluate in Pygmo
            verbosity (int): solver's verbosity
            solver (str): Pygmo solver
        """
        # Define Pygmo problem
        opt_prob = pg.problem(udp=rqtof_udp(self._obj_fcn, self._stage, self._tune_bounds))
        print(opt_prob)

        # Define Pygmo algorithm
        if solver == "neldermead":
            nl_algo = pg.nlopt(solver="neldermead")
            nl_algo.xtol_rel = 1e-3
            nl_algo.ftol_rel = 1e-3
            nl_algo.maxtime = 1800
            algo = pg.algorithm(uda=nl_algo)
        elif solver == "de":
            de_algo = pg.de(gen=20, F=0.8, CR=0.9, variant=2, ftol=1e-6, xtol=1e-6, seed=0)
            algo = pg.algorithm(uda=de_algo)
        algo.set_verbosity(verbosity)

        # Define Pygmo population
        pop = pg.population(opt_prob, size=pop_size-1, seed=0)
        pop.push_back(x=self._x0)

        # Optimize
        pop = algo.evolve(pop)
        self._x = pop.champion_x
        self._f = pop.champion_f
        self._x_at_scale = self._x * self._tune_bounds
        return 

    def tune_parallel(self, pop_size=1, n_isl=4, verbosity=1): # Does not seem to be working as expected -> investigate
        """Tune the RQ-Law algo using the Nelder-Mead algorithm in parallel.

        Args:
            pop_size (int): size of the population of chromosomes to evaluate in Pygmo
            verbosity (int): solver's verbosity
            n_isl (int): number of islands to parallelize optimization over
        """
        # Define Pygmo problem
        opt_prob = pg.problem(udp=rqtof_udp(self._obj_fcn, self._stage, self._tune_bounds))
        print(opt_prob)

        # Define Pygmo algorithm
        nl_algo = pg.nlopt(solver="neldermead")
        nl_algo.xtol_rel = 1e-3
        nl_algo.ftol_rel = 1e-3
        nl_algo.maxtime = 1800
        algo = pg.algorithm(uda=nl_algo)
        algo.set_verbosity(verbosity)

        # Define Pygmo archipelago
        archi = pg.archipelago(n=n_isl, algo=algo, prob=opt_prob, pop_size=pop_size, seed=0)
        print(archi)

        # Optimize
        archi.evolve()
        archi.wait()
        self._x = archi.get_champions_x()
        self._f = archi.get_champions_f()
        self._x_at_scale = np.array([self._x[i] * self._tune_bounds for i in range(n_isl)])
        return 

    def pretty(self):
        print(f"\nInitial weights (scaled)  : {self._x0}")
        print(f"Final weights (scaled)    : {self._x}")
        print(f"Initial weights           : {self._x0_at_scale}")
        print(f"Final weights             : {self._x_at_scale}")
        print(f"Final tof                 : {self._f}")
        return

    

    
            

    