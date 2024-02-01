import numpy as np
import multiprocessing as mp
import tables as pt
import sys
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Gillespie_backend/')
import Gillespie_backend as gil

def compute_MSD(gillespie, output, step_tot, check_steps):
    """
    Compute the Mean Squared Displacement (MSD) over the simulation.
    """
    msd_time = np.zeros((step_tot // check_steps, check_steps, 2), dtype=float)
    msd_tot = np.zeros((step_tot, 2), dtype=float)
    abs_time = 0.0
    sim_initial_position = gillespie.get_r(periodic=True)
    index_tot = 0  # Changed from float to int for proper indexing
    
    for i in range(step_tot // check_steps):
        initial_positions = gillespie.get_r(periodic=True)
        current_time = 0.0
        
        for step in range(check_steps):
            move, dt = gillespie.evolve()
            current_time += dt[0]
            abs_time += dt[0]
            msd_time[i, step] = [current_time, np.mean(np.linalg.norm(gillespie.get_r(periodic=True) - initial_positions, axis=1)**2)]
            msd_tot[index_tot] = [abs_time, np.mean(np.linalg.norm(gillespie.get_r(periodic=True) - sim_initial_position, axis=1)**2)]
            index_tot += 1
    
    output.put(('create_array', ('/Evolution_of_MSD', 'MSD_' + hex(gillespie.seed), msd_time)))
    output.put(('create_array', ('/MSD_tot', 'MSD_tot' + hex(gillespie.seed), msd_tot)))

def run_simulation(inqueue, output, step_tot, check_steps):
    """
    Run the simulation for each set of parameters fetched from the input queue.
    """
    for args in iter(inqueue.get, None):
        gillespie = initialize_gillespie(*args)
        compute_MSD(gillespie, output, step_tot, check_steps)

def handle_output(output, filename, header):
    """
    Handles writing simulation results to an HDF5 file.
    """
    with pt.open_file(filename, mode='w') as hdf:
        hdf.root._v_attrs.file_header = header
        hdf.create_group('/', 'Evolution_of_MSD', 'MSD at different time')
        hdf.create_group('/', 'MSD_tot', 'MSD total')
        
        while True:
            task = output.get()
            if task is None: break  # Signal to terminate
            
            method, args = task
            getattr(hdf, method)(*args)

def initialize_gillespie(ell_tot, Energy, kdiff, seed, Nlinker, dimension):
    """
    Initialize the Gillespie simulation system with the given parameters.
    """
    # Assuming gil.Gillespie is the correct way to initialize your Gillespie object
    return gil.Gillespie(ell_tot=ell_tot, rho0=0., BindingEnergy=Energy, kdiff=kdiff,
                         seed=seed, sliding=False, Nlinker=Nlinker, old_gillespie=None, dimension=dimension)

def parallel_MSD_evolution(args, step_tot, check_steps, filename):
    """
    Coordinate parallel execution of MSD evolution simulations.
    """
    num_process = mp.cpu_count()
    output = mp.Queue()
    inqueue = mp.Queue()
    
    header = make_header(args, [step_tot, check_steps])
    proc = mp.Process(target=handle_output, args=(output, filename, header))
    proc.start()
    
    jobs = [mp.Process(target=run_simulation, args=(inqueue, output, step_tot, check_steps)) for _ in range(num_process)]
    
    for job in jobs:
        job.daemon = True
        job.start()
    
    for arg in args:
        inqueue.put(arg)
    
    for _ in jobs:
        inqueue.put(None)
    
    for job in jobs:
        job.join()
    
    output.put(None)  # Signal to `handle_output` to terminate
    proc.join()

def make_header(args, sim_arg):
    """
    Create a header string for the HDF5 file.
    """
    header = f"This file computes the mean squared displacement. Parameters of the simulation:\n"
    labels = ['ell_tot', 'Energy', 'kdiff', 'seed', 'Nlinker', 'dimension', 'step_tot', 'check_steps']
    values = args + sim_arg
    header += '\n'.join([f"{label} = {value}" for label, value in zip(labels, values)])
    return header
