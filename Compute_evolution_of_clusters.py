import numpy as np
import multiprocessing as mp
import math
from scipy.spatial import distance_matrix
from multiprocessing import shared_memory
import sys
import queue
import copy
import tables as pt
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Gillespie_backend/')
import Gillespie_backend as gil
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Analysis/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Analysis/')
from ToolBox import *


def cluster_points(points, max_distance):
    # Reshape the points array to shape (-1, 3) if it's (3, N)
    if points.shape[0] == 3:
        points = points.T

    N = points.shape[0]
    clusters = []
    for i in range(N):
        # Initialize a new cluster with the current point
        new_cluster = [points[i]]

        for cluster in clusters:
            for point in cluster:
                # If the current point is within the max_distance of a point in an existing cluster
                if np.linalg.norm(points[i]- point) <= max_distance:
                    # Add the current point to the cluster and break out of the loop
                    cluster.append(points[i])
                    break
            else:
                # Continue the loop if the inner loop wasn't broken
                continue

            # Break the outer loop if the inner loop was broken
            break
        else:
            # If the point wasn't added to any cluster, we add the new_cluster to the list of clusters
            clusters.append(new_cluster)

    return [np.array(cluster) for cluster in clusters]
    
def compute_av_cluster_size(gillespie,output,step_tot,check_steps,max_distance):
    c_size_time = np.zeros((step_tot//check_steps,2),dtype=float)
    current_time = 0.
    prev_c_size = np.mean([c.__len__() for c in cluster_points(gillespie.get_R(),max_distance)])
    for i in range(step_tot//check_steps):
        t_tot = 0.
        av_c_size = 0.
        for t in range(check_steps):
            move,time = gillespie.evolve()
            current_time+=time[0]
            t_tot+=time[0]
            #np.mean([c.__len__() for c in cluster_points(gillespie.get_R(),max_distance)])
            av_c_size +=prev_c_size * time[0]
            prev_c_size = np.mean([c.__len__() for c in cluster_points(gillespie.get_R(),max_distance)])  
        av_c_size /=t_tot
        c_size_time[i] = [current_time,av_c_size]
    output.put(('create_array',('/','av_clust_size_'+hex(gillespie.seed),c_size_time)))

def  Run(inqueue,output,step_tot,check_steps,max_distance):
    # simulation_name is a "f_"+float.hex() 
    """
    Each run process fetch a set of parameters called args, and run the associated simulation until the set of arg is empty.
    The simulation consists of evolving the gillespie, every check_steps it checks if the entropy of the system is close enough
    to a given entropy function. If it is the case it adds the position of the linkers associated to this state + the value of the entropy
    and the time associated to this measurement. the position of the linkers is a (Nlinker,3) array to which we add the value of the
    entropy S, and time t as [S, Nan, Nan], and [t,Nan,nan].
    parameters:
    inqueue (multiprocessing.queue) : each entry of q is  a set of parameters associated with a specific gillespie simulation.
    output (multiprocessing.queue) : it just fetch the data that has to be outputed inside this queue
    step_tot (int) : total number of steps in the simulation
    check_step (int) : number of steps between two checking
    epsilon (float): minimum distances (in entropy unit) for the picture to be taken
    X,Y : the average entropy curve of reference.
    """
    for args in iter(inqueue.get,None):
        # create the associated gillespie system
        Nlinker = args[4] 
        ell_tot = args[0]
        kdiff = args[2]
        Energy = args[1]
        seed = args[3]
        dimension = args[5]
        # create the system
        gillespie = gil.Gillespie(ell_tot=ell_tot, rho0=0., BindingEnergy=Energy, kdiff=kdiff,
                            seed=seed, sliding=False, Nlinker=Nlinker, old_gillespie=None, dimension=dimension)
        # pass it as an argument, R returns an array of size (step_tot//check_steps,Nlinker+2,3)
        #output.put(('create_group',('/','bin_hist_'+hex(seed))))
        #Compute_Pair_Correlation_Function(gillespie,output,'bin_hist_'+hex(seed),step_tot,check_steps,num_bins,max_distance,linked)
        compute_av_cluster_size(gillespie,output,step_tot,check_steps,max_distance)
        #output.put(('create_array',('/',"R_"+hex(seed),R)))

def handle_output(output,filename,header):
    """
    This function handles the output queue from the Simulation function.
    It uses the PyTables (tables) library to create and write to an HDF5 file.

    Parameters:
    output (multiprocessing.Queue): The queue from which to fetch output data.

    The function retrieves tuples from the output queue, each of which 
    specifies a method to call on the HDF5 file (either 'createGroup' 
    or 'createArray') and the arguments for that method. 

    The function continues to retrieve and process data from the output 
    queue until it encounters a None value, signaling that all simulations 
    are complete. At this point, the function closes the HDF5 file and terminates.
    """
    hdf = pt.open_file(filename, mode='w') # open a hdf5 file
    while True: # run until we get a False
        args = output.get() # access the last element (if there is no element, it keeps waiting for one)
        if args: # if it has an element access it
            if args.__len__() == 3:
                method, args,time = args # the elements should be tuple, the first element is a method second is the argument.
                array = getattr(hdf, method)(*args) # execute the method of hdf with the given args
                array.attrs['time'] = time
            else :
                method, args = args # the elements should be tuple, the first element is a method second is the argument.
                array = getattr(hdf, method)(*args) # execute the method of hdf with the given args
        else: # once it receive a None
            break # it break and close
    hdf.close()
def make_header(args,sim_arg):
    header ='is close enough to the average entropy curve (that has been computed by averaging 50 to 100 systems) '
    header += 'the file is composed of arrays, each array name can be written : h_X...X where X...X represent an hexadecimal '
    header+= 'name for an integer that corresponds to the seed of the simulation. Each array is made of the position of N '
    header+= 'linkers. Additionnally, the two first entry of the array are [S,NaN,Nan] and [t,NaN,NaN] that are respectively  '
    header+= 'the value of the entropy and time of the given picture.\n'
    header += 'Parameters of the simulation : '
    header +='Nlinker = '+str(args[4])+'\n'
    header +='ell_tot = '+str(args[0])+'\n'
    header += 'kdiff = '+str(args[2])+'\n'
    header += 'Energy =  '+str(args[1])+'\n'
    header += 'seed = '+str(args[3])+'\n'
    header += 'dimension = '+str(args[5])+'\n'
    header+='step_tot = '+str(sim_arg[0])+'\n'
    header+='check_steps = '+str(sim_arg[1])+'\n'

def parallel_cluster_size_evolution(args,step_tot,check_steps,filename,max_distance):
    num_process = mp.cpu_count()
    output = mp.Queue() # shared queue between process for the output
    inqueue = mp.Queue() # shared queue between process for the inputs used to have less process than simulations
    jobs = [] # list of the jobs for  the simulation
    header = make_header(args,[step_tot,check_steps])
    proc = mp.Process(target=handle_output, args=(output,filename,header)) # start the process handle_output, that will only end at the very end
    proc.start() # start it
    for i in range(num_process):
        p = mp.Process(target=Run, args=(inqueue, output,step_tot,check_steps,max_distance)) # start all the 12 processes that do nothing until we add somthing to the queue
        jobs.append(p)
        p.daemon = True
        p.start()
    for arg in args:
        inqueue.put(arg)  # put all the list of tuple argument inside the input queue.
    for i in range(num_process): # add a false at the very end of the queue of argument
        inqueue.put(None) # we add one false per process we started... We need to terminate each of them
    for p in jobs: # wait for the end of all processes
        p.join()
    output.put(False) # now send the signal for ending the output.
    proc.join() # wait for the end of the last process