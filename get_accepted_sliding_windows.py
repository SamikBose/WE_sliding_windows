# This code provides the functions to extract the time-lagged dataset
# from WEPY simulation data, where merging and cloning has taken
# place as a part of the enhanced sampling strategy.
#
# This is essential for running TICA or building the transition counts matrices.
#
# A time-lagged data has a starting point (t0) and an end-point (t1), which are
# related by the formula t1 = t0 _ \del_T where \del_T is the lag-time.
# 
# As per weighted ensemble strategy, a trajectory can clone into multiple trahectories
# with any given time-step of the simulation. Hence, there is a probability that the trajectory at t0 will split
# into multiple trajectories within a given lag-time. Hence, for a given conformation (C0) at t0, 
# instead of a single conformation (C1) at t1, we may end up with multiple conformations at t1 (C1, C1', C1" etc...).
# So, the concept of straight-forward time-lagged dataset does not work with the WE based strategy.
#
# The solution:
# Firstly, the time-lagged datapoints (C0 at t0 and C1, C1', C1"... at t1) can not have any weight imbalance, i.e.,
# the weight of the root node (at t0) and the weight of it's time-lagged children (at t1) should be exactly same. 
# Secondly, even if there is no weight-imbalance, still there may arise a particular scenario where we need to discard the
# time-lagged datapoint from the dataset. Consider a case where a walker (weight=w) from a separate tree (no historical connection), 
# merges into a node which connects the root node (at t0) with its time-lagged children (at t1). 
# In general this would add-up weights by 'w' amount to the subtree of the root node under observation and hence such time-lagged
# parent-children pair will be discarded straightaway. However, if we have a squashing event within the (t0 to t1) subtree as well
# where the sqaushed walker had the exactly same weight ('w') before being squashed, then mathematically we don't observe the
# difference in the weights of root node (t0) and its children (t1). Still such time-lagged parent-children pair are biased by out-of-tree
# merging and must be discarded as well. 

# These solutions are undertaken in the following functions and we get corrected time-lagged datapoints which 
# are used to build the MSMs in our work.
#

import numpy as np
from wepy.hdf5 import WepyHDF5
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.analysis.parents import resampling_panel, parent_panel, net_parent_table, sliding_window, ancestors, ParentForest
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from itertools import groupby
import pickle as pkl
import os.path as osp
import os
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
import mdtraj as mdj
import numpy as np
import time
import sys
import networkx as nx


def gen_all_trees(h5_path,run_idx):
    wepy_h5 =  WepyHDF5(h5_path, mode='r')

    with wepy_h5:
        n_walkers = wepy_h5.num_init_walkers(run_idx)
        n_cycles = wepy_h5.num_run_cycles(run_idx)

        # Make Parent Table
        resampling_rec = wepy_h5.resampling_records([run_idx])
        resamp_panel = resampling_panel(resampling_rec)
        par_panel = parent_panel(MultiCloneMergeDecision, resamp_panel)
        net_par_table = net_parent_table(par_panel)
        parent_forest = ParentForest(parent_table = net_par_table)
    all_subtrees = parent_forest.trees
    return(all_subtrees, resamp_panel)


#
def effective_subtrees(all_subtrees, window_length):
    
    eff_subtrees = []
    sorted_node_list = []
    for subtree in all_subtrees:
        sorted_nodes = sorted(subtree.nodes, key=lambda x: x[0])
        if sorted_nodes[-1][0] >= window_length:
            eff_subtrees.append(subtree)
            sorted_node_list.append(sorted_nodes)
    return(eff_subtrees, sorted_node_list)


#
def match_weights(eff_subtree, root_node, window_length, run_idx, h5_path):
    SIGNAL = True
    init_cycle = root_node[0]
    init_walker = root_node[1]

    end_cycle  = root_node[0] + window_length
    final_wt = 0.0000000000000000000000000000

    wepy_h5 = WepyHDF5(h5_path, mode='r')
    wepy_h5.open()
    init_wt = wepy_h5.h5[f'runs/{run_idx}/trajectories/{init_walker}/weights/'][init_cycle][0]

    dict_all = nx.predecessor(eff_subtree, source=root_node, cutoff=window_length)
    tl_node_list = []
    for node in dict_all.keys():
        if node[0] == end_cycle:
            final_wt += wepy_h5.h5[f'runs/{run_idx}/trajectories/{node[1]}/weights/'][node[0]][0]
            tl_node_list.append(node)
    wepy_h5.close()

    if init_wt != final_wt:
        SIGNAL = False
    if init_wt == final_wt:
        SIGNAL = True
    return(SIGNAL, init_wt, final_wt, dict_all, tl_node_list)


# This is the second layer: Check if there is any out of tree merging inside the weight-matched
# list of lagged datapoints.
def check_out_of_tree_merge(all_node_dict, resamp_pan, sorted_node_list):

    signal_merging = True
    for keys in  all_node_dict.keys():
        if len(all_node_dict[keys]) > 0:
            merged_nodes_list = []
            parent_cyc = all_node_dict[keys][0][0]
            parent_walker = all_node_dict[keys][0][1]
            resamp_pan_cyc = resamp_pan[parent_cyc][0]
            for idx, item in enumerate(resamp_pan_cyc):
                if parent_walker in item[1] and item[0] == 3:
                    signal_merging = False
                    merged_nodes_list.append((parent_cyc,idx))
                    print(f'Nodes merged into {(parent_cyc,parent_walker)} are {merged_nodes_list}')
                    #print(keys, merged_nodes_list)

            for merged_node in merged_nodes_list:
                if merged_node in sorted_node_list:
                    signal_merging = True
                    print(f'Inside tree merging: Node in Lag-tree {(parent_cyc,parent_walker)} , Parent node {merged_node}...')

                else:
                    print(f'Out of tree merging: Node in Lag-tree {(parent_cyc,parent_walker)} , Parent node {merged_node} is out of tree...')
                    signal_merging = False
                    break

        if signal_merging == False:
            break

    return(signal_merging)


# This is the main function being called by the MSM code
# Needs the run_idx, h5_path and lag_time
def get_accepted_sw(h5_path, run_idx, window_length):

    accepted_sw = []

    all_subtrees, resamp_pan = gen_all_trees(h5_path,run_idx)
    eff_subtrees_list, sorted_node_list = effective_subtrees(all_subtrees, window_length)

    #for index in range(3,4):
    for index in range(len(eff_subtrees_list)):

        print(f'Running for effective subtree {index}: length {len(eff_subtrees_list[index])}')
        t1 = time.time()
        true_count = 0
        false_count = 0
        false_count_merging = 0

        #storing the accepted subtree, it's sorted nodes and the final cycle index...
        eff_subtree = eff_subtrees_list[index]
        sorted_nodes = sorted_node_list[index]
        final_cycle_in_subtree =  sorted_nodes[-1][0]

        for each_sorted_node in sorted_nodes:
            current_cycle =  each_sorted_node[0]

            if current_cycle <= final_cycle_in_subtree - window_length:
                signal, init_wt, final_wt, all_node_dict, tl_nodes_list = match_weights(eff_subtree, each_sorted_node, window_length, run_idx, h5_path)


                if signal == True:
                    signal_merging = check_out_of_tree_merge(all_node_dict, resamp_pan, sorted_nodes)

                    if signal_merging == True:

                        ## testing here
                        true_count += 1

                        for tl_node in tl_nodes_list:
                            accepted_sw.append([(each_sorted_node[1], each_sorted_node[0]), (tl_node[1],tl_node[0])])

                    else:
                        false_count_merging += 1
                else:
                    false_count += 1


        t2 = time.time()
        print(index, len(sorted_nodes), true_count, false_count, false_count_merging, t2 - t1)
    return(accepted_sw)


for i in range(10):
    t_start = time.time()
    h5_path = '/dickson/s1/bosesami/KSL_unbinding/KSL_19/simulation_folder/openmm/clone.h5' 
    tld_path = '/dickson/s1/bosesami/time_lagged_data_WE/tl_datasets'
    run_idx = i
    window_length = int(sys.argv[1])
    t_end = time.time()

    accepted_sliding_windows = get_accepted_sw(h5_path, run_idx, window_length)

    pkl.dump(accepted_sliding_windows,  open(f'{tld_path}/tld_run{run_idx}_lagtime{window_length}.pkl','wb'))
