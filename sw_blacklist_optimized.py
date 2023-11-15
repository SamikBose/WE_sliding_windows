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


## This function generates all the walker trees and the resampling panel of a run
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


# This returns all the trees that are larger in length than the lag time
# It also returns the sorted node list in terms of cycles. 
def effective_subtrees(all_subtrees, window_length):
    
    eff_subtrees = []
    sorted_node_list = []
    for subtree in all_subtrees:
        sorted_nodes = sorted(subtree.nodes, key=lambda x: x[0])
        if sorted_nodes[-1][0] >= window_length:
            eff_subtrees.append(subtree)
            sorted_node_list.append(sorted_nodes)
    return(eff_subtrees, sorted_node_list)


# This returns a signal suggesting if the root node and its time lag child(ren)
# has conserved weights or not. In case the weights are not conserved we straightaway
# can discard the t0, t1 lagged datapoint pair(s).
def match_weights(eff_subtree, root_node, window_length, run_idx, all_wts):
    SIGNAL = True
    init_cycle = root_node[0]
    init_walker = root_node[1]

    end_cycle  = root_node[0] + window_length
    final_wt = 0.0000000000000000000000000000
    
    init_wt = all_wts[init_walker][init_cycle]

    dict_all = nx.predecessor(eff_subtree, source=root_node, cutoff=window_length)
    tl_node_list = []
    for node in dict_all.keys():
        if node[0] == end_cycle:
            final_wt += all_wts[node[1]][node[0]]
            tl_node_list.append(node)

    if init_wt != final_wt:
        SIGNAL = False
    if init_wt == final_wt:
        SIGNAL = True
    return(SIGNAL, init_wt, final_wt, dict_all, tl_node_list)


# This creates a blacklist which has all the nodes where out of tree 
# merging has taken place.
def primary_blacklist(resamp_pan, sorted_node_list_idx, search_range):
    t1 = time.time()
    blacklist = []
    length = len(sorted_node_list_idx)
    for idx, node in enumerate(sorted_node_list_idx):
        walker, cyc = node[1] , node[0]
        merged_nodes_list = []
        search_range_lower = 0 if idx < search_range else (idx - search_range)
        search_range_upper = length if idx > (length - search_range) else (idx + search_range)

        cyc_resamp_pan = resamp_pan[cyc][0]
        for idx2, item in enumerate(cyc_resamp_pan):
            if walker in item[1] and item[0] == 3:
                merged_nodes_list.append((cyc-1,idx2))

        
        for merged_node in merged_nodes_list:
            if merged_node not in sorted_node_list_idx[search_range_lower:search_range_upper]:
                blacklist.append(node)
                break

        #if idx % 10000 == 0:
        #    print(idx, time.time() - t1)

    t2 = time.time()
    print(f'Time taken to compute the blacklist for this tree: {t2-t1}, Blacklist length {len(blacklist)}')
    return(blacklist)



def check_blacklist_node(blacklist, all_node_dict):
    signal_blacklist = True
    counter = 0
    for keys in all_node_dict.keys():
        if len(all_node_dict[keys]) > 0:
            parent_node = all_node_dict[keys]
            counter += 1
            if parent_node[0] in blacklist:
                signal_blacklist = False
                break
                 
            if counter == len(all_node_dict.keys()):
                if keys in blacklist:
                    print('final nodes are blacklist...')
                    signal_blacklist = False
                    break
    return(signal_blacklist)


# This is the main function being called by the MSM code
# Needs the run_idx, h5_path, lag_time and path to the all_weights file
def get_accepted_sw(h5_path, run_idx, window_length, wt_path):

    t = time.time()

    accepted_sw = []
    all_subtrees, resamp_pan = gen_all_trees(h5_path,run_idx)
    eff_subtrees_list, sorted_node_list = effective_subtrees(all_subtrees, window_length)

    all_wts = pkl.load(open(wt_path,'rb'))
    print(len(eff_subtrees_list))
    
    for index in range(len(eff_subtrees_list)):

        print()
        print(f'Running for tree: {index}, number of nodes: {len(sorted_node_list[index])}')
        t1 = time.time()
        true_count = 0
        false_count_blacklist = 0
        false_count_wt =0

        final_cycle_in_subtree =  sorted_node_list[index][-1][0]
        blacklist = primary_blacklist(resamp_pan, sorted_node_list[index], 200)

        for each_sorted_node in sorted_node_list[index]:
            
            current_cycle =  each_sorted_node[0]
            if current_cycle <= final_cycle_in_subtree - window_length:
                signal, init_wt, final_wt, all_node_dict, tl_nodes_list = match_weights(eff_subtrees_list[index], each_sorted_node, window_length, run_idx, all_wts)
        
                if signal == True:
                    signal_blacklist = check_blacklist_node(blacklist, all_node_dict)
            
                    if signal_blacklist == True:
                        true_count += 1
                
                        for tl_node in tl_nodes_list:
                            accepted_sw.append([(each_sorted_node[1], each_sorted_node[0]), (tl_node[1],tl_node[0])])

                    else:
                        #print(f'Blacklisted node encountered in the subtree... Root node {each_sorted_node}')
                        false_count_blacklist += 1
                else:
                    #print(f'Wts do not match: Node {each_sorted_node}')
                    false_count_wt += 1

        print(time.time() - t1, true_count, false_count_wt, false_count_blacklist)

    #print(f'Total time taken {time.time() - t} seconds')
    return(accepted_sw)




####
####
## Main code ###
t_start = time.time()

run_idx = int(sys.argv[1])
window_length = int(sys.argv[2])

dummy_run_idx = 0
h5_path = f'/dickson/s1/bosesami/time_lagged_data_WE/randomwalk_sims/1D_15states_100000cycles_100walkers_RUN{run_idx}/wepy.results.h5' 
wt_path = f'/dickson/s1/bosesami/time_lagged_data_WE/randomwalk_sims/all_wts_run{run_idx}.pkl'

#all_subtrees, resamp_pan = gen_all_trees(h5_path,dummy_run_idx)
#eff_subtrees_list, sorted_node_list = effective_subtrees(all_subtrees, window_length)
#print(len(eff_subtrees_list))

#for index in range(len(eff_subtrees_list)):
#    blacklist = primary_blacklist(resamp_pan, sorted_node_list[index], eff_subtrees_list[index])


accepted_sliding_windows = get_accepted_sw(h5_path, dummy_run_idx, window_length, wt_path)

#print(f'Run idx: {run_idx}, length of accepted tld:{len(accepted_sliding_windows)}, {accept_count}, {reject_wt_count}, {reject_blist_count}')

pkl.dump(accepted_sliding_windows,  open(f'rw_tld_run{run_idx}_lagtime{window_length}_BL_optimized.pkl','wb'))

t_end = time.time()
print(f'Run idx: {run_idx}, length of accepted tld:{len(accepted_sliding_windows)}, time taken: {t_end - t_start}')
