#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:13:04 2022

@author: tizianolatino
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from scipy.stats import randint


def data_load(data_type):
    
    if data_type == 'neg_bin_not_reduced':
        data = pd.read_csv('data/raw_GM12878_1Mb_prepro.csv')
        metadata = pd.read_csv('data/metadata_prepro.csv')
        metadata_bin = pd.read_csv('data/metadata_neg_bin.csv')
        data_toy = pd.read_csv('data/data_toy_neg_bin.csv')
        print(' negative binomial model with data original data dimension:',data.shape)
    
    if data_type == 'bin_not_reduced':
        data = pd.read_csv('data/raw_GM12878_1Mb_prepro.csv')
        metadata = pd.read_csv('data/metadata_prepro.csv')
        metadata_bin = pd.read_csv('data/metadata_bin.csv')
        #str chr adjusting
        for i in range(0,len(metadata)):
            metadata['chr'][i] = metadata['chr'][i].replace("'", '')
        #str chr adjusting
        for i in range(0,len(metadata_bin)):
            metadata_bin['chr'][i] = metadata_bin['chr'][i].replace("'", '')
        data_toy = pd.read_csv('data/data_toy.csv')
        print(' binomial model with data original data dimension:',data.shape)
    
    if data_type == 'neg_bin_reduced':
        data = pd.read_csv('data/raw_GM12878_1Mb_prepro_reduced.csv')
        metadata = pd.read_csv('data/metadata_prepro_reduced.csv')
        metadata_bin = pd.read_csv('data/metadata_neg_bin_reduced.csv')
        #str chr adjusting
        for i in range(0,len(metadata)):
            metadata['chr'][i] = metadata['chr'][i].replace("'", '')
        #str chr adjusting
        for i in range(0,len(metadata_bin)):
            metadata_bin['chr'][i] = metadata_bin['chr'][i].replace("'", '')
        data_toy = pd.read_csv('data/data_toy_neg_reduced.csv')
        print('negative binomial model with data dimension reduced:',data.shape)
        
    if data_type == 'bin_reduced':
        data = pd.read_csv('data/raw_GM12878_1Mb_prepro_reduced.csv')
        metadata = pd.read_csv('data/metadata_prepro_reduced.csv')
        metadata_bin = pd.read_csv('data/metadata_bin_reduced.csv')
        #str chr adjusting
        for i in range(0,len(metadata)):
            metadata['chr'][i] = metadata['chr'][i].replace("'", '')
        #str chr adjusting
        for i in range(0,len(metadata_bin)):
            metadata_bin['chr'][i] = metadata_bin['chr'][i].replace("'", '')
        data_toy = pd.read_csv('data/data_toy_reduced.csv')
        print(' binomial model with data dimension reduced:',data.shape)
    
    return data, metadata, metadata_bin, data_toy 

#data_type = 'neg_bin_reduced'
#data, metadata, metadata_bin, data_toy = data_load(data_type)


#####################################################################################################
#------------------------------------TOY DATASET-------------------------------------------------------#


def toy_df(data, metadata, metadata_bin, save_name ):
    
    data_toy = data.copy(deep=True)
    
    #intrachromosomal
    for i in range(0,len(metadata)):
        #block chr
        chri = metadata['chr'][i]
        df = data.iloc[metadata['start'][i]:metadata['end'][i]+1,
                           metadata['start'][i]:metadata['end'][i]+1]

        #diagonal binomial
        for dist in range(1,len(df)):
            diag = np.diag(df, k=dist)
            df_dis = metadata_bin[(metadata_bin == dist).any(axis=1)]
            row = df_dis[(df_dis == str(chri)).any(axis=1)]

            '''
            #binomial
            n = row['n'].values[0]
            p = row['p'].values[0]
            diag_ = np.random.binomial(n, p, len(diag))
            for d in range(0,len(diag_)): 
                df.iloc[d][d+dist] = diag_[d]
                df.iloc[d+dist][d] = diag_[d]
            '''
            #negative binomial
            r = row['r'].values[0]
            p = row['p'].values[0]
            diag_ = np.random.negative_binomial(r, p, len(diag))
            for d in range(0,len(diag_)): 
                df.iloc[d][d+dist] = diag_[d]
                df.iloc[d+dist][d] = diag_[d]
        
        #fill original dataframe      
        data_toy.iloc[metadata['start'][i]:metadata['end'][i]+1,
                           metadata['start'][i]:metadata['end'][i]+1] = df
    #interchromosomal
    for i in range(1,len(metadata)):
        df = data.iloc[metadata['start'][i]:metadata['end'][i]+1,
                           0:metadata['start'][i]]
        '''
        #uniform binomial distr
        n = metadata_bin['n'][len(metadata_bin)-1]
        p = metadata_bin['p'][len(metadata_bin)-1]
        df = np.random.binomial(n, p, df.shape)
        '''
        #uniform negative binomial distr
        r = metadata_bin['r'][len(metadata_bin)-1]
        p = metadata_bin['p'][len(metadata_bin)-1]
        df = np.random.negative_binomial(r, p, df.shape)
        data_toy.iloc[metadata['start'][i]:metadata['end'][i]+1,
                           0:metadata['start'][i]] = df
        
    #symmetrize datframe interchr
    def symmetrize(df):
        i_upper = np.triu_indices(len(df), 0)
        matrix = df.to_numpy()
        matrix[i_upper] = matrix.T[i_upper] 
        df = pd.DataFrame(matrix)
        
    
    symmetrize(data_toy)
    
    #check if is symmetric
    arr = data_toy.to_numpy()
    (arr.transpose() == arr).all()
    
    if save_name != False:
        data_toy.to_csv('data/'+save_name+'.csv', index=False)
    
    return data_toy
    

#plt.hist(df,bins=int(np.sqrt(df.shape[0]*df.shape[1])))

######################################################################################################
#------------------------------------ADJACENCY MATRIX------------------------------------------------#
    

import matplotlib.patches as patches
import matplotlib as mpl

def adj_vis(A,metadata ,box_vis): 
    fig, ax = plt.subplots()
    
    
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (0,0,0,0)
    for i in range(1,50):
        cmaplist[i] = (.5, .5, .5, 1.0)
    
    
    
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    bounds0 = np.linspace(0, 1, 1).tolist()
    bounds1 = np.linspace(1, 25, 2).tolist()
    bounds2 = np.linspace(50, 100, 2).tolist()
    bounds3 = np.linspace(500, 1000, 2).tolist()
    bounds4 = np.linspace(2500, 5000, 2).tolist()
    bounds5 = np.linspace(10000, 20000, 3).tolist()
    bounds = bounds0 + bounds1 + bounds2 + bounds3 + bounds4 + bounds5
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    
    im = ax.imshow(A, cmap=cmap, norm=norm)
    
    yticks = []
    chrs_len_tot = 0
    for i in range(0,len(metadata)):
        chri_len = metadata['end'][i]-metadata['start'][i]
        rect = patches.Rectangle((chrs_len_tot-0.5, chrs_len_tot -0.5), 
                                 chri_len+1, chri_len+1, 
                                   linewidth=2, edgecolor='k', facecolor='none', linestyle='--')
        if box_vis == True:
            ax.add_patch(rect)
        middle_point = int(chri_len*0.5)+chrs_len_tot
        yticks.append(middle_point)
        chrs_len_tot += chri_len+1
        
    plt.yticks(yticks, labels=metadata['chr'], fontsize=10)
 
        
    plt.xlabel('Mbp', fontsize=13)
   
    fig.colorbar(im,location='right', anchor=(0, 0.3), shrink=0.7, ticks=bounds)


#############################################################################################
#------------------------------------NETWORK------------------------------------------------#


#zeroes counting

#sum(data_toy[data_toy==0].count())

#create weighted undirected grapg from dataframe
#G = nx.from_numpy_matrix(data_toy.values)

#len(G.edges)

                
'''            
#plotting weight
weights = []
distances = []

for edge in G.edges(0):
    w = G[edge[0]][edge[1]]["weight"]
    weights.append(w)
    distances.append(edge[1])
    
plt.plot(distances,weights)
#plt.loglog(distances,weights, 'bx-')
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.ylabel('Contact Probability',fontsize='xx-large')
plt.show()    
'''

'''
#graph of chr1
G_chr1 = nx.Graph()

chr_1= []

for i in range(0,int(chr2_max+1)):
    chr_1.append(i)

clust = chr_1

for sub_node in clust:
    edges_sub_node = list(G.edges(sub_node))
    for sub_edge in edges_sub_node:
        G_chr1.add_edge(float(sub_edge[0]),float(sub_edge[1]), weight=G[sub_edge[0]][sub_edge[1]]["weight"])        

'''  
##################################################################################################
#------------------------------------TRANSLOCATED DATAFRAME------------------------------------------------# 


from itertools import combinations_with_replacement, permutations, combinations

def df_translocation(data_toy_, chrs_name_,translocation_,metadata_,metadata_bin_,t_type_):
    
    data_toy_trans_ = data_toy_.copy(deep=True)
    
    if t_type_ == 'rbt':
        #reciprocal balanced translocation
        
        for i in range(0,len(chrs_name_)):
                
                #chr line construction
                t_i = translocation_[i]
                print('rbt-',i,t_i)
                t_chr_1 = list(np.arange(t_i[0][0],t_i[0][1]+1))
                t_chr_2 = list(np.arange(t_i[1][0],t_i[1][1]+1))
                chr_1_row = metadata_.loc[metadata_['chr'].replace("'", '') ==  chrs_name_[i][0]]
                chr_1 = list(np.arange(chr_1_row['start'].iloc[0],chr_1_row['end'].iloc[0]+1))
                chr_2_row = metadata_.loc[metadata_['chr'].replace("'", '') ==  chrs_name_[i][1]]
                chr_2 = list(np.arange(chr_2_row['start'].iloc[0],chr_2_row['end'].iloc[0]+1))
                chr_brpt_1 = chr_1.index(t_chr_1[0])
                chr_brpt_2 = chr_2.index(t_chr_2[0])
                chr_1 = chr_1[:chr_brpt_1]+t_chr_2+chr_1[chr_brpt_1:]
                chr_2 = chr_2[:chr_brpt_2]+t_chr_1+chr_2[chr_brpt_2:]
                chr_1_t = [x for x in chr_1 if x not in t_chr_1]
                chr_2_t = [x for x in chr_2 if x not in t_chr_2]
                     
                #dataframe chr_1 modification 
                chr_1_t_ij = list(combinations(chr_1_t,2))
                for j in chr_1_t_ij:
                    dist = np.abs(chr_1_t.index(j[0])-chr_1_t.index(j[1]))
                    chr_name = chrs_name_[i][0]
                    df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                    row = df_dis[(df_dis == chr_name).any(axis=1)]
                    r = row['r'].values[0]
                    p = row['p'].values[0]
                    contacts_ij = np.random.negative_binomial(r, p)
                    data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                    data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
                chr_1_ = list(np.arange(chr_1_row['start'].iloc[0],chr_1_row['end'].iloc[0]+1))
                chr_1_ = [x for x in chr_1_ if x not in t_chr_1]
                for j in t_chr_1:
                    for jj in chr_1_:
                        r = metadata_bin_['r'][len(metadata_bin_)-1]
                        p = metadata_bin_['p'][len(metadata_bin_)-1]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j][jj] = contacts_ij
                        data_toy_trans_.loc[jj][j] = contacts_ij
                        
                #dataframe chr_2 modification 
                chr_2_t_ij = list(combinations(chr_2_t,2))
                for j in chr_2_t_ij:
                    dist = np.abs(chr_2_t.index(j[0])-chr_2_t.index(j[1]))
                    chr_name = chrs_name_[i][1]
                    df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                    row = df_dis[(df_dis == chr_name).any(axis=1)]
                    r = row['r'].values[0]
                    p = row['p'].values[0]
                    contacts_ij = np.random.negative_binomial(r, p)
                    data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                    data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
                chr_2_ = list(np.arange(chr_2_row['start'].iloc[0],chr_2_row['end'].iloc[0]+1))
                chr_2_ = [x for x in chr_2_ if x not in t_chr_2]
                for j in t_chr_2:
                    for jj in chr_2_:
                        r = metadata_bin_['r'][len(metadata_bin_)-1]
                        p = metadata_bin_['p'][len(metadata_bin_)-1]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j][jj] = contacts_ij
                        data_toy_trans_.loc[jj][j] = contacts_ij

    if t_type_ == 'nrbt':
        #non reciprocal balanced translocation
        
        for i in range(0,len(chrs_name_)):
                #chr line construction
                t_i = translocation_[i]
                print('nrbt-',i,t_i)
                t_chr_1 = list(np.arange(t_i[0][0],t_i[0][1]+1))         
                chr_1_row = metadata_.loc[metadata_['chr'].replace("'", '') ==  chrs_name_[i][0]]
                chr_1 = list(np.arange(chr_1_row['start'].iloc[0],chr_1_row['end'].iloc[0]+1))
                chr_2_row = metadata_.loc[metadata_['chr'].replace("'", '') ==  chrs_name_[i][1]]
                chr_2 = list(np.arange(chr_2_row['start'].iloc[0],chr_2_row['end'].iloc[0]+1))
                chr_brpt_1 = chr_1.index(t_chr_1[1])
                chr_brpt_2 = chr_2.index(t_i[1][0])
                chr_2_t = chr_2[:chr_brpt_2]+t_chr_1+chr_2[chr_brpt_2:]
                chr_1_t = [x for x in chr_1 if x not in t_chr_1]
                
                
                #dataframe chr_1 modification 
                chr_1_t_ij = list(combinations(chr_1_t,2))
                for j in chr_1_t_ij:              
                    dist = np.abs(chr_1_t.index(j[0])-chr_1_t.index(j[1]))
                    chr_name = chrs_name_[i][0]
                    df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                    row = df_dis[(df_dis == chr_name).any(axis=1)]
                    r = row['r'].values[0]
                    p = row['p'].values[0]
                    contacts_ij = np.random.negative_binomial(r, p)
                    data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                    data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
                chr_1_ = list(np.arange(chr_1_row['start'].iloc[0],chr_1_row['end'].iloc[0]+1))
                chr_1_ = [x for x in chr_1_ if x not in t_chr_1]
                for j in t_chr_1:
                    for jj in chr_1_:
                        r = metadata_bin_['r'][len(metadata_bin_)-1]
                        p = metadata_bin_['p'][len(metadata_bin_)-1]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j][jj] = contacts_ij
                        data_toy_trans_.loc[jj][j] = contacts_ij
                        
                #dataframe chr_2 modification 
                chr_2_t_ij = list(combinations(chr_2_t,2))
                for j in chr_2_t_ij:
                    dist = np.abs(chr_2_t.index(j[0])-chr_2_t.index(j[1]))
                    chr_name = chrs_name_[i][1]
                    df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                    row = df_dis[(df_dis == chr_name).any(axis=1)]
                    if len(row) == 0:
                        dist = len(metadata_bin_[(metadata_bin_ == chr_name).any(axis=1)])-2
                        df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                        row = df_dis[(df_dis == chr_name).any(axis=1)]
                        r = row['r'].values[0]
                        p = row['p'].values[0]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                        data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
                    else:  
                        r = row['r'].values[0]
                        p = row['p'].values[0]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                        data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
              
    if t_type_ == 'ut':
        #non reciprocal balanced translocation
        
        for i in range(0,len(chrs_name_)):
                #chr line construction
                t_i = translocation_[i]
                print('ut-',i,t_i)
                t_chr_1 = list(np.arange(t_i[0][0],t_i[0][1]+1))         
                t_chr_2 = list(np.arange(t_i[1][0],t_i[1][1]+1))  
                chr_1_row = metadata_.loc[metadata_['chr'].replace("'", '') ==  chrs_name_[i][0]]
                chr_1 = list(np.arange(chr_1_row['start'].iloc[0],chr_1_row['end'].iloc[0]+1))
                chr_2_row = metadata_.loc[metadata_['chr'].replace("'", '') ==  chrs_name_[i][1]]
                chr_2 = list(np.arange(chr_2_row['start'].iloc[0],chr_2_row['end'].iloc[0]+1))
                chr_brpt_1 = chr_1.index(t_chr_1[1])
                chr_brpt_2 = chr_2.index(t_i[1][0])
                chr_2 = chr_2[:chr_brpt_2]+t_chr_1+chr_2[chr_brpt_2:]
                chr_1_t = [x for x in chr_1 if x not in t_chr_1]
                chr_2_t = [x for x in chr_2 if x not in t_chr_2]
                
                #dataframe chr_1 modification 
                chr_1_t_ij = list(combinations(chr_1_t,2))
                for j in chr_1_t_ij:              
                    dist = np.abs(chr_1_t.index(j[0])-chr_1_t.index(j[1]))
                    chr_name = chrs_name_[i][0]
                    df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                    row = df_dis[(df_dis == chr_name).any(axis=1)]
                    r = row['r'].values[0]
                    p = row['p'].values[0]
                    contacts_ij = np.random.negative_binomial(r, p)
                    data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                    data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
                chr_1_ = list(np.arange(chr_1_row['start'].iloc[0],chr_1_row['end'].iloc[0]+1))
                chr_1_ = [x for x in chr_1_ if x not in t_chr_1]
                for j in t_chr_1:
                    for jj in chr_1_:
                        r = metadata_bin_['r'][len(metadata_bin_)-1]
                        p = metadata_bin_['p'][len(metadata_bin_)-1]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j][jj] = contacts_ij
                        data_toy_trans_.loc[jj][j] = contacts_ij
                        
                #dataframe chr_2 modification 
                chr_2_t_ij = list(combinations(chr_2_t,2))
                for j in chr_2_t_ij:
                    dist = np.abs(chr_2_t.index(j[0])-chr_2_t.index(j[1]))
                    chr_name = chrs_name_[i][1]
                    df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                    row = df_dis[(df_dis == chr_name).any(axis=1)]
                    if len(row) == 0:
                        dist = len(metadata_bin[(metadata_bin_ == chr_name).any(axis=1)])-1
                        df_dis = metadata_bin_[(metadata_bin_ == dist).any(axis=1)]
                        row = df_dis[(df_dis == chr_name).any(axis=1)]
                        r = row['r'].values[0]
                        p = row['p'].values[0]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                        data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
                    else:  
                        r = row['r'].values[0]
                        p = row['p'].values[0]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j[0]][j[1]] = contacts_ij
                        data_toy_trans_.loc[j[1]][j[0]] = contacts_ij
                all_chr = list(np.arange(0,len(data_toy_trans_)))  
                for j in t_chr_2:
                    for jj in all_chr:
                        r = metadata_bin_['r'][len(metadata_bin_)-1]
                        p = metadata_bin_['p'][len(metadata_bin_)-1]
                        contacts_ij = np.random.negative_binomial(r, p)
                        data_toy_trans_.loc[j][jj] = contacts_ij
                        data_toy_trans_.loc[jj][j] = contacts_ij
                     
    return data_toy_trans_


#########################################################################################################
#---------------------------------------------SUB NETWORK-----------------------------------------------#


def unique_combination(list_1,list_2):
    unique_combinations = []
    for ii in range(len(list_1)):
        for jj in range(len(list_2)):
            unique_combinations.append((list_1[ii], list_2[jj]))

    return unique_combinations

#scegliere sempre come primo chr: chr1-------da correggere
def sub_network(chrs_name_,data_toy_,metadata):
    chrs_name_ = [item for sublist in chrs_name_ for item in sublist]
    
    #metadata_sub creation
    metadata_sub_col = {'chr': [], 'start': [], 'end': []}
    metadata_sub_ = pd.DataFrame(metadata_sub_col)
    for chrs_name_i in chrs_name_:
        chr_i_row = metadata.loc[metadata['chr'] ==  chrs_name_i]
        row_i = [chr_i_row['chr'].iloc[0], chr_i_row['start'].iloc[0], chr_i_row['end'].iloc[0]]
        metadata_sub_.loc[len(metadata_sub_)] = row_i
    metadata_sub_ = metadata_sub_.sort_values('start')
    metadata_sub_ = metadata_sub_.reset_index(drop=True)
    
    #data_toy_sub intrachr creation
    data_dim = 0
    for i in range(0,len(metadata_sub_)):     
        start = metadata_sub_['start'][i]
        end = metadata_sub_['end'][i]
        dim = end-start+1
        data_dim += dim
    data_array = np.zeros((data_dim, data_dim))
    ij = 0
    for i in range(0,len(metadata_sub_)):
        start = metadata_sub_['start'][i]
        end = metadata_sub_['end'][i]
        dist = end-start+1
        df = data_toy_.iloc[start:end+1,start:end+1]
        data_array[ij:ij+dist,ij:ij+dist] = df
        ij+=dist
    
    #data_toy_sub interchr creation
    dist_row = 0
    for i in range(0,len(metadata_sub_)):
        start_row = metadata_sub_['start'][i]
        end_row = metadata_sub_['end'][i]
        dist_col=0
        dist_i = end_row-start_row+1
        for j in range(0,len(metadata_sub_)):
            start_col = metadata_sub_['start'][j]
            end_col = metadata_sub_['end'][j]
            dist_j = end_col-start_col+1
            if (start_row,end_row) == (start_col,end_col):  
                dist_col += dist_j
                continue
            else:
                df = data_toy_.iloc[start_row:end_row+1,start_col:end_col+1]
                data_array[dist_row:dist_row+dist_i,dist_col:dist_col+dist_j] = df
            dist_col += dist_j
        dist_row += dist_i
               
    data_toy_sub = pd.DataFrame(data_array) 
    
    #sub network creation   
    G_sub = nx.Graph()          
    dist_chr = []
    dist = 0
    for i in range(0,len(metadata_sub_)):
        start_row = metadata_sub_['start'][i]
        end_row = metadata_sub_['end'][i]
        dist += end_row-start_row+1
        dist_chr.append(dist-1)
    
    for i in range(0,len(metadata_sub_)):
        start_row = metadata_sub_['start'][i]
        end_row = metadata_sub_['end'][i]
        dist = end_row-start_row
        nodes_graph_chr_i = list(np.arange(start_row,end_row+1))
        nodes_df_chr_i = list(np.arange(dist_chr[i]-dist,dist_chr[i]+1))  
        for j in range(i,len(metadata_sub_)):
            start_row = metadata_sub_['start'][j]
            end_row = metadata_sub_['end'][j]
            dist = end_row-start_row
            nodes_graph_chr_j = list(np.arange(start_row,end_row+1))
            nodes_df_chr_j = list(np.arange(dist_chr[j]-dist,dist_chr[j]+1))
            comb_df = unique_combination(nodes_df_chr_i,nodes_df_chr_j)
            comb_graph = unique_combination(nodes_graph_chr_i,nodes_graph_chr_j)
            for ij in range(0,len(comb_df)):
                w = data_toy_sub[comb_df[ij][0]][comb_df[ij][1]]
                G_sub.add_edge(comb_graph[ij][0],comb_graph[ij][1], weight=w) 
       
    return metadata_sub_, G_sub, data_toy_sub


######################################################################################################
#---------------------------------------------LABELS-------------------------------------------------#

def labels_creation(chrs_name_,translocation,metadata,plot_type,G):
    chrs_name__ = [item for sublist in chrs_name_ for item in sublist]
    count = 0
    df_labels_col = {'chr': [], 'start': [], 'end': [], 'labels': []}
    df_labels = pd.DataFrame(df_labels_col)
    flat_list = [item for sublist in translocation for item in sublist]
    translocation = [item for sublist in flat_list for item in sublist]
    
    for i in range(0,len(metadata)):
        
        if plot_type == 'all_translocation':
            start = metadata['start'][i]
            end = metadata['end'][i]       
            for t in range(1, len(translocation), 2): 
                t_end = translocation[t]
                t_start = translocation[t-1]            
                if t_start >= start and t_end <= end:
                    row_i = ['nt'+metadata['chr'][i],metadata['start'][i],t_start-1,len(df_labels)-count]
                    row_t = ['t'+metadata['chr'][i],t_start,t_end,len(df_labels)+1-count]
                    row_f = ['nt'+metadata['chr'][i],t_end+1,metadata['end'][i],len(df_labels)-count]
                    df_labels.loc[len(df_labels)] = row_i
                    df_labels.loc[len(df_labels)] = row_t
                    df_labels.loc[len(df_labels)] = row_f
                    count = count+1
                    break          
                elif t_end == translocation[len(translocation)-1]:
                    row = ['nt'+metadata['chr'][i],metadata['start'][i],metadata['end'][i],len(df_labels)-count]
                    df_labels.loc[len(df_labels)] = row
                    break
                
        if plot_type == 'all_no_translocation':
            row = [metadata['chr'][i],metadata['start'][i],metadata['end'][i], i]
            df_labels.loc[len(df_labels)] = row
            
        if plot_type == 'chrs_no_translocation':
            for j in range(0,len(chrs_name__)):
                if metadata['chr'][i].replace("'", '') == chrs_name__[j]:
                    row = [metadata['chr'][i],metadata['start'][i],metadata['end'][i], j]
                    df_labels.loc[len(df_labels)] = row
          
        if plot_type == 'chrs_translocation':
            start = metadata['start'][i]
            end = metadata['end'][i]
            for t in range(1, len(translocation), 2): 
                t_end = translocation[t]
                t_start = translocation[t-1]
                if t_start >= start and t_end <= end:
                    row_i = ['nt'+metadata['chr'][i],metadata['start'][i],t_start-1,len(df_labels)-count]
                    row_t = ['t'+metadata['chr'][i],t_start,t_end,len(df_labels)+1-count]
                    row_f = ['nt'+metadata['chr'][i],t_end+1,metadata['end'][i],len(df_labels)-count]
                    df_labels.loc[len(df_labels)] = row_i
                    df_labels.loc[len(df_labels)] = row_t
                    df_labels.loc[len(df_labels)] = row_f
                    count = count+1
                    break
                  
    #str chr adjusting
    #for i in range(0,len(df_labels)):
        #df_labels['chr'][i] = df_labels['chr'][i].replace("'", '')
    
    
    #label assignment
    node_labels = []
    
    labels = []
    
    def labels_assig(network, df):
        
        labels.append([x for x in df['chr'] ])
        
        for node in sorted(network.nodes()):  
            for lab in range(0,len(df)):
                start = df['start'][lab]
                end = df['end'][lab]
                if start <= node <= end: 
                    node_labels.append(df['labels'][lab])
                    break
        return
    
    labels_assig(G, df_labels)
    
    node_labels = np.array(node_labels)
    
    return node_labels, labels, df_labels




###############################################################################################
#-------------------------------------------NODE2VEC------------------------------------------#


#from node2vec import Node2Vec
from fastnode2vec import Graph, Node2Vec
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import time
import itertools
from itertools import permutations
from itertools import combinations_with_replacement 
from sklearn.metrics import silhouette_score



def plot_embedding(graph, G, df_labels,node_labels, parameters, perm, par_to_cycle, save,folder, SI_text):
    time_series = []
    # Print the obtained permutations 
    for index, i in enumerate(perm):
        if len(perm[0])==6:
            parameters[par_to_cycle[0]] = i[0]
            parameters[par_to_cycle[1]] = i[1]
            parameters[par_to_cycle[2]] = i[2]
            parameters[par_to_cycle[3]] = i[3]
            parameters[par_to_cycle[4]] = i[4]
            parameters[par_to_cycle[5]] = i[5]
        else:
            parameters[par_to_cycle[0]] = i[0]
                
        print(index,'---',parameters)
        
        start = time.time()
        
        
        edges = []
        for u, v in G.edges():
            edges.append((str(u), str(v), G[u][v]['weight']))
            
        fastG = Graph(edges,
              directed=False, weighted=True)

        

        n2v = Node2Vec(fastG, dim=parameters[0], 
                              walk_length=parameters[1], 
                              context=parameters[2], 
                              p=parameters[3], 
                              q=parameters[4], 
                              workers=parameters[5],
                              )

        n2v.train(epochs=100, progress_bar=False)
        '''node2vec = Node2Vec(G, dimensions=parameters[0],
                            walk_length=parameters[1], 
                            num_walks=parameters[2], 
                            workers=1, 
                            p=parameters[3], 
                            q=parameters[4], 
                            seed=42,
                            workers=4)'''
        # Embed nodes
        #model = node2vec.fit(window=1,min_count=1,batch_words=4) # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        #model.save('EMBEDDING_MODEL_FILENAME.model')
        end = time.time()
        print("The time of execution of above program is :",
          (end-start)/60 , "min")
        if len(perm[0])==6:
            time_series.append((i[0],i[1],i[2],i[3],i[4],i[5],(end-start)/60))
        else:
            time_series.append((i[0],(end-start)/60))
        #convert to dataframe
        emb_df = (
        pd.DataFrame(
            #[model.wv.get_vector(str(int(n))) for n in sorted(graph.nodes())],index = sorted(graph.nodes)))
            [n2v.wv[str(n)] for n in sorted(graph.nodes())],index = sorted(graph.nodes)))
    
        #Visualize Embeddings
        pca = PCA(n_components = 2, random_state = 7)
        pca_mdl = pca.fit_transform(emb_df)
    
        emb_df_PCA = (
        pd.DataFrame(
            pca_mdl,
            columns=['x','y'],
            index = emb_df.index))
        
        #silhouette_score
        SI = silhouette_score(emb_df_PCA,node_labels)
        text = 'Silhouette Index:'+str(np.round(SI,3))
        print(text)
    
        fig = plt.figure(figsize=(20,12))
        scatter_x = np.array(emb_df_PCA['x'])
        scatter_y = np.array(emb_df_PCA['y'])

        cdict = {1: 'red',2: 'green', 3: 'darkblue', 4: 'yellow', 5: 'black', 6: 'gray', 7: 'lightsteelblue',
                 8: 'violet', 9: 'orange', 10: 'pink', 11: 'tan', 12: 'saddlebrown', 13: 'olive', 
                 14: 'indigo', 15: 'khaki', 16: 'teal', 17: 'goldenrod', 18: 'lime', 19: 'aqua',
                 20: 'crimson', 21: 'palegreen', 22: 'silver', 23: 'sandybrown'}

            
        labels_printed = []
        for j in range(0,len(df_labels)):
            label_name = df_labels['chr'][j]
            
            if label_name[0] == 't':
                if label_name in labels_printed: continue
                labels_printed.append(label_name)
                alpha = 0.35
                if label_name[-1] == 'X':
                    color = cdict[len(cdict)]
                else:
                    color = cdict[int(label_name.replace('tchr', ''))]
                ix = np.where(node_labels == df_labels['labels'][j])         
                scatter = plt.scatter(scatter_x[ix], scatter_y[ix], c = color, 
                                      s = 10, alpha = alpha, label=label_name)#marker='*'
                
                
            elif label_name[0] == 'n':
                if label_name in labels_printed: continue
                labels_printed.append(label_name)
                alpha = 1
                if label_name[-1] == 'X':
                    color = cdict[len(cdict)]
                else:
                    
                    color = cdict[int(label_name.replace('ntchr', ''))]
                ix = np.where(node_labels == df_labels['labels'][j])           
                scatter = plt.scatter(scatter_x[ix], scatter_y[ix], c = color, 
                                      s = 10, alpha = alpha, label=label_name)
                
            else:
                alpha = 1
                if label_name[-1] == 'X':
                    color = cdict[len(cdict)]
                else:
                    color = cdict[int(label_name.replace('chr', ''))]
                ix = np.where(node_labels == df_labels['labels'][j])           
                scatter = plt.scatter(scatter_x[ix], scatter_y[ix], c = color, 
                                      s = 10, alpha = alpha, label=label_name)
    
        plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0)
        if SI_text == True:
            plt.text(0.02, 0.95, text, style='italic', fontsize=12,transform=plt.gca().transAxes,
                     bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.title('PCA Visualization')
        plt.grid()
        plt.plot()
        
        if save == True:
                plt.savefig(folder+''+str(i)+str(index)+'.png')
                
        #del node2vec
        #del n2v
        
    return time_series

'''
#time execution plotting   
    
    d_wl_series = []
    time_ = []
    
    for t in range(0,len(time_series)):
        d_wl_series.append((time_series[t][0]))
        time_.append(time_series[t][1])
    
    plt.plot(range(0,len(time_)),time_, 'bx-')
    plt.xticks(np.arange(0,len(time_)),d_wl_series)
    plt.xlabel('parameter workers')
    plt.ylabel('Execution time (min)')
    plt.show()    
'''


###############################################################################################################
#-------------------------------------------MAIN-------------------------------------------------------------#


if __name__ == "__main__":
    
#dataset loading
    #'bin_not_reduced'-'bin_reduced'-'neg_bin_reduced'-'neg_bin_not_reduced'
    data_type = 'neg_bin_not_reduced'
    data, metadata, metadata_bin, data_toy = data_load(data_type)
    #adj_vis(data,metadata, box_vis =False)
    
#datatoy creation
    #data_toy = toy_df(data, metadata, metadata_bin, save_name='data_toy_neg_bin')
    #adj_vis(data_toy,metadata, box_vis =True)
    
#translocated datatoy creation
    #chrs_name = [['chr1','chrX'],['chr6','chr20']]
    chrs_name = [['chr6','chr20']]
    #'rbt'-'nrbt'-'ut'
    t_type = 'ut'
    #for chr1 chrX
    t_1 = [[5, 10],[1370,1375]]
    #for crh6 chr20
    t_2 = [[1050, 1055],[2700,2705]]
    translocation = [t_2]
    #data_toy_trans = df_translocation(data_toy,chrs_name,translocation,metadata,metadata_bin,t_type)
    #adj_vis(data_toy_trans,metadata, box_vis =False)
    
#network creation
    G = nx.from_numpy_matrix(data_toy.values)
    
#sub network creation
    metadata_sub, G_sub, data_toy_sub = sub_network(chrs_name, data_toy, metadata)
    adj_vis(data_toy_sub,metadata_sub, box_vis =False)

#labels
    #'all_no_translocation'-'all_translocation'-'chrs_no_translocation'-'chrs_translocation'
    plot_type = 'chrs_no_translocation'
    node_labels, labels, df_labels = labels_creation(chrs_name,translocation,metadata,plot_type,G)
    
#node2vec:
    #folder
    #folder = 'results/(tG,tG_sub)/nrbt/chr1_chrX_chr6_chr14/'

    folder = 'results/GM12878/(G,G_sub)/'
    folder_ = folder[:-1]
    if not os.path.exists(folder_):
        os.makedirs(folder_)
        
    #network to use for PCA   
    graph = G_sub
   
    #values =[0.01,0.05,0.1,0.25,0.5,0.75]
    #perm = [p for p in itertools.product(values, repeat=2)]
    
    '''
    perm = []
    dim_list =(4,5,7,10)
    wl_list= (100,150,250)
    cont_list=(10,50,100,200)
    p_list = (0.3,0.5)
    q_list = (5,10)
    wor_list=(20)
    for d in dim_list:
        for wl in wl_list:
            for p in p_list:
                for q in q_list:
                    for c in cont_list:
                        if wl == 150 and p == 0.5 and q ==10  :
                            perm.append([d,wl,c,p,q,20])'''

    perm = [[7,150,5,0.5,10,20]]
    
    parameters = [10,50,100,1,1,20]
    par_to_cycle = [0,1,2,3,4,5]
    time_series = plot_embedding(graph,G,df_labels,node_labels,
                                 parameters,perm,par_to_cycle,
                                 save=True,folder=folder,SI_text=False)
    
    '''
    #save txt times execution
    with open(folder+'time_series.txt', 'w') as file:
        for item in time_series:
            # write each item on a new line
            if len(item)== 3:
                file.write('%i,%i,%f\n' %item)
            else :
                file.write('%i,%f\n' %item)
    '''
        
        
        
        
    
    