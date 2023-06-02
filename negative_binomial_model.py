#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:43:09 2023

@author: tizianolatino
"""


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


#data = pd.read_csv('data/raw_GM12878_1Mb_prepro_reduced.csv')
#metadata = pd.read_csv('data/metadata_prepro_reduced.csv')
data = pd.read_csv('data/raw_GM12878_1Mb_prepro.csv')
metadata = pd.read_csv('data/metadata_prepro.csv')

##########################################################################################################
#------------------------------------INTRACHROMOSOMAL NEGATIVE  BINOMIAL-------------------------------------------#


#plt.hist(diag,bins=int(np.sqrt(len(diag))))

metadata_bin_col = {'chr': [], 'distance': [], 'r': [], 'mean': [], 'p': []}
  
  

metadata_bin = pd.DataFrame(metadata_bin_col)

#chr_to_plot = "'chr21'"

for chr_i in range(0,len(metadata)):
    
    df = data.iloc[metadata['start'][chr_i]:metadata['end'][chr_i]+1,
                   metadata['start'][chr_i]:metadata['end'][chr_i]+1]
    n=df.sum().sum()
    for diag_i in range(1,len(df)):
        diag = np.diag(df, k=diag_i)
        if not diag.any():
            mean = np.mean(diag)  
            r = 1 
            p = 1
            row = [metadata['chr'][chr_i], int(diag_i), r, mean, p ]
            metadata_bin.loc[len(metadata_bin)] = row
        elif len(diag) <= 2:
            diag = np.append(diag,np.mean(diag)*2)
            diag = np.append(diag,0)
            mean = np.mean(diag)   
            var = np.var(diag)        
            p = mean/var
            r = mean**2/(var-mean)
            if p!=p:print(diag,mean,var,p,r)
            if var<mean:print(diag,mean,var,p,r)
            row = [metadata['chr'][chr_i], int(diag_i), r, mean, p ]
            metadata_bin.loc[len(metadata_bin)] = row
        else:          
            mean = np.mean(diag)   
            var = np.var(diag)        
            p = mean/var
            r = mean**2/(var-mean)
            if p!=p:print(diag,mean,var,p,r)
            if var<mean:
                diag = np.append(diag,np.mean(diag)*2)
                diag = np.append(diag,0)
                mean = np.mean(diag)   
                var = np.var(diag)        
                p = mean/var
                r = mean**2/(var-mean)
                if p!=p:print(diag,mean,var,p,r)
                if var<mean:print(diag,mean,var,p,r)
                row = [metadata['chr'][chr_i], int(diag_i), r, mean, p ]
                metadata_bin.loc[len(metadata_bin)] = row
                
            row = [metadata['chr'][chr_i], int(diag_i), r, mean, p ]
            metadata_bin.loc[len(metadata_bin)] = row
            
        
    
'''   
    if metadata['chr'][chr_i] == chr_to_plot :
    #if 0 != 1:        
        
        trials_tot = metadata_bin['p'].loc[metadata_bin['chr'] == metadata['chr'][chr_i]]
        distances =  list(range(1,len(trials_tot)+1))
        
        #linear plot
        plt.plot(distances, trials_tot,label=metadata['chr'][chr_i])
        plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
        plt.grid()
        plt.legend()
        plt.ylabel('Total Number of Contacts',fontsize='xx-large')
        plt.show()    

        #loglog plot 
        plt.loglog(distances, trials_tot,label=metadata['chr'][chr_i])
        plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
        plt.grid()
        plt.ylabel('Total Number of Contacts(log)',fontsize='xx-large')
        plt.show()       
'''
    
    

##########################################################################################################
#------------------------------------INTERCHROMOSOMAL NEGATIVE BINOMIAL-------------------------------------------#

'''
#based on distance
metadata_bin_col = {'chr': [], 'distance': [], 'n': [], 'mean': [], 'p': []}

metadata_bin_inter = pd.DataFrame(metadata_bin_col)

def interchr(df, dim, weights_mean):
    weights = np.empty((dim,)+(0,)).tolist()
    for dis in range(1,5):
        print(dis)
        diagonal = list(np.diag(df, k=-dis)) 
        print('lunghezza diagonale:',len(diagonal))
        diag_index = []
        for i in range(0,len(diagonal)):
            diag_index.append((i,i+dis))
        print(diag_index[0])
        #seleziono i valori della diagonale fuori dai chr
        for i in range(0,len(diag_index)):
            for index, row in metadata.iterrows():
                if row['start'] <=diag_index[i][0]<=row['end']:
                    start_i = index 
                if row['start'] <=diag_index[i][1]<=row['end']:
                    end_i = index                
            if start_i != end_i: weights[dis].append(diagonal[i])
    print(weights)
    for dist in range(1,dim):
        print(len(weights[dist]))
        mean = np.mean(weights[dist])
        weights_mean.append(mean)
     
    return weights_mean
 
weights_mean = []       
interchr(data,5,weights_mean)
'''


#uniform 
interchr_values = []

for i in range(1,len(metadata)):
    x = 0
    values = sum(data.iloc[metadata['start'][i]:metadata['end'][i]+1,0:metadata['start'][i]].values.tolist(),[])
    interchr_values.append(values)


interchr_tot = np.asarray(sum(interchr_values, []))


#negative binomial
n = sum(interchr_tot)
mean = np.mean(interchr_tot)   
var = np.var(interchr_tot)

p = mean/var
r = mean**2/(var-mean)

#s = np.random.negative_binomial(r, p, 100000)

row = ['interchr', 0., r, mean, p ]
metadata_bin.loc[len(metadata_bin)] = row

for i in range(0,len(metadata_bin)):
    metadata_bin['chr'][i] = metadata_bin['chr'][i].replace("'", '')

#metadata_bin.to_csv('data/metadata_neg_bin_reduced.csv', index=False)
metadata_bin.to_csv('data/metadata_neg_bin.csv', index=False)


for i in range(0,len(metadata)):
    metadata['chr'][i] = metadata['chr'][i].replace("'", '')

#metadata_bin.to_csv('data/metadata_neg_bin_reduced.csv', index=False)
metadata.to_csv('data/metadata_prepo.csv', index=False)




