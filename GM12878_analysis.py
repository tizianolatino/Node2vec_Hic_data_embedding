#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:46:56 2023

@author: tizianolatino
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import interpolate


############################################################################################################
#---------------------------------------DATA PROCESSING----------------------------------------------------#

data = pd.read_csv('data/raw_GM12878_1Mb.csv', header=None)
metadata = pd.read_csv('data/metadata.csv')


#fig, ax = plt.subplots()
#im = ax.imshow(data)


#drop chrY rows
data.drop(range(metadata.iloc[20]['start']-1,metadata.iloc[20]['end']),inplace=True)
#drop chrY columns
data.drop(data.columns[range(metadata.iloc[20]['start']-1,metadata.iloc[20]['end'])], axis=1,inplace=True)
#reset index and columns
data.reset_index(inplace = True, drop = True)
data.set_axis(list(range(0,data.shape[1])), axis=1, inplace=True)
#metadata start from zero
metadata[['start','end']] = metadata[['start','end']].sub(1, axis='columns')
#metadata drop chrY
metadata.drop(20,inplace=True)
metadata.reset_index(inplace = True, drop = True)
#metadata subtract chrY nodes from the last 3 chrs
metadata.loc[20:23, "end"] = metadata.loc[20:23, "end"].apply(lambda x: x - 60)
metadata.loc[20:23, "start"] = metadata.loc[20:23, "start"].apply(lambda x: x - 60)

#check isolated nodes
isoleted_nodes = []
for row in range(0,data.shape[0]):
    if sum(data.loc[row]) == 0: isoleted_nodes.append(row)

#remove from metadata the isolated node
isoleted_nodes.reverse()
isolated_nodes_chr = [0]*23
for nodes in isoleted_nodes:
    for row in range(0,metadata.shape[0]):
        if nodes <= metadata['end'].loc[row] and nodes >= metadata['start'].loc[row]:
            #print('nodes:',nodes,'rows:',row)       
            isolated_nodes_chr[row] +=1
            break
for i in range(0,len(isolated_nodes_chr)):
    metadata.loc[i:, "end"] = metadata.loc[i:, "end"].apply(lambda x: x - isolated_nodes_chr[i])
    metadata.loc[i+1:, "start"] = metadata.loc[i+1:, "start"].apply(lambda x: x - isolated_nodes_chr[i])

#remove from data the isolated node
data.drop(isoleted_nodes,inplace=True)
data.drop(data.columns[isoleted_nodes], axis=1,inplace=True)
data.reset_index(inplace = True, drop = True)
data.set_axis(list(range(0,data.shape[1])), axis=1, inplace=True)  

#check main diagonal zeroes
diag = np.diag(data, k=0)
np.where(diag!=0)

#save datframe
#data.to_csv('data/raw_GM12878_1Mb_prepro.csv', index=False)
#metadata.to_csv('data/metadata_prepro.csv', index=False)


############################################################################################################
#---------------------------------------DATA REDUCTION-----------------------------------------------------#


#select raw to remove

percent = 0.8
nodes_to_remove = []
for i in range(0,len(metadata)):
    start = metadata['start'][i]
    end = metadata['end'][i]
    interv = end-start+1
    len_row_to_remove = int(interv*percent)
    for row in range(0,len_row_to_remove):
        nodes_to_remove.append(end-row)
    

#remove from metadata the isolated node
nodes_to_remove.sort()
nodes_to_remove_chr = [0]*23
for nodes in nodes_to_remove:
    for row in range(0,metadata.shape[0]):
        if nodes <= metadata['end'].loc[row] and nodes >= metadata['start'].loc[row]:
            #print('nodes:',nodes,'rows:',row)       
            nodes_to_remove_chr[row] +=1
            break
for i in range(0,len(nodes_to_remove_chr)):
    metadata.loc[i:, "end"] = metadata.loc[i:, "end"].apply(lambda x: x - nodes_to_remove_chr[i])
    metadata.loc[i+1:, "start"] = metadata.loc[i+1:, "start"].apply(lambda x: x - nodes_to_remove_chr[i])

#remove from data the isolated node
data.drop(nodes_to_remove,inplace=True)
data.drop(data.columns[nodes_to_remove], axis=1,inplace=True)
data.reset_index(inplace = True, drop = True)
data.set_axis(list(range(0,data.shape[1])), axis=1, inplace=True)  

#save datframe
#data.to_csv('data/raw_GM12878_1Mb_prepro_reduced.csv', index=False)
#metadata.to_csv('data/metadata_prepro_reduced.csv', index=False)

############################################################################################################
#----------------------------------------ADJ MATRIX PLOT--------------------------------------------------#

import matplotlib.patches as patches

def adj_vis(A):
    fig, ax = plt.subplots()
    #im = ax.imshow(A,cmap='Reds')
    im = ax.imshow(A)
    
    yticks = []
    for i in range(0,len(metadata)):
        chri_len = metadata['end'][i]-metadata['start'][i]
        rect = patches.Rectangle((metadata['start'][i]-0.5, metadata['start'][i]-0.5), 
                                 chri_len+1, chri_len+1, 
                                   linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        middle_point = int(chri_len*0.5)+metadata['start'][i]
        yticks.append(middle_point)
    
    plt.yticks(yticks, labels=metadata['chr'], fontsize=10)
    plt.xlabel('Mbp', fontsize=13)
    
    fig.colorbar(im,location='right', anchor=(0, 0.3), shrink=0.7)

adj_vis(data)

############################################################################################################
#----------------------------------------NETWORK CREATION--------------------------------------------------#

#create weighted undirected grapg from data_norm
G = nx.from_numpy_matrix(data.values)

#save graph
#G = nx.read_gpickle('myGraph.gpickle')

'''
df2 = pd.DataFrame(np.array([[0,10,5,2], [10,0,20,7], [5,20,0,15],[2,7,15,0]]),columns=[0,1,2,3])
df2_norm = norm(df2)

graph = nx.from_numpy_matrix(df2.values)

graph_sub= graph.subgraph([2, 3])
'''

############################################################################################################
#----------------------------------------NETWORK ANALYSIS----------------------------------------------------#

'''
#total link density
density_tot = nx.density(G)

#density of single chr
density_chr = []
for chr_i in range(0,len(metadata)):
    density_chr_i = nx.density(G.subgraph(list(range(metadata['start'][chr_i],metadata['end'][chr_i]+1))))
    density_chr.append(density_chr_i)
    



#intrachrmosomal weight respect to the distance
weights = np.empty((300,)+(0,)).tolist()

for chr_i in range(0,len(metadata)):
    graph = G.subgraph(list(range(metadata['start'][chr_i],metadata['end'][chr_i]+1)))
    for edge in graph.edges():
        w = graph[edge[0]][edge[1]]["weight"]
        dist = abs(edge[0]-edge[1])
        weights[dist].append(w)

weights_mean = [0]
for dist in range(1,242):
    mean = np.mean(weights[dist])
    weights_mean.append(mean)
    
#linear plot
plt.plot(weights_mean)
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.ylabel('Average Intrachromosomal Contact Probability',fontsize='xx-large')
plt.show()    

#logplot with best fit line
fig=plt.figure()
ax = fig.add_subplot(111)
z=np.arange(len(weights_mean))
logA = np.log(z) #no need for list comprehension since all z values >= 1
logB = np.log(weights_mean)

m, c = np.polyfit(logA, logB, 4) # fit log(y) = m*log(x) + c
y_fit = np.exp(m*logA + c) # calculate the fitted values of y 

plt.plot(z, weights_mean, color = 'r')
plt.plot(z, y_fit, ':')

ax.set_yscale('symlog')
ax.set_xscale('symlog')
#slope, intercept = np.polyfit(logA, logB, 1)
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.ylabel('Average Intrachromosomal Contact Probability(log)',fontsize='xx-large')
ax.set_title('Pre Referral URL Popularity distribution')
plt.show()
'''
############################################################################################################
#----------------------------------------DENSITY ANALYSIS----------------------------------------------------#

#total zeros density
zeros_density = []

for dis in range(1,len(data)):
    zeros_density.append(sum(np.diag(data, k=dis)==0)/len(np.diag(data, k=dis)))

distances = list(range(1,len(zeros_density)+1))
#linear plot
plt.plot(distances, zeros_density)
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.ylabel('Zeros Density',fontsize='xx-large')
plt.show()    

############################################################################################################
#----------------------------------------INTRACHROMOSOMAL ANALYSIS-----------------------------------------#

data = pd.read_csv('data/raw_GM12878_1Mb_prepro.csv')
metadata = pd.read_csv('data/metadata_prepro.csv')


#intrachrmosomal weight respect to the distance
weights_intra = np.empty((250,)+(0,)).tolist()



for chr_i in range(0,len(metadata)):
    df = data.iloc[metadata['start'][chr_i]:metadata['end'][chr_i]+1,
                   metadata['start'][chr_i]:metadata['end'][chr_i]+1]
    for dis in range(1,len(df)):
        w = list(np.diag(df, k=-dis))
        for i in w:
            if i != 0.:
                weights_intra[dis].append(i)

weights_mean_intra = []
for dist in range(1,242):
    mean = np.mean(weights_intra[dist])
    weights_mean_intra.append(int(mean))

distances_intra = list(range(1,len(weights_mean_intra)+1))

#linear plot
plt.plot(distances_intra, weights_mean_intra)
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.ylabel('Average Intrachromosomal Contacts',fontsize='xx-large')
plt.show()    

#loglog plot 

plt.loglog(distances_intra, weights_mean_intra)
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.ylabel('Average Intrachromosomal Contacts (log)',fontsize='xx-large')
plt.show()       

#loglog plot with tangent line
from scipy import interpolate
import math


x = np.array(distances_intra)
y = np.array(weights_mean_intra)



def slope_cal(x0):
    x1 = [math.log(x[x0[0]],10),math.log(x[x0[1]],10)]
    y1 = [math.log(y[x0[0]],10),math.log(y[x0[1]],10)]
    slope = np.diff(y1)/np.diff(x1)
    return list(slope)

x0 = [7,29]
slope_x0 = np.round(slope_cal(x0)[0],3)

x1 = [0,2]
slope_x1 = np.round(slope_cal(x1)[0],3)

x2 = [2,7]
slope_x2 = np.round(slope_cal(x2)[0],3)

x3 = [29,120]
slope_x3 = np.round(slope_cal(x3)[0],3)

plt.loglog(x,y,label='intrachromosomal', linewidth=2)

plt.plot(x[x1],y[x1], "ob", markersize=5)
plt.axline((x[x1[0]], y[x1[0]]),(x[x1[1]], y[x1[1]]), linewidth=0.9, color='g', linestyle='--',
           label=str(slope_x1)+' , ['+str(x[x1[0]])+','+str(x[x1[1]])+'] Mbp')

plt.plot(x[x2],y[x2], "ob", markersize=5)
plt.axline((x[x2[0]], y[x2[0]]),(x[x2[1]], y[x2[1]]), linewidth=0.9, color='k', linestyle='--',
           label=str(slope_x2)+' , ['+str(x[x2[0]])+','+str(x[x2[1]])+'] Mbp')

plt.plot(x[x0],y[x0], "ob", markersize=5)
plt.axline((x[x0[0]], y[x0[0]]),(x[x0[1]], y[x0[1]]), linewidth=0.9, color='r', linestyle='--',
           label=str(slope_x0)+' , ['+str(x[x0[0]])+','+str(x[x0[1]])+'] Mbp')

plt.plot(x[x3],y[x3], "ob", markersize=5)
plt.axline((x[x3[0]], y[x3[0]]),(x[x3[1]], y[x3[1]]), linewidth=0.9, color='y', linestyle='--',
           label=str(slope_x3)+' , ['+str(x[x3[0]])+','+str(x[x3[1]])+'] Mbp')

plt.legend()
plt.tick_params(which = 'both')
plt.grid(which = 'major')
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.ylabel('Average Intrachromosomal Contact Probability(log)',fontsize='xx-large')
plt.show()

    


############################################################################################################
#----------------------------------------INTERCHROMOSOMAL ANALYSIS------------------------------------------#

#interchrmosomal weight respect to the distance

def interchr(df, dim, weights_mean):
    weights = np.empty((dim,)+(0,)).tolist()
    for dis in range(1,len(df)):
        print(dis)
        diagonal = list(np.diag(df, k=-dis))
        diag_index = []
        for i in range(0,len(diagonal)):
            diag_index.append((i,i+dis))
        #seleziono i valori della diagonale fuori dai chr
        for i in range(0,len(diag_index)):
            for index, row in metadata.iterrows():
                if row['start'] <=diag_index[i][0]<=row['end']:
                    start_i = index 
                if row['start'] <=diag_index[i][1]<=row['end']:
                    end_i = index                
            if start_i != end_i: weights[dis].append(diagonal[i])

    for dist in range(1,dim):
        mean = np.mean(weights[dist])
        weights_mean.append(mean)
     
    return weights_mean
 
weights_mean = []       
interchr(data,3000,weights_mean)

distances = list(range(1,len(weights_mean)+1))


#linear plot 
plt.plot(distances, weights_mean)
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.ylabel('Average Interchromosomal Contacts',fontsize='xx-large')
plt.show()    

#loglog plot 
plt.loglog(distances, weights_mean)
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.ylabel('Average Interhromosomal Contacts(log)',fontsize='xx-large')
plt.show()      

###########################################################################################################
#-----------------------------------------INTRA AND INTER PLOTTING ----------------------------------------------------#

#linear plot
plt.plot(distances, weights_mean, color='r', linestyle='--',label='Interchromosomal')
plt.plot(distances_intra, weights_mean_intra,label='Intrachromosomal')
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.legend(fontsize="15",loc ="center right")
plt.ylabel('Average Contacts',fontsize='xx-large')
plt.show()    

#loglog plot 

plt.loglog(distances, weights_mean, color='r', linestyle='--',label='Interchromosomal')
plt.loglog(distances_intra, weights_mean_intra,label='Intrachromosomal')
plt.xlabel('Genomic Distance(Mbp)', fontsize='xx-large')
plt.grid()
plt.legend(fontsize="15",loc ="upper right")
plt.ylabel('Average Contacts',fontsize='xx-large')
plt.show()      

###########################################################################################################
#-----------------------------------------SAVE NETWORK----------------------------------------------------#

nx.write_gpickle(G,'myGraph.gpickle')

G = nx.read_gpickle('myGraph.gpickle')


