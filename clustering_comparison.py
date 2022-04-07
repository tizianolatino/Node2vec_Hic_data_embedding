#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:48:04 2022

@author: Tiziano Latino
"""
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

#clustering method
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def performance_int(data, labels_pred, metrica):
    SI = silhouette_score(data, labels_pred,metric=metrica)
    return SI


def clustering_comparison(k, data, metrica):
    
    clust_kmns = []
    clust_hier = []
    clust_GM = []
    clust_spec = []
    
    perf_models = {}
        
    #kmeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels_kmns = kmeans.labels_
    perf_models['clust_kmns'] = performance_int(data, labels_kmns, k)

    #hierarchical
    hier = AgglomerativeClustering(n_clusters=k, affinity='euclidean',linkage='ward')  
    labels_hier = hier.fit_predict(data)
    perf_models['clust_hier'] = performance_int(data, labels_hier, metrica)
       
    #spectral
    spec = SpectralClustering(n_clusters=k,random_state=0).fit(data)
    labels_spec = spec.labels_
    perf_models['clust_spec'] = performance_int(data, labels_spec, metrica) 
                                               
    # gaussian micture model
    GM = GaussianMixture(n_components=k).fit(data)
    labels_GM = GM.predict(data)
    perf_models['clust_GM'] = performance_int(data, labels_GM, metrica) 
    
    clust_kmns.append(perf_models.get('clust_kmns'))
    clust_hier.append(perf_models.get('clust_hier'))
    clust_GM.append(perf_models.get('clust_GM'))
    clust_spec.append(perf_models.get('clust_spec'))
     
    return clust_kmns, clust_hier, clust_spec, clust_GM
