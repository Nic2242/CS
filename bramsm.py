# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:31:27 2024

@author: 531725ns
"""

import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from brat import get_model_words_value

def sameShop(pi, pj):
    return pi['shop'] == pj['shop']

def diffBrand(pi, pj):
    return pi['brand'] != pj['brand']

def candidatePair(i , j, candidate_pairs):
    return candidate_pairs[i,j] == 1 or candidate_pairs[j,i] == 1
    
def intersect(set1, set2):
    smallest = set1 if len(set1) <= len(set2) else set2
    largest = set1 if len(set1) > len(set2) else set2

    return {i for i in smallest if i in largest}

def jaccard(value1, value2):
    if len(value1) == 0 and len(value2) == 0:
        return 0

    n_intersect = len(intersect(value1, value2))
    return n_intersect / (len(value1) + len(value2) - n_intersect)

def qShingle(text, q):
    # Create q-grams for both strings
    qgrams_text = [text[i:i+q] for i in range(len(text) - q + 1)]
    return set(qgrams_text)

def extract_feature_model_words(attributes):
    result = set()
    for value in attributes.values():
        for mw in get_model_words_value(value):
            result.add(mw)
    return result


def MSM_Clusters(products, candidate_pairs, q, epsilon = 0.9, gamma=0.775,  mu=0.65):
    inf_distance = 1000
    N = len(products)  # Total number of products
    distances = np.zeros((N, N))  # Initialize distance matrix
    np.fill_diagonal(distances, inf_distance)

    def set_distance(i, j, value):
        distances[i, j] = value
        distances[j, i] = value

    for i, p_i in tqdm(enumerate(products), desc="Clustering", leave=False):
        for j, p_j in tqdm(enumerate(products), desc="Clustering", leave=False):
        # Now you can use both the index (i, j) and the product (product_i, product_j)

            if i>=j:
                continue
    
            if sameShop(p_i, p_j) or diffBrand(p_i, p_j) or not candidatePair(i,j,candidate_pairs):
                set_distance(i, j, inf_distance)
                continue
         
            sim = 0
            m = 0 #number of matches
            w = 0 #weight of matches
    
            nmki = dict(p_i["featuresMap"])  # Non-matching keys of pi
            nmkj = dict(p_j["featuresMap"])  # Non-matching keys of pj

  
            for q_key, q_value in p_i["featuresMap"].items():
                for r_key, r_value in p_j["featuresMap"].items():
                    key_similarity = jaccard(qShingle(q_key, q),
                                             qShingle(r_key, q))
    
                    if key_similarity > gamma:
                        value_similarity = jaccard(qShingle(q_value, q),
                                                   qShingle(r_value, q))
                        weight = key_similarity
                        sim += weight * value_similarity
                        m += 1
                        w += weight
                        nmki.pop(q_key)  # Remove matching key-value pair
                        nmkj.pop(r_key)
                        
            avg_sim = sim / w if w > 0 else 0
            mw_perc = jaccard(extract_feature_model_words(nmki), extract_feature_model_words(nmkj))
            title_sim = jaccard(qShingle(p_i["title"].lower(), q),
                                qShingle(p_j["title"].lower(), q))
    
            min_features = min(len(p_i["featuresMap"]), len(p_j["featuresMap"]))
            h_sim: float
            if title_sim < .5:
                theta1 = m / min_features
                theta2 = 1 - theta1
                h_sim = theta1 * avg_sim + theta2 * mw_perc
            else:
                theta1 = (1 - mu) * m / min_features
                theta2 = 1 - mu - theta1
                h_sim = theta1 * avg_sim + theta2 * mw_perc + mu * title_sim
    
            set_distance(i, j, 1 - h_sim)

    # perform clustering
    model = AgglomerativeClustering(distance_threshold=epsilon,
                                    n_clusters=None,
                                    linkage='single',
                                    metric='precomputed')
    model.fit(distances)

    # sys.setrecursionlimit(10000)
    # plt.title("Hierarchical Clustering Dendrogram")
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(model)
    # plt.xlabel("Duplicate products")
    # plt.show()

    comparisons_made = np.zeros((N, N))

    for i, j in np.argwhere((distances > 0) & (distances < inf_distance)):
        comparisons_made[min(i, j), max(i, j)] = 1

    num_comparisons_made = comparisons_made.sum()

    return model, num_comparisons_made
