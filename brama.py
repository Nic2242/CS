# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:30:25 2024

@author: 531725ns
"""

import json
import hashlib
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from typing import Dict, List, Set
from brami import generate_vectors, minhash
from bral import lsh
from bramsm import MSM_Clusters


brand_names = ["akai", "alba", "apple", "arcam", "arise", "bang", "bpl", "bush", "cge", "changhong", "coby", "compal", "curtis",
          "durabrand", "element", "finlux", "fujitsu", "funai", "haier", "hisense", "hitachi", "itel", "insignia",
          "jensen", "jvc", "kogan", "konka", "lg", "loewe", "magnavox", "marantz", "memorex", "micromax", "metz",
          "onida", "panasonic", "pensonic", "philips", "planar", "proscan", "rediffusion", "saba", "salora", "samsung",
          "sansui", "sanyo", "seiki", "sharp", "skyworth", "sony", "tatung", "tcl", "telefunken", "thomson", "toshiba",
          "tpv", "tp vision", "vestel", "videocon", "vizio", "vu", "walton", "westinghouse", "xiaomi", "zenith"]  # Example brand names

def load_data():
    with open("C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Computer Science for Business Analytics/Data.json") as f:
        data = json.load(f)
    
    Pages = {}

    # Counter for generating unique page_ids
    page_counter = 0

    # Iterate over the raw data (model_id -> list of product pages)
    for model_id, page_list in data.items():
        for page in page_list:
            # Generate a unique page_id (e.g., "page_1", "page_2", ...)
            page_id = f"page_{page_counter}"
            page_counter += 1
            
            # Initialize brand to None by default
            brand = None
            
            # Extract words from the title, convert to lowercase for case-insensitive comparison
            title_words = set(page["title"].lower().split())

            # Compare words in title to the list of brand names
            for brand_name in brand_names:
                if brand_name in title_words:
                    brand = brand_name  # Assign the matched brand to the 'brand' key
                    break  # Exit loop after finding the first match

            # Add the product data to the `Pages` dictionary
            Pages[page_id] = {
                "model_id": model_id,
                "title": page["title"],
                "featuresMap": page["featuresMap"],
                "shop": page["shop"],
                "brand": brand  # Add the brand to the page data
            }

    N = len(Pages)
    duplicates_matrix = np.zeros((N, N)).astype(int)
    
    products = list(Pages.values())
    for i, p1 in enumerate(products):
        for j, p2 in enumerate(products):
            if i != j and p1["model_id"] == p2["model_id"]:
                duplicates_matrix[i, j] = 1


    print(f'\nN={N} (of which {len(data)} unique)\n')
    duplicates_matrix.sum()/2

    return Pages, duplicates_matrix

def performance_metrics(prefix: str,
                        predicted_duplicates,
                        actual_duplicates,
                        num_comparisons_made):
    duplicates_found = np.sum(predicted_duplicates * actual_duplicates) / 2
    total_num_duplicates = np.sum(actual_duplicates) / 2

    pair_quality = duplicates_found / num_comparisons_made
    pair_completeness = duplicates_found / total_num_duplicates
    f1 = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness)

    return {
        f'{prefix}f1': f1,
        f'{prefix}PQ': pair_quality,
        f'{prefix}PC': pair_completeness,
        f'{prefix}D_f': duplicates_found,
        f'{prefix}N_c': num_comparisons_made,
        f'{prefix}D_n': total_num_duplicates,
    }

def main():
    # Define your list of brand names to search for in the title
   
    Pages, duplicates_matrix = load_data()
    products = list(Pages.values())

    replications = 10
    n = 1200
    q = 3
   
  
    experiment_results: List[Dict] = []
    index_range = range(len(products))
    for bootstrap in tqdm(range(replications), desc="Replications"):

        current_indices = random.choices(index_range, k=len(products))
        current_products = [products[i] for i in current_indices]

        # Remove duplicates from both products and indices, preserving order
        seen = set()
        unique_indices = []
        unique_products = []
        
        for idx, product in zip(current_indices, current_products):
            if idx not in seen:
                unique_indices.append(idx)
                unique_products.append(product)
                seen.add(idx)
                
        current_products = unique_products   
        current_indices = unique_indices
        current_duplicates = np.array(
            [[duplicates_matrix[i, j] for j in current_indices] for i in current_indices])
        current_num_duplicates = np.sum(current_duplicates) / 2

        N=len(current_products)
        for r in tqdm([r for r in range(2, n) if n % r == 0], desc="(r,b) combinations", leave=False):
            b = round(n / r)
            vectors = generate_vectors(current_products)
            # create signature matrix
            signatures = minhash(vectors, r * b, len(vectors[:,0]))

            if np.isinf(signatures).sum() > 0:
                raise ValueError(f"M still contains infinite values")

            # apply LSH
            candidate_pair_matrix = lsh(signatures, b, r)
           
            # perform clustering
            model, num_comparisons_made = MSM_Clusters(current_products, candidate_pair_matrix, q)
            predicted_duplicates = np.array(
                [[int(model.labels_[i] == model.labels_[j]) for j in range(N)] for i in range(N)])
            np.fill_diagonal(predicted_duplicates, 0)

            experiment_results.append({
                'bootstrap': bootstrap,
                'n': n,
                'b': b,
                'r': r,
                'N': N,
                'num_duplicates': current_num_duplicates,
                **performance_metrics('lsh__',
                                      candidate_pair_matrix,
                                      current_duplicates,
                                      np.sum(candidate_pair_matrix)/2),
                **performance_metrics('clu__',
                                      predicted_duplicates,
                                      current_duplicates,
                                      num_comparisons_made)
            })
            
    df_results = pd.DataFrame(experiment_results)
    print(df_results)
    df_results.to_csv("results_new.csv")      

   

if __name__ == '__main__':
     main()
