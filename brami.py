# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:27:28 2024

@author: 531725ns
"""

import numpy as np

from brat import get_model_words_title, get_model_words_value

def ExtractMW(page):
    MW_title = set()
    MW_feature = set()
    
    # Process the title first
    title = page['title']
    MW_title = get_model_words_title(title)
        
    # Now process the feature attributes (key-value pairs in featuresMap)
    features_map = page['featuresMap']
    for feature in features_map.values():
        MW_feature.update(get_model_words_value(feature))
    return MW_title.union(MW_feature)

def AllMW(products):
   MW = set()
   for product in products:
       MW.update(ExtractMW(product))
   return MW
   

def generate_vectors(products):
    # Number of pages (N) and number of model words (n)
    N = len(products)
    MW=AllMW(products)
    n = len(MW)
    # Create an empty matrix (n, N) filled with 0s
    vectors = np.zeros((n, N), dtype=int)
    
    # Create a mapping from model words to row indices
    word_to_index = {word: idx for idx, word in enumerate(MW)}
    
    # Fill the matrix
    for product_idx, product in enumerate(products):
        # For each page, check which model words are present and update the matrix
        for word in ExtractMW(product):
            if word in word_to_index:
                row_idx = word_to_index[word]
                vectors[row_idx, product_idx] = 1
    return vectors


#vocab size = #model words = #hash evaluations
def minhash(vectors, n, vocab_size):
    # Random coefficients for hash functions (a * r + b) % 
    a = np.random.randint(low=2147483648, high=4294967296, size=n, dtype=np.int64)
    b = np.random.randint(low=2147483648, high=4294967296, size=n, dtype=np.int64)
    
    p = 864158203  # A large prime number to use in the hash function
    
    if p < vocab_size:
        raise ValueError("p < vocab_size")
    
    # Initialize signature matrix with infinity values
    signatures = np.inf * np.ones((n, vectors.shape[1]))  # n rows, N columns (documents)
    
    # Create h_ir hash values for each row (feature index) in the vocabulary
    h_ir = np.array([(a * r + b) % p for r in range(vocab_size)]).T  # Hash values for each feature index
   
    # Iterate through each column (document) in vectors
    for c in range(vectors.shape[1]):  # columns are documents
        indices = set(np.where(vectors[:, c] == 1)[0])  # Get the non-zero indices for the document
        
        # Update the signature matrix for the current document
        for r in indices:
            # For each non-zero feature index, take the minimum of the current value in the signature and the hash value for this feature
            signatures[:, c] = np.minimum(signatures[:, c], h_ir[:, r])
    
    return signatures.astype(int)
