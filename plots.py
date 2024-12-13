# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:04:13 2024

@author: 531725ns
"""

import pandas as pd
import matplotlib.pyplot as plt


data_bench = pd.read_csv("C:/Users/531725ns/OneDrive - Erasmus University Rotterdam/Master/Computer Science for Business Analytics/results_all_bench.csv", delimiter = ';')
data_new = pd.read_csv("C:/Users/531725ns/OneDrive - Erasmus University Rotterdam/Master/Computer Science for Business Analytics/results_new.csv")

data_new_new = data_new.rename(columns={col: 'new_' + col for col in data_new.columns})
df_results = pd.concat([data_bench, data_new_new], axis=1)

df_results['frac_comparison'] = df_results['lsh__N_c'] / (df_results['N']*df_results['N'] - df_results['N'])
result = df_results.groupby(['r','b']).agg('mean')
result = result.reset_index()
result.set_index(['r','b'],inplace = True)

plt.figure(figsize=(8,6))
plt.plot(result['frac_comparison'], result['lsh__f1'], color='black', label='LSH_f1')
plt.plot(result['frac_comparison'], result['new_lsh__f1'], color='red', label='new_LSH_f1')
plt.xlabel('frac_comparison')
plt.ylabel('LSH_f1')
plt.legend()
plt.savefig('LSH_f1_comparison.png', format='png')

# Plot LSH_PC and new_LSH_PC against frac_comparison
plt.figure(figsize=(8,6))
plt.plot(result['frac_comparison'], result['lsh__PC'], color='black', label='LSH_PC')
plt.plot(result['frac_comparison'], result['new_lsh__PC'], color='red', label='new_LSH_PC')
plt.xlabel('frac_comparison')
plt.ylabel('LSH_PC')
plt.legend()
plt.savefig('LSH_PC_comparison.png', format='png')

# Plot LSH_PQ and new_LSH_PQ against frac_comparison
plt.figure(figsize=(8,6))
plt.plot(result['frac_comparison'], result['lsh__PQ'], color='black', label='LSH_PQ')
plt.plot(result['frac_comparison'], result['new_lsh__PQ'], color='red', label='new_LSH_PQ')
plt.xlabel('frac_comparison')
plt.ylabel('LSH_PQ')
plt.legend()
plt.savefig('LSH_PQ_comparison.png', format='png')

# Plot MSM_f1 and new_MSM_f1 against frac_comparison
plt.figure(figsize=(8,6))
plt.plot(result['frac_comparison'], result['clu__f1'], color='black', label='MSM_f1')
plt.plot(result['frac_comparison'], result['new_clu__f1'], color='red', label='new_MSM_f1')
plt.xlabel('frac_comparison')
plt.ylabel('MSM_f1')
plt.legend()
plt.savefig('MSM_f1_comparison.png', format='png')

plt.legend()
plt.show()


