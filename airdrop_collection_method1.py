import os
import pandas as pd

# Set project and file paths
project = 'not'
input_file = f'./data/{project}_airdrop_cex_outflow.csv'
output_file = './result/cex_flowchain_sybil.csv'

# Load data
cex_flowchain_sybil = pd.read_csv(input_file)

cex_flowchain_sybil['comment_tx'] = cex_flowchain_sybil['comment_tx'].astype(str)
cex_flowchain_sybil['cluster_id'] = cex_flowchain_sybil['to_address'] + '_' + cex_flowchain_sybil['comment_tx']

# Calculate cluster size by counting unique 'from_address' for each 'cluster_id'
cluster_size = cex_flowchain_sybil.groupby('cluster_id')['from_address'].nunique().reset_index(name='cluster_size')

cex_flowchain_sybil = cex_flowchain_sybil.merge(cluster_size, on='cluster_id', how='left')
cex_flowchain_sybil = cex_flowchain_sybil.sort_values(by=['cluster_size', 'cluster_id'], ascending=False)

# Save the result to a CSV file
cex_flowchain_sybil.to_csv('./result/cex_flowchain_sybil.csv', index=False)

print('done') 