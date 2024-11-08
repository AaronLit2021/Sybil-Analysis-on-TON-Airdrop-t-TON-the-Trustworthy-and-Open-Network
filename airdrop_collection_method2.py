import pandas as pd
import os

# Set project and load data
project = 'not'
data_path = f"./data/{project}_airdrop_cex_transfer_comment_flowin.csv"
jetton_outflow_sybil = pd.read_csv(data_path)

# Create a combined DataFrame with 'from_address' and 'to_address'
jetton_outflow_sybil_list = pd.concat([
    jetton_outflow_sybil[['from_address', 'cluster_id']].rename(columns={'from_address': 'address'}),
    jetton_outflow_sybil[['to_address', 'cluster_id']].rename(columns={'to_address': 'address'})
])

jetton_outflow_sybil_list.columns = ['address', 'center_addr_inflow_cluster_id']
jetton_outflow_sybil_list = jetton_outflow_sybil_list.drop_duplicates()

address_counts = jetton_outflow_sybil_list.groupby('center_addr_inflow_cluster_id')['address'].nunique().reset_index()
address_counts.columns = ['center_addr_inflow_cluster_id', 'unique_address_count']

# Filter for clusters with at least 5 unique addresses
valid_clusters = address_counts[address_counts['unique_address_count'] >= 5]['center_addr_inflow_cluster_id']

# Filter original data to retain only valid clusters
jetton_outflow_sybil_list = jetton_outflow_sybil_list[
    jetton_outflow_sybil_list['center_addr_inflow_cluster_id'].isin(valid_clusters)
]

# Merge the unique address count back into the filtered DataFrame
jetton_outflow_sybil_list = jetton_outflow_sybil_list.merge(
    address_counts,
    on='center_addr_inflow_cluster_id',
    how='left'
)

# Prefix 'center_addr_inflow_cluster_id' with 'aggregator_'
jetton_outflow_sybil_list['center_addr_inflow_cluster_id'] = 'aggregator_' + jetton_outflow_sybil_list['center_addr_inflow_cluster_id']

# Sort by 'unique_address_count' in descending order
jetton_outflow_sybil_list = jetton_outflow_sybil_list.sort_values(by='unique_address_count', ascending=False)

# Save the result to a CSV file
output_dir = "./result"
os.makedirs(output_dir, exist_ok=True)
jetton_outflow_sybil_list.to_csv(f"{output_dir}/aggregator_sybil_list.csv", index=False)

print('done') 