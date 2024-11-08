import os
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
from tqdm import tqdm  # Import tqdm for progress bar display

# Set project and data path
project = 'not'
data_jetton = pd.read_csv(f'./data/{project}_jetton_outflow_tx.csv')

# Ensure correct date parsing, using 'block_date' as the date column
data_jetton['date'] = data_jetton['block_date'].map(lambda x: str(x).split("_")[-1] if isinstance(x, str) else None)
date_len = data_jetton['date'].nunique()  # Get the number of unique dates

# DataFrame to store community information (only contains nodes and cluster_id)
community_df = pd.DataFrame(columns=['node', 'cluster_id'])

# Iterate over all unique dates with a tqdm progress bar
for index, tgt_date in enumerate(tqdm(data_jetton['date'].unique().tolist(), desc="Processing Dates")):
    print("=" * 30)
    print(f"Processing date: {tgt_date}")

    # Filter data to only process the current date
    data_jetton_tgt = data_jetton[data_jetton['date'] == tgt_date]

    G = nx.Graph()  # Use an undirected graph

    # Add edges to the graph, checking for valid nodes
    edges = data_jetton_tgt[['from_address', 'to_address', 'block_date', 'adjusted_amount']].values
    for edge in edges:
        from_address = edge[0]
        to_address = edge[1]
        amount = edge[3]

        # Check for invalid addresses
        if pd.isnull(from_address) or pd.isnull(to_address):
            print(f"Skipping invalid edge: from {from_address} to {to_address}")
            continue
        
        G.add_edge(from_address, to_address, value=amount)

    print('Searching for connected subgraphs')
    # Print the number of nodes and edges in the graph to confirm connectivity
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Perform community detection using the Louvain algorithm
    partition = community_louvain.best_partition(G, random_state=123)

    # Store nodes and their corresponding cluster_id for each community
    for community_idx, community_id in enumerate(set(partition.values())):
        community_nodes = [node for node, community in partition.items() if community == community_id]
        if len(community_nodes) >= 5:  # Only save communities with 5 or more nodes
            # Create a DataFrame with nodes and cluster_id
            node_data = pd.DataFrame({
                'node': community_nodes,
                'cluster_id': f"{community_idx}_{tgt_date}"  # Use community_idx and date combination
            })
            community_df = pd.concat([community_df, node_data], ignore_index=True)

    # Output results after processing each date
    print(f"Finished processing {tgt_date}, current number of communities: {community_df['cluster_id'].nunique()}")

# Save the final DataFrame as a CSV file
output_path = "./result"
os.makedirs(output_path, exist_ok=True)
community_df_save = community_df.copy()
community_df_save.to_csv(f"{output_path}/node_info_jetton.csv", index=False)
print('done') 