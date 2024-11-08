import os
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain

# Load data
project = 'not'
data_first = pd.read_csv(f'./data/{project}_airdrop_first_tx.csv')

# Community detection
def detect_communities(data):
    community_df = pd.DataFrame(columns=['node', 'cluster_id'])

    print("Overview of data_first:")
    print(data.head())
    print("Number of rows in data_first:", len(data))

    for tgt_date in data['block_date'].unique():
        data_tgt = data[data['block_date'] == tgt_date]
        G = nx.Graph()
        edges = data_tgt[['from_address', 'to_address', 'value']].values
        G.add_edges_from((edge[0], edge[1], {'value': edge[2]}) for edge in edges)

        print(f"Graph for date {tgt_date}:")
        print("Number of nodes:", G.number_of_nodes())
        print("Number of edges:", G.number_of_edges())

        for component_idx, component in enumerate(nx.connected_components(G)):
            subgraph = G.subgraph(component)
            partition = community_louvain.best_partition(subgraph, random_state=123)

            for community_idx, community_id in enumerate(set(partition.values())):
                community_nodes = [node for node, community in partition.items() if community == community_id]

                print(f"Community {community_idx} on date {tgt_date} with {len(community_nodes)} nodes")

                if len(community_nodes) >= 5:
                    node_data = pd.DataFrame({'node': community_nodes, 'cluster_id': f"{component_idx}_{community_idx}_{tgt_date}"})
                    community_df = pd.concat([community_df, node_data], ignore_index=True)

    return community_df

community_df = detect_communities(data_first)

# Save results
os.makedirs("./result", exist_ok=True)
community_df.to_csv("./result/sybil_net_category.csv", index=False)
print("The sybil network categories have been successfully saved to the './result' directory.")