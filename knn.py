import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple

# Load Action Data
data_action = pd.read_csv("./data_action/not_airdrop_data_action.csv")

# Preprocess for High and Low Activity
min_date_cnt = 0
min_days_from_start = 0
min_score = 0
low_activity_threshold = 3

data_action = data_action.rename(columns={'from_address': 'from_addr', 'first_to_address': 'to_addr'})
data_action['total_fees_round'] = np.round(data_action['total_fees'], 2)

data_action = data_action[(data_action['ext_in_all_date_cnt'] >= min_date_cnt) &
                          (data_action['days_from_start'] >= min_days_from_start)]

low_activity_df = data_action[data_action['ext_in_all_hash_cnt'] <= low_activity_threshold]
high_activity_df = data_action[data_action['ext_in_all_hash_cnt'] > low_activity_threshold]


# High Activity Processing
def knn_process(df: pd.DataFrame, cols: List[str], ratio_limit: List[Tuple[float, float]], k_input: int,
                first_action_cols: List[str], last_action_cols: List[str],
                row_num: int = 10000) -> pd.DataFrame:
    centroid_df = df.groupby(cols).agg({'from_addr': 'nunique'}).rename(columns={'from_addr': 'cid_cnt'})
    centroid_df = centroid_df[centroid_df['cid_cnt'] >= 5].reset_index()
    centroids = centroid_df[cols].drop_duplicates().values
    feature_data = df[cols].values

    def assign_data_to_particles(centroids: np.ndarray, feature_data: np.ndarray,
                                 ratio_limit: List[Tuple[float, float]], k_input: int) -> np.ndarray:
        kdtree = KDTree(centroids)
        distances, closest_centroids_indices = kdtree.query(feature_data, k=k_input)
        labels = np.zeros((feature_data.shape[0], k_input), dtype=int)

        for i in range(feature_data.shape[0]):
            for j in range(k_input):
                index = closest_centroids_indices[i][j]
                centroid = centroids[index]
                new_point = feature_data[i]
                con1 = all((centroid[d] - new_point[d]) < (centroid[d] * ratio_limit[d][0] + ratio_limit[d][1]) for d in range(centroids.shape[1]))
                con2 = all((centroid[d] - new_point[d]) >= 0 for d in range(centroids.shape[1]))
                labels[i][j] = index if (con1 & con2) else -999

        return labels

    label_list = []
    for i in range(0, len(feature_data), row_num):
        labels = assign_data_to_particles(centroids, feature_data[i:(i + row_num)], ratio_limit, k_input)
        label_list.append(labels)

    label_list_new = [item for sublist in label_list for item in sublist.tolist()]
    addr_list = df['from_addr'].tolist()
    df_result = pd.DataFrame({'from_addr': addr_list, 'cid': label_list_new}).explode('cid')
    df_result = df_result[df_result['cid'] != -999]

    def process_cluster_data(cluster_df: pd.DataFrame, original_df: pd.DataFrame,
                             first_action_cols: List[str], last_action_cols: List[str], min_group_size: int) -> pd.DataFrame:
        cluster_size_df = cluster_df.groupby(['cid']).agg({'from_addr': 'nunique'}).rename(columns={'from_addr': 'cluster_size'})
        cluster_size_df = cluster_size_df[cluster_size_df['cluster_size'] >= min_group_size]
        cluster_df = cluster_df.merge(cluster_size_df, how='inner', on=['cid'])

        cols = ['from_addr', 'first_block_date', 'to_addr', 'first_method', 'last_block_date', 'last_to_address', 'last_method']
        cluster_df = cluster_df.merge(original_df[cols], how='left', on=['from_addr'])

        first_cluster_cols = ['cid'] + first_action_cols
        first_cluster_df = cluster_df.groupby(first_cluster_cols).agg({'from_addr': 'nunique'}).reset_index()
        first_cluster_df = first_cluster_df[first_cluster_df['from_addr'] >= min_group_size]
        first_cluster_df['knn_id'] = range(len(first_cluster_df))
        first_cluster_df['knn_id'] = first_cluster_df['knn_id'].astype(str) + '_first'
        del first_cluster_df['from_addr']

        knn_first = cluster_df.merge(first_cluster_df, how='inner', on=first_cluster_cols)[['from_addr', 'knn_id']]

        last_cluster_cols = ['cid'] + last_action_cols
        last_cluster_df = cluster_df.groupby(last_cluster_cols).agg({'from_addr': 'nunique'}).reset_index()
        last_cluster_df = last_cluster_df[last_cluster_df['from_addr'] >= min_group_size]
        last_cluster_df['knn_id'] = range(len(last_cluster_df))
        last_cluster_df['knn_id'] = last_cluster_df['knn_id'].astype(str) + '_last'
        del last_cluster_df['from_addr']

        knn_last = cluster_df.merge(last_cluster_df, how='inner', on=last_cluster_cols)[['from_addr', 'knn_id']]

        return pd.concat([knn_first, knn_last], axis=0)

    return process_cluster_data(df_result, df, first_action_cols, last_action_cols, min_group_size)

k_input = 5
min_group_size = 5
ratio_limit = [(0.1, 1), (0.1, 1), (0.1, 1), (0.1, 1), (0.1, 0), (0.1, 0), (0.01, 1), (0.01, 1)]
knn_feat_cols = ['ext_in_all_hash_cnt', 'ext_in_all_date_cnt', 'all_method_cnt', 'all_to_addr_cnt', 'ext_in_all_week_cnt', 'ext_in_all_month_cnt', 'date_range', 'days_from_start']
first_action_cols = ['first_block_date', 'to_addr', 'first_method', 'last_block_date']
last_action_cols = ['first_block_date', 'last_block_date', 'last_to_address', 'last_method']

knn_result = knn_process(high_activity_df, knn_feat_cols, ratio_limit, k_input, first_action_cols, last_action_cols)

# Low Activity Processing
def process_low_activity(df: pd.DataFrame, cols: List[str], min_group_size: int) -> pd.DataFrame:
    low_activity_cluster_df = df.groupby(cols).agg({'from_addr': 'nunique'}).rename(columns={'from_addr': 'cluster_size'})
    low_activity_cluster_df = low_activity_cluster_df[low_activity_cluster_df['cluster_size'] >= min_group_size]
    
    # Convert MultiIndex to a single string for 'knn_id'
    low_activity_cluster_df['knn_id'] = low_activity_cluster_df.index.to_frame().apply(lambda row: '_'.join(row.astype(str)), axis=1) + '_lowactive'

    return pd.merge(df, low_activity_cluster_df, on=cols, how='inner')[['from_addr', 'knn_id']]

low_activity_cols = ['ext_in_all_date_cnt', 'ext_in_all_hash_cnt', 'all_method_cnt', 'all_to_addr_cnt', 'first_block_date', 'to_addr', 'first_method', 'last_block_date', 'last_to_address', 'last_method']
low_activity_result = process_low_activity(low_activity_df, low_activity_cols, min_group_size)

# Combine High and Low Activity Results
def sybil_result_concat(knn_result: pd.DataFrame, low_activity_result: pd.DataFrame) -> pd.DataFrame:
    sybil_result = pd.concat([knn_result, low_activity_result], axis=0)
    cluster_size_df = sybil_result.groupby('knn_id')['from_addr'].nunique().reset_index().rename(columns={'from_addr': 'cluster_size'})
    return sybil_result.merge(cluster_size_df, on='knn_id', how='left').sort_values(by='cluster_size', ascending=False)

def generate_detail(sybil_result: pd.DataFrame, data_action: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(sybil_result, data_action, on='from_addr', how='inner').sort_values(by=['cluster_size', 'knn_id'], ascending=False)

sybil_result = sybil_result_concat(knn_result, low_activity_result)
sybil_detail = generate_detail(sybil_result, pd.concat([low_activity_df, high_activity_df]))

def generate_knn_id_new(sybil_result: pd.DataFrame, sybil_detail: pd.DataFrame, knn_feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    check_agg_msg = ['knn_id'] + knn_feat_cols
    agg_size_desc = sybil_detail.groupby(check_agg_msg).agg({'from_addr': 'nunique'}).rename(columns={'from_addr': 'agg_size'}).reset_index()
    agg_size_desc = agg_size_desc.drop_duplicates(['knn_id'])
    agg_size_desc['knn_id_agg'] = agg_size_desc.apply(lambda row: "_".join(row.astype(str)), axis=1)
    sybil_result = sybil_result.merge(agg_size_desc[['knn_id', 'knn_id_agg']], how='left', on='knn_id')
    sybil_detail = sybil_detail.merge(agg_size_desc[['knn_id', 'knn_id_agg']], how='left', on='knn_id')
    sybil_detail['knn_id'] = sybil_detail['knn_id_agg']
    sybil_result['knn_id'] = sybil_result['knn_id_agg']
    return sybil_result.drop(columns='knn_id_agg'), sybil_detail.drop(columns='knn_id_agg')

sybil_result, sybil_detail = generate_knn_id_new(sybil_result, sybil_detail, knn_feat_cols)

# Ensure the result directory exists
os.makedirs("./result", exist_ok=True)

sybil_result.to_csv("./result/not_airdrop_knn_result.csv", index=False)
sybil_detail.to_csv("./result/not_airdrop_knn_detail.csv", index=False)

# Print confirmation message
print("The sybil_result and sybil_detail have been successfully saved to the './result' directory.")
