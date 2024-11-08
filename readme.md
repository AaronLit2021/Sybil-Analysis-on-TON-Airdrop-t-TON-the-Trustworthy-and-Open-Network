# Project Name

## Introduction
This project analyzes airdrop and CEX collection data to identify Sybil clusters and community structures using network analysis techniques. By leveraging Gas Transfer Graphs (GTGs), Jetton Transfer Graphs (JTGs), and centralized exchange deposit patterns, this framework enables comprehensive detection of Sybil behaviors.

## Features
- Community detection using the Louvain method.
- Identification of Sybil clusters based on transaction patterns.
- Graph classification to determine network structures.
- Comprehensive analysis of Sybil behaviors through multiple approaches.


## Usage

Sybil attackers exploit bots and automated scripts to manage multiple accounts, enabling them to interact extensively with the TON chain. These bulk-operated accounts exhibit interconnected and similar behaviors, allowing them to be grouped into a single cluster for identification. We have developed three approaches targeting each phase of Sybil cluster formation—generation, transfer, and consolidation to identify TON Sybil accounts.

### Approach 1: Two-Phase Approach
In EVM, Trusta Labs has developed a two-stage framework that uses AI and machine learning clustering algorithms to identify Sybil clusters. Please refer to [Trusta's AI and Machine Learning Framework](https://medium.com/@trustalabs.ai/trustas-ai-and-machine-learning-framework-for-robust-sybil-resistance-in-airdrops-ba17059ec5b7) for a comprehensive overview of our approach.

- **Stage One**: Employs community detection algorithms such as Louvain and K-CORE to analyze Gas Transfer Graphs (GTGs) and Jetton Transfer Graphs (JTGs). This stage aims to detect tightly connected and suspicious Sybil groups.
  - **Script**: `gtg.py` and `jtg.py`
  - **Command**: 
    ```bash
    python gtg.py
    python jtg.py
    ```

- **Stage Two**: Focuses on analyzing account profiles and on-chain behaviors. The K-means algorithm is applied to further refine the results of Stage One.
  - **Script**: `knn.py`
  - **Command**: 
    ```bash
    python knn.py
    ```

### Approach 2: Highly Suspicious GTGs and JTGs
Based solely on graph mining algorithms on Gas Transfer Graphs (GTGs) and Jetton Transfer Graphs (JTGs), we can identify highly suspicious patterns such as Tree-Structured and Chain-like formations.
- **Script**: `gtg.py` and `jtg.py`
- **Command**: 
  ```bash
  python gtg.py
  python jtg.py
  ```

### Approach 3: Airdrop Collection into CEX
Sybil accounts often deposit airdropped tokens into the same centralized exchange (CEX) account during the token collection phase. Trusta has developed an approach to identify convergent token flows into the same CEX account.
- **Script**: `airdrop_collection_method1.py` and `airdrop_collection_method2.py`
- **Command**: 
  ```bash
  python airdrop_collection_method1.py
  python airdrop_collection_method2.py
  ```

## Data
Due to upload size limitations, the repository only contains sample data for specific dates. Please replace the sample data with your full dataset for complete analysis.

## Configuration
Ensure that the data files are placed in the `./data` directory with the correct naming conventions as specified in the scripts.
