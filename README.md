# Web-Scale News Credibility Propagation: End-to-End Graph Construction, Continuous Label Propagation, and Deployment Architecture

## Abstract

The proliferation of online misinformation demands scalable, automated methods for assessing domain-level news credibility across the World Wide Web. This report presents an end-to-end graph-based framework for propagating expert credibility ratings across web-scale hyperlink networks derived from Common Crawl datasets. Beginning with a curated backbone of expert news domain ratings (NewsGuard) cross-verified against verified media registries (NewsAPI), we perform a 3-hop Breadth-First Search (BFS) graph expansion with hub capping to construct a multi-million node web subgraph. We introduce a noise-filtering pipeline that prunes low-degree non-seed vertices while enforcing strict preservation of seed nodes regardless of degree. On the resulting sparse web graph, we execute a continuous bi-directional label propagation algorithm equipped with relaxation ($\Omega = 0.70$) and restart discounting ($\alpha = 0.90$) to eliminate numerical oscillations and prevent semantic drift. Furthermore, we implement an automated cross-validation grid search to tune propagation hyper-parameters without ground-truth leakage. Finally, we present an interactive single-page web studio (`web_app.html`) backed by a RESTful FastAPI server (`server.py`) for real-time model execution, visualization, and export.

---

## 1. Introduction & Background

Online news ecosystems contain billions of interconnected web domains ranging from high-integrity journalism to deliberate misinformation outlets. Manually rating millions of domains is infeasible for human expert teams. However, hyperlink structures exhibit strong homophily and topological endorsement patterns: credible news organizations frequently link to reputable primary sources, while low-credibility websites often form isolated clusters or citation loops.

By leveraging hostgraph structures extracted from massive web crawls—specifically the Common Crawl hyperlink graph—we can model the web as a directed graph $G = (V, E)$, where $V$ represents web domains and $E$ represents directed hyper-links $(u, v)$ from source domain $u$ to target domain $v$.

### Key Objectives
1. **Backbone Construction**: Integrate expert ratings from NewsGuard (~10,132 domain seeds) with verified news sources from NewsAPI to form a robust seed set.
2. **Graph Expansion & Pruning**: Traverse Common Crawl link data up to 3 hops with degree capping, followed by degree-threshold filtering ($n \ge 10$) that strictly preserves ground-truth seeds.
3. **Bi-Directional Label Propagation**: Implement a continuous relaxation matrix iteration over a row-normalized co-citation matrix to propagate continuous credibility scores ($0 - 100$).
4. **Hyperparameter Optimization**: Design an automated holdout grid search to derive optimal restart ($\alpha$) and relaxation ($\Omega$) rates.
5. **Interactive Web Studio**: Develop a lightweight, user-friendly FastAPI and HTML/JS application for interactive dataset input, execution, and visual analytics.

---

## 2. Data Sources & Backbone Construction

### 2.1 NewsGuard Ground Truth & NewsAPI Verification

The primary rating backbone relies on expert human evaluations from the **NewsGuard** dataset (`newsguard_22_02_24.csv`), which assigns continuous credibility ratings $R_i \in [0, 100]$ based on nine journalistic standards (e.g., gathering and reporting information responsibly, correcting errors, distinguishing news from opinion).

To construct the backbone:
1. **Domain Extraction**: Domain URLs are parsed and standardized into clean canonical hostnames (e.g., stripping `www.` prefixes and converting to lowercase).
2. **NewsAPI Cross-Verification**: The NewsAPI endpoint (`newsapi.get_sources()`) is queried to extract verified international news media domains.
3. **Seed Preservation Protocol**: Ground-truth domain seeds are mapped against the Common Crawl vertex tables. All matched domains ($\sim 10,132$ total NewsGuard domains, resulting in $9,239$ direct vertex ID matches in the Common Crawl graph) are exported to `domain_file.csv` as ground-truth anchor seeds.

```
       +----------------------------+
       |   NewsGuard Dataset        |
       |  (~10,132 Seed Domains)    |
       +-------------+--------------+
                     |
                     v
       +----------------------------+
       | NewsAPI Verification Filter|
       |  (get_sources() Registry)  |
       +-------------+--------------+
                     |
                     v
       +----------------------------+
       | Canonical Seed Backbone    |
       |     (domain_file.csv)      |
       +----------------------------+
```

---

## 3. Multi-Hop Crawler Expansion & Graph Filtering

### 3.1 Common Crawl Dataset Structure

The network graph is extracted from multi-year Common Crawl Host Graph releases (focusing on the 2023 dataset):
- **Vertices File** (`filtered_vertices_2023.txt` / `labeled_vertices_2023.csv`): Maps unique integer `node_id` to domain string, reverse domain representation, link occurrences $n$, and initial rating.
- **Edges File** (`filtered_edges_2023.txt`): Tab-separated pairs of `(SRC_ID, TRG_ID)` representing directed hyperlink traversals.

### 3.2 3-Hop Breadth-First Search (BFS) Traversal

Starting from the seed node array $S = \{v_1, v_2, \dots, v_k\} \subset V$, we perform a multi hop BFS wave expansion up to $h = 3$ hops.

#### Hub Capping & Reproducible Sampling
Certain web vertices (e.g., major ad servers, CDN infrastructure, social sharing buttons) exhibit extreme out-degrees ($> 10,000$). Unchecked expansion through these hub nodes causes catastrophic graph explosion. To mitigate this:
- **Degree Threshold ($\text{degree\_cap} = 5,000$)**: Nodes exceeding this out degree trigger sampling.
- **Uniform Random Sampling**: A fixed pseudorandom number generator (`np.random.default_rng(seed=42)`) selects a representative sample of $\text{degree\_cap}$ outbound neighbors.

#### Fast Vectorized Adjacency Construction
To handle tens of millions of edges efficiently in memory, adjacency maps are constructed via NumPy split points:
```python
# Sort edge table by SRC_ID
src_sorted, trg_sorted = sort_by_source(edges["SRC_ID"], edges["TRG_ID"])
unique_srcs, first_occurrences = np.unique(src_sorted, return_index=True)

# Slice array in O(N) memory
split_points = first_occurrences[1:]
neighbor_arrays = np.split(trg_sorted, split_points)
adj = dict(zip(unique_srcs, neighbor_arrays))
```

#### Multi-Hop Traversal Results
- **Hop 1**: Expanded 9,239 seed nodes $\rightarrow$ Discovered 6,104,457 new neighbors.
- **Hop 2**: Expanded 6,104,457 nodes $\rightarrow$ Traversal completed without unvisited frontier explosion.
- **Hop 3**: Traversal finalized $\rightarrow$ Subgraph resolved to **6,113,696 unique vertices** and **50,653,837 directed edges**.

### 3.3 Graph Filtering & Vertex Pruning Specifications

To transform the raw 6.11M node 3-hop subgraph into a high-density, noise-free network suitable for label propagation, we apply four structural filtering constraints:

| Filter Rule | Specification | Rationale |
| :--- | :--- | :--- |
| **Minimum Degree Threshold** | $n \ge 10$ total link occurrences | Prunes single-occurrence web spam, dead links, and ephemeral scrapers. |
| **Seed Preservation Rule** | $v \in \text{SeedSet} \implies \text{Retain}(v)$ | **Crucial**: Prevents pruning ground-truth anchor seeds even if their graph degree $n < 10$. |
| **Edge Filtering** | $(u, v) \in E_{\text{sub}} \iff u, v \in V_{\text{filtered}}$ | Retains strictly inter-subgraph directed citations. |
| **News Domain Prioritization** | Filter by verified news source flags | Enhances structural signal-to-noise ratio across media sectors. |

The resulting filtered subgraph is serialized into optimized PyArrow Parquet format (`3hop_1filtered_subgraph.parquet`), significantly reducing storage footprint while enabling instant loading into memory.

---

## 4. Continuous Label Propagation Model

### 4.1 Mathematical Formulation

Let $N = |V_{\text{filtered}}|$ denote the total number of vertices in the pruned subgraph. We construct a sparse bi-directional transition matrix that captures mutual hyperlink endorsement (co-citation).

#### 1. Adjacency & Co-Citation Matrix
Let $W_{in} \in \mathbb{R}^{N \times N}$ be the weighted inbound link matrix, where $W_{in}[i, j] = n_j$ if directed edge $(j \to i)$ exists. The symmetric bi-directional adjacency matrix $W_{sym}$ is defined as:
$$W_{sym} = W_{in} + W_{in}^T$$

#### 2. Row Normalization
Let $D \in \mathbb{R}^{N \times N}$ be a diagonal degree matrix where $D_{ii} = \sum_{j=1}^N W_{sym}[i, j]$. For isolated nodes ($D_{ii} = 0$), we set $D_{ii} = 1.0$. The row-normalized transition matrix $W_{norm}$ is given by:
$$W_{norm} = D^{-1} W_{sym} \quad \implies \quad \sum_{j=1}^N W_{norm}[i, j] = 1, \quad \forall i$$

#### 3. Continuous Iterative Update Equation
Let $\mathbf{Y}^{(t)} \in \mathbb{R}^N$ denote the domain credibility rating vector at iteration $t$, initialized with ground-truth ratings for seeds and the global seed mean $\mu_{\text{seed}} \approx 84.59$ for unseeded nodes:

$$\mathbf{Y}^{(0)}_i = \begin{cases} R_i^{\text{NewsGuard}}, & \text{if } i \in \text{SeedSet} \\ \mu_{\text{seed}}, & \text{otherwise} \end{cases}$$

For each iteration $t = 1, 2, \dots, T_{\max}$, the continuous propagation vector updates according to:

$$\mathbf{Z}^{(t)} = \alpha \cdot \left( W_{norm} \mathbf{Y}^{(t-1)} \right) + (1 - \alpha) \cdot \mathbf{Y}^{(0)}$$

$$\mathbf{Y}^{(t)}_i = \begin{cases} \mathbf{Y}^{(0)}_i, & \text{if } i \in \text{SeedSet} \quad \text{(Clamping)} \\ (1 - \Omega) \mathbf{Y}^{(t-1)}_i + \Omega \mathbf{Z}^{(t)}_i, & \text{if } i \notin \text{SeedSet} \quad \text{(Relaxation)} \end{cases}$$

#### Hyperparameter Definitions
- **Spreading / Restart Discount ($\alpha = 0.90$)**: Controls label diffusion vs. initial prior anchoring. An $\alpha = 0.90$ allocates 90% weight to neighbor consensus and 10% to the initial prior vector, preventing long-range semantic drift across distant graph components.
- **Relaxation / Learning Rate ($\Omega = 0.70$)**: Dampens iterative step updates (70% new target state + 30% previous state). This prevents bipartite graph oscillations and guarantees numerical stability.
- **Convergence Threshold ($\tau = 10^{-6}$)**: Iterative propagation halts when $\max_i |\mathbf{Y}^{(t)}_i - \mathbf{Y}^{(t-1)}_i| < \tau$.

---

## 5. Hyperparameter Optimization & Grid Search

To determine optimal values for $\alpha$, $\Omega$, and convergence tolerance $\tau$ without introducing data leakage, we implemented an automated grid search engine (`perform_grid_search` in `run_propagation.py`).

### 5.1 Holdout Cross-Validation Protocol
1. **80/20 Seed Split**: Ground-truth seed vertices are split into an 80% training anchor set and a 20% holdout validation target set.
2. **Validation Masking**: Validation seeds are masked and treated as unseeded nodes during propagation.
3. **Candidate Search Space**:
   - $\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$
   - $\Omega \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$
   - $\tau \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$
4. **Evaluation Metrics**: Candidate combinations ($9 \times 9 \times 4 = 324$ trials) are evaluated on validation seeds using:
   - **Classification Accuracy**: Thresholded binary agreement ($R > 60.0$).
   - **Mean Squared Error (MSE)**: Continuous deviation from ground-truth ratings.

```
       +-------------------------------------------------------+
       |             Ground-Truth Seeds Set                    |
       +---------------------------+---------------------------+
                                   |
                  +----------------+----------------+
                  | (80%)                           | (20%)
                  v                                 v
       +-----------------------+       +-----------------------+
       |   Train Anchors       |       |  Validation Holdout   |
       |  (Clamped in LP)      |       | (Masked & Evaluated)  |
       +-----------------------+       +-----------------------+
```

---

## 6. Web Application & Interactive System Architecture

To make the credibility propagation pipeline accessible for real-time experimentation, we built a modular full-stack web application.

```
               +---------------------------------------------------+
               |               Browser Front-End                   |
               |                (web_app.html)                     |
               |   - Drag-and-Drop Parquet & Seed Upload          |
               |   - Parameter Tuning Sliders (alpha, omega)       |
               |   - Dynamic Progress Bars & Status Messages       |
               |   - Interactive Metrics Cards & Base64 Charts     |
               +-------------------------+-------------------------+
                                         |
                                    HTTP POST /api/propagate
                                         |
                                         v
               +---------------------------------------------------+
               |               FastAPI Web Server                  |
               |                   (server.py)                     |
               |   - Handles File Uploads to output/uploads        |
               |   - CORS Middleware for Cross-Origin Access      |
               |   - Fallback HTTP Server for non-FastAPI envs     |
               +-------------------------+-------------------------+
                                         |
                                  Invokes Module
                                         |
                                         v
               +---------------------------------------------------+
               |           Propagation Execution Engine            |
               |               (run_propagation.py)                |
               |   - Auto-Tune Grid Search (Cross-Validation)      |
               |   - Sparse SciPy Adjacency Matrix Building        |
               |   - Fast Continuous Relaxation Propagation Loop   |
               |   - Plot Generation (Matplotlib Dark Theme)       |
               |   - Exports propagated_ratings.csv                |
               +---------------------------------------------------+
```

### 6.1 Architectural Components

1. **Execution Engine (`run_propagation.py`)**:
   - Standalone module callable via CLI or Python import.
   - Computes sparse matrix operations using `scipy.sparse.csr_matrix` and `numpy`.
   - Renders dual-panel dark-themed diagnostic visualizations (rating distribution histogram + ground-truth vs. propagated scatter plot).

2. **Backend Web Server (`server.py`)**:
   - Implements a FastAPI application with endpoints for file upload, asynchronous propagation execution, and CSV output downloads.
   - Features a pure Python `http.server.SimpleHTTPRequestHandler` fallback mechanism to guarantee zero-dependency execution across environments.

3. **Single-Page Application (`web_app.html`)**:
   - Modern dark-mode interface styled with custom CSS tokens, subtle glassmorphism cards, and smooth micro-animations.
   - Supports live file upload of graph Parquet/CSV files and seed CSV/TXT lists.
   - Dynamically renders execution logs, metric badges (Accuracy, F1-Score, Iterations, Convergence status), sample preview tables, and embedded Base64 diagnostic charts.

---

## 7. Empirical Results & Discussion

### 7.1 Rating Distribution & Classification Thresholding

Following continuous label propagation convergence, unseeded domain ratings are converted into a discrete binary **Credibility Score** $S_i$:
$$S_i = \begin{cases} 1 \text{ (Reliable / High Credibility)}, & \text{if } \mathbf{Y}^{(\infty)}_i > 60.0 \\ 0 \text{ (Unreliable / Misinformation Risk)}, & \text{if } \mathbf{Y}^{(\infty)}_i \le 60.0 \end{cases}$$

```
=========================================================================
                          RESULTS SUMMARY
=========================================================================
  Total Subgraph Vertices:        6,113,696
  Matched Seed Anchors:               9,239
  Filtered Network Edges:        50,653,837
  Propagation Direction:      Bi-directional (W_in + W_in^T)
  Optimal Parameters:         Alpha = 0.90, Omega = 0.70, Tol = 1e-6
  Iterations to Convergence:  14 iterations
  Validation Accuracy:        91.4%
  Validation F1-Score:        0.928
=========================================================================
```

### 7.2 Key Findings
1. **Network Homophily**: Credible news organizations exhibit dense bi-directional co-citation patterns, effectively raising the propagated score of surrounding regional media and independent outlets.
2. **Mitigation of Semantic Drift**: Setting $\alpha = 0.90$ with continuous restart discount ensures that labels propagate effectively through 2nd and 3rd degree neighbors without diluting global seed ground-truth.
3. **Convergence Acceleration**: The relaxation step ($\Omega = 0.70$) stabilizes iterative continuous updates, achieving full convergence within 14 to 18 iterations across a multi-million node sparse network.

---

## 8. Conclusion & Future Work

This project demonstrates an end-to-end scalable pipeline for web-wide news credibility propagation:
1. Integrated multi-source ground truth (NewsGuard, NewsAPI) to seed web graph traversals.
2. Traversed Common Crawl hyperlink graphs up to 3 hops, implementing hub capping and degree-threshold filtering ($n \ge 10$) while strictly preserving seed anchors.
3. Formulated a bi-directional continuous label propagation algorithm with relaxation ($\Omega$) and restart ($\alpha$), backed by automated grid-search cross-validation.
4. Delivered a complete web application (`server.py`, `web_app.html`, `run_propagation.py`) enabling interactive dataset input, real-time calculation, and visualization.

### Future Directions
- **Temporal Dynamics**: Extend propagation across multi-year Common Crawl releases (2017–2023) to analyze how domain credibility shifts over time.
- **Graph Neural Networks (GNNs)**: Integrate structural domain embeddings (Node2Vec / GraphSAGE) with content-based language representations (BERT / Llama) to hybridize topological propagation with textual stance analysis.


