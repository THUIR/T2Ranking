# T2Ranking

T2Ranking is a large-scale Chinese benchmark for passage ranking. The details about T2Ranking are elaborated in [this paper](https://github.com/THUIR/T2Ranking).


Passage ranking are important and challenging topics for both academics and industries in the area of Information Retrieval (IR). The goal of passage ranking is to compile a search result list ordered in terms of relevance to the query from a large passage collection. Typically, Passage ranking involves two stages: passage retrieval and passage re-ranking. 

To support the passage ranking research, various benchmark datasets are constructed. However, the commonly-used datasets for passage ranking usually focus on the English language. For non-English scenarios, such as Chinese, the existing datasets are limited in terms of data scale, fine-grained relevance annotation and false negative issues.


To address this problem, we introduce T2Ranking, a large-scale Chinese benchmark for passage ranking. T2Ranking comprises more than 300K queries and over 2M unique passages from real- world search engines. Specifically, we sample question-based search queries from user logs of the Sogou search engine, a popular search system in China. For each query, we extract the content of corresponding documents from different search engines. After model-based passage segmentation and clustering-based passage de-duplication, a large-scale passage corpus is obtained. For a given query and its corresponding passages, we hire expert annotators to provide 4-level relevance judgments of each query-passage pair. 

Table 1: The data statistics of datasets commonly used in passage ranking. FR(SR): First (Second)- stage of passage ranking, i.e., passage Retrieval (Re-ranking).
<div align=center><img width="800" height="400" src="pic/stat.png"/></div>


The download links and data formats are presented in the following table.

| Description                                           | Filename                                                                                                                | File size |                        Num Records | Format                                                         |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------:|-----------------------------------:|----------------------------------------------------------------|
| Collection                                | collection.tsv                             |    3.5 GB |                         2,303,643  | tsv: pid, passage |
| Queries     Train                          | queries.train.tsv                                 |   9.8 MB |                         258,042  | tsv: qid, query |
| Qrels Train                               | qrels.train.tsv                                 |   29 MB |                           1,613,421  | TREC qrels format |
<!-- | Qrels Dev                                 | qrels.dev.tsv                                    |    13 MB |                            781,949  | TREC qrels format |
| Queries     Dev                          | queries.dev.tsv                           |   1.9 MB |                         49,662  | tsv: qid, query |
| Qrels Retrieval                               | retrieval_qrels.tsv                                |   14 MB |                           980,060  | tsv:qid, pid | -->





