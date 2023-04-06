# T2Ranking

## Introduction
T2Ranking is a large-scale Chinese benchmark for passage ranking. The details about T2Ranking are elaborated in [this paper](https://github.com/THUIR/T2Ranking).


Passage ranking are important and challenging topics for both academics and industries in the area of Information Retrieval (IR). The goal of passage ranking is to compile a search result list ordered in terms of relevance to the query from a large passage collection. Typically, Passage ranking involves two stages: passage retrieval and passage re-ranking. 

To support the passage ranking research, various benchmark datasets are constructed. However, the commonly-used datasets for passage ranking usually focus on the English language. For non-English scenarios, such as Chinese, the existing datasets are limited in terms of data scale, fine-grained relevance annotation and false negative issues.


To address this problem, we introduce T2Ranking, a large-scale Chinese benchmark for passage ranking. T2Ranking comprises more than 300K queries and over 2M unique passages from real- world search engines. Specifically, we sample question-based search queries from user logs of the Sogou search engine, a popular search system in China. For each query, we extract the content of corresponding documents from different search engines. After model-based passage segmentation and clustering-based passage de-duplication, a large-scale passage corpus is obtained. For a given query and its corresponding passages, we hire expert annotators to provide 4-level relevance judgments of each query-passage pair. 


<div align=center><img width="600" height="200" src="pic/stat.png"/></div>
<div align=center>Table 1: The data statistics of datasets commonly used in passage ranking. FR(SR): First (Second)- stage of passage ranking, i.e., passage Retrieval (Re-ranking).</div>



Compared with existing datasets, T2Ranking dataset has the following characteristics and advantages:
* The proposed dataset focus on the Chinese search scenario, and has advantages in data scale compared with existing Chinese passage ranking datasets, which can better support the design of deep learning algorithms
* The proposed dataset has a large number of fine-grained relevance annotations, which is helpful for mining fine-grained relationship between queries and passages and constructing more accurate ranking algorithms.
* By retrieving passage results from multiple commercial search engines and providing complete annotation, we ease the false negative problem to some extent, which is beneficial to providing more accurate evaluation.
* We design multiple strategies to ensure the high quality of our dataset, such as using a passage segment model and a passage clustering model to enhance the semantic integrity and diversity of passages and employing active learning for annotation method to improve the efficiency and quality of data annotation.

## Data
The whole dataset can be downloaded through [huggingface](https://huggingface.co/datasets/THUIR/T2Ranking), and the data formats are presented in the following table.


| Description                                           | Filename                                                                                                                |                        Num Records | Format                                                         |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------:|-----------------------------------:|----------------------------------------------------------------|
| Collection                                | collection.tsv                             |                           2,303,643  | tsv: pid, passage |
| Queries     Train                          | queries.train.tsv                                 |                  258,042  | tsv: qid, query |
| Queries     Dev                          | queries.dev.tsv                                 |                 24,832  | tsv: qid, query |
| Queries     Test                          | queries.test.tsv                                 |                24,832  | tsv: qid, query |
| Qrels Train                               | qrels.train.tsv                                 |             1,613,421  | TREC qrels format |
| Qrels Dev                                 | qrels.dev.tsv                                    |      400,536    | TREC qrels format |
| Qrels Retrieval Train                               | qrels.retrieval.train.tsv                                |            744,663  | TREC qrels format |
| Qrels Retrieval Dev                               | qrels.retrieval.dev.tsv                                |            118,933  | TREC qrels format |
|  BM25 Negatives                               | train.bm25.tsv                                |            200,359,731  | tsv: qid, pid, index |
|  Hard Negatives                               | train.mined.tsv                                |            200,376,001  | tsv: qid, pid, index, score |



## Training and Evaluation
The performance reported in our paper can be reproduced by running the following command:

```bash
sh train_dual_encoder.sh
```

