# Multiple choice question answering with language models + injected external knowledge from ConceptNet.

For each question and answer pair, this model finds a shortest path (using Dijkstra from networkx) between concepts identified in the question and the answer candidates.
For a set of such identified paths, the most semantically relevant path is chosen by embedding the QA context and each path candidate into the same vector space and scoring them using cosine similarity.
This is done using the Sentence Transformer. The best scored path is concatenated to the input sequence before being processed as a normal MCQA classification task.

The script currently supports classification on the CommonsenseQA and OpenbookQA datasets.
