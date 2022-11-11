# Offensive Language Detection in Social Media

This project was realized for the Human Language Technologies course and the purpose was to develop an offensive language detector that, given a tweet, can recognize whether it contains offensive content or not. \
The dataset used to build the models was obtained from SemEval-2019 Task 6, more details about this [task](https://sites.google.com/site/offensevalsharedtask/offenseval2019). More information about the project can be found in the [report](https://github.com/michisco/OffensEval_HLT/blob/main/report.pdf).

## Methodology
I analyzed and trained several models/methods including [Bidirectional LSTM](https://paperswithcode.com/method/bilstm), [BERT](https://arxiv.org/abs/1810.04805), RoBERTa ([Base](https://arxiv.org/abs/1907.11692) and [trained with ~58M tweets](https://github.com/cardiffnlp/tweeteval)), [DistilBERT](https://arxiv.org/abs/1910.01108), and [BERTweet](https://aclanthology.org/2020.emnlp-demos.2/).

## How to run
All the models can be run on Colab or Kaggle, it is strongly recommended to change the runtime type to GPU to speed up the notebooks' execution.

## Reference
Credit to [SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval)](https://arxiv.org/abs/1903.08983)
