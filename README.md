## Learning to Bootstrap for Entity Set Expansion
Source code for [EMNLP 2019](https://www.emnlp-ijcnlp2019.org) paper: Learning to Bootstrap for Entity Set Expansion.


### Preprocess:
- To accelarate, we pre-process and cache the dataset file first:
```shell
python preprocess.py
```

### Execution
- To execute our code, please run:
```shell
python main.py --dataset "file path to your cached dataset"
```
### Citation
Please cite the following paper if you find our code is helpful.
```bibtex
@inproceedings{yan-etal-2019-learning,
    title = "Learning to Bootstrap for Entity Set Expansion",
    author = "Yan, Lingyong  and
      Han, Xianpei  and
      Sun, Le  and
      He, Ben",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1028",
    doi = "10.18653/v1/D19-1028",
    pages = "292--301",
    abstract = "Bootstrapping for Entity Set Expansion (ESE) aims at iteratively acquiring new instances of a specific target category. Traditional bootstrapping methods often suffer from two problems: 1) delayed feedback, i.e., the pattern evaluation relies on both its direct extraction quality and extraction quality in later iterations. 2) sparse supervision, i.e., only few seed entities are used as the supervision. To address the above two problems, we propose a novel bootstrapping method combining the Monte Carlo Tree Search (MCTS) algorithm with a deep similarity network, which can efficiently estimate delayed feedback for pattern evaluation and adaptively score entities given sparse supervision signals. Experimental results confirm the effectiveness of the proposed method.",
}
```
