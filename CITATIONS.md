# Academic & Technical Citations

This project builds upon the following datasets, models, foundational papers, and open-source libraries:

---

## 1. Datasets

```bibtex
@misc{thoughtvector2017twcs,
  author = {ThoughtVector},
  title = {Customer Support on Twitter: Over 3 Million Inquiries and Responses},
  year = {2017},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter},
  note = {Sub-sampled and filtered for brand AmazonHelp}
}
```

---

## 2. Foundational Papers & Methodology

### Retrieval-Augmented Generation (RAG)
```bibtex
@article{lewis2020retrieval,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```

### LLM-as-a-Judge
```bibtex
@article{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

### Dense Vector Similarity & Search
```bibtex
@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```

### Non-Parametric Bootstrap Confidence Intervals
```bibtex
@article{efron1979bootstrap,
  title={Bootstrap methods: another look at the jackknife},
  author={Efron, Bradley},
  journal={The Annals of Statistics},
  volume={7},
  number={1},
  pages={1--26},
  year={1979}
}
```

### Inter-Rater Reliability (Cohen's Kappa)
```bibtex
@article{cohen1968weighted,
  title={Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit},
  author={Cohen, Jacob},
  journal={Psychological Bulletin},
  volume={70},
  number={4},
  pages={213--220},
  year={1968}
}
```

---

## 3. Software Libraries & Models

- **Google Gemini Models:** `models/gemini-flash-latest`, `models/gemini-pro-latest`, `models/gemini-embedding-001` via `google-generativeai` SDK.
- **FAISS (Facebook AI Similarity Search):** `faiss-cpu` (v1.10.0+), flat inner-product dense index.
- **Scikit-Learn:** `scikit-learn` (v1.6.1+) for TF-IDF vectorization, logistic regression baselines, classification metrics, and Cohen's kappa score.
- **Pytest:** `pytest` (v9.1.1+) test execution harness.
- **Tenacity:** Retry and exponential backoff handling for API calls.
- **Tabulate & PyYAML:** Formatted terminal outputs and structured taxonomy management.
