# Fine-Tuning GPT-2 for Writing Style Replication

A low-resource evaluation framework for style replication using **GPT-2 small (117M)**, with analysis through **Style Capacity**, **Cross-Topic Confusion**, and **Style Drift**.

This repository contains the implementation and experiment notebook accompanying the paper:

> **Fine-Tuning GPT-2 for Writing Style Replication: A Low-Resource Evaluation Framework Using Style Capacity, Confusion, and Drift Analysis**

---

## Overview

This project studies whether very small amounts of topic-specific training data (25, 50, 100 lines) can shift GPT-2’s output style in a measurable way.

Instead of author-level datasets, the experiments use **AG News desk categories** as style proxies:
- World
- Sports
- Business
- Science/Technology

The model is fine-tuned separately for each desk and training size, producing **12 checkpoints** (4 topics × 3 data budgets).

Generated text is scored using sentence-embedding cosine similarity in the **all-MiniLM-L6-v2** space, referred to as **StyleSim**.

---

## Research Questions

1. **Capacity**: Does style overlap improve as training examples increase from 25 → 50 → 100?
2. **Confusion**: Do 100-line models align most strongly with their own desk, or leak into other desks?
3. **Drift**: During recursive self-conditioning, how quickly does stylistic alignment degrade?

---

## Methodology Snapshot

### Model and Training
- Base model: `gpt2` (GPT-2 small, 117M)
- Fine-tuning objective: causal next-token LM loss
- Epochs: 2
- Learning rate: `5e-5`
- Batch size: 8
- Sequence length: 64 tokens (packed)
- Precision: FP16 when CUDA is available

### Decoding
- Unconditional generation from `<|endoftext|>`
- `top_k = 50`, `top_p = 0.95`, `temperature = 0.9`
- 50 new tokens per decode
- 20 decodes per checkpoint

### Scoring
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Metric: mean pairwise cosine between generated set and held-out references (**StyleSim**)
- Typical value range observed in this setup: ~0.02 to ~0.11

---

## Key Findings (from the paper)

- **Sports** shows the strongest gain with data scale, reaching **0.104 at 100 lines** (~58% relative increase from 25 lines).
- **Science/Technology** improves only marginally (flattest capacity curve).
- In confusion analysis (100-line models), **3/4 desks retain diagonal dominance**.
- In World-desk drift, StyleSim peaks at **hop 3 (~0.096)** and declines afterward, indicating style decay in longer recursive chains.

---

## Repository Structure

- `gpt2_style_emulation.ipynb` — Main experiment notebook (data preparation, fine-tuning loop, generation, evaluation, plotting)
- `README.md` — Project documentation
- `LICENSE` — Apache 2.0

---

## How to Run

1. Open the notebook in Google Colab or local Jupyter.
2. Install dependencies (see below).
3. Run cells in order to:
   - load/filter AG News,
   - train checkpoints for each desk and sample size,
   - generate outputs,
   - compute StyleSim metrics,
   - reproduce capacity/confusion/drift plots.

> The notebook is designed for low-resource experimentation and was run on a Colab T4-class GPU.

---

## Dependencies

Typical packages used:

- `torch`
- `transformers`
- `datasets`
- `sentence-transformers`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`

If needed, install with:

```bash
pip install torch transformers datasets sentence-transformers scikit-learn numpy matplotlib seaborn
```

---

## Experimental Notes

- Fixed random seed (42) was used for reproducibility.
- Model artifacts and generated outputs can be cached and reused for re-scoring without retraining.
- Exact bit-for-bit reproduction is still sensitive to hardware/runtime differences and stochastic decoding behavior.

---

## Limitations

- AG News desk labels are **topic/style proxies**, not verified authorship identities.
- StyleSim is an embedding-based comparative indicator, not a definitive authorship score.
- Unconditional generation lowers absolute cosine values because content entities rarely overlap exactly with references.
- Results are encoder-dependent (MiniLM used consistently for comparability).

---

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{kodgire2026gpt2style,
  title={Fine-Tuning GPT-2 for Writing Style Replication: A Low-Resource Evaluation Framework Using Style Capacity, Confusion, and Drift Analysis},
  author={Kodgire, Prathamesh and Dubey, Yashi and Anand, Kunal and Rajput, Rohit Singh and Dhawase, Dhammjyoti V. and Dubey, Shubh},
  year={2026},
  note={Preprint / Manuscript}
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See [`LICENSE`](./LICENSE) for details.

---

## Contact

For questions or collaboration, please use the repository Issues section or contact the authors listed in the manuscript.
