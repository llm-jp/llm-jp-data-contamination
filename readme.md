##LM Contamination Task

This repo works for llm-jp eval-tuning-wg task9: [データリークの評価](https://github.com/llm-jp/eval-tuning-wg/issues/9#top)

[Oscar Sainz, et al.](https://hitz-zentroa.github.io/lm-contamination/blog/) firstly proposed the idea that the model is contaminated if it is able to generate examples of the dataset.
However, recent works show that this method can be unreliable and subject to failure. [S. Golchin & M. Surdeanu](https://arxiv.org/pdf/2308.08493.pdf)(https://arxiv.org/pdf/2311.06233.pdf) argue that such failures can result either from the sparsity introduced by the request to reproduce the first instances of a dataset split or from the inability to bypass the safety filters set by the model provider when the model is asked to generate copyrighted content like dataset instances. 

Osainz has posted the related works on [huggingface community](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/472)

[Time Travel in LLMs: Tracing Data Contamination in Large Language Models (Golchin and Surdeanu, 2023)][reference]
[reference](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/472)

[Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation (Li 2023)][reference]
[reference](https://arxiv.org/pdf/2309.10677.pdf)

[Detecting Pretraining Data from Large Language Models (Shi et al., 2023)][reference]
[reference](https://arxiv.org/pdf/2310.16789.pdf)

[Proving Test Set Contamination in Black Box Language Models (Oren et al., 2023)][reference]
[reference](https://arxiv.org/pdf/2310.17623.pdf)

[Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models (Golchin and Surdeanu, 2023)][reference]
[reference](https://arxiv.org/pdf/2311.06233.pdf)

[Investigating Data Contamination in Modern Benchmarks for Large Language Models (Deng et al., 2023)][reference]
[reference](https://arxiv.org/pdf/2311.09783.pdf)

[Rethinking Benchmark and Contamination for Language Models with Rephrased Samples (Yang et al., 2023)][reference]
[reference](https://arxiv.org/pdf/2311.04850.pdf)


So far, this repo implementated part of [S. Golchin & M. Surdeanu](https://arxiv.org/pdf/2308.08493.pdf)(https://arxiv.org/pdf/2311.06233.pdf)'s work.

**More details please refer to [データリークの評価](https://github.com/llm-jp/eval-tuning-wg/issues/9#top). 
