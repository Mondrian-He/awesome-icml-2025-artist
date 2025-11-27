# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 10 of 100

## 目录 (Table of Contents)

1. [Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It](#Hide-Seek-Transformer-Symmetries-Obscure-Sharpness-Riemannian-Geometry-Finds-It)
2. [Catoni Contextual Bandits are Robust to Heavy-tailed Rewards](#Catoni-Contextual-Bandits-are-Robust-to-Heavy-tailed-Rewards)
3. [Implicit Language Models are RNNs: Balancing Parallelization and Expressivity](#Implicit-Language-Models-are-RNNs-Balancing-Parallelization-and-Expressivity)
4. [Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models](#Raptor-Scalable-Train-Free-Embeddings-for-3D-Medical-Volumes-Leveraging-Pretrained-2D-Foundation-Models)
5. [CoPINN: Cognitive Physics-Informed Neural Networks](#CoPINN-Cognitive-Physics-Informed-Neural-Networks)
6. [ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals](#ResQ-Mixed-Precision-Quantization-of-Large-Language-Models-with-Low-Rank-Residuals)
7. [Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration](#Soft-Reasoning-Navigating-Solution-Spaces-in-Large-Language-Models-through-Controlled-Embedding-Exploration)
8. [Linearization Turns Neural Operators into Function-Valued Gaussian Processes](#Linearization-Turns-Neural-Operators-into-Function-Valued-Gaussian-Processes)
9. [An Analysis for Reasoning Bias of Language Models with Small Initialization](#An-Analysis-for-Reasoning-Bias-of-Language-Models-with-Small-Initialization)
10. [RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents against Human Experts](#RE-Bench-Evaluating-Frontier-AI-RD-Capabilities-of-Language-Model-Agents-against-Human-Experts)
11. [CVE-Bench: A Benchmark for AI Agents’ Ability to Exploit Real-World Web Application Vulnerabilities](#CVE-Bench-A-Benchmark-for-AI-Agents-Ability-to-Exploit-Real-World-Web-Application-Vulnerabilities)
12. [Great Models Think Alike and this Undermines AI Oversight](#Great-Models-Think-Alike-and-this-Undermines-AI-Oversight)
13. [Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner](#Overcoming-Multi-step-Complexity-in-Multimodal-Theory-of-Mind-Reasoning-A-Scalable-Bayesian-Planner)
14. [Doubly Robust Conformalized Survival Analysis with Right-Censored Data](#Doubly-Robust-Conformalized-Survival-Analysis-with-Right-Censored-Data)
15. [Training Deep Learning Models with Norm-Constrained LMOs](#Training-Deep-Learning-Models-with-Norm-Constrained-LMOs)
16. [Generalized Random Forests Using Fixed-Point Trees](#Generalized-Random-Forests-Using-Fixed-Point-Trees)
17. [Upweighting Easy Samples in Fine-Tuning Mitigates Forgetting](#Upweighting-Easy-Samples-in-Fine-Tuning-Mitigates-Forgetting)
18. [SAFE: Finding Sparse and Flat Minima to Improve Pruning](#SAFE-Finding-Sparse-and-Flat-Minima-to-Improve-Pruning)
19. [Emergence and Effectiveness of Task Vectors in In-Context Learning: An Encoder Decoder Perspective](#Emergence-and-Effectiveness-of-Task-Vectors-in-In-Context-Learning-An-Encoder-Decoder-Perspective)
20. [Taming Knowledge Conflicts in Language Models](#Taming-Knowledge-Conflicts-in-Language-Models)
21. [Continuous Bayesian Model Selection for Multivariate Causal Discovery](#Continuous-Bayesian-Model-Selection-for-Multivariate-Causal-Discovery)
22. [MP-Nav: Enhancing Data Poisoning Attacks against Multimodal Learning](#MP-Nav-Enhancing-Data-Poisoning-Attacks-against-Multimodal-Learning)
23. [CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging](#CAT-Merging-A-Training-Free-Approach-for-Resolving-Conflicts-in-Model-Merging)
24. [Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation](#Pretraining-Generative-Flow-Networks-with-Inexpensive-Rewards-for-Molecular-Graph-Generation)
25. [SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity](#SpikeVideoFormer-An-Efficient-Spike-Driven-Video-Transformer-with-Hamming-Attention-and-mathcalOT-Complexity)
26. [Dequantified Diffusion-Schrödinger Bridge for Density Ratio Estimation](#Dequantified-Diffusion-Schrödinger-Bridge-for-Density-Ratio-Estimation)
27. [DAMA: Data- and Model-aware Alignment of Multi-modal LLMs](#DAMA-Data-and-Model-aware-Alignment-of-Multi-modal-LLMs)
28. [Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models](#Visual-Attention-Never-Fades-Selective-Progressive-Attention-ReCalibration-for-Detailed-Image-Captioning-in-Multimodal-Large-Language-Models)
29. [A Mathematical Framework for AI-Human Integration in Work](#A-Mathematical-Framework-for-AI-Human-Integration-in-Work)
30. [XAttnMark: Learning Robust Audio Watermarking with Cross-Attention](#XAttnMark-Learning-Robust-Audio-Watermarking-with-Cross-Attention)
31. [Dimension-Free Adaptive Subgradient Methods with Frequent Directions](#Dimension-Free-Adaptive-Subgradient-Methods-with-Frequent-Directions)
32. [Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](#Dynamic-Mixture-of-Curriculum-LoRA-Experts-for-Continual-Multimodal-Instruction-Tuning)
33. [PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](#PARM-Multi-Objective-Test-Time-Alignment-via-Preference-Aware-Autoregressive-Reward-Model)

---


## Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It

### Images

![014130e170b571bd6583299897710e3aa40abc0825547c5ce2ded81a3904699a.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/014130e170b571bd6583299897710e3aa40abc0825547c5ce2ded81a3904699a.jpg)

![06206ef5c6a954a3e4b09af2ca28d9829e622cc379d7915fbfa615d5c7f08e06.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/06206ef5c6a954a3e4b09af2ca28d9829e622cc379d7915fbfa615d5c7f08e06.jpg)

![309308aa3da51b798d747a2f9e832fa46dc2ab963c8dbdd3dd8785779aacf79c.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/309308aa3da51b798d747a2f9e832fa46dc2ab963c8dbdd3dd8785779aacf79c.jpg)

![319fe7a34294dca334ebba8d5d2e32d1bd2b703ad88076ff646452bb924aea3d.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/319fe7a34294dca334ebba8d5d2e32d1bd2b703ad88076ff646452bb924aea3d.jpg)

![38cc07c319eda1010245e923e1e85886f80b6541afcecc6b5d6e013318f395d0.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/38cc07c319eda1010245e923e1e85886f80b6541afcecc6b5d6e013318f395d0.jpg)

![3ee9472e1007141b2acaab80c1e8a312ba4f0f15ac2fe906614a426e4fd1b97a.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/3ee9472e1007141b2acaab80c1e8a312ba4f0f15ac2fe906614a426e4fd1b97a.jpg)

![5776ab079dbbe83c13479e1f7448dc6bb4a6a7c0aa2efe3ebadde67bb6bebd19.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/5776ab079dbbe83c13479e1f7448dc6bb4a6a7c0aa2efe3ebadde67bb6bebd19.jpg)

![7a5361ef4a8ccc1977c71fcaa4548b7b787bd6890dc7b77dc14e536c4d656b65.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/7a5361ef4a8ccc1977c71fcaa4548b7b787bd6890dc7b77dc14e536c4d656b65.jpg)

![859c5a5a95bf35ce96d747bd0f68592af28337fd7501aed0d0a4888420ded405.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/859c5a5a95bf35ce96d747bd0f68592af28337fd7501aed0d0a4888420ded405.jpg)

![a88ff11a8d213cb5d908533f5d14680ffb92d7a8df226358cd82488102d3b280.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/a88ff11a8d213cb5d908533f5d14680ffb92d7a8df226358cd82488102d3b280.jpg)

![abf4f75d25c8ee4542385c7a2cfa40f8812d7bdba216fa1405cb39e420256dba.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/images/abf4f75d25c8ee4542385c7a2cfa40f8812d7bdba216fa1405cb39e420256dba.jpg)

### Tables

![698eee1a3a884a7728aff556f0f60c48813fb77c3bcbb0899bc43aece5b94837.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/tables/698eee1a3a884a7728aff556f0f60c48813fb77c3bcbb0899bc43aece5b94837.jpg)

![922f2d426c0aec1f3b4ef4cc23f1a44374f9d0cb8b66a4cf34c4f0bb7851be90.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/tables/922f2d426c0aec1f3b4ef4cc23f1a44374f9d0cb8b66a4cf34c4f0bb7851be90.jpg)

![b26e471d829661a3069dab81d976c279812d94c1b0f55c9265c39a85abf2a4d5.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/tables/b26e471d829661a3069dab81d976c279812d94c1b0f55c9265c39a85abf2a4d5.jpg)

![b48ffec195f24edd0e5e3913c658f04d32ec0527ba176ad587aaba48b55305cc.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/tables/b48ffec195f24edd0e5e3913c658f04d32ec0527ba176ad587aaba48b55305cc.jpg)

![c96bd4c150e55b841f23b282e4cdda58c0030a2000574c5582fcdf7ea80ffac6.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/tables/c96bd4c150e55b841f23b282e4cdda58c0030a2000574c5582fcdf7ea80ffac6.jpg)

![fbaa834d2f0957dca79fffdfd858a359b9170afe7e7204870097fa096eddf4e8.jpg](../icml_results/299_Everything%20Everywhere%20All%20at%20Once_%20LLMs%20can%20In-Context%20Learn%20Multiple%20Tasks%20in%20Superposition/tables/fbaa834d2f0957dca79fffdfd858a359b9170afe7e7204870097fa096eddf4e8.jpg)

## Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It


### Images

![12c711f8442d640661d2797970d91ee865f51d954c3310f395deff9a4e1e0aa7.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/12c711f8442d640661d2797970d91ee865f51d954c3310f395deff9a4e1e0aa7.jpg)

![43a018feb0a524358e67e40f33d1806cd7d91166d70143bb4a98a551a7e8368c.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/43a018feb0a524358e67e40f33d1806cd7d91166d70143bb4a98a551a7e8368c.jpg)

![587dbedb0292bee47a2b035c115a84082dcf31426d92e342cba244dd4202137a.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/587dbedb0292bee47a2b035c115a84082dcf31426d92e342cba244dd4202137a.jpg)

![67cbd5fcc6d8a7e681384b899c4287a53f077a5f631abf7ed0a5948bae6077c8.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/67cbd5fcc6d8a7e681384b899c4287a53f077a5f631abf7ed0a5948bae6077c8.jpg)

![7cc1439dfb801df3d9bc71b6b5d85c6d537aa09db393c7705ab990b914c50931.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/7cc1439dfb801df3d9bc71b6b5d85c6d537aa09db393c7705ab990b914c50931.jpg)

![810144bcb2b5f573d385b7f8584455969f56d678c81b3bb913f99a1503315f65.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/810144bcb2b5f573d385b7f8584455969f56d678c81b3bb913f99a1503315f65.jpg)

![b6d14856bcebc8469e4a4f34de52c319665bdbfd9eca855af7a7a36f1278063f.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/b6d14856bcebc8469e4a4f34de52c319665bdbfd9eca855af7a7a36f1278063f.jpg)

![b8956894229c36878b8533a4c8fbc2686b4b4526446d332c55d92581307ab3bc.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/b8956894229c36878b8533a4c8fbc2686b4b4526446d332c55d92581307ab3bc.jpg)

![c681636219f576cdd9eab39ecea7bf7b62e723e80081c57049e277048857674e.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/c681636219f576cdd9eab39ecea7bf7b62e723e80081c57049e277048857674e.jpg)

![ec393e52ab1ac2d76744ff6cac3a12947d2afa3d9e2197e353196ceb8c986fc8.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/images/ec393e52ab1ac2d76744ff6cac3a12947d2afa3d9e2197e353196ceb8c986fc8.jpg)

### Tables

![7a6f6f913552f6a819ca6964c8d7e6b402324edaec2529961b9d46b3d4e991bd.jpg](../icml_results/300_Hide%20%26%20Seek_%20Transformer%20Symmetries%20Obscure%20Sharpness%20%26%20Riemannian%20Geometry%20Finds%20It/tables/7a6f6f913552f6a819ca6964c8d7e6b402324edaec2529961b9d46b3d4e991bd.jpg)

## Catoni Contextual Bandits are Robust to Heavy-tailed Rewards


### Tables

![ad6924b2b57e8c07e79e506c83020bdfa88d46bd1e6f8074bce47bb18452810b.jpg](../icml_results/301_Catoni%20Contextual%20Bandits%20are%20Robust%20to%20Heavy-tailed%20Rewards/tables/ad6924b2b57e8c07e79e506c83020bdfa88d46bd1e6f8074bce47bb18452810b.jpg)

![adfb732e351cfdf6f091a321b7169c7d7cfcda87a0f777afa3f92cdc2029a6d4.jpg](../icml_results/301_Catoni%20Contextual%20Bandits%20are%20Robust%20to%20Heavy-tailed%20Rewards/tables/adfb732e351cfdf6f091a321b7169c7d7cfcda87a0f777afa3f92cdc2029a6d4.jpg)

![b838d4035049b41cbb19501f522bf21ec5df46874b98548580da5ddd38992ba0.jpg](../icml_results/301_Catoni%20Contextual%20Bandits%20are%20Robust%20to%20Heavy-tailed%20Rewards/tables/b838d4035049b41cbb19501f522bf21ec5df46874b98548580da5ddd38992ba0.jpg)

## Implicit Language Models are RNNs: Balancing Parallelization and Expressivity


### Images

![1fd87ae975451188979ce59c6785389f7ef1b6369da8c8d185037ede342c1114.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/1fd87ae975451188979ce59c6785389f7ef1b6369da8c8d185037ede342c1114.jpg)

![2a470dc5b84ea8905593edb471d6dde54821ebce7840a64dc453b07fe00fb653.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/2a470dc5b84ea8905593edb471d6dde54821ebce7840a64dc453b07fe00fb653.jpg)

![31ea09c00565d7b9397f11a0918d53cb949da27b74dd71af59847c1105e08bdd.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/31ea09c00565d7b9397f11a0918d53cb949da27b74dd71af59847c1105e08bdd.jpg)

![4ccd3449c0936d8a5fe40efdef85d1fd925b2c1feef10e13781f705799ef09bd.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/4ccd3449c0936d8a5fe40efdef85d1fd925b2c1feef10e13781f705799ef09bd.jpg)

![79c28cfe930e2c368cb2925a33a7073a14126199213d2bc3f9066f662ab3f9af.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/79c28cfe930e2c368cb2925a33a7073a14126199213d2bc3f9066f662ab3f9af.jpg)

![8194628307539080d56e689759f70c69d1c291fd81b25208ca0b48bbe356624d.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/8194628307539080d56e689759f70c69d1c291fd81b25208ca0b48bbe356624d.jpg)

![841d5f7ee11a1b01c1266612b2e0cd3d4ed2cbaf49c389ee3af1f443a0110582.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/841d5f7ee11a1b01c1266612b2e0cd3d4ed2cbaf49c389ee3af1f443a0110582.jpg)

![abedad1115db99c62f1da30fa62c3eeb9504b0d09c9db15b02340169ebb271ee.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/abedad1115db99c62f1da30fa62c3eeb9504b0d09c9db15b02340169ebb271ee.jpg)

![b93604bf2846ba53bef3659c676d25e89e5f364b1c62913aa3d37d923d8cf507.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/b93604bf2846ba53bef3659c676d25e89e5f364b1c62913aa3d37d923d8cf507.jpg)

![c10050b8ce0eee16e85e86cb2e37200e649c9eedfd2be85c71ee2b6e71c6b822.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/c10050b8ce0eee16e85e86cb2e37200e649c9eedfd2be85c71ee2b6e71c6b822.jpg)

![c1704f9e041341c7aab664a5a06dd7407b6ff3739a0581941f0c103eca0d166e.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/c1704f9e041341c7aab664a5a06dd7407b6ff3739a0581941f0c103eca0d166e.jpg)

![e42e9d7cb2859d2a4db1944facc6c8718e727d18e2d339c83c5c11a8dcb73231.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/e42e9d7cb2859d2a4db1944facc6c8718e727d18e2d339c83c5c11a8dcb73231.jpg)

![ec3c962e24e95292dbafd1233c522d98797e346e3191c07c51903def3896a5f4.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/ec3c962e24e95292dbafd1233c522d98797e346e3191c07c51903def3896a5f4.jpg)

![fd4b0231c8fbff38667a11039e6b18cf8b87b0fd51c36f017dba84870ce630a5.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/images/fd4b0231c8fbff38667a11039e6b18cf8b87b0fd51c36f017dba84870ce630a5.jpg)

### Tables

![01c4d304f70da6dfb5565f3b97e71353259df07ff297124071b1813a7a43aeeb.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/tables/01c4d304f70da6dfb5565f3b97e71353259df07ff297124071b1813a7a43aeeb.jpg)

![2cdb0a909e0149356eff485b736ef273690bb0b12770a2f6e606dc2b055a056b.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/tables/2cdb0a909e0149356eff485b736ef273690bb0b12770a2f6e606dc2b055a056b.jpg)

![2e450d5c5d92e21415194875f5a3b7a4637d098bd192bbe2f822101896bc6070.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/tables/2e450d5c5d92e21415194875f5a3b7a4637d098bd192bbe2f822101896bc6070.jpg)

![75ec6bda5b762bd3ee09290fb231b0613ac22151afc673031424f7350b97853d.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/tables/75ec6bda5b762bd3ee09290fb231b0613ac22151afc673031424f7350b97853d.jpg)

![7b25104af7a22fd8de5d54d6395c0ca9206c53b82cba8e1523f38560a8d7f596.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/tables/7b25104af7a22fd8de5d54d6395c0ca9206c53b82cba8e1523f38560a8d7f596.jpg)

![fe002a41901c17cd58a65e5a0e5339eb522fc944cf7686d1cf14d5fa629c948f.jpg](../icml_results/302_Implicit%20Language%20Models%20are%20RNNs_%20Balancing%20Parallelization%20and%20Expressivity/tables/fe002a41901c17cd58a65e5a0e5339eb522fc944cf7686d1cf14d5fa629c948f.jpg)

## Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models


### Images

![0bde28b49f1028ad76b86bd70a092e82a8c820c3fa9f85909cd6afe7a4bc0bbf.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/0bde28b49f1028ad76b86bd70a092e82a8c820c3fa9f85909cd6afe7a4bc0bbf.jpg)

![29e8f77b634013f288d9ef154fb11ff949fd2aa7e844d38ce04389d261faf9c6.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/29e8f77b634013f288d9ef154fb11ff949fd2aa7e844d38ce04389d261faf9c6.jpg)

![5844f13f399c951977a191965d7d58ab3fb513fc3f170b9cdb0c4eaf02f8a01e.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/5844f13f399c951977a191965d7d58ab3fb513fc3f170b9cdb0c4eaf02f8a01e.jpg)

![609f729e9612c29f79b6826d766f5f9e8e53d5d3d0fb08304bcb2d19be9bdace.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/609f729e9612c29f79b6826d766f5f9e8e53d5d3d0fb08304bcb2d19be9bdace.jpg)

![638297d32f244702df2c106d5aaae0a9cda99567d7d77a0e24817a7a0453e59a.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/638297d32f244702df2c106d5aaae0a9cda99567d7d77a0e24817a7a0453e59a.jpg)

![712b823d4c32c6559903ba92447b0bac6e6e1a6f5b1c7fda456abd97dc62c2eb.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/712b823d4c32c6559903ba92447b0bac6e6e1a6f5b1c7fda456abd97dc62c2eb.jpg)

![72550a835a97bad259ab4fc26c552687ac9552d094672266d11185cd71746fa9.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/72550a835a97bad259ab4fc26c552687ac9552d094672266d11185cd71746fa9.jpg)

![d40aa4efb4ad7312c8cd89f778c49daafcffa9cee59976400680b7c6a3e4e114.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/d40aa4efb4ad7312c8cd89f778c49daafcffa9cee59976400680b7c6a3e4e114.jpg)

![e7009748b158e5be53d774bb5dcb2c9b42b543aafe3345c9cf58716a92ba41f6.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/e7009748b158e5be53d774bb5dcb2c9b42b543aafe3345c9cf58716a92ba41f6.jpg)

![f15f94ef0396216c7256fb268bbf044cfcf5cac9e1efcfa0a76f133c152ac860.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/images/f15f94ef0396216c7256fb268bbf044cfcf5cac9e1efcfa0a76f133c152ac860.jpg)

### Tables

![014883b9677a668d9ece295a027866b9767b7b278c84cae283ab019d1b82113c.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/014883b9677a668d9ece295a027866b9767b7b278c84cae283ab019d1b82113c.jpg)

![0e77aea71d5e1f8bcee728ea1a897de5dde08755edd3c696530291a831be19a9.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/0e77aea71d5e1f8bcee728ea1a897de5dde08755edd3c696530291a831be19a9.jpg)

![1443e83ab9497ea6c63b6b1eb6025dd67dc46e722be9b197d16398f499d742d0.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/1443e83ab9497ea6c63b6b1eb6025dd67dc46e722be9b197d16398f499d742d0.jpg)

![211c98a9c9a8037a7d333e540e591f910a4b5abb7caab547db6a12bbd4e4f1fc.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/211c98a9c9a8037a7d333e540e591f910a4b5abb7caab547db6a12bbd4e4f1fc.jpg)

![21713d0409d1912d6b4944729161119faf7d75ef119739c0a8baf04277a2ee28.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/21713d0409d1912d6b4944729161119faf7d75ef119739c0a8baf04277a2ee28.jpg)

![2ab387a0fd7560354af233c1695911575c8f43d00264d4937b28ee3afc70f51e.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/2ab387a0fd7560354af233c1695911575c8f43d00264d4937b28ee3afc70f51e.jpg)

![2d5b6c0e110ee92a0f530fc7797b8047f046809a672cd7ed852429f4d709586c.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/2d5b6c0e110ee92a0f530fc7797b8047f046809a672cd7ed852429f4d709586c.jpg)

![5c62015462e5807418fbdfca1f27bf6d188e7e6b67b5c67ef73a32dfdc1b3f6d.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/5c62015462e5807418fbdfca1f27bf6d188e7e6b67b5c67ef73a32dfdc1b3f6d.jpg)

![5d0f9f664fffe9fcee480acaf54ba252435f179051450beef9eeb98a46c76530.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/5d0f9f664fffe9fcee480acaf54ba252435f179051450beef9eeb98a46c76530.jpg)

![78c2106c57b4b049cfd37a6653fc52da2f4ee0816b4c4522ddee68b6463cd19b.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/78c2106c57b4b049cfd37a6653fc52da2f4ee0816b4c4522ddee68b6463cd19b.jpg)

![825177ec5797ba813481d2a131dc33799843aa4dda313b18b3146a9ad2d8290e.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/825177ec5797ba813481d2a131dc33799843aa4dda313b18b3146a9ad2d8290e.jpg)

![9be4c70e9607d12faa4ddb0108a88c6590e24beafa64b5c7cf876050e756a6da.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/9be4c70e9607d12faa4ddb0108a88c6590e24beafa64b5c7cf876050e756a6da.jpg)

![bcfa41cfe90165ab2b39a605b603b5578e00042d14690e658d813192b37e6623.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/bcfa41cfe90165ab2b39a605b603b5578e00042d14690e658d813192b37e6623.jpg)

![d03383e788c94e32fea17aa8c71f227d14393cfefd43719e52c900a869752a5e.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/d03383e788c94e32fea17aa8c71f227d14393cfefd43719e52c900a869752a5e.jpg)

![d9c17c522a938b13f1ccd052888a06b114b6be59cba5ccdf372e7eaef49edb53.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/d9c17c522a938b13f1ccd052888a06b114b6be59cba5ccdf372e7eaef49edb53.jpg)

![f26b99ea48b43a7b2ea62309af59a94a3a05d33d0305abf9d389c59b31e0d20b.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/f26b99ea48b43a7b2ea62309af59a94a3a05d33d0305abf9d389c59b31e0d20b.jpg)

![ff83ee22f5e7d8c909896adcfadef4b8e931e477e1f7b6c37979ace2b3b19eae.jpg](../icml_results/303_Raptor_%20Scalable%20Train-Free%20Embeddings%20for%203D%20Medical%20Volumes%20Leveraging%20Pretrained%202D%20Foundation%20Mo/tables/ff83ee22f5e7d8c909896adcfadef4b8e931e477e1f7b6c37979ace2b3b19eae.jpg)

## CoPINN: Cognitive Physics-Informed Neural Networks


### Images

![03b6f3ccc2dd9a0da1b79148223e4d3aed578b9cacc65674e539ba42bc6eed12.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/03b6f3ccc2dd9a0da1b79148223e4d3aed578b9cacc65674e539ba42bc6eed12.jpg)

![489423e3cdf4a8800f9d14b260d64ee0cb71e68cde0806d47053699652105ec9.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/489423e3cdf4a8800f9d14b260d64ee0cb71e68cde0806d47053699652105ec9.jpg)

![50524051e97c3f921ea22097f2837d157b5a7e5fbdd5a35d9fb3d1ed5dc0bccb.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/50524051e97c3f921ea22097f2837d157b5a7e5fbdd5a35d9fb3d1ed5dc0bccb.jpg)

![5ac30b12032b9ca9f98de5066d2722616d2af5acdc09f58a47c1e1b0c36ef2b0.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/5ac30b12032b9ca9f98de5066d2722616d2af5acdc09f58a47c1e1b0c36ef2b0.jpg)

![6fa03db37f0c86a96c4377f94a8f1616ab7c46f9ebf9eb6bc016bb35c2542a07.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/6fa03db37f0c86a96c4377f94a8f1616ab7c46f9ebf9eb6bc016bb35c2542a07.jpg)

![7f097664795259a2461abc9efb174264995e56789cd55275905f82f121eb7fd7.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/7f097664795259a2461abc9efb174264995e56789cd55275905f82f121eb7fd7.jpg)

![8bf44b1515b50a5a581dd3eb7fd43574fb71cad1771946e99f8a110c9276b6fc.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/8bf44b1515b50a5a581dd3eb7fd43574fb71cad1771946e99f8a110c9276b6fc.jpg)

![b63675fa499ddccb4821ea6f4f618004e60710ebbf0a2f0b32a836616525d13b.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/b63675fa499ddccb4821ea6f4f618004e60710ebbf0a2f0b32a836616525d13b.jpg)

![d61d2e653915c1ddf4d3bcf44cb138fad22ce2fe20620073b93b48f43546a86b.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/d61d2e653915c1ddf4d3bcf44cb138fad22ce2fe20620073b93b48f43546a86b.jpg)

![e0dbf6ab5b4e84222b28b4cf3082ec657d59c150f681525f31df9a2b88665dc3.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/images/e0dbf6ab5b4e84222b28b4cf3082ec657d59c150f681525f31df9a2b88665dc3.jpg)

### Tables

![2cd73f8cce35013804c88d92a03c43c39a81a56bc7c7a0dcd491643bcc26d7fa.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/tables/2cd73f8cce35013804c88d92a03c43c39a81a56bc7c7a0dcd491643bcc26d7fa.jpg)

![41e43f3725a4ae8bc942ad22ea7ca59f11579f8823b4d7eccdcddad6ed0c9c6d.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/tables/41e43f3725a4ae8bc942ad22ea7ca59f11579f8823b4d7eccdcddad6ed0c9c6d.jpg)

![5ae7c8155682834ab03a531e777dbea054c1c873af0e36905248d73e7a13f437.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/tables/5ae7c8155682834ab03a531e777dbea054c1c873af0e36905248d73e7a13f437.jpg)

![a8ac5242d92dd525e567f6a98eda914f35186104c7186c821ee54872e6a42f96.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/tables/a8ac5242d92dd525e567f6a98eda914f35186104c7186c821ee54872e6a42f96.jpg)

![d0acb5eb26f7ab7c28044452ca0339cf0bdbaff13144139fc48ab6633ee14ace.jpg](../icml_results/304_CoPINN_%20Cognitive%20Physics-Informed%20Neural%20Networks/tables/d0acb5eb26f7ab7c28044452ca0339cf0bdbaff13144139fc48ab6633ee14ace.jpg)

## ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals


### Images

![389706894affd852f29b7b85a3a6452cab01139215299627526edef98ed4dee5.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/images/389706894affd852f29b7b85a3a6452cab01139215299627526edef98ed4dee5.jpg)

![53267a1d147adf4a095b3e31a1dbb016ebc00347bf035ae68a2fedad3cfd7d54.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/images/53267a1d147adf4a095b3e31a1dbb016ebc00347bf035ae68a2fedad3cfd7d54.jpg)

![6f8b0d656ba03aaef0ef97e99798babc07cebb2f21f5e010b549c52befc22118.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/images/6f8b0d656ba03aaef0ef97e99798babc07cebb2f21f5e010b549c52befc22118.jpg)

![b944e9c1056338c447d67973e28d6fa28901c39e0a6a4b2fc3147c15fd13fb79.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/images/b944e9c1056338c447d67973e28d6fa28901c39e0a6a4b2fc3147c15fd13fb79.jpg)

![d609b2d7d5cf2029adcd13050d14c6baf23c7fe76ade6507853e4a09be310672.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/images/d609b2d7d5cf2029adcd13050d14c6baf23c7fe76ade6507853e4a09be310672.jpg)

![f03d8247a91b4623e1a8681fac0bb811014b3fbadfbcf9de3c769718b2805c77.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/images/f03d8247a91b4623e1a8681fac0bb811014b3fbadfbcf9de3c769718b2805c77.jpg)

![fb65572080de0e9a50ca6851b968bb6dc23046c55e02e80309eb1d1c97a7d795.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/images/fb65572080de0e9a50ca6851b968bb6dc23046c55e02e80309eb1d1c97a7d795.jpg)

### Tables

![28e0cb6e7e6bbc6d799a1f7e36151776db7d1780a198fbfa6f5757c671355f6a.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/28e0cb6e7e6bbc6d799a1f7e36151776db7d1780a198fbfa6f5757c671355f6a.jpg)

![30ff5ba6da5a5b633da83967446c22b511fe42f2ee840c675f9b49d798f8e734.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/30ff5ba6da5a5b633da83967446c22b511fe42f2ee840c675f9b49d798f8e734.jpg)

![340f320df0eb56b8760dcfa35ce0a4d1ed7ee3edf64d950d1d1e08c5f6aad080.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/340f320df0eb56b8760dcfa35ce0a4d1ed7ee3edf64d950d1d1e08c5f6aad080.jpg)

![359701f6ff9550e86bc190260e9f1b23aaa35aaa2b2cb0f482024ed42fa96398.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/359701f6ff9550e86bc190260e9f1b23aaa35aaa2b2cb0f482024ed42fa96398.jpg)

![512db45c4583a81b9b6df3302250645f05b47c361cb680a447e405f05631a2f4.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/512db45c4583a81b9b6df3302250645f05b47c361cb680a447e405f05631a2f4.jpg)

![524f5ae26222b1bd0bb0c0eebce0c774bffcf926391ed96067bea64435f89c7a.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/524f5ae26222b1bd0bb0c0eebce0c774bffcf926391ed96067bea64435f89c7a.jpg)

![56a07e65e53e66e23a1ae01039a88eb15489f074834047b034a3bc1b18d7c2e1.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/56a07e65e53e66e23a1ae01039a88eb15489f074834047b034a3bc1b18d7c2e1.jpg)

![5a18de5f0c2caa44827d0143499d31a454f88cf244a3868f1963f74a149d3678.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/5a18de5f0c2caa44827d0143499d31a454f88cf244a3868f1963f74a149d3678.jpg)

![624a0ae1d3b451646f70317b067748a5f5ec6779b4c8cce7d3761a116e7379d1.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/624a0ae1d3b451646f70317b067748a5f5ec6779b4c8cce7d3761a116e7379d1.jpg)

![64fb9d15607ba69c95ff132b65e6c25ec41f1d8119a673f30f330de67a4effbb.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/64fb9d15607ba69c95ff132b65e6c25ec41f1d8119a673f30f330de67a4effbb.jpg)

![758b64ce5f723d9d60a098cea302488273c809f97670f99f536ecce88fe6f9cb.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/758b64ce5f723d9d60a098cea302488273c809f97670f99f536ecce88fe6f9cb.jpg)

![831b564bb3c8e41b632bb041b1d703b1678d6370113bed48d01ffe9e4fa8419a.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/831b564bb3c8e41b632bb041b1d703b1678d6370113bed48d01ffe9e4fa8419a.jpg)

![88eedc76f5a78297b7782397b72435818917e1a1e418ee742ad037607f748d70.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/88eedc76f5a78297b7782397b72435818917e1a1e418ee742ad037607f748d70.jpg)

![938617420b7176638ae33bc2956f604cf210c22b88dc6c0bfc1098e6fcd44562.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/938617420b7176638ae33bc2956f604cf210c22b88dc6c0bfc1098e6fcd44562.jpg)

![af589f3acbd39a6f1e95eba88dc7ca81893ad0a6fda71140c9e5a7b55ca94775.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/af589f3acbd39a6f1e95eba88dc7ca81893ad0a6fda71140c9e5a7b55ca94775.jpg)

![b934d91ac939ed195855d933f76b2133a1e7232e7965d32c5fd46c592b4b9e5d.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/b934d91ac939ed195855d933f76b2133a1e7232e7965d32c5fd46c592b4b9e5d.jpg)

![bb938265767eb3d87b8b044696c64ee84d836e25e7df1edde3860cf150bc0179.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/bb938265767eb3d87b8b044696c64ee84d836e25e7df1edde3860cf150bc0179.jpg)

![e3e2b9e16a8a5ee9f7b92e4603bae98f1192ea2488ed5bf3df1abc1c03da9989.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/e3e2b9e16a8a5ee9f7b92e4603bae98f1192ea2488ed5bf3df1abc1c03da9989.jpg)

![e51c2ac263b780f0742b1d9527901e3f0d7dbd478a346db4aade6f48ff264278.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/e51c2ac263b780f0742b1d9527901e3f0d7dbd478a346db4aade6f48ff264278.jpg)

![f8fd719acef552852e66fdf528f1eb93d1e367133237dc59be7ccf86ae1af281.jpg](../icml_results/305_ResQ_%20Mixed-Precision%20Quantization%20of%20Large%20Language%20Models%20with%20Low-Rank%20Residuals/tables/f8fd719acef552852e66fdf528f1eb93d1e367133237dc59be7ccf86ae1af281.jpg)

## Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration


### Images

![2ec3bc0a0026b840947912657c1b00ff24d54ba2ddf440b9b2055fcace474560.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/2ec3bc0a0026b840947912657c1b00ff24d54ba2ddf440b9b2055fcace474560.jpg)

![4d2ab9e0e35eb54e74a3199ab134407ecb5e39f6cb13af3419d9086d53e76c71.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/4d2ab9e0e35eb54e74a3199ab134407ecb5e39f6cb13af3419d9086d53e76c71.jpg)

![53abcb376a7238873efa778d983b10ea2b8bba7e25c8db527b9f818ac3652a83.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/53abcb376a7238873efa778d983b10ea2b8bba7e25c8db527b9f818ac3652a83.jpg)

![5776d97c098facbe119840d6414c9bcad69671e309b3c792e0b8c888aeeecd44.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/5776d97c098facbe119840d6414c9bcad69671e309b3c792e0b8c888aeeecd44.jpg)

![7eefa041d63415e50c82cff2454907d045208441105f687d7f088ec55eab7921.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/7eefa041d63415e50c82cff2454907d045208441105f687d7f088ec55eab7921.jpg)

![8c5b227b69ede2c4e7f08deb8064a9bddd9e2d085fad2e74bf30747fbe750e43.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/8c5b227b69ede2c4e7f08deb8064a9bddd9e2d085fad2e74bf30747fbe750e43.jpg)

![d03e45f63f4cf9ef51418bd4465cf7a92da500778566e69c427a061cfb2acc5c.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/d03e45f63f4cf9ef51418bd4465cf7a92da500778566e69c427a061cfb2acc5c.jpg)

![e181b786a017fc9a839f32753c43d3062e43f7e66291136b048c6ade2227d452.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/images/e181b786a017fc9a839f32753c43d3062e43f7e66291136b048c6ade2227d452.jpg)

### Tables

![0610891dbeb5296d83824fb876d0737a2c4d65b80683a9d747e893877b2e9f88.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/0610891dbeb5296d83824fb876d0737a2c4d65b80683a9d747e893877b2e9f88.jpg)

![207015577c73a3a1bf5c19bf0d029a0e7d8ca2a1e465e2d5170c6c82d7af81cb.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/207015577c73a3a1bf5c19bf0d029a0e7d8ca2a1e465e2d5170c6c82d7af81cb.jpg)

![2c22733bdcb795446bd1f01f760f2016e7b5e31b6758ce443de75ef3e23794cb.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/2c22733bdcb795446bd1f01f760f2016e7b5e31b6758ce443de75ef3e23794cb.jpg)

![44e57f6690f752fc195dc39c3d8341f1f8975e8357839f8fdbd3a1caa372b6e7.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/44e57f6690f752fc195dc39c3d8341f1f8975e8357839f8fdbd3a1caa372b6e7.jpg)

![581f7f84c9bdba745d738386c01c069297af8db145eebfc38e05dc408cac03a5.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/581f7f84c9bdba745d738386c01c069297af8db145eebfc38e05dc408cac03a5.jpg)

![62272237b03b36e248882186af8daec9abbef4cd763b7e69e639db3825968986.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/62272237b03b36e248882186af8daec9abbef4cd763b7e69e639db3825968986.jpg)

![83a38713db466ab99a4edc8cc5943b06763255d3462128db4e226bfd9f902987.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/83a38713db466ab99a4edc8cc5943b06763255d3462128db4e226bfd9f902987.jpg)

![9b76d65a9ab850214d0db2a79b8e0b2cb4048ab17b4b3455095263aae880be70.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/9b76d65a9ab850214d0db2a79b8e0b2cb4048ab17b4b3455095263aae880be70.jpg)

![c4fff4a6186263f506cd4ccd5a3decc16bf789d223174a3194d986932076e941.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/c4fff4a6186263f506cd4ccd5a3decc16bf789d223174a3194d986932076e941.jpg)

![c9f313c1637a5bd9506bc24b9e069e61a0815916872dfcbcdcdc268e2c0e15dc.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/c9f313c1637a5bd9506bc24b9e069e61a0815916872dfcbcdcdc268e2c0e15dc.jpg)

![e44a70ff36607feaee1beabcfd6f054f6d5598b75e551092922f0c27e1f7fa8b.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/e44a70ff36607feaee1beabcfd6f054f6d5598b75e551092922f0c27e1f7fa8b.jpg)

![eaf17195a128dad740f0c0ad9c790dd23234c352e20ff21f37eb3358f576458f.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/eaf17195a128dad740f0c0ad9c790dd23234c352e20ff21f37eb3358f576458f.jpg)

![ed11614060463c8fb1942e19002193aa24cdfcb72209138336081a6198bbf0a0.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/ed11614060463c8fb1942e19002193aa24cdfcb72209138336081a6198bbf0a0.jpg)

![ed632bf2880529989c96b5ac39949bf6c8f8cac0526b788e89b80ac366d1721d.jpg](../icml_results/306_Soft%20Reasoning_%20Navigating%20Solution%20Spaces%20in%20Large%20Language%20Models%20through%20Controlled%20Embedding%20Exp/tables/ed632bf2880529989c96b5ac39949bf6c8f8cac0526b788e89b80ac366d1721d.jpg)

## Linearization Turns Neural Operators into Function-Valued Gaussian Processes


### Images

![10159df4500964920b492e7df05ebc677eaeba95e237fc74b84901b1b6b35e5a.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/images/10159df4500964920b492e7df05ebc677eaeba95e237fc74b84901b1b6b35e5a.jpg)

![2483a5a816f4ab910c18371d542ab26717ce99f7f35184541a42ba3b5099a48a.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/images/2483a5a816f4ab910c18371d542ab26717ce99f7f35184541a42ba3b5099a48a.jpg)

![73072d296eb30567278a8685e78f83aa0da1d74e549c298b16b3f7e801e19084.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/images/73072d296eb30567278a8685e78f83aa0da1d74e549c298b16b3f7e801e19084.jpg)

![883d6c6b7ae53b4afd0425f7ec5d3688b2f2cb754e26b1d16c34955d196347ed.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/images/883d6c6b7ae53b4afd0425f7ec5d3688b2f2cb754e26b1d16c34955d196347ed.jpg)

![89376ef0bfa1b6a265907ba07d6df2a9d07fe07e9950fece2ce8ded4ead8fffc.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/images/89376ef0bfa1b6a265907ba07d6df2a9d07fe07e9950fece2ce8ded4ead8fffc.jpg)

![c0f1012044e80e059443fce9733601d0d5137a1a5a2d91ac4e906f67557853ec.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/images/c0f1012044e80e059443fce9733601d0d5137a1a5a2d91ac4e906f67557853ec.jpg)

![eba65764a9a82a0899ffe5d81107cb05fd18e83f37cdf315726675aeba3eb142.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/images/eba65764a9a82a0899ffe5d81107cb05fd18e83f37cdf315726675aeba3eb142.jpg)

### Tables

![492783e1b9277e944fadaa2da6b9a767d85bebfac364e2b48e2d99c5f8c231f2.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/492783e1b9277e944fadaa2da6b9a767d85bebfac364e2b48e2d99c5f8c231f2.jpg)

![5046f5b0f334a41fa7e9f20a07593e50c718b04812f105c99dffb2378bb36ab6.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/5046f5b0f334a41fa7e9f20a07593e50c718b04812f105c99dffb2378bb36ab6.jpg)

![509eafd748d71435ce3bac9a745c0bb81edf14b3351bbf35bf863f9755f32d60.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/509eafd748d71435ce3bac9a745c0bb81edf14b3351bbf35bf863f9755f32d60.jpg)

![530b19cef4874a871f2d3a2fb3ed3b452e43bec0f4cc58f4cab4fcb23b7e732c.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/530b19cef4874a871f2d3a2fb3ed3b452e43bec0f4cc58f4cab4fcb23b7e732c.jpg)

![614b76acd2870aeab5c69849bf8c3f6077d1ff41c9d4453e1b9b455d26bfd748.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/614b76acd2870aeab5c69849bf8c3f6077d1ff41c9d4453e1b9b455d26bfd748.jpg)

![695cfa0006fe9d8071d88e4e447d6243e89e6b6c9b3b20badd1a229f08b81fbd.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/695cfa0006fe9d8071d88e4e447d6243e89e6b6c9b3b20badd1a229f08b81fbd.jpg)

![92c5f248fff578139e0479bc79fec3286a40842564f9530b41461f92748ab4a8.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/92c5f248fff578139e0479bc79fec3286a40842564f9530b41461f92748ab4a8.jpg)

![c196936e7eaee303475d41ef68d00de231215da471ae64dc90e5b99926017ad0.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/c196936e7eaee303475d41ef68d00de231215da471ae64dc90e5b99926017ad0.jpg)

![da23bc5563ff40ca1a235798235724298d2a04c60af284b8d42d6719b140ae47.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/da23bc5563ff40ca1a235798235724298d2a04c60af284b8d42d6719b140ae47.jpg)

![ee72cef0945f88aee901509c8d2415d3c767eaf30e0f2e92c13366108270a87f.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/ee72cef0945f88aee901509c8d2415d3c767eaf30e0f2e92c13366108270a87f.jpg)

![ef3926cd7d865740899be357da15046d9fc523adb97d5751c53d33a9262e8841.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/ef3926cd7d865740899be357da15046d9fc523adb97d5751c53d33a9262e8841.jpg)

![f2c3c31884b240b0ad45e1ca894f7c121b0c813de5a922ea1b011cd2bee3ea59.jpg](../icml_results/307_Linearization%20Turns%20Neural%20Operators%20into%20Function-Valued%20Gaussian%20Processes/tables/f2c3c31884b240b0ad45e1ca894f7c121b0c813de5a922ea1b011cd2bee3ea59.jpg)

## An Analysis for Reasoning Bias of Language Models with Small Initialization


### Images

![0d340ed688503707a55ca542d8d6e476145d14ff30c9ba576045eaf5ab0cb5db.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/0d340ed688503707a55ca542d8d6e476145d14ff30c9ba576045eaf5ab0cb5db.jpg)

![1c67b84d3073ea102db6c50b6af75f044926a1bcf78c537439722adc424b4030.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/1c67b84d3073ea102db6c50b6af75f044926a1bcf78c537439722adc424b4030.jpg)

![30835cfd27d1f678d074ed442ce3f201d6c0c58c28ad3f2bf85ffcba7085eb8d.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/30835cfd27d1f678d074ed442ce3f201d6c0c58c28ad3f2bf85ffcba7085eb8d.jpg)

![3b0a34d80f0b3b2f0076a8bd9bfd36eba157af59b5506951365c8714b04bda1c.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/3b0a34d80f0b3b2f0076a8bd9bfd36eba157af59b5506951365c8714b04bda1c.jpg)

![5f0ac82f563b3ab69af02ea24a810aebb50b0e2cc6a3c80b17c1b525276ac920.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/5f0ac82f563b3ab69af02ea24a810aebb50b0e2cc6a3c80b17c1b525276ac920.jpg)

![782debb2588208ab859f6fb88bf3e69c80bfc56def751a30fb042812c5b67358.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/782debb2588208ab859f6fb88bf3e69c80bfc56def751a30fb042812c5b67358.jpg)

![802f214204c3279571f13407c4f5577cded419c71665591eaf129f5549cccca2.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/802f214204c3279571f13407c4f5577cded419c71665591eaf129f5549cccca2.jpg)

![8d4dec056c41ba055ca7b6ad5ae723bbc9b827565f6f9e0ddb12c9f06ba89300.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/8d4dec056c41ba055ca7b6ad5ae723bbc9b827565f6f9e0ddb12c9f06ba89300.jpg)

![8f990f67d4fd67400611a2e26b8e3013413987f343f8954c6f9a7b5528757d89.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/8f990f67d4fd67400611a2e26b8e3013413987f343f8954c6f9a7b5528757d89.jpg)

![9574c9b3744b256458c301be5b12f4054a92ce0c0247b6effa7db4424099a675.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/9574c9b3744b256458c301be5b12f4054a92ce0c0247b6effa7db4424099a675.jpg)

![96a6420251c5da8da8ad7ba712dd09baa2657648638df91198938f8df98b1e84.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/96a6420251c5da8da8ad7ba712dd09baa2657648638df91198938f8df98b1e84.jpg)

![9d0d09f245f09324d8912e6c301954b9df6a49c710fc0baff6a2154f5db21c24.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/9d0d09f245f09324d8912e6c301954b9df6a49c710fc0baff6a2154f5db21c24.jpg)

![c01720d983d54d94ae38880b12488ca482572743ae1f60fd751f16ed86193d3e.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/c01720d983d54d94ae38880b12488ca482572743ae1f60fd751f16ed86193d3e.jpg)

![c1303814955faaf4f0294543fc18e4bc2d53d79940a2597cea54e98eb5d48e49.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/c1303814955faaf4f0294543fc18e4bc2d53d79940a2597cea54e98eb5d48e49.jpg)

![d7a7ae17f622832a793de4307a89e0a43192e28ad4a4fd92a10255b1c330a3a2.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/d7a7ae17f622832a793de4307a89e0a43192e28ad4a4fd92a10255b1c330a3a2.jpg)

![e65191f5700a23f20371c6a7a2e3b3e604b8c5a0918c880612de563f6ed53f66.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/e65191f5700a23f20371c6a7a2e3b3e604b8c5a0918c880612de563f6ed53f66.jpg)

![eb6ec93020d5f307f8398d5745d659a79c2158cb14adfe04ac35cffb11b5f57f.jpg](../icml_results/308_An%20Analysis%20for%20Reasoning%20Bias%20of%20Language%20Models%20with%20Small%20Initialization/images/eb6ec93020d5f307f8398d5745d659a79c2158cb14adfe04ac35cffb11b5f57f.jpg)

## RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents against Human Experts


### Images

![01bb92762dfa5d9bfab827d6ee193033acd0876c1a3b9127f009a52bf4e65ed2.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/01bb92762dfa5d9bfab827d6ee193033acd0876c1a3b9127f009a52bf4e65ed2.jpg)

![06c4f500ba545488cabb2f7b648d14b4a70fc32c8347801560939c197fe3321d.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/06c4f500ba545488cabb2f7b648d14b4a70fc32c8347801560939c197fe3321d.jpg)

![2872db325040615048bfd998d5a6b688fab9c8b45dbaed1ddef49ded6c287378.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/2872db325040615048bfd998d5a6b688fab9c8b45dbaed1ddef49ded6c287378.jpg)

![2f540ef3266a2d845c9a09be21072b28f3516b6230401d3fff622787eeecba85.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/2f540ef3266a2d845c9a09be21072b28f3516b6230401d3fff622787eeecba85.jpg)

![326939eba079061e2a560da7dee91899395785f5717dc1dfc2ca804f6b53c36c.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/326939eba079061e2a560da7dee91899395785f5717dc1dfc2ca804f6b53c36c.jpg)

![32b88948bdd0a2c6246b1898c0a38ef46498dda6a9288420e4152b2c52848166.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/32b88948bdd0a2c6246b1898c0a38ef46498dda6a9288420e4152b2c52848166.jpg)

![3a004ddf1d7ed5e90bc40178c61edff7eb9af20fe96a14ff63154e1566433a31.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/3a004ddf1d7ed5e90bc40178c61edff7eb9af20fe96a14ff63154e1566433a31.jpg)

![49dac80492d54d544bb04af27172f615ba5894d197381c43723cfe2366df3280.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/49dac80492d54d544bb04af27172f615ba5894d197381c43723cfe2366df3280.jpg)

![4fe549a5a41102ebfe11f00e178c825fb46738bf83a27f6a49cc91b076d72ded.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/4fe549a5a41102ebfe11f00e178c825fb46738bf83a27f6a49cc91b076d72ded.jpg)

![54f886c38a82c99c84d376b8b86e2c532c24475f145562bea6b18cfba0a6a5c1.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/54f886c38a82c99c84d376b8b86e2c532c24475f145562bea6b18cfba0a6a5c1.jpg)

![550b9e6c8952dfe221198995442b4181adc0dcf761781c737194afa9713fc3e8.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/550b9e6c8952dfe221198995442b4181adc0dcf761781c737194afa9713fc3e8.jpg)

![5851d7335b20b8f28543edaa943b583e07d72c5ba831049211a737be53277061.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/5851d7335b20b8f28543edaa943b583e07d72c5ba831049211a737be53277061.jpg)

![59946f306e534605874f03e15b54a78bfee1efce302d6ce0c2a692bcbada8a55.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/59946f306e534605874f03e15b54a78bfee1efce302d6ce0c2a692bcbada8a55.jpg)

![61a07bb4b2712e4d31fa4c4a45f9f74b9e1bae5ca6d7953f5df0a51bf2389310.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/61a07bb4b2712e4d31fa4c4a45f9f74b9e1bae5ca6d7953f5df0a51bf2389310.jpg)

![67ec6de7bf7cb0944db1efe1886f4a53442ead1c1ba97b7864b2b00db675cd49.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/67ec6de7bf7cb0944db1efe1886f4a53442ead1c1ba97b7864b2b00db675cd49.jpg)

![689aa927a199e26b7bbe42749623bddb9543e5acb68ba8161bb96bedd1116934.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/689aa927a199e26b7bbe42749623bddb9543e5acb68ba8161bb96bedd1116934.jpg)

![6ad1662913023be56ad0d211d6272a5cb03da262664a3af12781caa6288fbb98.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/6ad1662913023be56ad0d211d6272a5cb03da262664a3af12781caa6288fbb98.jpg)

![6ae9d95249d59f3107a0be5792ee7dabf2c0d95c1731bfb0a9c5df1e009b7a17.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/6ae9d95249d59f3107a0be5792ee7dabf2c0d95c1731bfb0a9c5df1e009b7a17.jpg)

![7abd0d50514eda5d77d769451b91169e82ffaacb2eda3e3ee79e06481e9f50b5.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/7abd0d50514eda5d77d769451b91169e82ffaacb2eda3e3ee79e06481e9f50b5.jpg)

![7d9f7b5ef50a785bd329c16df5bde5040bcd3ef94d7e2baed53a9471f2f338c4.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/7d9f7b5ef50a785bd329c16df5bde5040bcd3ef94d7e2baed53a9471f2f338c4.jpg)

![85951a9c29bde8c8b02efb4d32afd7f4c9102faa6b7121c08e9ac3a7b570b533.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/85951a9c29bde8c8b02efb4d32afd7f4c9102faa6b7121c08e9ac3a7b570b533.jpg)

![87105518c48f1d146baeab35f3d9a863ac0a8e6e143828987727156ca57aea89.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/87105518c48f1d146baeab35f3d9a863ac0a8e6e143828987727156ca57aea89.jpg)

![894f39f9ad61c5325330ef59bc34e7df53fa67e25af357dddba86c40f7d507ff.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/894f39f9ad61c5325330ef59bc34e7df53fa67e25af357dddba86c40f7d507ff.jpg)

![8a0b602e616e406cc0386d15047f327aa4e53b1b5ac54cb9de2652bc202b0592.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/8a0b602e616e406cc0386d15047f327aa4e53b1b5ac54cb9de2652bc202b0592.jpg)

![8f2b11c0fed7afb06dfb967de72cb4b116023511689addc5cf58baca985c6b6b.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/8f2b11c0fed7afb06dfb967de72cb4b116023511689addc5cf58baca985c6b6b.jpg)

![93eebc8a25991431b5db9ba73088f0dadb6402c7c848a84da3131750dcd6c83c.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/93eebc8a25991431b5db9ba73088f0dadb6402c7c848a84da3131750dcd6c83c.jpg)

![96009f4e3babea3dadc0b20143352d500ba922d64a7513dda31591e61628bce7.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/96009f4e3babea3dadc0b20143352d500ba922d64a7513dda31591e61628bce7.jpg)

![97cc9cbc786e0d62e263466f34200d80ea82b9d48f5318b9bdc978ba2e6a5fb6.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/97cc9cbc786e0d62e263466f34200d80ea82b9d48f5318b9bdc978ba2e6a5fb6.jpg)

![9ea9e46573ff2031a1d4940a30cfdb496c5d6ee8ac0e5d9bfbbc78e9d809d9c0.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/9ea9e46573ff2031a1d4940a30cfdb496c5d6ee8ac0e5d9bfbbc78e9d809d9c0.jpg)

![a1bbac03c92ca1dfac505d64c367ce452168c9d1328f59612b073f0ee5895f40.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/a1bbac03c92ca1dfac505d64c367ce452168c9d1328f59612b073f0ee5895f40.jpg)

![abdac2963e9fb75856d0c6d7fed0168e0a7c1e4ac440940e7ab1e9ba2a89f257.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/abdac2963e9fb75856d0c6d7fed0168e0a7c1e4ac440940e7ab1e9ba2a89f257.jpg)

![af2a0242c58a0a80e9d60a5731c56f71a41c666bb1b4e0625db4d9a54f9693f1.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/af2a0242c58a0a80e9d60a5731c56f71a41c666bb1b4e0625db4d9a54f9693f1.jpg)

![b4c7d5fbc9d2dbc9e22461bd931265d3ae890a7813776e6b8e37182972cc4149.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/b4c7d5fbc9d2dbc9e22461bd931265d3ae890a7813776e6b8e37182972cc4149.jpg)

![b74aeb4d868508d0ec28afdcc1f1bab9d3174a1fe7c839172bb6815e6079ba5b.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/b74aeb4d868508d0ec28afdcc1f1bab9d3174a1fe7c839172bb6815e6079ba5b.jpg)

![b8537e7dae746353e753ca67ca50b6f79bdcef7705ef4f45405abc4e6155d16a.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/b8537e7dae746353e753ca67ca50b6f79bdcef7705ef4f45405abc4e6155d16a.jpg)

![b98aeae98009a8b582c988e52f599b65c9a6fa9fe498aa903aba1c910347ebee.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/b98aeae98009a8b582c988e52f599b65c9a6fa9fe498aa903aba1c910347ebee.jpg)

![c53e5d2d7148fbcc39c796ee78829c0cdc73ddf4be6b4731e7a5fbae81d7f229.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/c53e5d2d7148fbcc39c796ee78829c0cdc73ddf4be6b4731e7a5fbae81d7f229.jpg)

![c6a02fd151df1c638c9cb098ac98d87088592e9260c5ead33075ac7fcfac7703.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/c6a02fd151df1c638c9cb098ac98d87088592e9260c5ead33075ac7fcfac7703.jpg)

![cd02f4308d76451aa1b8e42af987c592ca369423e3da5e2d6fff514f71fab0df.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/cd02f4308d76451aa1b8e42af987c592ca369423e3da5e2d6fff514f71fab0df.jpg)

![d6165d72ec7057fe768b703a70b7092e99a7b5824f3eca53ab2c36fef14ab667.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/d6165d72ec7057fe768b703a70b7092e99a7b5824f3eca53ab2c36fef14ab667.jpg)

![de667e84212a6cdfe43779f31688d7c1d4a828bcbe601e8fb98f3856de7b2a23.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/de667e84212a6cdfe43779f31688d7c1d4a828bcbe601e8fb98f3856de7b2a23.jpg)

![e4c00c7f46e765f69be8892e14bead121de663e0abc3fb4e09214710aaceb864.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/e4c00c7f46e765f69be8892e14bead121de663e0abc3fb4e09214710aaceb864.jpg)

![efd77292b0f59112b618a10645a7e2a34f8c1ce9db44772e23f7a69dd9990240.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/efd77292b0f59112b618a10645a7e2a34f8c1ce9db44772e23f7a69dd9990240.jpg)

![f0f2221201a1450d1150cd0d522a55db667258b76b9b622229a0b375f245dfb3.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/f0f2221201a1450d1150cd0d522a55db667258b76b9b622229a0b375f245dfb3.jpg)

![f52800451c9ee2b7a14479f9da112d92fabcf98fe8f8868f5e0fcf2dd5227109.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/f52800451c9ee2b7a14479f9da112d92fabcf98fe8f8868f5e0fcf2dd5227109.jpg)

![f699ce50b586ace854266dd2b8b106a57b626bf31a6a2d7dd8240e1ec9ae5f6c.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/f699ce50b586ace854266dd2b8b106a57b626bf31a6a2d7dd8240e1ec9ae5f6c.jpg)

![f8cb5fb78b961e387d48f73ec56a26194c9962a0c20baf76c51bda40fa654cc7.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/f8cb5fb78b961e387d48f73ec56a26194c9962a0c20baf76c51bda40fa654cc7.jpg)

![fd2b1df9d98a8ea6ae2a229c286ade1ed6b66d55ad96ea84bf2ade80cdf0193d.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/images/fd2b1df9d98a8ea6ae2a229c286ade1ed6b66d55ad96ea84bf2ade80cdf0193d.jpg)

### Tables

![3f6f6d84b393a9e818ca849ba22a4a4cad67a6e3782fbaca91e6f58c2194e77e.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/3f6f6d84b393a9e818ca849ba22a4a4cad67a6e3782fbaca91e6f58c2194e77e.jpg)

![4afcbb0105069f1fc99e4cc79d3f0e2bbd58820f7c8a15afdc0a969433601a2a.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/4afcbb0105069f1fc99e4cc79d3f0e2bbd58820f7c8a15afdc0a969433601a2a.jpg)

![6807939584f442e2f2f5b88447878b9d2215d6f1b5d2eb3b33575365b8b7224a.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/6807939584f442e2f2f5b88447878b9d2215d6f1b5d2eb3b33575365b8b7224a.jpg)

![72d97b5a800b02145429f8235271169de2fd26983aaaff5ded5b1847453e5926.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/72d97b5a800b02145429f8235271169de2fd26983aaaff5ded5b1847453e5926.jpg)

![aa1712d8a9d6668bc62df60ba409c7e98b7d9c351aaf9d66f562a4543bf5cb8a.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/aa1712d8a9d6668bc62df60ba409c7e98b7d9c351aaf9d66f562a4543bf5cb8a.jpg)

![e873116f47a3bdbf38aa036da0d8686284024950e304f8bd422d656000a5951b.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/e873116f47a3bdbf38aa036da0d8686284024950e304f8bd422d656000a5951b.jpg)

![eb0750fb859c8bf8c57bfe47e093b727984f5b73294dbdd6267dd24de5b849f4.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/eb0750fb859c8bf8c57bfe47e093b727984f5b73294dbdd6267dd24de5b849f4.jpg)

![ebf0c940b05d1177e6c15025ef1039bb7fb46a038a990efeae7f6e1b26cf6da1.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/ebf0c940b05d1177e6c15025ef1039bb7fb46a038a990efeae7f6e1b26cf6da1.jpg)

![ee26c497dd911534bdc988838242f0bc1dcfaa7ce770d1263420f5a4bf9fe25c.jpg](../icml_results/309_RE-Bench_%20Evaluating%20Frontier%20AI%20R%26D%20Capabilities%20of%20Language%20Model%20Agents%20against%20Human%20Experts/tables/ee26c497dd911534bdc988838242f0bc1dcfaa7ce770d1263420f5a4bf9fe25c.jpg)

## CVE-Bench: A Benchmark for AI Agents’ Ability to Exploit Real-World Web Application Vulnerabilities


### Images

![2416d077cae8524c99f497eaee2409f563a55c34d9889b2d182210506c10cd4d.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/images/2416d077cae8524c99f497eaee2409f563a55c34d9889b2d182210506c10cd4d.jpg)

![337096305870cb34fa2e785f63fcf5cb309c9b5f2c16873a8e7aed3339341fd5.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/images/337096305870cb34fa2e785f63fcf5cb309c9b5f2c16873a8e7aed3339341fd5.jpg)

![389580307f89033f9bab73a537fae118f445a6168ae8c81f33aa726858bd7f01.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/images/389580307f89033f9bab73a537fae118f445a6168ae8c81f33aa726858bd7f01.jpg)

![40fe33deb4590047508809a14f3a3f94d912009592c97dd83a858a69c7d39853.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/images/40fe33deb4590047508809a14f3a3f94d912009592c97dd83a858a69c7d39853.jpg)

![bf31f30e993a4306fdf385ac26eee3438cb0dcbacb5ab12effd4d4c49731d382.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/images/bf31f30e993a4306fdf385ac26eee3438cb0dcbacb5ab12effd4d4c49731d382.jpg)

![cfc2f70770a3e77bfc1a7c9fade5cabc2b6ea9f99d618be56df6aca51e43a328.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/images/cfc2f70770a3e77bfc1a7c9fade5cabc2b6ea9f99d618be56df6aca51e43a328.jpg)

![ef6c31faef65c7e5d2008716c429b7df8a524a493a5956693933f1d0c580b038.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/images/ef6c31faef65c7e5d2008716c429b7df8a524a493a5956693933f1d0c580b038.jpg)

### Tables

![21d58d169e42b38a0494e4927df2e47f57aae1e1de09d2537cf2d9e191a5be9c.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/tables/21d58d169e42b38a0494e4927df2e47f57aae1e1de09d2537cf2d9e191a5be9c.jpg)

![6349c8216372bdba9618a82f334428f6e6ba8e33be98a1382a5aa8653aad84ff.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/tables/6349c8216372bdba9618a82f334428f6e6ba8e33be98a1382a5aa8653aad84ff.jpg)

![6ff057a7ecc8b000a90e873f4682abfe7bb78ec6ff3d3c90627ca2e27e8bd7ac.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/tables/6ff057a7ecc8b000a90e873f4682abfe7bb78ec6ff3d3c90627ca2e27e8bd7ac.jpg)

![864336b08899c6119e2369066770b47c7ec258326549c8a1d9d46744e515561e.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/tables/864336b08899c6119e2369066770b47c7ec258326549c8a1d9d46744e515561e.jpg)

![a34ce7b3e4e82116442d966694e71087ee9fd4d6609ed5dde859a70dfab36bb7.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/tables/a34ce7b3e4e82116442d966694e71087ee9fd4d6609ed5dde859a70dfab36bb7.jpg)

![d33cc64d220ce203fd951220449802c4cf6d77f652fb1c930c946aa85a28dd3e.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/tables/d33cc64d220ce203fd951220449802c4cf6d77f652fb1c930c946aa85a28dd3e.jpg)

![e13480c936371997c798916c5ed4d0fdd3b8275d545d6da5f811da5bf66eddc3.jpg](../icml_results/310_CVE-Bench_%20A%20Benchmark%20for%20AI%20Agents%E2%80%99%20Ability%20to%20Exploit%20Real-World%20Web%20Application%20Vulnerabilities/tables/e13480c936371997c798916c5ed4d0fdd3b8275d545d6da5f811da5bf66eddc3.jpg)

## Great Models Think Alike and this Undermines AI Oversight


### Images

![05cdb626972752da7953bb35d88ca92c2749670a7be563215e70623955e2f729.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/05cdb626972752da7953bb35d88ca92c2749670a7be563215e70623955e2f729.jpg)

![089b9befe8c762a2dbc27acd2798e7f65914cc69a2e050844dadeecece04e9fa.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/089b9befe8c762a2dbc27acd2798e7f65914cc69a2e050844dadeecece04e9fa.jpg)

![10d2b9a9a890cc8d661364ecc47e62914b75a0a430b2c55486f7da7e84fcb212.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/10d2b9a9a890cc8d661364ecc47e62914b75a0a430b2c55486f7da7e84fcb212.jpg)

![2f0b3bec0cded920466ec28c8a21283b640e08525e00dda586744df28518273d.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/2f0b3bec0cded920466ec28c8a21283b640e08525e00dda586744df28518273d.jpg)

![333ce8845b2dbce93a4a3e40fb4d49609539fd89ac9046fb04b6830b19dbe8c4.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/333ce8845b2dbce93a4a3e40fb4d49609539fd89ac9046fb04b6830b19dbe8c4.jpg)

![3ecfb312bb6251b69baa18257bf07059692f2d376f5a8bf1091becf8c0bbfd28.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/3ecfb312bb6251b69baa18257bf07059692f2d376f5a8bf1091becf8c0bbfd28.jpg)

![4da9066f118bedc2f2e097a848740561ba04c426992b61882bf8510e4ae1a2c6.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/4da9066f118bedc2f2e097a848740561ba04c426992b61882bf8510e4ae1a2c6.jpg)

![57134e07d6edec6819c5a12c8659c94d251f5ab2c4faad11c2a8dd79134c17f0.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/57134e07d6edec6819c5a12c8659c94d251f5ab2c4faad11c2a8dd79134c17f0.jpg)

![5e2e97fc230770c053f7de6bfbcf9056d15990058a411ca51b434f048f9faf7e.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/5e2e97fc230770c053f7de6bfbcf9056d15990058a411ca51b434f048f9faf7e.jpg)

![6b79a9a8e6853aacc0d44025cf391e1c72d64b72999c04441d048fec254cf489.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/6b79a9a8e6853aacc0d44025cf391e1c72d64b72999c04441d048fec254cf489.jpg)

![6ebeac02402b1edb032215e1c86abb1ad31939d95fe5af930561dd9fe4d17fd6.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/6ebeac02402b1edb032215e1c86abb1ad31939d95fe5af930561dd9fe4d17fd6.jpg)

![783545539b82d5f81c57766e3c0bcd7ce3d3b66ebd02dce558c4aacc2eedaef3.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/783545539b82d5f81c57766e3c0bcd7ce3d3b66ebd02dce558c4aacc2eedaef3.jpg)

![835493988a1339953b03d36ab1acd6eb61d1f41a41f6873e1fa5ecd9fb428750.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/835493988a1339953b03d36ab1acd6eb61d1f41a41f6873e1fa5ecd9fb428750.jpg)

![9399ea5783f5ef31bbb35caf94af54ef787017e92e3c6135d4c97603378e1cd8.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/9399ea5783f5ef31bbb35caf94af54ef787017e92e3c6135d4c97603378e1cd8.jpg)

![b100dfcabf78f2c12064dee8f782d76161f2a2b6fff02a78096cc5e01c83b057.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/b100dfcabf78f2c12064dee8f782d76161f2a2b6fff02a78096cc5e01c83b057.jpg)

![bed2b3f8f73cc158c59e0ef0b92d245f6c8201f2ff3c15e9a9a50ad6f6c4f62c.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/bed2b3f8f73cc158c59e0ef0b92d245f6c8201f2ff3c15e9a9a50ad6f6c4f62c.jpg)

![c185d290d1091b905351036cc88c0e709f55ebc3b8446519392bdbec6a96c423.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/c185d290d1091b905351036cc88c0e709f55ebc3b8446519392bdbec6a96c423.jpg)

![c38c5cdd2400f41d1443fec962086c804887576628876818541ddfa158bbcc89.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/c38c5cdd2400f41d1443fec962086c804887576628876818541ddfa158bbcc89.jpg)

![c41c9208f7eb5bad16de8f13e31ba2ec633af2cce3d2204d06dcf0c1b20fe7ff.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/c41c9208f7eb5bad16de8f13e31ba2ec633af2cce3d2204d06dcf0c1b20fe7ff.jpg)

![cb4c31746799c218adbebaf7ebd01c8023f91ac734d5d91d77f709bb70264479.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/cb4c31746799c218adbebaf7ebd01c8023f91ac734d5d91d77f709bb70264479.jpg)

![dad894ae027ebbccd42499aa11919ac0fbcd127d4bb5ef5a73d0ba4defa4fd47.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/dad894ae027ebbccd42499aa11919ac0fbcd127d4bb5ef5a73d0ba4defa4fd47.jpg)

![edeabf28ebfa4139c1e432f7b0eff07c093b97f3ab9fa14bff1d7f83ce76844e.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/edeabf28ebfa4139c1e432f7b0eff07c093b97f3ab9fa14bff1d7f83ce76844e.jpg)

![ee72956233b8405bb740d54b2895312ebdc7ee866ea319c25b883b0764ca94c9.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/ee72956233b8405bb740d54b2895312ebdc7ee866ea319c25b883b0764ca94c9.jpg)

![f3bd8ec7ebf874157fd8acdcf8c2b2de318c3dc16b30ea3210dcaa7657b9e839.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/images/f3bd8ec7ebf874157fd8acdcf8c2b2de318c3dc16b30ea3210dcaa7657b9e839.jpg)

### Tables

![0116ea57e8ca00305016971d5c43b8f1378d33b9943ef3548180faf43786e030.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/0116ea57e8ca00305016971d5c43b8f1378d33b9943ef3548180faf43786e030.jpg)

![0bd75179f567f45c73eaf2aed9e8a888666c6bc006290743c7ad833e6adb146e.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/0bd75179f567f45c73eaf2aed9e8a888666c6bc006290743c7ad833e6adb146e.jpg)

![0d4d97242aaee465eb1384b6eb1e12920e28b5c03a080684fce4b873fbaf71dc.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/0d4d97242aaee465eb1384b6eb1e12920e28b5c03a080684fce4b873fbaf71dc.jpg)

![18396fb5c914b7f3acdb53138fd8fe74791b3a25e3069db19b864ca5feda0fe9.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/18396fb5c914b7f3acdb53138fd8fe74791b3a25e3069db19b864ca5feda0fe9.jpg)

![1936e3936a20e2f65bf4a39cc9e70f21acaf55cdf8e8723d248a9776467f7582.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/1936e3936a20e2f65bf4a39cc9e70f21acaf55cdf8e8723d248a9776467f7582.jpg)

![1a3c5c08253e9c40380b4560cfa0009bb1f77fae463e1ea6b99f1a82ffc2f724.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/1a3c5c08253e9c40380b4560cfa0009bb1f77fae463e1ea6b99f1a82ffc2f724.jpg)

![3de7b8c25dfb1c28acb325de5fdf795ae6f95ff660087069f96b1eb6b9bb16ca.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/3de7b8c25dfb1c28acb325de5fdf795ae6f95ff660087069f96b1eb6b9bb16ca.jpg)

![46cf3e1065a1ab536bb7b5881d187eca224b99debeff104ee928e5eb3ceae50c.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/46cf3e1065a1ab536bb7b5881d187eca224b99debeff104ee928e5eb3ceae50c.jpg)

![5687f97ca515407a6fcf165170ba49d4188d9ec05546c3c3801b28dfc46dd3f4.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/5687f97ca515407a6fcf165170ba49d4188d9ec05546c3c3801b28dfc46dd3f4.jpg)

![6ef28fe5536c9c41fbeedc0c6c319728593392b7b5cf666063269fe5e1dd15d2.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/6ef28fe5536c9c41fbeedc0c6c319728593392b7b5cf666063269fe5e1dd15d2.jpg)

![6f3507affdd8bcb5c5799c5b816f6e6fa58467338935764f42d846b8b088c81f.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/6f3507affdd8bcb5c5799c5b816f6e6fa58467338935764f42d846b8b088c81f.jpg)

![71d235df83acb648d2bc9bbf0305014c3790f3d913ec6a755a6ae80bc92a59f7.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/71d235df83acb648d2bc9bbf0305014c3790f3d913ec6a755a6ae80bc92a59f7.jpg)

![75fdae1e6b1898220cedee08ad7632b2438ec08bf1121438e73853aea420c81e.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/75fdae1e6b1898220cedee08ad7632b2438ec08bf1121438e73853aea420c81e.jpg)

![7bf58fffbb4a68b2ef2848ae58c2e0630396ac912fc318543a6e5ca611ee649a.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/7bf58fffbb4a68b2ef2848ae58c2e0630396ac912fc318543a6e5ca611ee649a.jpg)

![844e1eee4c22a36844230f9dde617cf4dab201dd4b4418293271eb63403e9282.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/844e1eee4c22a36844230f9dde617cf4dab201dd4b4418293271eb63403e9282.jpg)

![84b54ee8d268fcb724a9c35de724f2526f69057893e47e4a9c6f3acd109fed29.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/84b54ee8d268fcb724a9c35de724f2526f69057893e47e4a9c6f3acd109fed29.jpg)

![8a0d2ff3af09fd614afc31330e9390abd563d6c2122015b19b9aecab74ed2248.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/8a0d2ff3af09fd614afc31330e9390abd563d6c2122015b19b9aecab74ed2248.jpg)

![8a31fa59613877a3490a8eeac94709b7dc86024cb82954761ce4a5391e8c1f01.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/8a31fa59613877a3490a8eeac94709b7dc86024cb82954761ce4a5391e8c1f01.jpg)

![94b3aeba8eaf11226109579111d7b2880b15d334dcfa190a664d3ce2b8390470.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/94b3aeba8eaf11226109579111d7b2880b15d334dcfa190a664d3ce2b8390470.jpg)

![99cb1326ad282d24f8da8b23b9db426316d2317c1079012740a0d8c790fad797.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/99cb1326ad282d24f8da8b23b9db426316d2317c1079012740a0d8c790fad797.jpg)

![a6f3bb49e6c73f4aaaf0355112cd6c3988a793b0e0c9226e4ecf66e8603538c7.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/a6f3bb49e6c73f4aaaf0355112cd6c3988a793b0e0c9226e4ecf66e8603538c7.jpg)

![ae3e1a282c24387f4772f6efdc2b9759f167ddb5de75b1aff42badbe21b686c3.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/ae3e1a282c24387f4772f6efdc2b9759f167ddb5de75b1aff42badbe21b686c3.jpg)

![b12a85ebc971d798221c3085aaa33310d2e3949d639ea2eacc29814dfaf9eb0a.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/b12a85ebc971d798221c3085aaa33310d2e3949d639ea2eacc29814dfaf9eb0a.jpg)

![b669cf802886394e801c1ee3288eeac963181f517523f4df79a3ca263782ae42.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/b669cf802886394e801c1ee3288eeac963181f517523f4df79a3ca263782ae42.jpg)

![b6a417f4683a28fbdbdfdfb9f2245e667f0caa868db1a679c603becbb0595cae.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/b6a417f4683a28fbdbdfdfb9f2245e667f0caa868db1a679c603becbb0595cae.jpg)

![b6bcf76042a0893c705b5fc1be9e1968a4c26cf883c2f12bb3e05bde6a83685a.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/b6bcf76042a0893c705b5fc1be9e1968a4c26cf883c2f12bb3e05bde6a83685a.jpg)

![c4b1e62f8972a820e49ee134bb87e1e5c676b27bd78ea4faa0924fd789cf4d97.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/c4b1e62f8972a820e49ee134bb87e1e5c676b27bd78ea4faa0924fd789cf4d97.jpg)

![ca95e3a69d558838d83d4ff99eae7b7228422b72dbe2e6abf671a81238602a2c.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/ca95e3a69d558838d83d4ff99eae7b7228422b72dbe2e6abf671a81238602a2c.jpg)

![ce9293cab5529c55d716fdaf06c3fd04705dd4eb7f8c24d558666c815e828dfd.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/ce9293cab5529c55d716fdaf06c3fd04705dd4eb7f8c24d558666c815e828dfd.jpg)

![d3694aaed64c77f5e06c01d314f5933a4dcb7fb227488d33ae8e50ee7ba7e3ae.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/d3694aaed64c77f5e06c01d314f5933a4dcb7fb227488d33ae8e50ee7ba7e3ae.jpg)

![d757e8fc3af944a9b01dd175861d15cd09e9636c96a2efd598ea8f7e8a0f14db.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/d757e8fc3af944a9b01dd175861d15cd09e9636c96a2efd598ea8f7e8a0f14db.jpg)

![de5b3d5dd2afa33a7063ba12f8e0c960f9657764d535eb7c739f33dabea4a8bf.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/de5b3d5dd2afa33a7063ba12f8e0c960f9657764d535eb7c739f33dabea4a8bf.jpg)

![f55961d39d27845952632972047a831334f5f0804fd6ba48bb28dd7e36674cb7.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/f55961d39d27845952632972047a831334f5f0804fd6ba48bb28dd7e36674cb7.jpg)

![f9a201d2e892e3e7ddaf9681abd451af299f6eda2221293883b9dff20669cf9b.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/f9a201d2e892e3e7ddaf9681abd451af299f6eda2221293883b9dff20669cf9b.jpg)

![fa686679aaf9135d9ec09af4c4f1fb5f44bedb85d467010cdec1885962926a72.jpg](../icml_results/311_Great%20Models%20Think%20Alike%20and%20this%20Undermines%20AI%20Oversight/tables/fa686679aaf9135d9ec09af4c4f1fb5f44bedb85d467010cdec1885962926a72.jpg)

## Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner


### Images

![33add81a37a328ba16e33c1648a2531533337b60c64a13be68dd3777ddce2432.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/images/33add81a37a328ba16e33c1648a2531533337b60c64a13be68dd3777ddce2432.jpg)

![5637e596ebd78284085fddf31f3f2c47d25d6597bc1a3cbcae99bcc782d185ce.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/images/5637e596ebd78284085fddf31f3f2c47d25d6597bc1a3cbcae99bcc782d185ce.jpg)

![8d69ce3363762b7f26ad5be4e5d63be41c4d6a6481975a5476603d214fa8dc36.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/images/8d69ce3363762b7f26ad5be4e5d63be41c4d6a6481975a5476603d214fa8dc36.jpg)

![8de5d1ad366eef6309cf74e3f1aa06a33ad5013c3862098c0d945ff1aa9aa1b9.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/images/8de5d1ad366eef6309cf74e3f1aa06a33ad5013c3862098c0d945ff1aa9aa1b9.jpg)

![9d3808756be6b63a017f3d9ba0b32445e3823ce415ac7ce26fa8addcc39a0074.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/images/9d3808756be6b63a017f3d9ba0b32445e3823ce415ac7ce26fa8addcc39a0074.jpg)

![fa7f24fd3c11ead5b9cd2a36aa938d8dd827e53995a370d282e1107a06f25739.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/images/fa7f24fd3c11ead5b9cd2a36aa938d8dd827e53995a370d282e1107a06f25739.jpg)

### Tables

![0eb49b2729922cf6301f5916bde07b78f82dbe8ce3ee65fa49d4a1ad0b1b7458.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/0eb49b2729922cf6301f5916bde07b78f82dbe8ce3ee65fa49d4a1ad0b1b7458.jpg)

![1e0ec9018442dd6ae6b0ca3669858e6dded8202c1ff7ce74eb74c54f3d52013b.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/1e0ec9018442dd6ae6b0ca3669858e6dded8202c1ff7ce74eb74c54f3d52013b.jpg)

![51d0ec8b1737d84902ee3e40660b5aa5b71d37bdf06bf7a0f0baf55fa08ac473.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/51d0ec8b1737d84902ee3e40660b5aa5b71d37bdf06bf7a0f0baf55fa08ac473.jpg)

![57b4e4b8cba6bd901cc5c4c7924bf85fd9a120c831cf7efb18258cd9c578d79b.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/57b4e4b8cba6bd901cc5c4c7924bf85fd9a120c831cf7efb18258cd9c578d79b.jpg)

![638057ca32d8a450255f5f0893613cc96c1aada83bd0a82b265cd9348fc36591.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/638057ca32d8a450255f5f0893613cc96c1aada83bd0a82b265cd9348fc36591.jpg)

![798446315f23c369002ad44f2611ffd0cb482e414e7036eb0fe4dd6a0ace41cb.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/798446315f23c369002ad44f2611ffd0cb482e414e7036eb0fe4dd6a0ace41cb.jpg)

![79c4e16d71854a12086120d3d0da5b506a80c90ea33c86d6147c074b8ce37941.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/79c4e16d71854a12086120d3d0da5b506a80c90ea33c86d6147c074b8ce37941.jpg)

![8048cf4377899fb818cb1c9c8ccc1d35bded064d6f8537c0db606723d728b3e2.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/8048cf4377899fb818cb1c9c8ccc1d35bded064d6f8537c0db606723d728b3e2.jpg)

![837d123e10f5b9b08b20521e210fdd5927a8e2a39f7d147e6ff06f57902ed92a.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/837d123e10f5b9b08b20521e210fdd5927a8e2a39f7d147e6ff06f57902ed92a.jpg)

![8cfcaf9d07b74cf6facdca535110e871dd6e7cb548248d960d52ddcec557939a.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/8cfcaf9d07b74cf6facdca535110e871dd6e7cb548248d960d52ddcec557939a.jpg)

![a80abc2851f26df685c99e7a9d09564ab0803058c8d47007ed0838238d0aaba4.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/a80abc2851f26df685c99e7a9d09564ab0803058c8d47007ed0838238d0aaba4.jpg)

![ac1393ba23b6946702fa59f67fabacff0f9091992561ae812fa5cf62e186932d.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/ac1393ba23b6946702fa59f67fabacff0f9091992561ae812fa5cf62e186932d.jpg)

![e8d90f48616093adb105207646d975f99970f0d686a699ddd36100f3a9ef9002.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/e8d90f48616093adb105207646d975f99970f0d686a699ddd36100f3a9ef9002.jpg)

![f1d60226f0d1cfc064d40a229197aedd2a7ce0d316ab93dbbf42375315110657.jpg](../icml_results/312_Overcoming%20Multi-step%20Complexity%20in%20Multimodal%20Theory-of-Mind%20Reasoning_%20A%20Scalable%20Bayesian%20Planner/tables/f1d60226f0d1cfc064d40a229197aedd2a7ce0d316ab93dbbf42375315110657.jpg)

## Doubly Robust Conformalized Survival Analysis with Right-Censored Data


### Images

![02ee8ec44d873b5df85f11a7c60202f3e1b641f46c95532e0e4b27eece911d42.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/02ee8ec44d873b5df85f11a7c60202f3e1b641f46c95532e0e4b27eece911d42.jpg)

![0865a8a595cc82977d1671a33b75ff27f2411f12e86699f88a0fbd09248b62aa.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/0865a8a595cc82977d1671a33b75ff27f2411f12e86699f88a0fbd09248b62aa.jpg)

![126dfce649471c062cf9f83b73079b1c0928b6675c22dba82c720941d7fb9f8c.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/126dfce649471c062cf9f83b73079b1c0928b6675c22dba82c720941d7fb9f8c.jpg)

![2db31677bf7f9eb4d15a94f0cac121c8ae18f2619e9e0ed3058aebeb1c46f142.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/2db31677bf7f9eb4d15a94f0cac121c8ae18f2619e9e0ed3058aebeb1c46f142.jpg)

![353544c97b6fe86976bf263eb9254b8c12036df4478bafb45c9069c361687601.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/353544c97b6fe86976bf263eb9254b8c12036df4478bafb45c9069c361687601.jpg)

![361db69a505ab9dba0866f6a46d651ed5121bfa75cc5a4610ab84d85cd8d472b.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/361db69a505ab9dba0866f6a46d651ed5121bfa75cc5a4610ab84d85cd8d472b.jpg)

![4a86a7db96217b0856e159e8118e11a83c5c4a1be81f7efea40d02ab541a172f.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/4a86a7db96217b0856e159e8118e11a83c5c4a1be81f7efea40d02ab541a172f.jpg)

![620ee2072034eddcd4edb0db9642a43cfa05691dbd7e8a5b0e66e90752475995.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/620ee2072034eddcd4edb0db9642a43cfa05691dbd7e8a5b0e66e90752475995.jpg)

![6f44fa1b9956418d80fa67cf53925a73fbc716c5e68f64c8d3bc8f1c88395b93.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/6f44fa1b9956418d80fa67cf53925a73fbc716c5e68f64c8d3bc8f1c88395b93.jpg)

![764f59b2d2c704f8bdb425154cb1530a7d104bee2a7074b4f0c89abfd6245369.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/764f59b2d2c704f8bdb425154cb1530a7d104bee2a7074b4f0c89abfd6245369.jpg)

![79f7a86c7e4bbe34aa490aa72f1c65ddba933c603523213c415e85903e57a623.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/79f7a86c7e4bbe34aa490aa72f1c65ddba933c603523213c415e85903e57a623.jpg)

![896df44387e2ebc462a126c57d883faf0534cf520989254a676701323ba25eb6.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/896df44387e2ebc462a126c57d883faf0534cf520989254a676701323ba25eb6.jpg)

![9a03e83738355278ae81b22e8f44a6502c9f3d10005d9b0f108521f6a890d13c.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/9a03e83738355278ae81b22e8f44a6502c9f3d10005d9b0f108521f6a890d13c.jpg)

![ab8ccbd2bbced908c446ad97f743ee1f9fcfd780edf81fd06164aa601a3c3885.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/ab8ccbd2bbced908c446ad97f743ee1f9fcfd780edf81fd06164aa601a3c3885.jpg)

![b92f56c69b6b157e952301e76ac0bd39d000309e89a305f46e812b14d2aedd4d.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/b92f56c69b6b157e952301e76ac0bd39d000309e89a305f46e812b14d2aedd4d.jpg)

![c3388393f49dc04dbdac6d32303be65066488f882c4729e87bc47b4e27282a0c.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/c3388393f49dc04dbdac6d32303be65066488f882c4729e87bc47b4e27282a0c.jpg)

![c6a01bc12596c3a4224f0a74e5bb8aeb891dbecffedbdb463d7ec5bbf97f4339.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/c6a01bc12596c3a4224f0a74e5bb8aeb891dbecffedbdb463d7ec5bbf97f4339.jpg)

![c783f394f4e63dab4dfbd8f7101d35fa23c2096b38fb048fa29daa6e066b12fe.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/c783f394f4e63dab4dfbd8f7101d35fa23c2096b38fb048fa29daa6e066b12fe.jpg)

![d6d3839a32e530e6f0db7493e671a65949af35bd90a206439c308eced17a3340.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/d6d3839a32e530e6f0db7493e671a65949af35bd90a206439c308eced17a3340.jpg)

![d845292937b23e482cf5aa4887d194036530419de16828603e4f48425b1ffff6.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/d845292937b23e482cf5aa4887d194036530419de16828603e4f48425b1ffff6.jpg)

![f0b197bdcf6902cd276c0db12da5f7c040b2a55bd6e4308cc59e4bbb8eac4656.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/images/f0b197bdcf6902cd276c0db12da5f7c040b2a55bd6e4308cc59e4bbb8eac4656.jpg)

### Tables

![1894b99b35d71f9f1e544834616c722092912c24d12f85e5fc02a14b2b82729c.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/tables/1894b99b35d71f9f1e544834616c722092912c24d12f85e5fc02a14b2b82729c.jpg)

![717f09d95892d78fe94405776341f2db803549e23b9cc0c67d07adff70169f34.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/tables/717f09d95892d78fe94405776341f2db803549e23b9cc0c67d07adff70169f34.jpg)

![74c6cc4b3e4aec0706010187ab93ac7ec92fa1fb2f788951c83744f3dfa22616.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/tables/74c6cc4b3e4aec0706010187ab93ac7ec92fa1fb2f788951c83744f3dfa22616.jpg)

![834753450e90f4c7bba7c86e146d65bf5a4359052d0fc2c4013c6c3373496c38.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/tables/834753450e90f4c7bba7c86e146d65bf5a4359052d0fc2c4013c6c3373496c38.jpg)

![aab57e8a3cd0c57a3518f97753aa70077e933f9fa7d5e036bc455ef5e4e63791.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/tables/aab57e8a3cd0c57a3518f97753aa70077e933f9fa7d5e036bc455ef5e4e63791.jpg)

![c5525628167c2b64a00aa1b1639309faff2fac16248d98ccf4f1a270707fafe0.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/tables/c5525628167c2b64a00aa1b1639309faff2fac16248d98ccf4f1a270707fafe0.jpg)

![d12ed2f7890d94752f41fdc3414b90e6807376cc91e4e14aaf015c7f6aecbb3e.jpg](../icml_results/313_Doubly%20Robust%20Conformalized%20Survival%20Analysis%20with%20Right-Censored%20Data/tables/d12ed2f7890d94752f41fdc3414b90e6807376cc91e4e14aaf015c7f6aecbb3e.jpg)

## Training Deep Learning Models with Norm-Constrained LMOs


### Images

![02153a9d886a4fc2cde250052da074841f2a1fd9a0bb1d77fd251a20bb824aa3.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/02153a9d886a4fc2cde250052da074841f2a1fd9a0bb1d77fd251a20bb824aa3.jpg)

![1c871438445812d8bfec87c262808f1b8bd221422af77d7e0b0612c5b00112ba.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/1c871438445812d8bfec87c262808f1b8bd221422af77d7e0b0612c5b00112ba.jpg)

![4f0b67879eaee355ed8995adc346893bb07acc50314804c9c5953fb98c0553ed.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/4f0b67879eaee355ed8995adc346893bb07acc50314804c9c5953fb98c0553ed.jpg)

![5578dec2e5ef9eccb7ecc59a5d5fc3193bd91d0d4a089ccf1213174e17ba81a4.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/5578dec2e5ef9eccb7ecc59a5d5fc3193bd91d0d4a089ccf1213174e17ba81a4.jpg)

![5d3591e4219a7008e15c862107725176b3dd99f656b1efaae3385b1a4a1be456.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/5d3591e4219a7008e15c862107725176b3dd99f656b1efaae3385b1a4a1be456.jpg)

![89122304a3c908de9968097d80ab55038b5f215467725aac9e20f0e74279e514.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/89122304a3c908de9968097d80ab55038b5f215467725aac9e20f0e74279e514.jpg)

![8ceea63e354107ce72ec29d23f5099e60e0d3a49cf9633dc3b8475d2b2e521d5.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/8ceea63e354107ce72ec29d23f5099e60e0d3a49cf9633dc3b8475d2b2e521d5.jpg)

![8f182474dd7a13170d4aae4ce523555e9742af8abb78a015ebbb98a79fcc190a.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/8f182474dd7a13170d4aae4ce523555e9742af8abb78a015ebbb98a79fcc190a.jpg)

![94cd003d3ac9af409a9b3178ce80e04ad59de0942cf9b9cdbf950f917d376544.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/94cd003d3ac9af409a9b3178ce80e04ad59de0942cf9b9cdbf950f917d376544.jpg)

![ad2674c67527cc34fd823b8d5d116e2383e4698d8635e0e710d86a07dad5b2d8.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/ad2674c67527cc34fd823b8d5d116e2383e4698d8635e0e710d86a07dad5b2d8.jpg)

![c9492f3db10c6b813a8f5ca70a78593bc493e1259be5449aaed55d82eacbe3cd.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/c9492f3db10c6b813a8f5ca70a78593bc493e1259be5449aaed55d82eacbe3cd.jpg)

![d81bfcb57e3887ab2a2fd762baa5b8a1e479ec335fcd2a891ee80ce70c931217.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/images/d81bfcb57e3887ab2a2fd762baa5b8a1e479ec335fcd2a891ee80ce70c931217.jpg)

### Tables

![043a401248486fca79a0bba04409a17a74447b0136569ba05be837328725076d.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/043a401248486fca79a0bba04409a17a74447b0136569ba05be837328725076d.jpg)

![1f2290627bd0a1cbe23bb1ab8bb56241e23b6deb0faf7a20931dfb3802f99d44.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/1f2290627bd0a1cbe23bb1ab8bb56241e23b6deb0faf7a20931dfb3802f99d44.jpg)

![47981e014b47cbf6aaa106221c746db34b0104f1bcb57e5ca3881866e4e6ed04.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/47981e014b47cbf6aaa106221c746db34b0104f1bcb57e5ca3881866e4e6ed04.jpg)

![5e6c8c652a487fb03410a081214fe8ab78f35e8f60a3dd39b710644b4ce25662.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/5e6c8c652a487fb03410a081214fe8ab78f35e8f60a3dd39b710644b4ce25662.jpg)

![706e209f198778803c3c3870574906317de2da9898f250c45ec784755a8dc06d.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/706e209f198778803c3c3870574906317de2da9898f250c45ec784755a8dc06d.jpg)

![784bff760fdaeb1143dde636de261a1f9ea8a2cef957c4123906a87dcff81512.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/784bff760fdaeb1143dde636de261a1f9ea8a2cef957c4123906a87dcff81512.jpg)

![89c1d3482afcef8d1f2c6f84d7cc39aedf859235bf6aa2649aad935ec78462a3.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/89c1d3482afcef8d1f2c6f84d7cc39aedf859235bf6aa2649aad935ec78462a3.jpg)

![c03300f0348fde1d948c72986f34f4e730ddb76c85e4ddfd0d4a3926c6f66585.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/c03300f0348fde1d948c72986f34f4e730ddb76c85e4ddfd0d4a3926c6f66585.jpg)

![d25cec7362b8d3ef41ebd56d611655caa353fdaca841ae0e7ccaedf5202c7c6c.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/d25cec7362b8d3ef41ebd56d611655caa353fdaca841ae0e7ccaedf5202c7c6c.jpg)

![d27468140a9c23b85d95477b7624d1ac8a409efcf6fdbaa2665773f9f71fbfeb.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/d27468140a9c23b85d95477b7624d1ac8a409efcf6fdbaa2665773f9f71fbfeb.jpg)

![febf1d84243d79b78df6de92f1d4eac256c6ea522990ebb12a6a94e233354909.jpg](../icml_results/314_Training%20Deep%20Learning%20Models%20with%20Norm-Constrained%20LMOs/tables/febf1d84243d79b78df6de92f1d4eac256c6ea522990ebb12a6a94e233354909.jpg)

## Generalized Random Forests Using Fixed-Point Trees


### Images

![160a9b35ac892c8d528da5114f48ca2d6d84231339174b9cf8ff7d156f6dab1a.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/160a9b35ac892c8d528da5114f48ca2d6d84231339174b9cf8ff7d156f6dab1a.jpg)

![1e526f20d4483bec1ca75151ddcd947e07d758e9c839972e0d044d4962713c9b.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/1e526f20d4483bec1ca75151ddcd947e07d758e9c839972e0d044d4962713c9b.jpg)

![308190d6a14114c6a21b41336272fa114a125f3db7ee7204463c9f1b86c7ecc2.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/308190d6a14114c6a21b41336272fa114a125f3db7ee7204463c9f1b86c7ecc2.jpg)

![3a26192a0699808be27021feab2b40429c89013b6d2e65078cc998153790bf61.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/3a26192a0699808be27021feab2b40429c89013b6d2e65078cc998153790bf61.jpg)

![44b747733ea7e281b51075bf9005246d90fc3a38d76ed11fdb9878997018438e.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/44b747733ea7e281b51075bf9005246d90fc3a38d76ed11fdb9878997018438e.jpg)

![49555702d46473872122e5228ffcb72a9d5b92739768d9b5ce1da263621fcfa2.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/49555702d46473872122e5228ffcb72a9d5b92739768d9b5ce1da263621fcfa2.jpg)

![4c4d2f14d30e745f87edd21d683af16a8d4baf040ffad499998dab38f3d5e4e9.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/4c4d2f14d30e745f87edd21d683af16a8d4baf040ffad499998dab38f3d5e4e9.jpg)

![50f5dada077d60380c177bb8e5d6a788cd1d59a28a73537dc680a292fb0e0d2e.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/50f5dada077d60380c177bb8e5d6a788cd1d59a28a73537dc680a292fb0e0d2e.jpg)

![7402f6ad74220b59dd1d087059333eb53f58224ad34503dc7e166ff04d5afdb8.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/7402f6ad74220b59dd1d087059333eb53f58224ad34503dc7e166ff04d5afdb8.jpg)

![7f6c02926419c356e87ebd31fbde80d1843996f11d35bee33856052ba13913b2.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/7f6c02926419c356e87ebd31fbde80d1843996f11d35bee33856052ba13913b2.jpg)

![84910df48da9090dae12f3b499d2324eba84f2a1736ea87e25053bf8f45c2951.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/84910df48da9090dae12f3b499d2324eba84f2a1736ea87e25053bf8f45c2951.jpg)

![86c3a5a996ad854d18e972295df7a4493795bf6f60eec9e41cd19bd3f2afdfe4.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/86c3a5a996ad854d18e972295df7a4493795bf6f60eec9e41cd19bd3f2afdfe4.jpg)

![a6983fc3cf2406a39adb51859c96f3a718597028750c790267377fb651657b04.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/a6983fc3cf2406a39adb51859c96f3a718597028750c790267377fb651657b04.jpg)

![a873bb7bf88f315655504d82547db05f768b848f78e66a5eb9ef463ebf9e466e.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/a873bb7bf88f315655504d82547db05f768b848f78e66a5eb9ef463ebf9e466e.jpg)

![b0182fa18a6a88413999c43f6d526c7722f6afbfb7d4e16e4a9cfd3fd7a9a4a4.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/b0182fa18a6a88413999c43f6d526c7722f6afbfb7d4e16e4a9cfd3fd7a9a4a4.jpg)

![d9992f9649de7537df24a688048e82d768aafcb29d31670032a3f2081c1d0a9c.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/d9992f9649de7537df24a688048e82d768aafcb29d31670032a3f2081c1d0a9c.jpg)

![dbc484e4aa9c51fae90ce0a14c5311b7dcab59cc3d21dc89e2c85570bf0bbb2a.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/dbc484e4aa9c51fae90ce0a14c5311b7dcab59cc3d21dc89e2c85570bf0bbb2a.jpg)

![e6e875a0d9837f1d82ae2eb10f0de2ea0d3d71a5ceca724a3b01090590ead58d.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/e6e875a0d9837f1d82ae2eb10f0de2ea0d3d71a5ceca724a3b01090590ead58d.jpg)

![eef73c6ea7a43212d981fc031e3d4684a875e3f2b102a63f4bf3ed3bea04d319.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/eef73c6ea7a43212d981fc031e3d4684a875e3f2b102a63f4bf3ed3bea04d319.jpg)

![f91eeda0f02071755b15e0cabcf70bcd6e404db7a1266c02f6414fc7afb9b748.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/images/f91eeda0f02071755b15e0cabcf70bcd6e404db7a1266c02f6414fc7afb9b748.jpg)

### Tables

![34ce0460f7f31c6d2446fcca7e9248c2e5f8404d1bc0c915e4d4efe437cf7d0d.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/34ce0460f7f31c6d2446fcca7e9248c2e5f8404d1bc0c915e4d4efe437cf7d0d.jpg)

![49fdd832f82e4b3be3a0a6f1884718ac50dd5381331045480aeb41870e455df8.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/49fdd832f82e4b3be3a0a6f1884718ac50dd5381331045480aeb41870e455df8.jpg)

![5bbf1fffdd8e8d2a3dc5ffb0de4cfac8b99064c69e4ddc09074a437e8cdd5ed4.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/5bbf1fffdd8e8d2a3dc5ffb0de4cfac8b99064c69e4ddc09074a437e8cdd5ed4.jpg)

![5de330881045b5e98676b49e241a68200990e30cf3f5eaecc771cbed132e40f8.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/5de330881045b5e98676b49e241a68200990e30cf3f5eaecc771cbed132e40f8.jpg)

![6f91623f11fa5d24b01fc0e16583ba814f891a22d5d0b1379ac4fe8ca3414fb7.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/6f91623f11fa5d24b01fc0e16583ba814f891a22d5d0b1379ac4fe8ca3414fb7.jpg)

![794c7ed55cb01e9822bd7f093b6c476b305de2c9026659de01d3118b89e467e1.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/794c7ed55cb01e9822bd7f093b6c476b305de2c9026659de01d3118b89e467e1.jpg)

![c3bec2eb8cb19889906962a5f4865edc4b0b91124b7083f03c72bc3df70f227c.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/c3bec2eb8cb19889906962a5f4865edc4b0b91124b7083f03c72bc3df70f227c.jpg)

![d56787b4cd000dd4979a32d836543e23bae457d173d1951c13cd8cc2f0307158.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/d56787b4cd000dd4979a32d836543e23bae457d173d1951c13cd8cc2f0307158.jpg)

![d766c680a34105016ca7b64813d81d30013c9b839a4ca19b5a4da3cf1e0c8803.jpg](../icml_results/315_Generalized%20Random%20Forests%20Using%20Fixed-Point%20Trees/tables/d766c680a34105016ca7b64813d81d30013c9b839a4ca19b5a4da3cf1e0c8803.jpg)

## Upweighting Easy Samples in Fine-Tuning Mitigates Forgetting


### Images

![33810b0cff6594167bf1e6ab4dbfb7067e8e0f920bfea8baae8385e05dfd72fc.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/images/33810b0cff6594167bf1e6ab4dbfb7067e8e0f920bfea8baae8385e05dfd72fc.jpg)

![58a62c769573dff2e82b445ebe53c62995f1ad98dda517c9cdbbea78bbd12fd2.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/images/58a62c769573dff2e82b445ebe53c62995f1ad98dda517c9cdbbea78bbd12fd2.jpg)

![bed9d6799bd101d8c893114adc39f569a5af7fc83e8c7107000bdac0fe2f004c.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/images/bed9d6799bd101d8c893114adc39f569a5af7fc83e8c7107000bdac0fe2f004c.jpg)

![cda2dd78036f173fd7a85f0a289b0312a2c286178badb803ba0eecba709251f4.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/images/cda2dd78036f173fd7a85f0a289b0312a2c286178badb803ba0eecba709251f4.jpg)

### Tables

![012c944332b156737a718e05922eacc682e697b7228b227cc67661eaab99051b.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/012c944332b156737a718e05922eacc682e697b7228b227cc67661eaab99051b.jpg)

![1cb6a608ff8129a2fa8034fdbb5b3fc383d65ece31012ec2fd11b232238fa99c.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/1cb6a608ff8129a2fa8034fdbb5b3fc383d65ece31012ec2fd11b232238fa99c.jpg)

![57382dbec5a30061622d596dd39eae8b8af4be739d79c79faabe9db194d5d6d1.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/57382dbec5a30061622d596dd39eae8b8af4be739d79c79faabe9db194d5d6d1.jpg)

![5925a279b1d0088b010896af3ec7ca3ae2a4e00263fbe893a990bb17013bc92b.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/5925a279b1d0088b010896af3ec7ca3ae2a4e00263fbe893a990bb17013bc92b.jpg)

![6cfcc34670a453f04d0d227ac7bd688ceaa3f0f69252109a7aeb3fbe6a35317a.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/6cfcc34670a453f04d0d227ac7bd688ceaa3f0f69252109a7aeb3fbe6a35317a.jpg)

![79f1f96e8f6f0ca705971325cc20c545211005e76187c97c232064a8b7336f6b.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/79f1f96e8f6f0ca705971325cc20c545211005e76187c97c232064a8b7336f6b.jpg)

![94b85419c8c825e174278c54721579d1fdfe116249918cca9201b792bf66d2ba.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/94b85419c8c825e174278c54721579d1fdfe116249918cca9201b792bf66d2ba.jpg)

![959d0c048cea1aa746e95fcfbf8f7646daef5a023c97b48684361dd01fcf8fc9.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/959d0c048cea1aa746e95fcfbf8f7646daef5a023c97b48684361dd01fcf8fc9.jpg)

![a1192880a64548f299d5b3dfa8c4341eab827c204a783067975b56e0fb8cc03b.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/a1192880a64548f299d5b3dfa8c4341eab827c204a783067975b56e0fb8cc03b.jpg)

![a8fb1166aaf559ba5d7b219efe194255de3478370a63a8d47a83a37d6e6cc222.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/a8fb1166aaf559ba5d7b219efe194255de3478370a63a8d47a83a37d6e6cc222.jpg)

![aafa2949b83a7b42c55d942c329e10595763242423b78d827f9a767acc90ca69.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/aafa2949b83a7b42c55d942c329e10595763242423b78d827f9a767acc90ca69.jpg)

![ba65141e6edf149ee91ff332b7fbe06d70d08d322256976794acaeda62e0f493.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/ba65141e6edf149ee91ff332b7fbe06d70d08d322256976794acaeda62e0f493.jpg)

![bb63663a6dd5572a3ac00f6dcbdcb7e6432a9dfb761b9f217c8d31df8c0b4d27.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/bb63663a6dd5572a3ac00f6dcbdcb7e6432a9dfb761b9f217c8d31df8c0b4d27.jpg)

![cb51e803d56cd61cc806db526ed0cde9ed6841999ab38695c4ecf108f8092b4b.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/cb51e803d56cd61cc806db526ed0cde9ed6841999ab38695c4ecf108f8092b4b.jpg)

![d6f9e540453ad8142387e466b4e6a9cd05395ac5d468da52b8d84fbb730a82ea.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/d6f9e540453ad8142387e466b4e6a9cd05395ac5d468da52b8d84fbb730a82ea.jpg)

![f3be7a72574d9f4b0c9c6483f8d949a75dfa6f4d912a75470610c319dae61dce.jpg](../icml_results/316_Upweighting%20Easy%20Samples%20in%20Fine-Tuning%20Mitigates%20Forgetting/tables/f3be7a72574d9f4b0c9c6483f8d949a75dfa6f4d912a75470610c319dae61dce.jpg)

## SAFE: Finding Sparse and Flat Minima to Improve Pruning


### Images

![46050bad2e075dd0ced308fb62dcf572884f872ae8c8e6de5f027e2c825f03b7.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/images/46050bad2e075dd0ced308fb62dcf572884f872ae8c8e6de5f027e2c825f03b7.jpg)

![524f6a6010e59e500a633ab5e47b5525a91371ccb90f83d0317a02768d5926a1.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/images/524f6a6010e59e500a633ab5e47b5525a91371ccb90f83d0317a02768d5926a1.jpg)

![7f7bb73cb84cff131f2481232112c158505c23f2d5eff729c5e7d41494f9987f.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/images/7f7bb73cb84cff131f2481232112c158505c23f2d5eff729c5e7d41494f9987f.jpg)

![9fe5a04a9479632c19bed2124e98419a51c650e3ca66144a04f4aa5acbfc214c.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/images/9fe5a04a9479632c19bed2124e98419a51c650e3ca66144a04f4aa5acbfc214c.jpg)

![cfd419dd4ed32952eb921efa809269e19eb2629ea8285498ef084f8cdf2d8089.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/images/cfd419dd4ed32952eb921efa809269e19eb2629ea8285498ef084f8cdf2d8089.jpg)

### Tables

![073cbe37149154a7effc66713a5720d9b6f1f3f409eba8efd525862c016c28d6.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/073cbe37149154a7effc66713a5720d9b6f1f3f409eba8efd525862c016c28d6.jpg)

![1079c45204229f1bb522d4dbf7556e1204c1ad5db6eb9a2cd154c4432183df94.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/1079c45204229f1bb522d4dbf7556e1204c1ad5db6eb9a2cd154c4432183df94.jpg)

![27cb0c059e2ecebe9c2497acd52dc9a4a4eb7206bd53f583cf95108d4b530613.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/27cb0c059e2ecebe9c2497acd52dc9a4a4eb7206bd53f583cf95108d4b530613.jpg)

![28c67303941d370c31eba2f533069d02f6843b9f572e3680c79836c72318cca1.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/28c67303941d370c31eba2f533069d02f6843b9f572e3680c79836c72318cca1.jpg)

![2ace5b3e368587d8319308f7820dd741e20e1124c74a58df2f138b61cc49ec8b.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/2ace5b3e368587d8319308f7820dd741e20e1124c74a58df2f138b61cc49ec8b.jpg)

![3b8f513b7233c4671428bfce241de551bef11ccf064fa789eb69193a6d9c6443.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/3b8f513b7233c4671428bfce241de551bef11ccf064fa789eb69193a6d9c6443.jpg)

![3d07e940278604cee105bbe05d26aad13abafb80b154a402008fe24dc3dd7dcd.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/3d07e940278604cee105bbe05d26aad13abafb80b154a402008fe24dc3dd7dcd.jpg)

![66945872168f24dbce5bb3ea30d8430a54a6cb2fcb55cdb9e11dfbf82d7601e1.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/66945872168f24dbce5bb3ea30d8430a54a6cb2fcb55cdb9e11dfbf82d7601e1.jpg)

![6dcf72b42fe731d0022c3290042ca1f1b1c8adbe0938cb16c14b6086d9d4a459.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/6dcf72b42fe731d0022c3290042ca1f1b1c8adbe0938cb16c14b6086d9d4a459.jpg)

![8c503aa8a1c4ff3a5f0b94dabe8490148d4264f0532e59fc47f649d5cb378fe1.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/8c503aa8a1c4ff3a5f0b94dabe8490148d4264f0532e59fc47f649d5cb378fe1.jpg)

![95b8e1ca3674489edd37abdb8b45d4d081fa526d65b156b1a9b9d8eaf9decafd.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/95b8e1ca3674489edd37abdb8b45d4d081fa526d65b156b1a9b9d8eaf9decafd.jpg)

![b22d2170f6cfe55ecfd63e4c6641a3adb0fef1c7d92db26b37c930ec4398ee36.jpg](../icml_results/317_SAFE_%20Finding%20Sparse%20and%20Flat%20Minima%20to%20Improve%20Pruning/tables/b22d2170f6cfe55ecfd63e4c6641a3adb0fef1c7d92db26b37c930ec4398ee36.jpg)

## Emergence and Effectiveness of Task Vectors in In-Context Learning: An Encoder Decoder Perspective


### Images

![09c0143f2dc2a6e38881804184542fd61e413e89a5bea341578b6c4ad16cc4dc.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/09c0143f2dc2a6e38881804184542fd61e413e89a5bea341578b6c4ad16cc4dc.jpg)

![0ba9df2357fcd26fe8bff9b48ea0ef89709f4bf16cb8f43b792a6b1aa92c44b5.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/0ba9df2357fcd26fe8bff9b48ea0ef89709f4bf16cb8f43b792a6b1aa92c44b5.jpg)

![0fee91a32502863b33fa742ea1c9a40f9cf0485c9cfe267857059551fbedeb87.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/0fee91a32502863b33fa742ea1c9a40f9cf0485c9cfe267857059551fbedeb87.jpg)

![241d3915140981cf33be56f04f72abf115542de42af39bb2e925a87bbb3e2843.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/241d3915140981cf33be56f04f72abf115542de42af39bb2e925a87bbb3e2843.jpg)

![25450b2ff2858d9fc2431fc81513659bd819248a9a793ca7f5b2e65e8d9aded9.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/25450b2ff2858d9fc2431fc81513659bd819248a9a793ca7f5b2e65e8d9aded9.jpg)

![263bccff7edfba4cff45bb652584424cef5bc999cfda2b7b3277f867cbc56eaf.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/263bccff7edfba4cff45bb652584424cef5bc999cfda2b7b3277f867cbc56eaf.jpg)

![26ff06cf840ef23bea6f09eea546a6b9bf74351df061d897917a69fd0d2aaca8.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/26ff06cf840ef23bea6f09eea546a6b9bf74351df061d897917a69fd0d2aaca8.jpg)

![27acbc47353112b4db31858dc3282f02dc6142a0b8a95c92b2c270459f81ba9d.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/27acbc47353112b4db31858dc3282f02dc6142a0b8a95c92b2c270459f81ba9d.jpg)

![31095f0b2021055b7a312f09c754428f469c27c29a715a69e7a2ec7fc1331295.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/31095f0b2021055b7a312f09c754428f469c27c29a715a69e7a2ec7fc1331295.jpg)

![3b53ca62fcd0eac7f94a3611e43f643179486610fab70f7f79cf708daea9b157.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/3b53ca62fcd0eac7f94a3611e43f643179486610fab70f7f79cf708daea9b157.jpg)

![484eaa7450a5b748a95f69c3fef97ed0a2ffc717cedc81e4401683d91e565793.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/484eaa7450a5b748a95f69c3fef97ed0a2ffc717cedc81e4401683d91e565793.jpg)

![506543778d9d0d617fd16727b0a53f448124e47c8716527b1bf93cbbaecfb29f.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/506543778d9d0d617fd16727b0a53f448124e47c8716527b1bf93cbbaecfb29f.jpg)

![566e8c1e6e88a320e5c704a2662ac59a7c3eba17afceea774c38a190c469e2c2.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/566e8c1e6e88a320e5c704a2662ac59a7c3eba17afceea774c38a190c469e2c2.jpg)

![5888d9ca6189902a204dbc93ff199430367645a92279b6cc905a0c1130fcfa2d.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/5888d9ca6189902a204dbc93ff199430367645a92279b6cc905a0c1130fcfa2d.jpg)

![591a90debcec17e265b94786483b95465bf9de1351b05424ab0d91a4070fffa6.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/591a90debcec17e265b94786483b95465bf9de1351b05424ab0d91a4070fffa6.jpg)

![5cde6de7d641d6830d40b8a8d1350262af90b547ac42fe86ce8fe02d358bcbee.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/5cde6de7d641d6830d40b8a8d1350262af90b547ac42fe86ce8fe02d358bcbee.jpg)

![6adb2d5a677063dc8545694d7f4ccc3dde5c3a9b555e7121a6e63ec5caa04a0a.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/6adb2d5a677063dc8545694d7f4ccc3dde5c3a9b555e7121a6e63ec5caa04a0a.jpg)

![8a8dacb64512c940ed699f95b47ce9c0ca7cbeb7afa578f1281d532cbeacafb1.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/8a8dacb64512c940ed699f95b47ce9c0ca7cbeb7afa578f1281d532cbeacafb1.jpg)

![8b50504ec4499bd50a713a21a5db221a7636b140c3f6d79ad067b64a0cc66e6d.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/8b50504ec4499bd50a713a21a5db221a7636b140c3f6d79ad067b64a0cc66e6d.jpg)

![8b75b3692460850464756d58fb536f16807a7bfc37a431a4429d41db408000d5.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/8b75b3692460850464756d58fb536f16807a7bfc37a431a4429d41db408000d5.jpg)

![94dadc5f3405e69caa0d6f6b27cc1958b673927cf97b7353a93de757faf9a157.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/94dadc5f3405e69caa0d6f6b27cc1958b673927cf97b7353a93de757faf9a157.jpg)

![a76b66db291d38dad731eac4074fe70f9812f74817ef1466a70341fc9f83cb89.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/a76b66db291d38dad731eac4074fe70f9812f74817ef1466a70341fc9f83cb89.jpg)

![c4a0bf052ea37deab1524b111206f605c99b9a9d1a42178a0186c0d16650a52b.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/c4a0bf052ea37deab1524b111206f605c99b9a9d1a42178a0186c0d16650a52b.jpg)

![cdbaad770e49df0e5bf1df78ab7de735488b3a3e23438d37be6543eec852582b.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/cdbaad770e49df0e5bf1df78ab7de735488b3a3e23438d37be6543eec852582b.jpg)

![cf06d7720de63dfecd37eb6989828961f060f42811a3444b1b2c5c4fc258d59a.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/cf06d7720de63dfecd37eb6989828961f060f42811a3444b1b2c5c4fc258d59a.jpg)

![d30a42a3b323f3b29c3609644f9755bbb48b362352045137381dec940e6d4df2.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/d30a42a3b323f3b29c3609644f9755bbb48b362352045137381dec940e6d4df2.jpg)

![e8f4edc4abf5fbee72fabbfd7a0a299a6633eb19cd36883d0a9b81d5e678ff7c.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/e8f4edc4abf5fbee72fabbfd7a0a299a6633eb19cd36883d0a9b81d5e678ff7c.jpg)

![eb592664ec7c17c8b992390f1331716eff80e56676d75f570a5149db9151eee4.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/eb592664ec7c17c8b992390f1331716eff80e56676d75f570a5149db9151eee4.jpg)

![f50018fc143e052405dcb84e6089ca06179711e9f20be9055246fa2f35f6c373.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/images/f50018fc143e052405dcb84e6089ca06179711e9f20be9055246fa2f35f6c373.jpg)

### Tables

![38d38e949b97f8eb72eb22bcc6512b2296e7c6c05070334b8ebb5046ea2b878d.jpg](../icml_results/318_Emergence%20and%20Effectiveness%20of%20Task%20Vectors%20in%20In-Context%20Learning_%20An%20Encoder%20Decoder%20Perspective/tables/38d38e949b97f8eb72eb22bcc6512b2296e7c6c05070334b8ebb5046ea2b878d.jpg)

## Taming Knowledge Conflicts in Language Models


### Images

![10f05bd9d62c25148d4a8710fe8c32dad0f89ca31b01cd5571097b1e8d1330ea.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/images/10f05bd9d62c25148d4a8710fe8c32dad0f89ca31b01cd5571097b1e8d1330ea.jpg)

![33931b1b1b80f1fc2797a06b1610a6a8cbbaaa4f4a93fe10e3cb7b5c412e6bf4.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/images/33931b1b1b80f1fc2797a06b1610a6a8cbbaaa4f4a93fe10e3cb7b5c412e6bf4.jpg)

![43696c803d685a574e8ba7f678ee9f465015fafb3a64643607757b5782e08425.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/images/43696c803d685a574e8ba7f678ee9f465015fafb3a64643607757b5782e08425.jpg)

![6492314d625048faff19c171e726c97725032a2dd0190e0ff2736c79aa8192c8.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/images/6492314d625048faff19c171e726c97725032a2dd0190e0ff2736c79aa8192c8.jpg)

![6496fe14b4eed8dad1eeb5f2f27681a28160d0a3dbd999f98a9df82bd4e7ad0f.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/images/6496fe14b4eed8dad1eeb5f2f27681a28160d0a3dbd999f98a9df82bd4e7ad0f.jpg)

![b201b98ff393a53eba455fe3573e825515b0e9c4ce6260a54469123fa359770e.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/images/b201b98ff393a53eba455fe3573e825515b0e9c4ce6260a54469123fa359770e.jpg)

![d1d7c72d90d858ccd8d18c7cd498675c28f96afdf792ee622be40ed4c6c6f3d0.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/images/d1d7c72d90d858ccd8d18c7cd498675c28f96afdf792ee622be40ed4c6c6f3d0.jpg)

### Tables

![1cce566d0235f12dc7a0cb59ad74f7d5cfe6f9788e967b7ddd3f842a9eef84be.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/1cce566d0235f12dc7a0cb59ad74f7d5cfe6f9788e967b7ddd3f842a9eef84be.jpg)

![23df91fd36e9deca99fdb617272fbab9c43a033d4aca4be8d28df35549632add.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/23df91fd36e9deca99fdb617272fbab9c43a033d4aca4be8d28df35549632add.jpg)

![2d8ebedede03c347077798ac2747beb2ae9aa98a0c66441dfa135e2c6c6713ae.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/2d8ebedede03c347077798ac2747beb2ae9aa98a0c66441dfa135e2c6c6713ae.jpg)

![5fef1dcc64fe6b7699083de7bb64f40eec69320321e44759c3ae090a65d12164.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/5fef1dcc64fe6b7699083de7bb64f40eec69320321e44759c3ae090a65d12164.jpg)

![624587fda19d8c3064c7461e21deab3a545c173fbc32d3a3afb601c50ce76462.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/624587fda19d8c3064c7461e21deab3a545c173fbc32d3a3afb601c50ce76462.jpg)

![6c215dc86f6892210d0d35832519570eed7d37df0881d7b3a8587fb9952b1f33.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/6c215dc86f6892210d0d35832519570eed7d37df0881d7b3a8587fb9952b1f33.jpg)

![ed60c53180ef22ba79c1827f436d5bde31fc3b909fdb6f531a2be98fa156edaa.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/ed60c53180ef22ba79c1827f436d5bde31fc3b909fdb6f531a2be98fa156edaa.jpg)

![edf30659194895fcca5a67790cabaddf17a14f6c12b1aee101a6e2767c6df5a2.jpg](../icml_results/319_Taming%20Knowledge%20Conflicts%20in%20Language%20Models/tables/edf30659194895fcca5a67790cabaddf17a14f6c12b1aee101a6e2767c6df5a2.jpg)

## Continuous Bayesian Model Selection for Multivariate Causal Discovery


### Images

![1fc02e963f9b33f6b6095ca86a4366d8fc83562aebb133fc5fc10f6ce5386f02.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/images/1fc02e963f9b33f6b6095ca86a4366d8fc83562aebb133fc5fc10f6ce5386f02.jpg)

![4596874e2d7e09d9fc1828e59d13e66f35de88c0f7a3325edcda1afbf0cbc56e.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/images/4596874e2d7e09d9fc1828e59d13e66f35de88c0f7a3325edcda1afbf0cbc56e.jpg)

![6c0da5f7222c4c3ecaef322b8420784999cd535ba2d607596c3935176df7ad22.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/images/6c0da5f7222c4c3ecaef322b8420784999cd535ba2d607596c3935176df7ad22.jpg)

![cac17cc6b28d2c736d9d51286c35514119d5d470f7158657fb814fdb46c9d183.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/images/cac17cc6b28d2c736d9d51286c35514119d5d470f7158657fb814fdb46c9d183.jpg)

![de4835cecc4a36318dfbf7dd047936fd966589df01a7c7670588efbfd26f47d6.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/images/de4835cecc4a36318dfbf7dd047936fd966589df01a7c7670588efbfd26f47d6.jpg)

![f9b8fe07f6987100322fe535a1ffd32fcbe85b6fb245dad758cf6fe182479bc8.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/images/f9b8fe07f6987100322fe535a1ffd32fcbe85b6fb245dad758cf6fe182479bc8.jpg)

![ff94155e28465ab61649ecaf19bd6399e64d6ccdb3d621ff5a4e649fa6c0ea23.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/images/ff94155e28465ab61649ecaf19bd6399e64d6ccdb3d621ff5a4e649fa6c0ea23.jpg)

### Tables

![6d41305e893de0e0dcd1232062caa020ded9c84143c35cff209f7e59a08e9131.jpg](../icml_results/320_Continuous%20Bayesian%20Model%20Selection%20for%20Multivariate%20Causal%20Discovery/tables/6d41305e893de0e0dcd1232062caa020ded9c84143c35cff209f7e59a08e9131.jpg)

## MP-Nav: Enhancing Data Poisoning Attacks against Multimodal Learning


### Images

![198f8c92375da17c3d91d978e95167961eadded5981fc3887aebac430c22d9cd.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/images/198f8c92375da17c3d91d978e95167961eadded5981fc3887aebac430c22d9cd.jpg)

![2a6bacaba22288197e252ae562f354f0aa83cd663aea1030116bafcbad51cf76.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/images/2a6bacaba22288197e252ae562f354f0aa83cd663aea1030116bafcbad51cf76.jpg)

![b9e231f998ce9da07a0714683063f3b5955c0e408bde80fae6d8a0cbafe3c673.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/images/b9e231f998ce9da07a0714683063f3b5955c0e408bde80fae6d8a0cbafe3c673.jpg)

![d33f41dea7dc46f8a1770a3dd08810e0a354596c9eedb1414683abc61e2eba88.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/images/d33f41dea7dc46f8a1770a3dd08810e0a354596c9eedb1414683abc61e2eba88.jpg)

![d995b98570bdb8320483856202890d44fb7c93bd0120ce8ad85f30e24b78d81b.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/images/d995b98570bdb8320483856202890d44fb7c93bd0120ce8ad85f30e24b78d81b.jpg)

![e675236ea092980dd8858bc7601779c018f0e2194766a9128d98d0fe511c94ab.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/images/e675236ea092980dd8858bc7601779c018f0e2194766a9128d98d0fe511c94ab.jpg)

### Tables

![12043c9175944758ecc569aeaeabd88e21f6fa3c0c1d93ef2914cff097006303.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/tables/12043c9175944758ecc569aeaeabd88e21f6fa3c0c1d93ef2914cff097006303.jpg)

![5a75f3c822aaf1da7875d520f11b48fc1a39c478f40e87762859887e9401b9ca.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/tables/5a75f3c822aaf1da7875d520f11b48fc1a39c478f40e87762859887e9401b9ca.jpg)

![6526a85fe37bdc246fd3bf7af60c090a63d234fc922e301f5cfc474ac6c35633.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/tables/6526a85fe37bdc246fd3bf7af60c090a63d234fc922e301f5cfc474ac6c35633.jpg)

![82cec11a485f817ca7e1847029128888ad828b5cf6eebe9c261b84fd963e4f17.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/tables/82cec11a485f817ca7e1847029128888ad828b5cf6eebe9c261b84fd963e4f17.jpg)

![d04ef26345596d9525c13567526841e0382645ddf885da2c9358631f75f60c73.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/tables/d04ef26345596d9525c13567526841e0382645ddf885da2c9358631f75f60c73.jpg)

![eff13363635defe14d60a2707d5205bab75fb33004e31d7aefa2151088d1fd1e.jpg](../icml_results/321_MP-Nav_%20Enhancing%20Data%20Poisoning%20Attacks%20against%20Multimodal%20Learning/tables/eff13363635defe14d60a2707d5205bab75fb33004e31d7aefa2151088d1fd1e.jpg)

## CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging


### Images

![3a3c106a5ed6b93a94108468c3ca62cd8541b4df90d332564ba75eb440f48cda.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/images/3a3c106a5ed6b93a94108468c3ca62cd8541b4df90d332564ba75eb440f48cda.jpg)

![48d6b4c24896cd42d464ab2a2b969bedd08bcf4262ff254455dc6f9347a40aaf.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/images/48d6b4c24896cd42d464ab2a2b969bedd08bcf4262ff254455dc6f9347a40aaf.jpg)

![4bd7696772ae3d3eb7fd5f539fe3e2bc754fa6a3a209f497d66d605c9a86a37b.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/images/4bd7696772ae3d3eb7fd5f539fe3e2bc754fa6a3a209f497d66d605c9a86a37b.jpg)

![8277aff04b34ebea96621c7ab74d52a5c8c11f1f1b8331fed6db529d1e8e3a12.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/images/8277aff04b34ebea96621c7ab74d52a5c8c11f1f1b8331fed6db529d1e8e3a12.jpg)

![c2de75dff8f2904052ae715a59aff1c14d1b1c557bdd36767a75da27bd2f724f.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/images/c2de75dff8f2904052ae715a59aff1c14d1b1c557bdd36767a75da27bd2f724f.jpg)

### Tables

![088b08ffbeb1dcfd08b32961d6ada3394f6937b1c51ac43cabc827ca4b544001.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/088b08ffbeb1dcfd08b32961d6ada3394f6937b1c51ac43cabc827ca4b544001.jpg)

![2135240387be70afaecf42d3c434fd094d14f7fd59c1f531749b53b549e992eb.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/2135240387be70afaecf42d3c434fd094d14f7fd59c1f531749b53b549e992eb.jpg)

![29b5a3781ff5df395a266c2512eb24cbe727e00ad2aad01ee7886761fbbf9c84.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/29b5a3781ff5df395a266c2512eb24cbe727e00ad2aad01ee7886761fbbf9c84.jpg)

![331ea22f9864870e0d6b68917b725c2559a4aba9d019c0752fe9c27956b52aab.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/331ea22f9864870e0d6b68917b725c2559a4aba9d019c0752fe9c27956b52aab.jpg)

![5745ce418ad88d5d67ba77c71d082eb951444da7313b7afa2ae48ca9070c2394.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/5745ce418ad88d5d67ba77c71d082eb951444da7313b7afa2ae48ca9070c2394.jpg)

![705c5a75a890b673e015c77d1e8daa0048e2e942c3f7fd502973d791dbc0af52.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/705c5a75a890b673e015c77d1e8daa0048e2e942c3f7fd502973d791dbc0af52.jpg)

![7f34ee6ed8f5d1453fb9587926b273162960ba395e408d756d96132477cadad4.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/7f34ee6ed8f5d1453fb9587926b273162960ba395e408d756d96132477cadad4.jpg)

![8372f3d03d8fc6887c1e63782a57a92222f968912290808990717779d06cd329.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/8372f3d03d8fc6887c1e63782a57a92222f968912290808990717779d06cd329.jpg)

![86e19336a5831ca836fecaafb783c047478c5df5ab9f0cd5c6370124ea869727.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/86e19336a5831ca836fecaafb783c047478c5df5ab9f0cd5c6370124ea869727.jpg)

![9f8e190c48c4a552de3d492b7fe7958f6d36aa682fb3268bdb44de08c0126317.jpg](../icml_results/322_CAT%20Merging_%20A%20Training-Free%20Approach%20for%20Resolving%20Conflicts%20in%20Model%20Merging/tables/9f8e190c48c4a552de3d492b7fe7958f6d36aa682fb3268bdb44de08c0126317.jpg)

## Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation


### Images

![08299d067b541d17db1da0f6062be09e105bfb2a4ea492b3f05a844d7e578e4d.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/08299d067b541d17db1da0f6062be09e105bfb2a4ea492b3f05a844d7e578e4d.jpg)

![1643407ecf1874e15a8b1634044758076ed19f3627a819727e1c385b5b7278b5.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/1643407ecf1874e15a8b1634044758076ed19f3627a819727e1c385b5b7278b5.jpg)

![191efc696c17947457ad7bc357ddcbf69f47df7a88e994d0be9e82178c075c5d.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/191efc696c17947457ad7bc357ddcbf69f47df7a88e994d0be9e82178c075c5d.jpg)

![2b631bad0ffeaed6ffb35708f20059e218dbec605a2769a583470299fc23c582.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/2b631bad0ffeaed6ffb35708f20059e218dbec605a2769a583470299fc23c582.jpg)

![32c26a4f78e719e7c815eb82eb61d8dca2a8c6ad1e4753b45f3b541c5c0e07ec.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/32c26a4f78e719e7c815eb82eb61d8dca2a8c6ad1e4753b45f3b541c5c0e07ec.jpg)

![465ae060113c5370ad94c626d54d93b1e129d94ad89d20f83fdb08ff83434a3f.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/465ae060113c5370ad94c626d54d93b1e129d94ad89d20f83fdb08ff83434a3f.jpg)

![496700c228713962cf9d769e8ac3bc0f63e279522a803ffa5ff3517bb37cc6b7.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/496700c228713962cf9d769e8ac3bc0f63e279522a803ffa5ff3517bb37cc6b7.jpg)

![59e2603947aaa2df8d087910d6cccdee58cf89e32efd9a7e0670e86fd357f461.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/59e2603947aaa2df8d087910d6cccdee58cf89e32efd9a7e0670e86fd357f461.jpg)

![5fca1cc57112490d16b26ad4e0e80d259493243b6ccb62674664105408b86e0f.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/5fca1cc57112490d16b26ad4e0e80d259493243b6ccb62674664105408b86e0f.jpg)

![6bf02de339863cd60ec7506317b6472896d9f31786a6cf8deb9efa02640a0f3e.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/6bf02de339863cd60ec7506317b6472896d9f31786a6cf8deb9efa02640a0f3e.jpg)

![6c276ae1dbcf68ad399d37cc6c036d8408b32039e5251629ee7d3577e9153e10.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/6c276ae1dbcf68ad399d37cc6c036d8408b32039e5251629ee7d3577e9153e10.jpg)

![9719471e36970cb6febda05e30777a0c1458dc90b4d54be047b2a867f92dcc03.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/9719471e36970cb6febda05e30777a0c1458dc90b4d54be047b2a867f92dcc03.jpg)

![9edd9a0bbb6c5f434a4d1848ce65abace2ce0c68c9588206cf894eedb4bd05d9.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/9edd9a0bbb6c5f434a4d1848ce65abace2ce0c68c9588206cf894eedb4bd05d9.jpg)

![b7a42d021eda7a865918b469af34cd0bd07c0224a82a5a05039361d09784aceb.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/b7a42d021eda7a865918b469af34cd0bd07c0224a82a5a05039361d09784aceb.jpg)

![bf22a0ca7106f7f66f9d968171b9725e584235c4efbb101a655be78674e9824a.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/bf22a0ca7106f7f66f9d968171b9725e584235c4efbb101a655be78674e9824a.jpg)

![c1f1701c07cd4e49ec88e74a78a96a43ca7ffcec2eee66006692f6e0d6a31713.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/c1f1701c07cd4e49ec88e74a78a96a43ca7ffcec2eee66006692f6e0d6a31713.jpg)

![cc8e2f6e08c795170b78df8a5942eb36157d77a549ec117a07f7bdf2e10cee5b.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/cc8e2f6e08c795170b78df8a5942eb36157d77a549ec117a07f7bdf2e10cee5b.jpg)

![cd853b1e003994905054fdd0a12eb4c48368bef1e82f942d5a5596f0c727a41b.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/cd853b1e003994905054fdd0a12eb4c48368bef1e82f942d5a5596f0c727a41b.jpg)

![cdee55e6f6320e9e92e807d66001a4e3624eaddbea9d81c61c41542253828e21.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/cdee55e6f6320e9e92e807d66001a4e3624eaddbea9d81c61c41542253828e21.jpg)

![d83e2e8906e04765fd0a910f0a235616a3e96629c19daf249006039517c26b01.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/d83e2e8906e04765fd0a910f0a235616a3e96629c19daf249006039517c26b01.jpg)

![dea43cd5b25fab00c750b5057aec079a4fdf0d7e7fe4a95cc6e65e768b8d55a6.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/dea43cd5b25fab00c750b5057aec079a4fdf0d7e7fe4a95cc6e65e768b8d55a6.jpg)

![e16b28e0c11b7fe15a77448cabf34f09936495eea1677767d6e43e2f1654035a.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/e16b28e0c11b7fe15a77448cabf34f09936495eea1677767d6e43e2f1654035a.jpg)

![e1e9fa398163ac5681d0166b44a3905cfe29dd52d4aec2c59b011acbd3240377.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/e1e9fa398163ac5681d0166b44a3905cfe29dd52d4aec2c59b011acbd3240377.jpg)

![e82b09e3c222c905fe567dca4100479630d61b16aba199e439f64d1ee39242a7.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/e82b09e3c222c905fe567dca4100479630d61b16aba199e439f64d1ee39242a7.jpg)

![e956e9b3c2cbe226e0acbaf7da6f711deb6e7217c9ff1e5b20c5325fb6b1335b.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/e956e9b3c2cbe226e0acbaf7da6f711deb6e7217c9ff1e5b20c5325fb6b1335b.jpg)

![ec39a84dc455f36b9c48309687bcf941442a5d1b0e1ca46b16a0b738741210fe.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/ec39a84dc455f36b9c48309687bcf941442a5d1b0e1ca46b16a0b738741210fe.jpg)

![fceb35195d9bcf82d5b16ca3a56943f33bb12dd1c7252f4e5afe1f3efefba376.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/images/fceb35195d9bcf82d5b16ca3a56943f33bb12dd1c7252f4e5afe1f3efefba376.jpg)

### Tables

![21774df4481d3a09044978a3a10a5f546bf533a9a51e7a7d0e085d93908e300b.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/21774df4481d3a09044978a3a10a5f546bf533a9a51e7a7d0e085d93908e300b.jpg)

![25032ee4d3a44a9637ddf64f1d04c4808304e9cff678fff558d16c1957abb4bb.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/25032ee4d3a44a9637ddf64f1d04c4808304e9cff678fff558d16c1957abb4bb.jpg)

![257e53736a2d4422968962b0c8e2231fbe5cacc11f30e8bc424b782a3f86ff9b.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/257e53736a2d4422968962b0c8e2231fbe5cacc11f30e8bc424b782a3f86ff9b.jpg)

![28160cd839e3353a24770c7d299dc0e284199c03207e94c36fc68903eba80775.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/28160cd839e3353a24770c7d299dc0e284199c03207e94c36fc68903eba80775.jpg)

![31b810780f770f314d80d2322fd70bc8ad64d3d0b183dfa0b8e4db054738685a.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/31b810780f770f314d80d2322fd70bc8ad64d3d0b183dfa0b8e4db054738685a.jpg)

![44c0c5f8d07580a2d445ba5328bb1a684a4cae891077ac7341f72cc09661a76c.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/44c0c5f8d07580a2d445ba5328bb1a684a4cae891077ac7341f72cc09661a76c.jpg)

![5f641e7cba3a1fe641b46177fa2c43e2545307c04accb073e63ab4b23ceb0245.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/5f641e7cba3a1fe641b46177fa2c43e2545307c04accb073e63ab4b23ceb0245.jpg)

![75fa70cb048f8f9bf86285d97e37edc7c16183498fe771fb7973a2d4e4c255d0.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/75fa70cb048f8f9bf86285d97e37edc7c16183498fe771fb7973a2d4e4c255d0.jpg)

![7baefc6ae202e5dbc5b53e38a1f39da885899da273a6b9519fee44d344a30bd8.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/7baefc6ae202e5dbc5b53e38a1f39da885899da273a6b9519fee44d344a30bd8.jpg)

![7f3c7548447f0049e88f0f83ebf4b49aa0590761198a6f3e69cb0d7a39941fc7.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/7f3c7548447f0049e88f0f83ebf4b49aa0590761198a6f3e69cb0d7a39941fc7.jpg)

![9376a7f1d35b978c13a98bb82c4cbf58db299a254f3330ed94c577ff06514edc.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/9376a7f1d35b978c13a98bb82c4cbf58db299a254f3330ed94c577ff06514edc.jpg)

![b0fd69264211b0ad0773db1ba38f6d8ea297d48bb200513f53dfdb3fb8048b05.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/b0fd69264211b0ad0773db1ba38f6d8ea297d48bb200513f53dfdb3fb8048b05.jpg)

![c4874faef2156a98c35ff83cf538c073921a35c50c0982ee3fc914875755e9d2.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/c4874faef2156a98c35ff83cf538c073921a35c50c0982ee3fc914875755e9d2.jpg)

![c5f285622298396ad0673866d532883f93fdf433301880ebc6b676d70b1754d3.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/c5f285622298396ad0673866d532883f93fdf433301880ebc6b676d70b1754d3.jpg)

![d651a1a6f1335cb8fa4e3e9e77429a99d7d3810dcb4400c387f1947ec7275609.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/d651a1a6f1335cb8fa4e3e9e77429a99d7d3810dcb4400c387f1947ec7275609.jpg)

![d894163bf715272476ccccd601b86c82bcceeccd0a54479c9fb19ef4036af81e.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/d894163bf715272476ccccd601b86c82bcceeccd0a54479c9fb19ef4036af81e.jpg)

![e2d959049069e7d5550822918b7804c162856aa6639045fc9da6492ebd902aca.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/e2d959049069e7d5550822918b7804c162856aa6639045fc9da6492ebd902aca.jpg)

![e3f7f0f181ef4f378355ed6603d52511a444ae3b3f6dc3d9a74b429c5520b956.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/e3f7f0f181ef4f378355ed6603d52511a444ae3b3f6dc3d9a74b429c5520b956.jpg)

![ec185e5f85bb6bb5683fee30af35fad251501ff9e6cbcb38d1420c43fb064e23.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/ec185e5f85bb6bb5683fee30af35fad251501ff9e6cbcb38d1420c43fb064e23.jpg)

![f64e562dfb77c9c856fcb4a459b2bc38a5606d701d97a4c0831f1be561acd707.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/f64e562dfb77c9c856fcb4a459b2bc38a5606d701d97a4c0831f1be561acd707.jpg)

![fcfbc1d092c2ab7b1e94ce21f067b4ded286edfb9fd668ee36572eae5714136d.jpg](../icml_results/323_Pretraining%20Generative%20Flow%20Networks%20with%20Inexpensive%20Rewards%20for%20Molecular%20Graph%20Generation/tables/fcfbc1d092c2ab7b1e94ce21f067b4ded286edfb9fd668ee36572eae5714136d.jpg)

## SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity


### Images

![0cec4c7d7d964550a14d35c4fea40fc6a8431a98279eac889e49b10c68b71728.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/0cec4c7d7d964550a14d35c4fea40fc6a8431a98279eac889e49b10c68b71728.jpg)

![39d7825a2c9677387e58c6bcd6ab1b9a8d68c57154f68fdbbd6737ee9837fb87.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/39d7825a2c9677387e58c6bcd6ab1b9a8d68c57154f68fdbbd6737ee9837fb87.jpg)

![44315a2a4fa6190b2f46907480b681e633a2bbb4d72a29518b90451ab73176f1.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/44315a2a4fa6190b2f46907480b681e633a2bbb4d72a29518b90451ab73176f1.jpg)

![464268853550a2242932fb9413e4aa11eea37d4d7a46806591572244bfc78128.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/464268853550a2242932fb9413e4aa11eea37d4d7a46806591572244bfc78128.jpg)

![4ccafd34af47d370dbab29e887ea2734c9480b0595c0a0cf9833487c1dcee31b.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/4ccafd34af47d370dbab29e887ea2734c9480b0595c0a0cf9833487c1dcee31b.jpg)

![6c5063712c96cfeb2cb0acf1a63e77ada8010d01745690d1be39710c652d2e0d.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/6c5063712c96cfeb2cb0acf1a63e77ada8010d01745690d1be39710c652d2e0d.jpg)

![9011570433667b4e846cf130ce07203114dabb5322be4402ae4e7c8f4484effc.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/9011570433667b4e846cf130ce07203114dabb5322be4402ae4e7c8f4484effc.jpg)

![c247fef0c4d2707ad1a8b80c6959c791bc8f82c0790db25253484f01e6b7c6dc.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/c247fef0c4d2707ad1a8b80c6959c791bc8f82c0790db25253484f01e6b7c6dc.jpg)

![c2a0b99911fc7af8c5db68ae5c501c5f01cdd66d97febc93c65d5476b01bced5.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/c2a0b99911fc7af8c5db68ae5c501c5f01cdd66d97febc93c65d5476b01bced5.jpg)

![e3c202f96ef17a3449fb38999d9747409cce8d27b2692447018f33918415d2b9.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/images/e3c202f96ef17a3449fb38999d9747409cce8d27b2692447018f33918415d2b9.jpg)

### Tables

![0d029a33b1577dcb58e65f6865302ec11fc36ddc701d7a3de628a9ce560b362a.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/0d029a33b1577dcb58e65f6865302ec11fc36ddc701d7a3de628a9ce560b362a.jpg)

![377bb2475861686ffa93d0633873bdf739ebd1d138b328a81ced9a5b647e7605.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/377bb2475861686ffa93d0633873bdf739ebd1d138b328a81ced9a5b647e7605.jpg)

![4336b802009eaec62d614789c3c78ca0bc561220d79a2ad5d5607b4bc2a34941.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/4336b802009eaec62d614789c3c78ca0bc561220d79a2ad5d5607b4bc2a34941.jpg)

![67510cf4dafad76a7ecdce77d45a24929888406e4ce907e75024db73d4c39144.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/67510cf4dafad76a7ecdce77d45a24929888406e4ce907e75024db73d4c39144.jpg)

![974fbe3b802865ace5563558ef40fef3cb9f7b0a2129df18f3be3b325f3f38e3.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/974fbe3b802865ace5563558ef40fef3cb9f7b0a2129df18f3be3b325f3f38e3.jpg)

![a02802f787aa97492d9bd3b15efb8dca9758f0a55e7bfba3f5670c06f994f267.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/a02802f787aa97492d9bd3b15efb8dca9758f0a55e7bfba3f5670c06f994f267.jpg)

![a4ee3724c6a15d5fc8351e22f42e184bf0fcc74107475398230a98cab9cde6dd.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/a4ee3724c6a15d5fc8351e22f42e184bf0fcc74107475398230a98cab9cde6dd.jpg)

![b38e9680712eb58f66741d50bc44543a533402cdd4821e31e0f8de339f859d02.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/b38e9680712eb58f66741d50bc44543a533402cdd4821e31e0f8de339f859d02.jpg)

![cfae257ca484ef0617fea10562fc28ef9427f9f46b84cfa8b15508237eee0692.jpg](../icml_results/324_SpikeVideoFormer_%20An%20Efficient%20Spike-Driven%20Video%20Transformer%20with%20Hamming%20Attention%20and%20%24_mathcal%7BO/tables/cfae257ca484ef0617fea10562fc28ef9427f9f46b84cfa8b15508237eee0692.jpg)

## Dequantified Diffusion-Schrödinger Bridge for Density Ratio Estimation


### Images

![0e97fcc5ad37eda7a31d8c7f656c15463d4f58902a73eec30d2721e45245c4ab.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/0e97fcc5ad37eda7a31d8c7f656c15463d4f58902a73eec30d2721e45245c4ab.jpg)

![14dcb1a1ec261b019b175ba4e1c2748af66ee97fc87485917d40d54b72443dbc.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/14dcb1a1ec261b019b175ba4e1c2748af66ee97fc87485917d40d54b72443dbc.jpg)

![1f9ec2a07c90e118c3c8b4f97612fc841a24d13f09e45236bd6699889b13cac6.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/1f9ec2a07c90e118c3c8b4f97612fc841a24d13f09e45236bd6699889b13cac6.jpg)

![30de82c5af602632c6d96c2aed6045587e42cbed2709c1cbdb5f9b8f4de744ee.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/30de82c5af602632c6d96c2aed6045587e42cbed2709c1cbdb5f9b8f4de744ee.jpg)

![3274724360f7b15c3a2c692471d3f42a338322c604768e66c81dcb646de82344.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/3274724360f7b15c3a2c692471d3f42a338322c604768e66c81dcb646de82344.jpg)

![3301d4b48094b1874e95c7994bf0c8df46e3638863c6ba92069eaa72a2dfcb0d.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/3301d4b48094b1874e95c7994bf0c8df46e3638863c6ba92069eaa72a2dfcb0d.jpg)

![33fed44d5a50db281e8e0c6a46e448ef122a468c506eca17f47c2ce3b21f02e4.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/33fed44d5a50db281e8e0c6a46e448ef122a468c506eca17f47c2ce3b21f02e4.jpg)

![53152b91b31dd5820a23a385ed3628380cb5cb0fcecf4ad3622ce2b07c98218b.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/53152b91b31dd5820a23a385ed3628380cb5cb0fcecf4ad3622ce2b07c98218b.jpg)

![55815dc9447559ea65dfff19a5e19eeb1aed6485d39cd06928ad3896922982ef.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/55815dc9447559ea65dfff19a5e19eeb1aed6485d39cd06928ad3896922982ef.jpg)

![6dad7c6719fd64fd649792b110aff9ef8d4a7b40cd0c453898f89cf768cbf18d.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/6dad7c6719fd64fd649792b110aff9ef8d4a7b40cd0c453898f89cf768cbf18d.jpg)

![6f2b747363fe05cc983ddd55d544c5c6581cc6a218e8b2c840a0399aecba8cfb.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/6f2b747363fe05cc983ddd55d544c5c6581cc6a218e8b2c840a0399aecba8cfb.jpg)

![75b153556b1f2ddc693e8bd2c14fc676031d47e2f63abd42446134aeccbc396f.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/75b153556b1f2ddc693e8bd2c14fc676031d47e2f63abd42446134aeccbc396f.jpg)

![8efe4b78a73e65a410b982f4ade3b7bec5e9d1f54cff76d96f2b0d0f0a049f54.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/8efe4b78a73e65a410b982f4ade3b7bec5e9d1f54cff76d96f2b0d0f0a049f54.jpg)

![910a67315c29487e693191acaec15e439af4209fb8210604f47462d125e39f56.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/910a67315c29487e693191acaec15e439af4209fb8210604f47462d125e39f56.jpg)

![94b53cc6654320632f13cbadb40ee9b87d4bb0af86848a9665c9c932d8af1fb4.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/94b53cc6654320632f13cbadb40ee9b87d4bb0af86848a9665c9c932d8af1fb4.jpg)

![a69f044fa6503bec28ced78e3bc35899bdec8138f0342c2266a199deb3ea5f2f.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/a69f044fa6503bec28ced78e3bc35899bdec8138f0342c2266a199deb3ea5f2f.jpg)

![bbe57a2bba7045960366378dfed7d6e96399c451470c4afb7513f6acd87ae8bf.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/bbe57a2bba7045960366378dfed7d6e96399c451470c4afb7513f6acd87ae8bf.jpg)

![bcd89862a564bf6515fb1313df126bf05bdba2c1e9b5fb97e6e2991d415bf1eb.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/bcd89862a564bf6515fb1313df126bf05bdba2c1e9b5fb97e6e2991d415bf1eb.jpg)

![cbbe5557eb82cda7307e01f8d5800a5648601f48b8769079fcfc49731292fbd6.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/cbbe5557eb82cda7307e01f8d5800a5648601f48b8769079fcfc49731292fbd6.jpg)

![d0aedb5ed849837bb7aa6631917c3a0b448eaafe005122d410549ba466a15e27.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/d0aedb5ed849837bb7aa6631917c3a0b448eaafe005122d410549ba466a15e27.jpg)

![d37c7eb2d609e81d7b3e9278ab3b46c49c838a91d2c9b1dad249b78f81807dc6.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/d37c7eb2d609e81d7b3e9278ab3b46c49c838a91d2c9b1dad249b78f81807dc6.jpg)

![d3c01cd114093f28a62dd0cae18d37d02065a548c04e0742dfbc7e9d0dfdd7e6.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/d3c01cd114093f28a62dd0cae18d37d02065a548c04e0742dfbc7e9d0dfdd7e6.jpg)

![efdb9840608ce543115012c50c8992d8b50737bcffd67545a3a8e7c55e9f21b7.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/efdb9840608ce543115012c50c8992d8b50737bcffd67545a3a8e7c55e9f21b7.jpg)

![f0367681e0aa44f04e1b4809b521cf4b86e214cc80e3b2d2880d049a07f90f31.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/images/f0367681e0aa44f04e1b4809b521cf4b86e214cc80e3b2d2880d049a07f90f31.jpg)

### Tables

![9508d6dcba9e083ece9136decb51949f8b8edfac86a87b67674705f7f737f8df.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/tables/9508d6dcba9e083ece9136decb51949f8b8edfac86a87b67674705f7f737f8df.jpg)

![c030ee86647937e17021ebc240326050da495bc3365389fc84fd928b6444445f.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/tables/c030ee86647937e17021ebc240326050da495bc3365389fc84fd928b6444445f.jpg)

![eb301a934fe7b48aa95f5abf03a38d22483aedac4d3ae0e3adc4281a202e8978.jpg](../icml_results/325_Dequantified%20Diffusion-Schr%C3%B6dinger%20Bridge%20for%20Density%20Ratio%20Estimation/tables/eb301a934fe7b48aa95f5abf03a38d22483aedac4d3ae0e3adc4281a202e8978.jpg)

## DAMA: Data- and Model-aware Alignment of Multi-modal LLMs


### Images

![33bad441328570a45ae71573a4911ab412e5ecb063ec528548380133914952a5.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/33bad441328570a45ae71573a4911ab412e5ecb063ec528548380133914952a5.jpg)

![5479d59b3342cbdc4fcad411c639d49d8774c23ec396f34cafa61b37c5f600d8.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/5479d59b3342cbdc4fcad411c639d49d8774c23ec396f34cafa61b37c5f600d8.jpg)

![563cfb72d1e65c537066590c61ef65065cf229a93fe7cac352004d259c54b383.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/563cfb72d1e65c537066590c61ef65065cf229a93fe7cac352004d259c54b383.jpg)

![60456ff2358e838e8156c18989c727f76a76fac285e4aa6fd46ec730e9295352.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/60456ff2358e838e8156c18989c727f76a76fac285e4aa6fd46ec730e9295352.jpg)

![67fabcc5de0dc6b7af5d48bb7c22c47d81a88a4a73d5d890651ba9252ae12b71.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/67fabcc5de0dc6b7af5d48bb7c22c47d81a88a4a73d5d890651ba9252ae12b71.jpg)

![7b609f5c96c2bbeb7c84cfc674a753f3c4295f225bf32369be54b0679f4cf66c.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/7b609f5c96c2bbeb7c84cfc674a753f3c4295f225bf32369be54b0679f4cf66c.jpg)

![98c968b72bdbfc12b6291fc00a9602eb212f58c2113272f01231fd4f25bc2589.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/98c968b72bdbfc12b6291fc00a9602eb212f58c2113272f01231fd4f25bc2589.jpg)

![9c56c3f273be6fb8e0cd910b875e83b6b1b3998cd90d4bd1a890fe7e48e89b7e.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/9c56c3f273be6fb8e0cd910b875e83b6b1b3998cd90d4bd1a890fe7e48e89b7e.jpg)

![ae17a9312dcfc4c731f4395f670cc7cedaa5a7bb9034bc484c1b45b676a9cbde.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/ae17a9312dcfc4c731f4395f670cc7cedaa5a7bb9034bc484c1b45b676a9cbde.jpg)

![c01347acac7e0e27d920f298ae71a15b43c6a8fee0fa6e9f370bc7cc4903c177.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/c01347acac7e0e27d920f298ae71a15b43c6a8fee0fa6e9f370bc7cc4903c177.jpg)

![df44abde4c2b5be3f22ed07559f8539056e5ec6ab1327361b941857f4ac5ba09.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/images/df44abde4c2b5be3f22ed07559f8539056e5ec6ab1327361b941857f4ac5ba09.jpg)

### Tables

![09eb79825cccfb81a340d9a5e4d0034c4b182add2314af68ae3743d196640caa.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/tables/09eb79825cccfb81a340d9a5e4d0034c4b182add2314af68ae3743d196640caa.jpg)

![31d1ef9ce6bb24f34d8bea9bdcdb5cb7aa5be0d46ea720ff3b627d2cff0c4368.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/tables/31d1ef9ce6bb24f34d8bea9bdcdb5cb7aa5be0d46ea720ff3b627d2cff0c4368.jpg)

![9a28ea477e53f4c2c108b46550fc52702ac9e27705442e3722e77e27c2ec2620.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/tables/9a28ea477e53f4c2c108b46550fc52702ac9e27705442e3722e77e27c2ec2620.jpg)

![bafdcd364bdc6dd5cfab8e25048e1608673b80b9ff1fc2ab26cff72496ea7eeb.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/tables/bafdcd364bdc6dd5cfab8e25048e1608673b80b9ff1fc2ab26cff72496ea7eeb.jpg)

![f721faaeba282b30171e038247efc16cb632cd41b92d67ddbf29d745de68a85a.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/tables/f721faaeba282b30171e038247efc16cb632cd41b92d67ddbf29d745de68a85a.jpg)

![fe1319185422c5b3f99bd59090daca5ade35f768f17d2bff6d54426d6809fb3e.jpg](../icml_results/326_DAMA_%20Data-%20and%20Model-aware%20Alignment%20of%20Multi-modal%20LLMs/tables/fe1319185422c5b3f99bd59090daca5ade35f768f17d2bff6d54426d6809fb3e.jpg)

## Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models


### Images

![0b48665facdfbeb2e1cea238d3a4606abc0ccdebae1e773022b2dbd3fdeb7def.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/0b48665facdfbeb2e1cea238d3a4606abc0ccdebae1e773022b2dbd3fdeb7def.jpg)

![0ee4bc5e20d01ee8bff491f45a20d994f5bea9baee92b8c74a77a150dbf6dc05.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/0ee4bc5e20d01ee8bff491f45a20d994f5bea9baee92b8c74a77a150dbf6dc05.jpg)

![1bd2af2da09be9b36dc88b539ea71f93d47e5c95a55b9314d79471f4acc7d6de.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/1bd2af2da09be9b36dc88b539ea71f93d47e5c95a55b9314d79471f4acc7d6de.jpg)

![289656424433b411f9f74fbb851a40713f9cc78abc1d8db3dd65ffd4d65993cc.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/289656424433b411f9f74fbb851a40713f9cc78abc1d8db3dd65ffd4d65993cc.jpg)

![29a237bdee09f69758f0eb1c325bb0311312bd1774e1536328ed1def05774736.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/29a237bdee09f69758f0eb1c325bb0311312bd1774e1536328ed1def05774736.jpg)

![2ff78990cb78e64563f7c1155e2dbaa901c8fcf95087aabd6e1e8327c74874bf.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/2ff78990cb78e64563f7c1155e2dbaa901c8fcf95087aabd6e1e8327c74874bf.jpg)

![302ade4c233dcc64a39a9615cd13afd30fa77fa77b2aac0c4b5633b6b7ec38ef.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/302ade4c233dcc64a39a9615cd13afd30fa77fa77b2aac0c4b5633b6b7ec38ef.jpg)

![3d4acf4e9775f96b8170a1b232b110c5aa45a536d651fc9beefb521ebb556d32.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/3d4acf4e9775f96b8170a1b232b110c5aa45a536d651fc9beefb521ebb556d32.jpg)

![6107657ebf28241ee2215cba48b9d06c803b8a4eeddb24ac4595ccc4e067f48a.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/6107657ebf28241ee2215cba48b9d06c803b8a4eeddb24ac4595ccc4e067f48a.jpg)

![9778bf481e6cd25d38958a32a3f2312343b1b8e507b6a29429c73f52c55226d0.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/9778bf481e6cd25d38958a32a3f2312343b1b8e507b6a29429c73f52c55226d0.jpg)

![a6dfd15972302d10e8589b17d1c1b07c8b29c958653d65ca2d33fe514b592e91.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/a6dfd15972302d10e8589b17d1c1b07c8b29c958653d65ca2d33fe514b592e91.jpg)

![a6e18c376aac5b159fdda7a91e249377c91a08ac5a67f1c0e5663e6c27a708ff.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/a6e18c376aac5b159fdda7a91e249377c91a08ac5a67f1c0e5663e6c27a708ff.jpg)

![ae2329f27ca62e2e43ca3b545572b8e35603c1fd3baf3ac743c0ec2dfb22e7c9.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/ae2329f27ca62e2e43ca3b545572b8e35603c1fd3baf3ac743c0ec2dfb22e7c9.jpg)

![cd7e6b65b7bf334faca63b4a8150e194b8c6a2e947ce066c181d90d278b8479b.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/cd7e6b65b7bf334faca63b4a8150e194b8c6a2e947ce066c181d90d278b8479b.jpg)

![d29f809d14e3597de541d22fbf219877cd09932ad1dc9d005a5bad2856731dbc.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/d29f809d14e3597de541d22fbf219877cd09932ad1dc9d005a5bad2856731dbc.jpg)

![d99d8d4912bb151b8b758446bab5d96509450697c3008c45b8f5dd981abc3e26.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/d99d8d4912bb151b8b758446bab5d96509450697c3008c45b8f5dd981abc3e26.jpg)

![e0d46b7379611a63b6a32800f16496934c77476824d3c1ba2d3a90027f31dbe9.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/e0d46b7379611a63b6a32800f16496934c77476824d3c1ba2d3a90027f31dbe9.jpg)

![eb09737590403014a848f2a00c534acfbba99b8c1b2adff8a47af5ceeb920a73.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/eb09737590403014a848f2a00c534acfbba99b8c1b2adff8a47af5ceeb920a73.jpg)

![f0be822f11317d6327f9ab8cb84f48e6d59fd5cbdc4112345bd3f725b9c34836.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/f0be822f11317d6327f9ab8cb84f48e6d59fd5cbdc4112345bd3f725b9c34836.jpg)

![f75a481c04369f4bde196a59434cd3242c9e900a462afd9c9b8d541ea50c3156.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/f75a481c04369f4bde196a59434cd3242c9e900a462afd9c9b8d541ea50c3156.jpg)

![faa3ceacca833e65e585ed775e285a9ac56345ca475d716e539a8e91b3597fc5.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/faa3ceacca833e65e585ed775e285a9ac56345ca475d716e539a8e91b3597fc5.jpg)

![fd80c06a3769a915df880c48415bc0f68cba9830eb8124069216381285ddcd7a.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/images/fd80c06a3769a915df880c48415bc0f68cba9830eb8124069216381285ddcd7a.jpg)

### Tables

![17960c5857c5199dbc79dde4e912b01432af73ed9a95f4fc6bc42565ec08044a.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/17960c5857c5199dbc79dde4e912b01432af73ed9a95f4fc6bc42565ec08044a.jpg)

![26bf4b0e90711ef628a6a4c99111f1a2e4331fdc2a7b97decaa304308d46d0c1.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/26bf4b0e90711ef628a6a4c99111f1a2e4331fdc2a7b97decaa304308d46d0c1.jpg)

![32b1855d709e6e2f67d6091f0d057479e3c49941aa4fd36975ba8c44c337f7dc.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/32b1855d709e6e2f67d6091f0d057479e3c49941aa4fd36975ba8c44c337f7dc.jpg)

![7f0eda56dbe6466426bd3403ff2ac263032f2334f3ce367507e63cf4e83ad63d.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/7f0eda56dbe6466426bd3403ff2ac263032f2334f3ce367507e63cf4e83ad63d.jpg)

![8e81c7e69265e1528bb130e0c5428ab3b8aa9cbaaf04f18ba499126ce7f3c9c8.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/8e81c7e69265e1528bb130e0c5428ab3b8aa9cbaaf04f18ba499126ce7f3c9c8.jpg)

![92f5a51c4ef9823fa4a16cf03e132f2b69d842ff7c121867a28129da89bf0876.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/92f5a51c4ef9823fa4a16cf03e132f2b69d842ff7c121867a28129da89bf0876.jpg)

![9df0a3dc739c3a367143f3d801d26867acc9b5551f03d6d165c5a0e1e3af9783.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/9df0a3dc739c3a367143f3d801d26867acc9b5551f03d6d165c5a0e1e3af9783.jpg)

![b8ea6b94689c479c429fbcdba3ca88873501bd989fa80d9f417e50100eccbd1f.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/b8ea6b94689c479c429fbcdba3ca88873501bd989fa80d9f417e50100eccbd1f.jpg)

![d2271d6bdd7808b5d19983f687da33f46528b09a9c9a5fb3c758ebc9192a2820.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/d2271d6bdd7808b5d19983f687da33f46528b09a9c9a5fb3c758ebc9192a2820.jpg)

![d47668873a3808d3f48f896c5f7e79c70354dc9143f76de1d1022672737dbed8.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/d47668873a3808d3f48f896c5f7e79c70354dc9143f76de1d1022672737dbed8.jpg)

![dd065cbdd12d4f192a1e3b9ef1451e6f3825d708014af27cb63ed9edfdc7c2f1.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/dd065cbdd12d4f192a1e3b9ef1451e6f3825d708014af27cb63ed9edfdc7c2f1.jpg)

![f6d62f7601e3fd4300679f72abfe2c1a1a168f79cf5c31df0bc0f0cffc6f7026.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/f6d62f7601e3fd4300679f72abfe2c1a1a168f79cf5c31df0bc0f0cffc6f7026.jpg)

![ffee229c76a714816aaec2282814c95b46a480e04d1eb6ea0665f4783ad95639.jpg](../icml_results/327_Visual%20Attention%20Never%20Fades_%20Selective%20Progressive%20Attention%20ReCalibration%20for%20Detailed%20Image%20Capti/tables/ffee229c76a714816aaec2282814c95b46a480e04d1eb6ea0665f4783ad95639.jpg)

## A Mathematical Framework for AI-Human Integration in Work


### Images

![01e7d309cc927c51d25dd4cd57c9662e20541366318c9ae231b6e6187ea5caaa.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/01e7d309cc927c51d25dd4cd57c9662e20541366318c9ae231b6e6187ea5caaa.jpg)

![044adfc7d13116e033f51c977d5f6d5b6bef71a277726314872b77bef60aed47.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/044adfc7d13116e033f51c977d5f6d5b6bef71a277726314872b77bef60aed47.jpg)

![0fd64dfbe44d005fcd23469972f8a5e7cb404fc1c963502df9df5f8318c2cd2f.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/0fd64dfbe44d005fcd23469972f8a5e7cb404fc1c963502df9df5f8318c2cd2f.jpg)

![1a6f961f1194c233316077f26e30e7fedebd4de0c43ccff09b9380e9cb0fddf7.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/1a6f961f1194c233316077f26e30e7fedebd4de0c43ccff09b9380e9cb0fddf7.jpg)

![21f39170ad17a5cb1ef20c51f02073c4f74d49a9a0584d8d1ee818a51d176ba7.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/21f39170ad17a5cb1ef20c51f02073c4f74d49a9a0584d8d1ee818a51d176ba7.jpg)

![26f3cc877c2a89ff314045036fbf3961d10d2d67f9c59121178d78cbc35ac5ae.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/26f3cc877c2a89ff314045036fbf3961d10d2d67f9c59121178d78cbc35ac5ae.jpg)

![3cccd3854820da257eb1151fe445d0a8f6d80a8d7979b39968717f0189f092d9.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/3cccd3854820da257eb1151fe445d0a8f6d80a8d7979b39968717f0189f092d9.jpg)

![5fa47b66e5b296a4d2254a2cba45e822db55b3bda69be68711c047d1c5441202.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/5fa47b66e5b296a4d2254a2cba45e822db55b3bda69be68711c047d1c5441202.jpg)

![619f04f4a860005d33381930afa1cd3b09d2cbe021f509e334a969ed8c92df75.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/619f04f4a860005d33381930afa1cd3b09d2cbe021f509e334a969ed8c92df75.jpg)

![61d26b0c4165b630251a9a0fb731d5947442be7c34b5ddc18118a1806edf4b3f.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/61d26b0c4165b630251a9a0fb731d5947442be7c34b5ddc18118a1806edf4b3f.jpg)

![681786c8b5c5e01666106e0dd7b961d10f3294e2f67a6c2c20556f42ee17170a.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/681786c8b5c5e01666106e0dd7b961d10f3294e2f67a6c2c20556f42ee17170a.jpg)

![695c09280049c5f977e679090eb451327a3a03bee972d0e83d2127597724c5e4.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/695c09280049c5f977e679090eb451327a3a03bee972d0e83d2127597724c5e4.jpg)

![6e2036af8ab30b5ba4e6c43eaabc5e218bd09a95cebc3e94a35f0725d3dc28ca.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/6e2036af8ab30b5ba4e6c43eaabc5e218bd09a95cebc3e94a35f0725d3dc28ca.jpg)

![772ce6c5900e6e10109b6787eb72327fa992e04abd2cca9f662bb9fa2f8ba05c.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/772ce6c5900e6e10109b6787eb72327fa992e04abd2cca9f662bb9fa2f8ba05c.jpg)

![776a0e8ed04edd05e48f358085e2009d3e3836ffe670569e930bfde74d55ad46.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/776a0e8ed04edd05e48f358085e2009d3e3836ffe670569e930bfde74d55ad46.jpg)

![8a6ae4dfe055ba2b02af28993c292610232c81c47033132c246f1589aec832d1.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/8a6ae4dfe055ba2b02af28993c292610232c81c47033132c246f1589aec832d1.jpg)

![8cc29e2c042cae6d25918191f322b3f140cdcaece965034eca38be5b9938e4d3.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/8cc29e2c042cae6d25918191f322b3f140cdcaece965034eca38be5b9938e4d3.jpg)

![955b5fe858e7398ec46b36d9add4ee2c44ea6ad1f0100352319bea6c7a447211.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/955b5fe858e7398ec46b36d9add4ee2c44ea6ad1f0100352319bea6c7a447211.jpg)

![966ab6c39504dc62adcee76aaf48eafba9b4fb00139f90dd1276359957469bef.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/966ab6c39504dc62adcee76aaf48eafba9b4fb00139f90dd1276359957469bef.jpg)

![96c937bafc620987bd18ba889f1909f756d6db7f4c233e7f9046273475233b98.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/96c937bafc620987bd18ba889f1909f756d6db7f4c233e7f9046273475233b98.jpg)

![b5e4198cf6e6529c5df1eacc4ed8d5a3987fc3ce2149741ea214ff11928e5ca8.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/b5e4198cf6e6529c5df1eacc4ed8d5a3987fc3ce2149741ea214ff11928e5ca8.jpg)

![ef67f526407ad5329533f487752008c76d7bc8d30e9035319a71be627de2fbc7.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/ef67f526407ad5329533f487752008c76d7bc8d30e9035319a71be627de2fbc7.jpg)

![f49108eeac68983e14d0334a1b46bcc96e9d48ea1ac6103525e1ce6e53320afa.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/images/f49108eeac68983e14d0334a1b46bcc96e9d48ea1ac6103525e1ce6e53320afa.jpg)

### Tables

![42638e1f819371135cf2420d02ef03eec4c2182b9a8ff66b55dbb35ff5b75470.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/tables/42638e1f819371135cf2420d02ef03eec4c2182b9a8ff66b55dbb35ff5b75470.jpg)

![48bafa75f8887d2cdebc38a71ddb27fc2cae3a9ad20e0ecaa11578a7bf995ac5.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/tables/48bafa75f8887d2cdebc38a71ddb27fc2cae3a9ad20e0ecaa11578a7bf995ac5.jpg)

![6299cd1172f4eba7a78d604185d5ddd282add9f2cb169285369e363655ae6d13.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/tables/6299cd1172f4eba7a78d604185d5ddd282add9f2cb169285369e363655ae6d13.jpg)

![dce2b756ff1eeaff584ba823ce29b5f93aa40a1c57c8b1fbd12d5ef4ac14e425.jpg](../icml_results/328_A%20Mathematical%20Framework%20for%20AI-Human%20Integration%20in%20Work/tables/dce2b756ff1eeaff584ba823ce29b5f93aa40a1c57c8b1fbd12d5ef4ac14e425.jpg)

## XAttnMark: Learning Robust Audio Watermarking with Cross-Attention


### Images

![003c040d74dfe36ed7453e76691790aa81e28eac14fe22735d8a2142d932f64a.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/003c040d74dfe36ed7453e76691790aa81e28eac14fe22735d8a2142d932f64a.jpg)

![074670bf6ffa7db5fbc9e8bf5044755b978033967c9504096b813487f44bf8aa.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/074670bf6ffa7db5fbc9e8bf5044755b978033967c9504096b813487f44bf8aa.jpg)

![0cae15023ff07284a0652d1a7c0287d5e95ee31a5c44f09ca8505313d61e5ea5.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/0cae15023ff07284a0652d1a7c0287d5e95ee31a5c44f09ca8505313d61e5ea5.jpg)

![119b71ad414997a6cde1652523684300ab2a3594dee083b41979dc52b54946b2.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/119b71ad414997a6cde1652523684300ab2a3594dee083b41979dc52b54946b2.jpg)

![3240e2966d0c388c4bffb5969c35835f7fbf78910a0eba3fbaf9508f1210dea7.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/3240e2966d0c388c4bffb5969c35835f7fbf78910a0eba3fbaf9508f1210dea7.jpg)

![43c3ae5fd1b3bded38d0e480d598bfeb82f76116e117beb98070ff52a0593b94.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/43c3ae5fd1b3bded38d0e480d598bfeb82f76116e117beb98070ff52a0593b94.jpg)

![5da899fbf3f6855c8bb9d4c75a327eafe9bab845272272f936219e8879a8def1.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/5da899fbf3f6855c8bb9d4c75a327eafe9bab845272272f936219e8879a8def1.jpg)

![67e97878fad384c8fb619cfed9481baef16794f723788005862cf55af2e4d147.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/67e97878fad384c8fb619cfed9481baef16794f723788005862cf55af2e4d147.jpg)

![6e53626a54de2bdf5c63b8f0f69b161d87a24d18fcc72b4dc6cb9e00b55816db.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/6e53626a54de2bdf5c63b8f0f69b161d87a24d18fcc72b4dc6cb9e00b55816db.jpg)

![7366310e920f9660156d75dfce55c114b83c0cd45826e801966788b696022eba.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/7366310e920f9660156d75dfce55c114b83c0cd45826e801966788b696022eba.jpg)

![7554929f23f24af0f7f3b8b9172184500c13c5497e0421d346f6ba610c71d5dc.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/7554929f23f24af0f7f3b8b9172184500c13c5497e0421d346f6ba610c71d5dc.jpg)

![7c5a52e2a09544d19fec7f5991944241d182c922bc487dd47bdff37627b9c12e.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/7c5a52e2a09544d19fec7f5991944241d182c922bc487dd47bdff37627b9c12e.jpg)

![811f68fe19ec083e599f664ca577d8b736e76b020f6e258c80c3a2ee7249d536.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/811f68fe19ec083e599f664ca577d8b736e76b020f6e258c80c3a2ee7249d536.jpg)

![87a77278a56d39a0e0c2866a6be224f3e3a2a70dc9643ba862de758540185477.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/87a77278a56d39a0e0c2866a6be224f3e3a2a70dc9643ba862de758540185477.jpg)

![a3ec5059162110d2527334925a5290aee382eac6ed4a0c1fd63012679049da7a.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/a3ec5059162110d2527334925a5290aee382eac6ed4a0c1fd63012679049da7a.jpg)

![acd5dbb73f5979d072586f625de99a21c3e4a974a117a8d907f5ff03dfe45ff5.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/acd5dbb73f5979d072586f625de99a21c3e4a974a117a8d907f5ff03dfe45ff5.jpg)

![cb94196ed44d04dae4a46652c28b9cfe0ec0dbdcfed4fd0e0eb85fc96034db95.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/cb94196ed44d04dae4a46652c28b9cfe0ec0dbdcfed4fd0e0eb85fc96034db95.jpg)

![cba0faf18b4914739420e5a9f7bbe12110093465a85a17d3e6b22897472f8f4b.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/images/cba0faf18b4914739420e5a9f7bbe12110093465a85a17d3e6b22897472f8f4b.jpg)

### Tables

![046f6180e25963aaffef740aebd90b1f919dbc3b00db7d80a3f3fbf8f62e88d2.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/046f6180e25963aaffef740aebd90b1f919dbc3b00db7d80a3f3fbf8f62e88d2.jpg)

![1d69d67a6078cc1bda725f6e31fd619528d71565c9e42b23e18f170145a1b54e.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/1d69d67a6078cc1bda725f6e31fd619528d71565c9e42b23e18f170145a1b54e.jpg)

![227dcb75330a68b3362fee1d69c4c85df597d889dff3eab6aec5cf4f266a96c3.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/227dcb75330a68b3362fee1d69c4c85df597d889dff3eab6aec5cf4f266a96c3.jpg)

![22d7f852330f02d58fc5aba01548dd4f83b7b740d4bfc40ec2b0ab0f619edd3a.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/22d7f852330f02d58fc5aba01548dd4f83b7b740d4bfc40ec2b0ab0f619edd3a.jpg)

![25944dd1a1301900a6c642a11360181bbaee5a258decc535631fb8d4ddc1a5e5.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/25944dd1a1301900a6c642a11360181bbaee5a258decc535631fb8d4ddc1a5e5.jpg)

![439cd54ddfa03ff67cfc6da1681b67983f2841e07ded92f80b3c53b836fa825d.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/439cd54ddfa03ff67cfc6da1681b67983f2841e07ded92f80b3c53b836fa825d.jpg)

![5f5175be0c64b0f0e27ab3752442dfbc30c08aa1ddc9daedd560d6b3e6eef770.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/5f5175be0c64b0f0e27ab3752442dfbc30c08aa1ddc9daedd560d6b3e6eef770.jpg)

![677257eeb7acf9a684a3f71e348dae8b43e0a7fb59a467a76637be2e68b73104.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/677257eeb7acf9a684a3f71e348dae8b43e0a7fb59a467a76637be2e68b73104.jpg)

![7657ed96c1aa9ca39d5faa0228f789057bfd7a2cd740cc0621e3404a0a382133.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/7657ed96c1aa9ca39d5faa0228f789057bfd7a2cd740cc0621e3404a0a382133.jpg)

![ad87f6940844c272462c0ffb5d009bebc9ba19bcff9fbda9e631720d3c11dc05.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/ad87f6940844c272462c0ffb5d009bebc9ba19bcff9fbda9e631720d3c11dc05.jpg)

![e360ff5d157d6cfd5843b5a031782c7db6a4e22fdbcc0154981d6c6c32767b6a.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/e360ff5d157d6cfd5843b5a031782c7db6a4e22fdbcc0154981d6c6c32767b6a.jpg)

![f52a28aba54ed274edd677fef3419e1e0b41bac1698842220c55f3818e81f4e6.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/f52a28aba54ed274edd677fef3419e1e0b41bac1698842220c55f3818e81f4e6.jpg)

![fba5ddea29074297b7ee98daa6132e0c716e89a4d14cac7a05d0af3df0ae0810.jpg](../icml_results/329_XAttnMark_%C2%A0Learning%C2%A0Robust%C2%A0Audio%C2%A0Watermarking%C2%A0with%C2%A0Cross-Attention/tables/fba5ddea29074297b7ee98daa6132e0c716e89a4d14cac7a05d0af3df0ae0810.jpg)

## Dimension-Free Adaptive Subgradient Methods with Frequent Directions


### Images

![29cef3a7119ced5e3d65c15af08f6a09577974a39ef23d5c28334ac105549674.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/images/29cef3a7119ced5e3d65c15af08f6a09577974a39ef23d5c28334ac105549674.jpg)

![2cd59fd9c0c5e03e332b88d3f254f060686fc58c9904386b5df3cf9c2e3ac2b0.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/images/2cd59fd9c0c5e03e332b88d3f254f060686fc58c9904386b5df3cf9c2e3ac2b0.jpg)

![7894904b8fd37965f217398805959591398f32b26271dd9af7b8aae82456f513.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/images/7894904b8fd37965f217398805959591398f32b26271dd9af7b8aae82456f513.jpg)

![8358c6b281017483f4b9be8be35b5ce720c77a4ba110e62a8fdb7a15ca3aae89.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/images/8358c6b281017483f4b9be8be35b5ce720c77a4ba110e62a8fdb7a15ca3aae89.jpg)

![8ec7e7d3d782a6814c48076bb4c3470676c9ba72271a503f91d2edc9d1a4aa06.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/images/8ec7e7d3d782a6814c48076bb4c3470676c9ba72271a503f91d2edc9d1a4aa06.jpg)

![b1e384d095c398dbb47d1b51a0bc60e427ac112c4eca03fd4bce173c4bc701c7.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/images/b1e384d095c398dbb47d1b51a0bc60e427ac112c4eca03fd4bce173c4bc701c7.jpg)

![fee58739e3230228c268a41609d5de4f7ea476d5144a1c40aa95316209feaba6.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/images/fee58739e3230228c268a41609d5de4f7ea476d5144a1c40aa95316209feaba6.jpg)

### Tables

![129ba8b3f3611126f48022d05f08cd8746d485a860172d814f88282db687249e.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/tables/129ba8b3f3611126f48022d05f08cd8746d485a860172d814f88282db687249e.jpg)

![2d0262c0a3076d679e0f04dd68bbf2a2584575dd72f53ad4a21483986586b8a4.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/tables/2d0262c0a3076d679e0f04dd68bbf2a2584575dd72f53ad4a21483986586b8a4.jpg)

![b18d64b58ce8367883de49b5930edccd78d866f093b12490ac04b9e5485e99e9.jpg](../icml_results/330_Dimension-Free%20Adaptive%20Subgradient%20Methods%20with%20Frequent%20Directions/tables/b18d64b58ce8367883de49b5930edccd78d866f093b12490ac04b9e5485e99e9.jpg)

## Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning


### Images

![36fed1f39f0d1d69ea47dc869b828f0fac9310d54522853680b797ac13a0b825.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/images/36fed1f39f0d1d69ea47dc869b828f0fac9310d54522853680b797ac13a0b825.jpg)

![426b5efb47cf9e7f46444bfc6afcbff7d85cddb8f306e98da186f0d6fbd20567.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/images/426b5efb47cf9e7f46444bfc6afcbff7d85cddb8f306e98da186f0d6fbd20567.jpg)

![4ff0bd92aff1023f75fc9be4671e6992ed846dc14e175c520c83730d2624c578.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/images/4ff0bd92aff1023f75fc9be4671e6992ed846dc14e175c520c83730d2624c578.jpg)

![57cec971857e55d71cd65502c001fcde088d5ebfb9ed0276ade7f8a098fbf36e.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/images/57cec971857e55d71cd65502c001fcde088d5ebfb9ed0276ade7f8a098fbf36e.jpg)

![59f4f5fd55469380a25aea5f5a4b6522ed2b287dcb3f243eeddd957a9681bf40.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/images/59f4f5fd55469380a25aea5f5a4b6522ed2b287dcb3f243eeddd957a9681bf40.jpg)

![77f5d705e424e71ef4e34211dab42bdb79b6eb915f6f35982ec1912743d8c8ba.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/images/77f5d705e424e71ef4e34211dab42bdb79b6eb915f6f35982ec1912743d8c8ba.jpg)

![d20207a325febca9fa0d5338c1090f326155d66df5fa66c799d6852b03d2c878.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/images/d20207a325febca9fa0d5338c1090f326155d66df5fa66c799d6852b03d2c878.jpg)

### Tables

![0689449a10fc2e920eacfaa439a54025fa0a4054292250225d7adf82bed13f84.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/0689449a10fc2e920eacfaa439a54025fa0a4054292250225d7adf82bed13f84.jpg)

![1b7dfba5ca99991864b22f33f9207275fb6e6373d2cbbf14a94cc5b6c5657604.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/1b7dfba5ca99991864b22f33f9207275fb6e6373d2cbbf14a94cc5b6c5657604.jpg)

![216fdedbcd28790034c4b37734aa85687aa58dd904e204bd77b9ad5166b80ed7.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/216fdedbcd28790034c4b37734aa85687aa58dd904e204bd77b9ad5166b80ed7.jpg)

![2b4be1e6157697d4102f1d1316bbe09c8c4381924d5af7a43f0760e2921e1423.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/2b4be1e6157697d4102f1d1316bbe09c8c4381924d5af7a43f0760e2921e1423.jpg)

![37c7932d49b0ad94b3510d53743f725de3209b856d49db5540013c553c89fda1.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/37c7932d49b0ad94b3510d53743f725de3209b856d49db5540013c553c89fda1.jpg)

![3cc1a3fda13c41335aa3f6b6cc76d9127ff2f221131ec2aef749a6a9f0f45da9.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/3cc1a3fda13c41335aa3f6b6cc76d9127ff2f221131ec2aef749a6a9f0f45da9.jpg)

![71bebae43b15e2c53568ae32fbb403d8d09977ff7602b76a662609062f4ad7f1.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/71bebae43b15e2c53568ae32fbb403d8d09977ff7602b76a662609062f4ad7f1.jpg)

![8b9b802eed1efaa2b255e1e0c9b30b6834d367bed8523bbe0310e7496e8b2e11.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/8b9b802eed1efaa2b255e1e0c9b30b6834d367bed8523bbe0310e7496e8b2e11.jpg)

![983be32e88a70453584cc6dda8966eb0f119c9f87ac1fd3e49ef7ff67d50c60f.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/983be32e88a70453584cc6dda8966eb0f119c9f87ac1fd3e49ef7ff67d50c60f.jpg)

![bc067438aa2a094782b78476ca4265d68ed9db71fd22505d8e5445161433a206.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/bc067438aa2a094782b78476ca4265d68ed9db71fd22505d8e5445161433a206.jpg)

![e81caf5a305dffa416bc904ab01dff19b7d39c5415d900ff2e11408a4ce56eaf.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/e81caf5a305dffa416bc904ab01dff19b7d39c5415d900ff2e11408a4ce56eaf.jpg)

![f50928ff52db0eb42355facfd9ae0f9d2b98dcef4c2f6056543bcb50f1f9fb23.jpg](../icml_results/331_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning/tables/f50928ff52db0eb42355facfd9ae0f9d2b98dcef4c2f6056543bcb50f1f9fb23.jpg)

## PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model

### Images

![2e38abb36355e75beaf5e880cc2e33938b31b443207f436c55925a83f777581e.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/images/2e38abb36355e75beaf5e880cc2e33938b31b443207f436c55925a83f777581e.jpg)

![399d367a03eadc0b400076cea47b6adc5f99165b838a47b55331a631f5b0b12b.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/images/399d367a03eadc0b400076cea47b6adc5f99165b838a47b55331a631f5b0b12b.jpg)

![3aed2778eff9a13c0a9b77540ade12fe6ee3065af83554cfcee5f2ebacc56c97.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/images/3aed2778eff9a13c0a9b77540ade12fe6ee3065af83554cfcee5f2ebacc56c97.jpg)

![f83ec7b00fa10e8483c1e5e29a943342171991aaca9ab466d8075b2a729a6be5.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/images/f83ec7b00fa10e8483c1e5e29a943342171991aaca9ab466d8075b2a729a6be5.jpg)

### Tables

![21536d94e1eda4a2295b099ce91202e680e22eb44a6929b3c0a0303d59019c7c.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/tables/21536d94e1eda4a2295b099ce91202e680e22eb44a6929b3c0a0303d59019c7c.jpg)

![76046d4c7505ff3290a8a561aa53db317a5b9e87a3a201d97b8ee1d0e4699675.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/tables/76046d4c7505ff3290a8a561aa53db317a5b9e87a3a201d97b8ee1d0e4699675.jpg)

![8ab9fa9a0426d3604efc1fce7f266ae3895c670fde0878605101ce3ed36081d0.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/tables/8ab9fa9a0426d3604efc1fce7f266ae3895c670fde0878605101ce3ed36081d0.jpg)

![9b1d3b3602bb3a592f6b67e15f0c3bbe247b6c5e1c4bd68ed9a43e79851349db.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/tables/9b1d3b3602bb3a592f6b67e15f0c3bbe247b6c5e1c4bd68ed9a43e79851349db.jpg)

![a73d9e95614d07e37eb906eb97880925bf531962ca58920ec055e0e7ebbe7bf5.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/tables/a73d9e95614d07e37eb906eb97880925bf531962ca58920ec055e0e7ebbe7bf5.jpg)

![cc0e73da96ace8a7dafe7b828ad04d41edb50a6d261091d08005dd1db0e09573.jpg](../icml_results/332_PARM_%20Multi-Objective%20Test-Time%20Alignment%20via%20Preference-Aware%20Autoregressive%20Reward%20Model/tables/cc0e73da96ace8a7dafe7b828ad04d41edb50a6d261091d08005dd1db0e09573.jpg)
