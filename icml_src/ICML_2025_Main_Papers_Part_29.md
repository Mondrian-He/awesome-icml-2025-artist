# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 29 of 100

## 目录 (Table of Contents)

1. [Diffusion Models are Secretly Exchangeable: Parallelizing DDPMs via Auto Speculation](#Diffusion-Models-are-Secretly-Exchangeable-Parallelizing-DDPMs-via-Auto-Speculation)
2. [Toward a Unified Theory of Gradient Descent under Generalized Smoothness](#Toward-a-Unified-Theory-of-Gradient-Descent-under-Generalized-Smoothness)
3. [Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models](#Ferret-Federated-Full-Parameter-Tuning-at-Scale-for-Large-Language-Models)
4. [Improving the Statistical Efficiency of Cross-Conformal Prediction](#Improving-the-Statistical-Efficiency-of-Cross-Conformal-Prediction)
5. [Towards Cost-Effective Reward Guided Text Generation](#Towards-Cost-Effective-Reward-Guided-Text-Generation)
6. [Pareto-Optimal Fronts for Benchmarking Symbolic Regression Algorithms](#Pareto-Optimal-Fronts-for-Benchmarking-Symbolic-Regression-Algorithms)
7. [Tensorized Multi-View Multi-Label Classification via Laplace Tensor Rank](#Tensorized-Multi-View-Multi-Label-Classification-via-Laplace-Tensor-Rank)
8. [PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable Reinforcement Learning](#PIGDreamer-Privileged-Information-Guided-World-Models-for-Safe-Partially-Observable-Reinforcement-Learning)
9. [Enhancing Logits Distillation with Plug&Play Kendall's  $\tau$ Ranking Loss](#Enhancing-Logits-Distillation-with-PlugPlay-Kendalls-tau-Ranking-Loss)
10. [Fusing Reward and Dueling Feedback in Stochastic Bandits](#Fusing-Reward-and-Dueling-Feedback-in-Stochastic-Bandits)
11. [CogMath: Assessing LLMs' Authentic Mathematical Ability from a Human Cognitive Perspective](#CogMath-Assessing-LLMs-Authentic-Mathematical-Ability-from-a-Human-Cognitive-Perspective)
12. [PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs](#PAK-UCB-Contextual-Bandit-An-Online-Learning-Approach-to-Prompt-Aware-Selection-of-Generative-Models-and-LLMs)
13. [Beyond Topological Self-Explainable GNNs: A Formal Explainability Perspective](#Beyond-Topological-Self-Explainable-GNNs-A-Formal-Explainability-Perspective)
14. [Primitive Vision: Improving Diagram Understanding in MLLMs](#Primitive-Vision-Improving-Diagram-Understanding-in-MLLMs)
15. [HYGMA: Hypergraph Coordination Networks with Dynamic Grouping for Multi-Agent Reinforcement Learning](#HYGMA-Hypergraph-Coordination-Networks-with-Dynamic-Grouping-for-Multi-Agent-Reinforcement-Learning)
16. [MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention](#MMInference-Accelerating-Pre-filling-for-Long-Context-Visual-Language-Models-via-Modality-Aware-Permutation-Sparse-Attention)
17. [Permutation Equivariant Neural Networks for Symmetric Tensors](#Permutation-Equivariant-Neural-Networks-for-Symmetric-Tensors)
18. [Fundamental Limits of Visual Autoregressive Transformers: Universal Approximation Abilities](#Fundamental-Limits-of-Visual-Autoregressive-Transformers-Universal-Approximation-Abilities)
19. [How Expressive are Knowledge Graph Foundation Models?](#How-Expressive-are-Knowledge-Graph-Foundation-Models)
20. [Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo](#Inverse-Problem-Sampling-in-Latent-Space-Using-Sequential-Monte-Carlo)
21. [Proto Successor Measure: Representing the Behavior Space of an RL Agent](#Proto-Successor-Measure-Representing-the-Behavior-Space-of-an-RL-Agent)
22. [RISE: Radius of Influence based Subgraph Extraction for 3D Molecular Graph Explanation](#RISE-Radius-of-Influence-based-Subgraph-Extraction-for-3D-Molecular-Graph-Explanation)
23. [AEQA-NAT : Adaptive End-to-end Quantization Alignment Training Framework for Non-autoregressive Machine Translation](#AEQA-NAT-Adaptive-End-to-end-Quantization-Alignment-Training-Framework-for-Non-autoregressive-Machine-Translation)
24. [Competing Bandits in Matching Markets via Super Stability](#Competing-Bandits-in-Matching-Markets-via-Super-Stability)
25. [Boosting Adversarial Robustness with CLAT: Criticality Leveraged Adversarial Training](#Boosting-Adversarial-Robustness-with-CLAT-Criticality-Leveraged-Adversarial-Training)
26. [Reasoning-as-Logic-Units: Scaling Test-Time Reasoning in Large Language Models Through Logic Unit Alignment](#Reasoning-as-Logic-Units-Scaling-Test-Time-Reasoning-in-Large-Language-Models-Through-Logic-Unit-Alignment)
27. [DEALing with Image Reconstruction: Deep Attentive Least Squares](#DEALing-with-Image-Reconstruction-Deep-Attentive-Least-Squares)
28. [Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners](#Boosting-Masked-ECG-Text-Auto-Encoders-as-Discriminative-Learners)
29. [Optimization for Neural Operators can Benefit from Width](#Optimization-for-Neural-Operators-can-Benefit-from-Width)
30. [TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer](#TTFSFormer-A-TTFS-based-Lossless-Conversion-of-Spiking-Transformer)
31. [Synthetic Text Generation for Training Large Language Models via Gradient Matching](#Synthetic-Text-Generation-for-Training-Large-Language-Models-via-Gradient-Matching)
32. [Learning from Loss Landscape: Generalizable Mixed-Precision Quantization via Adaptive Sharpness-Aware Gradient Aligning](#Learning-from-Loss-Landscape-Generalizable-Mixed-Precision-Quantization-via-Adaptive-Sharpness-Aware-Gradient-Aligning)
33. [Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning](#Exploring-Criteria-of-Loss-Reweighting-to-Enhance-LLM-Unlearning)

---


## Diffusion Models are Secretly Exchangeable: Parallelizing DDPMs via Auto Speculation

### Images

![0073227233278bc7f3fcd6c6173ec0871b10e9775e87ce1dabb62553dea5c4af.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/0073227233278bc7f3fcd6c6173ec0871b10e9775e87ce1dabb62553dea5c4af.jpg)

![01bc1c0bd51cd0c3ef9f2d154398d721690beef2962391023654250aa4daaafd.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/01bc1c0bd51cd0c3ef9f2d154398d721690beef2962391023654250aa4daaafd.jpg)

![0fbda1cf8ba0b85f6ba951ddfc71b7bc6f999b7f3a5bdafc113381a08fbc108c.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/0fbda1cf8ba0b85f6ba951ddfc71b7bc6f999b7f3a5bdafc113381a08fbc108c.jpg)

![3cfd624ad519aeee8083e345dff4edf8dd4bce38863ea4f5f6fd4d8192ae6954.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/3cfd624ad519aeee8083e345dff4edf8dd4bce38863ea4f5f6fd4d8192ae6954.jpg)

![3e6775ee8970492d45359c101e9c9bd8808e28edf655cfe0a627c01ce153925e.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/3e6775ee8970492d45359c101e9c9bd8808e28edf655cfe0a627c01ce153925e.jpg)

![9b2e613ff4cb0b9933d9148dcea90d873ffcb30262743d1301cec9af8572fdb0.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/9b2e613ff4cb0b9933d9148dcea90d873ffcb30262743d1301cec9af8572fdb0.jpg)

![f23eb2cc01751b3e6e723d9484ac46d784d051565a49f5259a1914666154693c.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/f23eb2cc01751b3e6e723d9484ac46d784d051565a49f5259a1914666154693c.jpg)

![f62a50fb5abe51cf2203860eefbace2538af780b1df530a222b3f60f140194ec.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/f62a50fb5abe51cf2203860eefbace2538af780b1df530a222b3f60f140194ec.jpg)

![fb17b1ac6714363924dfc23766e3bfd456fb2f795b28aff861fbaa17e592bb8a.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/images/fb17b1ac6714363924dfc23766e3bfd456fb2f795b28aff861fbaa17e592bb8a.jpg)

### Tables

![671496559bc5bb5c6ec6aebaefc58375d3a695c2de1e3405e0013f3c6f0c1d75.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/tables/671496559bc5bb5c6ec6aebaefc58375d3a695c2de1e3405e0013f3c6f0c1d75.jpg)

![6a7170427e6b590b18a71bc625ed0806266cbdfc36ba560910de6c9cba67200c.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/tables/6a7170427e6b590b18a71bc625ed0806266cbdfc36ba560910de6c9cba67200c.jpg)

![7278282853bed0ab3ef452bb6cbb72ef14abd185ba34f1983cb83bfb661542cd.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/tables/7278282853bed0ab3ef452bb6cbb72ef14abd185ba34f1983cb83bfb661542cd.jpg)

![e4ae58f37fbd18d832d424bb29c627def9ea8b399fa8c23895636e5dfd3bfb92.jpg](../icml_results/932_Hyper_%20Hyperparameter%20Robust%20Efficient%20Exploration%20in%20Reinforcement%20Learning/tables/e4ae58f37fbd18d832d424bb29c627def9ea8b399fa8c23895636e5dfd3bfb92.jpg)

## Diffusion Models are Secretly Exchangeable: Parallelizing DDPMs via Auto Speculation


### Images

![1c408ee14470c05dbe4325aad0b5e21a8c97acae00df46e36f375ab8806ed462.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/1c408ee14470c05dbe4325aad0b5e21a8c97acae00df46e36f375ab8806ed462.jpg)

![2eb3d3386bcb1ac44686ad62949568de16ba6d43c726c99b3716ab48af3db40a.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/2eb3d3386bcb1ac44686ad62949568de16ba6d43c726c99b3716ab48af3db40a.jpg)

![53c035dfbd88428c0843d4e59b3227cc88a868faec6fd0f17671c6aa923f11b9.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/53c035dfbd88428c0843d4e59b3227cc88a868faec6fd0f17671c6aa923f11b9.jpg)

![8554b80d0c4fcd116477fba5f855e7f3ab1918dd659a757bd69a5d6dbda277c6.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/8554b80d0c4fcd116477fba5f855e7f3ab1918dd659a757bd69a5d6dbda277c6.jpg)

![91a184bc5e078ea1325bfbf57cc5386a02ad81951a3807d06146c872aa0d0fe1.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/91a184bc5e078ea1325bfbf57cc5386a02ad81951a3807d06146c872aa0d0fe1.jpg)

![df200d637cad7aaf6229aef33447ba2e7c6189555833f4d8bebca60924f25ed4.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/df200d637cad7aaf6229aef33447ba2e7c6189555833f4d8bebca60924f25ed4.jpg)

![dfa236c997c8bbcb065690dbba195001b1a881511c29bd54c507dae37d43dad1.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/dfa236c997c8bbcb065690dbba195001b1a881511c29bd54c507dae37d43dad1.jpg)

![eb15677b22c9dbfead750b2de5aa135726ebb47d3c002f8e8ab3eec19140bba5.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/images/eb15677b22c9dbfead750b2de5aa135726ebb47d3c002f8e8ab3eec19140bba5.jpg)

### Tables

![02aa9bcb82fd0f2aebcf09f73aa7314d1a9de4af533035920224169925d9b178.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/tables/02aa9bcb82fd0f2aebcf09f73aa7314d1a9de4af533035920224169925d9b178.jpg)

![30e857b96c873c643f111a87d492424063a5589d6e7f14c6202d6f16a7fc4f00.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/tables/30e857b96c873c643f111a87d492424063a5589d6e7f14c6202d6f16a7fc4f00.jpg)

![3b199877466a0682a5cab9e2c4b0db6a9fef1501cfe355281fa37135675250ad.jpg](../icml_results/933_Diffusion%20Models%20are%20Secretly%20Exchangeable_%20Parallelizing%20DDPMs%20via%20Auto%20Speculation/tables/3b199877466a0682a5cab9e2c4b0db6a9fef1501cfe355281fa37135675250ad.jpg)

## Toward a Unified Theory of Gradient Descent under Generalized Smoothness


### Images

![5d6f345dc9cfc4a9de610b4faa71a76675dd03f2e05bbbc1c69fb5c6d57f6183.jpg](../icml_results/934_Toward%20a%20Unified%20Theory%20of%20Gradient%20Descent%20under%20Generalized%20Smoothness/images/5d6f345dc9cfc4a9de610b4faa71a76675dd03f2e05bbbc1c69fb5c6d57f6183.jpg)

![ea9fd8636389a38f6cb0857f22f1aa20348f22816ff1ee4ef3599ccc6bd3f2d0.jpg](../icml_results/934_Toward%20a%20Unified%20Theory%20of%20Gradient%20Descent%20under%20Generalized%20Smoothness/images/ea9fd8636389a38f6cb0857f22f1aa20348f22816ff1ee4ef3599ccc6bd3f2d0.jpg)

### Tables

![42caf17c643579f69e99447edb9ce370dd86b0367eb49cb827820f219592d3b9.jpg](../icml_results/934_Toward%20a%20Unified%20Theory%20of%20Gradient%20Descent%20under%20Generalized%20Smoothness/tables/42caf17c643579f69e99447edb9ce370dd86b0367eb49cb827820f219592d3b9.jpg)

![828656e3527f06a5bddcbf3720daa867d1fd8b4991625c2ec0ba1bf3decf4b9a.jpg](../icml_results/934_Toward%20a%20Unified%20Theory%20of%20Gradient%20Descent%20under%20Generalized%20Smoothness/tables/828656e3527f06a5bddcbf3720daa867d1fd8b4991625c2ec0ba1bf3decf4b9a.jpg)

![9245da7b245b5884a1fc280c7d0a6e1b97926d1146e611c6a060d9eaca114b7d.jpg](../icml_results/934_Toward%20a%20Unified%20Theory%20of%20Gradient%20Descent%20under%20Generalized%20Smoothness/tables/9245da7b245b5884a1fc280c7d0a6e1b97926d1146e611c6a060d9eaca114b7d.jpg)

## Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models


### Images

![27135559a4dc4fe09d477b9140d9970aae9768b1c6f40c9a5128e52f1426cb9b.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/27135559a4dc4fe09d477b9140d9970aae9768b1c6f40c9a5128e52f1426cb9b.jpg)

![3deca932edc285e79b608e7a3d45968ac9b41019143d26586f959172927885ad.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/3deca932edc285e79b608e7a3d45968ac9b41019143d26586f959172927885ad.jpg)

![473fa0661f3848d859e8d3a827546c35450a42c7483a83ff01d183f401f1d5f2.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/473fa0661f3848d859e8d3a827546c35450a42c7483a83ff01d183f401f1d5f2.jpg)

![5d43e246f5f7f6931b7904421ae3a0648532dd2a5ea22fdd48eb7cc6933dfe6e.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/5d43e246f5f7f6931b7904421ae3a0648532dd2a5ea22fdd48eb7cc6933dfe6e.jpg)

![73b643286b2d5403831b485d28c14352835da6799d0f535cfdf038101eda5f41.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/73b643286b2d5403831b485d28c14352835da6799d0f535cfdf038101eda5f41.jpg)

![ac7d5db68e7e69d8067b029b97623d046c209c1dfb3815f004be94fec570e79f.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/ac7d5db68e7e69d8067b029b97623d046c209c1dfb3815f004be94fec570e79f.jpg)

![bf9d0092cf19389e7b079885f25711e85003f0f055b832831cceb3e20e46cd60.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/bf9d0092cf19389e7b079885f25711e85003f0f055b832831cceb3e20e46cd60.jpg)

![c4892b6fdbb5b849812d735bbfaacf09c7f74f9769ac5d82a6b2d5544346046b.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/c4892b6fdbb5b849812d735bbfaacf09c7f74f9769ac5d82a6b2d5544346046b.jpg)

![d689bc23b9aab856810dcc53b4f8448324149364e26cfbcf5d1cd09557103df6.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/images/d689bc23b9aab856810dcc53b4f8448324149364e26cfbcf5d1cd09557103df6.jpg)

### Tables

![1e56e469255ffd07386952271e74ee8b6d558d863166e63edceae91aa611c058.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/1e56e469255ffd07386952271e74ee8b6d558d863166e63edceae91aa611c058.jpg)

![35d24a718fb9df693ed2cd19690bb3425fa2b8dfa6bc02163be564c6f03eb1a0.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/35d24a718fb9df693ed2cd19690bb3425fa2b8dfa6bc02163be564c6f03eb1a0.jpg)

![43ad1a8ba9a3406ba29548b53740de6fbd85b1099f390fd81aa111ca588aa146.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/43ad1a8ba9a3406ba29548b53740de6fbd85b1099f390fd81aa111ca588aa146.jpg)

![46c02bc9cca5e12940ad895db50dfcbca80b64a9e91804354bf1f062b4afb200.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/46c02bc9cca5e12940ad895db50dfcbca80b64a9e91804354bf1f062b4afb200.jpg)

![73f9937aa28a0136a9470ae1f0b803ae259fa1dd2a45b6bd7990cd1d9fb8a191.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/73f9937aa28a0136a9470ae1f0b803ae259fa1dd2a45b6bd7990cd1d9fb8a191.jpg)

![7ab5cf06d670ce1d323a59e4a389c6666a896df4dd3d92a598294049edab9af2.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/7ab5cf06d670ce1d323a59e4a389c6666a896df4dd3d92a598294049edab9af2.jpg)

![8f6e8536b01c6ac26b01224ce8a8359c66e1a609b29b068b523108d29cb7d012.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/8f6e8536b01c6ac26b01224ce8a8359c66e1a609b29b068b523108d29cb7d012.jpg)

![922991309e5c6549e4e4b72075940434619c419a8830fb37473b12e3eec4b379.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/922991309e5c6549e4e4b72075940434619c419a8830fb37473b12e3eec4b379.jpg)

![ad8d803190169073d7df63852abfb0275ecc6af713058c99c6295b131ea6d0f7.jpg](../icml_results/935_Ferret_%20Federated%20Full-Parameter%20Tuning%20at%20Scale%20for%20Large%20Language%20Models/tables/ad8d803190169073d7df63852abfb0275ecc6af713058c99c6295b131ea6d0f7.jpg)

## Improving the Statistical Efficiency of Cross-Conformal Prediction


### Images

![3b02bc79f687ccb30a09ad789c2cb313bc24153765c4b881d84e57cfd4ade5bb.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/images/3b02bc79f687ccb30a09ad789c2cb313bc24153765c4b881d84e57cfd4ade5bb.jpg)

![46fb473a1b1f783ec92fc8080b912f70b6964dc683ce44f94171ee843b6dd723.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/images/46fb473a1b1f783ec92fc8080b912f70b6964dc683ce44f94171ee843b6dd723.jpg)

![998bd5ffd05d150ff8e748b9a5a4b91c43e19f3eb6cd34fb53f98555fdd95d9f.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/images/998bd5ffd05d150ff8e748b9a5a4b91c43e19f3eb6cd34fb53f98555fdd95d9f.jpg)

![abf92dcd672b369c7756a77ecf31fbeaebe25f04766cdce4b4ab4edf3f43e47b.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/images/abf92dcd672b369c7756a77ecf31fbeaebe25f04766cdce4b4ab4edf3f43e47b.jpg)

![cc5bdf02754a241ec48c956279e1fcf3a64f753d07fc5ef139162564aee93d0d.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/images/cc5bdf02754a241ec48c956279e1fcf3a64f753d07fc5ef139162564aee93d0d.jpg)

![d3095b746483715ad053195667c00b6b0776ac05f23ed65768daa77eb4f09b4f.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/images/d3095b746483715ad053195667c00b6b0776ac05f23ed65768daa77eb4f09b4f.jpg)

### Tables

![12f47894f8a39febb23aad419dee6f0d8f02d53c23166d53a8c2b4f0e89ab0bd.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/12f47894f8a39febb23aad419dee6f0d8f02d53c23166d53a8c2b4f0e89ab0bd.jpg)

![20a9bf61ae789492c891f034c89e46bd6d30a8ca70d5728db6e706c1e010cda9.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/20a9bf61ae789492c891f034c89e46bd6d30a8ca70d5728db6e706c1e010cda9.jpg)

![361c85f6eeb6e5374447fb313cf6425b5ee1990a83f210ead9473d1ec40d4055.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/361c85f6eeb6e5374447fb313cf6425b5ee1990a83f210ead9473d1ec40d4055.jpg)

![3ec442d9076e2e5db01d6dfd67458bd4760f36e1e500836c0887c04d261dd18d.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/3ec442d9076e2e5db01d6dfd67458bd4760f36e1e500836c0887c04d261dd18d.jpg)

![6bb0975a2fef779ba257fe8de92aacc801ecea7f402f9a4b9400de5aa1d0f42f.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/6bb0975a2fef779ba257fe8de92aacc801ecea7f402f9a4b9400de5aa1d0f42f.jpg)

![7aa013a1bcbeb6b7610e1c8881e242386487e42df85a952fa29c8016710e8f0a.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/7aa013a1bcbeb6b7610e1c8881e242386487e42df85a952fa29c8016710e8f0a.jpg)

![82895629e91d76816fe4beb24fc055855f3320a68b3b4369e7f9f2b39a8bd5b3.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/82895629e91d76816fe4beb24fc055855f3320a68b3b4369e7f9f2b39a8bd5b3.jpg)

![892d67ef58f19556d8319fdb256a88c13962627c632cdaef119a80cedde9deb1.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/892d67ef58f19556d8319fdb256a88c13962627c632cdaef119a80cedde9deb1.jpg)

![99a99e6bba880269cb0e07e8cd7b24b25aaaa01a233fb9e2b18160ef2553760f.jpg](../icml_results/936_Improving%20the%20Statistical%20Efficiency%20of%20Cross-Conformal%20Prediction/tables/99a99e6bba880269cb0e07e8cd7b24b25aaaa01a233fb9e2b18160ef2553760f.jpg)

## Towards Cost-Effective Reward Guided Text Generation


### Images

![2b4189ed6b1d895cfe264c8dcf0b94ebcee4297c2ce68acaf79b167fe6096b7d.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/images/2b4189ed6b1d895cfe264c8dcf0b94ebcee4297c2ce68acaf79b167fe6096b7d.jpg)

![c21363703b369f9b52cfc464114fb0d4ea512cde6a2af078abe0ff119a35fefe.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/images/c21363703b369f9b52cfc464114fb0d4ea512cde6a2af078abe0ff119a35fefe.jpg)

### Tables

![09b09ba19542e167584dfbc42f57db22c5062191dc3b533223858d0e09852c13.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/09b09ba19542e167584dfbc42f57db22c5062191dc3b533223858d0e09852c13.jpg)

![1f1141cb1ea62c2cb01170631c0f7fcc80656a0a2c6efdd811099b59fdaabb8b.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/1f1141cb1ea62c2cb01170631c0f7fcc80656a0a2c6efdd811099b59fdaabb8b.jpg)

![2462724e405cb4c21c3dbc1f69f008dfd68577b9a7369f7476996716a2ba2743.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/2462724e405cb4c21c3dbc1f69f008dfd68577b9a7369f7476996716a2ba2743.jpg)

![2eb08a3c2ef988740ed1c790e421214dbd95bbb23b9ba51d7e93dd0cca669fd8.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/2eb08a3c2ef988740ed1c790e421214dbd95bbb23b9ba51d7e93dd0cca669fd8.jpg)

![3d83bf6a12c90180de45018af1157583e58655b50bfb2b2b59412e7f11e97b81.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/3d83bf6a12c90180de45018af1157583e58655b50bfb2b2b59412e7f11e97b81.jpg)

![43e35bdf96e79ab09e281197ce8ccc0fcc8f049ffbfde590582fa2fe6fc9ae34.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/43e35bdf96e79ab09e281197ce8ccc0fcc8f049ffbfde590582fa2fe6fc9ae34.jpg)

![8b1c77953fb80de867a27762dbcc89318d904a52caa30c5e032658fa1196409d.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/8b1c77953fb80de867a27762dbcc89318d904a52caa30c5e032658fa1196409d.jpg)

![91b185bcd6b1bae068fb884f7ed1891db13024bb95c3ca5fd53e67e1bc3bd0ba.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/91b185bcd6b1bae068fb884f7ed1891db13024bb95c3ca5fd53e67e1bc3bd0ba.jpg)

![9aa03d332fe8f4606e5f20de80ae0c4152d370f41188be7695c1983b992e6a10.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/9aa03d332fe8f4606e5f20de80ae0c4152d370f41188be7695c1983b992e6a10.jpg)

![b1bb936f94c290675030f7a39d00369ebb2d6cef999dc2f08a249f1d35d1455d.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/b1bb936f94c290675030f7a39d00369ebb2d6cef999dc2f08a249f1d35d1455d.jpg)

![c85b4e2699a7f560a799bd268c4da0f92ef1a3824603fade1b159a63b39b587d.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/c85b4e2699a7f560a799bd268c4da0f92ef1a3824603fade1b159a63b39b587d.jpg)

![ca0359d35faaea7a16a30b8d5cfe1481815784f0d2dabd48815522ae701990d2.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/ca0359d35faaea7a16a30b8d5cfe1481815784f0d2dabd48815522ae701990d2.jpg)

![d22fab63abeb8f380d3a0f80721273aa7c4c63a2713a0d75807e2f9ebfbb0b69.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/d22fab63abeb8f380d3a0f80721273aa7c4c63a2713a0d75807e2f9ebfbb0b69.jpg)

![e9a9c61b52631f42349c435e27280f73fcb1e2ac544c150944ac072b32d58987.jpg](../icml_results/937_Towards%20Cost-Effective%20Reward%20Guided%20Text%20Generation/tables/e9a9c61b52631f42349c435e27280f73fcb1e2ac544c150944ac072b32d58987.jpg)

## Pareto-Optimal Fronts for Benchmarking Symbolic Regression Algorithms


### Images

![2273d8c03ce5626102ed009f7cab580636be197d62f2b92e11e7d46cf67ee58e.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/images/2273d8c03ce5626102ed009f7cab580636be197d62f2b92e11e7d46cf67ee58e.jpg)

![22f1be0d8d704fc8e9fc59cf8bd23a21de59aee4f3f4e2be33b21e82aaa27bc5.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/images/22f1be0d8d704fc8e9fc59cf8bd23a21de59aee4f3f4e2be33b21e82aaa27bc5.jpg)

![3d9dfb2bc7da16680c5617b0700b1da0660d7485e20e7b2071df16cb32d84730.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/images/3d9dfb2bc7da16680c5617b0700b1da0660d7485e20e7b2071df16cb32d84730.jpg)

![7f53a3924d0d162bb9e3e1aa03aeca170082cc75f1aca68bfea1c7c482f83e38.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/images/7f53a3924d0d162bb9e3e1aa03aeca170082cc75f1aca68bfea1c7c482f83e38.jpg)

![82db91d4ad12232e696721485b4677e7920e2101f9ad6a92a4a0dc91d30299fb.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/images/82db91d4ad12232e696721485b4677e7920e2101f9ad6a92a4a0dc91d30299fb.jpg)

![bf1107dd82d86c10856489fa176c58431e9ec6ce745a58f799225a85c5a44d45.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/images/bf1107dd82d86c10856489fa176c58431e9ec6ce745a58f799225a85c5a44d45.jpg)

![fa570f164ade0967e2aa46e05194151eaa37867161d128fda0007b1766d4951d.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/images/fa570f164ade0967e2aa46e05194151eaa37867161d128fda0007b1766d4951d.jpg)

### Tables

![13e24b7337ed41b046a0f9f395689f6cb21f8985333afd027a0f91be206db4a9.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/tables/13e24b7337ed41b046a0f9f395689f6cb21f8985333afd027a0f91be206db4a9.jpg)

![21c26648a106e80b9e4a3b6dc2d014c1c281498724298bab07fcbe8eb184cd8a.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/tables/21c26648a106e80b9e4a3b6dc2d014c1c281498724298bab07fcbe8eb184cd8a.jpg)

![3a09bb3139c50c533ec475df60a9ef68e18349d4418107d333a629e6770a58cc.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/tables/3a09bb3139c50c533ec475df60a9ef68e18349d4418107d333a629e6770a58cc.jpg)

![5a3c2e6578d3f3327e90e2e4bfad31908371c6133ecb75b6d30d489af9e990a7.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/tables/5a3c2e6578d3f3327e90e2e4bfad31908371c6133ecb75b6d30d489af9e990a7.jpg)

![af04894790107504ca726f3ce62abc3216b34ed51f7cf687ad66524cca3dfe59.jpg](../icml_results/938_Pareto-Optimal%20Fronts%20for%20Benchmarking%20Symbolic%20Regression%20Algorithms/tables/af04894790107504ca726f3ce62abc3216b34ed51f7cf687ad66524cca3dfe59.jpg)

## Tensorized Multi-View Multi-Label Classification via Laplace Tensor Rank


### Images

![36a3fc850df65feeec3689a64e381282dde240c365f6a487656ea665d337d69d.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/images/36a3fc850df65feeec3689a64e381282dde240c365f6a487656ea665d337d69d.jpg)

![3cffc357f2ae62eab246d5899a52f5e890358829eb6ac2910ab221c79aa02036.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/images/3cffc357f2ae62eab246d5899a52f5e890358829eb6ac2910ab221c79aa02036.jpg)

![43c67c1ea229cca1a596590dab314895e93343ed324fc5718ad12760068544a5.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/images/43c67c1ea229cca1a596590dab314895e93343ed324fc5718ad12760068544a5.jpg)

![6f0f367acfc712018f9c8a67329844e15404930b29077ad05e9760cc19343d99.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/images/6f0f367acfc712018f9c8a67329844e15404930b29077ad05e9760cc19343d99.jpg)

![8327dbff42cdd2425cf2b91efca6fa9574a378f09f0fc468722af7d978044623.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/images/8327dbff42cdd2425cf2b91efca6fa9574a378f09f0fc468722af7d978044623.jpg)

![aa597ad387c38e57b15b71adf2e2677b723398efee05a2f5d90db88295a05906.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/images/aa597ad387c38e57b15b71adf2e2677b723398efee05a2f5d90db88295a05906.jpg)

![d14f42188f4db1213eea8263b6e5c93e7a155ca4dd8a8245e70c0ff7ea0f005c.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/images/d14f42188f4db1213eea8263b6e5c93e7a155ca4dd8a8245e70c0ff7ea0f005c.jpg)

### Tables

![022a6cefe533ca0804ea9983e5aa0ae6b03d47307bf03a04e817c64487a16600.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/tables/022a6cefe533ca0804ea9983e5aa0ae6b03d47307bf03a04e817c64487a16600.jpg)

![4adb4840a2191011bdc9c40a03ab002d00f123f96573977e2b580fae7f141d90.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/tables/4adb4840a2191011bdc9c40a03ab002d00f123f96573977e2b580fae7f141d90.jpg)

![6dbdea7b95747e142b7cb5e9cc4d77ea3f530ff840135232da997776bbfb0b8d.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/tables/6dbdea7b95747e142b7cb5e9cc4d77ea3f530ff840135232da997776bbfb0b8d.jpg)

![774f4a5eeff7d7f28b4a45c3d04a9b97f0c6c34f71f63201793988866da6ced5.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/tables/774f4a5eeff7d7f28b4a45c3d04a9b97f0c6c34f71f63201793988866da6ced5.jpg)

![afd8f55a82b3e340c9b4507e65b861997dffe99b39e396b4b76683c228c1f9fc.jpg](../icml_results/939_Tensorized%20Multi-View%20Multi-Label%20Classification%20via%20Laplace%20Tensor%20Rank/tables/afd8f55a82b3e340c9b4507e65b861997dffe99b39e396b4b76683c228c1f9fc.jpg)

## PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable Reinforcement Learning


### Images

![199665ef79ef200eeea7f77cbab2ce456762c71dacb5cf5781b8305fb77b2731.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/199665ef79ef200eeea7f77cbab2ce456762c71dacb5cf5781b8305fb77b2731.jpg)

![2a8b430c985cfd9cf9a42e10eb3a6bcda35ffdd5310940e31a0415f382cedf79.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/2a8b430c985cfd9cf9a42e10eb3a6bcda35ffdd5310940e31a0415f382cedf79.jpg)

![559ce2d5db6c31eb1a73b2dbd5fcf0e2452a2010381f798b5332f4d63ab544e5.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/559ce2d5db6c31eb1a73b2dbd5fcf0e2452a2010381f798b5332f4d63ab544e5.jpg)

![5c66208c9b9941639cc327dc74932b5840156e6ba3bde87cbd91ff8a31a803b6.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/5c66208c9b9941639cc327dc74932b5840156e6ba3bde87cbd91ff8a31a803b6.jpg)

![64ab9b5497b76ffbf80f3651015814ac31cde77c78cce5de4403d8faf0f06c11.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/64ab9b5497b76ffbf80f3651015814ac31cde77c78cce5de4403d8faf0f06c11.jpg)

![67ae87dcb04b35ac911162e2a05eab992e91f7ed34317ebe44d7044e73b90d55.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/67ae87dcb04b35ac911162e2a05eab992e91f7ed34317ebe44d7044e73b90d55.jpg)

![68f5b3aecf9a8fb0d2a6f65ff74d7e87feb78f96b9334c7fd7700aa1cf1ce3c7.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/68f5b3aecf9a8fb0d2a6f65ff74d7e87feb78f96b9334c7fd7700aa1cf1ce3c7.jpg)

![b24256e63574664be0073d0c703713457e3922ced3fe7cc3ec1c1b80ce255c6a.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/b24256e63574664be0073d0c703713457e3922ced3fe7cc3ec1c1b80ce255c6a.jpg)

![c5b3758f542e325eb1f64281b9196a00a0a11eaff2b49fa65f4bdda2e78034e2.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/c5b3758f542e325eb1f64281b9196a00a0a11eaff2b49fa65f4bdda2e78034e2.jpg)

![e0b43691892a5755fd17361e16947a61bffabae6836d236007530c863ab9632a.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/images/e0b43691892a5755fd17361e16947a61bffabae6836d236007530c863ab9632a.jpg)

### Tables

![01ff3cba7680cee91df40a45f4e460ae41c75ff058117c9734bf850b9ac97cb8.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/01ff3cba7680cee91df40a45f4e460ae41c75ff058117c9734bf850b9ac97cb8.jpg)

![0d8467301c6dda54b8764cee55f4307f483f226a3269d7b14809e594b4c481da.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/0d8467301c6dda54b8764cee55f4307f483f226a3269d7b14809e594b4c481da.jpg)

![123b04b9be6dc47704376b12e163955deca648abb46f55dc8a3975e7d8f2ea9e.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/123b04b9be6dc47704376b12e163955deca648abb46f55dc8a3975e7d8f2ea9e.jpg)

![2fce8f2b089851d223505bc4d09c06aa9589ebab0497e871601128d6b5715d27.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/2fce8f2b089851d223505bc4d09c06aa9589ebab0497e871601128d6b5715d27.jpg)

![4f07de9de957e00c94d6f92da2169995db429021593a11d363d0137386d3d85f.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/4f07de9de957e00c94d6f92da2169995db429021593a11d363d0137386d3d85f.jpg)

![650837f220fa5d07b7c879a977a2da107c00f10ad4137cc0a6b4b911924de63d.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/650837f220fa5d07b7c879a977a2da107c00f10ad4137cc0a6b4b911924de63d.jpg)

![bf4c08415ec199e97cba7e8538be122a5383a6bd85566e0a4c452528496834d1.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/bf4c08415ec199e97cba7e8538be122a5383a6bd85566e0a4c452528496834d1.jpg)

![cc1391fb26a8c43a61428036e31f98e7c96770477898a0f2f2bcfbaea4db441e.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/cc1391fb26a8c43a61428036e31f98e7c96770477898a0f2f2bcfbaea4db441e.jpg)

![ed21c416fb3d075144f66132c8ee954091a81edeaabd2dbe3fa97fc031de7c84.jpg](../icml_results/940_PIGDreamer_%20Privileged%20Information%20Guided%20World%20Models%20for%20Safe%20Partially%20Observable%20Reinforcement%20L/tables/ed21c416fb3d075144f66132c8ee954091a81edeaabd2dbe3fa97fc031de7c84.jpg)

## Enhancing Logits Distillation with Plug&Play Kendall's  $\tau$ Ranking Loss


### Images

![15c30ce7e218bf3492a4e3327990b1e15275da96259f1dde0e8be0cbdeebc259.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/15c30ce7e218bf3492a4e3327990b1e15275da96259f1dde0e8be0cbdeebc259.jpg)

![37adc1b5d98ff65d7cbc0714818df732a57ef98b947246c6865a043475545525.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/37adc1b5d98ff65d7cbc0714818df732a57ef98b947246c6865a043475545525.jpg)

![995bacf0b57deb84b6e16c26ce897795cacc5587e1df4aea1f77a1295feb63a2.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/995bacf0b57deb84b6e16c26ce897795cacc5587e1df4aea1f77a1295feb63a2.jpg)

![9ba621ebf628cf0cb51386edb7353468528ed07615fe11a46e80d7647babb8fb.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/9ba621ebf628cf0cb51386edb7353468528ed07615fe11a46e80d7647babb8fb.jpg)

![9fa8b66fbd38601d34082dd6fb664fc35609e3dfc6d8ea8fe1d7fbea78ec6d4b.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/9fa8b66fbd38601d34082dd6fb664fc35609e3dfc6d8ea8fe1d7fbea78ec6d4b.jpg)

![b2896fdb62e9e3f838ac2a4f34d3a5d934419c97ba5f4a22da46ee9160b162c1.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/b2896fdb62e9e3f838ac2a4f34d3a5d934419c97ba5f4a22da46ee9160b162c1.jpg)

![e9c8e02188452286a308f6cb7bd50653e2406f807178999583254c1e0670e605.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/e9c8e02188452286a308f6cb7bd50653e2406f807178999583254c1e0670e605.jpg)

![fb97b9f7537ef30253c3bcf4c2eee65e58dc09e7481521385e3f906ade59492e.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/fb97b9f7537ef30253c3bcf4c2eee65e58dc09e7481521385e3f906ade59492e.jpg)

![fe84db74bde9e572a8726053323ebb323986e416260db0871be17ea1f45bba9b.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/images/fe84db74bde9e572a8726053323ebb323986e416260db0871be17ea1f45bba9b.jpg)

### Tables

![15231050de27331c23c3a2b8891d722655cef363d04cad21669d30f711ed0a3d.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/15231050de27331c23c3a2b8891d722655cef363d04cad21669d30f711ed0a3d.jpg)

![2210c4c90712fb207bf57adb5ffa060e621c924867f96a97358888869a484c79.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/2210c4c90712fb207bf57adb5ffa060e621c924867f96a97358888869a484c79.jpg)

![2d76835edcbc3a0ea35b5931b9e81f75e36a6820ec99f3045faa1516b6f20ca2.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/2d76835edcbc3a0ea35b5931b9e81f75e36a6820ec99f3045faa1516b6f20ca2.jpg)

![3b811a8cf1aaa64d859fad2f331d68c30471404b79059066454991a89f9934fc.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/3b811a8cf1aaa64d859fad2f331d68c30471404b79059066454991a89f9934fc.jpg)

![58d3d9070cf6b1772018c2c32b104794ffa0b67a2569bf8f71ba34fafceaeeab.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/58d3d9070cf6b1772018c2c32b104794ffa0b67a2569bf8f71ba34fafceaeeab.jpg)

![a2334dc38eb8240c15ef5b1ae09a0f4a40ee03764139cb2e5c616a0c67db22a6.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/a2334dc38eb8240c15ef5b1ae09a0f4a40ee03764139cb2e5c616a0c67db22a6.jpg)

![b643556d0377a96cb64619c78a348960d602a4209168f5c12302b435d2d125d8.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/b643556d0377a96cb64619c78a348960d602a4209168f5c12302b435d2d125d8.jpg)

![c38bb341017a905d5027ca6bcb129efde957b3e48516ba25b0e867dbbcd5e64e.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/c38bb341017a905d5027ca6bcb129efde957b3e48516ba25b0e867dbbcd5e64e.jpg)

![d813ebdcb3e859eda3bb9f3c73ad098346171dc666317fca4fc029524664ecb2.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/d813ebdcb3e859eda3bb9f3c73ad098346171dc666317fca4fc029524664ecb2.jpg)

![e04f9daa0da215c25d88e42a0b2b2114b5ffcc49ae305000850e53a6477f2686.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/e04f9daa0da215c25d88e42a0b2b2114b5ffcc49ae305000850e53a6477f2686.jpg)

![e62f523ca2b93ec36c8bddcb99a130694ae9c010799b065a2210ae6f95fe09c4.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/e62f523ca2b93ec36c8bddcb99a130694ae9c010799b065a2210ae6f95fe09c4.jpg)

![f5d33b7a03159404e2aa76b25fd576e5c6effaa69fa4f0e7523bffbed762e7a8.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/f5d33b7a03159404e2aa76b25fd576e5c6effaa69fa4f0e7523bffbed762e7a8.jpg)

![fca09b1f354bdb4c128aaa99e1ecadafab87b70c4c3a31b2dfae8cac20022fd1.jpg](../icml_results/941_Enhancing%20Logits%20Distillation%20with%20Plug%26Play%20Kendall%27s%20%20%24_tau%24%20Ranking%20Loss/tables/fca09b1f354bdb4c128aaa99e1ecadafab87b70c4c3a31b2dfae8cac20022fd1.jpg)

## Fusing Reward and Dueling Feedback in Stochastic Bandits


### Images

![0b603c8d6fd76bfd051c8a1cbe55ed44eb0a38bb19831e1e863df8f973168aac.jpg](../icml_results/942_Fusing%20Reward%20and%20Dueling%20Feedback%20in%20Stochastic%20Bandits/images/0b603c8d6fd76bfd051c8a1cbe55ed44eb0a38bb19831e1e863df8f973168aac.jpg)

![3c56b54cae08eb6455bbda0012bc5ed0f664c9eb1f992411e3ad5c69671c8c7b.jpg](../icml_results/942_Fusing%20Reward%20and%20Dueling%20Feedback%20in%20Stochastic%20Bandits/images/3c56b54cae08eb6455bbda0012bc5ed0f664c9eb1f992411e3ad5c69671c8c7b.jpg)

![acaaea4199d9ff42eba0626b1dc826deab696f29dfcd5bdfca99c609d48b445d.jpg](../icml_results/942_Fusing%20Reward%20and%20Dueling%20Feedback%20in%20Stochastic%20Bandits/images/acaaea4199d9ff42eba0626b1dc826deab696f29dfcd5bdfca99c609d48b445d.jpg)

### Tables

![060e592c83189b36718f89e1bb2de5fde88c2d11b2dce7a31fa55757153ebf36.jpg](../icml_results/942_Fusing%20Reward%20and%20Dueling%20Feedback%20in%20Stochastic%20Bandits/tables/060e592c83189b36718f89e1bb2de5fde88c2d11b2dce7a31fa55757153ebf36.jpg)

![1a3753e6ff43a493763d29d5ff4766dc0cbd0576bf372b6ea72c1780ba124b4b.jpg](../icml_results/942_Fusing%20Reward%20and%20Dueling%20Feedback%20in%20Stochastic%20Bandits/tables/1a3753e6ff43a493763d29d5ff4766dc0cbd0576bf372b6ea72c1780ba124b4b.jpg)

## CogMath: Assessing LLMs' Authentic Mathematical Ability from a Human Cognitive Perspective


### Images

![46482aef08daeaf399a4616397159eaffe99a0ebd8a3a9922f64e3be33cc3b86.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/images/46482aef08daeaf399a4616397159eaffe99a0ebd8a3a9922f64e3be33cc3b86.jpg)

![abc47c222ffb89a3cc93889b248bab8a5e3f8d0682050c94ffd33782debe32a2.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/images/abc47c222ffb89a3cc93889b248bab8a5e3f8d0682050c94ffd33782debe32a2.jpg)

![e8995f0923f1d264ff6778d591c953be94297432118c0267edaed942d2f8868e.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/images/e8995f0923f1d264ff6778d591c953be94297432118c0267edaed942d2f8868e.jpg)

### Tables

![0835cde93714b675eeecf3281148a038d9882e9dc44ee6ef761d2f87c815c245.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/0835cde93714b675eeecf3281148a038d9882e9dc44ee6ef761d2f87c815c245.jpg)

![41c1fe0764fe4220784d0d028fe7628ad6f4489efb45a338908715b01ad8c71c.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/41c1fe0764fe4220784d0d028fe7628ad6f4489efb45a338908715b01ad8c71c.jpg)

![4bc4aeff0e55ab7867d691438d262de1a3bdbdd11c4e9c6bb08729b8003f91b1.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/4bc4aeff0e55ab7867d691438d262de1a3bdbdd11c4e9c6bb08729b8003f91b1.jpg)

![4c8043075e879fa35090aff02d4769e1cd5ef26db9802784cd776e0d0a57d57d.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/4c8043075e879fa35090aff02d4769e1cd5ef26db9802784cd776e0d0a57d57d.jpg)

![675a0df08076988e56ce36c9f309d29d271e74e82084a5e34fbb916bf184f427.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/675a0df08076988e56ce36c9f309d29d271e74e82084a5e34fbb916bf184f427.jpg)

![a2de132db39f44fe5f268b2b3a88d57d4ed74aeace6ebbfafd78357757bb4a70.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/a2de132db39f44fe5f268b2b3a88d57d4ed74aeace6ebbfafd78357757bb4a70.jpg)

![af3bf0e55fa9001fef9eb6d4a49569e017526cb4c4a1eaa9ad81260d030a1f9b.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/af3bf0e55fa9001fef9eb6d4a49569e017526cb4c4a1eaa9ad81260d030a1f9b.jpg)

![ce78932616061c2d76caac68881fc2a27ab56c20dfee674486e7335ce8d235dd.jpg](../icml_results/943_CogMath_%20Assessing%20LLMs%27%20Authentic%20Mathematical%20Ability%20from%20a%20Human%20Cognitive%20Perspective/tables/ce78932616061c2d76caac68881fc2a27ab56c20dfee674486e7335ce8d235dd.jpg)

## PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs


### Images

![03d022d5dcb99f504901bba2fb48368ad1128f9b77b5453f01d9bea8644e0134.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/03d022d5dcb99f504901bba2fb48368ad1128f9b77b5453f01d9bea8644e0134.jpg)

![0443366a00d194c4d38ea04043f514aba734f3d67e296102edb08c0c80d3599b.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/0443366a00d194c4d38ea04043f514aba734f3d67e296102edb08c0c80d3599b.jpg)

![0bc6a9a70fd16c4d8c3fab6089ecfd2976950f49f1dd029a65225e4c3f74748e.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/0bc6a9a70fd16c4d8c3fab6089ecfd2976950f49f1dd029a65225e4c3f74748e.jpg)

![0e137b05187f42c77cf27abbab2dbbfd31d01b553ffcee082ad1aad308469591.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/0e137b05187f42c77cf27abbab2dbbfd31d01b553ffcee082ad1aad308469591.jpg)

![0eb25a79366c6025f33294a5ca7905c02bdcfdfab21df447189a779d8e2dffaf.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/0eb25a79366c6025f33294a5ca7905c02bdcfdfab21df447189a779d8e2dffaf.jpg)

![12971204aba3c53535a00aadc8432ff392343c04860948938674fbb57509ba90.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/12971204aba3c53535a00aadc8432ff392343c04860948938674fbb57509ba90.jpg)

![1beddb02199c97d6afa762e89f5678f9a0fae77b64fbda32ff30143c98ea90d1.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/1beddb02199c97d6afa762e89f5678f9a0fae77b64fbda32ff30143c98ea90d1.jpg)

![1c94ba7dbaeb467b8fefd9acc0925da96e4617c5b27169ff0a94030c25845bc4.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/1c94ba7dbaeb467b8fefd9acc0925da96e4617c5b27169ff0a94030c25845bc4.jpg)

![2070b1274c811ebd1dfa729703396bc3451576210b2e7545bde11345c96b6b8b.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/2070b1274c811ebd1dfa729703396bc3451576210b2e7545bde11345c96b6b8b.jpg)

![26357d17f57ddc47b3fc67cb89167a2895494ef0c4f3b97ecde67e17d46022e8.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/26357d17f57ddc47b3fc67cb89167a2895494ef0c4f3b97ecde67e17d46022e8.jpg)

![2da89dd44e12b5d0c1a775cd53722677962ad6404070095df7b22c34258e15bf.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/2da89dd44e12b5d0c1a775cd53722677962ad6404070095df7b22c34258e15bf.jpg)

![2fd4f8dcc90cadcd0e9ae975c091c5affae891fdfc81bb8f3e0adcc1c913bec1.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/2fd4f8dcc90cadcd0e9ae975c091c5affae891fdfc81bb8f3e0adcc1c913bec1.jpg)

![35652fcb99a72644d081bbbeb006d681b207dfd137155dae4a06589cddd392f8.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/35652fcb99a72644d081bbbeb006d681b207dfd137155dae4a06589cddd392f8.jpg)

![475d217d2abef6bd0a78882e2803ab1d5464434c584c4393f619a73d9b686d47.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/475d217d2abef6bd0a78882e2803ab1d5464434c584c4393f619a73d9b686d47.jpg)

![4aca6c0d990746412b6706421ccaa4b1cd2d2fa185895d432dbef766e0320370.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/4aca6c0d990746412b6706421ccaa4b1cd2d2fa185895d432dbef766e0320370.jpg)

![4b6989fc44a774ef17ad6b54a8b7197e25fe31edd2a1dae53fe063ef57bb1b46.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/4b6989fc44a774ef17ad6b54a8b7197e25fe31edd2a1dae53fe063ef57bb1b46.jpg)

![5bb5ddd16925b35d046158610cb3d514e3caeb0e113c5dd574ec661598a646db.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/5bb5ddd16925b35d046158610cb3d514e3caeb0e113c5dd574ec661598a646db.jpg)

![5d1197b023551400afdab6087f016f752e8de013430c950d3ba721fc4c312360.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/5d1197b023551400afdab6087f016f752e8de013430c950d3ba721fc4c312360.jpg)

![5df32a0d49fe4502ee56178020838b68fad73b3093f458a2e8805da9e9189de7.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/5df32a0d49fe4502ee56178020838b68fad73b3093f458a2e8805da9e9189de7.jpg)

![73df3e3bbd629e102bf6a1200a5c2c94d6c9fee19dcb1da51743b57de3b1e6a8.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/73df3e3bbd629e102bf6a1200a5c2c94d6c9fee19dcb1da51743b57de3b1e6a8.jpg)

![7ac610cd7129881da484e9eb9ad908c95946c414fbac93bfe693ce58121a846e.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/7ac610cd7129881da484e9eb9ad908c95946c414fbac93bfe693ce58121a846e.jpg)

![7ca7e57caedd58a9a74e9e275e264a2314d39d2ccdfbd160b6fef438f29d09fe.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/7ca7e57caedd58a9a74e9e275e264a2314d39d2ccdfbd160b6fef438f29d09fe.jpg)

![832a8763a1d94b777c5d56267fa90f938b80ec81c3802ff610c949ec7d012179.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/832a8763a1d94b777c5d56267fa90f938b80ec81c3802ff610c949ec7d012179.jpg)

![8357de24dfcc31f1528f2f61ee648cf06ac3c81996c1e89e3ca6fb6274f18ca4.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/8357de24dfcc31f1528f2f61ee648cf06ac3c81996c1e89e3ca6fb6274f18ca4.jpg)

![849d78d63467b11ca916e4677c83c6afa319ae8c982a8ff5f07433c529a981fd.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/849d78d63467b11ca916e4677c83c6afa319ae8c982a8ff5f07433c529a981fd.jpg)

![94e2256b12efd923d2b0bcd8dd81f1178216424002f94236438ea3e7622b1e93.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/94e2256b12efd923d2b0bcd8dd81f1178216424002f94236438ea3e7622b1e93.jpg)

![a23ce0eb7366b078641b95a419e559d140c5bfc03eadcf54102582f7c1798da3.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/a23ce0eb7366b078641b95a419e559d140c5bfc03eadcf54102582f7c1798da3.jpg)

![abcc686d0f572dc438f0f5aafb8b6ebecf1290360f7d9b0a47aa9705209da049.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/abcc686d0f572dc438f0f5aafb8b6ebecf1290360f7d9b0a47aa9705209da049.jpg)

![aef12e4396c27b40a6473cd4a8421805f70ace3c3b42bd4b63ed66f3af11694c.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/aef12e4396c27b40a6473cd4a8421805f70ace3c3b42bd4b63ed66f3af11694c.jpg)

![b761db8c51f912b0800480f11dd2b9b8ba8c797848df3d5155100b67c93f2e18.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/b761db8c51f912b0800480f11dd2b9b8ba8c797848df3d5155100b67c93f2e18.jpg)

![c4a9218f888f9e17c3d4c28ab04f80396879b727dfb84922d178193961e900e7.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/c4a9218f888f9e17c3d4c28ab04f80396879b727dfb84922d178193961e900e7.jpg)

![d8f4bc5e45209febec17c0d7ba01dc0e5c6aaa38bad347109bcd260f50b81fae.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/d8f4bc5e45209febec17c0d7ba01dc0e5c6aaa38bad347109bcd260f50b81fae.jpg)

![ebf306294dadbd7da16611be47b744c388bfbe197d8eb25a0fd7d28ae5ad199d.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/ebf306294dadbd7da16611be47b744c388bfbe197d8eb25a0fd7d28ae5ad199d.jpg)

![f3fc47d06bbcf0e0a663093537f3018448cc9547070054c8bc239ab9dbf38a80.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/f3fc47d06bbcf0e0a663093537f3018448cc9547070054c8bc239ab9dbf38a80.jpg)

![f888916d8aa8f02641c05cd7763bd7ddd019c7f6e5f8fadfb09514369ffdcde9.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/f888916d8aa8f02641c05cd7763bd7ddd019c7f6e5f8fadfb09514369ffdcde9.jpg)

![f93daada026935992c04f80a813ebc65959c0e3df7f34d9f1eedea2585b651c3.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/f93daada026935992c04f80a813ebc65959c0e3df7f34d9f1eedea2585b651c3.jpg)

![ff8f5f5a69b25d05b416b7b9e19f87248c58f7fcea1b4525dab5093efd5f1787.jpg](../icml_results/944_PAK-UCB%20Contextual%20Bandit_%20An%20Online%20Learning%20Approach%20to%20Prompt-Aware%20Selection%20of%20Generative%20Model/images/ff8f5f5a69b25d05b416b7b9e19f87248c58f7fcea1b4525dab5093efd5f1787.jpg)

## Beyond Topological Self-Explainable GNNs: A Formal Explainability Perspective


### Images

![158aa03de074d497449e861e865672263be251f2832efcda1719571ba18acdb9.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/158aa03de074d497449e861e865672263be251f2832efcda1719571ba18acdb9.jpg)

![2660fef7b1040cbdc5eacfbf22d39af97055ca20d9597abb40719e459c7edc25.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/2660fef7b1040cbdc5eacfbf22d39af97055ca20d9597abb40719e459c7edc25.jpg)

![2e6823dd338299079c55d1443a9c563d224f5d8851141c879325c471dff07866.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/2e6823dd338299079c55d1443a9c563d224f5d8851141c879325c471dff07866.jpg)

![735b85d32e5a3b244e85341e713583b4779643ec5d4190cfbca0e080a40b95e5.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/735b85d32e5a3b244e85341e713583b4779643ec5d4190cfbca0e080a40b95e5.jpg)

![7db9935aaef336ea6c362c4593ffd7e31a21305512973eb58657cd2059f58b86.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/7db9935aaef336ea6c362c4593ffd7e31a21305512973eb58657cd2059f58b86.jpg)

![86e834970d114cacde08ca593da5a17186f563c94d9a9c6682fb31948203c811.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/86e834970d114cacde08ca593da5a17186f563c94d9a9c6682fb31948203c811.jpg)

![890287618cd661449fb92eafd8c9ae8cbe8f507b6a8641c396e579cb4d3b6833.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/890287618cd661449fb92eafd8c9ae8cbe8f507b6a8641c396e579cb4d3b6833.jpg)

![9744ddf49e53530676c48a3216e975a1817cb5a639e4baefa1a4a59fee907b13.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/9744ddf49e53530676c48a3216e975a1817cb5a639e4baefa1a4a59fee907b13.jpg)

![a710948f0e6b4bc16753bf44501d20ad643b18827df4ee9a872de57044d4a3ed.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/a710948f0e6b4bc16753bf44501d20ad643b18827df4ee9a872de57044d4a3ed.jpg)

![af0a494c2920c5d0cadc850e74e0aaec7fcc86d9fae047e52f4f9f62e149575e.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/af0a494c2920c5d0cadc850e74e0aaec7fcc86d9fae047e52f4f9f62e149575e.jpg)

![b3a8ca192a8ba86ff4a8c46b9af5818536947f4039d76d39833e3dab756bc7e8.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/b3a8ca192a8ba86ff4a8c46b9af5818536947f4039d76d39833e3dab756bc7e8.jpg)

![cab79b3cb29eec2a9800be3613ced974a5958fa2155ef4c8979b33111085fc14.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/cab79b3cb29eec2a9800be3613ced974a5958fa2155ef4c8979b33111085fc14.jpg)

![d19280bde41c442b6671d9890bda800280c613eed7099305ca115c898feaa215.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/d19280bde41c442b6671d9890bda800280c613eed7099305ca115c898feaa215.jpg)

![d532206cdade0d0b832c0e465412cf175abd2986d7cd6e7392173afb9c29e184.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/d532206cdade0d0b832c0e465412cf175abd2986d7cd6e7392173afb9c29e184.jpg)

![e6acf50f4d091f8bf77181bf829ce1c688c3a706b239fb09517c9c519f91617f.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/e6acf50f4d091f8bf77181bf829ce1c688c3a706b239fb09517c9c519f91617f.jpg)

![e76d74c8d5862ec963bfad65bd058971257399081e75344709976c2f0cd0bf38.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/images/e76d74c8d5862ec963bfad65bd058971257399081e75344709976c2f0cd0bf38.jpg)

### Tables

![02e9c68b18ddfe167c9361eb98e89d938f8972ce5d451095bde56b3901e72c96.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/02e9c68b18ddfe167c9361eb98e89d938f8972ce5d451095bde56b3901e72c96.jpg)

![323c3d49643fe8681481c7bd3bc9be892e6b64965322c46999f8624e8dcaa27b.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/323c3d49643fe8681481c7bd3bc9be892e6b64965322c46999f8624e8dcaa27b.jpg)

![4096f76832e8dc4acd6fd06c06738493cfa07d8c7957832677569c3959616fcb.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/4096f76832e8dc4acd6fd06c06738493cfa07d8c7957832677569c3959616fcb.jpg)

![5640a315ea3b6d125681fdfdb70fe1422f816b769388aa1e2f2b29ab4a7e3dbc.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/5640a315ea3b6d125681fdfdb70fe1422f816b769388aa1e2f2b29ab4a7e3dbc.jpg)

![5c90dcbfc4d143937982663e723f4502aeeddc39dccb38cb2b1143a33047d097.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/5c90dcbfc4d143937982663e723f4502aeeddc39dccb38cb2b1143a33047d097.jpg)

![6c87858614942cee73e59e5d3684b21f2bcbd4b7c994c82de9eddf9a9073921a.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/6c87858614942cee73e59e5d3684b21f2bcbd4b7c994c82de9eddf9a9073921a.jpg)

![6e58529c197ee7c02c47e4a9569db0c99fd7383b826b6567dd92384d8d4e6013.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/6e58529c197ee7c02c47e4a9569db0c99fd7383b826b6567dd92384d8d4e6013.jpg)

![97a2b68daad6a0a98cf0e0256623f54b2f1cd627b1548d89d73f90eb94d7ac4c.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/97a2b68daad6a0a98cf0e0256623f54b2f1cd627b1548d89d73f90eb94d7ac4c.jpg)

![9e880e24d7ddabfe0c1f896c4cd224d421fdbaf389bc0770fd20d95cf61e6229.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/9e880e24d7ddabfe0c1f896c4cd224d421fdbaf389bc0770fd20d95cf61e6229.jpg)

![f6784e9874481fe7dc6a9710b4d979fe9b5470fd1a99d2ef168c4771a01df170.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/f6784e9874481fe7dc6a9710b4d979fe9b5470fd1a99d2ef168c4771a01df170.jpg)

![f97ed288ddcd619110fa5cefc52f77f5e8f19f4ee8d511a79ad2fd1f8dac22ad.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/f97ed288ddcd619110fa5cefc52f77f5e8f19f4ee8d511a79ad2fd1f8dac22ad.jpg)

![fcb621e61eee3d48dd1d160d0d199b61265dfb6b8f84dc5a453b61b190160ab9.jpg](../icml_results/945_Beyond%20Topological%20Self-Explainable%20GNNs_%20A%20Formal%20Explainability%20Perspective/tables/fcb621e61eee3d48dd1d160d0d199b61265dfb6b8f84dc5a453b61b190160ab9.jpg)

## Primitive Vision: Improving Diagram Understanding in MLLMs


### Images

![07810e26384cfe925f1508b6bde2ef21401b3b899491686fd9d9fce9b6e6ef8b.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/07810e26384cfe925f1508b6bde2ef21401b3b899491686fd9d9fce9b6e6ef8b.jpg)

![0fc1efe4e9ef27d4bd980ed9123fd5b512a87a6d93f1add5c19deafb1fb71716.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/0fc1efe4e9ef27d4bd980ed9123fd5b512a87a6d93f1add5c19deafb1fb71716.jpg)

![2d3e4980201ea31da6128f23eee3d83fbf3e2f2269cb68db0c85bd04e7278c69.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/2d3e4980201ea31da6128f23eee3d83fbf3e2f2269cb68db0c85bd04e7278c69.jpg)

![43cfa3e762f84488904cae09550626ec07a27a5a8f054bb441405590071d9845.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/43cfa3e762f84488904cae09550626ec07a27a5a8f054bb441405590071d9845.jpg)

![49b3bef31e7c6d012c493f22bcdfa45c152a6b52c138a3338f76903438b365e7.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/49b3bef31e7c6d012c493f22bcdfa45c152a6b52c138a3338f76903438b365e7.jpg)

![4d92a5231718f861b7dce8f74edad90080c03b5c93d45dd23b06fbd166e865ac.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/4d92a5231718f861b7dce8f74edad90080c03b5c93d45dd23b06fbd166e865ac.jpg)

![4e38b4842417f2c64248e2d83da45ddb082e0159460ad90c85a73d7e1d50a0e6.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/4e38b4842417f2c64248e2d83da45ddb082e0159460ad90c85a73d7e1d50a0e6.jpg)

![4ea59e3f67dae116ff71f110786ffc5f2424a34bd97ac4162cbab64dda8623b6.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/4ea59e3f67dae116ff71f110786ffc5f2424a34bd97ac4162cbab64dda8623b6.jpg)

![71517af7faa4b54c97508d9a338cfa34e66d12e11a6371f90811215f23dc527a.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/71517af7faa4b54c97508d9a338cfa34e66d12e11a6371f90811215f23dc527a.jpg)

![76614439911fdcf8d131463421b539f43747d65536305a50f6935a94395b469f.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/76614439911fdcf8d131463421b539f43747d65536305a50f6935a94395b469f.jpg)

![8bcecd34c979f6aa5a2ed2568f787b21e19da48574ebb99c6a449cd4bf13ad6a.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/8bcecd34c979f6aa5a2ed2568f787b21e19da48574ebb99c6a449cd4bf13ad6a.jpg)

![8cb97bfedf5ff0002c02a800ce7e0abda787e0c27e3200975d79d05420243f54.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/8cb97bfedf5ff0002c02a800ce7e0abda787e0c27e3200975d79d05420243f54.jpg)

![a1603f6e5ec12368786b899d679480000cd2aa70c0ea862006d4a21fa36ab976.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/a1603f6e5ec12368786b899d679480000cd2aa70c0ea862006d4a21fa36ab976.jpg)

![cbbbedbcdbb66d836cfba69d1bf8d368e880f8160ea0aed4f0e6b479b726ab44.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/cbbbedbcdbb66d836cfba69d1bf8d368e880f8160ea0aed4f0e6b479b726ab44.jpg)

![d5c51ce0dda440c725849e9919226cd9adb743a6a685d79b7830f9f137a615cf.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/d5c51ce0dda440c725849e9919226cd9adb743a6a685d79b7830f9f137a615cf.jpg)

![d9bff4b730860bdb6ef96e89e4d9d66d8e7114e4da8e3e6d89939d6377dd00c5.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/d9bff4b730860bdb6ef96e89e4d9d66d8e7114e4da8e3e6d89939d6377dd00c5.jpg)

![e92b8148bb2d53f867a02ad3233620f9f8e37344c0acb39242f0458340e8cae2.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/e92b8148bb2d53f867a02ad3233620f9f8e37344c0acb39242f0458340e8cae2.jpg)

![efce6a583c3e2ba7ce8077b73a1480c01cfefaf8f6457efffd6a8b3223d222c8.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/efce6a583c3e2ba7ce8077b73a1480c01cfefaf8f6457efffd6a8b3223d222c8.jpg)

![f536d844c0539d83172abd0e275823435b27f27df848f32234a51c52219f2f1e.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/images/f536d844c0539d83172abd0e275823435b27f27df848f32234a51c52219f2f1e.jpg)

### Tables

![0331ebe1a33bec75f0658a6bc380c3ec362d4d3313579fd092585239bbb929b0.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/0331ebe1a33bec75f0658a6bc380c3ec362d4d3313579fd092585239bbb929b0.jpg)

![110e97235d0392b7020c53623ff9d63d69d0e89f587af721401d0b6408936218.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/110e97235d0392b7020c53623ff9d63d69d0e89f587af721401d0b6408936218.jpg)

![1f79e13fd29396a498e823de5920038997da2d01471d57e7f9a557cfe38e53fd.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/1f79e13fd29396a498e823de5920038997da2d01471d57e7f9a557cfe38e53fd.jpg)

![284584b7fd37f429113cdc6a43e0943f72deb74dc796880fd8791669adff6f0d.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/284584b7fd37f429113cdc6a43e0943f72deb74dc796880fd8791669adff6f0d.jpg)

![3acf84329ec00566d68d29384169276cce9ed84af675c58e47c2dbf9c63f81d8.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/3acf84329ec00566d68d29384169276cce9ed84af675c58e47c2dbf9c63f81d8.jpg)

![43a04db08a7b2dc55cb509720a01d5b872b6fc6b6b506343a4712c92f0bbd4a9.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/43a04db08a7b2dc55cb509720a01d5b872b6fc6b6b506343a4712c92f0bbd4a9.jpg)

![771e157a0dd17ce0cdd86dd4fd1290356d1f202a0022ec519d6ae5ddb1e3d94b.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/771e157a0dd17ce0cdd86dd4fd1290356d1f202a0022ec519d6ae5ddb1e3d94b.jpg)

![9b7590d8dc710375dabcd0b4bb2986c5b4e8312882d3a1118024e41619f0b9e5.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/9b7590d8dc710375dabcd0b4bb2986c5b4e8312882d3a1118024e41619f0b9e5.jpg)

![aee0a8839261b66054936b4adf2e4f04012215693a48a8f218df688be91e6c8e.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/aee0a8839261b66054936b4adf2e4f04012215693a48a8f218df688be91e6c8e.jpg)

![d0af4923791868b71875718ca739598e5e15e1001dffd842240f74c07b6f8db6.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/d0af4923791868b71875718ca739598e5e15e1001dffd842240f74c07b6f8db6.jpg)

![d6d473fdc44775483bbf1eebd857f1d9431bb85994bc5b635fcc1ba39f4f4198.jpg](../icml_results/946_Primitive%20Vision_%20Improving%20Diagram%20Understanding%20in%20MLLMs/tables/d6d473fdc44775483bbf1eebd857f1d9431bb85994bc5b635fcc1ba39f4f4198.jpg)

## HYGMA: Hypergraph Coordination Networks with Dynamic Grouping for Multi-Agent Reinforcement Learning


### Images

![056cd514c5e8332803c074d152ace03bf112ec20c4187d3f20ee07592a8a865a.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/images/056cd514c5e8332803c074d152ace03bf112ec20c4187d3f20ee07592a8a865a.jpg)

![83cafeee42a0e0124de85bc3ff6a3e9b06adf5dbdff075fdcf1fb6ef595ded62.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/images/83cafeee42a0e0124de85bc3ff6a3e9b06adf5dbdff075fdcf1fb6ef595ded62.jpg)

![a5fb9a23a40d23d57eb5bd957728b0709d273ad17dfb73d9edfb0cdde5990fdd.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/images/a5fb9a23a40d23d57eb5bd957728b0709d273ad17dfb73d9edfb0cdde5990fdd.jpg)

![d12aabbea05f8236ba213d5a3778277345a8ba29c7dd8b35f6ba695de435e710.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/images/d12aabbea05f8236ba213d5a3778277345a8ba29c7dd8b35f6ba695de435e710.jpg)

![e671523f9f0370b2b9011f37292ef511569278c592b4549f5946193c9b189304.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/images/e671523f9f0370b2b9011f37292ef511569278c592b4549f5946193c9b189304.jpg)

### Tables

![4f1957252fcf2068e094390e2b845c4b93c2add04de6cf7c94900574aedbc48f.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/tables/4f1957252fcf2068e094390e2b845c4b93c2add04de6cf7c94900574aedbc48f.jpg)

![58c83cd4162cc9815110707fec5344cb5da955543afd0d525cce8da267fd729d.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/tables/58c83cd4162cc9815110707fec5344cb5da955543afd0d525cce8da267fd729d.jpg)

![7c483d76870cdd0b48b8b3e842dc307ae41e95b177488f2b42c853ec04e57e03.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/tables/7c483d76870cdd0b48b8b3e842dc307ae41e95b177488f2b42c853ec04e57e03.jpg)

![a9ff38a2b177a560dda95c8b5f3f51699e29a52d0d25e21452e3683a6671e94d.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/tables/a9ff38a2b177a560dda95c8b5f3f51699e29a52d0d25e21452e3683a6671e94d.jpg)

![b28fb118be278f7dcea3eabadc998fb169c0c1d4b936b0888f2ac3cc3407541c.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/tables/b28fb118be278f7dcea3eabadc998fb169c0c1d4b936b0888f2ac3cc3407541c.jpg)

![e203faa09d00f7d213c47bea61938be4e57359f7bd30cc4d707a9b27d2d7683b.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/tables/e203faa09d00f7d213c47bea61938be4e57359f7bd30cc4d707a9b27d2d7683b.jpg)

![f8e37aebd543c06255cdc49235f4249554bfe0798ec2becfbe114a94e3379145.jpg](../icml_results/947_HYGMA_%20Hypergraph%20Coordination%20Networks%20with%20Dynamic%20Grouping%20for%20Multi-Agent%20Reinforcement%20Learning/tables/f8e37aebd543c06255cdc49235f4249554bfe0798ec2becfbe114a94e3379145.jpg)

## MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention


### Images

![0a782db95c2348f8939843a28c34760ae514047d335e46b34bd6e2d5cdcc78c4.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/0a782db95c2348f8939843a28c34760ae514047d335e46b34bd6e2d5cdcc78c4.jpg)

![15636de26be0ed420b1bf7a0d4ccf566f87cc0e763d36bbe181afd4e9bb49266.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/15636de26be0ed420b1bf7a0d4ccf566f87cc0e763d36bbe181afd4e9bb49266.jpg)

![173fec3e341a00c1558d2be000e77c8c1d59b6ea0b4926fbf41b679a9efdd609.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/173fec3e341a00c1558d2be000e77c8c1d59b6ea0b4926fbf41b679a9efdd609.jpg)

![3214edbb559b2f79b0f3d74179d33969db92ab07e67c64fc6160477183deb773.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/3214edbb559b2f79b0f3d74179d33969db92ab07e67c64fc6160477183deb773.jpg)

![4f7f3c8e6ac76a386d3df39e43de4b6d0dcbd4d22876c9ffecf5c058527bac94.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/4f7f3c8e6ac76a386d3df39e43de4b6d0dcbd4d22876c9ffecf5c058527bac94.jpg)

![4fbb6fa763ef222e6ebe752296b24b3155e7d40dce1ebc71aca4ce04f97dce4b.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/4fbb6fa763ef222e6ebe752296b24b3155e7d40dce1ebc71aca4ce04f97dce4b.jpg)

![65a2c19f564991a1ccb79b39d925e1a02228f7683060b1d773e42e968211e8a9.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/65a2c19f564991a1ccb79b39d925e1a02228f7683060b1d773e42e968211e8a9.jpg)

![7058003a883624d43c91c3979d01176551ab7932bbe3d6404625e93a783e959b.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/7058003a883624d43c91c3979d01176551ab7932bbe3d6404625e93a783e959b.jpg)

![717bfe0a31e764accdc9d1a56428d42207577f777e4cc93d132d9cf96b44869e.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/717bfe0a31e764accdc9d1a56428d42207577f777e4cc93d132d9cf96b44869e.jpg)

![7c2102f2ff99a44df5ec0f2ba59a45cf76053be0159ac6f8586f86dc60ad539e.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/7c2102f2ff99a44df5ec0f2ba59a45cf76053be0159ac6f8586f86dc60ad539e.jpg)

![866e5277a44c76581bba8d1ce80428beb376b8cee6124e18d5b9bbda293b1a44.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/866e5277a44c76581bba8d1ce80428beb376b8cee6124e18d5b9bbda293b1a44.jpg)

![8747322579efbc2871859509105ade968eb2e91afb062568c98233d778c30352.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/8747322579efbc2871859509105ade968eb2e91afb062568c98233d778c30352.jpg)

![93a67226b8036114cb202491ad63adf6fd51ff635504269202dd3857a628368c.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/93a67226b8036114cb202491ad63adf6fd51ff635504269202dd3857a628368c.jpg)

![9568205e363243839e757a56ea9edd2f73c14608cf57df15be8bcd3559b68900.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/9568205e363243839e757a56ea9edd2f73c14608cf57df15be8bcd3559b68900.jpg)

![c2bc5a2a3db1c41a096d9b0d0d6f1aac0b1c7f5be8e19a30aa7c0b6d8d81cfe3.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/c2bc5a2a3db1c41a096d9b0d0d6f1aac0b1c7f5be8e19a30aa7c0b6d8d81cfe3.jpg)

![c9a767f4b3947bbb704aaaece83de5e147b16a59f01302ddf1dec475bf66f862.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/c9a767f4b3947bbb704aaaece83de5e147b16a59f01302ddf1dec475bf66f862.jpg)

![cf49dabd97cf37d679eb737ac57c4862d55cd5623b8bfbe01ed14dec889691d7.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/cf49dabd97cf37d679eb737ac57c4862d55cd5623b8bfbe01ed14dec889691d7.jpg)

![d89dd23779f47b020fce3e560f16b4f4a4f98edf359c3e0d8c0dd8ea19d401a4.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/d89dd23779f47b020fce3e560f16b4f4a4f98edf359c3e0d8c0dd8ea19d401a4.jpg)

![dfbb4c535bdeeba47025067196b85b0eea162f947e1e4998f68c0653f2239ae6.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/dfbb4c535bdeeba47025067196b85b0eea162f947e1e4998f68c0653f2239ae6.jpg)

![e6958bf6db4c156d11b7b7256e5d1d959ec2db629aff9e727f3fb0622464cc1b.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/e6958bf6db4c156d11b7b7256e5d1d959ec2db629aff9e727f3fb0622464cc1b.jpg)

![ea2838063f62eec4b5628b20baa4aa8a4896fcedf90ff2c31db676942894d677.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/ea2838063f62eec4b5628b20baa4aa8a4896fcedf90ff2c31db676942894d677.jpg)

![f571ebb942b9dc1a5d8b597bc572ab60219613177d688f0cf10ec5a0ab92ad31.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/images/f571ebb942b9dc1a5d8b597bc572ab60219613177d688f0cf10ec5a0ab92ad31.jpg)

### Tables

![49efd5ecfa55a7407209e74592c14eff768a0e99ae93d9fc7dc10e3074205f06.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/tables/49efd5ecfa55a7407209e74592c14eff768a0e99ae93d9fc7dc10e3074205f06.jpg)

![79c7e3ca2959fb14fb0b094dcfeda2f7f49007b2c79e35bd13bb7ba8837cc5f0.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/tables/79c7e3ca2959fb14fb0b094dcfeda2f7f49007b2c79e35bd13bb7ba8837cc5f0.jpg)

![97e90bd401ddd91f580bdb0d971e29b7542798082794b8adf5a97cb7a2465396.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/tables/97e90bd401ddd91f580bdb0d971e29b7542798082794b8adf5a97cb7a2465396.jpg)

![ea49840c6fadc83f9e09ef1f1801f7d85d25b454f229098afe2efa83187dc877.jpg](../icml_results/948_MMInference_%20Accelerating%20Pre-filling%20for%20Long-Context%20Visual%20Language%20Models%20via%20Modality-Aware%20Per/tables/ea49840c6fadc83f9e09ef1f1801f7d85d25b454f229098afe2efa83187dc877.jpg)

## Permutation Equivariant Neural Networks for Symmetric Tensors


### Images

![03401011fadb3a7e84513444f432afdcd36ed3d66e04e38daba1a54e25c97a6c.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/03401011fadb3a7e84513444f432afdcd36ed3d66e04e38daba1a54e25c97a6c.jpg)

![0a61829da2b174c26ac1651436ddb1c0dc70fc2688ee436287033b5c3831ac4b.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/0a61829da2b174c26ac1651436ddb1c0dc70fc2688ee436287033b5c3831ac4b.jpg)

![0b0a56f1563b3f20fbe733ad4262351873d216d3c31ac4cafaa1ce8b0e332122.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/0b0a56f1563b3f20fbe733ad4262351873d216d3c31ac4cafaa1ce8b0e332122.jpg)

![0e16e1e628b4dc40ff04bf727c84f142ceb7dc7a5fb4b2ceef93049ea34e4fe7.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/0e16e1e628b4dc40ff04bf727c84f142ceb7dc7a5fb4b2ceef93049ea34e4fe7.jpg)

![11307c7cedf061fa7b1184398b6af02e3518e1e0b99365d2c2ad6a89899cbc8a.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/11307c7cedf061fa7b1184398b6af02e3518e1e0b99365d2c2ad6a89899cbc8a.jpg)

![212c74fcb127da915e77c591d028e92391dba4ec3ef631a0baeb5188640f3d83.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/212c74fcb127da915e77c591d028e92391dba4ec3ef631a0baeb5188640f3d83.jpg)

![490e540815c5fde489fbc2a18bde6e8049804f676d666db11160124a01c50c06.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/490e540815c5fde489fbc2a18bde6e8049804f676d666db11160124a01c50c06.jpg)

![4be61d66f0f3c6ea874b8efa4b6473b439195515f8f4f1a364c5214e6933e4ba.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/4be61d66f0f3c6ea874b8efa4b6473b439195515f8f4f1a364c5214e6933e4ba.jpg)

![4e09e1f4196fd5cb14aa266d79c1041b0756519b87a1f3b267f929ca981d4ab4.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/4e09e1f4196fd5cb14aa266d79c1041b0756519b87a1f3b267f929ca981d4ab4.jpg)

![4e2da15ecc0cfadd654fc00dc57f58c1c0d6ec6ef717491ca5c7c8adcba97f43.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/4e2da15ecc0cfadd654fc00dc57f58c1c0d6ec6ef717491ca5c7c8adcba97f43.jpg)

![5d455a12286ad001cea271c61fcb53a92afae2bcc85d214dee5747bed88744be.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/5d455a12286ad001cea271c61fcb53a92afae2bcc85d214dee5747bed88744be.jpg)

![6c4965ac5c61bfa96366ccee5e6b331024efde03f22613dfa3fbda31fb2fc2bf.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/6c4965ac5c61bfa96366ccee5e6b331024efde03f22613dfa3fbda31fb2fc2bf.jpg)

![73efdb4d18762fe20b46987a2842964aee4220fca6fd073700c300cf3f27bb5f.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/73efdb4d18762fe20b46987a2842964aee4220fca6fd073700c300cf3f27bb5f.jpg)

![7aa2c46e91ae9fd9940e4ce704d549a030b9743397b96105624405ded0d1b8b1.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/7aa2c46e91ae9fd9940e4ce704d549a030b9743397b96105624405ded0d1b8b1.jpg)

![81a84c15320180c7575c2f13dc5af25a4df414467c9e4fa14f711bb0e8914853.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/81a84c15320180c7575c2f13dc5af25a4df414467c9e4fa14f711bb0e8914853.jpg)

![8708d5d0a31ef75b80ef28ef5f85f26ae001f69a4eea541be110116e38485cb5.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/8708d5d0a31ef75b80ef28ef5f85f26ae001f69a4eea541be110116e38485cb5.jpg)

![8aab904f6e8f5da4699ecf7302cb99b211d957ad5337c19b9234b8297432614e.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/8aab904f6e8f5da4699ecf7302cb99b211d957ad5337c19b9234b8297432614e.jpg)

![8bfebc3e888fcc00e88ea8442b26be74f9da8b60d50eaf7dfcc86e489b71e8c9.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/8bfebc3e888fcc00e88ea8442b26be74f9da8b60d50eaf7dfcc86e489b71e8c9.jpg)

![927dbcdf741afff765107c16fd5349db0526bacc6fb52f92b909e3434539a84b.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/927dbcdf741afff765107c16fd5349db0526bacc6fb52f92b909e3434539a84b.jpg)

![944646fe0e679752463c020a8d94dde5f5cee7743f57824b5f155c4b86d83b09.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/944646fe0e679752463c020a8d94dde5f5cee7743f57824b5f155c4b86d83b09.jpg)

![975ca0f0f14ab376749495c999bf6e47ec089eb831ef4298c0564d6b44640ed6.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/975ca0f0f14ab376749495c999bf6e47ec089eb831ef4298c0564d6b44640ed6.jpg)

![9ddabeec2283151e00c824a3857567d8d8667fe90bfe8daaabc982e8f73c0575.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/9ddabeec2283151e00c824a3857567d8d8667fe90bfe8daaabc982e8f73c0575.jpg)

![9f433ef4df04620d9305f1dda627733f6d257fa470b95ada7ab7dbc22649e2e0.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/9f433ef4df04620d9305f1dda627733f6d257fa470b95ada7ab7dbc22649e2e0.jpg)

![a12dca3ded90c393e38d1308af6aecca692cdead10a2bc33658b376c4bbcd9f7.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/a12dca3ded90c393e38d1308af6aecca692cdead10a2bc33658b376c4bbcd9f7.jpg)

![acf135de5a51a5b875b86a9165bbba639ed385128751c1d6c3c95730393ea08a.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/acf135de5a51a5b875b86a9165bbba639ed385128751c1d6c3c95730393ea08a.jpg)

![bc64b89e19292f8dd0fadb6c0fce1fe7b37fd1610d5dc1de2eac23275117f5e1.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/bc64b89e19292f8dd0fadb6c0fce1fe7b37fd1610d5dc1de2eac23275117f5e1.jpg)

![cda3e2f91434bd90cf1fba0f3d3877e02e7573b91c2eb28a865b8b1f4b614cf8.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/cda3e2f91434bd90cf1fba0f3d3877e02e7573b91c2eb28a865b8b1f4b614cf8.jpg)

![d3fe94a622045ce1dcc49fec182b9b35605f8411ee33b4039ce2951485e6f389.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/d3fe94a622045ce1dcc49fec182b9b35605f8411ee33b4039ce2951485e6f389.jpg)

![dfdf91156b363904f06aec7e090855e2e68e7bb5e56d992486cf9491720fb7cc.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/dfdf91156b363904f06aec7e090855e2e68e7bb5e56d992486cf9491720fb7cc.jpg)

![e1db324084e306d94a3bb4122dbd31444a060c9edbb266908d3673219c9221ee.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/e1db324084e306d94a3bb4122dbd31444a060c9edbb266908d3673219c9221ee.jpg)

![e9c0a747a8a6ca9e0f317397516150760cab23ca15b1e1d5f153c9cd39141940.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/e9c0a747a8a6ca9e0f317397516150760cab23ca15b1e1d5f153c9cd39141940.jpg)

![eab33b7f862ae81d042128b34995914a80ae931b6138e941b37941621da77e95.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/eab33b7f862ae81d042128b34995914a80ae931b6138e941b37941621da77e95.jpg)

![ef06eadaec4609cde7f78085b41a5be7344ef95de8f77fe8049eb584a735b486.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/ef06eadaec4609cde7f78085b41a5be7344ef95de8f77fe8049eb584a735b486.jpg)

![ef478007676e9e3b55b9ee6434451268793269062b44207d892472ea653fa3a3.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/ef478007676e9e3b55b9ee6434451268793269062b44207d892472ea653fa3a3.jpg)

![f478a6ab136bebbde19ed08909c787bb427fdf53bd318b65d1cd090863cd667e.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/f478a6ab136bebbde19ed08909c787bb427fdf53bd318b65d1cd090863cd667e.jpg)

![f49f82a5db998d971b4ba3bdbf9d14d801cb6b16237819a771ef0d0ae9f19a10.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/f49f82a5db998d971b4ba3bdbf9d14d801cb6b16237819a771ef0d0ae9f19a10.jpg)

![f870808bad134f755c984f9c3aeb9863592df1aacaa5445696fb8d270bd85492.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/f870808bad134f755c984f9c3aeb9863592df1aacaa5445696fb8d270bd85492.jpg)

![fc7ea59401c92e253daca441716c167be5824cb363fb2f133ad2c5e48e26a8b3.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/images/fc7ea59401c92e253daca441716c167be5824cb363fb2f133ad2c5e48e26a8b3.jpg)

### Tables

![04913f7d6360bf29d467537edc5cec30ee7578dcc33e0dd049c1553ec4568588.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/tables/04913f7d6360bf29d467537edc5cec30ee7578dcc33e0dd049c1553ec4568588.jpg)

![f0fc9eedc4a645d87fc127d08ab7f9a0c8b4df96af4a5d16234e7eb51a710b7d.jpg](../icml_results/949_Permutation%20Equivariant%20Neural%20Networks%20for%20Symmetric%20Tensors/tables/f0fc9eedc4a645d87fc127d08ab7f9a0c8b4df96af4a5d16234e7eb51a710b7d.jpg)

## Fundamental Limits of Visual Autoregressive Transformers: Universal Approximation Abilities


### Images

![fb7272e213483fbb86e75863b7816448216017860396ff79994337544f09ffcd.jpg](../icml_results/950_Fundamental%20Limits%20of%20Visual%20Autoregressive%20Transformers_%20Universal%20Approximation%20Abilities/images/fb7272e213483fbb86e75863b7816448216017860396ff79994337544f09ffcd.jpg)

## How Expressive are Knowledge Graph Foundation Models?


### Images

![0b78f062541c9afe06d9b7f155d7c859293b616e66c02e9b392fd05dca4587cf.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/0b78f062541c9afe06d9b7f155d7c859293b616e66c02e9b392fd05dca4587cf.jpg)

![1e9542f3e00e3107dcd044171f42edbc701a7166d4b712e1f140c6502ed98d0f.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/1e9542f3e00e3107dcd044171f42edbc701a7166d4b712e1f140c6502ed98d0f.jpg)

![23da6b0accaabade086b5bbd23b458b4d2444240655fb5e765314e7c5b3839f4.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/23da6b0accaabade086b5bbd23b458b4d2444240655fb5e765314e7c5b3839f4.jpg)

![2c9b6e6bc534da3c7548a80d840f771262a46c716024840c6011e346a4e78c9d.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/2c9b6e6bc534da3c7548a80d840f771262a46c716024840c6011e346a4e78c9d.jpg)

![5b26e20ebc0232e2f3355033a1c5695f2f0c8d9772cc5a2e7495f38d0623b59b.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/5b26e20ebc0232e2f3355033a1c5695f2f0c8d9772cc5a2e7495f38d0623b59b.jpg)

![651fa9a214feb55b773e2b80a9b3bb53d49434fd5a469385fb55cd4c90d1bba6.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/651fa9a214feb55b773e2b80a9b3bb53d49434fd5a469385fb55cd4c90d1bba6.jpg)

![89a8cdd5e6527ab017ba5a8f907c2aadd2a00049731580be6488dac53d62dc02.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/89a8cdd5e6527ab017ba5a8f907c2aadd2a00049731580be6488dac53d62dc02.jpg)

![961ac240e27c8584510d661aad44a875d921f2098cc2f39970dde7b5be44278c.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/961ac240e27c8584510d661aad44a875d921f2098cc2f39970dde7b5be44278c.jpg)

![ba3b9aae818198828c760febea7761832fc73ae7e691794b88f2594a9fb5fa15.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/ba3b9aae818198828c760febea7761832fc73ae7e691794b88f2594a9fb5fa15.jpg)

![f4c00020826d1da12752ad2ad748c789352a4ad18c21c6b95076996cb48bf23d.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/images/f4c00020826d1da12752ad2ad748c789352a4ad18c21c6b95076996cb48bf23d.jpg)

### Tables

![39c7005db11e2e432e1a7754274e064d9154fa954c06e5741f878485c548ea59.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/39c7005db11e2e432e1a7754274e064d9154fa954c06e5741f878485c548ea59.jpg)

![5029ddd837f926a4b4c0de7b9b3a8478b82505374d671bc706e20dc9ce01196b.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/5029ddd837f926a4b4c0de7b9b3a8478b82505374d671bc706e20dc9ce01196b.jpg)

![515495b639ecb1f91d032af398c0f5628cf0e349f0a26a544e4cd27fe8408a7b.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/515495b639ecb1f91d032af398c0f5628cf0e349f0a26a544e4cd27fe8408a7b.jpg)

![5caa0d21a45065c5ce4f565d25ad75d4e797f936f04c188627052ea223129a16.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/5caa0d21a45065c5ce4f565d25ad75d4e797f936f04c188627052ea223129a16.jpg)

![5dcb4a62bee668571140cea16e87e89ba34c90d919f29676430c8f7f420d676d.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/5dcb4a62bee668571140cea16e87e89ba34c90d919f29676430c8f7f420d676d.jpg)

![6a1566d77c61459a45d2b1c3d08074fd1f90a81a2685a58528fd158bf3b70054.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/6a1566d77c61459a45d2b1c3d08074fd1f90a81a2685a58528fd158bf3b70054.jpg)

![73ef2fc407d671287bd99d72b16e3e7ffdeb6fed176a562738622340b18bd7d3.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/73ef2fc407d671287bd99d72b16e3e7ffdeb6fed176a562738622340b18bd7d3.jpg)

![759bec57ff70fe28510376d699765886045275a1838b5c92daa584f0adcfa0df.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/759bec57ff70fe28510376d699765886045275a1838b5c92daa584f0adcfa0df.jpg)

![7bb2d85351fbe34245b7d163019ad78f18c737d95cab2828f15fe1f77db3cd14.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/7bb2d85351fbe34245b7d163019ad78f18c737d95cab2828f15fe1f77db3cd14.jpg)

![7d298ba28b9f99d34f27bf01422cd49e28f464ebef430dc2e7aa72a52ab480cf.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/7d298ba28b9f99d34f27bf01422cd49e28f464ebef430dc2e7aa72a52ab480cf.jpg)

![8e183fc15525d6154ccaedf5620ccd1daf0f0f0c1dd01d6d29a25a76f58de656.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/8e183fc15525d6154ccaedf5620ccd1daf0f0f0c1dd01d6d29a25a76f58de656.jpg)

![b0a06afdffab6cf2972ff6a0c8c323a196a35c61f6e94e6ea58e340301050353.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/b0a06afdffab6cf2972ff6a0c8c323a196a35c61f6e94e6ea58e340301050353.jpg)

![bce2353e8c4379977ed368f5acf9288e6de401b077d30b4b225d5bb227bde6bf.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/bce2353e8c4379977ed368f5acf9288e6de401b077d30b4b225d5bb227bde6bf.jpg)

![d683b4b9054794eb6bd7e6f1b744c5fce8514f25518352cb46e9f3ed22c1f5ea.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/d683b4b9054794eb6bd7e6f1b744c5fce8514f25518352cb46e9f3ed22c1f5ea.jpg)

![d964feb5c5c56a87ff88694f7f41cd0b3eb6959d6540e0ffa71f8f724fea1647.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/d964feb5c5c56a87ff88694f7f41cd0b3eb6959d6540e0ffa71f8f724fea1647.jpg)

![f07dff244c8f18731e6ee2aa96b4f89c1524c6a35125a154e88f505c229a3c84.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/f07dff244c8f18731e6ee2aa96b4f89c1524c6a35125a154e88f505c229a3c84.jpg)

![ff980081423b8ad8d52c7091145196e32941117c0a7f8b14e07291d628f427ef.jpg](../icml_results/951_How%20Expressive%20are%20Knowledge%20Graph%20Foundation%20Models_/tables/ff980081423b8ad8d52c7091145196e32941117c0a7f8b14e07291d628f427ef.jpg)

## Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo


### Images

![1526ed6abde22de2dd347fca7dd93a53b7ae6796985e80a6376c63cad3b4e218.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/1526ed6abde22de2dd347fca7dd93a53b7ae6796985e80a6376c63cad3b4e218.jpg)

![1a988ad78b9255ab86d0093a7091814414f518ac9be8a6f47cc67f8c0a924b14.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/1a988ad78b9255ab86d0093a7091814414f518ac9be8a6f47cc67f8c0a924b14.jpg)

![34b3cc7fc789e75af654986ca73bef49901affefb94f04ab3a68fb29c59ac21d.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/34b3cc7fc789e75af654986ca73bef49901affefb94f04ab3a68fb29c59ac21d.jpg)

![4cbf948e64c2cd94788b5673aa4f67c89aea4d519ba6a374366e4eeb30366578.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/4cbf948e64c2cd94788b5673aa4f67c89aea4d519ba6a374366e4eeb30366578.jpg)

![4d1434210b932fa88a0486000df158b246b53709fe8de08c4ba7973cc8fa95c9.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/4d1434210b932fa88a0486000df158b246b53709fe8de08c4ba7973cc8fa95c9.jpg)

![526c0f2a0bd7f5dc16521bec18ebec3225ab5df871528fceead804e51688da63.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/526c0f2a0bd7f5dc16521bec18ebec3225ab5df871528fceead804e51688da63.jpg)

![8552797ee39d402daf7350630b985cf8b2b5f4f5be1f056a0c0a0827872c950b.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/8552797ee39d402daf7350630b985cf8b2b5f4f5be1f056a0c0a0827872c950b.jpg)

![a2a90735e8883f9ddfb98fb6375a61e1cbdc312a258abd677feed9b5ce297df1.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/a2a90735e8883f9ddfb98fb6375a61e1cbdc312a258abd677feed9b5ce297df1.jpg)

![b15cfd599379e395fa095718b3d5e2967be5b016101c6d0f4a69ef4652891efa.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/b15cfd599379e395fa095718b3d5e2967be5b016101c6d0f4a69ef4652891efa.jpg)

![d90f184cfbe5ff14203fea6335dc635ef03f5b31e1198db6828cc32fbcd45973.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/d90f184cfbe5ff14203fea6335dc635ef03f5b31e1198db6828cc32fbcd45973.jpg)

![ea4b7ee30737247df772a38bb453352f72aca7707c3bc7aab5012f4088dcc357.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/images/ea4b7ee30737247df772a38bb453352f72aca7707c3bc7aab5012f4088dcc357.jpg)

### Tables

![1686482737ed159dfcf6db31a3482ea139f4b2522c48fd08449691b934ad0f3c.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/1686482737ed159dfcf6db31a3482ea139f4b2522c48fd08449691b934ad0f3c.jpg)

![3aafa21fe8f4f8ad92593b80f9e23ac7a5953ab8a1c3d6b80ed7bf5aa4ce8a31.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/3aafa21fe8f4f8ad92593b80f9e23ac7a5953ab8a1c3d6b80ed7bf5aa4ce8a31.jpg)

![67eedc940d3cde4b9da4da70ae1b079ebdb0fceab44307ec89889b904e4f10f5.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/67eedc940d3cde4b9da4da70ae1b079ebdb0fceab44307ec89889b904e4f10f5.jpg)

![6ebe31d78d407d1f8827caf37aebb3b43c75493bcb39a904dce86291b48e1eb0.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/6ebe31d78d407d1f8827caf37aebb3b43c75493bcb39a904dce86291b48e1eb0.jpg)

![7b622e4a77bb3254608fdd86948a7e6262df4fa9b81772817cf5e5ebe926d32d.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/7b622e4a77bb3254608fdd86948a7e6262df4fa9b81772817cf5e5ebe926d32d.jpg)

![86363e8dbe4fbc02fb3fccce131a4720240a117246c19edab29735dc6f8fe985.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/86363e8dbe4fbc02fb3fccce131a4720240a117246c19edab29735dc6f8fe985.jpg)

![8cde4fe92d7d3baf48c51adfbcdf51c720c6dfda69fa928e50a19a8b4057df78.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/8cde4fe92d7d3baf48c51adfbcdf51c720c6dfda69fa928e50a19a8b4057df78.jpg)

![b17fe91f1bf4b3f09b0b98e3237b181d1c9672a4694fcecb0daf94410518248e.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/b17fe91f1bf4b3f09b0b98e3237b181d1c9672a4694fcecb0daf94410518248e.jpg)

![b6020a4dca3d8c7040481fe472ed812810c1e782e0ef89350874c839cf42319b.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/b6020a4dca3d8c7040481fe472ed812810c1e782e0ef89350874c839cf42319b.jpg)

![b6752bc684e6e4ad637ac417238add561d9441108148553ce1ff43c1914b987d.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/b6752bc684e6e4ad637ac417238add561d9441108148553ce1ff43c1914b987d.jpg)

![bacc9938b0030869a1ae07023fa57954521cf1fde91e0d8b375d6dde59ee1a04.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/bacc9938b0030869a1ae07023fa57954521cf1fde91e0d8b375d6dde59ee1a04.jpg)

![cf7ff08d51ba539a1c0b5e1c154cd7f6329f6ca22d0d01baccebd4c0527c1216.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/cf7ff08d51ba539a1c0b5e1c154cd7f6329f6ca22d0d01baccebd4c0527c1216.jpg)

![e04ffb2536c6c4827045315aca607d374e8e5b0400d9eb8e4f50540daf291254.jpg](../icml_results/952_Inverse%20Problem%20Sampling%20in%20Latent%20Space%20Using%20Sequential%20Monte%20Carlo/tables/e04ffb2536c6c4827045315aca607d374e8e5b0400d9eb8e4f50540daf291254.jpg)

## Proto Successor Measure: Representing the Behavior Space of an RL Agent


### Images

![19899f7876261f3e6472c0eb9ffc38f63ad4e7f481dbd5f462ac5de0e1ea9b9c.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/images/19899f7876261f3e6472c0eb9ffc38f63ad4e7f481dbd5f462ac5de0e1ea9b9c.jpg)

![35bc4ec9ea653ad4a3572167ddbd9f6856fea1b5110ecd5b347fb536180b7ffa.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/images/35bc4ec9ea653ad4a3572167ddbd9f6856fea1b5110ecd5b347fb536180b7ffa.jpg)

![3e704c415bb83921f60a79fb878302abec130677e36853ae4d5d5cd65b1e3644.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/images/3e704c415bb83921f60a79fb878302abec130677e36853ae4d5d5cd65b1e3644.jpg)

![5b081715d782bc9f4aeeb48c7037929e4be8c906ddf3ee70ee8dbf96edd1d9e3.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/images/5b081715d782bc9f4aeeb48c7037929e4be8c906ddf3ee70ee8dbf96edd1d9e3.jpg)

![7430766c31412dc8ac645f398da4d16fc9f74db62910d77c088fe04a09c2a3c1.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/images/7430766c31412dc8ac645f398da4d16fc9f74db62910d77c088fe04a09c2a3c1.jpg)

![81f46c83694fd8df8c5e9c23b9890240184cafb54e2f182be774a694071866cb.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/images/81f46c83694fd8df8c5e9c23b9890240184cafb54e2f182be774a694071866cb.jpg)

### Tables

![317f8ec57e9fc2e4b8be330bc5273e1ed2eb4446cd5bc5a066902babe7ff0345.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/tables/317f8ec57e9fc2e4b8be330bc5273e1ed2eb4446cd5bc5a066902babe7ff0345.jpg)

![b95ff60fa0df62693d3f69ff6ec89c39f5e73a4262ae25544898f7d48c052b74.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/tables/b95ff60fa0df62693d3f69ff6ec89c39f5e73a4262ae25544898f7d48c052b74.jpg)

![df79b2ce88f1228726fa36596aaba748de5316d764824f12d2d519e0a4d4cd26.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/tables/df79b2ce88f1228726fa36596aaba748de5316d764824f12d2d519e0a4d4cd26.jpg)

![ffc68d946589ea36fec5113b24cf3588ce699fcc60810648b8050f401cbb68df.jpg](../icml_results/953_Proto%20Successor%20Measure_%20Representing%20the%20Behavior%20Space%20of%20an%20RL%20Agent/tables/ffc68d946589ea36fec5113b24cf3588ce699fcc60810648b8050f401cbb68df.jpg)

## RISE: Radius of Influence based Subgraph Extraction for 3D Molecular Graph Explanation


### Images

![125463de9ad718cc6034c76a8ba14b1c3cc78af424beb01b9dd6b003a846ab17.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/125463de9ad718cc6034c76a8ba14b1c3cc78af424beb01b9dd6b003a846ab17.jpg)

![41a5214e08842db624028b9a22381c284d7c34678e6956795650dcb1799e9d34.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/41a5214e08842db624028b9a22381c284d7c34678e6956795650dcb1799e9d34.jpg)

![6546d8309aaf016bea98e1571f1c75bc6ea795ef39227a174b236b1b9b58a3fe.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/6546d8309aaf016bea98e1571f1c75bc6ea795ef39227a174b236b1b9b58a3fe.jpg)

![92003773c1e2ef034f04907324f518ab10074d99e85bcdf8d3884ad10aa13d67.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/92003773c1e2ef034f04907324f518ab10074d99e85bcdf8d3884ad10aa13d67.jpg)

![b736f37f3e2b7068c1b2cf54f603626f0cf86bb31a8fe8c0bab5409ffb0eacb5.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/b736f37f3e2b7068c1b2cf54f603626f0cf86bb31a8fe8c0bab5409ffb0eacb5.jpg)

![c98d9bb78001bf1bc69b04f13f4c91f403d090068f2028db12cf192d86df0128.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/c98d9bb78001bf1bc69b04f13f4c91f403d090068f2028db12cf192d86df0128.jpg)

![e30e2dce1d19958d08b58630c56792746d128d2e67bedd44a22a64d3e3ab2890.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/e30e2dce1d19958d08b58630c56792746d128d2e67bedd44a22a64d3e3ab2890.jpg)

![efd029d751dd2b134209b04282588dde80f70b20267736ef9080855d78d8ce99.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/efd029d751dd2b134209b04282588dde80f70b20267736ef9080855d78d8ce99.jpg)

![f8e504974620a400d1e2fb1ff7de547646fc9d19c5f62c0b7c5ef9d0557fe768.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/images/f8e504974620a400d1e2fb1ff7de547646fc9d19c5f62c0b7c5ef9d0557fe768.jpg)

### Tables

![2607c96c2485a6812447f973e16898e1a2aa25e813a1ca7150229efaf65eab23.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/2607c96c2485a6812447f973e16898e1a2aa25e813a1ca7150229efaf65eab23.jpg)

![2e9344bec351ea27fe4fbb7ad0ac68b0215b95bf639215e5f1a9c50f38cc68f8.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/2e9344bec351ea27fe4fbb7ad0ac68b0215b95bf639215e5f1a9c50f38cc68f8.jpg)

![652350a82d4d4d526cd70cd041e82ce30de5983e068ee7f1e3804172df555253.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/652350a82d4d4d526cd70cd041e82ce30de5983e068ee7f1e3804172df555253.jpg)

![71252e265dfb0c57c5571457aebb32894a973b587053aa87bf8caed24a73dc71.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/71252e265dfb0c57c5571457aebb32894a973b587053aa87bf8caed24a73dc71.jpg)

![c461588038ef114ca4b777018e54d79c108e75aaf82fa59d5f8b00d18ad00429.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/c461588038ef114ca4b777018e54d79c108e75aaf82fa59d5f8b00d18ad00429.jpg)

![c533ad674953a7fdc90c6ce11ad740e4515ed40815391d494414b75d65b909ef.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/c533ad674953a7fdc90c6ce11ad740e4515ed40815391d494414b75d65b909ef.jpg)

![e4011487e463fd0fafcf18335262e0a285c93c8b2a79b7aeb3dd7fe98a7d5bf5.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/e4011487e463fd0fafcf18335262e0a285c93c8b2a79b7aeb3dd7fe98a7d5bf5.jpg)

![f49f229b47714ad45a3c6422e25fb1e23ae1529233336e7b49094a8a53fb28ca.jpg](../icml_results/954_RISE_%20Radius%20of%20Influence%20based%20Subgraph%20Extraction%20for%203D%20Molecular%20Graph%20Explanation/tables/f49f229b47714ad45a3c6422e25fb1e23ae1529233336e7b49094a8a53fb28ca.jpg)

## AEQA-NAT : Adaptive End-to-end Quantization Alignment Training Framework for Non-autoregressive Machine Translation


### Images

![6fe3eca63e174138b10b87b735fdd19956db44cf56056e9c34fdadb38a19afde.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/images/6fe3eca63e174138b10b87b735fdd19956db44cf56056e9c34fdadb38a19afde.jpg)

![7ae0dc8f3f23a8e92e4a2dbfdd8efceb9beae8034eaf3ab9ce139234a414cff1.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/images/7ae0dc8f3f23a8e92e4a2dbfdd8efceb9beae8034eaf3ab9ce139234a414cff1.jpg)

![8025c5ffd0f2d9cceada3a336671d485b63a64e1a2c748027807b1818ebcbe96.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/images/8025c5ffd0f2d9cceada3a336671d485b63a64e1a2c748027807b1818ebcbe96.jpg)

![9e2c999f9b73581ab05f2c9fc73c30c52b692dae279fd2e7a315cef3c01e8dbb.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/images/9e2c999f9b73581ab05f2c9fc73c30c52b692dae279fd2e7a315cef3c01e8dbb.jpg)

![cfaf7cfd7491f50a030619cf4751c69769b4c62e419c799388def6d5b68039f7.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/images/cfaf7cfd7491f50a030619cf4751c69769b4c62e419c799388def6d5b68039f7.jpg)

![e6fd461e462d6b086a9aab80926465aea4c25174e53cfb7f366042111d335873.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/images/e6fd461e462d6b086a9aab80926465aea4c25174e53cfb7f366042111d335873.jpg)

![f0a0004d07686db0aea87039d560c4901088c2706eb534b686894141c1b0dc56.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/images/f0a0004d07686db0aea87039d560c4901088c2706eb534b686894141c1b0dc56.jpg)

### Tables

![0c90fad97edc58bf32ace55f256309bd57dad393dfb54e73f63d0b715b4a96f9.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/0c90fad97edc58bf32ace55f256309bd57dad393dfb54e73f63d0b715b4a96f9.jpg)

![3f17a342159403f52b991480df6a58245bcd5535cbcca0d1d5c05f5e628fd7aa.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/3f17a342159403f52b991480df6a58245bcd5535cbcca0d1d5c05f5e628fd7aa.jpg)

![4e5463631d9a8d36e127ba6ad6f9823ea0f916df84718d6019b5f78836eaf1bb.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/4e5463631d9a8d36e127ba6ad6f9823ea0f916df84718d6019b5f78836eaf1bb.jpg)

![6f206044619727c75d8a97a1dfd915be3bfaee964c69ecf43e81e9e95aae327f.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/6f206044619727c75d8a97a1dfd915be3bfaee964c69ecf43e81e9e95aae327f.jpg)

![7a1f79198aeb2bc37ac53e6029d58943ee08a6244d9ada861328eec2dbd57ccf.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/7a1f79198aeb2bc37ac53e6029d58943ee08a6244d9ada861328eec2dbd57ccf.jpg)

![89efad1c6f6f1e6a846b89dba60b75851bcaeac38f719b104d6d0b50e71e9044.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/89efad1c6f6f1e6a846b89dba60b75851bcaeac38f719b104d6d0b50e71e9044.jpg)

![bd7c37fa7772411e580929b85df7ecbf2fa366511d908460eef91ffaf40576e9.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/bd7c37fa7772411e580929b85df7ecbf2fa366511d908460eef91ffaf40576e9.jpg)

![f4093e183f7cb4988f48d2f4039e3560c69dca94a9e5dcde740b3963d635abdd.jpg](../icml_results/955_AEQA-NAT%20_%20Adaptive%20End-to-end%20Quantization%20Alignment%20Training%20Framework%20for%20Non-autoregressive%20Mach/tables/f4093e183f7cb4988f48d2f4039e3560c69dca94a9e5dcde740b3963d635abdd.jpg)

## Competing Bandits in Matching Markets via Super Stability


### Images

![08d9c8b834fb342260082265b65e4d7ddf3d5d34e560ffd357470b2c0efffc57.jpg](../icml_results/956_Competing%20Bandits%20in%20Matching%20Markets%20via%20Super%20Stability/images/08d9c8b834fb342260082265b65e4d7ddf3d5d34e560ffd357470b2c0efffc57.jpg)

![2a74acdd53ac5b3cd4616167fbdd78a39468ff214f4b81333ec08f15870893a2.jpg](../icml_results/956_Competing%20Bandits%20in%20Matching%20Markets%20via%20Super%20Stability/images/2a74acdd53ac5b3cd4616167fbdd78a39468ff214f4b81333ec08f15870893a2.jpg)

![303de2695766967d0bc6b7079cc5c8fada1a2b5c2b8d05d369a0fbbfc5fdaa43.jpg](../icml_results/956_Competing%20Bandits%20in%20Matching%20Markets%20via%20Super%20Stability/images/303de2695766967d0bc6b7079cc5c8fada1a2b5c2b8d05d369a0fbbfc5fdaa43.jpg)

![85b2caf0695cdf9312d5c9b8dbb484dfe58b1158db48c38dc53d68fc4777fe8b.jpg](../icml_results/956_Competing%20Bandits%20in%20Matching%20Markets%20via%20Super%20Stability/images/85b2caf0695cdf9312d5c9b8dbb484dfe58b1158db48c38dc53d68fc4777fe8b.jpg)

![b25a8146a2ff4393d6030b1992637626dd74fc4f143716f0f19fa3b9cd814a00.jpg](../icml_results/956_Competing%20Bandits%20in%20Matching%20Markets%20via%20Super%20Stability/images/b25a8146a2ff4393d6030b1992637626dd74fc4f143716f0f19fa3b9cd814a00.jpg)

### Tables

![233ac1d6f616ef184c1d0e789649150dc6b294fe3ec7fa804a2815f1fcc3d024.jpg](../icml_results/956_Competing%20Bandits%20in%20Matching%20Markets%20via%20Super%20Stability/tables/233ac1d6f616ef184c1d0e789649150dc6b294fe3ec7fa804a2815f1fcc3d024.jpg)

## Boosting Adversarial Robustness with CLAT: Criticality Leveraged Adversarial Training


### Images

![007651a91f56516d89d6c881bb0523b8ed05645012d8b08fefef7544b4fdde62.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/007651a91f56516d89d6c881bb0523b8ed05645012d8b08fefef7544b4fdde62.jpg)

![1e72753a5ba2929c3eab56b2ec48cbbd1c0215a5d3465a14ffdb3c3fbacfa74a.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/1e72753a5ba2929c3eab56b2ec48cbbd1c0215a5d3465a14ffdb3c3fbacfa74a.jpg)

![31b36d523bf11360a165899d76f028abbbea826dc58426d8636848f8c2c12d4a.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/31b36d523bf11360a165899d76f028abbbea826dc58426d8636848f8c2c12d4a.jpg)

![c6550a5db5833997963f8ec86cfba1e61d9a18c57b0df3afc035e5b01c80db6f.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/c6550a5db5833997963f8ec86cfba1e61d9a18c57b0df3afc035e5b01c80db6f.jpg)

![cbfc27a62ae163025b564479c2eeaa676091fdecf6333e95d043938190d02d06.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/cbfc27a62ae163025b564479c2eeaa676091fdecf6333e95d043938190d02d06.jpg)

![d0399f067304283d9b805b5bef2c347471fcfaadc13e094497e4448a54c7bca4.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/d0399f067304283d9b805b5bef2c347471fcfaadc13e094497e4448a54c7bca4.jpg)

![d3052c7622c5ecd184a0498cdfe33c6c411d3485070d4f4632ed12b3d2a44333.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/d3052c7622c5ecd184a0498cdfe33c6c411d3485070d4f4632ed12b3d2a44333.jpg)

![dcbe7c08f87e1b169cb2ff8dd21f7d5b535948173675dfd63b9beaac74179c61.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/images/dcbe7c08f87e1b169cb2ff8dd21f7d5b535948173675dfd63b9beaac74179c61.jpg)

### Tables

![261f099dbef652fa75752d16fedc5f52d311c8e2b0fb3d4b5ae6c730a9c1f031.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/261f099dbef652fa75752d16fedc5f52d311c8e2b0fb3d4b5ae6c730a9c1f031.jpg)

![2b428153f9d3660956043802eb8d945acb503ab0a4e437e292a7ecfd3da72d69.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/2b428153f9d3660956043802eb8d945acb503ab0a4e437e292a7ecfd3da72d69.jpg)

![2cac1a736f442076ebf86584d83edcd532f619338cbace8c696924b61d5ae618.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/2cac1a736f442076ebf86584d83edcd532f619338cbace8c696924b61d5ae618.jpg)

![39854478b72902f1d82cf77d94daad770c8a2e1773d15e799027af014e4fb820.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/39854478b72902f1d82cf77d94daad770c8a2e1773d15e799027af014e4fb820.jpg)

![39cc1fc5d2d188ddb53bf3d343c335cd2347ea087295f01db8b11d0fc261a4df.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/39cc1fc5d2d188ddb53bf3d343c335cd2347ea087295f01db8b11d0fc261a4df.jpg)

![3d3314dd6716dc02ec252d2571bdc38cb4f8cb106da6b8debf0fbfa848fa53cc.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/3d3314dd6716dc02ec252d2571bdc38cb4f8cb106da6b8debf0fbfa848fa53cc.jpg)

![57a8458e0a75778ae038ac9122c6471c5cd091e55d7e0f31bad50739d8af2f52.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/57a8458e0a75778ae038ac9122c6471c5cd091e55d7e0f31bad50739d8af2f52.jpg)

![5c560f78ef951b864d056ac02900d59a5fccacad4b9b396169d002d20586a7e9.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/5c560f78ef951b864d056ac02900d59a5fccacad4b9b396169d002d20586a7e9.jpg)

![6ad8c671e240fb6c68f6330331ef0f6d70222cac0859f7e600d0bf24f4b5956a.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/6ad8c671e240fb6c68f6330331ef0f6d70222cac0859f7e600d0bf24f4b5956a.jpg)

![7ef3d76cc4c53f480ace0dab3d37dcfb52909c37aa3881d387e34c1491b63906.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/7ef3d76cc4c53f480ace0dab3d37dcfb52909c37aa3881d387e34c1491b63906.jpg)

![91bfec944a0eb4bad05384437be9fcb759f44f0425c0216038026518a4428f00.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/91bfec944a0eb4bad05384437be9fcb759f44f0425c0216038026518a4428f00.jpg)

![95cc3ffea2815113e125acd82243e3e00383e3ad425077ffdbb86551257c83ae.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/95cc3ffea2815113e125acd82243e3e00383e3ad425077ffdbb86551257c83ae.jpg)

![a108ffd8ec458dcc0a433335fb20c7c395eeb1976235cb2a3ca3aa1b956a4017.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/a108ffd8ec458dcc0a433335fb20c7c395eeb1976235cb2a3ca3aa1b956a4017.jpg)

![ac20ae8473e5c4a61b1f2bb407a3a893339d3c1de10368ae25cb15f3614faf5b.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/ac20ae8473e5c4a61b1f2bb407a3a893339d3c1de10368ae25cb15f3614faf5b.jpg)

![afe8d95ac9ad26f03a6116dc5788cf70c9687fb489e5406ac5bb1d973f434efe.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/afe8d95ac9ad26f03a6116dc5788cf70c9687fb489e5406ac5bb1d973f434efe.jpg)

![c4297d729f8261c282e9f24a34b83fc0d1677fa79e5667cbba39e9c69b2df122.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/c4297d729f8261c282e9f24a34b83fc0d1677fa79e5667cbba39e9c69b2df122.jpg)

![c6863fcf8b3c4f86eebb120bb11ed3d5da450613564209c7ceae6d542f44f93c.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/c6863fcf8b3c4f86eebb120bb11ed3d5da450613564209c7ceae6d542f44f93c.jpg)

![c7787e4e6aeb5b9d9bf2f956ae8188baa9a6d644e6dee1d55daf7de285cfa6f4.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/c7787e4e6aeb5b9d9bf2f956ae8188baa9a6d644e6dee1d55daf7de285cfa6f4.jpg)

![c9c78b030f3cb8f9f348f52374c34f47a42aed6ee1c1b72a0e9659dfcfe380b7.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/c9c78b030f3cb8f9f348f52374c34f47a42aed6ee1c1b72a0e9659dfcfe380b7.jpg)

![d10b02ff47e9231fd88fc45c9e2e8dc53ef4575f6a7bfc7700ddd69b9d6a8530.jpg](../icml_results/957_Boosting%20Adversarial%20Robustness%20with%20CLAT_%20Criticality%20Leveraged%20Adversarial%20Training/tables/d10b02ff47e9231fd88fc45c9e2e8dc53ef4575f6a7bfc7700ddd69b9d6a8530.jpg)

## Reasoning-as-Logic-Units: Scaling Test-Time Reasoning in Large Language Models Through Logic Unit Alignment


### Images

![2be7b78e7644556d6a5a33fdb470d6d1bbcc9f3c0e2d4cb61fa5314cc77b2f89.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/images/2be7b78e7644556d6a5a33fdb470d6d1bbcc9f3c0e2d4cb61fa5314cc77b2f89.jpg)

![6eb45005cc6171a9ea481a9cd54f288a3818b66275e7bfb5eaa04d6ef72051eb.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/images/6eb45005cc6171a9ea481a9cd54f288a3818b66275e7bfb5eaa04d6ef72051eb.jpg)

![79ebee08b5f2a80e49cb73741b5529618bac1fa846ca46650928b26028d05e22.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/images/79ebee08b5f2a80e49cb73741b5529618bac1fa846ca46650928b26028d05e22.jpg)

![7ded3792addcd0a57ac4707df746848821b5a783e1c2acf32434582d64519a9c.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/images/7ded3792addcd0a57ac4707df746848821b5a783e1c2acf32434582d64519a9c.jpg)

![b34f4185b3a3241638a9a5b518bb8935782157c71f072868352eb040de749758.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/images/b34f4185b3a3241638a9a5b518bb8935782157c71f072868352eb040de749758.jpg)

![ef2b9bf25c2a08107a87a24bee910ccd36077566f2e34c2249ca20d8dd8b90d1.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/images/ef2b9bf25c2a08107a87a24bee910ccd36077566f2e34c2249ca20d8dd8b90d1.jpg)

### Tables

![533b653e660371fb44290bf317015cb67459c4646e860a24738da66cfcd46705.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/tables/533b653e660371fb44290bf317015cb67459c4646e860a24738da66cfcd46705.jpg)

![79becf72d20fcf70d50328ca64790f040713831201678bc9d7abb72877669ac6.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/tables/79becf72d20fcf70d50328ca64790f040713831201678bc9d7abb72877669ac6.jpg)

![864ecf094aac6c4ed01ca85851f0fce28f813b1458896ec275248717dbcd5122.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/tables/864ecf094aac6c4ed01ca85851f0fce28f813b1458896ec275248717dbcd5122.jpg)

![aaf988ee1055777e4f3c567d87ab11cfbfdd99e84eb110b7baf0f4d9478ccd1a.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/tables/aaf988ee1055777e4f3c567d87ab11cfbfdd99e84eb110b7baf0f4d9478ccd1a.jpg)

![acbf89383224f742a0144dedb285f11031f51a60ff1ed662fafd05e4310c2b83.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/tables/acbf89383224f742a0144dedb285f11031f51a60ff1ed662fafd05e4310c2b83.jpg)

![cb4d3b82c9ef2eb9a86d1a91ba5ff07d1597a3608971c505cf52eaba88cfb685.jpg](../icml_results/958_Reasoning-as-Logic-Units_%20Scaling%20Test-Time%20Reasoning%20in%20Large%20Language%20Models%20Through%20Logic%20Unit%20Al/tables/cb4d3b82c9ef2eb9a86d1a91ba5ff07d1597a3608971c505cf52eaba88cfb685.jpg)

## DEALing with Image Reconstruction: Deep Attentive Least Squares


### Images

![04c8a173e93cef76bc9003ad7740e339448cf605574f1a8886b060f0d3857e2b.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/04c8a173e93cef76bc9003ad7740e339448cf605574f1a8886b060f0d3857e2b.jpg)

![09f72d6f000fde8378af320ee93886e8d90c122700c16c307df5532e3e9101df.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/09f72d6f000fde8378af320ee93886e8d90c122700c16c307df5532e3e9101df.jpg)

![3e02974ffe2befd6f7fbea793e9ded5b3820b2cad53fb5c3170e772e325fea1b.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/3e02974ffe2befd6f7fbea793e9ded5b3820b2cad53fb5c3170e772e325fea1b.jpg)

![4a4b79f701da1cc7dae353590fe5b4a784861c2bc698e3e110f85aa805dc7793.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/4a4b79f701da1cc7dae353590fe5b4a784861c2bc698e3e110f85aa805dc7793.jpg)

![5483e172c6e687ea06b91754e5a4ce8c32d1bdc1cfd3d819a58404c4e7dbe5eb.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/5483e172c6e687ea06b91754e5a4ce8c32d1bdc1cfd3d819a58404c4e7dbe5eb.jpg)

![58737d73a8f83fd87a8d03cea9f1b555754a32f41b2c06f379104efceed10af2.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/58737d73a8f83fd87a8d03cea9f1b555754a32f41b2c06f379104efceed10af2.jpg)

![6313cda8c7f90623f4e0f77b7d1f545aa1d43761d287f3483a98a81da4bd1d6e.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/6313cda8c7f90623f4e0f77b7d1f545aa1d43761d287f3483a98a81da4bd1d6e.jpg)

![69515e06cc3f3ad0765945708da1ce4c1f9be01d19bcf44faeb97f42121c1242.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/69515e06cc3f3ad0765945708da1ce4c1f9be01d19bcf44faeb97f42121c1242.jpg)

![7eba4e21f572ae78b1d10e23088b9931bb59b780cfeba70c5c331ac3ded89ba0.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/7eba4e21f572ae78b1d10e23088b9931bb59b780cfeba70c5c331ac3ded89ba0.jpg)

![99a6d94fcc02c702ad3e88aa1a808138cc0048807f490f75e622aa14e608d7d5.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/99a6d94fcc02c702ad3e88aa1a808138cc0048807f490f75e622aa14e608d7d5.jpg)

![a1728d823198fdc83cbf1391aea0cf8ba0f3e38f8e044699e05988a1a080fc30.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/a1728d823198fdc83cbf1391aea0cf8ba0f3e38f8e044699e05988a1a080fc30.jpg)

![e80c97e8bfd919ac3c20c6fc407d3a7bfc71a105e3fcf9940e582fac0a548902.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/e80c97e8bfd919ac3c20c6fc407d3a7bfc71a105e3fcf9940e582fac0a548902.jpg)

![ecac93230d3612c8e7f162193262d0a6b2f8b6af7732292f2c62ce1cd36ea740.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/ecac93230d3612c8e7f162193262d0a6b2f8b6af7732292f2c62ce1cd36ea740.jpg)

![ee230dd6a89b0e9f6a6a08b459bbc4ca09e13246065ff68545cb1671e8877572.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/ee230dd6a89b0e9f6a6a08b459bbc4ca09e13246065ff68545cb1671e8877572.jpg)

![ef27b3c705f47d125cd40e0c7f3ed861fe1c1a47c931cc2d1b0061946e16ed38.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/images/ef27b3c705f47d125cd40e0c7f3ed861fe1c1a47c931cc2d1b0061946e16ed38.jpg)

### Tables

![1b92d510f2067c373f8a9408a3c6182e535bbb6f71efb72267251d29297a5cbd.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/1b92d510f2067c373f8a9408a3c6182e535bbb6f71efb72267251d29297a5cbd.jpg)

![37b0046dc7acd9b448e49b2ce5a909fbc919cea3c02d3e190978b972ff541bd0.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/37b0046dc7acd9b448e49b2ce5a909fbc919cea3c02d3e190978b972ff541bd0.jpg)

![4c9597afbc116d454565bf8fd3a6d68bdfbab4e6200f4279253b3165e7d615e0.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/4c9597afbc116d454565bf8fd3a6d68bdfbab4e6200f4279253b3165e7d615e0.jpg)

![717c72035f17a312dd4aec8da84c193978732328b4372b67a1c3c17f004e1bbd.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/717c72035f17a312dd4aec8da84c193978732328b4372b67a1c3c17f004e1bbd.jpg)

![75b7522eae9e90cd14a95b3e708a02d0a456a7aa874f89eca6272a5783741b7d.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/75b7522eae9e90cd14a95b3e708a02d0a456a7aa874f89eca6272a5783741b7d.jpg)

![8f609efae093b8131e3099d44230b2fd75f92a55f125259d005d68cc0a956001.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/8f609efae093b8131e3099d44230b2fd75f92a55f125259d005d68cc0a956001.jpg)

![c5de3e1d33b1a287a4d4bfc48671287e3fd4057a7efa16bde2c8c036d2912282.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/c5de3e1d33b1a287a4d4bfc48671287e3fd4057a7efa16bde2c8c036d2912282.jpg)

![cc9870288b90247f7d0e370ac53fbcd78962b2e3545016d8659e6f3c8258995c.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/cc9870288b90247f7d0e370ac53fbcd78962b2e3545016d8659e6f3c8258995c.jpg)

![e34139a7cb1b1e79c117d174ad6b0a88d109dad28b5fc90a496d00af12d9757d.jpg](../icml_results/959_DEALing%20with%20Image%20Reconstruction_%20Deep%20Attentive%20Least%20Squares/tables/e34139a7cb1b1e79c117d174ad6b0a88d109dad28b5fc90a496d00af12d9757d.jpg)

## Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners


### Images

![06a659aaaca8d39396b84411c8f1c69e52cdfd2886ee5ec83037116d2f2cb41e.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/06a659aaaca8d39396b84411c8f1c69e52cdfd2886ee5ec83037116d2f2cb41e.jpg)

![0e49fc5060eb4c683a8cf02c200f59c1f35dd860aeaf0f3b2303c7e729eddb1b.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/0e49fc5060eb4c683a8cf02c200f59c1f35dd860aeaf0f3b2303c7e729eddb1b.jpg)

![272f8e778b0797003e8018342c007f58ebc7527202972d0c7cb5effe850b9cf1.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/272f8e778b0797003e8018342c007f58ebc7527202972d0c7cb5effe850b9cf1.jpg)

![4f951de2ddda8b29d0120000fecc56e523610b8408040d305f49fa639a7521b9.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/4f951de2ddda8b29d0120000fecc56e523610b8408040d305f49fa639a7521b9.jpg)

![87d4f7ba3e21376005adffb3bee79a0bde6e70ed748bd39aad3b8d91295d005b.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/87d4f7ba3e21376005adffb3bee79a0bde6e70ed748bd39aad3b8d91295d005b.jpg)

![8cde28ecf85dd2d645e8c0e9859c896394c8289f655b9700e7e5afccd65a769d.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/8cde28ecf85dd2d645e8c0e9859c896394c8289f655b9700e7e5afccd65a769d.jpg)

![961863ebcd0a92e75da087cbe02e5b32589b06e8649241a944fdc514e9a323b4.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/961863ebcd0a92e75da087cbe02e5b32589b06e8649241a944fdc514e9a323b4.jpg)

![98725b65f0a2a41d775bf0f628098e393486d082764bfbc8c5f8d11d51a3d71f.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/98725b65f0a2a41d775bf0f628098e393486d082764bfbc8c5f8d11d51a3d71f.jpg)

![a7a2d51b9a4af61dc66893b573093fba8148a0ca8b52d49476730bcc48ee9b24.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/a7a2d51b9a4af61dc66893b573093fba8148a0ca8b52d49476730bcc48ee9b24.jpg)

![be3860ceda8a6980bbd9814a53d2554e40c5ce8aa553fcf60551caff9804a755.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/be3860ceda8a6980bbd9814a53d2554e40c5ce8aa553fcf60551caff9804a755.jpg)

![ce2c44aaf5b9aa3ebf6097f741806ef64227acd55801d08fe8581b7d16641299.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/ce2c44aaf5b9aa3ebf6097f741806ef64227acd55801d08fe8581b7d16641299.jpg)

![ec9702d1da997d4fa110668a31aa4ffaf5d67a85caa27c277f11737661a36770.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/images/ec9702d1da997d4fa110668a31aa4ffaf5d67a85caa27c277f11737661a36770.jpg)

### Tables

![0948a285ed9df19633dbc769fea22396b9741ce6b6ed41c4a5c985db0ce0557e.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/0948a285ed9df19633dbc769fea22396b9741ce6b6ed41c4a5c985db0ce0557e.jpg)

![1b36baa62ed1fda141fa6b68c031168e8dc9b2dae33f37f7f093b2abe8f6d4c0.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/1b36baa62ed1fda141fa6b68c031168e8dc9b2dae33f37f7f093b2abe8f6d4c0.jpg)

![1e9d41984764b10d8593dd27fecb3b9491fbeebdd69365283c3873381a753dcd.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/1e9d41984764b10d8593dd27fecb3b9491fbeebdd69365283c3873381a753dcd.jpg)

![2b37198050030d9e77b41efd3daab5b6f4ffb9952bc9b378b869a3540b3874dc.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/2b37198050030d9e77b41efd3daab5b6f4ffb9952bc9b378b869a3540b3874dc.jpg)

![440b3162a64170ea9b1690f2ec317e89f29d1a1f30faac2383feab7bd38f70f5.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/440b3162a64170ea9b1690f2ec317e89f29d1a1f30faac2383feab7bd38f70f5.jpg)

![68a93cdc2566e18e45a1fd0986de88a7de5f57355f7982342aa9b90f7ba6ff3d.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/68a93cdc2566e18e45a1fd0986de88a7de5f57355f7982342aa9b90f7ba6ff3d.jpg)

![7301bb042e095dc2365cdb26f8c28219b8b500be3d431dd7b351088c4f980d69.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/7301bb042e095dc2365cdb26f8c28219b8b500be3d431dd7b351088c4f980d69.jpg)

![76a4d061ce92b303b1157c07b5a15de5699d80d1658620fad7404343966cd403.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/76a4d061ce92b303b1157c07b5a15de5699d80d1658620fad7404343966cd403.jpg)

![bddf4dd4a03c197cb527d7bf844b3c69d694c8a39e61d60083a1ce23ec6c953f.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/bddf4dd4a03c197cb527d7bf844b3c69d694c8a39e61d60083a1ce23ec6c953f.jpg)

![cb10a44a48b0fb5256ded13b407c78c551cc4197a3f7b13fc5b6e01768d5aaeb.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/cb10a44a48b0fb5256ded13b407c78c551cc4197a3f7b13fc5b6e01768d5aaeb.jpg)

![cd71e4ddbee05aa68f38ec26c7f4ea471aafe1488498d59e754749ad7e2e9efc.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/cd71e4ddbee05aa68f38ec26c7f4ea471aafe1488498d59e754749ad7e2e9efc.jpg)

![e1ca01180a72ab7132e481ecbb13c20d7f845822ae6b1b5ded222281870ea70b.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/e1ca01180a72ab7132e481ecbb13c20d7f845822ae6b1b5ded222281870ea70b.jpg)

![e73e7fdaf609622181b71b527aa26bb69c58186ac802a26279ef0fe68fd6be72.jpg](../icml_results/960_Boosting%20Masked%20ECG-Text%20Auto-Encoders%20as%20Discriminative%20Learners/tables/e73e7fdaf609622181b71b527aa26bb69c58186ac802a26279ef0fe68fd6be72.jpg)

## Optimization for Neural Operators can Benefit from Width


### Images

![01bf75f773e665e09daca05f5127560a40244e9849f4b1fa0c36361fe3d28635.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/01bf75f773e665e09daca05f5127560a40244e9849f4b1fa0c36361fe3d28635.jpg)

![037b03fc207bcc7b58e6d42c205ac2f3278754a59ac5d5539a0269e7d238ccd4.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/037b03fc207bcc7b58e6d42c205ac2f3278754a59ac5d5539a0269e7d238ccd4.jpg)

![0a68f73a34c6d7a74a8627da739e582cf2ac306a80e26457e71989ed25ac5b48.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/0a68f73a34c6d7a74a8627da739e582cf2ac306a80e26457e71989ed25ac5b48.jpg)

![3af8edca3fe714acdfec9020ad2c274ef6f8d86aa96c8b06f8c931f321ce38c9.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/3af8edca3fe714acdfec9020ad2c274ef6f8d86aa96c8b06f8c931f321ce38c9.jpg)

![3b5525246f22a7953340ee94fc4b11a5590d8861dceb7f00db8d978ae597b691.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/3b5525246f22a7953340ee94fc4b11a5590d8861dceb7f00db8d978ae597b691.jpg)

![413e24cd614a6b1ef4bcfbce66574a12ebf723444f73f3ac7732c7eed926325c.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/413e24cd614a6b1ef4bcfbce66574a12ebf723444f73f3ac7732c7eed926325c.jpg)

![45b5ccee5528e0f146f9523cc89f2aa471e5c51a6d19d5d9dc1a6b7748a78152.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/45b5ccee5528e0f146f9523cc89f2aa471e5c51a6d19d5d9dc1a6b7748a78152.jpg)

![6500323a823bf58732b088ffad06c017c49ab7c96a2ed5d100cac318f05773bd.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/6500323a823bf58732b088ffad06c017c49ab7c96a2ed5d100cac318f05773bd.jpg)

![673121f48ed77b3db326a31ea219997697517cd5c605b77b5f0eb3bb19161a1f.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/673121f48ed77b3db326a31ea219997697517cd5c605b77b5f0eb3bb19161a1f.jpg)

![eee820631be587d82df717d89d0d0c55907f945f0ad77c5a6450cc2002799124.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/images/eee820631be587d82df717d89d0d0c55907f945f0ad77c5a6450cc2002799124.jpg)

### Tables

![09fc56da74cc4f2e9ac707a6ef1dfb1ec4a0450c530c0456176ecd3818424303.jpg](../icml_results/961_Optimization%20for%20Neural%20Operators%20can%20Benefit%20from%20Width/tables/09fc56da74cc4f2e9ac707a6ef1dfb1ec4a0450c530c0456176ecd3818424303.jpg)

## TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer


### Images

![373d572010014dd1c82e3a2abb228d1c36f098506137beb33b8eb4dc038f0bfe.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/images/373d572010014dd1c82e3a2abb228d1c36f098506137beb33b8eb4dc038f0bfe.jpg)

![41a8a4d136f5dafbf410e68490dcae537431f40703699beeb972572d513c5a9c.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/images/41a8a4d136f5dafbf410e68490dcae537431f40703699beeb972572d513c5a9c.jpg)

![969a5f05caf0e9ec4c867e9ddad2b36bb0a5facc285a0517dafb2d58bcf2aeb5.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/images/969a5f05caf0e9ec4c867e9ddad2b36bb0a5facc285a0517dafb2d58bcf2aeb5.jpg)

![c33ba4e5bd9bac23b0a24abe6ecfc3a6858ffe4bb79c7ac7fb18ed4f3a649cc4.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/images/c33ba4e5bd9bac23b0a24abe6ecfc3a6858ffe4bb79c7ac7fb18ed4f3a649cc4.jpg)

### Tables

![59bab25f1c7c9246805034fca56a9e36c35c7d8eb807c114332decbcef381461.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/tables/59bab25f1c7c9246805034fca56a9e36c35c7d8eb807c114332decbcef381461.jpg)

![6b61c165d9960ddd3deae7f5a07086a1e9a59a785c232bbbb37c27537685194f.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/tables/6b61c165d9960ddd3deae7f5a07086a1e9a59a785c232bbbb37c27537685194f.jpg)

![c86b99497e06bfb99a258da5e776d89046e202582b3796d2577e26c32470b3ec.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/tables/c86b99497e06bfb99a258da5e776d89046e202582b3796d2577e26c32470b3ec.jpg)

![f6a5977997924de7c7886618e8d449647af72ab7e1632643df1c2ec5e55c8fa7.jpg](../icml_results/962_TTFSFormer_%20A%20TTFS-based%20Lossless%20Conversion%20of%20Spiking%20Transformer/tables/f6a5977997924de7c7886618e8d449647af72ab7e1632643df1c2ec5e55c8fa7.jpg)

## Synthetic Text Generation for Training Large Language Models via Gradient Matching


### Images

![3a0e273b7b988bdd36013529ee0ae35ffac9230ffa1ff6c897fe91f31d1867db.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/images/3a0e273b7b988bdd36013529ee0ae35ffac9230ffa1ff6c897fe91f31d1867db.jpg)

![4997e40cdb28db3d35700cf643624c768c174adcba1924338b7505d582c1fafd.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/images/4997e40cdb28db3d35700cf643624c768c174adcba1924338b7505d582c1fafd.jpg)

![71bf1c7e83be8217efa4f22dd7bcc0776570d19fe6752a4d148c55ad2c48241c.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/images/71bf1c7e83be8217efa4f22dd7bcc0776570d19fe6752a4d148c55ad2c48241c.jpg)

![800506a8085b76eca1a1b4edf31f1a6e8366448b7573cb270ec2c542098d0890.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/images/800506a8085b76eca1a1b4edf31f1a6e8366448b7573cb270ec2c542098d0890.jpg)

![8613cbedc91810778b00d63c475b8cb43966d5b1ef70626a9b3299cbe40ebcbf.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/images/8613cbedc91810778b00d63c475b8cb43966d5b1ef70626a9b3299cbe40ebcbf.jpg)

![e8ab96b6bbe1d42a0455554f226881bddebb37c2a7735db5265e3424fab76909.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/images/e8ab96b6bbe1d42a0455554f226881bddebb37c2a7735db5265e3424fab76909.jpg)

![fb6cac1493a2080ef3c3d4aec695bd7e593247d058ca6891fb60b7cf47a2b76b.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/images/fb6cac1493a2080ef3c3d4aec695bd7e593247d058ca6891fb60b7cf47a2b76b.jpg)

### Tables

![3f02aa5cb2455ed36a65c615676bee9cdc5ff205f68a709fb65f0b0bd17a81bd.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/tables/3f02aa5cb2455ed36a65c615676bee9cdc5ff205f68a709fb65f0b0bd17a81bd.jpg)

![668f5d455bc8427f97e9d56950debb38f1569c6c6265dea5c8371616fcfe2696.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/tables/668f5d455bc8427f97e9d56950debb38f1569c6c6265dea5c8371616fcfe2696.jpg)

![851a5f5d63a667c335f97089f2f189b0fcf1ff45004f9907ffbd345ba9e723d1.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/tables/851a5f5d63a667c335f97089f2f189b0fcf1ff45004f9907ffbd345ba9e723d1.jpg)

![961b055309aba2363809dfbb24104d121e3560d6229f21fa7715b2791f2e6d52.jpg](../icml_results/963_Synthetic%20Text%20Generation%20for%20Training%20Large%20Language%20Models%20via%20Gradient%20Matching/tables/961b055309aba2363809dfbb24104d121e3560d6229f21fa7715b2791f2e6d52.jpg)

## Learning from Loss Landscape: Generalizable Mixed-Precision Quantization via Adaptive Sharpness-Aware Gradient Aligning


### Images

![2551f17670660d3e6b777b916d10de7b54ad71989134bb0901d876d525edc3f2.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/2551f17670660d3e6b777b916d10de7b54ad71989134bb0901d876d525edc3f2.jpg)

![41b37f3e58e3da3b649641c93432de5fd9705ccacfaa9b587fa18a2293988a6b.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/41b37f3e58e3da3b649641c93432de5fd9705ccacfaa9b587fa18a2293988a6b.jpg)

![53e7f0175160aeb9af3f01f843e88a9526baefcba77d3dad11e871513e33e911.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/53e7f0175160aeb9af3f01f843e88a9526baefcba77d3dad11e871513e33e911.jpg)

![82e218e46a3e140366b653187d8b091eec357d79f8000364d620d2a321be04ef.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/82e218e46a3e140366b653187d8b091eec357d79f8000364d620d2a321be04ef.jpg)

![b55b13764a6941a03869f34b578acdb6f7dec2f3550a4a1cec8d8bb189513927.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/b55b13764a6941a03869f34b578acdb6f7dec2f3550a4a1cec8d8bb189513927.jpg)

![beb69f59ee24aae9d20ddf1c050d150669f142b89341161a7f5cab4bf7a191f1.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/beb69f59ee24aae9d20ddf1c050d150669f142b89341161a7f5cab4bf7a191f1.jpg)

![c65b27e07f7ff3c19bbd4302ac2777b1065b295c74af3493e63d37894bfd912e.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/c65b27e07f7ff3c19bbd4302ac2777b1065b295c74af3493e63d37894bfd912e.jpg)

![f67aee94609d2074b160dfb055b26afeb47806a2e7db8c31b5b92cf165bf58ab.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/images/f67aee94609d2074b160dfb055b26afeb47806a2e7db8c31b5b92cf165bf58ab.jpg)

### Tables

![10786f7b6a2cfcc39b0bb6eacd2fe499e009b2ac5b3b935a6919f7d5772b8bf5.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/tables/10786f7b6a2cfcc39b0bb6eacd2fe499e009b2ac5b3b935a6919f7d5772b8bf5.jpg)

![34051c221d1f19b505f667cb857b5b2ffb7862905f785d3c9b4a6f66acbd6245.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/tables/34051c221d1f19b505f667cb857b5b2ffb7862905f785d3c9b4a6f66acbd6245.jpg)

![7cb0f6eacb506103bccfd65305a38f2e426dde1c3c7bc593c4f700ad2e92eefb.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/tables/7cb0f6eacb506103bccfd65305a38f2e426dde1c3c7bc593c4f700ad2e92eefb.jpg)

![d0990a33c690cb2193a9b74ec4e06deb169cb71aa2faf1b078867910d7b17cd0.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/tables/d0990a33c690cb2193a9b74ec4e06deb169cb71aa2faf1b078867910d7b17cd0.jpg)

![e4aaac3f94a2260f5352bf3e4a94efe51284262d08c2a2fc48189b2685d8bff7.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/tables/e4aaac3f94a2260f5352bf3e4a94efe51284262d08c2a2fc48189b2685d8bff7.jpg)

![eaaaaed4f34fe17332deb81dd0a0732921941aca84862609a286a1d08f32b7d9.jpg](../icml_results/964_Learning%20from%20Loss%20Landscape_%20Generalizable%20Mixed-Precision%20Quantization%20via%20Adaptive%20Sharpness-Awar/tables/eaaaaed4f34fe17332deb81dd0a0732921941aca84862609a286a1d08f32b7d9.jpg)

## Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning

### Images

![027375bc5e740ba03dea9adb1666869a89cf3ec90a66a38c4d24b059915b535c.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/027375bc5e740ba03dea9adb1666869a89cf3ec90a66a38c4d24b059915b535c.jpg)

![06c0fc7e4db4170e0f54ac41ecd55c70302f03dcaf7bdae4c7e2536ef1f65aff.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/06c0fc7e4db4170e0f54ac41ecd55c70302f03dcaf7bdae4c7e2536ef1f65aff.jpg)

![30f7f29bc7b01b5016cada607aec31cd1602bf292fb219e9e02ce8a2ced22b4a.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/30f7f29bc7b01b5016cada607aec31cd1602bf292fb219e9e02ce8a2ced22b4a.jpg)

![35274af862dd3bf247ff8a39d950014ec1a1eeac25b6d7510c355340854849ff.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/35274af862dd3bf247ff8a39d950014ec1a1eeac25b6d7510c355340854849ff.jpg)

![433bf077863f8e9632a453f30d78d6e6c0cd8ad6e325da1e143ca0353c2cafc5.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/433bf077863f8e9632a453f30d78d6e6c0cd8ad6e325da1e143ca0353c2cafc5.jpg)

![489a3d9a920b8859b4ef6f070c60981241eff9189dd0ac714cf469b4c8015fe3.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/489a3d9a920b8859b4ef6f070c60981241eff9189dd0ac714cf469b4c8015fe3.jpg)

![4c91edeca278b59f6ac34ba80426c85a28cddcd28995697c3efb96db9312743a.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/4c91edeca278b59f6ac34ba80426c85a28cddcd28995697c3efb96db9312743a.jpg)

![4e055c9bfafa558470fe9dbc12ed0abc56b0e98f8ef5eac420774ba8f69a9958.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/4e055c9bfafa558470fe9dbc12ed0abc56b0e98f8ef5eac420774ba8f69a9958.jpg)

![5cb81cf0cdcc46bf417ac97f226d2f91900873b2d5de04330fa494a66166daa1.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/5cb81cf0cdcc46bf417ac97f226d2f91900873b2d5de04330fa494a66166daa1.jpg)

![7038350bbaadb6b6f9b02fb2d667bde3ec6b39fb4ea3a39e167f80668f3bb661.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/7038350bbaadb6b6f9b02fb2d667bde3ec6b39fb4ea3a39e167f80668f3bb661.jpg)

![8ad45f981121acaa662f194d163e77329549019f79157c062af5931416fb063a.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/8ad45f981121acaa662f194d163e77329549019f79157c062af5931416fb063a.jpg)

![92475fbc787944f735f50a5ff5ba02a66a85adb7a042884e370fa6db3d0b6016.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/92475fbc787944f735f50a5ff5ba02a66a85adb7a042884e370fa6db3d0b6016.jpg)

![95e3e9871c06c7c3733d43880cedff8ccc022e4a9223b97fe64fd26da71132d1.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/95e3e9871c06c7c3733d43880cedff8ccc022e4a9223b97fe64fd26da71132d1.jpg)

![a140ef10a9cce44036cd654c897936e7f27be6ceafe66ae99aa71f29ebc6642e.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/a140ef10a9cce44036cd654c897936e7f27be6ceafe66ae99aa71f29ebc6642e.jpg)

![a6f93e99c49ce629e8b9e0f2f2c35ab0eee0a1c2de8add0f4938dfa1678c41e3.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/a6f93e99c49ce629e8b9e0f2f2c35ab0eee0a1c2de8add0f4938dfa1678c41e3.jpg)

![b3d7e7ad3a2099f7fe9f689d0476de264f4d655b606c778f50830dfb70ab754a.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/b3d7e7ad3a2099f7fe9f689d0476de264f4d655b606c778f50830dfb70ab754a.jpg)

![cc00f23931cb5b8cb6ba8d823503c20a5fc083cf62efc7d75f6c8e641a51747d.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/cc00f23931cb5b8cb6ba8d823503c20a5fc083cf62efc7d75f6c8e641a51747d.jpg)

![fb0df99273e2d114a56c65eac4d9452942f8c7de14a52b96cca32a834e4721c9.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/images/fb0df99273e2d114a56c65eac4d9452942f8c7de14a52b96cca32a834e4721c9.jpg)

### Tables

![00bdc921f997135a05073223a86428c63f48c55a31e4fe6087ffcbe580bf1d61.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/00bdc921f997135a05073223a86428c63f48c55a31e4fe6087ffcbe580bf1d61.jpg)

![20f7bf7be8b170b3eda66c29371daab72cf59ac5d4b67ed791450a1df92cdc81.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/20f7bf7be8b170b3eda66c29371daab72cf59ac5d4b67ed791450a1df92cdc81.jpg)

![39f978c7eebc563a3f97313b96f2713691f524098070e1869e419ed0be0dd4eb.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/39f978c7eebc563a3f97313b96f2713691f524098070e1869e419ed0be0dd4eb.jpg)

![3e505eaf2dc1c1eee100ff601b503db34dbc141704dbcc58b311bea49106f248.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/3e505eaf2dc1c1eee100ff601b503db34dbc141704dbcc58b311bea49106f248.jpg)

![46eac523da15226793fdb1f9466174034a4b66c5f89eb4cc2741a40ea1d1e4cf.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/46eac523da15226793fdb1f9466174034a4b66c5f89eb4cc2741a40ea1d1e4cf.jpg)

![4a3407e64bf8ca314cf5991095f1f59e723e29d57afbf373fa356575601baae4.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/4a3407e64bf8ca314cf5991095f1f59e723e29d57afbf373fa356575601baae4.jpg)

![4f3c6dec889c32620321c5f2ae645e8e55e5de2b62754d2e5bb0ed38b73cccc3.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/4f3c6dec889c32620321c5f2ae645e8e55e5de2b62754d2e5bb0ed38b73cccc3.jpg)

![8151f9664959e5115394d20022c587daabfa2ef5068084896bef78759a8861e2.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/8151f9664959e5115394d20022c587daabfa2ef5068084896bef78759a8861e2.jpg)

![8a73d4b0dd60911fc86e32673f06d48523c8ce18cf7a9373ce586ff4dca5358d.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/8a73d4b0dd60911fc86e32673f06d48523c8ce18cf7a9373ce586ff4dca5358d.jpg)

![9078b9765c4e95ee22773ce1d550a06629fbe478e3b332e22be834bd3bfcc70d.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/9078b9765c4e95ee22773ce1d550a06629fbe478e3b332e22be834bd3bfcc70d.jpg)

![95156b620dc85f778f3e7901a4672ee9f58869dde67b1c0b1de68fc3f874448f.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/95156b620dc85f778f3e7901a4672ee9f58869dde67b1c0b1de68fc3f874448f.jpg)

![aa323698c04b67961cc4a80420fca19471464277c5cac31668fb7eacd1245ef5.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/aa323698c04b67961cc4a80420fca19471464277c5cac31668fb7eacd1245ef5.jpg)

![b6a04dfd0adc8d8dabf2d96607cad941b2239d77803ce2f5a077f7edefae5221.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/b6a04dfd0adc8d8dabf2d96607cad941b2239d77803ce2f5a077f7edefae5221.jpg)

![d418f239758595903b4a7b77c9b838621ef91026b717f1ba3ad11f69a49c721b.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/d418f239758595903b4a7b77c9b838621ef91026b717f1ba3ad11f69a49c721b.jpg)

![e4796b3f2f2f973d0332952069dc7d182c8b147d2589874b88fe42b934ee9d78.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/e4796b3f2f2f973d0332952069dc7d182c8b147d2589874b88fe42b934ee9d78.jpg)

![f0a6436c664abfdd3171954ed9de3a5831e1cd72fc96f7bea929e54dc9d10004.jpg](../icml_results/965_Exploring%20Criteria%20of%20Loss%20Reweighting%20to%20Enhance%20LLM%20Unlearning/tables/f0a6436c664abfdd3171954ed9de3a5831e1cd72fc96f7bea929e54dc9d10004.jpg)
