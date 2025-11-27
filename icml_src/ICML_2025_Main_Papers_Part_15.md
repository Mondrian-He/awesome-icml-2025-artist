# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 15 of 100

## 目录 (Table of Contents)

1. [Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification](#Sample-Scrutinize-and-Scale-Effective-Inference-Time-Search-by-Scaling-Verification)
2. [Learning Mixtures of Experts with EM: A Mirror Descent Perspective](#Learning-Mixtures-of-Experts-with-EM-A-Mirror-Descent-Perspective)
3. [Bayesian Optimization from Human Feedback: Near-Optimal Regret Bounds](#Bayesian-Optimization-from-Human-Feedback-Near-Optimal-Regret-Bounds)
4. [Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding](#Rethinking-Addressing-in-Language-Models-via-Contextualized-Equivariant-Positional-Encoding)
5. [DocVXQA: Context-Aware Visual Explanations for Document Question Answering](#DocVXQA-Context-Aware-Visual-Explanations-for-Document-Question-Answering)
6. [Bayesian Active Learning for Bivariate Causal Discovery](#Bayesian-Active-Learning-for-Bivariate-Causal-Discovery)
7. [Ensemble Distribution Distillation via Flow Matching](#Ensemble-Distribution-Distillation-via-Flow-Matching)
8. [Automatic Differentiation of Optimization Algorithms with Time-Varying Updates](#Automatic-Differentiation-of-Optimization-Algorithms-with-Time-Varying-Updates)
9. [How Contaminated Is Your Benchmark? Measuring Dataset Leakage in Large Language Models with Kernel Divergence](#How-Contaminated-Is-Your-Benchmark-Measuring-Dataset-Leakage-in-Large-Language-Models-with-Kernel-Divergence)
10. [Revisiting Chain-of-Thought in Code Generation: Do Language Models Need to Learn Reasoning before Coding?](#Revisiting-Chain-of-Thought-in-Code-Generation-Do-Language-Models-Need-to-Learn-Reasoning-before-Coding)
11. [A Theoretical Study of (Hyper) Self-Attention through the Lens of Interactions: Representation, Training, Generalization](#A-Theoretical-Study-of-Hyper-Self-Attention-through-the-Lens-of-Interactions-Representation-Training-Generalization)
12. [EnsLoss: Stochastic Calibrated Loss Ensembles for Preventing Overfitting in Classification](#EnsLoss-Stochastic-Calibrated-Loss-Ensembles-for-Preventing-Overfitting-in-Classification)
13. [CRANE: Reasoning with constrained LLM generation](#CRANE-Reasoning-with-constrained-LLM-generation)
14. [Collapse-Proof Non-Contrastive Self-Supervised Learning](#Collapse-Proof-Non-Contrastive-Self-Supervised-Learning)
15. [Relating Misfit to Gain in Weak-to-Strong Generalization Beyond the Squared Loss](#Relating-Misfit-to-Gain-in-Weak-to-Strong-Generalization-Beyond-the-Squared-Loss)
16. [On Exact Bit-level Reversible Transformers Without Changing Architecture](#On-Exact-Bit-level-Reversible-Transformers-Without-Changing-Architecture)
17. [Clipped SGD Algorithms for Performative Prediction: Tight Bounds for Stochastic Bias and Remedies](#Clipped-SGD-Algorithms-for-Performative-Prediction-Tight-Bounds-for-Stochastic-Bias-and-Remedies)
18. [The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions](#The-Hidden-Dimensions-of-LLM-Alignment-A-Multi-Dimensional-Analysis-of-Orthogonal-Safety-Directions)
19. [GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers](#GrokFormer-Graph-Fourier-Kolmogorov-Arnold-Transformers)
20. [TtBA: Two-third Bridge Approach for Decision-Based Adversarial Attack](#TtBA-Two-third-Bridge-Approach-for-Decision-Based-Adversarial-Attack)
21. [Dissecting Submission Limit in Desk-Rejections: A Mathematical Analysis of Fairness in AI Conference Policies](#Dissecting-Submission-Limit-in-Desk-Rejections-A-Mathematical-Analysis-of-Fairness-in-AI-Conference-Policies)
22. [Optimizing Noise Distributions for Differential Privacy](#Optimizing-Noise-Distributions-for-Differential-Privacy)
23. [What Makes a Good Feedforward Computational Graph?](#What-Makes-a-Good-Feedforward-Computational-Graph)
24. [From Thousands to Billions: 3D Visual Language Grounding via Render-Supervised Distillation from 2D VLMs](#From-Thousands-to-Billions-3D-Visual-Language-Grounding-via-Render-Supervised-Distillation-from-2D-VLMs)
25. [A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features](#A-Cross-Modal-Knowledge-Distillation-Data-Augmentation-Recipe-for-Improving-Transcriptomics-Representations-through-Morphological-Features)
26. [Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks](#Efficient-Robust-Conformal-Prediction-via-Lipschitz-Bounded-Networks)
27. [Outlier-Aware Post-Training Quantization for Discrete Graph Diffusion Models](#Outlier-Aware-Post-Training-Quantization-for-Discrete-Graph-Diffusion-Models)
28. [ReverB-SNN: Reversing Bit of the Weight and Activation for Spiking Neural Networks](#ReverB-SNN-Reversing-Bit-of-the-Weight-and-Activation-for-Spiking-Neural-Networks)
29. [Rethinking Confidence Scores and Thresholds in Pseudolabeling-based SSL](#Rethinking-Confidence-Scores-and-Thresholds-in-Pseudolabeling-based-SSL)
30. [Fast Min-$\epsilon$ Segmented Regression using Constant-Time Segment Merging](#Fast-Min-epsilon-Segmented-Regression-using-Constant-Time-Segment-Merging)
31. [MTL-UE: Learning to Learn Nothing for Multi-Task Learning](#MTL-UE-Learning-to-Learn-Nothing-for-Multi-Task-Learning)
32. [Adaptive Sensitivity Analysis for Robust Augmentation against Natural Corruptions in Image Segmentation](#Adaptive-Sensitivity-Analysis-for-Robust-Augmentation-against-Natural-Corruptions-in-Image-Segmentation)
33. [No-Regret is not enough! Bandits with General Constraints through Adaptive Regret Minimization](#No-Regret-is-not-enough-Bandits-with-General-Constraints-through-Adaptive-Regret-Minimization)

---


## Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification

### Images

![27a06a56a0300b11bfbd6e32aa3592f04cf1529f323b042457a4fd830c4397c2.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/images/27a06a56a0300b11bfbd6e32aa3592f04cf1529f323b042457a4fd830c4397c2.jpg)

![327fe3501888c887fe4f1fb8a1914d0f130e98b72d9b3169f9873ae09ae78797.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/images/327fe3501888c887fe4f1fb8a1914d0f130e98b72d9b3169f9873ae09ae78797.jpg)

![4acc659361d15f76e4d4f879f06f66250d7e1fb7613d925282151254480f7ce1.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/images/4acc659361d15f76e4d4f879f06f66250d7e1fb7613d925282151254480f7ce1.jpg)

![74cfc1d9e6907ac258a34b147e816353faf0024d1f9359bcbe8de0a1813e50f8.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/images/74cfc1d9e6907ac258a34b147e816353faf0024d1f9359bcbe8de0a1813e50f8.jpg)

![d583f93e38787bee9aa1c078c314ee028ad093697aafb9d8d094114aaf0517ba.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/images/d583f93e38787bee9aa1c078c314ee028ad093697aafb9d8d094114aaf0517ba.jpg)

![d6229a1c629cc25935f5592993eb28f8612099f69d94bfbba4f08bf325556401.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/images/d6229a1c629cc25935f5592993eb28f8612099f69d94bfbba4f08bf325556401.jpg)

![d711ed7556612252a909b87a7db1ed089e1bdf18caa2d6476fe7b5f3fc5f195b.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/images/d711ed7556612252a909b87a7db1ed089e1bdf18caa2d6476fe7b5f3fc5f195b.jpg)

### Tables

![1c2edddf9d4d71b71e8513ecdf75270995cfba1fa51e8129e2e7834856ae3d4a.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/1c2edddf9d4d71b71e8513ecdf75270995cfba1fa51e8129e2e7834856ae3d4a.jpg)

![387b284e0f3dc2b8402e2cb49de5f7a73d611abe863313e52514dcb0c3dd98cd.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/387b284e0f3dc2b8402e2cb49de5f7a73d611abe863313e52514dcb0c3dd98cd.jpg)

![500da089d1ce3980fc4f53fa0a4d0d00b69337eed76e0e0cdf2c244fe6a36a9d.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/500da089d1ce3980fc4f53fa0a4d0d00b69337eed76e0e0cdf2c244fe6a36a9d.jpg)

![683deb5563fda671650b98b6d48dab64dd48f2202c48eb056ce4ff19be925176.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/683deb5563fda671650b98b6d48dab64dd48f2202c48eb056ce4ff19be925176.jpg)

![6ee13ce6aa61bc9c7f33ad40a86f0502a97a622476fd3fc05076cbb38d80cd75.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/6ee13ce6aa61bc9c7f33ad40a86f0502a97a622476fd3fc05076cbb38d80cd75.jpg)

![8b5d0386d5adc3f14226f5658681590fbb8fada13d1cf7d069f8fc16ebc553ce.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/8b5d0386d5adc3f14226f5658681590fbb8fada13d1cf7d069f8fc16ebc553ce.jpg)

![c3bbbb8d2ee1e2f49101f5a8335f90de522d3c54ed03afb0ea02d2ef8ab0e76d.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/c3bbbb8d2ee1e2f49101f5a8335f90de522d3c54ed03afb0ea02d2ef8ab0e76d.jpg)

![c5d9903c44d2f6199ddb8d3ea98cdc218b774ac7d97cda05b707a53ab0be034f.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/c5d9903c44d2f6199ddb8d3ea98cdc218b774ac7d97cda05b707a53ab0be034f.jpg)

![c9c3f12afa51ceafac159277431e01c238a747998659716d89d336ba8766f0bd.jpg](../icml_results/465_Less%20is%20More_%20Federated%20Graph%20Learning%20with%20Alleviating%20Topology%20Heterogeneity%20from%20A%20Causal%20Perspec/tables/c9c3f12afa51ceafac159277431e01c238a747998659716d89d336ba8766f0bd.jpg)

## Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification


### Images

![12c179f51bdefa83e5444cc977f2922cad0dc5af128d24cbfd9f8889c0da5657.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/12c179f51bdefa83e5444cc977f2922cad0dc5af128d24cbfd9f8889c0da5657.jpg)

![23c946ad552b1947b9ac457ca1226d9216e1e7d53790f7eed18de188c9ef3655.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/23c946ad552b1947b9ac457ca1226d9216e1e7d53790f7eed18de188c9ef3655.jpg)

![3eb34fca8fe7cc9ccca76505f6206631c013b79cda999b91144b6b882ecc267b.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/3eb34fca8fe7cc9ccca76505f6206631c013b79cda999b91144b6b882ecc267b.jpg)

![48629bd47dd1b835a0608819dfa70823d6da235d44e0ce1fe30cce6d36ad6982.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/48629bd47dd1b835a0608819dfa70823d6da235d44e0ce1fe30cce6d36ad6982.jpg)

![620d9f9f756f15ac2b4fd73989773caf294e440fb0013df4e4e447825a70afa9.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/620d9f9f756f15ac2b4fd73989773caf294e440fb0013df4e4e447825a70afa9.jpg)

![8f8a9e162187afd104dc42789a700c45dd7bede831f6615ab43fab211e1942ff.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/8f8a9e162187afd104dc42789a700c45dd7bede831f6615ab43fab211e1942ff.jpg)

![9d7bea07009a76915540f8924d09929aea30b8d79f361640f1cd40b7769ed154.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/9d7bea07009a76915540f8924d09929aea30b8d79f361640f1cd40b7769ed154.jpg)

![ea6a455778312e0d312b80ffd8b0e69f92c1208f6b7d33f7c6089fd90147af60.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/ea6a455778312e0d312b80ffd8b0e69f92c1208f6b7d33f7c6089fd90147af60.jpg)

![ebac538a19b72d8502e672b098027c16bc2cc1d284cfdd3ff5ac86aa10d25c08.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/images/ebac538a19b72d8502e672b098027c16bc2cc1d284cfdd3ff5ac86aa10d25c08.jpg)

### Tables

![20ac50c7ac20735782f1b4d34377eb13630e56c4650fa05e5750ac08b02b907d.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/20ac50c7ac20735782f1b4d34377eb13630e56c4650fa05e5750ac08b02b907d.jpg)

![210630417899615705e83fa79ec5706a673e400d7efad8743a680f12b9be4efd.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/210630417899615705e83fa79ec5706a673e400d7efad8743a680f12b9be4efd.jpg)

![4123619a594366f41d9a77fbb4a117f9889f8ba3302de24170b56c7ca4773c58.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/4123619a594366f41d9a77fbb4a117f9889f8ba3302de24170b56c7ca4773c58.jpg)

![94ea0de52febf3f6b538f211e58f7d1abfd7c84434a19b71a4731d7192c26feb.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/94ea0de52febf3f6b538f211e58f7d1abfd7c84434a19b71a4731d7192c26feb.jpg)

![b009e76ebe2efe382d28cd6f74eb2b57003e551e8fc18e345b4d5c00eab5f8b7.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/b009e76ebe2efe382d28cd6f74eb2b57003e551e8fc18e345b4d5c00eab5f8b7.jpg)

![bcddc0afeef84dca4139600e43263c4e7186e1994f8ff722df829be0d3e3795e.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/bcddc0afeef84dca4139600e43263c4e7186e1994f8ff722df829be0d3e3795e.jpg)

![bf3c26d19dddf05ff83ca545b42713f4cba27aab51947b465d65eda3d73a56d7.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/bf3c26d19dddf05ff83ca545b42713f4cba27aab51947b465d65eda3d73a56d7.jpg)

![dedaf9b05c8f70609092b3e8928c2730947c23457246c0aeab7fb12671f1e842.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/dedaf9b05c8f70609092b3e8928c2730947c23457246c0aeab7fb12671f1e842.jpg)

![e3f46993e352f75d84da5adb09f826f66d2890dba14fcb83b1ff0fa26ae0de76.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/e3f46993e352f75d84da5adb09f826f66d2890dba14fcb83b1ff0fa26ae0de76.jpg)

![f341739a7a3037fccb5b4ab8cf4b64d53d29b6bf1eecbca323a7a3d288f30fd7.jpg](../icml_results/466_Sample%2C%20Scrutinize%20and%20Scale_%20Effective%20Inference-Time%20Search%20by%20Scaling%20Verification/tables/f341739a7a3037fccb5b4ab8cf4b64d53d29b6bf1eecbca323a7a3d288f30fd7.jpg)

## Learning Mixtures of Experts with EM: A Mirror Descent Perspective


### Images

![2827524c00af1e859dd1581521a78a177289d8f0badccad77f90b72886bb7dac.jpg](../icml_results/467_Learning%20Mixtures%20of%20Experts%20with%20EM_%20A%20Mirror%20Descent%20Perspective/images/2827524c00af1e859dd1581521a78a177289d8f0badccad77f90b72886bb7dac.jpg)

![2dbe89b519206a0f8062da0c5e5ea9b7be783444397b9363d731deb982e68784.jpg](../icml_results/467_Learning%20Mixtures%20of%20Experts%20with%20EM_%20A%20Mirror%20Descent%20Perspective/images/2dbe89b519206a0f8062da0c5e5ea9b7be783444397b9363d731deb982e68784.jpg)

![77bfe0093a21f492818f470b5992c563e2888a2f06517fa6da3bbe1fbd0bda93.jpg](../icml_results/467_Learning%20Mixtures%20of%20Experts%20with%20EM_%20A%20Mirror%20Descent%20Perspective/images/77bfe0093a21f492818f470b5992c563e2888a2f06517fa6da3bbe1fbd0bda93.jpg)

![ff21955f5f7287b807b8366bc2685dccd3b564c0677f6d6dadff4e519647a77e.jpg](../icml_results/467_Learning%20Mixtures%20of%20Experts%20with%20EM_%20A%20Mirror%20Descent%20Perspective/images/ff21955f5f7287b807b8366bc2685dccd3b564c0677f6d6dadff4e519647a77e.jpg)

### Tables

![608ab9ee1b343f3dd5396f3050ef4b0b93c496549a6fefb276f22bf5c8b58afe.jpg](../icml_results/467_Learning%20Mixtures%20of%20Experts%20with%20EM_%20A%20Mirror%20Descent%20Perspective/tables/608ab9ee1b343f3dd5396f3050ef4b0b93c496549a6fefb276f22bf5c8b58afe.jpg)

![6fc76f0a9545c73cfeffa1195278041b9c1514146df25afb49c0b755d45ba829.jpg](../icml_results/467_Learning%20Mixtures%20of%20Experts%20with%20EM_%20A%20Mirror%20Descent%20Perspective/tables/6fc76f0a9545c73cfeffa1195278041b9c1514146df25afb49c0b755d45ba829.jpg)

## Bayesian Optimization from Human Feedback: Near-Optimal Regret Bounds


### Images

![b5de3a6a4348604964928249365c2a69e132c4e40685edeba78720baafab201c.jpg](../icml_results/468_Bayesian%20Optimization%20from%20Human%20Feedback_%20Near-Optimal%20Regret%20Bounds/images/b5de3a6a4348604964928249365c2a69e132c4e40685edeba78720baafab201c.jpg)

![bf338926a5354c63596209204635e48b9aa6f6fb17db241bf0587fb2c42f827e.jpg](../icml_results/468_Bayesian%20Optimization%20from%20Human%20Feedback_%20Near-Optimal%20Regret%20Bounds/images/bf338926a5354c63596209204635e48b9aa6f6fb17db241bf0587fb2c42f827e.jpg)

![fc76d7df60ffeae653ff716a0ae61442bad01822ee8860a5a77c28c88556851f.jpg](../icml_results/468_Bayesian%20Optimization%20from%20Human%20Feedback_%20Near-Optimal%20Regret%20Bounds/images/fc76d7df60ffeae653ff716a0ae61442bad01822ee8860a5a77c28c88556851f.jpg)

### Tables

![ef389ae8fd3b98344ea6167f4d85df575ec38126a801ef005d01a734cbb4e5f3.jpg](../icml_results/468_Bayesian%20Optimization%20from%20Human%20Feedback_%20Near-Optimal%20Regret%20Bounds/tables/ef389ae8fd3b98344ea6167f4d85df575ec38126a801ef005d01a734cbb4e5f3.jpg)

## Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding


### Images

![5e302528a8814756369790d6f5c2b1a078b36029691b8820bf3d99abed54b5e2.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/5e302528a8814756369790d6f5c2b1a078b36029691b8820bf3d99abed54b5e2.jpg)

![6e91aab0d629e4069032519b322d81f7d8492c32bf88793faf55cbbb375ec768.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/6e91aab0d629e4069032519b322d81f7d8492c32bf88793faf55cbbb375ec768.jpg)

![73f2b7bfc4328a014ab5f7380ff11b9153609b4b6884056ec2801041b554038f.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/73f2b7bfc4328a014ab5f7380ff11b9153609b4b6884056ec2801041b554038f.jpg)

![81323d25fad40a38e365973df7cb2aa411def8e77ecf187cb8514cb9b84e6841.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/81323d25fad40a38e365973df7cb2aa411def8e77ecf187cb8514cb9b84e6841.jpg)

![9e354b40ecf465a597b8954c3d1c963f743e8c485009c5593a5adc5d98f39e09.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/9e354b40ecf465a597b8954c3d1c963f743e8c485009c5593a5adc5d98f39e09.jpg)

![bce2ec923f0c21c07b9b79e300950a2eabb131c41e0c22bcbca5e2a9a3e4c8e8.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/bce2ec923f0c21c07b9b79e300950a2eabb131c41e0c22bcbca5e2a9a3e4c8e8.jpg)

![f6cfa2e7def7049f39bd981503b5ebeea8749399cfa2b1a0110a9791b6b23bbb.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/f6cfa2e7def7049f39bd981503b5ebeea8749399cfa2b1a0110a9791b6b23bbb.jpg)

![fd516bf6d8a727cb8241719228498e81b33b9eee625b91e924f76499a0e6fd0e.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/images/fd516bf6d8a727cb8241719228498e81b33b9eee625b91e924f76499a0e6fd0e.jpg)

### Tables

![00852055f5ab2dc9dc59aac4de3b50a3d25cb2621e58eaceab7863003ec83019.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/00852055f5ab2dc9dc59aac4de3b50a3d25cb2621e58eaceab7863003ec83019.jpg)

![105912ebdbf105bcb03b5cb24441e5fd97cbb8954809b5c2ca0a9c6b7e95ab86.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/105912ebdbf105bcb03b5cb24441e5fd97cbb8954809b5c2ca0a9c6b7e95ab86.jpg)

![1239f3b7f94abb2bcbf7e9c74b4285ab634128d431d47544863ec66259297ec3.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/1239f3b7f94abb2bcbf7e9c74b4285ab634128d431d47544863ec66259297ec3.jpg)

![280674ac2ba4bc79ef6b01cc86f79d159bd6f5c001963a766205ce8d74bb034e.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/280674ac2ba4bc79ef6b01cc86f79d159bd6f5c001963a766205ce8d74bb034e.jpg)

![295f97220daa0daa7645f2abfd0e8933180d8f8f5d7bd265a42017de327e618a.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/295f97220daa0daa7645f2abfd0e8933180d8f8f5d7bd265a42017de327e618a.jpg)

![3e3655e121d92085c7c99c175c5c90001a4ba1f1e7a899963883b7fe971fc57b.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/3e3655e121d92085c7c99c175c5c90001a4ba1f1e7a899963883b7fe971fc57b.jpg)

![865e5b5590af542ff969caa917093bcc0204106389e2b4562b0033617a38b097.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/865e5b5590af542ff969caa917093bcc0204106389e2b4562b0033617a38b097.jpg)

![9c111f6b79436948875d103d3780625fcd99dd5fac97951e827a120db2916be0.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/9c111f6b79436948875d103d3780625fcd99dd5fac97951e827a120db2916be0.jpg)

![a28e9614b0497c5e71d3bca42c5e37da8e56a4c70f360db73f6af57655d5a6a1.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/a28e9614b0497c5e71d3bca42c5e37da8e56a4c70f360db73f6af57655d5a6a1.jpg)

![abd03b76e709edf525efe7bd3cecdc06e6e7cdfff3cf182e673d997ed6599d60.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/abd03b76e709edf525efe7bd3cecdc06e6e7cdfff3cf182e673d997ed6599d60.jpg)

![ae42a01ab0281968a4cba12acc075c49699da09bba7c68afb78f0f392d13c988.jpg](../icml_results/469_Rethinking%20Addressing%20in%20Language%20Models%20via%20Contextualized%20Equivariant%20Positional%20Encoding/tables/ae42a01ab0281968a4cba12acc075c49699da09bba7c68afb78f0f392d13c988.jpg)

## DocVXQA: Context-Aware Visual Explanations for Document Question Answering


### Images

![08403449fddd20bbea088af86b7579cab4ebef62972d8c28397c18d2dc430eb1.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/08403449fddd20bbea088af86b7579cab4ebef62972d8c28397c18d2dc430eb1.jpg)

![1a3816d70c5c174a7935039a194895255f549f74784dab4a9fea2a4d3a365920.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/1a3816d70c5c174a7935039a194895255f549f74784dab4a9fea2a4d3a365920.jpg)

![29c74c4bbb6dad813a195b5f4172796b799df4df86c760a0dbaa2a17292a53a1.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/29c74c4bbb6dad813a195b5f4172796b799df4df86c760a0dbaa2a17292a53a1.jpg)

![2f3a48013016b0367d8d53636017406f6a897d832267a5ae370098f943f933ae.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/2f3a48013016b0367d8d53636017406f6a897d832267a5ae370098f943f933ae.jpg)

![477c09886b5744af2f9109b0071d218e84295fae1411a268bfcc229465fce741.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/477c09886b5744af2f9109b0071d218e84295fae1411a268bfcc229465fce741.jpg)

![6060a66de71a172a739e234f6dd250542a583414d83cf983728fe439f612b536.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/6060a66de71a172a739e234f6dd250542a583414d83cf983728fe439f612b536.jpg)

![682f35788ab6eb18e345a25b9616cb3dcfeb0211cda35a405a66f5a4dc188ed3.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/682f35788ab6eb18e345a25b9616cb3dcfeb0211cda35a405a66f5a4dc188ed3.jpg)

![76e06104cc5c9a4a2b24791de520b52d13957441f5537ceb3cd3b7be13a06f34.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/76e06104cc5c9a4a2b24791de520b52d13957441f5537ceb3cd3b7be13a06f34.jpg)

![78d76d04ac014783346f5e44d770a611319c0fef949298de4b94cb60452e3686.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/78d76d04ac014783346f5e44d770a611319c0fef949298de4b94cb60452e3686.jpg)

![7fa53fb394730dd1c975fdbe4f3ae35b4af3f5542ffb04a8a65b9a93f5d57742.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/7fa53fb394730dd1c975fdbe4f3ae35b4af3f5542ffb04a8a65b9a93f5d57742.jpg)

![89cb7314188db1681e2140a2aba2a17a63187c63bc5fcd40ba56df52aeed393f.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/89cb7314188db1681e2140a2aba2a17a63187c63bc5fcd40ba56df52aeed393f.jpg)

![950b4f7f2c3ee9308e17a140ccf1fe893c47f04372d428f0537cce69aafc8e11.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/950b4f7f2c3ee9308e17a140ccf1fe893c47f04372d428f0537cce69aafc8e11.jpg)

![977397cea5292062c986dae71359a281e796cc665cd25b9f30fcb02b41c254b4.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/977397cea5292062c986dae71359a281e796cc665cd25b9f30fcb02b41c254b4.jpg)

![9a455734d0ec5ca27f41b9a2f0c8e6e8efb6320b8caa97dd304fac4c6de98309.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/9a455734d0ec5ca27f41b9a2f0c8e6e8efb6320b8caa97dd304fac4c6de98309.jpg)

![9adf7e17fef6dae789a34489d188f6e2b1f0c421ee7c22c3d84b763498506a06.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/9adf7e17fef6dae789a34489d188f6e2b1f0c421ee7c22c3d84b763498506a06.jpg)

![a3cd5e787047a422d6b714021c0e1ceeb54c359e22cef0b9253c2f4172bbade3.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/a3cd5e787047a422d6b714021c0e1ceeb54c359e22cef0b9253c2f4172bbade3.jpg)

![a7c2c03ea4d58ceb614a08fe928557fc79b289d0157b3abfa79959921a7aa48c.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/a7c2c03ea4d58ceb614a08fe928557fc79b289d0157b3abfa79959921a7aa48c.jpg)

![a9e6b3c1beaba9da2d38a0b84d6d27ec17843b536bed88e8c0b506a458ff7b22.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/a9e6b3c1beaba9da2d38a0b84d6d27ec17843b536bed88e8c0b506a458ff7b22.jpg)

![b8ee71906d27176532b4f48c7e8dad1aa9d5f86c5f58d9c6cd3ca793349b8db2.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/b8ee71906d27176532b4f48c7e8dad1aa9d5f86c5f58d9c6cd3ca793349b8db2.jpg)

![bc8ff045fcd5a824d53e998221591310ebe838f32366ddaafb9fb02e7ea32290.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/bc8ff045fcd5a824d53e998221591310ebe838f32366ddaafb9fb02e7ea32290.jpg)

![bd324b36bd3d2365fafcac967c16a8e74feb5ce53ad8eae27481121ec6e05e8c.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/bd324b36bd3d2365fafcac967c16a8e74feb5ce53ad8eae27481121ec6e05e8c.jpg)

![c8961168d6a177a56d277ddc1e090fd2b312454b7c4ca3651d4069c2e7c853ad.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/c8961168d6a177a56d277ddc1e090fd2b312454b7c4ca3651d4069c2e7c853ad.jpg)

![cffdcd3d57a91718f03d307a2d2919412a55a058846e9ff0c428ada9525eeacc.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/cffdcd3d57a91718f03d307a2d2919412a55a058846e9ff0c428ada9525eeacc.jpg)

![d5b057d3c878d0e200f1a7d6636dd17edaefb08413d1410f37423b21176569c9.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/d5b057d3c878d0e200f1a7d6636dd17edaefb08413d1410f37423b21176569c9.jpg)

![e1564f6c880f21c538a580952befebe902e58279d093405b2cce079fb0171430.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/e1564f6c880f21c538a580952befebe902e58279d093405b2cce079fb0171430.jpg)

![eb2ce3bb573a7111380be684fe86e2fd63814ea4f304cc0b5b13219a3e9428b5.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/eb2ce3bb573a7111380be684fe86e2fd63814ea4f304cc0b5b13219a3e9428b5.jpg)

![f033fc0b06ee6fa1bf44a024092279c03107e902d467fcd258d18bf548b1373a.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/images/f033fc0b06ee6fa1bf44a024092279c03107e902d467fcd258d18bf548b1373a.jpg)

### Tables

![2c37b4a63c685c0056dd5430dff287cfa2a3fe06755b1a29db4f0db81076eaae.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/tables/2c37b4a63c685c0056dd5430dff287cfa2a3fe06755b1a29db4f0db81076eaae.jpg)

![7f97db900f6cb4ee3052307f1047dd78f7961bf08d759fa8ffab322fedff656d.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/tables/7f97db900f6cb4ee3052307f1047dd78f7961bf08d759fa8ffab322fedff656d.jpg)

![b6bd6b1d27fb089b9d87c21ef9e2b2aa862efdce0d81c1eb63563be36cb9f84c.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/tables/b6bd6b1d27fb089b9d87c21ef9e2b2aa862efdce0d81c1eb63563be36cb9f84c.jpg)

![c6e32a46459f588a068ed810dc0acff858a9f33f3ff441abf2195ccb3b5eabba.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/tables/c6e32a46459f588a068ed810dc0acff858a9f33f3ff441abf2195ccb3b5eabba.jpg)

![ff34ea17cd910f871fa596371f142bc059cd6cda4a9c2cc0d70e319f5ee5c20a.jpg](../icml_results/470_DocVXQA_%20Context-Aware%20Visual%20Explanations%20for%20Document%20Question%20Answering/tables/ff34ea17cd910f871fa596371f142bc059cd6cda4a9c2cc0d70e319f5ee5c20a.jpg)

## Bayesian Active Learning for Bivariate Causal Discovery


### Images

![00d9a858a9e0be7429e9c3cf5f8aa1a76a3760f0e419c2d0475dcf6a7f162e48.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/00d9a858a9e0be7429e9c3cf5f8aa1a76a3760f0e419c2d0475dcf6a7f162e48.jpg)

![0be5289773687ee70c07169793acd8fbf37c873b42464f8e80016f714032bcde.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/0be5289773687ee70c07169793acd8fbf37c873b42464f8e80016f714032bcde.jpg)

![1b77ce4217a132c42f6daf0444798694fccbeae20530cf992770de3bc6e798b5.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/1b77ce4217a132c42f6daf0444798694fccbeae20530cf992770de3bc6e798b5.jpg)

![343e6c743ed5d60308baf240dc93f118561df38fa419b60880bd89128cdc2f12.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/343e6c743ed5d60308baf240dc93f118561df38fa419b60880bd89128cdc2f12.jpg)

![8336c7429e83054d94c8308155ca04d293618b45c2d0a1b67c5e2b003fa52b73.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/8336c7429e83054d94c8308155ca04d293618b45c2d0a1b67c5e2b003fa52b73.jpg)

![84b69aa905409b9416d3619b21ce6233cdea86c4de70b0a9e785aa90ff4272f0.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/84b69aa905409b9416d3619b21ce6233cdea86c4de70b0a9e785aa90ff4272f0.jpg)

![85c58e461a2188cc8b0586696df061f261b15decd98c090a0a333e6ccb87e965.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/85c58e461a2188cc8b0586696df061f261b15decd98c090a0a333e6ccb87e965.jpg)

![87a6a3d0f024e451ea09d0fd39273c7869b9a08c3543cca1593df98578dc1ec3.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/87a6a3d0f024e451ea09d0fd39273c7869b9a08c3543cca1593df98578dc1ec3.jpg)

![b498a0854f9fcec72bba27083cd3ff99156b857f4936203bae1b6c4be3d53d51.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/b498a0854f9fcec72bba27083cd3ff99156b857f4936203bae1b6c4be3d53d51.jpg)

![bb7596c47e0439763cc20cbff9aca920394129629be4474bad397442bbef0c91.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/bb7596c47e0439763cc20cbff9aca920394129629be4474bad397442bbef0c91.jpg)

![c0c911a4279de537b44ebe46720af8aca3cced27b42653bff7fb47248a755fe8.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/c0c911a4279de537b44ebe46720af8aca3cced27b42653bff7fb47248a755fe8.jpg)

![c5d6ffd030c0a336083cb2c254e629d2f112161ca01686a45d936386b389657a.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/c5d6ffd030c0a336083cb2c254e629d2f112161ca01686a45d936386b389657a.jpg)

![c8c1067336e528f28837a215f54e76b57cb223a80053f9862a8cb45823b69e72.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/images/c8c1067336e528f28837a215f54e76b57cb223a80053f9862a8cb45823b69e72.jpg)

### Tables

![3bf9d7ad1b56f2314787bdbc1d24a55fe2c71ef3f0a6efd29b0ca3fbb95670ba.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/tables/3bf9d7ad1b56f2314787bdbc1d24a55fe2c71ef3f0a6efd29b0ca3fbb95670ba.jpg)

![5ca10e406dedd93ab49ecd66005eb9657152b6a90f6fb7977d83e0a5fb5149c1.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/tables/5ca10e406dedd93ab49ecd66005eb9657152b6a90f6fb7977d83e0a5fb5149c1.jpg)

![ee2f3c5f4658973323e1f363fd591dee8aecb376a34c457d601e1ff8c97cf4f3.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/tables/ee2f3c5f4658973323e1f363fd591dee8aecb376a34c457d601e1ff8c97cf4f3.jpg)

![f11cdb6da736bd4323f9b157610c4426038d90ecdfb80f7f5ec46fddd22c1c5d.jpg](../icml_results/471_Bayesian%20Active%20Learning%20for%20Bivariate%20Causal%20Discovery/tables/f11cdb6da736bd4323f9b157610c4426038d90ecdfb80f7f5ec46fddd22c1c5d.jpg)

## Ensemble Distribution Distillation via Flow Matching


### Images

![0dc6c5f1e5f7eff7d75599a95f5ffbade997e5fbf67c6e1f57657ee6f3bbc9de.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/images/0dc6c5f1e5f7eff7d75599a95f5ffbade997e5fbf67c6e1f57657ee6f3bbc9de.jpg)

![3ff75bc2be67abeb79d4827bed69cf85220bf4913df9373da40c8102286a8835.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/images/3ff75bc2be67abeb79d4827bed69cf85220bf4913df9373da40c8102286a8835.jpg)

![bfa9ff1bae9ae20287db1e240eff3dbd915a0ce943520e1e9ab648e931cf858f.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/images/bfa9ff1bae9ae20287db1e240eff3dbd915a0ce943520e1e9ab648e931cf858f.jpg)

![e91c805c32f41e4f1cb901c941ccf39c42f1d486512431f766460ed6b5f1e176.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/images/e91c805c32f41e4f1cb901c941ccf39c42f1d486512431f766460ed6b5f1e176.jpg)

![f754de4b2ed4d6f99e0fadac4f4d23f0027dc162c70b9c7e4d0b83f81dce4055.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/images/f754de4b2ed4d6f99e0fadac4f4d23f0027dc162c70b9c7e4d0b83f81dce4055.jpg)

### Tables

![03de71a70d386bbad147debcdae23285bc18a083b7e97fc57a0c7e311386a5f8.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/03de71a70d386bbad147debcdae23285bc18a083b7e97fc57a0c7e311386a5f8.jpg)

![0f85dcdeeca3bd03f6f6d5addd0f0094e20bd992dd104ab3541bc404c65800ba.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/0f85dcdeeca3bd03f6f6d5addd0f0094e20bd992dd104ab3541bc404c65800ba.jpg)

![31cc26f883adc5ea13cbc46de7149ef2b80593008f9b6d3b9e849124f7d03df0.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/31cc26f883adc5ea13cbc46de7149ef2b80593008f9b6d3b9e849124f7d03df0.jpg)

![4b8c62f1effc5e1a43ffc4aae9962f8563b0d9b919b48d60d639d784bd1fd6a3.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/4b8c62f1effc5e1a43ffc4aae9962f8563b0d9b919b48d60d639d784bd1fd6a3.jpg)

![520ce891c98b6e8fd5bfc7f0a5df949f11609466521d3398da17ac8b8d0abb72.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/520ce891c98b6e8fd5bfc7f0a5df949f11609466521d3398da17ac8b8d0abb72.jpg)

![5d39bcee39b6c684068a32397713fbcfe7b0571230a426cf2a4bff02c754b379.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/5d39bcee39b6c684068a32397713fbcfe7b0571230a426cf2a4bff02c754b379.jpg)

![5ec3adb3ff65dac12df9640cbead1562ec7d92ec79b302c5f24416cbe4e434c2.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/5ec3adb3ff65dac12df9640cbead1562ec7d92ec79b302c5f24416cbe4e434c2.jpg)

![7a9b1fc654924d44ce4d82601aa4f921aa9360de8f22e85980c982d654dce5cd.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/7a9b1fc654924d44ce4d82601aa4f921aa9360de8f22e85980c982d654dce5cd.jpg)

![a41ebf7243585f342b8570f750756c71134262636dcefaa0f4883aaf6269bc84.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/a41ebf7243585f342b8570f750756c71134262636dcefaa0f4883aaf6269bc84.jpg)

![fe3904e67e2ead60f0c525878a628188691e617f75d5da2174d3ed50055d8287.jpg](../icml_results/472_Ensemble%20Distribution%20Distillation%20via%20Flow%20Matching/tables/fe3904e67e2ead60f0c525878a628188691e617f75d5da2174d3ed50055d8287.jpg)

## Automatic Differentiation of Optimization Algorithms with Time-Varying Updates


### Images

![d4d19053a7094f7c7d5edf8d5bce556219f71c663cbd8e2417517cf6fdec72a8.jpg](../icml_results/473_Automatic%20Differentiation%20of%20Optimization%20Algorithms%20with%20Time-Varying%20Updates/images/d4d19053a7094f7c7d5edf8d5bce556219f71c663cbd8e2417517cf6fdec72a8.jpg)

## How Contaminated Is Your Benchmark? Measuring Dataset Leakage in Large Language Models with Kernel Divergence


### Images

![22508621e4b1b51722a4aa0646ce943c515da16b4fbd413fe22c055282977f82.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/images/22508621e4b1b51722a4aa0646ce943c515da16b4fbd413fe22c055282977f82.jpg)

![2d0ed54dcafd9f0b9eda6443b6dd1a96845f65a2cc19b14a5da32ec473a487bd.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/images/2d0ed54dcafd9f0b9eda6443b6dd1a96845f65a2cc19b14a5da32ec473a487bd.jpg)

![4bc3c7b58957f0fe382f798e544031773a167f42ecb8ef345e4f8334049fcfc3.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/images/4bc3c7b58957f0fe382f798e544031773a167f42ecb8ef345e4f8334049fcfc3.jpg)

![5ec12b54b296108206aa93249a0d778b4b1c2bd8ebe6e7eb9725ef590e879fa2.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/images/5ec12b54b296108206aa93249a0d778b4b1c2bd8ebe6e7eb9725ef590e879fa2.jpg)

![a5889fcbb089ec9d0578fa0e20bbe3586531cd48a177061c1ae974db5a5d7f11.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/images/a5889fcbb089ec9d0578fa0e20bbe3586531cd48a177061c1ae974db5a5d7f11.jpg)

![e471edd34211bb4a00e159d6ef97ea2b567b65c3d15e62eee4a1dbda386b1b98.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/images/e471edd34211bb4a00e159d6ef97ea2b567b65c3d15e62eee4a1dbda386b1b98.jpg)

![edd431ab7de77b892a159538bf9a49b3909df672e8b565a0674da3c9f594fa4b.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/images/edd431ab7de77b892a159538bf9a49b3909df672e8b565a0674da3c9f594fa4b.jpg)

### Tables

![03e6421a3f0c7fe287fec5e99cb0c368e7d7f945ebbd5640399cc9aa001561e7.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/03e6421a3f0c7fe287fec5e99cb0c368e7d7f945ebbd5640399cc9aa001561e7.jpg)

![2dbc1d5fbe3105fb2a963a2911f1bf64f9dedd776202cbe9ad9bfbbb8dc74466.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/2dbc1d5fbe3105fb2a963a2911f1bf64f9dedd776202cbe9ad9bfbbb8dc74466.jpg)

![39c41a5a0825026a60fa5528dedb0c45536c3c4368cae761fdb7220a42bf99b6.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/39c41a5a0825026a60fa5528dedb0c45536c3c4368cae761fdb7220a42bf99b6.jpg)

![4905a3b7f50ef73bb7e4fe98c75319c3e0a843303b75549ab899b382ce960d46.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/4905a3b7f50ef73bb7e4fe98c75319c3e0a843303b75549ab899b382ce960d46.jpg)

![509a7e8ae99fbdc5da1b11c16b879858fa000cad034a1276d86dfe351fee5f18.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/509a7e8ae99fbdc5da1b11c16b879858fa000cad034a1276d86dfe351fee5f18.jpg)

![6d1adbf6a2b419f9e890e2c772e48b822861f041ba7345bc501e9c61e69930bb.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/6d1adbf6a2b419f9e890e2c772e48b822861f041ba7345bc501e9c61e69930bb.jpg)

![750d333e3d3502e82c7ab4cf7cf32a45e4431f471b00bde8e51eb566cfe52cf1.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/750d333e3d3502e82c7ab4cf7cf32a45e4431f471b00bde8e51eb566cfe52cf1.jpg)

![810defa123ceb2e841de43547d67c4e472456fc0a92bef533d491120221c9a38.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/810defa123ceb2e841de43547d67c4e472456fc0a92bef533d491120221c9a38.jpg)

![abf0831e9f4d9e62227c21843f489709d46d2d09b284a53c2dbb1750e6b894a0.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/abf0831e9f4d9e62227c21843f489709d46d2d09b284a53c2dbb1750e6b894a0.jpg)

![d2e9f59c528e6ad3548b39ae0640fa046261232f619ddcb456d6154e7375b9bc.jpg](../icml_results/474_How%20Contaminated%20Is%20Your%20Benchmark_%20Measuring%20Dataset%20Leakage%20in%20Large%20Language%20Models%20with%20Kernel%20D/tables/d2e9f59c528e6ad3548b39ae0640fa046261232f619ddcb456d6154e7375b9bc.jpg)

## Revisiting Chain-of-Thought in Code Generation: Do Language Models Need to Learn Reasoning before Coding?


### Images

![0564482f3004246ea566e673e94f792a74ce60005cfd0c4bd99745911f81f899.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/0564482f3004246ea566e673e94f792a74ce60005cfd0c4bd99745911f81f899.jpg)

![0bff316d53950b80638fc2d7914ddeb9fe41d7903fd5e04bf0d68925f40d76d9.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/0bff316d53950b80638fc2d7914ddeb9fe41d7903fd5e04bf0d68925f40d76d9.jpg)

![11635b064cbfcaed5afab93cf41a783669dbabaaeae56005a3920caf73757e33.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/11635b064cbfcaed5afab93cf41a783669dbabaaeae56005a3920caf73757e33.jpg)

![1b1da400a6267e6cd5de7c3daa415085390c2eda4eda66116a4d6210240cca25.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/1b1da400a6267e6cd5de7c3daa415085390c2eda4eda66116a4d6210240cca25.jpg)

![28d0a640b8230f6fe6784e6b0333199d71dd28fa7bb8ad82f6770442876be6b8.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/28d0a640b8230f6fe6784e6b0333199d71dd28fa7bb8ad82f6770442876be6b8.jpg)

![2dd98232312f4e14bd8377ad83f1e8c9cb64693ec45ede77630fa68fb9d98e0a.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/2dd98232312f4e14bd8377ad83f1e8c9cb64693ec45ede77630fa68fb9d98e0a.jpg)

![648b78d10d5d62f9cfb78c004c253719c63acd6534573130e65f4a60a4654c97.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/648b78d10d5d62f9cfb78c004c253719c63acd6534573130e65f4a60a4654c97.jpg)

![676b1195c27c096bb95e7bd4b17c19b3a261e4ae681fddac7ef33ad174a23c52.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/676b1195c27c096bb95e7bd4b17c19b3a261e4ae681fddac7ef33ad174a23c52.jpg)

![76c59546cd7be4f215d351fe57f34419d096ba28e845a39c2d38bdd6b2a6d25f.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/76c59546cd7be4f215d351fe57f34419d096ba28e845a39c2d38bdd6b2a6d25f.jpg)

![9f055b62efdf7bba2ad1ba6054c9a4ff702af5f90aac6cf60b1117747dd331a7.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/9f055b62efdf7bba2ad1ba6054c9a4ff702af5f90aac6cf60b1117747dd331a7.jpg)

![bf9ea2629b22c86b3e76fcdbaf14c4707c8874fef59647f47ac8c61572f280f7.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/bf9ea2629b22c86b3e76fcdbaf14c4707c8874fef59647f47ac8c61572f280f7.jpg)

![d57a6645a9c80b05b64e066f1b18c1361af876e220afbd572cf0cc9bc3237a56.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/d57a6645a9c80b05b64e066f1b18c1361af876e220afbd572cf0cc9bc3237a56.jpg)

![e9fd6c2dad690b4e2377d519e23712553b936bba8a94ccbd29fb636eae366989.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/e9fd6c2dad690b4e2377d519e23712553b936bba8a94ccbd29fb636eae366989.jpg)

![f1e9a4fd62109010da2acd0bedfed36b26825a109cf306582862f11ea2f2920a.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/f1e9a4fd62109010da2acd0bedfed36b26825a109cf306582862f11ea2f2920a.jpg)

![f9b2fe627aac639debc656edfa24ddd7c9539843907c2e63ca96ff3ab86d7276.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/images/f9b2fe627aac639debc656edfa24ddd7c9539843907c2e63ca96ff3ab86d7276.jpg)

### Tables

![04186ce2c4ef94bf24cb20df93e68130df2dc67c3bc2a1b4ecb57acbf46a0574.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/04186ce2c4ef94bf24cb20df93e68130df2dc67c3bc2a1b4ecb57acbf46a0574.jpg)

![11dfdb70d15b36d23f483afbbb6ff1740b38cfe60c0d4a7400971e8b47ecace0.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/11dfdb70d15b36d23f483afbbb6ff1740b38cfe60c0d4a7400971e8b47ecace0.jpg)

![2fe229578b00ef4778cae8fb3f5d348ced30671feb0067c0bfe94c0f8620b859.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/2fe229578b00ef4778cae8fb3f5d348ced30671feb0067c0bfe94c0f8620b859.jpg)

![30f84b43290b17331c0e779c4550adc237810a05b23dde7a1d173becf151b204.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/30f84b43290b17331c0e779c4550adc237810a05b23dde7a1d173becf151b204.jpg)

![359e1a25c72a2efe9fecb11ec54eb6c6e84244f5aeda481ad78d65b570b13506.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/359e1a25c72a2efe9fecb11ec54eb6c6e84244f5aeda481ad78d65b570b13506.jpg)

![3a38de79d8ac9ec55827aa165e4c8604ea4ade84de8ce2b754243d449c37a806.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/3a38de79d8ac9ec55827aa165e4c8604ea4ade84de8ce2b754243d449c37a806.jpg)

![428a30236f80b24753130c580f0359d2818aed79a6083c7c91c948fbd7f4d340.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/428a30236f80b24753130c580f0359d2818aed79a6083c7c91c948fbd7f4d340.jpg)

![59018e982d6f4da8595a9ad73833605bfe176699b7b794e4aaa7e430d840d905.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/59018e982d6f4da8595a9ad73833605bfe176699b7b794e4aaa7e430d840d905.jpg)

![5eb826e8871f9102675ee91744285f8fadde85d23dc98ec6559ec67d6baed077.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/5eb826e8871f9102675ee91744285f8fadde85d23dc98ec6559ec67d6baed077.jpg)

![74fc19ca45235cb9ea5a9029935777297fed4c21fcb3427f6eb9f108df25d326.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/74fc19ca45235cb9ea5a9029935777297fed4c21fcb3427f6eb9f108df25d326.jpg)

![bf3c917e0df75fbaadfdc51a7ed35cd084e05044d253e55f23fbafee004a9eb5.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/bf3c917e0df75fbaadfdc51a7ed35cd084e05044d253e55f23fbafee004a9eb5.jpg)

![cf6a1101415ab5bd4eabf08afdd4d96f99bfeb426d0be2bf5650f5e0c60fa3d1.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/cf6a1101415ab5bd4eabf08afdd4d96f99bfeb426d0be2bf5650f5e0c60fa3d1.jpg)

![ee3f343f77c8e1a774696d169f65cf679b3f760ecdd54e9a573231ce6a152fc0.jpg](../icml_results/475_Revisiting%20Chain-of-Thought%20in%20Code%20Generation_%20Do%20Language%20Models%20Need%20to%20Learn%20Reasoning%20before%20Co/tables/ee3f343f77c8e1a774696d169f65cf679b3f760ecdd54e9a573231ce6a152fc0.jpg)

## A Theoretical Study of (Hyper) Self-Attention through the Lens of Interactions: Representation, Training, Generalization


### Images

![3d77163f3a64df6d1a42c3224546be83cc37f344b0701b7ec4ebe11fac450099.jpg](../icml_results/476_A%20Theoretical%20Study%20of%20%28Hyper) Self-Attention through the Lens of Interactions_ Representation, Trai/images/3d77163f3a64df6d1a42c3224546be83cc37f344b0701b7ec4ebe11fac450099.jpg)

![59dfd725ff511ef4264fbb8382cb12574ebbfc343c9f71b1a9434c1e32da20bd.jpg](../icml_results/476_A%20Theoretical%20Study%20of%20%28Hyper) Self-Attention through the Lens of Interactions_ Representation, Trai/images/59dfd725ff511ef4264fbb8382cb12574ebbfc343c9f71b1a9434c1e32da20bd.jpg)

![a05c466a4f93e88605f59eed7cc4cd6c254065a7240302396e690f72d78e4edb.jpg](../icml_results/476_A%20Theoretical%20Study%20of%20%28Hyper) Self-Attention through the Lens of Interactions_ Representation, Trai/images/a05c466a4f93e88605f59eed7cc4cd6c254065a7240302396e690f72d78e4edb.jpg)

![d3f8183619900a2e734ebb4c9185f7cda2c74e9e2d537af027f26035dacf2423.jpg](../icml_results/476_A%20Theoretical%20Study%20of%20%28Hyper) Self-Attention through the Lens of Interactions_ Representation, Trai/images/d3f8183619900a2e734ebb4c9185f7cda2c74e9e2d537af027f26035dacf2423.jpg)

### Tables

![11d6ade4c0cb1cbe1b090906c02f31975bf55547dfb8300b196c0fd92dac2b68.jpg](../icml_results/476_A%20Theoretical%20Study%20of%20%28Hyper) Self-Attention through the Lens of Interactions_ Representation, Trai/tables/11d6ade4c0cb1cbe1b090906c02f31975bf55547dfb8300b196c0fd92dac2b68.jpg)

![3b58b1a96d8aced65f9addec54744b394bc11724d5d2bb1f89a839b07a18c9cd.jpg](../icml_results/476_A%20Theoretical%20Study%20of%20%28Hyper) Self-Attention through the Lens of Interactions_ Representation, Trai/tables/3b58b1a96d8aced65f9addec54744b394bc11724d5d2bb1f89a839b07a18c9cd.jpg)

![4a4eb16fcba622fb00fb86eb7c7dd1777e7bf03f4a835c43e969579b657a212a.jpg](../icml_results/476_A%20Theoretical%20Study%20of%20%28Hyper) Self-Attention through the Lens of Interactions_ Representation, Trai/tables/4a4eb16fcba622fb00fb86eb7c7dd1777e7bf03f4a835c43e969579b657a212a.jpg)

## EnsLoss: Stochastic Calibrated Loss Ensembles for Preventing Overfitting in Classification


### Images

![0e7e8da110a0fb21d7f7576cc6d5a243d4dde349a53cbbb2eaf50ee0c2b5d707.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/images/0e7e8da110a0fb21d7f7576cc6d5a243d4dde349a53cbbb2eaf50ee0c2b5d707.jpg)

![249b756255815cbf65c926e78e3fa6af71a934053d314783122bc75b55bbb63e.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/images/249b756255815cbf65c926e78e3fa6af71a934053d314783122bc75b55bbb63e.jpg)

![474ca08445d199487ff5b960ed007b422c10dab8a8e644ce8b62700183d141d1.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/images/474ca08445d199487ff5b960ed007b422c10dab8a8e644ce8b62700183d141d1.jpg)

![b4720c04135770f694ad2857c94cee5eeb21ba0052a3906ff1c94417c1010bc0.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/images/b4720c04135770f694ad2857c94cee5eeb21ba0052a3906ff1c94417c1010bc0.jpg)

### Tables

![267523379701b80a83f1a5f636851512e88883805ccd5a6a7732435ede2a2f98.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/267523379701b80a83f1a5f636851512e88883805ccd5a6a7732435ede2a2f98.jpg)

![2ca3aac7e8ad34daf0663ee284d119ad23eb6e0f2343ff685766dd65fcdb72ec.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/2ca3aac7e8ad34daf0663ee284d119ad23eb6e0f2343ff685766dd65fcdb72ec.jpg)

![2d68a5ca9d333735c6ef5f8e181f8169456cdfa36a7ccc363bef6186a5a43c62.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/2d68a5ca9d333735c6ef5f8e181f8169456cdfa36a7ccc363bef6186a5a43c62.jpg)

![3112e0952087841fcc35e7eb2772b604a5518bd0883693dd1ee0ca47653fe2da.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/3112e0952087841fcc35e7eb2772b604a5518bd0883693dd1ee0ca47653fe2da.jpg)

![53a86cefc473cb2e5728f08648bf793c5d247ce8d62edf42cd6f413183e65bed.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/53a86cefc473cb2e5728f08648bf793c5d247ce8d62edf42cd6f413183e65bed.jpg)

![58760a52817958b1164601049ad8b8ba2c5a747febc395b7887bf575427b13c7.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/58760a52817958b1164601049ad8b8ba2c5a747febc395b7887bf575427b13c7.jpg)

![621f7e63109aa7769cb9607f201b93c042bb7851f2b447c308bf418f2a85875c.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/621f7e63109aa7769cb9607f201b93c042bb7851f2b447c308bf418f2a85875c.jpg)

![8b6f63416c6906fbeea314a74a4b10eb285533888cea00c47f5c1d1b52b107fc.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/8b6f63416c6906fbeea314a74a4b10eb285533888cea00c47f5c1d1b52b107fc.jpg)

![c5f4b0eb8c8ae76e2ba7f7ba63cb8ccdf08bd692564d3b40e65de0daeb005bcb.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/c5f4b0eb8c8ae76e2ba7f7ba63cb8ccdf08bd692564d3b40e65de0daeb005bcb.jpg)

![e0cd729a9a93d740b7940f4b2f4c917268a5a5cb59fc9694f18c6ed896e8b198.jpg](../icml_results/477_EnsLoss_%20Stochastic%20Calibrated%20Loss%20Ensembles%20for%20Preventing%20Overfitting%20in%20Classification/tables/e0cd729a9a93d740b7940f4b2f4c917268a5a5cb59fc9694f18c6ed896e8b198.jpg)

## CRANE: Reasoning with constrained LLM generation


### Images

![0d9d6eacdd6fd86f40e382a1c8d18b746fba9d3934c96244d5172c252278920d.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/images/0d9d6eacdd6fd86f40e382a1c8d18b746fba9d3934c96244d5172c252278920d.jpg)

![5276d2d7c80e39a031a96f3cf42c43ef5683afd38b064fd160666d605a91bbfa.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/images/5276d2d7c80e39a031a96f3cf42c43ef5683afd38b064fd160666d605a91bbfa.jpg)

![de47b696152e7dcb123aa8815f0300b79173fb26d69cd955ca1f3b6e8eb921bf.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/images/de47b696152e7dcb123aa8815f0300b79173fb26d69cd955ca1f3b6e8eb921bf.jpg)

![ef3c5d165493cbd3a3270310c30a647164fa746ff83cbaa4c451fb45326ae9c6.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/images/ef3c5d165493cbd3a3270310c30a647164fa746ff83cbaa4c451fb45326ae9c6.jpg)

### Tables

![23d0d62a99885fa0feb756bfd3748b6f4e31b60ea24377c0e79e15d6f49d9b05.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/tables/23d0d62a99885fa0feb756bfd3748b6f4e31b60ea24377c0e79e15d6f49d9b05.jpg)

![2672be3da1acb30e1276c6fd2f2f0ab80edc3fd6af9f5cd760211915e080bfea.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/tables/2672be3da1acb30e1276c6fd2f2f0ab80edc3fd6af9f5cd760211915e080bfea.jpg)

![27d652026966f0ebd60c17cd916746afeddf1581d4faa582a21f15b956e4ce1c.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/tables/27d652026966f0ebd60c17cd916746afeddf1581d4faa582a21f15b956e4ce1c.jpg)

![43596157eab1d2d7b232b1d3e3abfe4baf982360ce111f8cde57536ed6c7cc9b.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/tables/43596157eab1d2d7b232b1d3e3abfe4baf982360ce111f8cde57536ed6c7cc9b.jpg)

![7c247aa3b4835a30df10cf516998e94e22d91694115f6f3672a4a68822058546.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/tables/7c247aa3b4835a30df10cf516998e94e22d91694115f6f3672a4a68822058546.jpg)

![f9446bcd2c8c154b182021372e070e42015eeaade84ed9cd521a2a081bdcc119.jpg](../icml_results/478_CRANE_%20Reasoning%20with%20constrained%20LLM%20generation/tables/f9446bcd2c8c154b182021372e070e42015eeaade84ed9cd521a2a081bdcc119.jpg)

## Collapse-Proof Non-Contrastive Self-Supervised Learning


### Images

![029759b272a9361a922ab904c0b8fad632f4024564bb6cb2de2bd8bf7d8c661c.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/029759b272a9361a922ab904c0b8fad632f4024564bb6cb2de2bd8bf7d8c661c.jpg)

![14597089988c49fa431e5c27b34f544fdebb7c0360544d286c51f5dcd2160c73.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/14597089988c49fa431e5c27b34f544fdebb7c0360544d286c51f5dcd2160c73.jpg)

![1ebcc787a389329d44b09eb5b62e41610528ab13c216617e710b4c860d491511.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/1ebcc787a389329d44b09eb5b62e41610528ab13c216617e710b4c860d491511.jpg)

![3fb09e97be981ba8d2979faad50736266b56616e10f88c8777416016badeae6f.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/3fb09e97be981ba8d2979faad50736266b56616e10f88c8777416016badeae6f.jpg)

![44aef9b945d62aa1c52d14accb02b905d47cf3bce2a115629586c8ebf8df2ec8.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/44aef9b945d62aa1c52d14accb02b905d47cf3bce2a115629586c8ebf8df2ec8.jpg)

![5720fcce0bba3c2a6d6acb43bfb050a9fc11c5e1e86cc677909aea22925b4774.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/5720fcce0bba3c2a6d6acb43bfb050a9fc11c5e1e86cc677909aea22925b4774.jpg)

![89b5cda1335e4649c28caa843767a990a445e5c57288890e529c4443978e17e1.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/89b5cda1335e4649c28caa843767a990a445e5c57288890e529c4443978e17e1.jpg)

![a63ea670542dc2e8502107e4b2a60a99eb390786cfacf1bb57c7c6370c8e5d29.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/a63ea670542dc2e8502107e4b2a60a99eb390786cfacf1bb57c7c6370c8e5d29.jpg)

![ad5ce9e690417b930e68f7b2b822b1759d770e3841b3bb39efaabefd83e69238.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/ad5ce9e690417b930e68f7b2b822b1759d770e3841b3bb39efaabefd83e69238.jpg)

![c5274da43c82ef7a22c2745d4a73709b333c099d642c4b3db1ae522692fa3294.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/c5274da43c82ef7a22c2745d4a73709b333c099d642c4b3db1ae522692fa3294.jpg)

![e1b12efaec88a19f32be436a7fb8b4ff9de02fb10bc12ee799ade2f170b4cb37.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/e1b12efaec88a19f32be436a7fb8b4ff9de02fb10bc12ee799ade2f170b4cb37.jpg)

![e596c4378f113845be25aa924b6dc38f9f1a5d807229c67dfde43c512497dd9e.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/e596c4378f113845be25aa924b6dc38f9f1a5d807229c67dfde43c512497dd9e.jpg)

![e994f064c270809e34f64c0f5a41b72845766cd3889c28c0706a52c4161f35f1.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/e994f064c270809e34f64c0f5a41b72845766cd3889c28c0706a52c4161f35f1.jpg)

![efffb68094c47d22244e94c7431a051febe15892b0434937ced5b9bea7a8a106.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/efffb68094c47d22244e94c7431a051febe15892b0434937ced5b9bea7a8a106.jpg)

![f1a6300b34cd6ccc5f9ef02d5e9f72352e56d6f7156ccc037e6faf4c34692508.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/f1a6300b34cd6ccc5f9ef02d5e9f72352e56d6f7156ccc037e6faf4c34692508.jpg)

![fc8925e5880da9450d7efe94f56937bfcd5c895cada0a2ac11d7ebda26886d92.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/fc8925e5880da9450d7efe94f56937bfcd5c895cada0a2ac11d7ebda26886d92.jpg)

![ffcbc47c269ca0af2f4bdf1724ee75522acb53329e6e155b6597c67e44eb5dd6.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/images/ffcbc47c269ca0af2f4bdf1724ee75522acb53329e6e155b6597c67e44eb5dd6.jpg)

### Tables

![03d92163e1e7ccee447138d2ceb03ea6c45018925ce749657b5b528630947da6.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/03d92163e1e7ccee447138d2ceb03ea6c45018925ce749657b5b528630947da6.jpg)

![16d9fbf1134a9244638982928dd989d7148293e09ffc7b234cdb0b2e79128d48.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/16d9fbf1134a9244638982928dd989d7148293e09ffc7b234cdb0b2e79128d48.jpg)

![431b1b7f50bae8773182a75f33fe6e7cbaf60e6ab6320608249036a7eb22eaca.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/431b1b7f50bae8773182a75f33fe6e7cbaf60e6ab6320608249036a7eb22eaca.jpg)

![5e094c7322104cfa14df59df83ebd2e7c13a537b16795569c9769edb8744cd8e.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/5e094c7322104cfa14df59df83ebd2e7c13a537b16795569c9769edb8744cd8e.jpg)

![a32d9d14e88b8317cfd36f85e44b2ad6973e38aee0098fbb4b02cdf5a4ffc047.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/a32d9d14e88b8317cfd36f85e44b2ad6973e38aee0098fbb4b02cdf5a4ffc047.jpg)

![bb41bf8d70c81e1f31359c4568012fcf108c5da45e336b5e6c50a960252274e1.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/bb41bf8d70c81e1f31359c4568012fcf108c5da45e336b5e6c50a960252274e1.jpg)

![bc1beb5e42983b62a1a2a72433a8e610b562abc6195a87bc29c74f5b84382498.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/bc1beb5e42983b62a1a2a72433a8e610b562abc6195a87bc29c74f5b84382498.jpg)

![eb5618db2232af8c709bbefab3b00c80346e7d9ec0c755f440324c374646d7bc.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/eb5618db2232af8c709bbefab3b00c80346e7d9ec0c755f440324c374646d7bc.jpg)

![fcec7007341a8a33f21b37ad85ca4bcf06ea66b2cdca913d01178eba3158938b.jpg](../icml_results/479_Collapse-Proof%20Non-Contrastive%20Self-Supervised%20Learning/tables/fcec7007341a8a33f21b37ad85ca4bcf06ea66b2cdca913d01178eba3158938b.jpg)

## Relating Misfit to Gain in Weak-to-Strong Generalization Beyond the Squared Loss


### Images

![60ccd9437ead05a3b9d718af658caef44a89607360c1b089fbb80ea3411e4ef2.jpg](../icml_results/480_Relating%20Misfit%20to%20Gain%20in%20Weak-to-Strong%20Generalization%20Beyond%20the%20Squared%20Loss/images/60ccd9437ead05a3b9d718af658caef44a89607360c1b089fbb80ea3411e4ef2.jpg)

![946835a30c300bbd5dada825c90b9b66b8d8f9d368335ed85c6e8b0c35d791d6.jpg](../icml_results/480_Relating%20Misfit%20to%20Gain%20in%20Weak-to-Strong%20Generalization%20Beyond%20the%20Squared%20Loss/images/946835a30c300bbd5dada825c90b9b66b8d8f9d368335ed85c6e8b0c35d791d6.jpg)

![9b22baecd8add760ce65452f85357667d5190cc589fc3b2c0e8373c7b5081ba5.jpg](../icml_results/480_Relating%20Misfit%20to%20Gain%20in%20Weak-to-Strong%20Generalization%20Beyond%20the%20Squared%20Loss/images/9b22baecd8add760ce65452f85357667d5190cc589fc3b2c0e8373c7b5081ba5.jpg)

![cffb8e222affeff818bfa50a4b974c95923f3b667c542619dbb3fd34c49fe692.jpg](../icml_results/480_Relating%20Misfit%20to%20Gain%20in%20Weak-to-Strong%20Generalization%20Beyond%20the%20Squared%20Loss/images/cffb8e222affeff818bfa50a4b974c95923f3b667c542619dbb3fd34c49fe692.jpg)

### Tables

![7dac9ab93e8d45dff9ebec24ec878e35289e5fe681aea292a9edfea23e4aec4f.jpg](../icml_results/480_Relating%20Misfit%20to%20Gain%20in%20Weak-to-Strong%20Generalization%20Beyond%20the%20Squared%20Loss/tables/7dac9ab93e8d45dff9ebec24ec878e35289e5fe681aea292a9edfea23e4aec4f.jpg)

![aa698ba1d246c3faad017c410bc36e8a3516bafd5b6468bcb2da753965a58fb2.jpg](../icml_results/480_Relating%20Misfit%20to%20Gain%20in%20Weak-to-Strong%20Generalization%20Beyond%20the%20Squared%20Loss/tables/aa698ba1d246c3faad017c410bc36e8a3516bafd5b6468bcb2da753965a58fb2.jpg)

## On Exact Bit-level Reversible Transformers Without Changing Architecture


### Images

![134e2e81767e815d70d89857afd98c51d07aa4528bfc5586a31828810067f4f3.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/134e2e81767e815d70d89857afd98c51d07aa4528bfc5586a31828810067f4f3.jpg)

![14bda139f0fdd6d9441451b69fd103b569e22de960b92304dcb17b7eeb4819c0.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/14bda139f0fdd6d9441451b69fd103b569e22de960b92304dcb17b7eeb4819c0.jpg)

![1955674b20fb90f9be82d2c21fd349842e2404ee0d66cdb8376bd7b16e7dd77f.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/1955674b20fb90f9be82d2c21fd349842e2404ee0d66cdb8376bd7b16e7dd77f.jpg)

![1cc13cbf34426447a4737f92eece253cee0ce92ddbfc906462276e0c523666df.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/1cc13cbf34426447a4737f92eece253cee0ce92ddbfc906462276e0c523666df.jpg)

![26cb720cf54855a6a49b82f80f3255f6394f7f82d836ceea22643bec34bc3106.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/26cb720cf54855a6a49b82f80f3255f6394f7f82d836ceea22643bec34bc3106.jpg)

![468c4de060b96bcfcf46e8b8a09678fd222fecf4941abe9c2f4ca53067856c88.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/468c4de060b96bcfcf46e8b8a09678fd222fecf4941abe9c2f4ca53067856c88.jpg)

![595f0d03af516b325835159960c5b87d8363271373c2f997bc0799dd33b45f3e.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/595f0d03af516b325835159960c5b87d8363271373c2f997bc0799dd33b45f3e.jpg)

![59b848b27ca4582de9e0aaec62162f08db148234e495eb813f6ccf840609d25d.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/59b848b27ca4582de9e0aaec62162f08db148234e495eb813f6ccf840609d25d.jpg)

![976987a5093e1ca632666374b3edaef75cd2007946d52f098dcc632298daec34.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/images/976987a5093e1ca632666374b3edaef75cd2007946d52f098dcc632298daec34.jpg)

### Tables

![20bb93bcb7f20ac05156aca3e315f025f1ec8c670e40d7610a66856d35657404.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/tables/20bb93bcb7f20ac05156aca3e315f025f1ec8c670e40d7610a66856d35657404.jpg)

![31e0582141140d7df1c8e8e3e951da44e808ad6971e1564114ecdb9fea0202f0.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/tables/31e0582141140d7df1c8e8e3e951da44e808ad6971e1564114ecdb9fea0202f0.jpg)

![5df8fdd9453c6f30af905dabbc1715e09e5de1674775ccc2faac5fc456d00123.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/tables/5df8fdd9453c6f30af905dabbc1715e09e5de1674775ccc2faac5fc456d00123.jpg)

![a774c6652410f967706f9f3cdb358ec36ecd6153c3009e7b17cbb7d0fc5b40c9.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/tables/a774c6652410f967706f9f3cdb358ec36ecd6153c3009e7b17cbb7d0fc5b40c9.jpg)

![bc3f484b120218089a8abce35b4701f5fb2f065a18df1b0b6c189cc5bfa0755a.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/tables/bc3f484b120218089a8abce35b4701f5fb2f065a18df1b0b6c189cc5bfa0755a.jpg)

![dede618bb416faab53ee9e8f482b5b650e8af3399aec1b46bd255d2e7a80af14.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/tables/dede618bb416faab53ee9e8f482b5b650e8af3399aec1b46bd255d2e7a80af14.jpg)

![ef22b6ef3f3118812b000881e465a563f270dee98d82459a09e2067e8af92090.jpg](../icml_results/481_On%20Exact%20Bit-level%20Reversible%20Transformers%20Without%20Changing%20Architecture/tables/ef22b6ef3f3118812b000881e465a563f270dee98d82459a09e2067e8af92090.jpg)

## Clipped SGD Algorithms for Performative Prediction: Tight Bounds for Stochastic Bias and Remedies


### Images

![2d1cf4306e3d7f424295f0018cfd69b4635e9e67f08882b35bd770a2357f0908.jpg](../icml_results/482_Clipped%20SGD%20Algorithms%20for%20Performative%20Prediction_%20Tight%20Bounds%20for%20Stochastic%20Bias%20and%20Remedies/images/2d1cf4306e3d7f424295f0018cfd69b4635e9e67f08882b35bd770a2357f0908.jpg)

![3cc2313695f86a1d42ecb4a38f48c548b097ab9e689c4a50bcf18c5699993aad.jpg](../icml_results/482_Clipped%20SGD%20Algorithms%20for%20Performative%20Prediction_%20Tight%20Bounds%20for%20Stochastic%20Bias%20and%20Remedies/images/3cc2313695f86a1d42ecb4a38f48c548b097ab9e689c4a50bcf18c5699993aad.jpg)

![9773c1d63554559d6f738e65aebf109da969f4afd11ec64e07d8d34e568e50bc.jpg](../icml_results/482_Clipped%20SGD%20Algorithms%20for%20Performative%20Prediction_%20Tight%20Bounds%20for%20Stochastic%20Bias%20and%20Remedies/images/9773c1d63554559d6f738e65aebf109da969f4afd11ec64e07d8d34e568e50bc.jpg)

![9ecf52e9a4fb975fcce2892cd7eb0bfa3e75b16422c1868783f8dd80e0e55fcb.jpg](../icml_results/482_Clipped%20SGD%20Algorithms%20for%20Performative%20Prediction_%20Tight%20Bounds%20for%20Stochastic%20Bias%20and%20Remedies/images/9ecf52e9a4fb975fcce2892cd7eb0bfa3e75b16422c1868783f8dd80e0e55fcb.jpg)

![d00feb1973a9405b6696082d9f4e38c18e86296ac16e027d8ec5508195da361a.jpg](../icml_results/482_Clipped%20SGD%20Algorithms%20for%20Performative%20Prediction_%20Tight%20Bounds%20for%20Stochastic%20Bias%20and%20Remedies/images/d00feb1973a9405b6696082d9f4e38c18e86296ac16e027d8ec5508195da361a.jpg)

## The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions


### Images

![0bc4c4c599fb7c00e6b51ee4a7c9da55ebc0ed1915633ecc3bf0a83b3ecae777.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/0bc4c4c599fb7c00e6b51ee4a7c9da55ebc0ed1915633ecc3bf0a83b3ecae777.jpg)

![0df8eca4efc3f74894d98f9b77acc24525f1d8256fc1ea2afc46ba3e9134ac6b.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/0df8eca4efc3f74894d98f9b77acc24525f1d8256fc1ea2afc46ba3e9134ac6b.jpg)

![1b6465402d98f6c6bfb20886f25b43ff7b04e39915f404556183f49b5205bd05.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/1b6465402d98f6c6bfb20886f25b43ff7b04e39915f404556183f49b5205bd05.jpg)

![3367fe8f62fcc146cd6135002210d55612bde50a58b2971414d15dbcf3afa590.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/3367fe8f62fcc146cd6135002210d55612bde50a58b2971414d15dbcf3afa590.jpg)

![3b8b5a7e2bdb4af79efe4afa536aa5179042049848310dae70fd322647a82f83.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/3b8b5a7e2bdb4af79efe4afa536aa5179042049848310dae70fd322647a82f83.jpg)

![41a1bb1be12096ac8ce03f26cde8e5038faf63352bc701abc2eed75ef57604f8.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/41a1bb1be12096ac8ce03f26cde8e5038faf63352bc701abc2eed75ef57604f8.jpg)

![441912eaba3e400549b5a61e2c4c949cbf6c65a30ab3fdf63905a2218faf8380.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/441912eaba3e400549b5a61e2c4c949cbf6c65a30ab3fdf63905a2218faf8380.jpg)

![4cc2eaf619ebe801b6bd867cb695797b1db7f7e4990818e1b98e7b9794c389c7.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/4cc2eaf619ebe801b6bd867cb695797b1db7f7e4990818e1b98e7b9794c389c7.jpg)

![5b3de0e4efb5cd4316255220a289da5650ebe37a4ca364bdb037ea3711c88e6e.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/5b3de0e4efb5cd4316255220a289da5650ebe37a4ca364bdb037ea3711c88e6e.jpg)

![a3e3afbb07d0b5b10e9128fb21d10778c9c4f4f275893fd560ebb5b4ec9b15e2.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/a3e3afbb07d0b5b10e9128fb21d10778c9c4f4f275893fd560ebb5b4ec9b15e2.jpg)

![b57f6921560decf8d16874a0c7e632feebe7271ad0ccb550911a63debd9bfd48.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/b57f6921560decf8d16874a0c7e632feebe7271ad0ccb550911a63debd9bfd48.jpg)

![bda2759102ae37891eb20cdfe19469acd93a382964d3d87504e482c61a0712e9.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/bda2759102ae37891eb20cdfe19469acd93a382964d3d87504e482c61a0712e9.jpg)

![e417ce6c94a3f6f2bc7140ed8a3a79b28c5a3dd5c64a8cc629f2c55659d09a89.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/images/e417ce6c94a3f6f2bc7140ed8a3a79b28c5a3dd5c64a8cc629f2c55659d09a89.jpg)

### Tables

![2e41032c70f7d009a68ee604fe4b4b4abdf6ea4f8876c97da18efbd59c7ef195.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/2e41032c70f7d009a68ee604fe4b4b4abdf6ea4f8876c97da18efbd59c7ef195.jpg)

![3115363d0542628d1a24595128ad88533fcf041dd7c0628e1c4950e2f4f66826.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/3115363d0542628d1a24595128ad88533fcf041dd7c0628e1c4950e2f4f66826.jpg)

![501915d9717ef33ae293b423b941a4d8ffccec3b7f8b8ef1c72d91375db62ccf.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/501915d9717ef33ae293b423b941a4d8ffccec3b7f8b8ef1c72d91375db62ccf.jpg)

![64c719e805df7f5c59f32f9dddd8b0ba12e07568206140fa7e5d505225e07e34.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/64c719e805df7f5c59f32f9dddd8b0ba12e07568206140fa7e5d505225e07e34.jpg)

![6ce966311795a8d0261071fa92b6ec1023f2107eee5f8d711132767d77a82a16.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/6ce966311795a8d0261071fa92b6ec1023f2107eee5f8d711132767d77a82a16.jpg)

![860c60277a415050a480ad775f3a1cff4646a6dcc38107d81de62b08e6c576b5.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/860c60277a415050a480ad775f3a1cff4646a6dcc38107d81de62b08e6c576b5.jpg)

![888ab7ea4bd33cac312d015e91257ab01ec2b0f2bdd1716215742381d63ebf1a.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/888ab7ea4bd33cac312d015e91257ab01ec2b0f2bdd1716215742381d63ebf1a.jpg)

![b721996aa3f1c8a672bddb32b61effb23909da815ca0f92a146184412849ec02.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/b721996aa3f1c8a672bddb32b61effb23909da815ca0f92a146184412849ec02.jpg)

![c78f6a9053a8a5848d71a790088f041a5372d9b9750391598679c6688a971da1.jpg](../icml_results/483_The%20Hidden%20Dimensions%20of%20LLM%20Alignment_%20A%20Multi-Dimensional%20Analysis%20of%20Orthogonal%20Safety%20Directions/tables/c78f6a9053a8a5848d71a790088f041a5372d9b9750391598679c6688a971da1.jpg)

## GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers


### Images

![2d572e780ecaf9f63915162f51aafd4197fe75bb3b68fd84541c2092964f87e9.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/images/2d572e780ecaf9f63915162f51aafd4197fe75bb3b68fd84541c2092964f87e9.jpg)

![31ac392ce51564f332d4d937d7a26b8d360a228d6fae926d72e951fb696f7c3a.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/images/31ac392ce51564f332d4d937d7a26b8d360a228d6fae926d72e951fb696f7c3a.jpg)

![5c1e2e5b2b76f0439a33f5922e346638fa13be793305f12675818952c17fe0fe.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/images/5c1e2e5b2b76f0439a33f5922e346638fa13be793305f12675818952c17fe0fe.jpg)

![64f7ab04f50379795ba69ae4a215889b6ef1079d120c4bba806d52d9c401bd52.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/images/64f7ab04f50379795ba69ae4a215889b6ef1079d120c4bba806d52d9c401bd52.jpg)

![76dcd37abc010ca8c833e699d723cc47f9127cc7eb635194282f08baeab03acc.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/images/76dcd37abc010ca8c833e699d723cc47f9127cc7eb635194282f08baeab03acc.jpg)

![a14d82f6ae95201e1f044a4f5fb939ef169b0741c80ae253e950703b879ed472.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/images/a14d82f6ae95201e1f044a4f5fb939ef169b0741c80ae253e950703b879ed472.jpg)

![d8484fb25aed23313b296caffded4986e9dda9d3fd657a5c0566033147393501.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/images/d8484fb25aed23313b296caffded4986e9dda9d3fd657a5c0566033147393501.jpg)

### Tables

![286d28c2a47ca542f78069c371436160b8d1ab63070e4555c69026b67a4f5e8b.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/286d28c2a47ca542f78069c371436160b8d1ab63070e4555c69026b67a4f5e8b.jpg)

![2f4f68b1f31228d6e3ac8c66a195f865d5ef0ad2d40eab58ac3cc14e0e458459.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/2f4f68b1f31228d6e3ac8c66a195f865d5ef0ad2d40eab58ac3cc14e0e458459.jpg)

![3cdec861acb017a5a3b50b299818bc69996a68603ae3413bacf36525a5f25aa3.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/3cdec861acb017a5a3b50b299818bc69996a68603ae3413bacf36525a5f25aa3.jpg)

![3ff0b8af97c59e2b27adf666cf9adf7e6f9e4a1c606f9cdff643571138cbbf2a.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/3ff0b8af97c59e2b27adf666cf9adf7e6f9e4a1c606f9cdff643571138cbbf2a.jpg)

![7def0bc6776f7de50088ba1c3817efc00cabb1d81ed2966e1418c4c4772db8a5.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/7def0bc6776f7de50088ba1c3817efc00cabb1d81ed2966e1418c4c4772db8a5.jpg)

![8e49d25af053b1b01fef5952e24734ee6122fb983d5c0705e030b87f05607097.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/8e49d25af053b1b01fef5952e24734ee6122fb983d5c0705e030b87f05607097.jpg)

![90c6a7d97c241b5f631302230b2d413bd582e690976f79a09e2f06d5b7c2921d.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/90c6a7d97c241b5f631302230b2d413bd582e690976f79a09e2f06d5b7c2921d.jpg)

![a32c3bbd423ff97a6e65aa2f6032488ce16a58b525c217c5f70f61dd0a76dea9.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/a32c3bbd423ff97a6e65aa2f6032488ce16a58b525c217c5f70f61dd0a76dea9.jpg)

![a68edd863059ec27c8fada39c3c542bacdf1de897cf6896fef3a043b17423a6a.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/a68edd863059ec27c8fada39c3c542bacdf1de897cf6896fef3a043b17423a6a.jpg)

![bc442ae2728662a6cb61522d989d6221e5a98acbb81fc244d131fbebc8c8ec66.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/bc442ae2728662a6cb61522d989d6221e5a98acbb81fc244d131fbebc8c8ec66.jpg)

![d7a88d53e88d98088db5fd0363a9d89e60942f163fd3635a4305e2daa899c112.jpg](../icml_results/484_GrokFormer_%20Graph%20Fourier%20Kolmogorov-Arnold%20Transformers/tables/d7a88d53e88d98088db5fd0363a9d89e60942f163fd3635a4305e2daa899c112.jpg)

## TtBA: Two-third Bridge Approach for Decision-Based Adversarial Attack


### Images

![1a698e8d9d72ef351ae28ecaa90a9b37ce9a9ced2bb19bb7301a9b80a53e2ebb.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/images/1a698e8d9d72ef351ae28ecaa90a9b37ce9a9ced2bb19bb7301a9b80a53e2ebb.jpg)

![61ac8c933d03cf1ab2469b66bfc7d14c24172bab417b7a22e77b167bfb655ede.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/images/61ac8c933d03cf1ab2469b66bfc7d14c24172bab417b7a22e77b167bfb655ede.jpg)

![72514e097d08a90df28395150fdf4e9df02cbe3d5087a2a1e94fc69a9f8886d3.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/images/72514e097d08a90df28395150fdf4e9df02cbe3d5087a2a1e94fc69a9f8886d3.jpg)

![883e4cdd8c94549e682a40ffeab2a8354ad1bf4e4c449f6acd85016ec726e80f.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/images/883e4cdd8c94549e682a40ffeab2a8354ad1bf4e4c449f6acd85016ec726e80f.jpg)

![d175095b0b65b9e543067f6abf15bc7a906457b43aad653227a8138a76bd9cba.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/images/d175095b0b65b9e543067f6abf15bc7a906457b43aad653227a8138a76bd9cba.jpg)

![dea6df2d5e2161f104b4de1bc18067b967e66bbb77f8a34556db52ff7a10d5a4.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/images/dea6df2d5e2161f104b4de1bc18067b967e66bbb77f8a34556db52ff7a10d5a4.jpg)

### Tables

![3dc7950c69f76c679adb35aecbca1ca00c6f67b14b0f4797ea631c22c40282ab.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/tables/3dc7950c69f76c679adb35aecbca1ca00c6f67b14b0f4797ea631c22c40282ab.jpg)

![524b0f6529291957cedce4b7741460f4da1c5ab8f93e9e21605325a7f0ef74c3.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/tables/524b0f6529291957cedce4b7741460f4da1c5ab8f93e9e21605325a7f0ef74c3.jpg)

![5df6063202827eea2a4a98ab1016129a435d257dc91770bef96b671af839a639.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/tables/5df6063202827eea2a4a98ab1016129a435d257dc91770bef96b671af839a639.jpg)

![a056f77d70a1ec304b67beab3b5ce3f523d4e74914437b89ab54d25eefd7e066.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/tables/a056f77d70a1ec304b67beab3b5ce3f523d4e74914437b89ab54d25eefd7e066.jpg)

![d1fb15054db7bd25d3221a2086fa6e21faaabd26b2ad2458aed61992e7f76917.jpg](../icml_results/485_TtBA_%20Two-third%20Bridge%20Approach%20for%20Decision-Based%20Adversarial%20Attack/tables/d1fb15054db7bd25d3221a2086fa6e21faaabd26b2ad2458aed61992e7f76917.jpg)

## Dissecting Submission Limit in Desk-Rejections: A Mathematical Analysis of Fairness in AI Conference Policies


### Images

![2fda6b27631fc7efb967b9a207e3c77b7406262bcdc085d22a34d1e830d383eb.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/images/2fda6b27631fc7efb967b9a207e3c77b7406262bcdc085d22a34d1e830d383eb.jpg)

![9bf9d1d19c11e1b98451cdb050747ddd1d38b29be4980ec0b335fb75b7f6c038.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/images/9bf9d1d19c11e1b98451cdb050747ddd1d38b29be4980ec0b335fb75b7f6c038.jpg)

![b067c8aecd2283868b41f98146eb1d0f50392485395d74ce1cca660218c9721d.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/images/b067c8aecd2283868b41f98146eb1d0f50392485395d74ce1cca660218c9721d.jpg)

### Tables

![010dc4ed305bbfba16a1de50953563695ee3e2f627ed6a1cebfe4b07c5e0bf25.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/tables/010dc4ed305bbfba16a1de50953563695ee3e2f627ed6a1cebfe4b07c5e0bf25.jpg)

![283869ffcca785ac5b1cceaf25f4b3f72d047cf674b1bd21ca32be824562f334.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/tables/283869ffcca785ac5b1cceaf25f4b3f72d047cf674b1bd21ca32be824562f334.jpg)

![4ba49dd2b86c23fe406da10abf910f733ba67138bdbfbec7d5ca458badbb958e.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/tables/4ba49dd2b86c23fe406da10abf910f733ba67138bdbfbec7d5ca458badbb958e.jpg)

![9600e1508aaaa6edd541eddb8e307bf7af77a589ae9c915ec54b72455f238df0.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/tables/9600e1508aaaa6edd541eddb8e307bf7af77a589ae9c915ec54b72455f238df0.jpg)

![ead24736b58518be657e7c619df255e14eeff45a863b96d681678dd401e36a73.jpg](../icml_results/486_Dissecting%20Submission%20Limit%20in%20Desk-Rejections_%20A%20Mathematical%20Analysis%20of%20Fairness%20in%20AI%20Conference/tables/ead24736b58518be657e7c619df255e14eeff45a863b96d681678dd401e36a73.jpg)

## Optimizing Noise Distributions for Differential Privacy


### Images

![3d31a67c63265a3cef1049d0d3eabb630f4585c2f1b3cbc9b570206cd19c5854.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/images/3d31a67c63265a3cef1049d0d3eabb630f4585c2f1b3cbc9b570206cd19c5854.jpg)

![47fe95591f1c161ab701d10dea01ee682e45f5cc95f26e4799716e69526d14b4.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/images/47fe95591f1c161ab701d10dea01ee682e45f5cc95f26e4799716e69526d14b4.jpg)

![7560c14595dc275980d082ca11e57e709fbd29381be62e94b2018be4c9f9d214.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/images/7560c14595dc275980d082ca11e57e709fbd29381be62e94b2018be4c9f9d214.jpg)

![7dd23bb109b30f792eb417715557ac39c9e30f18b00ecbee449e4becb946e791.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/images/7dd23bb109b30f792eb417715557ac39c9e30f18b00ecbee449e4becb946e791.jpg)

![ae4798d2dd3d7c800b60bd7620368bd138b5132b8788abd16c168d8b52e9eece.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/images/ae4798d2dd3d7c800b60bd7620368bd138b5132b8788abd16c168d8b52e9eece.jpg)

![e042f08ae6c8309926c4576378d61cd32107e35e9eadb85ac27a9f91453a3cea.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/images/e042f08ae6c8309926c4576378d61cd32107e35e9eadb85ac27a9f91453a3cea.jpg)

![fc700e8843924b3e0943a53e9372b5db0d6c8623b99246c2c0a47e29932f6d21.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/images/fc700e8843924b3e0943a53e9372b5db0d6c8623b99246c2c0a47e29932f6d21.jpg)

### Tables

![a5622471febe68e5c5a2899d78052288dabaa91c4bc5308f833b0f55b21e797e.jpg](../icml_results/487_Optimizing%20Noise%20Distributions%20for%20Differential%20Privacy/tables/a5622471febe68e5c5a2899d78052288dabaa91c4bc5308f833b0f55b21e797e.jpg)

## What Makes a Good Feedforward Computational Graph?


### Images

![0c2d8dfe751e396adb12aca4111324f7f49ac5136d9ad9f86c0bca7477360a97.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/images/0c2d8dfe751e396adb12aca4111324f7f49ac5136d9ad9f86c0bca7477360a97.jpg)

![22fe54d8ed3a438dd99fecd681791529a8a2446cef193286f2eea4eaf3353110.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/images/22fe54d8ed3a438dd99fecd681791529a8a2446cef193286f2eea4eaf3353110.jpg)

![66f6c4048a7b71e9a4b6bc1ce78128f352f97fd5b9429a6a178e0d547a31135f.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/images/66f6c4048a7b71e9a4b6bc1ce78128f352f97fd5b9429a6a178e0d547a31135f.jpg)

![7c5fc28fc4bc8f2f6e5eca3bb9e6a9f4b5f0d366ff0e624af3dadc80cf54e133.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/images/7c5fc28fc4bc8f2f6e5eca3bb9e6a9f4b5f0d366ff0e624af3dadc80cf54e133.jpg)

![9ee3a8056e9b063f0518119cc30d3562e0a41c6387eaefd37a82e51807707e7f.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/images/9ee3a8056e9b063f0518119cc30d3562e0a41c6387eaefd37a82e51807707e7f.jpg)

![ac81c94de01ab79ca0aea3e4392eed1363f9ce4edadddf6f81e0c26e46ba9690.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/images/ac81c94de01ab79ca0aea3e4392eed1363f9ce4edadddf6f81e0c26e46ba9690.jpg)

![e09e760cf9c9656269f00c5becd9132b9d2d868be482c6f6ff273a4f107ca9a5.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/images/e09e760cf9c9656269f00c5becd9132b9d2d868be482c6f6ff273a4f107ca9a5.jpg)

### Tables

![640cdc982bb28396def1c97ec874406845eb8b66b87a8f057eed1d1dd1bd1dcc.jpg](../icml_results/488_What%20Makes%20a%20Good%20Feedforward%20Computational%20Graph_/tables/640cdc982bb28396def1c97ec874406845eb8b66b87a8f057eed1d1dd1bd1dcc.jpg)

## From Thousands to Billions: 3D Visual Language Grounding via Render-Supervised Distillation from 2D VLMs


### Images

![05eb822acc186f4450677efe07ef45dff4b93e7d3109328044870d7888fc34d0.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/05eb822acc186f4450677efe07ef45dff4b93e7d3109328044870d7888fc34d0.jpg)

![0aa5d6c34806ff2954cec1d797704374c091137d3dd060a3ea907b61797c29df.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/0aa5d6c34806ff2954cec1d797704374c091137d3dd060a3ea907b61797c29df.jpg)

![27efbf2368a084cfc885203693d68985706d7a7c769ac29845a0c0be2f8a7065.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/27efbf2368a084cfc885203693d68985706d7a7c769ac29845a0c0be2f8a7065.jpg)

![2aa51f27b27c4b672e8505151c7a30959eabe8b6a0cefa42237f06f1657689fb.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/2aa51f27b27c4b672e8505151c7a30959eabe8b6a0cefa42237f06f1657689fb.jpg)

![90085c98c0ddcd508d6e9ed5d0002dcab8ecba45e772560c250a88d6ae681921.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/90085c98c0ddcd508d6e9ed5d0002dcab8ecba45e772560c250a88d6ae681921.jpg)

![956bda0eba715a0cc28ba3d9e97a520f4fddbc8e4519916158616f4e9d792d52.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/956bda0eba715a0cc28ba3d9e97a520f4fddbc8e4519916158616f4e9d792d52.jpg)

![acbcd9be61ca5dfefa87debd74687b15448d1271fcb49842c91f19f9057d4367.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/acbcd9be61ca5dfefa87debd74687b15448d1271fcb49842c91f19f9057d4367.jpg)

![c317665c2062f238c0a6667edc0f74517f9faae031136dad212e3fe208782afc.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/c317665c2062f238c0a6667edc0f74517f9faae031136dad212e3fe208782afc.jpg)

![c652372ec9a6db44082230e6cd9badcd778a92dd835c7bdfda6068f726725c78.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/images/c652372ec9a6db44082230e6cd9badcd778a92dd835c7bdfda6068f726725c78.jpg)

### Tables

![42047bc6dd98c9e0aca09a38558d4ad30712b18f521a81126fbceec2eeca0c0b.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/42047bc6dd98c9e0aca09a38558d4ad30712b18f521a81126fbceec2eeca0c0b.jpg)

![4ba19db74ee86b20f037ab2cb8d551de587153afc215f1537dd03daa87a87bcf.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/4ba19db74ee86b20f037ab2cb8d551de587153afc215f1537dd03daa87a87bcf.jpg)

![595af9897720bf897242f10d4344f14800f45f8ca64f2ed8c9f8f05b97a43773.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/595af9897720bf897242f10d4344f14800f45f8ca64f2ed8c9f8f05b97a43773.jpg)

![8b1d506fdcd6aab70d096c82ead208d1cf5593245f7a246c1044c060a30b2f85.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/8b1d506fdcd6aab70d096c82ead208d1cf5593245f7a246c1044c060a30b2f85.jpg)

![8b1dc2b77c6a541340c923770b876243ff15dac835a4f74aa93ae03e8e1cd5ab.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/8b1dc2b77c6a541340c923770b876243ff15dac835a4f74aa93ae03e8e1cd5ab.jpg)

![dbe7c711b12656b11d83ec64c533fada1300cf701e48ff45924b1d708f9fcbef.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/dbe7c711b12656b11d83ec64c533fada1300cf701e48ff45924b1d708f9fcbef.jpg)

![eda9f89e51314c50ab375aef41e91aa489dea9ba242d16607c373e1e36bef75d.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/eda9f89e51314c50ab375aef41e91aa489dea9ba242d16607c373e1e36bef75d.jpg)

![fe35f56efbafcc0842f4f0f9d26f567deab2857bd1acbcf35220c41209618b10.jpg](../icml_results/489_From%20Thousands%20to%20Billions_%203D%20Visual%20Language%20Grounding%20via%20Render-Supervised%20Distillation%20from%202D%20/tables/fe35f56efbafcc0842f4f0f9d26f567deab2857bd1acbcf35220c41209618b10.jpg)

## A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features


### Images

![2a5340edd67e7bd1f7da0e4955b5b54f85143a5f10621e6859d6e337643a6ffa.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/images/2a5340edd67e7bd1f7da0e4955b5b54f85143a5f10621e6859d6e337643a6ffa.jpg)

![89f657f5b4defca46ddfdb5d240e6cc580e31585665ed2843f86ad6f5cc009be.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/images/89f657f5b4defca46ddfdb5d240e6cc580e31585665ed2843f86ad6f5cc009be.jpg)

![d0cbfb76980bfc07e2801eb776822e8745fa21b944b891adec63386b4164fffa.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/images/d0cbfb76980bfc07e2801eb776822e8745fa21b944b891adec63386b4164fffa.jpg)

![da241adc49f87bd2494b871e620d701e697dc081a6362d616b631fc449df98aa.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/images/da241adc49f87bd2494b871e620d701e697dc081a6362d616b631fc449df98aa.jpg)

![e782eb04988e79ef2d5149fea2ec5a533aa1ff44b4096e5ce2485799034cd44a.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/images/e782eb04988e79ef2d5149fea2ec5a533aa1ff44b4096e5ce2485799034cd44a.jpg)

![fdddeb22eac868fac28f7884dc99deb30eebf97529470a9d48f379faad6a3957.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/images/fdddeb22eac868fac28f7884dc99deb30eebf97529470a9d48f379faad6a3957.jpg)

### Tables

![5c5ebd6a92c96120d0c294f388504b6ee8f08308205457ab19508db3792ff10f.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/tables/5c5ebd6a92c96120d0c294f388504b6ee8f08308205457ab19508db3792ff10f.jpg)

![8cacb02d31791ebeb584723f31d954c79da22c77ec9b7a0a2f1f0dc3df3b721c.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/tables/8cacb02d31791ebeb584723f31d954c79da22c77ec9b7a0a2f1f0dc3df3b721c.jpg)

![f2f15dfb1f675432a9c0c049304656b5ec535746760fff25c12b568125cccccf.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/tables/f2f15dfb1f675432a9c0c049304656b5ec535746760fff25c12b568125cccccf.jpg)

![f40074570a3c9018de3e59983805e3bbf2a8aef3eb530666756b0d2a4cb12e2a.jpg](../icml_results/490_A%20Cross%20Modal%20Knowledge%20Distillation%20%26%20Data%20Augmentation%20Recipe%20for%20Improving%20Transcriptomics%20Repres/tables/f40074570a3c9018de3e59983805e3bbf2a8aef3eb530666756b0d2a4cb12e2a.jpg)

## Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks


### Images

![1f127f7ca28c3cad90185db3d3915d99ae6e7f95b016f58a7e274b620acc3d68.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/images/1f127f7ca28c3cad90185db3d3915d99ae6e7f95b016f58a7e274b620acc3d68.jpg)

![4129071c593b3e770ebd3bf27e7aed45af1fe579bc5b2909a9addda79936a659.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/images/4129071c593b3e770ebd3bf27e7aed45af1fe579bc5b2909a9addda79936a659.jpg)

![460e4ea05fc8c394c1647f6ac72836b5947b34e9e3977de6fa6bfa6bd296e35a.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/images/460e4ea05fc8c394c1647f6ac72836b5947b34e9e3977de6fa6bfa6bd296e35a.jpg)

![97ef129c89b082479828c811001406065821405b8873b7bae86ae89304cac86a.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/images/97ef129c89b082479828c811001406065821405b8873b7bae86ae89304cac86a.jpg)

![cbe3e322c61c82dbcbf6f7aabbc4838cd60713bbfece8d65786b2b92dac641a9.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/images/cbe3e322c61c82dbcbf6f7aabbc4838cd60713bbfece8d65786b2b92dac641a9.jpg)

![f2d0e2bbe8de6aa0966b4adc5b66b28919825aea705d40e91a4db1efd5a33940.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/images/f2d0e2bbe8de6aa0966b4adc5b66b28919825aea705d40e91a4db1efd5a33940.jpg)

### Tables

![2bc98d0dc6d83adb6bb0294c853ae67063b3e0b00424ef6e2b928a094d8e1331.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/2bc98d0dc6d83adb6bb0294c853ae67063b3e0b00424ef6e2b928a094d8e1331.jpg)

![316b62fdeda1b710b9a32ed577500655d97a6125b9cab30c313173628596a93a.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/316b62fdeda1b710b9a32ed577500655d97a6125b9cab30c313173628596a93a.jpg)

![63f4624e88e9f40d01ae6c65c408a4364749f653c17dc7bf073bcbff993b9706.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/63f4624e88e9f40d01ae6c65c408a4364749f653c17dc7bf073bcbff993b9706.jpg)

![71d07cc0e314ffb2e6c96d11bb3e94a7e7bcbe1ef9210ea4dd6288c74541d66b.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/71d07cc0e314ffb2e6c96d11bb3e94a7e7bcbe1ef9210ea4dd6288c74541d66b.jpg)

![8497d7bdc1d876596b568d39aa3024fedd7ed834e5a5e5bb9cfacaef82a31111.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/8497d7bdc1d876596b568d39aa3024fedd7ed834e5a5e5bb9cfacaef82a31111.jpg)

![94c260ab8831c12045e6b6b41087c89fe5b85ffb2dabd27713dc6924300e08a2.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/94c260ab8831c12045e6b6b41087c89fe5b85ffb2dabd27713dc6924300e08a2.jpg)

![975f61aa59bdefbcbbc414df872793d4537d28c12ac84a744e0923a256b0b67a.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/975f61aa59bdefbcbbc414df872793d4537d28c12ac84a744e0923a256b0b67a.jpg)

![b912a82fe2a3e97855adfbc34cfa25c45739bd82138ac98798bd53bedb87cc5d.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/b912a82fe2a3e97855adfbc34cfa25c45739bd82138ac98798bd53bedb87cc5d.jpg)

![c6139fcf7bab09a5f115d9c036fdde4a0c84944ceff89d29c41f818b723b0abd.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/c6139fcf7bab09a5f115d9c036fdde4a0c84944ceff89d29c41f818b723b0abd.jpg)

![d41c288eece55ec110168fa1a0615b0c6165ac4625ae28944c840ba353b2e454.jpg](../icml_results/491_Efficient%20Robust%20Conformal%20Prediction%20via%20Lipschitz-Bounded%20Networks/tables/d41c288eece55ec110168fa1a0615b0c6165ac4625ae28944c840ba353b2e454.jpg)

## Outlier-Aware Post-Training Quantization for Discrete Graph Diffusion Models


### Images

![42d5bc8242d17befb71730ae2c596adf8f86db036f22cc7773c97292f3790985.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/images/42d5bc8242d17befb71730ae2c596adf8f86db036f22cc7773c97292f3790985.jpg)

![6cbdcf4d7e3ee380c2949d8976b4c06b74d29cb311003da8951411208d478014.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/images/6cbdcf4d7e3ee380c2949d8976b4c06b74d29cb311003da8951411208d478014.jpg)

![83aaba62e0980b355586285a6d22439e955bc2f112c8fc1f8717e84882030244.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/images/83aaba62e0980b355586285a6d22439e955bc2f112c8fc1f8717e84882030244.jpg)

![c6483c6cd2904d4c5d7287a8220396231d42b69174b5c60de213f91defe64517.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/images/c6483c6cd2904d4c5d7287a8220396231d42b69174b5c60de213f91defe64517.jpg)

![c93f4df910eebbcdffe2d7aa641ffd3b118f156eda880f211c5783e6a3b0ccaa.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/images/c93f4df910eebbcdffe2d7aa641ffd3b118f156eda880f211c5783e6a3b0ccaa.jpg)

![ea5b83cdd408e083408b10196a6a3acc3ba2237fe43b7cc54bcd2ba8c362ba63.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/images/ea5b83cdd408e083408b10196a6a3acc3ba2237fe43b7cc54bcd2ba8c362ba63.jpg)

### Tables

![42b7308e5e75b6412a65608e98aa1d811374c463d31a25f5b8342cb3742a99eb.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/42b7308e5e75b6412a65608e98aa1d811374c463d31a25f5b8342cb3742a99eb.jpg)

![4840a61d51aa0c751ce7742a7d60f4e7d58d8f45861ba3179606fb8daf3a169b.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/4840a61d51aa0c751ce7742a7d60f4e7d58d8f45861ba3179606fb8daf3a169b.jpg)

![61fc92ca1f1dc4dfbf33954a752a9650457586e027e0feff833c01dcf391b238.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/61fc92ca1f1dc4dfbf33954a752a9650457586e027e0feff833c01dcf391b238.jpg)

![93dfa8ba1639cd59f255f2d0024ab46f2dd1cfde9fcdb291ef95a8892ac49ac2.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/93dfa8ba1639cd59f255f2d0024ab46f2dd1cfde9fcdb291ef95a8892ac49ac2.jpg)

![96b13be1cf48fbb26e38aab03d3b95fed4167356561e686daa26e555b67b68af.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/96b13be1cf48fbb26e38aab03d3b95fed4167356561e686daa26e555b67b68af.jpg)

![d92d6fddb6f2b28e8a86fbd3a8e500fc89d42140d750ee0a5da843467290838a.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/d92d6fddb6f2b28e8a86fbd3a8e500fc89d42140d750ee0a5da843467290838a.jpg)

![dfbbe6f351fa8c959bc7052143ef0f916a32be51cba5e5acac19715d45465b34.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/dfbbe6f351fa8c959bc7052143ef0f916a32be51cba5e5acac19715d45465b34.jpg)

![e38a99a43ad59901ab7a4e027990d07b07009a5697e6af68b00143c44f0184f4.jpg](../icml_results/492_Outlier-Aware%20Post-Training%20Quantization%20for%20Discrete%20Graph%20Diffusion%20Models/tables/e38a99a43ad59901ab7a4e027990d07b07009a5697e6af68b00143c44f0184f4.jpg)

## ReverB-SNN: Reversing Bit of the Weight and Activation for Spiking Neural Networks


### Images

![26afed1cc52726fda9ed38b56f422dd611613cc1872de2dbf45298051dfdacf6.jpg](../icml_results/493_ReverB-SNN_%20Reversing%20Bit%20of%20the%20Weight%20and%20Activation%20for%20Spiking%20Neural%20Networks/images/26afed1cc52726fda9ed38b56f422dd611613cc1872de2dbf45298051dfdacf6.jpg)

### Tables

![13ba4527bd022390f7ccf589f05bd2d17d04fe1acf6a8b186ff927cfa50e5347.jpg](../icml_results/493_ReverB-SNN_%20Reversing%20Bit%20of%20the%20Weight%20and%20Activation%20for%20Spiking%20Neural%20Networks/tables/13ba4527bd022390f7ccf589f05bd2d17d04fe1acf6a8b186ff927cfa50e5347.jpg)

![5aa20f9bdf5edab557e5f00648a44734587050854689fea6382dd243afc04ebb.jpg](../icml_results/493_ReverB-SNN_%20Reversing%20Bit%20of%20the%20Weight%20and%20Activation%20for%20Spiking%20Neural%20Networks/tables/5aa20f9bdf5edab557e5f00648a44734587050854689fea6382dd243afc04ebb.jpg)

![757842fb9277a08b6d4110cf53e9541ae1fc8fdd7dccf555ed1fe392a388409a.jpg](../icml_results/493_ReverB-SNN_%20Reversing%20Bit%20of%20the%20Weight%20and%20Activation%20for%20Spiking%20Neural%20Networks/tables/757842fb9277a08b6d4110cf53e9541ae1fc8fdd7dccf555ed1fe392a388409a.jpg)

![758769eb20a4ec98c982026bc4588672230680dc21bacfc100e061ef046b5768.jpg](../icml_results/493_ReverB-SNN_%20Reversing%20Bit%20of%20the%20Weight%20and%20Activation%20for%20Spiking%20Neural%20Networks/tables/758769eb20a4ec98c982026bc4588672230680dc21bacfc100e061ef046b5768.jpg)

![b8dabb8008f67a75795f8af87bf3a1ee5efb38c7f0fdbf08a5c594c630a30eb2.jpg](../icml_results/493_ReverB-SNN_%20Reversing%20Bit%20of%20the%20Weight%20and%20Activation%20for%20Spiking%20Neural%20Networks/tables/b8dabb8008f67a75795f8af87bf3a1ee5efb38c7f0fdbf08a5c594c630a30eb2.jpg)

## Rethinking Confidence Scores and Thresholds in Pseudolabeling-based SSL


### Images

![370846df86d406f8f68db390200c64826437a69956ba9fef4438cb746625342e.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/370846df86d406f8f68db390200c64826437a69956ba9fef4438cb746625342e.jpg)

![4d0d6e8b57ab84d28dd92980c70c326dfa8b98787451e3ade8c0c1cc2cca17d0.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/4d0d6e8b57ab84d28dd92980c70c326dfa8b98787451e3ade8c0c1cc2cca17d0.jpg)

![5c16e732fad7c63f11beeb6c2bd3e4927e11cf43df58a1b8948c29c02a6cf563.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/5c16e732fad7c63f11beeb6c2bd3e4927e11cf43df58a1b8948c29c02a6cf563.jpg)

![6a4897ad4bbe46cd10bb124739e2dc2963208618e701da38f982805a3e0f0ac7.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/6a4897ad4bbe46cd10bb124739e2dc2963208618e701da38f982805a3e0f0ac7.jpg)

![986f80775a45b5454314aae07505a873e6da54e3803c59d9ad5cfdc8653e54fb.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/986f80775a45b5454314aae07505a873e6da54e3803c59d9ad5cfdc8653e54fb.jpg)

![bc086ba0258d389c105a52c33da1c1185f53bae417a561eef00acda52abeba66.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/bc086ba0258d389c105a52c33da1c1185f53bae417a561eef00acda52abeba66.jpg)

![c85cac6a75dd5458b9267d43d22932a24eafdc6f2d7f5b29fde2bdd032d82941.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/c85cac6a75dd5458b9267d43d22932a24eafdc6f2d7f5b29fde2bdd032d82941.jpg)

![d899e39544d270afc7f22b76c1af3cbde13710c49bd8af31924b59334d9f58a5.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/images/d899e39544d270afc7f22b76c1af3cbde13710c49bd8af31924b59334d9f58a5.jpg)

### Tables

![29dff65701dab0618e8925345a0cc7e0cc5d0d7b0d6b308a312f915cf096da66.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/29dff65701dab0618e8925345a0cc7e0cc5d0d7b0d6b308a312f915cf096da66.jpg)

![46efaf7a0514c1b3a545f77376cc142a813c45752c64ad03bf19e2406c8e1a77.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/46efaf7a0514c1b3a545f77376cc142a813c45752c64ad03bf19e2406c8e1a77.jpg)

![4b19a3027204abf2d40d91c9632d579a851e20399c592258ec99d748c66f1f7d.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/4b19a3027204abf2d40d91c9632d579a851e20399c592258ec99d748c66f1f7d.jpg)

![6052d5a88e05b1267bbcca41b6ae9277986180cd16cf60f4eedc6a1bd638fcb1.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/6052d5a88e05b1267bbcca41b6ae9277986180cd16cf60f4eedc6a1bd638fcb1.jpg)

![6ad015cfb8da43dcdc16953842f48b34dbd9c16fa29c296c573de58f0b7c49f6.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/6ad015cfb8da43dcdc16953842f48b34dbd9c16fa29c296c573de58f0b7c49f6.jpg)

![d92875b0797ffa428c35897cddf043a8a6a3b0872306cb7a8f8ac2221ef8fa10.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/d92875b0797ffa428c35897cddf043a8a6a3b0872306cb7a8f8ac2221ef8fa10.jpg)

![e0b6144063522bea65f6fd6666f47e187dffb96d85b3728b2231a5dad59f45d5.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/e0b6144063522bea65f6fd6666f47e187dffb96d85b3728b2231a5dad59f45d5.jpg)

![e5880772a4efb472b6401505ff172fe0e20be4ffdd7ca60ebf91b0fce08fbc7b.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/e5880772a4efb472b6401505ff172fe0e20be4ffdd7ca60ebf91b0fce08fbc7b.jpg)

![e7a1a716bda7499669c05792199c5af3d0e11e5bd1dcaec17267061746fc46ba.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/e7a1a716bda7499669c05792199c5af3d0e11e5bd1dcaec17267061746fc46ba.jpg)

![f9992f15cbeb48a7b3bbf1990b81059028ddfc43298261846133f335692c256c.jpg](../icml_results/494_Rethinking%20Confidence%20Scores%20and%20Thresholds%20in%20Pseudolabeling-based%20SSL/tables/f9992f15cbeb48a7b3bbf1990b81059028ddfc43298261846133f335692c256c.jpg)

## Fast Min-$\epsilon$ Segmented Regression using Constant-Time Segment Merging


### Images

![054eab19074193eb67213a69e7c1cc7416bc06bc5258b45c552bde0ebbe6e1d4.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/054eab19074193eb67213a69e7c1cc7416bc06bc5258b45c552bde0ebbe6e1d4.jpg)

![06fb78db6cad2e461436499d1fabed3d42c6caefa8740d1d9b5a15976aee032e.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/06fb78db6cad2e461436499d1fabed3d42c6caefa8740d1d9b5a15976aee032e.jpg)

![16fb9a1270f3168e48210ac4bd275a517b790c28f001dc879f1bd0868d581157.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/16fb9a1270f3168e48210ac4bd275a517b790c28f001dc879f1bd0868d581157.jpg)

![20a8dbbee18e7eb79fa2ef285091e542021d66154e21f9ef9bdbea5b565b057a.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/20a8dbbee18e7eb79fa2ef285091e542021d66154e21f9ef9bdbea5b565b057a.jpg)

![74742aadd2efd289979186460ec4f1bdbe7176864a10b02cb510e32c1cd7adf3.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/74742aadd2efd289979186460ec4f1bdbe7176864a10b02cb510e32c1cd7adf3.jpg)

![9313a15ef067fdaa20565ab4ac8a08361f4706ef213fd19c2e1720de3098008e.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/9313a15ef067fdaa20565ab4ac8a08361f4706ef213fd19c2e1720de3098008e.jpg)

![f3533a9e678212cd0dc221e634086e88b14eb7f64a421829685924347cfe2c14.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/f3533a9e678212cd0dc221e634086e88b14eb7f64a421829685924347cfe2c14.jpg)

![fe453ceb18ec3505b3a3374ae6c38325eff23350678237821e1a1d11db4504d0.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/images/fe453ceb18ec3505b3a3374ae6c38325eff23350678237821e1a1d11db4504d0.jpg)

### Tables

![aec6d80f29c56fcad27caa304be686971e1dfe0329c408e60a0e62b08ccd91f2.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/tables/aec6d80f29c56fcad27caa304be686971e1dfe0329c408e60a0e62b08ccd91f2.jpg)

![d4832127572fc6c21941187bef963215a1dcd0054e8a33a331fe8d2ec2f00e25.jpg](../icml_results/495_Fast%20Min-%24_epsilon%24%20Segmented%20Regression%20using%20Constant-Time%20Segment%20Merging/tables/d4832127572fc6c21941187bef963215a1dcd0054e8a33a331fe8d2ec2f00e25.jpg)

## MTL-UE: Learning to Learn Nothing for Multi-Task Learning


### Images

![0dfec9a4a3353c0e83c9a5b8e5c33dfdd92ea816b1eb543b954e42ce4649966d.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/0dfec9a4a3353c0e83c9a5b8e5c33dfdd92ea816b1eb543b954e42ce4649966d.jpg)

![12b9ae80944ab4ffbe3b20cd44b789384e8f81c3d214db5798f167ba26d7e23e.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/12b9ae80944ab4ffbe3b20cd44b789384e8f81c3d214db5798f167ba26d7e23e.jpg)

![12cb3f2ec17a919f462f7bc66e0178ef067ba42be6ce73321da74dc52f047c1b.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/12cb3f2ec17a919f462f7bc66e0178ef067ba42be6ce73321da74dc52f047c1b.jpg)

![2e21331f63b42b0952423d4ff440c26d86993ed4996eae1ef33e33eeb05e1b96.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/2e21331f63b42b0952423d4ff440c26d86993ed4996eae1ef33e33eeb05e1b96.jpg)

![3281edbfba4317f5c8ebbd2310a075e36aff6fa96872199e8bebe4dc078bd3ef.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/3281edbfba4317f5c8ebbd2310a075e36aff6fa96872199e8bebe4dc078bd3ef.jpg)

![43f25c4197e5cbd347d0e52bae723907aa61bc34062ccfe7579faad5007ef52c.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/43f25c4197e5cbd347d0e52bae723907aa61bc34062ccfe7579faad5007ef52c.jpg)

![82df0c456d67a188290b6468bd2513b6ec7c5808f368c865214eb433ecbb0d61.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/82df0c456d67a188290b6468bd2513b6ec7c5808f368c865214eb433ecbb0d61.jpg)

![a7b4b5bd6db432e0e8b2be44c94437c4fe9bb5cbbf14afff4d409b2ba3a506d5.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/a7b4b5bd6db432e0e8b2be44c94437c4fe9bb5cbbf14afff4d409b2ba3a506d5.jpg)

![afc55e570a5c2457e93c7bc6b4a6c38a03f7cf21fa1629b8f373d67fe42f36b9.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/images/afc55e570a5c2457e93c7bc6b4a6c38a03f7cf21fa1629b8f373d67fe42f36b9.jpg)

### Tables

![0255764aee405808d5d7639f822b26218027617938e803972a3fe72dec4eb028.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/0255764aee405808d5d7639f822b26218027617938e803972a3fe72dec4eb028.jpg)

![1ff01f3703e972c33353abec53aef3e289ea5060ff137c5a998333743efe23e1.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/1ff01f3703e972c33353abec53aef3e289ea5060ff137c5a998333743efe23e1.jpg)

![4bedab96ac77623d225b37eb5a18ac111866a096a14fef0fe7e505905bbee07b.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/4bedab96ac77623d225b37eb5a18ac111866a096a14fef0fe7e505905bbee07b.jpg)

![5180c13122c59b2257abc972ac6174d6d378419f5d6823798715bdd65b4610af.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/5180c13122c59b2257abc972ac6174d6d378419f5d6823798715bdd65b4610af.jpg)

![53925cb729e47855ad84f8b931d2062c767b1c6a34de8f782848cf1cb90b0438.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/53925cb729e47855ad84f8b931d2062c767b1c6a34de8f782848cf1cb90b0438.jpg)

![6496b22142e754ec45bed63bbb0d1bc2ff53ecc2e5cab2039d08a55c0e262ae1.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/6496b22142e754ec45bed63bbb0d1bc2ff53ecc2e5cab2039d08a55c0e262ae1.jpg)

![69ee8f76198cf07eaaadde84a8b9de0848e285ef8e56ebf2f2bac3517d73c4a6.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/69ee8f76198cf07eaaadde84a8b9de0848e285ef8e56ebf2f2bac3517d73c4a6.jpg)

![6b06f8e009d68f63f21372b1659db03f1d51b46d38a5a053e0696eb1eeb375eb.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/6b06f8e009d68f63f21372b1659db03f1d51b46d38a5a053e0696eb1eeb375eb.jpg)

![83dc79ece0190329db8324d5b336155f264e670bff795b4d0a36901b94c407b8.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/83dc79ece0190329db8324d5b336155f264e670bff795b4d0a36901b94c407b8.jpg)

![888329c3d80279dcc30c441252dc4238893fec42caed785486337dc5867bf481.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/888329c3d80279dcc30c441252dc4238893fec42caed785486337dc5867bf481.jpg)

![943b36cc31e463db5716bce0c2a561a49190ab3f051049d7612541fc2deedd7e.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/943b36cc31e463db5716bce0c2a561a49190ab3f051049d7612541fc2deedd7e.jpg)

![98e00bdff65f9b2b0b235a5b4c00cc4223cac50c98eaeccf9710fac7eb27ea1c.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/98e00bdff65f9b2b0b235a5b4c00cc4223cac50c98eaeccf9710fac7eb27ea1c.jpg)

![ac3ac68add5c48de96bca1dcf07d4f06a8a599a3701fcc95ca04c1c78e4007db.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/ac3ac68add5c48de96bca1dcf07d4f06a8a599a3701fcc95ca04c1c78e4007db.jpg)

![b05256ff49799bdff5cdb3ac7578915d1b7c7334bcf85a2794ce053fbf259763.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/b05256ff49799bdff5cdb3ac7578915d1b7c7334bcf85a2794ce053fbf259763.jpg)

![fb144086b130cb165129e576ec432179270049e6da7ae3ef4ea9cf87c6f12625.jpg](../icml_results/496_MTL-UE_%20Learning%20to%20Learn%20Nothing%20for%20Multi-Task%20Learning/tables/fb144086b130cb165129e576ec432179270049e6da7ae3ef4ea9cf87c6f12625.jpg)

## Adaptive Sensitivity Analysis for Robust Augmentation against Natural Corruptions in Image Segmentation


### Images

![0402ee73b3acf834fd9cc2f72af2d9ca5644afec583ba9204670cadf0ef83383.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/0402ee73b3acf834fd9cc2f72af2d9ca5644afec583ba9204670cadf0ef83383.jpg)

![07968495fa5c082f7ffdf8402a66c1929971bf030190b8b215b3cc2a3c47efb6.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/07968495fa5c082f7ffdf8402a66c1929971bf030190b8b215b3cc2a3c47efb6.jpg)

![15224adf5420aafc923cef91a054eb1b110b58f9d9476dfa898b6d671882748a.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/15224adf5420aafc923cef91a054eb1b110b58f9d9476dfa898b6d671882748a.jpg)

![2c1526f47cf950ad402f297174ef0be40999840ac5510c91b8508ddcc6d4cd8b.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/2c1526f47cf950ad402f297174ef0be40999840ac5510c91b8508ddcc6d4cd8b.jpg)

![42e44695306342ccd881b08c999fea991165db5720ca7219a5c8e491a2fa0d86.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/42e44695306342ccd881b08c999fea991165db5720ca7219a5c8e491a2fa0d86.jpg)

![4344179406c1933daa08eed6fa567306fabe87303fdc5e7ad0f62761b27c3fb2.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/4344179406c1933daa08eed6fa567306fabe87303fdc5e7ad0f62761b27c3fb2.jpg)

![49ba7d003a3adbda6518955c0512bace4bbc70c9808bf3d6ac53348552615736.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/49ba7d003a3adbda6518955c0512bace4bbc70c9808bf3d6ac53348552615736.jpg)

![5e704cacf903c1f25da2fb7a98927daac7a844064de4820984330c5ee1f3d177.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/5e704cacf903c1f25da2fb7a98927daac7a844064de4820984330c5ee1f3d177.jpg)

![6c4e9ad10551f85d253419f8713ecae1adccd8191a5cf7f16988bbc1492717b3.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/6c4e9ad10551f85d253419f8713ecae1adccd8191a5cf7f16988bbc1492717b3.jpg)

![6d359d877155838273e07dbf158536219db97466b2c55b1674e05045f54df293.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/6d359d877155838273e07dbf158536219db97466b2c55b1674e05045f54df293.jpg)

![a4429f81f13bd8bdf7d9b87a974402aee4caee714845d742fad6f2133028de35.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/a4429f81f13bd8bdf7d9b87a974402aee4caee714845d742fad6f2133028de35.jpg)

![ab231569709740dced9a621e5cd3a46377a341f5f432295c9f0dc2bd6309b4ca.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/ab231569709740dced9a621e5cd3a46377a341f5f432295c9f0dc2bd6309b4ca.jpg)

![b6363693728eb126ae8554de51a77bcb48cd76b7e5e66d434d338f0762ee9d21.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/b6363693728eb126ae8554de51a77bcb48cd76b7e5e66d434d338f0762ee9d21.jpg)

![b93929f8a56baed9fe1ea08b2378c42ee17c11998e55323fe9787ed00bcadacd.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/b93929f8a56baed9fe1ea08b2378c42ee17c11998e55323fe9787ed00bcadacd.jpg)

![bab9c9e59cea8b80edf3ddb317be0a0c3fbe9a0d4f20347654850507b8751615.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/bab9c9e59cea8b80edf3ddb317be0a0c3fbe9a0d4f20347654850507b8751615.jpg)

![fcf34579fda2f9ed8bf8355307ef874d160ff85e0dc5d9c384fb56221654393f.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/images/fcf34579fda2f9ed8bf8355307ef874d160ff85e0dc5d9c384fb56221654393f.jpg)

### Tables

![01015713baa41124e69f6e7679a7468ec20e736c937ab4530bd5c7b01ca45325.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/01015713baa41124e69f6e7679a7468ec20e736c937ab4530bd5c7b01ca45325.jpg)

![01519a9f2530e8beaaab710a5ec027d8d920ccffb009a1c32594a4b96abfd09b.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/01519a9f2530e8beaaab710a5ec027d8d920ccffb009a1c32594a4b96abfd09b.jpg)

![266a1743460fa388e732bb3463507062ec0f19fc46b0a979d08941beaa27801f.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/266a1743460fa388e732bb3463507062ec0f19fc46b0a979d08941beaa27801f.jpg)

![2b4f4cf1c425ea9a7e9f01c9b5eabf743cef6359c8a4196c1baa30eae30be7c6.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/2b4f4cf1c425ea9a7e9f01c9b5eabf743cef6359c8a4196c1baa30eae30be7c6.jpg)

![475b4f0b5fd1eaf9a517b25c175ace4a5cb5b907397506d5e404ef7917903d9f.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/475b4f0b5fd1eaf9a517b25c175ace4a5cb5b907397506d5e404ef7917903d9f.jpg)

![575e5036621aaeabc1ae4a5f709dacca304cdf67ad37ee72d168dca546002cda.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/575e5036621aaeabc1ae4a5f709dacca304cdf67ad37ee72d168dca546002cda.jpg)

![96db76cf248fa24e5a6b2313bd98b87c0d6783039a25e784781595904766c7a3.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/96db76cf248fa24e5a6b2313bd98b87c0d6783039a25e784781595904766c7a3.jpg)

![9a583eabc1bb200b34875de00c2bf851172aa259a061a3397ae4e6011f8a63f8.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/9a583eabc1bb200b34875de00c2bf851172aa259a061a3397ae4e6011f8a63f8.jpg)

![9fb719fe6b532793133935e57d4a8c92f7fd2c35cc66b04f87427c028a616733.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/9fb719fe6b532793133935e57d4a8c92f7fd2c35cc66b04f87427c028a616733.jpg)

![9fc0640b0b3e1e32c01af5a83e9271da01afd38f0b4922e5168db52d36a0e194.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/9fc0640b0b3e1e32c01af5a83e9271da01afd38f0b4922e5168db52d36a0e194.jpg)

![a14d5994d82de5adeca80d15e982570b645a940dfc95528dd83953c4e0a37117.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/a14d5994d82de5adeca80d15e982570b645a940dfc95528dd83953c4e0a37117.jpg)

![bd51cae80b0af893576ec6ee94f34f6928540024e93e823b7935f3b228701462.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/bd51cae80b0af893576ec6ee94f34f6928540024e93e823b7935f3b228701462.jpg)

![c417541c2b5fc7d96065e90c23b752256f0edc42a90201749511b60855100d45.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/c417541c2b5fc7d96065e90c23b752256f0edc42a90201749511b60855100d45.jpg)

![e0ae187813c985ad7027e7206181ca14c962e182b3470807f171320b4ad59af0.jpg](../icml_results/497_Adaptive%20Sensitivity%20Analysis%20for%20Robust%20Augmentation%20against%20Natural%20Corruptions%20in%20Image%20Segmentat/tables/e0ae187813c985ad7027e7206181ca14c962e182b3470807f171320b4ad59af0.jpg)

## No-Regret is not enough! Bandits with General Constraints through Adaptive Regret Minimization

### Images

![c09237e1475368c9ae0656214d9cb53f6e5b50bae584ae183bf01ed494bfdd65.jpg](../icml_results/498_No-Regret%20is%20not%20enough%21%20Bandits%20with%20General%20Constraints%20through%20Adaptive%20Regret%20Minimization/images/c09237e1475368c9ae0656214d9cb53f6e5b50bae584ae183bf01ed494bfdd65.jpg)

![c8ed3c4f16894faa29a761562e384094d825abf8f793f5d015c812fc569b2b99.jpg](../icml_results/498_No-Regret%20is%20not%20enough%21%20Bandits%20with%20General%20Constraints%20through%20Adaptive%20Regret%20Minimization/images/c8ed3c4f16894faa29a761562e384094d825abf8f793f5d015c812fc569b2b99.jpg)
