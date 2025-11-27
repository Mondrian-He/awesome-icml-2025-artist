# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 25 of 100

## 目录 (Table of Contents)

1. [On-the-Fly Adaptive Distillation of Transformer to Dual-State  Linear Attention for Long-Context LLM Serving](#On-the-Fly-Adaptive-Distillation-of-Transformer-to-Dual-State-Linear-Attention-for-Long-Context-LLM-Serving)
2. [Rethinking Time Encoding via Learnable Transformation Functions](#Rethinking-Time-Encoding-via-Learnable-Transformation-Functions)
3. [GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models](#GANQ-GPU-Adaptive-Non-Uniform-Quantization-for-Large-Language-Models)
4. [Matryoshka Quantization](#Matryoshka-Quantization)
5. [Learning Representations of Instruments for Partial Identification of Treatment Effects](#Learning-Representations-of-Instruments-for-Partial-Identification-of-Treatment-Effects)
6. [Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](#Leveraging-Skills-from-Unlabeled-Prior-Data-for-Efficient-Online-Exploration)
7. [When Data-Free Knowledge Distillation Meets Non-Transferable Teacher: Escaping Out-of-Distribution Trap is All You Need](#When-Data-Free-Knowledge-Distillation-Meets-Non-Transferable-Teacher-Escaping-Out-of-Distribution-Trap-is-All-You-Need)
8. [Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](#Divide-and-Conquer-Grounding-LLMs-as-Efficient-Decision-Making-Agents-via-Offline-Hierarchical-Reinforcement-Learning)
9. [Random Policy Evaluation Uncovers Policies of Generative Flow Networks](#Random-Policy-Evaluation-Uncovers-Policies-of-Generative-Flow-Networks)
10. [One Stone, Two Birds: Enhancing Adversarial Defense Through the Lens of Distributional Discrepancy](#One-Stone-Two-Birds-Enhancing-Adversarial-Defense-Through-the-Lens-of-Distributional-Discrepancy)
11. [ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](#ICLShield-Exploring-and-Mitigating-In-Context-Learning-Backdoor-Attacks)
12. [Generalized Category Discovery via Reciprocal Learning and Class-Wise Distribution Regularization](#Generalized-Category-Discovery-via-Reciprocal-Learning-and-Class-Wise-Distribution-Regularization)
13. [Inductive Gradient Adjustment for Spectral Bias in Implicit Neural Representations](#Inductive-Gradient-Adjustment-for-Spectral-Bias-in-Implicit-Neural-Representations)
14. [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](#MxMoE-Mixed-precision-Quantization-for-MoE-with-Accuracy-and-Performance-Co-Design)
15. [Revisiting Convergence: Shuffling Complexity Beyond Lipschitz Smoothness](#Revisiting-Convergence-Shuffling-Complexity-Beyond-Lipschitz-Smoothness)
16. [Compact Matrix Quantum Group Equivariant Neural Networks](#Compact-Matrix-Quantum-Group-Equivariant-Neural-Networks)
17. [On the Duality between Gradient Transformations and Adapters](#On-the-Duality-between-Gradient-Transformations-and-Adapters)
18. [Causality Inspired Federated Learning for OOD Generalization](#Causality-Inspired-Federated-Learning-for-OOD-Generalization)
19. [On the Convergence of Continuous Single-timescale Actor-critic](#On-the-Convergence-of-Continuous-Single-timescale-Actor-critic)
20. [POROver: Improving Safety and Reducing Overrefusal in Large Language Models with Overgeneration and Preference Optimization](#POROver-Improving-Safety-and-Reducing-Overrefusal-in-Large-Language-Models-with-Overgeneration-and-Preference-Optimization)
21. [Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning](#Behavior-Regularized-Diffusion-Policy-Optimization-for-Offline-Reinforcement-Learning)
22. [Is Noise Conditioning Necessary for Denoising Generative Models?](#Is-Noise-Conditioning-Necessary-for-Denoising-Generative-Models)
23. [Learning Efficient Robotic Garment Manipulation with Standardization](#Learning-Efficient-Robotic-Garment-Manipulation-with-Standardization)
24. [Effective and Efficient Masked Image Generation Models](#Effective-and-Efficient-Masked-Image-Generation-Models)
25. [Efficient Heterogeneity-Aware Federated Active Data Selection](#Efficient-Heterogeneity-Aware-Federated-Active-Data-Selection)
26. [UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning](#UDora-A-Unified-Red-Teaming-Framework-against-LLM-Agents-by-Dynamically-Hijacking-Their-Own-Reasoning)
27. [SPEX: Scaling Feature Interaction Explanations for LLMs](#SPEX-Scaling-Feature-Interaction-Explanations-for-LLMs)
28. [Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens](#Reflect-then-Plan-Offline-Model-Based-Planning-through-a-Doubly-Bayesian-Lens)
29. [Preconditioned Riemannian Gradient Descent Algorithm for Low-Multilinear-Rank Tensor Completion](#Preconditioned-Riemannian-Gradient-Descent-Algorithm-for-Low-Multilinear-Rank-Tensor-Completion)
30. [Semantics-aware Test-time Adaptation for 3D Human Pose Estimation](#Semantics-aware-Test-time-Adaptation-for-3D-Human-Pose-Estimation)
31. [Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](#Reasoning-Through-Execution-Unifying-Process-and-Outcome-Rewards-for-Code-Generation)
32. [Robust Reward Alignment via Hypothesis Space Batch Cutting](#Robust-Reward-Alignment-via-Hypothesis-Space-Batch-Cutting)
33. [Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction](#Sample-Complexity-of-Distributionally-Robust-Off-Dynamics-Reinforcement-Learning-with-Online-Interaction)

---


## On-the-Fly Adaptive Distillation of Transformer to Dual-State  Linear Attention for Long-Context LLM Serving

### Images

![0302465edf87f6fab05a5fe8bc9b3671a454fe2234bee4727a38b6a8e0210ca3.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/0302465edf87f6fab05a5fe8bc9b3671a454fe2234bee4727a38b6a8e0210ca3.jpg)

![1a50fbcf438c9cef01cdae8bd55acb74d963f0f4bd8694f463bc6b6ca1ef6162.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/1a50fbcf438c9cef01cdae8bd55acb74d963f0f4bd8694f463bc6b6ca1ef6162.jpg)

![2abbf77088e25a54b1c94ff885566de4d96825c3e67efaf3c8775a95cd92cdfd.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/2abbf77088e25a54b1c94ff885566de4d96825c3e67efaf3c8775a95cd92cdfd.jpg)

![44814403eb246c329fb9734b427cce8d5176d8b9368298de7786696b467836ce.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/44814403eb246c329fb9734b427cce8d5176d8b9368298de7786696b467836ce.jpg)

![509920dd8f25be260513b9eb1436a438dd45feb5ff9181bc0d0dfb1759a64344.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/509920dd8f25be260513b9eb1436a438dd45feb5ff9181bc0d0dfb1759a64344.jpg)

![5497558544ebcc912674a6ee0560b74113f48f25b859e9afbb28666c5417e729.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/5497558544ebcc912674a6ee0560b74113f48f25b859e9afbb28666c5417e729.jpg)

![94346d946155b5c2e8a9856687701f4d385b6cd3f248e6d9ca09ca0cf6592dee.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/94346d946155b5c2e8a9856687701f4d385b6cd3f248e6d9ca09ca0cf6592dee.jpg)

![9780327b10c06a968f622c2f29cdeb5271a37c388982ee0157c084d86cc22265.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/9780327b10c06a968f622c2f29cdeb5271a37c388982ee0157c084d86cc22265.jpg)

![a6e4310db7392730591b8131e27b88e6b16148c759936568475ae28c51c0ec67.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/a6e4310db7392730591b8131e27b88e6b16148c759936568475ae28c51c0ec67.jpg)

![d80e99c1184f1f957c6b5771e1cb10d02877dfe43b40087354c09859da42cb8b.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/d80e99c1184f1f957c6b5771e1cb10d02877dfe43b40087354c09859da42cb8b.jpg)

![f75f7d53561cec404fbf467f92d4bf768b08b9f9d657ff6291e286a46edc2fd1.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/f75f7d53561cec404fbf467f92d4bf768b08b9f9d657ff6291e286a46edc2fd1.jpg)

![f8cc65280ec237a051f56240eb4b24fad74cc05f46ea6cba9646c523b9f4a245.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/f8cc65280ec237a051f56240eb4b24fad74cc05f46ea6cba9646c523b9f4a245.jpg)

![fdc83538b107065a97c84765e3a8f2654ee74251263e87cbdb7c29286895b337.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/images/fdc83538b107065a97c84765e3a8f2654ee74251263e87cbdb7c29286895b337.jpg)

### Tables

![0af51d5d6cbf4dd2d0998506dbb8e9e217bb25fd8d96aa86355af46aae614922.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/0af51d5d6cbf4dd2d0998506dbb8e9e217bb25fd8d96aa86355af46aae614922.jpg)

![0b89bfb2ca33d7d7537d3fa9a29e4efbc51e9aca956a81710b0ddf786ceb7816.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/0b89bfb2ca33d7d7537d3fa9a29e4efbc51e9aca956a81710b0ddf786ceb7816.jpg)

![14fbfe6f51c09fd41529e5d676b66e26f440ec7294bfa9f0964fa57869def79e.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/14fbfe6f51c09fd41529e5d676b66e26f440ec7294bfa9f0964fa57869def79e.jpg)

![1905f3fdf29cfea279a4b85af2ebe0fb25b22b0d28d1293b09f98d0917af5c4c.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/1905f3fdf29cfea279a4b85af2ebe0fb25b22b0d28d1293b09f98d0917af5c4c.jpg)

![1db8c0fb3fba316193b2c6cf14adb6fb6c88a0f48521cd86b26ba90225d42055.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/1db8c0fb3fba316193b2c6cf14adb6fb6c88a0f48521cd86b26ba90225d42055.jpg)

![3096b10528df82bc19b75108d0f933d29062eaac0515a729f0cce44b53eb0f42.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/3096b10528df82bc19b75108d0f933d29062eaac0515a729f0cce44b53eb0f42.jpg)

![34b32c674fee1ecefb09d53c81acc5e5882ce1e71a5d06f88a39e6e5213c3772.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/34b32c674fee1ecefb09d53c81acc5e5882ce1e71a5d06f88a39e6e5213c3772.jpg)

![3b074ea29f1f16c3331d17b9d1c90c7689959a983cb43b467fd8deb802bac848.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/3b074ea29f1f16c3331d17b9d1c90c7689959a983cb43b467fd8deb802bac848.jpg)

![46f5cb7a12ace1a931d5e213b5b9699d0dd0600abb5f69ddbf2348913be1b8a4.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/46f5cb7a12ace1a931d5e213b5b9699d0dd0600abb5f69ddbf2348913be1b8a4.jpg)

![49d4577c08ecd9ac5a3e2bde4f85aa815c6d8d8cf67e72ff5e3800a6987eb783.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/49d4577c08ecd9ac5a3e2bde4f85aa815c6d8d8cf67e72ff5e3800a6987eb783.jpg)

![59de07b34241a6d1deb284723db8425c6ea9d7853b133039faa8b3a28d315402.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/59de07b34241a6d1deb284723db8425c6ea9d7853b133039faa8b3a28d315402.jpg)

![5e049d833ba78379ee682d8e1b5afb395d7e2b2f12c9183def07f3e03e8d4aae.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/5e049d833ba78379ee682d8e1b5afb395d7e2b2f12c9183def07f3e03e8d4aae.jpg)

![7484ece8cd6cc03e2a12434923a2ee173821c785a0bd5e0e713ed13a59cf8bef.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/7484ece8cd6cc03e2a12434923a2ee173821c785a0bd5e0e713ed13a59cf8bef.jpg)

![99039ca25cee444b4b499404eabc2f7a83da6ea0ff9f90ea94b4fe4827b9ec5c.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/99039ca25cee444b4b499404eabc2f7a83da6ea0ff9f90ea94b4fe4827b9ec5c.jpg)

![a96f8ea4599dd9ee4f82bd0d0ac7808eae50cb6a63f6f3ca4dfb760c2ad61bd7.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/a96f8ea4599dd9ee4f82bd0d0ac7808eae50cb6a63f6f3ca4dfb760c2ad61bd7.jpg)

![b0c96420e8a6b1736a585111a8a5e157e490685975b1ba48a248f8014fff59c3.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/b0c96420e8a6b1736a585111a8a5e157e490685975b1ba48a248f8014fff59c3.jpg)

![c175d1d1a54694947833e553a02becebd272a6f675af72af16e4cb0934cea42f.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/c175d1d1a54694947833e553a02becebd272a6f675af72af16e4cb0934cea42f.jpg)

![c889d8e55e2fe0c56d2249f54c35ff916c1efeda8e138fbac60030554d3a8a50.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/c889d8e55e2fe0c56d2249f54c35ff916c1efeda8e138fbac60030554d3a8a50.jpg)

![e102aa303d92979f1fa1f4b38c32c1394732895685f96bf81b982db4242ee387.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/e102aa303d92979f1fa1f4b38c32c1394732895685f96bf81b982db4242ee387.jpg)

![e9530e50c76e0b09d142fe1aff8575815a5c3d9b8c654a79561a69f3f0d3ec39.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/e9530e50c76e0b09d142fe1aff8575815a5c3d9b8c654a79561a69f3f0d3ec39.jpg)

![f6eecce4f8a0d34f30dae2a3401881e4a385ae5a9db18e58015382d053e8d322.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/f6eecce4f8a0d34f30dae2a3401881e4a385ae5a9db18e58015382d053e8d322.jpg)

![f8b0b4a3688e028b33a0b9abc30cd7a42d098f3e9e849009aacee5d981b9d7e7.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/f8b0b4a3688e028b33a0b9abc30cd7a42d098f3e9e849009aacee5d981b9d7e7.jpg)

![fd823f3ce292791e91eec40c257dc6462d7bc25430fc21844ecf9b293769ce51.jpg](../icml_results/799_Information%20Bottleneck-guided%20MLPs%20for%20Robust%20Spatial-temporal%20Forecasting/tables/fd823f3ce292791e91eec40c257dc6462d7bc25430fc21844ecf9b293769ce51.jpg)

## On-the-Fly Adaptive Distillation of Transformer to Dual-State  Linear Attention for Long-Context LLM Serving


### Images

![16728cfc38d93930496abef0109662c2a10db54ed4bd127def57ca48f915474f.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/16728cfc38d93930496abef0109662c2a10db54ed4bd127def57ca48f915474f.jpg)

![2616cd2eb66f6f8639152bd4bda1177d72f7909767d016cf5e90ef6cc60d5b4d.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/2616cd2eb66f6f8639152bd4bda1177d72f7909767d016cf5e90ef6cc60d5b4d.jpg)

![2797e02e7d98806eaefac998aa931777a37a8e6dfeac4fdd8f4e39c809f46f6c.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/2797e02e7d98806eaefac998aa931777a37a8e6dfeac4fdd8f4e39c809f46f6c.jpg)

![4a2244834684879c124d00f8bf912ca7f95d0ac2df1542df75e8ded88bb68612.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/4a2244834684879c124d00f8bf912ca7f95d0ac2df1542df75e8ded88bb68612.jpg)

![52b46ff131b6953cfd85c5fa142129286132a49a088e294f9760231e2cd20759.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/52b46ff131b6953cfd85c5fa142129286132a49a088e294f9760231e2cd20759.jpg)

![5556e37dde5bb94e52b34ffd9cb371f94e0456d99150744896e966b406af3566.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/5556e37dde5bb94e52b34ffd9cb371f94e0456d99150744896e966b406af3566.jpg)

![5d910956b052c46ea7d1724404bbf29f646c3777210fbc5f5e9fc16ce5723812.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/5d910956b052c46ea7d1724404bbf29f646c3777210fbc5f5e9fc16ce5723812.jpg)

![63dee48bfd527956aaa73eb15bb459dc0c26d18febceed4ef330efc5f648cb8d.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/63dee48bfd527956aaa73eb15bb459dc0c26d18febceed4ef330efc5f648cb8d.jpg)

![84e71f1a103ca812086d3fcd30eff8fd18aea053b7a2ddf8b816e82574f12f37.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/84e71f1a103ca812086d3fcd30eff8fd18aea053b7a2ddf8b816e82574f12f37.jpg)

![a42f6be7086bb2b54f200b1c01b606411b00adb972c6d8988c78267743e14777.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/a42f6be7086bb2b54f200b1c01b606411b00adb972c6d8988c78267743e14777.jpg)

![d1853f9313459158fe4eac882a420b20d17cb8b7389f45b93b15f485d1c31ac7.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/d1853f9313459158fe4eac882a420b20d17cb8b7389f45b93b15f485d1c31ac7.jpg)

![e2461a77f78e8e6519d80ce571070e2125310948af022c3689d1f7843b16fa2e.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/e2461a77f78e8e6519d80ce571070e2125310948af022c3689d1f7843b16fa2e.jpg)

![ec5f8709476bcf639608aa0e4795eb1ae9b6b7d09bef95f51a54a0615d03eb85.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/images/ec5f8709476bcf639608aa0e4795eb1ae9b6b7d09bef95f51a54a0615d03eb85.jpg)

### Tables

![1c65b4240e35c13e66179abb0eab8b08ec3464d02c35aeb199f475fb86ccc220.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/1c65b4240e35c13e66179abb0eab8b08ec3464d02c35aeb199f475fb86ccc220.jpg)

![1eb8dc87a8270b4ab793b9930390363b83e83ab0681061d79a7ea3dca7e99e30.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/1eb8dc87a8270b4ab793b9930390363b83e83ab0681061d79a7ea3dca7e99e30.jpg)

![5346bb7378ed5190b83268821292e0d52796a95dfcdbbe7b9564096b33ecfe84.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/5346bb7378ed5190b83268821292e0d52796a95dfcdbbe7b9564096b33ecfe84.jpg)

![53c042b96ea2e5db7da4fa3181992407f6c767b9497002cce45d9ebd602e7488.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/53c042b96ea2e5db7da4fa3181992407f6c767b9497002cce45d9ebd602e7488.jpg)

![987dc80eb61036e686d9c2e3949745266d50a6e9fbb7f32de1f0a870fb914347.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/987dc80eb61036e686d9c2e3949745266d50a6e9fbb7f32de1f0a870fb914347.jpg)

![d648f36d69685f11f598546948f50e8d8287a90182beb7994c1881752477c519.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/d648f36d69685f11f598546948f50e8d8287a90182beb7994c1881752477c519.jpg)

![ef731ebe4a6562f7b11ac9115312704fc159b0e5b4b106fa4a479e97b488c990.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/ef731ebe4a6562f7b11ac9115312704fc159b0e5b4b106fa4a479e97b488c990.jpg)

![fc8293361567c5570f448fab82057b80eb2a18f1049ffffe3a5025e620f8cd77.jpg](../icml_results/800_On-the-Fly%20Adaptive%20Distillation%20of%20Transformer%20to%20Dual-State%20%20Linear%20Attention%20for%20Long-Context%20LLM/tables/fc8293361567c5570f448fab82057b80eb2a18f1049ffffe3a5025e620f8cd77.jpg)

## Rethinking Time Encoding via Learnable Transformation Functions


### Images

![05f189f8ace6d79ba381c502879c1ea2df671e575ebad28649405d8cf971f3ec.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/05f189f8ace6d79ba381c502879c1ea2df671e575ebad28649405d8cf971f3ec.jpg)

![082874b012b6623bb4a6c734a63c70c3fedff39abc717f968515b020b03c1b51.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/082874b012b6623bb4a6c734a63c70c3fedff39abc717f968515b020b03c1b51.jpg)

![117f45e0524c97aa8661b8382f876901949e2598d77a4772237f223672aa9dda.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/117f45e0524c97aa8661b8382f876901949e2598d77a4772237f223672aa9dda.jpg)

![167e4235d59363a600340fffd7c49691f0664e0fa36b655727f84b843b7fa710.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/167e4235d59363a600340fffd7c49691f0664e0fa36b655727f84b843b7fa710.jpg)

![1b05f9c343f6ad266453ffd2d97a242d8f27a63bae07bda7ad5d8994ab59bf0e.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/1b05f9c343f6ad266453ffd2d97a242d8f27a63bae07bda7ad5d8994ab59bf0e.jpg)

![339685173c14ee95beaf788f47cdec0522536137a463f00f940ddb9a97847b68.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/339685173c14ee95beaf788f47cdec0522536137a463f00f940ddb9a97847b68.jpg)

![36eabbdca366707548668f1c7eebfa4d0061d7b17a62e1f5552cb5ef337b0e4f.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/36eabbdca366707548668f1c7eebfa4d0061d7b17a62e1f5552cb5ef337b0e4f.jpg)

![3c13732c42457578a081be0b635d148d6868caf5fdaa6a2fe9b395d8dc0919e9.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/3c13732c42457578a081be0b635d148d6868caf5fdaa6a2fe9b395d8dc0919e9.jpg)

![668894e41b06e588a0f9dfe64ad31e9096a0e0bf80c518adb1b64a4df521b63f.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/668894e41b06e588a0f9dfe64ad31e9096a0e0bf80c518adb1b64a4df521b63f.jpg)

![74e3ac54cf2ef3b7e48c311d5a5b25ed490309ffe5c53e2a42635fe25eaab455.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/74e3ac54cf2ef3b7e48c311d5a5b25ed490309ffe5c53e2a42635fe25eaab455.jpg)

![7aacc717961446b1e8bbf0d48ff4379fdd24ff357841f38c73d0b19348be1fc5.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/7aacc717961446b1e8bbf0d48ff4379fdd24ff357841f38c73d0b19348be1fc5.jpg)

![88fda9cd15ad48ad151563c96145da710dc1c572b4fe039b9d8712b526939093.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/88fda9cd15ad48ad151563c96145da710dc1c572b4fe039b9d8712b526939093.jpg)

![a94e72f6fe1242e6755978a336912e4e3c6f502fa1f2bca1af2d973e48fff140.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/a94e72f6fe1242e6755978a336912e4e3c6f502fa1f2bca1af2d973e48fff140.jpg)

![b914bb5df4873dc66c1c0333f9415c49a3df987c94f5d9c27b9c01e9064a6b22.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/b914bb5df4873dc66c1c0333f9415c49a3df987c94f5d9c27b9c01e9064a6b22.jpg)

![c3b0f11fe0c9784c7543c9527c5e58e1b4b0cb2973412592fa0166b524bc3694.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/c3b0f11fe0c9784c7543c9527c5e58e1b4b0cb2973412592fa0166b524bc3694.jpg)

![deb60e0367db2488c00831768ef722c913b06327da4eb6b3684bc1c596484207.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/deb60e0367db2488c00831768ef722c913b06327da4eb6b3684bc1c596484207.jpg)

![f1264e37f9d23efe66395e59d600d6de2ef17551e1b9b424158645d0ddbe52ab.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/f1264e37f9d23efe66395e59d600d6de2ef17551e1b9b424158645d0ddbe52ab.jpg)

![f921e5d1b47b6cb7ceb7c0199adcbd6c75bf01b442937229244d9f7dc3314d0c.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/f921e5d1b47b6cb7ceb7c0199adcbd6c75bf01b442937229244d9f7dc3314d0c.jpg)

![fe27b8df4c63c1e47a9f9bceaf3281713c11a9ada01908b3681b1dcf9fdb3847.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/images/fe27b8df4c63c1e47a9f9bceaf3281713c11a9ada01908b3681b1dcf9fdb3847.jpg)

### Tables

![07faaab47bc19ca8b895a7928b52c3819ab94846b4bc17d9fb0e182f89a2f7d6.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/07faaab47bc19ca8b895a7928b52c3819ab94846b4bc17d9fb0e182f89a2f7d6.jpg)

![176b04edd113d4e64aad5ef593c3dc3cbacfeca7acef886c22d0e4a1b2d18298.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/176b04edd113d4e64aad5ef593c3dc3cbacfeca7acef886c22d0e4a1b2d18298.jpg)

![221867568a23451fb65e8a55de0b69208635de7b0eda68b67c32e172631cb487.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/221867568a23451fb65e8a55de0b69208635de7b0eda68b67c32e172631cb487.jpg)

![4a2ab00e9501830359214948775ec7c286f329db975bb387f307a6e9a36ee9b0.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/4a2ab00e9501830359214948775ec7c286f329db975bb387f307a6e9a36ee9b0.jpg)

![4d840cbfa13fcd6f9bb2fae993ebd4770147631e1842e8af4c21863634112b45.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/4d840cbfa13fcd6f9bb2fae993ebd4770147631e1842e8af4c21863634112b45.jpg)

![644f3e0d6bb0fb86edc75b7b778a2a949756fa63ca7fe2250f3aa857ea2d26e4.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/644f3e0d6bb0fb86edc75b7b778a2a949756fa63ca7fe2250f3aa857ea2d26e4.jpg)

![718dbc846f82c3a970124105753cf1665b8d1f18573e21d6794acbea5131068a.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/718dbc846f82c3a970124105753cf1665b8d1f18573e21d6794acbea5131068a.jpg)

![b02368a4cb6e247a24388b9a1253f1b64c03c1cdee3b1516fdc9dacc756e7e96.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/b02368a4cb6e247a24388b9a1253f1b64c03c1cdee3b1516fdc9dacc756e7e96.jpg)

![b36700f80a20a2d63c1687c641439c1181384745e086f838ebc86991f94a491c.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/b36700f80a20a2d63c1687c641439c1181384745e086f838ebc86991f94a491c.jpg)

![b7cd1a29d8b8ec5f52753c6f957ae2f6a01735e4e8f9edf97efd405738f81870.jpg](../icml_results/801_Rethinking%20Time%20Encoding%20via%20Learnable%20Transformation%20Functions/tables/b7cd1a29d8b8ec5f52753c6f957ae2f6a01735e4e8f9edf97efd405738f81870.jpg)

## GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models


### Images

![36bcf8a85e9d799e7697960179d490037cd39a6da7f0893a71793abe99ec8b54.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/images/36bcf8a85e9d799e7697960179d490037cd39a6da7f0893a71793abe99ec8b54.jpg)

![9c41489384c2af5d15762597f9821307faad25896fae904cc9bd6cedebda4f99.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/images/9c41489384c2af5d15762597f9821307faad25896fae904cc9bd6cedebda4f99.jpg)

![f51670b9841160525f15edda31d709a7386af29cfe9d7c4ab9c3a84ffd660697.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/images/f51670b9841160525f15edda31d709a7386af29cfe9d7c4ab9c3a84ffd660697.jpg)

### Tables

![20aea1c0fa05bb46e979fb2795ee790d2da74a669c36cb23c6fe8cc6a338197d.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/20aea1c0fa05bb46e979fb2795ee790d2da74a669c36cb23c6fe8cc6a338197d.jpg)

![2ac4f9b3d1a2cf4ebc00b720300bca18aec77cd00af4d8368ae6926d7bff8a89.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/2ac4f9b3d1a2cf4ebc00b720300bca18aec77cd00af4d8368ae6926d7bff8a89.jpg)

![4ae35c14b8a6c9707f547e35c65ac6ea23f3a74c6e986bc1ff8ceb0e63d55feb.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/4ae35c14b8a6c9707f547e35c65ac6ea23f3a74c6e986bc1ff8ceb0e63d55feb.jpg)

![5146903705fed0f64485b21a0ed19cb7c0b971e3420989bdb05c0b859e4be318.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/5146903705fed0f64485b21a0ed19cb7c0b971e3420989bdb05c0b859e4be318.jpg)

![70d2dcb93a9d04cb56a3a67454ae4fa41211cdc5dc044623c1a5adf1e9fd9b3b.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/70d2dcb93a9d04cb56a3a67454ae4fa41211cdc5dc044623c1a5adf1e9fd9b3b.jpg)

![ab4b1bb667d53c018c76f6553a2e055ba5ad36937d01b1f0ff75170b2f755c04.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/ab4b1bb667d53c018c76f6553a2e055ba5ad36937d01b1f0ff75170b2f755c04.jpg)

![b1e86cadf94c2e3a48d03460d42d7fc371d247ec335561793501a94e50cca97e.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/b1e86cadf94c2e3a48d03460d42d7fc371d247ec335561793501a94e50cca97e.jpg)

![bab0e66a70d3c8a90f507e6f349eb11e2cb3fc501da60896fc745199305ec651.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/bab0e66a70d3c8a90f507e6f349eb11e2cb3fc501da60896fc745199305ec651.jpg)

![bdfdb51c522b09ec6d10bb0f1868e0fbb67f612b8848000850e81d3bc8fcd46e.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/bdfdb51c522b09ec6d10bb0f1868e0fbb67f612b8848000850e81d3bc8fcd46e.jpg)

![bf79363c16415e0e26d1b3f2879abedf986aecd23b30aa20f9580448af39087e.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/bf79363c16415e0e26d1b3f2879abedf986aecd23b30aa20f9580448af39087e.jpg)

![f822bcebec5fc349d6e517e35030851a706ec626fb16eba879909c06bff2c660.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/f822bcebec5fc349d6e517e35030851a706ec626fb16eba879909c06bff2c660.jpg)

![fc4d4be71982ac6f6c68da5f178efe5fc4c64829e0420e9d3793dd7faf0e5086.jpg](../icml_results/802_GANQ_%20GPU-Adaptive%20Non-Uniform%20Quantization%20for%20Large%20Language%20Models/tables/fc4d4be71982ac6f6c68da5f178efe5fc4c64829e0420e9d3793dd7faf0e5086.jpg)

## Matryoshka Quantization


### Images

![4220d2c28e2d9bfecf25e7be32bf7c091add737e6a16f20c704f320206553969.jpg](../icml_results/803_Matryoshka%20Quantization/images/4220d2c28e2d9bfecf25e7be32bf7c091add737e6a16f20c704f320206553969.jpg)

![88e567b3e31f789c579a21e5530e0a91d28948e2df9734e5538d170fe601fd11.jpg](../icml_results/803_Matryoshka%20Quantization/images/88e567b3e31f789c579a21e5530e0a91d28948e2df9734e5538d170fe601fd11.jpg)

![f98dd8258ad751567205107e5cd73087ba86be25b537ca85c34ee002518dfe2c.jpg](../icml_results/803_Matryoshka%20Quantization/images/f98dd8258ad751567205107e5cd73087ba86be25b537ca85c34ee002518dfe2c.jpg)

![fadeea2273bed60df647095bf432ae48c2c233272d0d65e17d6aaa6a3b8ec166.jpg](../icml_results/803_Matryoshka%20Quantization/images/fadeea2273bed60df647095bf432ae48c2c233272d0d65e17d6aaa6a3b8ec166.jpg)

### Tables

![04df3029f3bfde1f14720233b9c75d0be6075e4e48a5cd8e43c25d91f555e9f1.jpg](../icml_results/803_Matryoshka%20Quantization/tables/04df3029f3bfde1f14720233b9c75d0be6075e4e48a5cd8e43c25d91f555e9f1.jpg)

![054ec928ffcff833f986c91620bf3bbfe98d7aee70909d7fd539155f702b3b8e.jpg](../icml_results/803_Matryoshka%20Quantization/tables/054ec928ffcff833f986c91620bf3bbfe98d7aee70909d7fd539155f702b3b8e.jpg)

![085498b189457dd8e2cc19e5b49231bfe79c590874e00cafa3799c729ff1feca.jpg](../icml_results/803_Matryoshka%20Quantization/tables/085498b189457dd8e2cc19e5b49231bfe79c590874e00cafa3799c729ff1feca.jpg)

![08cf24a80ffdbc3b1e97066182e615f2c8bb5da46d42c42379d5d42e15a7f447.jpg](../icml_results/803_Matryoshka%20Quantization/tables/08cf24a80ffdbc3b1e97066182e615f2c8bb5da46d42c42379d5d42e15a7f447.jpg)

![13a8e53d3ad6890e8864da18dd25211f4356d3cd368ad00724852cec7738d2c5.jpg](../icml_results/803_Matryoshka%20Quantization/tables/13a8e53d3ad6890e8864da18dd25211f4356d3cd368ad00724852cec7738d2c5.jpg)

![1adb28c65894cd37f597b773cda23c910b0cd3cea679b261599a4456b782d2ce.jpg](../icml_results/803_Matryoshka%20Quantization/tables/1adb28c65894cd37f597b773cda23c910b0cd3cea679b261599a4456b782d2ce.jpg)

![1ef7dd716d201c82fcd3ec603cee32303229dc482788dfc63988fe46b9ad1ddd.jpg](../icml_results/803_Matryoshka%20Quantization/tables/1ef7dd716d201c82fcd3ec603cee32303229dc482788dfc63988fe46b9ad1ddd.jpg)

![24da7343b2b35c03d4b954df73c34e353743bed70b5a98fe9e19363b6c8558c3.jpg](../icml_results/803_Matryoshka%20Quantization/tables/24da7343b2b35c03d4b954df73c34e353743bed70b5a98fe9e19363b6c8558c3.jpg)

![2b6df0b1daba8f33cf415bdd7820038788312241a3102891fc03ff87b216162d.jpg](../icml_results/803_Matryoshka%20Quantization/tables/2b6df0b1daba8f33cf415bdd7820038788312241a3102891fc03ff87b216162d.jpg)

![2dd20d81819e5ae797d21e81f160f3ef7544c8e6783d5ef03427a298a558d259.jpg](../icml_results/803_Matryoshka%20Quantization/tables/2dd20d81819e5ae797d21e81f160f3ef7544c8e6783d5ef03427a298a558d259.jpg)

![32bdb28b0be1968da46707d63596d6776bef0665c54f2b8f4958ef1d3627617f.jpg](../icml_results/803_Matryoshka%20Quantization/tables/32bdb28b0be1968da46707d63596d6776bef0665c54f2b8f4958ef1d3627617f.jpg)

![3979d97e6d3a5c9ec9d1b79c86f50b84c2af9eebb86110661eb68deb8822baa6.jpg](../icml_results/803_Matryoshka%20Quantization/tables/3979d97e6d3a5c9ec9d1b79c86f50b84c2af9eebb86110661eb68deb8822baa6.jpg)

![3af37fbcf5c52ffe52d450bef20caf75523b72104f1456c4048b9bc73f1c84a1.jpg](../icml_results/803_Matryoshka%20Quantization/tables/3af37fbcf5c52ffe52d450bef20caf75523b72104f1456c4048b9bc73f1c84a1.jpg)

![408ab3b1794f180645c58b916e9809f29648f69ecb8eeb42f384841c682d99de.jpg](../icml_results/803_Matryoshka%20Quantization/tables/408ab3b1794f180645c58b916e9809f29648f69ecb8eeb42f384841c682d99de.jpg)

![43491fd22a6a30e3bf73302e45ab3e3e9cc803a65553dc2d2d9f850c75a873d4.jpg](../icml_results/803_Matryoshka%20Quantization/tables/43491fd22a6a30e3bf73302e45ab3e3e9cc803a65553dc2d2d9f850c75a873d4.jpg)

![4aeda0d8dd7d1fd008c7efdfca0a3cb5a544e8d8453a85338a57bb61c330eac9.jpg](../icml_results/803_Matryoshka%20Quantization/tables/4aeda0d8dd7d1fd008c7efdfca0a3cb5a544e8d8453a85338a57bb61c330eac9.jpg)

![56f9488b58f989c9efbc54eb9eaf3385567696153557ecaef09fa23c5e45afc9.jpg](../icml_results/803_Matryoshka%20Quantization/tables/56f9488b58f989c9efbc54eb9eaf3385567696153557ecaef09fa23c5e45afc9.jpg)

![5774204f049694989083a68b04bd32f3bfeb232ba7a6d9c8f5cdbeec2f1c247c.jpg](../icml_results/803_Matryoshka%20Quantization/tables/5774204f049694989083a68b04bd32f3bfeb232ba7a6d9c8f5cdbeec2f1c247c.jpg)

![60e453cb98b9339b13c0562b5359ebf1e14979bf90b34306d90b6af876246535.jpg](../icml_results/803_Matryoshka%20Quantization/tables/60e453cb98b9339b13c0562b5359ebf1e14979bf90b34306d90b6af876246535.jpg)

![66812f962c677a706444711dccec4cec232b1c14bdffcbc9267b6ddc23b62dd2.jpg](../icml_results/803_Matryoshka%20Quantization/tables/66812f962c677a706444711dccec4cec232b1c14bdffcbc9267b6ddc23b62dd2.jpg)

![6d22bab913ddfb8d19f7afbdb3d143fb75d405449214e740d732e41e414d2eca.jpg](../icml_results/803_Matryoshka%20Quantization/tables/6d22bab913ddfb8d19f7afbdb3d143fb75d405449214e740d732e41e414d2eca.jpg)

![74f1949f666468af1c2f147e445179cc3a8b30e269cfa6d875ab30fc7d10a49e.jpg](../icml_results/803_Matryoshka%20Quantization/tables/74f1949f666468af1c2f147e445179cc3a8b30e269cfa6d875ab30fc7d10a49e.jpg)

![8fa7b0d54a3b8b700fc2fee162cda61faad448c66c69cb9b3b484bc9b7c12c51.jpg](../icml_results/803_Matryoshka%20Quantization/tables/8fa7b0d54a3b8b700fc2fee162cda61faad448c66c69cb9b3b484bc9b7c12c51.jpg)

![94a80d9846e95bbaf12a05b24106f24ed695f303f31fd3554a6ab7f436b3dcad.jpg](../icml_results/803_Matryoshka%20Quantization/tables/94a80d9846e95bbaf12a05b24106f24ed695f303f31fd3554a6ab7f436b3dcad.jpg)

![9fd7e11fca9171f587a04c211c5c77859ea3cd672dc5605dec68b8a2c7cba3e8.jpg](../icml_results/803_Matryoshka%20Quantization/tables/9fd7e11fca9171f587a04c211c5c77859ea3cd672dc5605dec68b8a2c7cba3e8.jpg)

![a5a59b0ee405085ac9884d25ada9fdb4427af47df21ffe1790a52b44ce46d7f3.jpg](../icml_results/803_Matryoshka%20Quantization/tables/a5a59b0ee405085ac9884d25ada9fdb4427af47df21ffe1790a52b44ce46d7f3.jpg)

![abee6325bc3dd726213181e05e5f52b55d58c1488bd931ae0ec5692d07ddae2a.jpg](../icml_results/803_Matryoshka%20Quantization/tables/abee6325bc3dd726213181e05e5f52b55d58c1488bd931ae0ec5692d07ddae2a.jpg)

![cfd8e03587b6994d0d82011d6c27801235dade0d714c2cd8ae3d5ab823a2fa62.jpg](../icml_results/803_Matryoshka%20Quantization/tables/cfd8e03587b6994d0d82011d6c27801235dade0d714c2cd8ae3d5ab823a2fa62.jpg)

![d09f3c10b6cc6fc3cf7b8634674fa613ebbdaee2833653c9951a1a03d3b4e35a.jpg](../icml_results/803_Matryoshka%20Quantization/tables/d09f3c10b6cc6fc3cf7b8634674fa613ebbdaee2833653c9951a1a03d3b4e35a.jpg)

![ea10fabe4ebb15b50b6e48c81cfa7a81d0a6277a9b72ea4f44336986bce80b89.jpg](../icml_results/803_Matryoshka%20Quantization/tables/ea10fabe4ebb15b50b6e48c81cfa7a81d0a6277a9b72ea4f44336986bce80b89.jpg)

## Learning Representations of Instruments for Partial Identification of Treatment Effects


### Images

![1f01da9dcc846418dadf501cc4b8a4f9118b10bdbbca4421ce60ff44155413c3.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/1f01da9dcc846418dadf501cc4b8a4f9118b10bdbbca4421ce60ff44155413c3.jpg)

![40eba824ce523117ac980e10231354dce106ef8f3d56e5996915d15f8fc32eb5.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/40eba824ce523117ac980e10231354dce106ef8f3d56e5996915d15f8fc32eb5.jpg)

![6890e7a9b9a30ded35bb3f7f69ad9954d527821b51584cbd7d9587e89ab45a8f.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/6890e7a9b9a30ded35bb3f7f69ad9954d527821b51584cbd7d9587e89ab45a8f.jpg)

![71a671b98f01584634d3f26ae444c98de6d149baff55469543b4e477eab6b869.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/71a671b98f01584634d3f26ae444c98de6d149baff55469543b4e477eab6b869.jpg)

![7b6ccfa05dbed73cad062eaeafff616e66f135b3f128d332dd22f54515d5a6b4.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/7b6ccfa05dbed73cad062eaeafff616e66f135b3f128d332dd22f54515d5a6b4.jpg)

![7c6afae67629b15e836f25b79a8a4b0cae438c5476f37afc0c91675cba133b5f.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/7c6afae67629b15e836f25b79a8a4b0cae438c5476f37afc0c91675cba133b5f.jpg)

![83f12325c0f13f77b52cf513d2feee3010390295f1bda8bce722e9555c4bf042.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/83f12325c0f13f77b52cf513d2feee3010390295f1bda8bce722e9555c4bf042.jpg)

![ab09a95db357d5d2e1b203a93f4db67894e83d1ee7af0235561a6cf7a109c65d.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/ab09a95db357d5d2e1b203a93f4db67894e83d1ee7af0235561a6cf7a109c65d.jpg)

![b5e9e4c329ff5d538b352291d520ff0b0a05a2f7292c18c99ddd8b0760bd003a.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/images/b5e9e4c329ff5d538b352291d520ff0b0a05a2f7292c18c99ddd8b0760bd003a.jpg)

### Tables

![20322af1b08acdb484963ea7a70ef34ba03a1ede2b6b294148cac2eaccc64f7e.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/20322af1b08acdb484963ea7a70ef34ba03a1ede2b6b294148cac2eaccc64f7e.jpg)

![569f59919b0877b944e79fb7d6d12d762d31a21f1ad0d8a00552185cf14dc0f7.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/569f59919b0877b944e79fb7d6d12d762d31a21f1ad0d8a00552185cf14dc0f7.jpg)

![590c5b8b2deb7f0fd6aaf737f7e301146fb81055634fee623f8bb6f19d291698.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/590c5b8b2deb7f0fd6aaf737f7e301146fb81055634fee623f8bb6f19d291698.jpg)

![79e0738a3039ae10fb104ae247746dd2ebdb2c32996f0de3d07c514430d480a0.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/79e0738a3039ae10fb104ae247746dd2ebdb2c32996f0de3d07c514430d480a0.jpg)

![87f6d6ce547044152ad7290524c299013e87c5645809178d7235c7f1cf1ad456.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/87f6d6ce547044152ad7290524c299013e87c5645809178d7235c7f1cf1ad456.jpg)

![e6097d63fe741907318b0888e8655722a047a0168920ca97902d2a62fa179184.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/e6097d63fe741907318b0888e8655722a047a0168920ca97902d2a62fa179184.jpg)

![ec85fa33bdb6919790df06be5da9643d5d74dfe3b26e877953399b434022125c.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/ec85fa33bdb6919790df06be5da9643d5d74dfe3b26e877953399b434022125c.jpg)

![f721d24f4e0235f009389ae4695b41d85819b4990cf312779345fd2759605c5f.jpg](../icml_results/804_Learning%20Representations%20of%20Instruments%20for%20Partial%20Identification%20of%20Treatment%20Effects/tables/f721d24f4e0235f009389ae4695b41d85819b4990cf312779345fd2759605c5f.jpg)

## Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration


### Images

![024af250aceafc2db5d705b20a44581795547d7a8d0c109dec476cffb89c880e.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/024af250aceafc2db5d705b20a44581795547d7a8d0c109dec476cffb89c880e.jpg)

![22e9287a934307137d00b10d6acfb3271a8220076291356ed1110a2f68358d39.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/22e9287a934307137d00b10d6acfb3271a8220076291356ed1110a2f68358d39.jpg)

![301164ec5d73ad6dad237b7d80a7c19166d3d1a2f1bd986b48f52154b438450c.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/301164ec5d73ad6dad237b7d80a7c19166d3d1a2f1bd986b48f52154b438450c.jpg)

![3309da8b996cab4263d3b580b5f8b638c8103ee30fc58c7e13c788196760e07a.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/3309da8b996cab4263d3b580b5f8b638c8103ee30fc58c7e13c788196760e07a.jpg)

![41a7404a1700b9c90e4dac17b6f9c906988f8a0da6329367736c023d394c37ac.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/41a7404a1700b9c90e4dac17b6f9c906988f8a0da6329367736c023d394c37ac.jpg)

![51c3c58cdf5f59060beec28a4f638a0a6e6d02f2c5428874cdd491f447ea4e47.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/51c3c58cdf5f59060beec28a4f638a0a6e6d02f2c5428874cdd491f447ea4e47.jpg)

![8521870d7d00d7881e67e470c67cf622bfdebc0e9fe8e3b5970730b9e4c4a68f.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/8521870d7d00d7881e67e470c67cf622bfdebc0e9fe8e3b5970730b9e4c4a68f.jpg)

![88ef40a611db483104c5a83fbd32fda357f3e0babf3195e0168c15f113030450.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/88ef40a611db483104c5a83fbd32fda357f3e0babf3195e0168c15f113030450.jpg)

![96f9258650e2ffe01b278e3c1fc25fc57a8785388ff60ddc70e904e94be40557.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/96f9258650e2ffe01b278e3c1fc25fc57a8785388ff60ddc70e904e94be40557.jpg)

![a04f79ea357b52759c801532597eebceb02779897ed1d50b84946714ae3ff787.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/a04f79ea357b52759c801532597eebceb02779897ed1d50b84946714ae3ff787.jpg)

![b71e6e9603b6082e0552f3be752a384518a9fa0d63522db1398807044a65a926.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/b71e6e9603b6082e0552f3be752a384518a9fa0d63522db1398807044a65a926.jpg)

![bec7a07cc2476f468c91cc0e215919ac8b9e1487472d24c020ec402fcffb3ef3.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/bec7a07cc2476f468c91cc0e215919ac8b9e1487472d24c020ec402fcffb3ef3.jpg)

![c09cbb9ee81b0290197ce21c94ee3fad2f4ea22646ea8e641cf7e4338e320543.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/c09cbb9ee81b0290197ce21c94ee3fad2f4ea22646ea8e641cf7e4338e320543.jpg)

![c561c63df6b5d06b6fbc4c1d62f9c58131ee082f6f369d4a7d09833540afe454.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/c561c63df6b5d06b6fbc4c1d62f9c58131ee082f6f369d4a7d09833540afe454.jpg)

![d1d226bebdf620152bc70f6c3912c54667af727550a6a2f8190f175bf10de3fa.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/d1d226bebdf620152bc70f6c3912c54667af727550a6a2f8190f175bf10de3fa.jpg)

![d82313ecd9617c5fbcd9c807b661d9a6db642d383cec56c1aed0fc81dbda388a.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/d82313ecd9617c5fbcd9c807b661d9a6db642d383cec56c1aed0fc81dbda388a.jpg)

![e38036c216b3362dc10afcbf54d02185364746e714fd4aa7b62c57464dc06617.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/e38036c216b3362dc10afcbf54d02185364746e714fd4aa7b62c57464dc06617.jpg)

![e664eb30b075b9dc7087c5bbfdeaafd05af40196b1570a2707edfb0ff356b8fb.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/e664eb30b075b9dc7087c5bbfdeaafd05af40196b1570a2707edfb0ff356b8fb.jpg)

![f22bf573874a0d58daa7987431c063e21fea2221ae53a2469d27c8b1ec03db11.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/f22bf573874a0d58daa7987431c063e21fea2221ae53a2469d27c8b1ec03db11.jpg)

![fcae27f6d865671ea8d2f936b03e4590a6c9e1aae724f24811ef65409c5bbe9a.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/fcae27f6d865671ea8d2f936b03e4590a6c9e1aae724f24811ef65409c5bbe9a.jpg)

![fcd7316a905231a199ff724c78dbccf7af47571057b442d30e69844cf6a6c6c0.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/images/fcd7316a905231a199ff724c78dbccf7af47571057b442d30e69844cf6a6c6c0.jpg)

### Tables

![03cffdcaedb1ffa7b05b9aba845f9df6e4a1ba4a0f9fbe60e7a35aa8c906eedd.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/tables/03cffdcaedb1ffa7b05b9aba845f9df6e4a1ba4a0f9fbe60e7a35aa8c906eedd.jpg)

![1930fb6a3376f59ea4f3770355d1ddb4610b05f744e451e4e7741f08df0c305b.jpg](../icml_results/805_Leveraging%20Skills%20from%20Unlabeled%20Prior%20Data%20for%20Efficient%20Online%20Exploration/tables/1930fb6a3376f59ea4f3770355d1ddb4610b05f744e451e4e7741f08df0c305b.jpg)

## When Data-Free Knowledge Distillation Meets Non-Transferable Teacher: Escaping Out-of-Distribution Trap is All You Need


### Images

![04eee64e22abd8f80a8f93dacc03b49753958d0628d69fe37522c584064b62ca.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/04eee64e22abd8f80a8f93dacc03b49753958d0628d69fe37522c584064b62ca.jpg)

![0c1e4db1fa46bc3ecbe747f655c6e6b101c8e224b11b1dedf69e650d300ec547.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/0c1e4db1fa46bc3ecbe747f655c6e6b101c8e224b11b1dedf69e650d300ec547.jpg)

![10f6eea7267bb8a50a56956a4ccd27dc4e3460b8230986183a9b06ee19414b90.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/10f6eea7267bb8a50a56956a4ccd27dc4e3460b8230986183a9b06ee19414b90.jpg)

![167427b37e2c9e212a83747a83f9f0e287af1544cd8725ed9925206adff0fb1f.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/167427b37e2c9e212a83747a83f9f0e287af1544cd8725ed9925206adff0fb1f.jpg)

![24386113fce5557ba1b428451a0e4f0bb2a902d629035a3ce1a3e25fc7f18414.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/24386113fce5557ba1b428451a0e4f0bb2a902d629035a3ce1a3e25fc7f18414.jpg)

![253682cf22201ffaf34f8ac719d9311185ec75ae9c1f3703f54cd7462c4a1774.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/253682cf22201ffaf34f8ac719d9311185ec75ae9c1f3703f54cd7462c4a1774.jpg)

![2610a1f859c0491b0ef2ca2cde9b02f5b4584dd18140423020e024b5a8b3d908.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/2610a1f859c0491b0ef2ca2cde9b02f5b4584dd18140423020e024b5a8b3d908.jpg)

![2e9ff8225532e3bca170ed28891a51949a61e1291199f249c59340b85eff2ef1.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/2e9ff8225532e3bca170ed28891a51949a61e1291199f249c59340b85eff2ef1.jpg)

![32ce57a91b492be5a756c587f16ea8860ea44fbb717c708bfe3517ddaf8b999b.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/32ce57a91b492be5a756c587f16ea8860ea44fbb717c708bfe3517ddaf8b999b.jpg)

![32e1f1e1626fd7889e1062284a1fc75c526490605aab3e058912c9075dfbc206.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/32e1f1e1626fd7889e1062284a1fc75c526490605aab3e058912c9075dfbc206.jpg)

![3b876f04d004c1747a2b7e99a8cdd14c133fa02cf620412af4aa436293f5a307.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/3b876f04d004c1747a2b7e99a8cdd14c133fa02cf620412af4aa436293f5a307.jpg)

![4277f0a17fa4757d03f7463fcde6e80c07aa1cdff35e6ea688a5bce401bf6077.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/4277f0a17fa4757d03f7463fcde6e80c07aa1cdff35e6ea688a5bce401bf6077.jpg)

![48dfabaf98ad18e0f41e61ae9d2114ff14db56e29cc7deb553666813cbaecba6.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/48dfabaf98ad18e0f41e61ae9d2114ff14db56e29cc7deb553666813cbaecba6.jpg)

![4bdfc80f889b68aa578f300f1c66244ed03406202b6a326fd9559af483a00732.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/4bdfc80f889b68aa578f300f1c66244ed03406202b6a326fd9559af483a00732.jpg)

![536c8cd2531d9d1cc77d8b5276f3ef73c52ade078e610c5de94929309d25d441.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/536c8cd2531d9d1cc77d8b5276f3ef73c52ade078e610c5de94929309d25d441.jpg)

![53b3b3d9d9a6dbd1a2aab86baca6890195eba4b927dfad9931c3560c7115e54e.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/53b3b3d9d9a6dbd1a2aab86baca6890195eba4b927dfad9931c3560c7115e54e.jpg)

![5619a26905a26a9ea03f21daff1ce41e60869f3bec3b1ebe8308fecfca97636e.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/5619a26905a26a9ea03f21daff1ce41e60869f3bec3b1ebe8308fecfca97636e.jpg)

![585c2b638725db1cd8927d953dd6f0cf62d1ddab1e98d667810d27a07ff66d6c.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/585c2b638725db1cd8927d953dd6f0cf62d1ddab1e98d667810d27a07ff66d6c.jpg)

![59bf2a93b00deeff8ef8e8de902ce2b8c8e93dbadc1a2403e4bd3badd0882087.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/59bf2a93b00deeff8ef8e8de902ce2b8c8e93dbadc1a2403e4bd3badd0882087.jpg)

![5d930878dd1728fa22d227a95e5e10b84c636b59c597b6383519212f8b95280d.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/5d930878dd1728fa22d227a95e5e10b84c636b59c597b6383519212f8b95280d.jpg)

![5e25dd5865b0d699c9293f4b6fd20632f378a78a08b261ee374151cbdcebd393.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/5e25dd5865b0d699c9293f4b6fd20632f378a78a08b261ee374151cbdcebd393.jpg)

![65786e44eaf535837585fa3f982464d917827127490c9d42d750acd7fa58b5b7.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/65786e44eaf535837585fa3f982464d917827127490c9d42d750acd7fa58b5b7.jpg)

![693a2d1a4653718e28ca5f748c1c41a78e71044c9378bb9d34ec503c0b57e4d4.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/693a2d1a4653718e28ca5f748c1c41a78e71044c9378bb9d34ec503c0b57e4d4.jpg)

![7353ad5aa018a483c4abee76f611ede726a2b4da10b90911e47c06b59f414989.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/7353ad5aa018a483c4abee76f611ede726a2b4da10b90911e47c06b59f414989.jpg)

![7afdbc35fad53358a70260d3816fa1de98c1cbc3b42b373dc90d285bded9fa4b.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/7afdbc35fad53358a70260d3816fa1de98c1cbc3b42b373dc90d285bded9fa4b.jpg)

![86c3b768b9521535af38c288dcc2c20833b889491172608ca99bace03ae0166a.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/86c3b768b9521535af38c288dcc2c20833b889491172608ca99bace03ae0166a.jpg)

![86eb9884aea9126bfe9f5e5177315bdab4c4cfe8257ac349aa315800512ba693.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/86eb9884aea9126bfe9f5e5177315bdab4c4cfe8257ac349aa315800512ba693.jpg)

![8700ee30b4a2445873e27a21bd89a50802eb897c0b0328fb949fac9be49783df.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/8700ee30b4a2445873e27a21bd89a50802eb897c0b0328fb949fac9be49783df.jpg)

![8b268cd4ddd4a357eca698f55b9442aa301d6b12ab08d2465a63808f067012b3.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/8b268cd4ddd4a357eca698f55b9442aa301d6b12ab08d2465a63808f067012b3.jpg)

![8df8a22fb5854b3935cb349c4e767bc0a80e57bfe3d08280783e1c99959b8435.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/8df8a22fb5854b3935cb349c4e767bc0a80e57bfe3d08280783e1c99959b8435.jpg)

![8f943caf7aeac368989a5de0911600be8abf85406aac1424e0531e7702f6da72.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/8f943caf7aeac368989a5de0911600be8abf85406aac1424e0531e7702f6da72.jpg)

![98e8f558b1b404b0678c71f8fd3a76e6b4965609f108cb10cafece1630b44f21.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/98e8f558b1b404b0678c71f8fd3a76e6b4965609f108cb10cafece1630b44f21.jpg)

![9c78b7246fbc6231f16b21d913ba765fe956c32d7ef65cfc6de73fdb43f1a8a1.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/9c78b7246fbc6231f16b21d913ba765fe956c32d7ef65cfc6de73fdb43f1a8a1.jpg)

![9d2e56beab502452fa34465e89efe751ff7a6cfd4e461101f574df24361ef1a0.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/9d2e56beab502452fa34465e89efe751ff7a6cfd4e461101f574df24361ef1a0.jpg)

![a17aa4a30687db67d877abb79a9dfe9d75c13b702362f813137a128023828f16.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/a17aa4a30687db67d877abb79a9dfe9d75c13b702362f813137a128023828f16.jpg)

![a6743eece13df9ff67f2bfdbc3a940b305d156c8b253734bf4295361b2e3a314.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/a6743eece13df9ff67f2bfdbc3a940b305d156c8b253734bf4295361b2e3a314.jpg)

![a7d7ab0d779b840b14efdbba89dea5ff8385f5289e6b35342f8f22d7053e2d3b.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/a7d7ab0d779b840b14efdbba89dea5ff8385f5289e6b35342f8f22d7053e2d3b.jpg)

![a923ad0f5cefdb7e50bb73efd298e048d56e8d4b69e156f805b464a6ff0e71d7.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/a923ad0f5cefdb7e50bb73efd298e048d56e8d4b69e156f805b464a6ff0e71d7.jpg)

![aa96fa5d7d9ce2af6988304c40dd568fb2ad253172f30e2adb6be6767f292f1a.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/aa96fa5d7d9ce2af6988304c40dd568fb2ad253172f30e2adb6be6767f292f1a.jpg)

![aad9f74b0b10a3161861de6dad6091fdefc9e4a714b7f1c4d56a2a92a265e3bb.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/aad9f74b0b10a3161861de6dad6091fdefc9e4a714b7f1c4d56a2a92a265e3bb.jpg)

![ab33aa41ff4d3a495dc4b8e8409400038b8ea5e11c1e4d9006ebae03f8015c36.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/ab33aa41ff4d3a495dc4b8e8409400038b8ea5e11c1e4d9006ebae03f8015c36.jpg)

![af75e12d9b7e1858d0bf2abb2fb7a6e210b64892486b6bb458d390b5b90e2f3d.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/af75e12d9b7e1858d0bf2abb2fb7a6e210b64892486b6bb458d390b5b90e2f3d.jpg)

![b46640973b383785d54cdc4d9d0d1a434142aed1920711ddadeb8c8fc5c47c0b.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/b46640973b383785d54cdc4d9d0d1a434142aed1920711ddadeb8c8fc5c47c0b.jpg)

![bc9b1256eaaf1b8c3a50f91721a9617f391a6e1ab1676f281d5ffdf3db44d833.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/bc9b1256eaaf1b8c3a50f91721a9617f391a6e1ab1676f281d5ffdf3db44d833.jpg)

![c578a5eb2e46cb671a019258f983a13b145f9c42260c07d8054d650fe86e3e22.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/c578a5eb2e46cb671a019258f983a13b145f9c42260c07d8054d650fe86e3e22.jpg)

![c6fc7e09451fb05b026b57f551a7b7f488024fdfc88bb0660416b88eadc92bed.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/c6fc7e09451fb05b026b57f551a7b7f488024fdfc88bb0660416b88eadc92bed.jpg)

![c97e36aea5eef8a42ff22f6d8343799e29a7dffbcace5ac816e879dcf4376062.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/c97e36aea5eef8a42ff22f6d8343799e29a7dffbcace5ac816e879dcf4376062.jpg)

![cc2beae20cf9d06864c28a2aea91cdc3f58e59fb02727342279fd20f17ba5600.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/cc2beae20cf9d06864c28a2aea91cdc3f58e59fb02727342279fd20f17ba5600.jpg)

![cc81d08fe8643ab2a33a203dbe8edbf710d7a3226bfbc289f3a7ad1f7634c986.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/cc81d08fe8643ab2a33a203dbe8edbf710d7a3226bfbc289f3a7ad1f7634c986.jpg)

![f6ac9fe1958418cbbc3c3a9d5f495e88dd7fe1bebbd17b05198d1cdedb426028.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/f6ac9fe1958418cbbc3c3a9d5f495e88dd7fe1bebbd17b05198d1cdedb426028.jpg)

![f70418b765497e12ac9be68becbf82dd9c8de6c39f7b2cacfd2f0a28621a059e.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/f70418b765497e12ac9be68becbf82dd9c8de6c39f7b2cacfd2f0a28621a059e.jpg)

![f9fbdb5e6e875733713124c1aa6d148cb7c51603cc68d2fab55cfcdbc12b262e.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/images/f9fbdb5e6e875733713124c1aa6d148cb7c51603cc68d2fab55cfcdbc12b262e.jpg)

### Tables

![015f45fdc777bae882b7b6e112eb601893daf0d5ff7ee5d1e5865fb883551aa8.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/015f45fdc777bae882b7b6e112eb601893daf0d5ff7ee5d1e5865fb883551aa8.jpg)

![30da56c091e69261b3873e2e81fbc3b85c023b07c75f0dbfbafc660f195782a1.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/30da56c091e69261b3873e2e81fbc3b85c023b07c75f0dbfbafc660f195782a1.jpg)

![4aa495183d784de53f5c37a4f168a1dd42badf96b19ff028372d22633c39de52.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/4aa495183d784de53f5c37a4f168a1dd42badf96b19ff028372d22633c39de52.jpg)

![6fc4d0d7e1fe723cf0a05c3d0d036c8ff3eb437f465b579afbaaaeff1efc035a.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/6fc4d0d7e1fe723cf0a05c3d0d036c8ff3eb437f465b579afbaaaeff1efc035a.jpg)

![7ede1241a1ed6009a7413e3eb3c13e1d532ac717f4f5f414609c1aea3a179bee.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/7ede1241a1ed6009a7413e3eb3c13e1d532ac717f4f5f414609c1aea3a179bee.jpg)

![b2138d0ed2b0bafee3955b8e407b93403f2cc55fa96cf68b6409dcaab9799682.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/b2138d0ed2b0bafee3955b8e407b93403f2cc55fa96cf68b6409dcaab9799682.jpg)

![b2f317fd38e768366c982a92be5f85e87dc4b433326cf59f7bcf66328dcdc367.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/b2f317fd38e768366c982a92be5f85e87dc4b433326cf59f7bcf66328dcdc367.jpg)

![c11c17e822c4a7ac89a68443c14195112676c6c0bc05aaff54b3cba7e7389013.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/c11c17e822c4a7ac89a68443c14195112676c6c0bc05aaff54b3cba7e7389013.jpg)

![c9e7fd45101cf60191bc4fb03437c4a4eb685b9e8844e8941d1e92b29d82a1da.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/c9e7fd45101cf60191bc4fb03437c4a4eb685b9e8844e8941d1e92b29d82a1da.jpg)

![d9298509dc402a4ae0a8df5dde02e6f444be92d7f86c0dbd65526a634410b04c.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/d9298509dc402a4ae0a8df5dde02e6f444be92d7f86c0dbd65526a634410b04c.jpg)

![e33baf96c4d7e0c1a2cd3436faf201adc7bb1f79a04ac0e483c86cf4ec6734c5.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/e33baf96c4d7e0c1a2cd3436faf201adc7bb1f79a04ac0e483c86cf4ec6734c5.jpg)

![f44afb9bfeb52dcdfcdd8cbd7a7ed39b5904b84a38346fbd702c1ac93f712bc4.jpg](../icml_results/806_When%20Data-Free%20Knowledge%20Distillation%20Meets%20Non-Transferable%20Teacher_%20Escaping%20Out-of-Distribution%20T/tables/f44afb9bfeb52dcdfcdd8cbd7a7ed39b5904b84a38346fbd702c1ac93f712bc4.jpg)

## Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning


### Images

![266be69d3eb856ecc8da8f91c23ec5cad80ba5dd520944c5ee0316f347288995.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/images/266be69d3eb856ecc8da8f91c23ec5cad80ba5dd520944c5ee0316f347288995.jpg)

![2d3b98ead088ce153d00b1ce2b3a2b393e47788662918568e7683956a5b6188e.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/images/2d3b98ead088ce153d00b1ce2b3a2b393e47788662918568e7683956a5b6188e.jpg)

![32f6d036155b8d30830644665e043de39f396e8a0db5bd38d36f2872985abc52.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/images/32f6d036155b8d30830644665e043de39f396e8a0db5bd38d36f2872985abc52.jpg)

![6c5f5a17d8944c589b92e99c670e5f7aadddc99a4f8df20ec9b1710343784f58.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/images/6c5f5a17d8944c589b92e99c670e5f7aadddc99a4f8df20ec9b1710343784f58.jpg)

![d0a4f9743cd5252294aca8eb40a52705d1ff31e4406c835ea1d795d231849319.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/images/d0a4f9743cd5252294aca8eb40a52705d1ff31e4406c835ea1d795d231849319.jpg)

### Tables

![32775e3c54f44733d4910533185342f80acc1c9a5039abb206002ad5cbb90a1a.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/tables/32775e3c54f44733d4910533185342f80acc1c9a5039abb206002ad5cbb90a1a.jpg)

![625f326b76ff75f49916b95085f1e38b513f2ec31679e92bd44350b250de524a.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/tables/625f326b76ff75f49916b95085f1e38b513f2ec31679e92bd44350b250de524a.jpg)

![7b2c352abfbc4005bef3c9c5f8e906e135aaf81052558d33fefa553185ae5c0c.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/tables/7b2c352abfbc4005bef3c9c5f8e906e135aaf81052558d33fefa553185ae5c0c.jpg)

![7ba25c80cb9844c7c03dd9990bbee26fcf74a02613910d0a8072a00c367ab1f7.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/tables/7ba25c80cb9844c7c03dd9990bbee26fcf74a02613910d0a8072a00c367ab1f7.jpg)

![89f7782e3866c723aa78bc7c4f9ed1a1312a8827d0a81f25dda820c60db39468.jpg](../icml_results/807_Divide%20and%20Conquer_%20Grounding%20LLMs%20as%20Efficient%20Decision-Making%20Agents%20via%20Offline%20Hierarchical%20Rein/tables/89f7782e3866c723aa78bc7c4f9ed1a1312a8827d0a81f25dda820c60db39468.jpg)

## Random Policy Evaluation Uncovers Policies of Generative Flow Networks


### Images

![6e4e0aa06133afd37f7ad9207f6288818ef86dffbc5cc76ac4623be58da1a0e7.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/6e4e0aa06133afd37f7ad9207f6288818ef86dffbc5cc76ac4623be58da1a0e7.jpg)

![79b7ab64887a8025280ce348c7cc5d2820209896cd8413509dc35f892f530cb3.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/79b7ab64887a8025280ce348c7cc5d2820209896cd8413509dc35f892f530cb3.jpg)

![91fe3fd5faedaa28db26f931452ec7f163a0652d73a70e1598c5fd26e4c53fe6.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/91fe3fd5faedaa28db26f931452ec7f163a0652d73a70e1598c5fd26e4c53fe6.jpg)

![9e01f5608845c0c900de3d781cb485e0732879cc24d929e4425f5c61d41b7d1e.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/9e01f5608845c0c900de3d781cb485e0732879cc24d929e4425f5c61d41b7d1e.jpg)

![bab355f4c12f2a8658b6b22ab67e2c8c2ced566cf01c30d619b4b2d96daf2373.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/bab355f4c12f2a8658b6b22ab67e2c8c2ced566cf01c30d619b4b2d96daf2373.jpg)

![dc90567721886a1cbb1eb5c35b9ab89e441b294657c18762041949bdb17c72ba.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/dc90567721886a1cbb1eb5c35b9ab89e441b294657c18762041949bdb17c72ba.jpg)

![e10e564dbebc59706b7be6b4c452a92a8a14f37b05f95e8d58258730a1f2c3b6.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/e10e564dbebc59706b7be6b4c452a92a8a14f37b05f95e8d58258730a1f2c3b6.jpg)

![e62fd8234e9454bf18fe67ceb70f710542354dd74abd467374dbf33b593ffba4.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/e62fd8234e9454bf18fe67ceb70f710542354dd74abd467374dbf33b593ffba4.jpg)

![ed6b781db09cf0be2014a16dbe9ed2df8f52e11234140b62a212c0511bb328bd.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/images/ed6b781db09cf0be2014a16dbe9ed2df8f52e11234140b62a212c0511bb328bd.jpg)

### Tables

![4f20d8634b5b2b7d21e847cee50e4e828ee0a3ec7106beb07cf6e8f1c0b37f22.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/tables/4f20d8634b5b2b7d21e847cee50e4e828ee0a3ec7106beb07cf6e8f1c0b37f22.jpg)

![b234de84ff5b7c15a9ca8353a0baec3f62d13d1119c9787c3175fedb8124d5c0.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/tables/b234de84ff5b7c15a9ca8353a0baec3f62d13d1119c9787c3175fedb8124d5c0.jpg)

![f13ecbad3a21419e15334f6f1c5764e439e11446c8e92ab746ed77474ae61fa0.jpg](../icml_results/808_Random%20Policy%20Evaluation%20Uncovers%20Policies%20of%20Generative%20Flow%20Networks/tables/f13ecbad3a21419e15334f6f1c5764e439e11446c8e92ab746ed77474ae61fa0.jpg)

## One Stone, Two Birds: Enhancing Adversarial Defense Through the Lens of Distributional Discrepancy


### Images

![7e4ba56e402e84bae6895b71b3c02a9125411bab9836df4fe9600b121d607361.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/images/7e4ba56e402e84bae6895b71b3c02a9125411bab9836df4fe9600b121d607361.jpg)

![eb7df7083729727904db6b0909d3511097d939bc34c9c31da50e49f214fb7deb.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/images/eb7df7083729727904db6b0909d3511097d939bc34c9c31da50e49f214fb7deb.jpg)

### Tables

![0aabcd00a5eff369c0cd025b42cad64f039a34e96e8e6b54fc48b084f714d674.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/0aabcd00a5eff369c0cd025b42cad64f039a34e96e8e6b54fc48b084f714d674.jpg)

![2ba4231486d597c3a15ff34693b3a6785f3d01715fc6556bf322739e5ae19939.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/2ba4231486d597c3a15ff34693b3a6785f3d01715fc6556bf322739e5ae19939.jpg)

![33a5d2945c9c45ad637bbc81c40a42f7cba86c48b453b0ae41fe4cb7c483a9c2.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/33a5d2945c9c45ad637bbc81c40a42f7cba86c48b453b0ae41fe4cb7c483a9c2.jpg)

![47506ca195fc9455d2ce6b7e8aa13a66118a837ad211810448a6fd046e9419b8.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/47506ca195fc9455d2ce6b7e8aa13a66118a837ad211810448a6fd046e9419b8.jpg)

![4db11c1e33f976ab45365d708c57f6d4663dca76fcbd272a72dfabdb91c57b2d.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/4db11c1e33f976ab45365d708c57f6d4663dca76fcbd272a72dfabdb91c57b2d.jpg)

![56b5e81b7cc0bace7945f76bb30734d69f4dc62ac7628dc26ec39a7b09c4bb14.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/56b5e81b7cc0bace7945f76bb30734d69f4dc62ac7628dc26ec39a7b09c4bb14.jpg)

![83359a64095d55d8de5b0f394beb084f3cde6980768b51dce1a98cca2d27b0cc.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/83359a64095d55d8de5b0f394beb084f3cde6980768b51dce1a98cca2d27b0cc.jpg)

![991b1d41dcdc2559618e1aa31a14b5fdf3bed4fab0464e4a97e9f64798d5d7cc.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/991b1d41dcdc2559618e1aa31a14b5fdf3bed4fab0464e4a97e9f64798d5d7cc.jpg)

![9ee575a5d8a83388d2ccd2c198bcf50ed7f25e3cc91e83ae01149fb3c24135a6.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/9ee575a5d8a83388d2ccd2c198bcf50ed7f25e3cc91e83ae01149fb3c24135a6.jpg)

![a56fc00cac479f026075567cb767b1a6ca548afb69b0ccf42d1d9b59e9278ec3.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/a56fc00cac479f026075567cb767b1a6ca548afb69b0ccf42d1d9b59e9278ec3.jpg)

![b01013130af4bc36ecae5c5bfc8e86c5d0db081b804006e4c053c187bfb6bc62.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/b01013130af4bc36ecae5c5bfc8e86c5d0db081b804006e4c053c187bfb6bc62.jpg)

![da9856e6700cad0840d1a7ea22a7f7bbc01727df0ac9d5831bbed7032b49403f.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/da9856e6700cad0840d1a7ea22a7f7bbc01727df0ac9d5831bbed7032b49403f.jpg)

![e2d2997c31d1a71e6a9451946291841cf7d344c6f822fe0f67c4198c11df9b08.jpg](../icml_results/809_One%20Stone%2C%20Two%20Birds_%20Enhancing%20Adversarial%20Defense%20Through%20the%20Lens%20of%20Distributional%20Discrepancy/tables/e2d2997c31d1a71e6a9451946291841cf7d344c6f822fe0f67c4198c11df9b08.jpg)

## ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks


### Images

![78e31c102819c70a788b50f99f205b6c4a601d04326aca33f22de048d0bea177.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/images/78e31c102819c70a788b50f99f205b6c4a601d04326aca33f22de048d0bea177.jpg)

![869a7e689a4cccb7b96239b0b0ed8354eb4900417c60f4093ec68e08900043c1.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/images/869a7e689a4cccb7b96239b0b0ed8354eb4900417c60f4093ec68e08900043c1.jpg)

![9aab04c25429ab737facc65bb0062a9f10ca25451a28cdf5747a4eb220969c79.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/images/9aab04c25429ab737facc65bb0062a9f10ca25451a28cdf5747a4eb220969c79.jpg)

![ce83ba5045353971ed847a8d3b18c399678c9ff48fb1ee89cfb1cb3b839c374a.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/images/ce83ba5045353971ed847a8d3b18c399678c9ff48fb1ee89cfb1cb3b839c374a.jpg)

![f25efb3c418cbaceb4f180349a2108bea8de61ebc4e9f730809712d888979a45.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/images/f25efb3c418cbaceb4f180349a2108bea8de61ebc4e9f730809712d888979a45.jpg)

### Tables

![3af5d9cda64c706b20871b416c751dade5e0fbda85f8a9e4d87f0722f5453274.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/tables/3af5d9cda64c706b20871b416c751dade5e0fbda85f8a9e4d87f0722f5453274.jpg)

![46a3a058edbd8a0fdb90846ebb18c64ba47506fe18ed06a68d2caa0736f7a770.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/tables/46a3a058edbd8a0fdb90846ebb18c64ba47506fe18ed06a68d2caa0736f7a770.jpg)

![6a07d9b8659a8f41f42e6c7d5be37f673176db0a4f063b26ab076373ee10be98.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/tables/6a07d9b8659a8f41f42e6c7d5be37f673176db0a4f063b26ab076373ee10be98.jpg)

![b8ce1eafdc6d6befcf234b7d725ba463beefd32eb4cc964fe4da332c0eb2eaba.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/tables/b8ce1eafdc6d6befcf234b7d725ba463beefd32eb4cc964fe4da332c0eb2eaba.jpg)

![efe86f311bccb0d829ccfc617e79f9438fddcf7f2202a3864143dd0856187ae0.jpg](../icml_results/810_ICLShield_%20Exploring%20and%20Mitigating%20In-Context%20Learning%20Backdoor%20Attacks/tables/efe86f311bccb0d829ccfc617e79f9438fddcf7f2202a3864143dd0856187ae0.jpg)

## Generalized Category Discovery via Reciprocal Learning and Class-Wise Distribution Regularization


### Images

![0c1103618d9cadf5334318095dac2339d0f2add0f37ba192cfa74b5ea66c1404.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/0c1103618d9cadf5334318095dac2339d0f2add0f37ba192cfa74b5ea66c1404.jpg)

![13222209da6c4ad4bb8f8778ed57395f8ba17c395e1992d23dc215f6e51b56c3.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/13222209da6c4ad4bb8f8778ed57395f8ba17c395e1992d23dc215f6e51b56c3.jpg)

![43779f2035676869384c13c1141bc2337af897fa8b15f46b28e820f6c00e7b2f.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/43779f2035676869384c13c1141bc2337af897fa8b15f46b28e820f6c00e7b2f.jpg)

![54a9743c2e432993f454775856a1e7ab4c925a3f49fd2eff6dd4c25d962c3d8a.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/54a9743c2e432993f454775856a1e7ab4c925a3f49fd2eff6dd4c25d962c3d8a.jpg)

![8de963492f3353309022a108804c9789323b063761ed2a0a5b625e1cc56df3a5.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/8de963492f3353309022a108804c9789323b063761ed2a0a5b625e1cc56df3a5.jpg)

![a7cc32383c5c34f552865c9ddc65b70f37984eccb64ed1f692362a91641394d5.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/a7cc32383c5c34f552865c9ddc65b70f37984eccb64ed1f692362a91641394d5.jpg)

![aa8f041c3d6b915b87e820002980833c03cc2593fa745aeb56f0d86acb77100f.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/aa8f041c3d6b915b87e820002980833c03cc2593fa745aeb56f0d86acb77100f.jpg)

![bc2847f23c761e216534394f8a8981833a0a378e9c61c89f7b6387b0d9852e69.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/bc2847f23c761e216534394f8a8981833a0a378e9c61c89f7b6387b0d9852e69.jpg)

![c767e57a916d1d1ce38343e142a723e8013c031ac0b119ada64276d89388bc3a.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/c767e57a916d1d1ce38343e142a723e8013c031ac0b119ada64276d89388bc3a.jpg)

![ef1cc88cbcf25e07a2f9b600ced9da97517d69e2259b03db62e9e433d7de941b.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/ef1cc88cbcf25e07a2f9b600ced9da97517d69e2259b03db62e9e433d7de941b.jpg)

![f50cbaafb99be0aacf1e19e1a46b26f0b598770f23a717bb5c6e49c69974581e.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/images/f50cbaafb99be0aacf1e19e1a46b26f0b598770f23a717bb5c6e49c69974581e.jpg)

### Tables

![32ddf9ffb280eb3e98de947e7ac4f283be653ce944a2ac6b755f5192bdaf816e.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/32ddf9ffb280eb3e98de947e7ac4f283be653ce944a2ac6b755f5192bdaf816e.jpg)

![398553739be2cd572a1b5dd9280a1b0c14a0ac5c38059ba891b827287a570851.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/398553739be2cd572a1b5dd9280a1b0c14a0ac5c38059ba891b827287a570851.jpg)

![3ce0244cd1e611774d680918aff03c24bbf125b16d41e32c874c547665349a23.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/3ce0244cd1e611774d680918aff03c24bbf125b16d41e32c874c547665349a23.jpg)

![5fb0332764eb1795639ff5285fc7a3a64ad5028a87127c49eb0088c94c942a7c.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/5fb0332764eb1795639ff5285fc7a3a64ad5028a87127c49eb0088c94c942a7c.jpg)

![60e735a81c9417426bff2e46ce533faa46b021ecae434fb6267897d299cd1d9a.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/60e735a81c9417426bff2e46ce533faa46b021ecae434fb6267897d299cd1d9a.jpg)

![6141271ccfa5a9f1ffd48a4686296c65c627f3e0ab96b25f574bfa29b61d5db0.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/6141271ccfa5a9f1ffd48a4686296c65c627f3e0ab96b25f574bfa29b61d5db0.jpg)

![779b94722faa32d3073cfc1a4d6393d2f8aaf2297d0cc59d1db0c710d4503203.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/779b94722faa32d3073cfc1a4d6393d2f8aaf2297d0cc59d1db0c710d4503203.jpg)

![9754e119f710b5e91ec5de9c0ad1839d1a6bec1ec835e4a814197a1a9e09c750.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/9754e119f710b5e91ec5de9c0ad1839d1a6bec1ec835e4a814197a1a9e09c750.jpg)

![994a747bb877cbfa7a1904fee78df7468ec06feeb27151265577b982a2064dd8.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/994a747bb877cbfa7a1904fee78df7468ec06feeb27151265577b982a2064dd8.jpg)

![c8296c64416270b88b1487bb7be4c768433a1b1c6a6f0e3a0c5c8a8e5f7e953e.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/c8296c64416270b88b1487bb7be4c768433a1b1c6a6f0e3a0c5c8a8e5f7e953e.jpg)

![ca480b37ef3efa4af1383c5c6171723bfdfac561276ae606dde21386fd441c8e.jpg](../icml_results/811_Generalized%20Category%20Discovery%20via%20Reciprocal%20Learning%20and%20Class-Wise%20Distribution%20Regularization/tables/ca480b37ef3efa4af1383c5c6171723bfdfac561276ae606dde21386fd441c8e.jpg)

## Inductive Gradient Adjustment for Spectral Bias in Implicit Neural Representations


### Images

![1b62386b5aad078f7d0fc0e1b22b4e44c53a46410f808f4151ade4da6fd6efe1.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/1b62386b5aad078f7d0fc0e1b22b4e44c53a46410f808f4151ade4da6fd6efe1.jpg)

![2183d5512c1b8e9a0fa5e5248d0a3f34b2e9c3c546023065c2603a8264e62c94.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/2183d5512c1b8e9a0fa5e5248d0a3f34b2e9c3c546023065c2603a8264e62c94.jpg)

![46f7b32f25214720b1bcf79fbc4891d4fc8ccca67263f80156c6caef33b18224.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/46f7b32f25214720b1bcf79fbc4891d4fc8ccca67263f80156c6caef33b18224.jpg)

![58f1b2439a5cd607b44d0f02d7770cbc9685b32e9abbf204876f70af1588a21a.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/58f1b2439a5cd607b44d0f02d7770cbc9685b32e9abbf204876f70af1588a21a.jpg)

![5a36babbe7800fcbd9b54b7530d403fd64ca610f532d4769f90287c3cb81dfda.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/5a36babbe7800fcbd9b54b7530d403fd64ca610f532d4769f90287c3cb81dfda.jpg)

![616423c1e89154c0991793e353113dfad333c3b0705cff4238a978e052e40ade.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/616423c1e89154c0991793e353113dfad333c3b0705cff4238a978e052e40ade.jpg)

![64210c5dfc1b0a1ffd66159d8af3f142e5b67e912cb646a0dbea5003b9eba2d5.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/64210c5dfc1b0a1ffd66159d8af3f142e5b67e912cb646a0dbea5003b9eba2d5.jpg)

![81d0363e0c46c00b6797d6c792ade06839029d15fda748250529d9ba354d687d.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/81d0363e0c46c00b6797d6c792ade06839029d15fda748250529d9ba354d687d.jpg)

![a8e2252dd011318b30f057454130191b8cdecca67c021e22ff3de234d8ccdb9d.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/a8e2252dd011318b30f057454130191b8cdecca67c021e22ff3de234d8ccdb9d.jpg)

![b480ac349995b6f6dc7bf98a69cfb25c20cc7f397217f1534c58065ca289dd39.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/b480ac349995b6f6dc7bf98a69cfb25c20cc7f397217f1534c58065ca289dd39.jpg)

![c08c7a85ba5bcd1852b26b42d8d91d69b4129682310f5c7ad0177dbfb615c856.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/c08c7a85ba5bcd1852b26b42d8d91d69b4129682310f5c7ad0177dbfb615c856.jpg)

![f091db76c3b982c68018e437b512bc7be9bbf9881a80b417db280f7720dab47e.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/f091db76c3b982c68018e437b512bc7be9bbf9881a80b417db280f7720dab47e.jpg)

![f3b0b17967085cc51d4db5ecfbf1fbef917d422c86bbc4f4113d3c81bcfd7def.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/images/f3b0b17967085cc51d4db5ecfbf1fbef917d422c86bbc4f4113d3c81bcfd7def.jpg)

### Tables

![027a271ad6296719684f5efecf739ee6c5c0dd2fd557afb69d1bae52be732a7d.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/027a271ad6296719684f5efecf739ee6c5c0dd2fd557afb69d1bae52be732a7d.jpg)

![25449fdd4ff84738c1517d3ffc9a8df563177d1177558624cb4698b356f33bd3.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/25449fdd4ff84738c1517d3ffc9a8df563177d1177558624cb4698b356f33bd3.jpg)

![2a2d6ad2c90b4976eb634e1a648d59581ad373d18ac86a0472c9c5cf2b1e9466.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/2a2d6ad2c90b4976eb634e1a648d59581ad373d18ac86a0472c9c5cf2b1e9466.jpg)

![2b1a4dbf3095132998e0a377ab39bdd67ec93a047b6abfd2c45570e2df045a61.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/2b1a4dbf3095132998e0a377ab39bdd67ec93a047b6abfd2c45570e2df045a61.jpg)

![432c3956e5106556315ed8399d31f2165351d68e7a5b5167d00ea11844628e77.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/432c3956e5106556315ed8399d31f2165351d68e7a5b5167d00ea11844628e77.jpg)

![6a6b3265c6d4144a8d6deefb1ff1194e3a324cbc24c9a23e9181ee2000ba3571.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/6a6b3265c6d4144a8d6deefb1ff1194e3a324cbc24c9a23e9181ee2000ba3571.jpg)

![7c3a5d4026a8d60f56caeaaa67717ba6cfba87c704a0d783c09abde5c26ab075.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/7c3a5d4026a8d60f56caeaaa67717ba6cfba87c704a0d783c09abde5c26ab075.jpg)

![82d2be48ed977515dfd430490c9cab0a928053c79a21e2a2934b240ebaf3c5b6.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/82d2be48ed977515dfd430490c9cab0a928053c79a21e2a2934b240ebaf3c5b6.jpg)

![91d1d4fd4e5d7b143a7e704b3ceff5a580cfb015292bde23bc34c6bc4fe5af48.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/91d1d4fd4e5d7b143a7e704b3ceff5a580cfb015292bde23bc34c6bc4fe5af48.jpg)

![bb68569f1aee85f027b219397e1a1ea323aea706fa5a961135a45a17b157a2ca.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/bb68569f1aee85f027b219397e1a1ea323aea706fa5a961135a45a17b157a2ca.jpg)

![caa98d015e63681221a015b337a0919d4f12571da006f9a06a4f8370006bff0e.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/caa98d015e63681221a015b337a0919d4f12571da006f9a06a4f8370006bff0e.jpg)

![d78a97c4d33d14b3d404ca67dcfd3b09e5c9412a2e5ea47fb8116db0b0cc2077.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/d78a97c4d33d14b3d404ca67dcfd3b09e5c9412a2e5ea47fb8116db0b0cc2077.jpg)

![f466edd873b19c7e729f388a9a4610a86696a3f482a347edd118ca7f493e8c69.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/f466edd873b19c7e729f388a9a4610a86696a3f482a347edd118ca7f493e8c69.jpg)

![fc3ad69af267c25cd28e0970701b51566b8b4bd79e202370dda9d7d6c77e6b10.jpg](../icml_results/812_Inductive%20Gradient%20Adjustment%20for%20Spectral%20Bias%20in%20Implicit%20Neural%20Representations/tables/fc3ad69af267c25cd28e0970701b51566b8b4bd79e202370dda9d7d6c77e6b10.jpg)

## MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design


### Images

![38835e4b5e1cab4c7ef9af0da9e318f991093cc52fdc933bce50f8b1849b21c3.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/images/38835e4b5e1cab4c7ef9af0da9e318f991093cc52fdc933bce50f8b1849b21c3.jpg)

![66795de81d814864420bb6155ad843c7fd9d2e1b764b474919fe29e47cdffdba.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/images/66795de81d814864420bb6155ad843c7fd9d2e1b764b474919fe29e47cdffdba.jpg)

![804a03dbee4c649e5c9b5441324bb517e80c1b06aecd357a8034252c7773aec0.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/images/804a03dbee4c649e5c9b5441324bb517e80c1b06aecd357a8034252c7773aec0.jpg)

![dabbb99892e941157abc037bedb927b625cc701d31976ed6a0da0a305215b3ec.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/images/dabbb99892e941157abc037bedb927b625cc701d31976ed6a0da0a305215b3ec.jpg)

![f66db94b48ad8b9aebf8842d8c68a939c1787cb41cfb0f43e2fea9aae95d4f81.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/images/f66db94b48ad8b9aebf8842d8c68a939c1787cb41cfb0f43e2fea9aae95d4f81.jpg)

![ff138e18bd08fb26db1fd4d976553d63bf779b228c0b80cc6ebf01cf18c01464.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/images/ff138e18bd08fb26db1fd4d976553d63bf779b228c0b80cc6ebf01cf18c01464.jpg)

### Tables

![0975d4463ad75631966984d7d33f63b49910f8abf60a5927eaebdff0831be8ed.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/0975d4463ad75631966984d7d33f63b49910f8abf60a5927eaebdff0831be8ed.jpg)

![4a73a965a5bccb4fa71592a1a4468f10a1c9c5bc201a8d859c582f60d560f419.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/4a73a965a5bccb4fa71592a1a4468f10a1c9c5bc201a8d859c582f60d560f419.jpg)

![8564af0070ec58d18a14b4d2c760dd0f56f0c39703ceeb494323af75c66fe7af.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/8564af0070ec58d18a14b4d2c760dd0f56f0c39703ceeb494323af75c66fe7af.jpg)

![9261c3356eb2f046cb46f5b2ca605c0b8b9b0026486d37ace391e3119b453491.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/9261c3356eb2f046cb46f5b2ca605c0b8b9b0026486d37ace391e3119b453491.jpg)

![a6bdb3d9fe2dc6ecea65fae972611a108226e1546073ebcaeec60f2c531de29f.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/a6bdb3d9fe2dc6ecea65fae972611a108226e1546073ebcaeec60f2c531de29f.jpg)

![b8e5d1b74e1ee2cdea2449906dd82f057f160165987e5a844c5b1cb275e1b119.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/b8e5d1b74e1ee2cdea2449906dd82f057f160165987e5a844c5b1cb275e1b119.jpg)

![bbe27f65dcf30369e48cefb15147b6e4fcb738a5b2573de177baaf2a3c8fa1ca.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/bbe27f65dcf30369e48cefb15147b6e4fcb738a5b2573de177baaf2a3c8fa1ca.jpg)

![d71e61ccadf8251a6b3b48a11e24f19e6b438e023ed6a0e35475435134014f7c.jpg](../icml_results/813_MxMoE_%20Mixed-precision%20Quantization%20for%20MoE%20with%20Accuracy%20and%20Performance%20Co-Design/tables/d71e61ccadf8251a6b3b48a11e24f19e6b438e023ed6a0e35475435134014f7c.jpg)

## Revisiting Convergence: Shuffling Complexity Beyond Lipschitz Smoothness


### Images

![5263fe65a8f39e44107e57f88e79c86f6ddec443b82bc9c3a3fdcf11b4ce02b0.jpg](../icml_results/814_Revisiting%20Convergence_%20Shuffling%20Complexity%20Beyond%20Lipschitz%20Smoothness/images/5263fe65a8f39e44107e57f88e79c86f6ddec443b82bc9c3a3fdcf11b4ce02b0.jpg)

![77428e1e13a0a68bcf46fa5b59c48897264e4fc2741ff13aae14aae7c1e6906e.jpg](../icml_results/814_Revisiting%20Convergence_%20Shuffling%20Complexity%20Beyond%20Lipschitz%20Smoothness/images/77428e1e13a0a68bcf46fa5b59c48897264e4fc2741ff13aae14aae7c1e6906e.jpg)

![e83dfe330c9e16a86c34140e2e5fb43bbbdb27afc623704740611f39cd2c3c30.jpg](../icml_results/814_Revisiting%20Convergence_%20Shuffling%20Complexity%20Beyond%20Lipschitz%20Smoothness/images/e83dfe330c9e16a86c34140e2e5fb43bbbdb27afc623704740611f39cd2c3c30.jpg)

## Compact Matrix Quantum Group Equivariant Neural Networks


### Images

![0cd5728960927cc162dda3100f75c2a213ed6a01270ced9eb67fdaa77cce6f0f.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/0cd5728960927cc162dda3100f75c2a213ed6a01270ced9eb67fdaa77cce6f0f.jpg)

![17b42afba829d3273221582d426f81662f97f2676bd3968ef601bc2432ecf7d7.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/17b42afba829d3273221582d426f81662f97f2676bd3968ef601bc2432ecf7d7.jpg)

![451769f3a62fa20f34d8eb539a9ab9930780af22dbc13df48cdf9df95978b52c.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/451769f3a62fa20f34d8eb539a9ab9930780af22dbc13df48cdf9df95978b52c.jpg)

![6ec56c6920033542d9b94ed242c8b4b1920c824e2e2139660c213fe52b6f6838.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/6ec56c6920033542d9b94ed242c8b4b1920c824e2e2139660c213fe52b6f6838.jpg)

![72fe17688f4ce91a5b55e72c2c88fcce105b6a35c9d9742b280d194264021e8e.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/72fe17688f4ce91a5b55e72c2c88fcce105b6a35c9d9742b280d194264021e8e.jpg)

![7d4a0a15ba28e25ee41e0b474706da37d68c52d6c44b992553155c8291b231b8.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/7d4a0a15ba28e25ee41e0b474706da37d68c52d6c44b992553155c8291b231b8.jpg)

![9c7223904eb172aef82d1311ead296d16130a8ab5a22d98118d49bc38a509f56.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/9c7223904eb172aef82d1311ead296d16130a8ab5a22d98118d49bc38a509f56.jpg)

![9d96be32daa844684455d3f20c534ca36ee99814789a60683a762bf8f6f17cca.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/9d96be32daa844684455d3f20c534ca36ee99814789a60683a762bf8f6f17cca.jpg)

![a79c0fcafad19195498d8a5a93ad2fb387fb61f1c5df38b8d89baa0ed62cbd9f.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/a79c0fcafad19195498d8a5a93ad2fb387fb61f1c5df38b8d89baa0ed62cbd9f.jpg)

![c378c2331319368ed4d241e5da1d291feedaddb58d96170fb99dd043762a90ba.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/c378c2331319368ed4d241e5da1d291feedaddb58d96170fb99dd043762a90ba.jpg)

![c7d385aab6d3de863c31adc3ede293a644720be07ee542b1731f2feac772d16c.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/c7d385aab6d3de863c31adc3ede293a644720be07ee542b1731f2feac772d16c.jpg)

![cc9148d93e1e4782ead5a2633594959a18d358b7ef3d79ab4b9b0205b3b6ea49.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/cc9148d93e1e4782ead5a2633594959a18d358b7ef3d79ab4b9b0205b3b6ea49.jpg)

![dd871f7f7352ee555456d5ca1e945ebe1761498604d5f80a0111a1dbeb4577ef.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/dd871f7f7352ee555456d5ca1e945ebe1761498604d5f80a0111a1dbeb4577ef.jpg)

![e316fbd646c0baa4dd7872382263c14d1ad9293a550ac4412b58627ae21e2e3e.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/e316fbd646c0baa4dd7872382263c14d1ad9293a550ac4412b58627ae21e2e3e.jpg)

![e359f49e4b76376996f0e0683e3bdf34880db64b95e91f596e1454e68dc61d60.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/e359f49e4b76376996f0e0683e3bdf34880db64b95e91f596e1454e68dc61d60.jpg)

![e5414ce658c11fed4baa3cc3a353451d036d5a2e033b8d66080b17b9ed221427.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/images/e5414ce658c11fed4baa3cc3a353451d036d5a2e033b8d66080b17b9ed221427.jpg)

### Tables

![d770e5e767d8287b1b45063516104746a40e287047016676ec1354d7bab58703.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/tables/d770e5e767d8287b1b45063516104746a40e287047016676ec1354d7bab58703.jpg)

![f2ade234fd8bcc2b93233f2cc2a0e4562eb4333f6f5f274860ea5f05fe23e4d8.jpg](../icml_results/815_Compact%20Matrix%20Quantum%20Group%20Equivariant%20Neural%20Networks/tables/f2ade234fd8bcc2b93233f2cc2a0e4562eb4333f6f5f274860ea5f05fe23e4d8.jpg)

## On the Duality between Gradient Transformations and Adapters


### Images

![451daa56f3959c8f314da7fd6fb6fcb3430787d4ff2920bd717e8456ee5a81e8.jpg](../icml_results/816_On%20the%20Duality%20between%20Gradient%20Transformations%20and%20Adapters/images/451daa56f3959c8f314da7fd6fb6fcb3430787d4ff2920bd717e8456ee5a81e8.jpg)

![8d4ab087a73b44c7b78e944027ed430fe3e678d5c8737f9b694034ea8630e479.jpg](../icml_results/816_On%20the%20Duality%20between%20Gradient%20Transformations%20and%20Adapters/images/8d4ab087a73b44c7b78e944027ed430fe3e678d5c8737f9b694034ea8630e479.jpg)

### Tables

![10e6c5583b1bd826b0a296101f1ec7b04161fff5fa49b7439818de7ac31ed50d.jpg](../icml_results/816_On%20the%20Duality%20between%20Gradient%20Transformations%20and%20Adapters/tables/10e6c5583b1bd826b0a296101f1ec7b04161fff5fa49b7439818de7ac31ed50d.jpg)

![3709e3621bf32a11595e83402d17d9ac22befd5a1f41f6d823f9f8ff00223e1f.jpg](../icml_results/816_On%20the%20Duality%20between%20Gradient%20Transformations%20and%20Adapters/tables/3709e3621bf32a11595e83402d17d9ac22befd5a1f41f6d823f9f8ff00223e1f.jpg)

![38753130fc1aa762bc3795985e4a06968456b0fb27f1ddc12fccb552dd7e799c.jpg](../icml_results/816_On%20the%20Duality%20between%20Gradient%20Transformations%20and%20Adapters/tables/38753130fc1aa762bc3795985e4a06968456b0fb27f1ddc12fccb552dd7e799c.jpg)

![3b3553ee00e4010158f3d35299c5cb0f58dd20dca4adbbc8b3b780d7cc87f63f.jpg](../icml_results/816_On%20the%20Duality%20between%20Gradient%20Transformations%20and%20Adapters/tables/3b3553ee00e4010158f3d35299c5cb0f58dd20dca4adbbc8b3b780d7cc87f63f.jpg)

![c66705d1ee2e550a494230cc200a68ffe84348ad4ba8da6f1502c4a3e2d487de.jpg](../icml_results/816_On%20the%20Duality%20between%20Gradient%20Transformations%20and%20Adapters/tables/c66705d1ee2e550a494230cc200a68ffe84348ad4ba8da6f1502c4a3e2d487de.jpg)

## Causality Inspired Federated Learning for OOD Generalization


### Images

![017a87409f9d1d3601888bee29c4c4edacda8526fab90e2c8d908b7dce38f01c.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/017a87409f9d1d3601888bee29c4c4edacda8526fab90e2c8d908b7dce38f01c.jpg)

![3bd03ca3321449231ab646f633bc12940df282d8c92c6c1b256f435d6d24fcca.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/3bd03ca3321449231ab646f633bc12940df282d8c92c6c1b256f435d6d24fcca.jpg)

![51f806b96fcaa2df323c5a17b53f9d04a7c9da026e422bbeedbdb6ca4ad68a92.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/51f806b96fcaa2df323c5a17b53f9d04a7c9da026e422bbeedbdb6ca4ad68a92.jpg)

![7ac7147fae795b3a5b0dcb965798a9115f9d0d55ab125322979ccb904109194a.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/7ac7147fae795b3a5b0dcb965798a9115f9d0d55ab125322979ccb904109194a.jpg)

![91cecee0f543bd36f262bf6744a2d0d39f918ccb5d21aea7f77a44738cde6b38.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/91cecee0f543bd36f262bf6744a2d0d39f918ccb5d21aea7f77a44738cde6b38.jpg)

![9846731d1392fcbca6c5dad4a0a61a0a7510a0dec00a7bb6e948852da1cc9c2f.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/9846731d1392fcbca6c5dad4a0a61a0a7510a0dec00a7bb6e948852da1cc9c2f.jpg)

![a6a7e7917f02384644ef66b7c88b6e897d0cd160a5b18ccf2f33106033dd6544.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/a6a7e7917f02384644ef66b7c88b6e897d0cd160a5b18ccf2f33106033dd6544.jpg)

![ce9fcc61634da06df3ae66efee1e94d398cdd3e3488237924c234167464c4e7d.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/ce9fcc61634da06df3ae66efee1e94d398cdd3e3488237924c234167464c4e7d.jpg)

![f126b76272cbed70d567803f2ad44d104c55f7ec95ae9786466eb0c3bf9f1bd1.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/images/f126b76272cbed70d567803f2ad44d104c55f7ec95ae9786466eb0c3bf9f1bd1.jpg)

### Tables

![1006068cef204e36a4a2dfca3975e9653a62e9d3fb1e4e60283e96fc6db9d0cd.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/tables/1006068cef204e36a4a2dfca3975e9653a62e9d3fb1e4e60283e96fc6db9d0cd.jpg)

![16e8b68a068c2afe320a43a87b3c1abcbe4b6638b9e62825523c9d44c59080c2.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/tables/16e8b68a068c2afe320a43a87b3c1abcbe4b6638b9e62825523c9d44c59080c2.jpg)

![4bff0310b79b00c0a5473225ce3d2172d556c7cabcc35f7ef97bdff101aa1664.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/tables/4bff0310b79b00c0a5473225ce3d2172d556c7cabcc35f7ef97bdff101aa1664.jpg)

![4cab4fc4e8401010b2526f7b420e4e52f9b2429d1ca430dbf3a3c0c8fab8481f.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/tables/4cab4fc4e8401010b2526f7b420e4e52f9b2429d1ca430dbf3a3c0c8fab8481f.jpg)

![5e6307501640570fda4242d93ebbae597f339c65739e698ccdf97c7f424f3e84.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/tables/5e6307501640570fda4242d93ebbae597f339c65739e698ccdf97c7f424f3e84.jpg)

![d89811401edcf80ec340a64439273f3cda7f09bb3abe5e7fcd414ec65bd247b6.jpg](../icml_results/817_Causality%20Inspired%20Federated%20Learning%20for%20OOD%20Generalization/tables/d89811401edcf80ec340a64439273f3cda7f09bb3abe5e7fcd414ec65bd247b6.jpg)

## On the Convergence of Continuous Single-timescale Actor-critic


### Tables

![0285e596c3dce13cf770e289cc4821ef926d50ddfee8c93226503f108d4f80ae.jpg](../icml_results/818_On%20the%20Convergence%20of%20Continuous%20Single-timescale%20Actor-critic/tables/0285e596c3dce13cf770e289cc4821ef926d50ddfee8c93226503f108d4f80ae.jpg)

![da61711f06ae234436c67530dba3bd72d8489bab84d1a9c7132aa5fa7bd1eb3a.jpg](../icml_results/818_On%20the%20Convergence%20of%20Continuous%20Single-timescale%20Actor-critic/tables/da61711f06ae234436c67530dba3bd72d8489bab84d1a9c7132aa5fa7bd1eb3a.jpg)

## POROver: Improving Safety and Reducing Overrefusal in Large Language Models with Overgeneration and Preference Optimization


### Images

![0c3840175da8e35599bce1a91f57949eaf4c5402596ad06ae57c42a87d2fe03e.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/0c3840175da8e35599bce1a91f57949eaf4c5402596ad06ae57c42a87d2fe03e.jpg)

![20173a940ace6c94b67c8bfd0110e261cad7257238032037174e8e93110a0baa.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/20173a940ace6c94b67c8bfd0110e261cad7257238032037174e8e93110a0baa.jpg)

![2ed7eb0545b413b7637a5f15d99b2aa534716ee84f2327ef9d63a7bb12c8e1e3.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/2ed7eb0545b413b7637a5f15d99b2aa534716ee84f2327ef9d63a7bb12c8e1e3.jpg)

![30becf29929cdd26be39bb4234943ed2ac419623a1f76bb1ea780e549a102793.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/30becf29929cdd26be39bb4234943ed2ac419623a1f76bb1ea780e549a102793.jpg)

![48a59cbd515a194741b472c6d214e0aee7880239322fa66be6484f3390a97105.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/48a59cbd515a194741b472c6d214e0aee7880239322fa66be6484f3390a97105.jpg)

![5ab7ebf66e31534a3abd373f41dbe5eeed37829118d764407c88e98a92d8f243.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/5ab7ebf66e31534a3abd373f41dbe5eeed37829118d764407c88e98a92d8f243.jpg)

![63ee04e1bd997853560494055fd28695783bd2398ee01da9c1128e2c23e64b40.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/63ee04e1bd997853560494055fd28695783bd2398ee01da9c1128e2c23e64b40.jpg)

![65c78874b11b113d62f12903dd079cd1b3f598ce3e5150217abeb625947a2e4f.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/65c78874b11b113d62f12903dd079cd1b3f598ce3e5150217abeb625947a2e4f.jpg)

![667142cb982dce601dc9b49991f326e8ff3ad7b703285b61b775d8c04645513e.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/667142cb982dce601dc9b49991f326e8ff3ad7b703285b61b775d8c04645513e.jpg)

![6dcd5243a5596b946ab5603d6283c14e7b24114d97a0f3f3e7ddc1ef91ce664c.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/6dcd5243a5596b946ab5603d6283c14e7b24114d97a0f3f3e7ddc1ef91ce664c.jpg)

![6ddfd19b2bf08fec9b562f11f6adffcb624af5971e90002340e6d98e09e9ef76.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/6ddfd19b2bf08fec9b562f11f6adffcb624af5971e90002340e6d98e09e9ef76.jpg)

![739164365a7046dd86446bbd55124c974f2bbc0107d429759aaebf091d88ea88.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/739164365a7046dd86446bbd55124c974f2bbc0107d429759aaebf091d88ea88.jpg)

![74236d87990aa9123f6c2c116fe83d05a6aca0839afe2b6b9fc75eae4396f2dd.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/74236d87990aa9123f6c2c116fe83d05a6aca0839afe2b6b9fc75eae4396f2dd.jpg)

![89f5b82df30d6bd7084add1debdae6ea3e88f4231a13658a2fce104e00b39a08.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/89f5b82df30d6bd7084add1debdae6ea3e88f4231a13658a2fce104e00b39a08.jpg)

![8a76ebab846090a1c27842d0ca48928ad65cb0771d19fda87b8e5745cc00d71e.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/8a76ebab846090a1c27842d0ca48928ad65cb0771d19fda87b8e5745cc00d71e.jpg)

![8aeae16b42c554a4ac02a1bb9934fd8beffce053e9421fc10dae3a35ed1ba5c3.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/8aeae16b42c554a4ac02a1bb9934fd8beffce053e9421fc10dae3a35ed1ba5c3.jpg)

![91e4e588b87435614d5dff91f9af2c085b40c05b13debea3c06da07bb95da657.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/91e4e588b87435614d5dff91f9af2c085b40c05b13debea3c06da07bb95da657.jpg)

![9749b4131d03311d4fa1719d6dd3c02eb18556189466d83db331048f09bba413.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/9749b4131d03311d4fa1719d6dd3c02eb18556189466d83db331048f09bba413.jpg)

![a093adebd2abfe646c0103af437b99043b571bd44356ed4e111be8e33118c35a.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/a093adebd2abfe646c0103af437b99043b571bd44356ed4e111be8e33118c35a.jpg)

![a199f88133593139cd424457296f2450b2c12751259198f7647cd9c5ad647766.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/a199f88133593139cd424457296f2450b2c12751259198f7647cd9c5ad647766.jpg)

![c2bcd154ffacb11e9b081d6ccb364a8a4176cd28763dbbd0224af0400107c35c.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/c2bcd154ffacb11e9b081d6ccb364a8a4176cd28763dbbd0224af0400107c35c.jpg)

![c5b39856ef04254d1ee3b285338897ebd901f2911c3cf57a2942413992c9b81b.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/c5b39856ef04254d1ee3b285338897ebd901f2911c3cf57a2942413992c9b81b.jpg)

![c7fead89010ca6e55e219cc5d5203a3730b44c00db9f2bb157d40255db1dacf0.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/c7fead89010ca6e55e219cc5d5203a3730b44c00db9f2bb157d40255db1dacf0.jpg)

![d2775c5ef1a506ecd81f1d28fa259bcefad8d59e12fbefe516fb027411b6970d.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/d2775c5ef1a506ecd81f1d28fa259bcefad8d59e12fbefe516fb027411b6970d.jpg)

![ee9a4ca3c1c67e3427184a98369b4ff8072a1dd585ed87752b11510aa3c2ea78.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/ee9a4ca3c1c67e3427184a98369b4ff8072a1dd585ed87752b11510aa3c2ea78.jpg)

![f19795eec3787ff8afff8646f4aa9ba9c00787418358a36ac78d606b9aa68f0b.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/f19795eec3787ff8afff8646f4aa9ba9c00787418358a36ac78d606b9aa68f0b.jpg)

![f3307e6269f48001598b09ec87f48e9ec42fac0003ea690bc69e2643cec92580.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/f3307e6269f48001598b09ec87f48e9ec42fac0003ea690bc69e2643cec92580.jpg)

![f9f80383b48dafd4b4cf198151c6ee0cb7e29e131dbcec16b992953d3cd81ea9.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/f9f80383b48dafd4b4cf198151c6ee0cb7e29e131dbcec16b992953d3cd81ea9.jpg)

![fee886dc07c65782d5f64059abce31178a9a08f248dde825e85b8c3fa40b26c2.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/images/fee886dc07c65782d5f64059abce31178a9a08f248dde825e85b8c3fa40b26c2.jpg)

### Tables

![04885abbe38441a12205131ea9e1b79e72381cc88bbc7a6d29536a64caec8be0.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/04885abbe38441a12205131ea9e1b79e72381cc88bbc7a6d29536a64caec8be0.jpg)

![1f07ec82052ca8f3aad06863827140c036899c93c72a35f83d4ce34cb7dd2340.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/1f07ec82052ca8f3aad06863827140c036899c93c72a35f83d4ce34cb7dd2340.jpg)

![1fe0fedf87cd28e692b6b726afa5cde4b9a916b9b7c10edcddcc18de01f7ee0c.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/1fe0fedf87cd28e692b6b726afa5cde4b9a916b9b7c10edcddcc18de01f7ee0c.jpg)

![27388ce6618672e58cb67b4c1094a87719dd74e0902a0784f47b94b19d7ac5fe.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/27388ce6618672e58cb67b4c1094a87719dd74e0902a0784f47b94b19d7ac5fe.jpg)

![521b6aca8743516de9f6f2d8fb36c5906f1e34556ea3627c10f458cf5bb2c74d.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/521b6aca8743516de9f6f2d8fb36c5906f1e34556ea3627c10f458cf5bb2c74d.jpg)

![5816eeced2c80b1ad054ab1b81047c10a02d5249d48e2bc70af3b412f26ff8ea.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/5816eeced2c80b1ad054ab1b81047c10a02d5249d48e2bc70af3b412f26ff8ea.jpg)

![5a92bcf7818d3a4f0a58b311da7a8ce641e50c2ff8c5a5d991ac8c92c842944b.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/5a92bcf7818d3a4f0a58b311da7a8ce641e50c2ff8c5a5d991ac8c92c842944b.jpg)

![81adefe90e7b940e6f3c81a37f1d67a17de3295ea5a348d180d0ce62288d2fd0.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/81adefe90e7b940e6f3c81a37f1d67a17de3295ea5a348d180d0ce62288d2fd0.jpg)

![bba2fb022b110a75913f33a9aeeed04373e6d48b09fc48def3b4ee8d49700444.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/bba2fb022b110a75913f33a9aeeed04373e6d48b09fc48def3b4ee8d49700444.jpg)

![dcb2f5279d577a7356f06a323aa230a571915c0ff6f4ba595d757a2de0c29256.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/dcb2f5279d577a7356f06a323aa230a571915c0ff6f4ba595d757a2de0c29256.jpg)

![dda7c8705168bf44ecc7d682475ea86fa28fbaaeff111529c6d3b13baf5f5096.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/dda7c8705168bf44ecc7d682475ea86fa28fbaaeff111529c6d3b13baf5f5096.jpg)

![e5655d01f0ae02cb8ded2424be5b3fae095034a9d54ea3a499944ea846cdbb09.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/e5655d01f0ae02cb8ded2424be5b3fae095034a9d54ea3a499944ea846cdbb09.jpg)

![ea927c749752a62167a3e405b5bfb30a40f8e214e3436712b80bec4a7fefcdba.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/ea927c749752a62167a3e405b5bfb30a40f8e214e3436712b80bec4a7fefcdba.jpg)

![fb95b3e7cf7d9828fc2cff2ba855f9c4e69d6feb82381c426ae652321e8cd0d2.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/fb95b3e7cf7d9828fc2cff2ba855f9c4e69d6feb82381c426ae652321e8cd0d2.jpg)

![fc79eda209bd5db5813209a2ec3e7f6015bdf2c40fbb537f8ddc258707f42057.jpg](../icml_results/819_POROver_%20Improving%20Safety%20and%20Reducing%20Overrefusal%20in%20Large%20Language%20Models%20with%20Overgeneration%20and%20/tables/fc79eda209bd5db5813209a2ec3e7f6015bdf2c40fbb537f8ddc258707f42057.jpg)

## Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning


### Images

![0d8365cafae598329fc91700c51032913db703a93f0a0b3b8ccd2161d367eb55.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/0d8365cafae598329fc91700c51032913db703a93f0a0b3b8ccd2161d367eb55.jpg)

![1b48736d7bf9c81deb97a0b76638f47d488259f83551378cd1ee2ec25ef23362.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/1b48736d7bf9c81deb97a0b76638f47d488259f83551378cd1ee2ec25ef23362.jpg)

![1f7ac00eb6aee0a8319863245870d179700b08ff9540f1bd64f6f3dba6c92391.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/1f7ac00eb6aee0a8319863245870d179700b08ff9540f1bd64f6f3dba6c92391.jpg)

![1fa634e5af57f32a3607e666d4fb0439f52890738a7dffa31d41f1edd1530bcd.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/1fa634e5af57f32a3607e666d4fb0439f52890738a7dffa31d41f1edd1530bcd.jpg)

![2fb78b28077b3e9d45e6c0954d28bb5357ff38e96a6c52483ee8846e21140b27.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/2fb78b28077b3e9d45e6c0954d28bb5357ff38e96a6c52483ee8846e21140b27.jpg)

![46f7036665547f763c858f1686d39c1e77f97c5c1962dfd0f877d487e2c245ea.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/46f7036665547f763c858f1686d39c1e77f97c5c1962dfd0f877d487e2c245ea.jpg)

![5369275b45135a86fa4bc79e313099af0656f88f9ca945a2af851981f3bbed7e.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/5369275b45135a86fa4bc79e313099af0656f88f9ca945a2af851981f3bbed7e.jpg)

![56400fdd0d180fe951b5e1b92f3204f21d20675220b2e346aa36c3ec52c0460f.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/56400fdd0d180fe951b5e1b92f3204f21d20675220b2e346aa36c3ec52c0460f.jpg)

![75d0788e5c7df4342bc16c92bcaaa3092521b918d1c6255380aca639f1323b81.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/75d0788e5c7df4342bc16c92bcaaa3092521b918d1c6255380aca639f1323b81.jpg)

![75d149d14fbfb12fc24c3b36cac5161975545a7eb27b150fd0860f6929653d40.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/75d149d14fbfb12fc24c3b36cac5161975545a7eb27b150fd0860f6929653d40.jpg)

![86dd8cc7783eb8ca1cd96047e31d2c789e1417f9568a2298d4d3bf685ef19e1f.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/86dd8cc7783eb8ca1cd96047e31d2c789e1417f9568a2298d4d3bf685ef19e1f.jpg)

![c3aca11cf581b0b5cf1fac41d62665a7a6bdaa359fd0c807e61eb513ff2edf1a.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/c3aca11cf581b0b5cf1fac41d62665a7a6bdaa359fd0c807e61eb513ff2edf1a.jpg)

![cab165e7ec6da09e3a01bd94650eab6b496db3a74805f576b31a677a8c76aa83.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/cab165e7ec6da09e3a01bd94650eab6b496db3a74805f576b31a677a8c76aa83.jpg)

![d219c78d496c26b213023fe71bac169780223fb2b5675cc02c2af9afdc6470f8.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/d219c78d496c26b213023fe71bac169780223fb2b5675cc02c2af9afdc6470f8.jpg)

![ee8e0c0c343d9ad4655d8052fbd6c9e475f179cfd36b4b3cc9d87dee0c80a3f0.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/ee8e0c0c343d9ad4655d8052fbd6c9e475f179cfd36b4b3cc9d87dee0c80a3f0.jpg)

![f252c043d01453baee48adc93c1eaeb68b05fdd3b048cb004ce1db6995d0777e.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/f252c043d01453baee48adc93c1eaeb68b05fdd3b048cb004ce1db6995d0777e.jpg)

![f4644984f5c13c1e091feabc851ac25e8d3702079e42dcf74ddeaa30bffad9b7.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/f4644984f5c13c1e091feabc851ac25e8d3702079e42dcf74ddeaa30bffad9b7.jpg)

![fdac20d731f3a187b3b615eb7df948307e662bc6c4e1555fecb3751ffb693ea0.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/images/fdac20d731f3a187b3b615eb7df948307e662bc6c4e1555fecb3751ffb693ea0.jpg)

### Tables

![0d40fc7385c837d483ecf811104df0b43067a72cece1fa21d31653f28d37d26e.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/tables/0d40fc7385c837d483ecf811104df0b43067a72cece1fa21d31653f28d37d26e.jpg)

![26250549de4297922c2a8913362025c372118fa1b50d72072a11d6b649dc0ae4.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/tables/26250549de4297922c2a8913362025c372118fa1b50d72072a11d6b649dc0ae4.jpg)

![599aa982bce5275116d5801a8aa5bc09c22327efbb28583d6157aabb24f74df6.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/tables/599aa982bce5275116d5801a8aa5bc09c22327efbb28583d6157aabb24f74df6.jpg)

![77e1be8f61e2fe3da438886b7fbb0ffc889d0053030a1c5939e9c91a23316f3e.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/tables/77e1be8f61e2fe3da438886b7fbb0ffc889d0053030a1c5939e9c91a23316f3e.jpg)

![d60dd55024f30fd36f86fff579f8a0c1b2ac47ceb2947aec15694981438229b5.jpg](../icml_results/820_Behavior-Regularized%20Diffusion%20Policy%20Optimization%20for%20Offline%20Reinforcement%20Learning/tables/d60dd55024f30fd36f86fff579f8a0c1b2ac47ceb2947aec15694981438229b5.jpg)

## Is Noise Conditioning Necessary for Denoising Generative Models?


### Images

![00be86001481354fa1255f91eb0f731b2b91d313a43ec08da54b9d5f13b03de8.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/00be86001481354fa1255f91eb0f731b2b91d313a43ec08da54b9d5f13b03de8.jpg)

![1731aa3edbd20ed7de84946e6145bbf9537bb71cafaff45259ce995587b0c116.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/1731aa3edbd20ed7de84946e6145bbf9537bb71cafaff45259ce995587b0c116.jpg)

![266af812ce29d0ec479f53001857d17243ab57f5a0b9932d072a0b381f043b10.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/266af812ce29d0ec479f53001857d17243ab57f5a0b9932d072a0b381f043b10.jpg)

![2f6a994d091e49ae148242ce47701807ed3a12f412b13e5b6229a5970c4169aa.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/2f6a994d091e49ae148242ce47701807ed3a12f412b13e5b6229a5970c4169aa.jpg)

![33f0295f53104be7f6a631e2f1f5cdd448691d75bc014d80739735063f9f9869.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/33f0295f53104be7f6a631e2f1f5cdd448691d75bc014d80739735063f9f9869.jpg)

![371a22bcc80880be0c139be119f98bfbce3155f17adabe293e35fc6baba044df.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/371a22bcc80880be0c139be119f98bfbce3155f17adabe293e35fc6baba044df.jpg)

![64dd2a8ce6e014f9e785ec27c9e822f9d12caa3dfc9124da215c4588a798011c.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/64dd2a8ce6e014f9e785ec27c9e822f9d12caa3dfc9124da215c4588a798011c.jpg)

![6c6292a805facd256e80cb4a29b81dbfe39b932594f27d9b0b0e602ee4d64ff0.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/6c6292a805facd256e80cb4a29b81dbfe39b932594f27d9b0b0e602ee4d64ff0.jpg)

![7562f00546b60d23af203d30d571d06166e6e4c15830e71633b936235f8a9b33.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/7562f00546b60d23af203d30d571d06166e6e4c15830e71633b936235f8a9b33.jpg)

![98e14989344a939c82106656b4bc3fc7c8dd4b6070a1c54e9fca9db5f229d293.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/98e14989344a939c82106656b4bc3fc7c8dd4b6070a1c54e9fca9db5f229d293.jpg)

![c13daeb3069c8452e23a798fef35ab72362ba5a59ac7a47455cd51be1e0f5a54.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/c13daeb3069c8452e23a798fef35ab72362ba5a59ac7a47455cd51be1e0f5a54.jpg)

![c214190eb119835b5475d8d46f500de7c3bcd84f983a6569f861d67186df7415.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/c214190eb119835b5475d8d46f500de7c3bcd84f983a6569f861d67186df7415.jpg)

![c709795316f650d3978c837dbad41c392681ad204cd38edf954a387f6ce8855c.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/c709795316f650d3978c837dbad41c392681ad204cd38edf954a387f6ce8855c.jpg)

![e55f3498e393fb04185721c18e2c48deefa5748bdcbcc263c9376963a093bcc3.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/images/e55f3498e393fb04185721c18e2c48deefa5748bdcbcc263c9376963a093bcc3.jpg)

### Tables

![01c68cf4381cab701a689b8f8c396828deef6236fc8c0a12e6fa2a3c7c239f85.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/01c68cf4381cab701a689b8f8c396828deef6236fc8c0a12e6fa2a3c7c239f85.jpg)

![28bccaa326465d4ad55196376309e2b22fab83dc0b96d18122e5a5da966f2cd8.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/28bccaa326465d4ad55196376309e2b22fab83dc0b96d18122e5a5da966f2cd8.jpg)

![4d9ee9acd039d2b36b02d2d51e6c876344f63a957166d2ce1168268f026e3d1c.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/4d9ee9acd039d2b36b02d2d51e6c876344f63a957166d2ce1168268f026e3d1c.jpg)

![8abe455b6f6647795d2d413546b81d512e4e0f8e170856d01461fdb410e41c29.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/8abe455b6f6647795d2d413546b81d512e4e0f8e170856d01461fdb410e41c29.jpg)

![9345334b0b6babb4adf26660e8730bccb5ab315788c27ef426a6acef5be9a5e0.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/9345334b0b6babb4adf26660e8730bccb5ab315788c27ef426a6acef5be9a5e0.jpg)

![a9a87666dc03dc166aa9a371afe247d902b1af0a0c16f1aae8caac46a49dad4e.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/a9a87666dc03dc166aa9a371afe247d902b1af0a0c16f1aae8caac46a49dad4e.jpg)

![ac12e9c68ca873092b49a8127a18e545d57c0431926ef382cf4200de82a0c892.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/ac12e9c68ca873092b49a8127a18e545d57c0431926ef382cf4200de82a0c892.jpg)

![c93b587b0a3c7dffafd267005a8ab17be10c0203375137709467824c2be6a058.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/c93b587b0a3c7dffafd267005a8ab17be10c0203375137709467824c2be6a058.jpg)

![f992fa19b037205fdc1044e31fd7cd1a95beab290fff04b5fedee0ec4eccf161.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/f992fa19b037205fdc1044e31fd7cd1a95beab290fff04b5fedee0ec4eccf161.jpg)

![fb612652752985caf83d0bc658bada4524da16f65c4169ac3804711eddb0f768.jpg](../icml_results/821_Is%20Noise%20Conditioning%20Necessary%20for%20Denoising%20Generative%20Models_/tables/fb612652752985caf83d0bc658bada4524da16f65c4169ac3804711eddb0f768.jpg)

## Learning Efficient Robotic Garment Manipulation with Standardization


### Images

![0e8c5587bb0fdd39c5f10843741e261e8f76eb7daa43e69ed8ca8db63270659a.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/0e8c5587bb0fdd39c5f10843741e261e8f76eb7daa43e69ed8ca8db63270659a.jpg)

![15e8f5f1a1fdd0420e5d2f0261994101c2a6f1166487df2483771d4e0375eeb2.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/15e8f5f1a1fdd0420e5d2f0261994101c2a6f1166487df2483771d4e0375eeb2.jpg)

![24ce0c534c2834778971c2602728927f9e0be519102980e918d9f4d5cbf90b30.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/24ce0c534c2834778971c2602728927f9e0be519102980e918d9f4d5cbf90b30.jpg)

![2926e94da0af3144da8d6ed8c8daf8f166ec2358e4b7b348cbe1474a61da0c76.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/2926e94da0af3144da8d6ed8c8daf8f166ec2358e4b7b348cbe1474a61da0c76.jpg)

![3149f818610243508f197db72344ff3e74a08e81ec53f3979572b0b93a2b9f4a.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/3149f818610243508f197db72344ff3e74a08e81ec53f3979572b0b93a2b9f4a.jpg)

![5404a3a99959295ace7bf37ea93f513562f71ebc1b3adf87eaef3d01b0583a38.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/5404a3a99959295ace7bf37ea93f513562f71ebc1b3adf87eaef3d01b0583a38.jpg)

![5b26d9f25027cf4026ff6cffc768c6e7e530e73d6431c182ef96daa7b96467f5.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/5b26d9f25027cf4026ff6cffc768c6e7e530e73d6431c182ef96daa7b96467f5.jpg)

![8705aa6f70a0bbf9c89182a7edb08196fa39958f2b9118bd4d07fbbd9f9f6ed0.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/8705aa6f70a0bbf9c89182a7edb08196fa39958f2b9118bd4d07fbbd9f9f6ed0.jpg)

![894dc67c71fcc8cef6389f52a79b84c380e38d81871ff59b57a1e7a751c98e2c.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/894dc67c71fcc8cef6389f52a79b84c380e38d81871ff59b57a1e7a751c98e2c.jpg)

![93f33dd97b147697bb6c078676b2baab40a94bf584a624a85b1185429b29d0cc.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/93f33dd97b147697bb6c078676b2baab40a94bf584a624a85b1185429b29d0cc.jpg)

![a7419372ea5d25cf5930d6d8736ffb00b9360a30c865cf6eecd0d223d3717767.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/a7419372ea5d25cf5930d6d8736ffb00b9360a30c865cf6eecd0d223d3717767.jpg)

![c2a36f63adb8b3989f314aa31cbef1b9f74d148729b4bdc4588378d37bcbfb1b.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/c2a36f63adb8b3989f314aa31cbef1b9f74d148729b4bdc4588378d37bcbfb1b.jpg)

![c48989a36b6b1eb9effc934e100f7301b4fcf4cac79a8c787335b097a1c52673.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/c48989a36b6b1eb9effc934e100f7301b4fcf4cac79a8c787335b097a1c52673.jpg)

![c72287b5a6a4beb3a3a2201123adbf36acbd20c53d44cd56f4b41b5284101ffa.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/c72287b5a6a4beb3a3a2201123adbf36acbd20c53d44cd56f4b41b5284101ffa.jpg)

![c80552c97a77d1fcaf2d1cd567383a0953869445bac6c2ec33a8f5c290f72363.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/c80552c97a77d1fcaf2d1cd567383a0953869445bac6c2ec33a8f5c290f72363.jpg)

![c8bd3d4575baa1653c2cff4fe4823d8e65e96f581fca9fd3d012615c960e1be0.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/c8bd3d4575baa1653c2cff4fe4823d8e65e96f581fca9fd3d012615c960e1be0.jpg)

![d184cc7f5253e81f747c8bc5e7b9aae9eddd0e46bd51dc30ce7f682006c54e0f.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/d184cc7f5253e81f747c8bc5e7b9aae9eddd0e46bd51dc30ce7f682006c54e0f.jpg)

![d9b817bb1b1b2934ccb3e64103c5bc78ad59644d3ad9ff0df720da739e84a62d.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/images/d9b817bb1b1b2934ccb3e64103c5bc78ad59644d3ad9ff0df720da739e84a62d.jpg)

### Tables

![229793c75fd578ba3d3920760bf30f738fb15a86c03d6356e1357b4ad5aef7ca.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/tables/229793c75fd578ba3d3920760bf30f738fb15a86c03d6356e1357b4ad5aef7ca.jpg)

![7667da367bc604e4843c6d03f3940436746e0d3d29c90cae24fa9c0b2145a3c6.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/tables/7667da367bc604e4843c6d03f3940436746e0d3d29c90cae24fa9c0b2145a3c6.jpg)

![8bf03022f1a4d0995aff90cb47d550b1ab9cb9b3e746f9577f365731c2c946b5.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/tables/8bf03022f1a4d0995aff90cb47d550b1ab9cb9b3e746f9577f365731c2c946b5.jpg)

![aa2d9ae5ce6228060f2bb7948409cc63adf0194a71d802a8b3421b3a658ab01e.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/tables/aa2d9ae5ce6228060f2bb7948409cc63adf0194a71d802a8b3421b3a658ab01e.jpg)

![c65b9721958cab14aa7a548a1f545234d3eacbabf1eb803f0527ec3e73402576.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/tables/c65b9721958cab14aa7a548a1f545234d3eacbabf1eb803f0527ec3e73402576.jpg)

![c764cacc5013f8eb2d67e177d71d38faac24756266c43b81fc9ab1edbd438159.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/tables/c764cacc5013f8eb2d67e177d71d38faac24756266c43b81fc9ab1edbd438159.jpg)

![ec4b811ce8f89f597d5392a1bd6fe910a3c90af6ca9229435fd6b474d923c509.jpg](../icml_results/822_Learning%20Efficient%20Robotic%20Garment%20Manipulation%20with%20Standardization/tables/ec4b811ce8f89f597d5392a1bd6fe910a3c90af6ca9229435fd6b474d923c509.jpg)

## Effective and Efficient Masked Image Generation Models


### Images

![104c5648ea876860af2e5ab48125b2d195d378e9cabe812436c878fc8aba4f46.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/104c5648ea876860af2e5ab48125b2d195d378e9cabe812436c878fc8aba4f46.jpg)

![2f1be3574efe3976d737f826d008d125c34486825e4b6b7bedb4eb859da4b734.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/2f1be3574efe3976d737f826d008d125c34486825e4b6b7bedb4eb859da4b734.jpg)

![4981aded5c645be9a8b928c758d4bb585421229c47c615c5cf8e3eaac4f6428b.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/4981aded5c645be9a8b928c758d4bb585421229c47c615c5cf8e3eaac4f6428b.jpg)

![69ce806c453ddc945d690335f118a5e76b5170a1db93c82733e580b7c9fe9e2e.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/69ce806c453ddc945d690335f118a5e76b5170a1db93c82733e580b7c9fe9e2e.jpg)

![7822b81c0c112fcb26f6cc18b5987aea3cd80f999e6b2a291239f2e7c58c1da7.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/7822b81c0c112fcb26f6cc18b5987aea3cd80f999e6b2a291239f2e7c58c1da7.jpg)

![7a2a800967b6db47575ba4b38dc24fe42a41ad99aed109ab29f65a6734aa38ff.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/7a2a800967b6db47575ba4b38dc24fe42a41ad99aed109ab29f65a6734aa38ff.jpg)

![7ab04b80c9897ee002215ee643c130f8c5ce598ce7ec8f4fdb8c7bfd241d8528.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/7ab04b80c9897ee002215ee643c130f8c5ce598ce7ec8f4fdb8c7bfd241d8528.jpg)

![d7ab8870fee5a3c81bf8bb4d974e14cbfeb3b12c84ba4ea99dca6998be877961.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/d7ab8870fee5a3c81bf8bb4d974e14cbfeb3b12c84ba4ea99dca6998be877961.jpg)

![eda942ca1fd72e5bbddd5085bd02f1b49627ada7d1332c7da4e44db2366b8c1c.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/images/eda942ca1fd72e5bbddd5085bd02f1b49627ada7d1332c7da4e44db2366b8c1c.jpg)

### Tables

![0d14ff86af2e4077c255bd99e59348406967e4c2ecc339df071b95a6e2f791ea.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/0d14ff86af2e4077c255bd99e59348406967e4c2ecc339df071b95a6e2f791ea.jpg)

![0d229db5edbb99368020ba5deaf150cf88822474576b510c8a254bbefb999afd.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/0d229db5edbb99368020ba5deaf150cf88822474576b510c8a254bbefb999afd.jpg)

![2daf2be27607f02446e85dfd6ec99d5f45b745514439ef7c784595590ddb6d59.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/2daf2be27607f02446e85dfd6ec99d5f45b745514439ef7c784595590ddb6d59.jpg)

![2f61216c0b13f60fbb5c776622ab84b3d7e6cbc1984eced8032467c6ff61006d.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/2f61216c0b13f60fbb5c776622ab84b3d7e6cbc1984eced8032467c6ff61006d.jpg)

![59dd4726eddc3392aaa43b186f410fdcddceab1f3ca729747896e2df72b8cea5.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/59dd4726eddc3392aaa43b186f410fdcddceab1f3ca729747896e2df72b8cea5.jpg)

![7826ff753758880ec3f098baf528f21b763b4b77b886667668ecc593f53eb1e5.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/7826ff753758880ec3f098baf528f21b763b4b77b886667668ecc593f53eb1e5.jpg)

![7ad8a64fdb2d762d6174fdd011c3b42fb33975450b9e6b4943d7ae72dabf3017.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/7ad8a64fdb2d762d6174fdd011c3b42fb33975450b9e6b4943d7ae72dabf3017.jpg)

![83c5d93f6e9d2dc44c857e6afe44598c9e4a668cb126bd7c6df41fd8c0a098d3.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/83c5d93f6e9d2dc44c857e6afe44598c9e4a668cb126bd7c6df41fd8c0a098d3.jpg)

![936f3727853ea654e63df18a15401a02fc23a8cd51a805d09d4c030b355e3997.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/936f3727853ea654e63df18a15401a02fc23a8cd51a805d09d4c030b355e3997.jpg)

![a26a1bf01548c4c963887b80463756d45f45c801c38c11237ff4edae1b8c9c8b.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/a26a1bf01548c4c963887b80463756d45f45c801c38c11237ff4edae1b8c9c8b.jpg)

![e71641e57be57f644ee4d78e83bae430c1b81308ea0e63c260d09bf52ff6cecc.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/e71641e57be57f644ee4d78e83bae430c1b81308ea0e63c260d09bf52ff6cecc.jpg)

![f04fb19c43df318cd8bd683aa9352bed6730e4862be71ac96108b543e86a70f0.jpg](../icml_results/823_Effective%20and%20Efficient%20Masked%20Image%20Generation%20Models/tables/f04fb19c43df318cd8bd683aa9352bed6730e4862be71ac96108b543e86a70f0.jpg)

## Efficient Heterogeneity-Aware Federated Active Data Selection


### Images

![0c5abddb75806c05799ddfb7ebc2b45132da646e10c6620219a5b2ca97c7e2c6.jpg](../icml_results/824_Efficient%20Heterogeneity-Aware%20Federated%20Active%20Data%20Selection/images/0c5abddb75806c05799ddfb7ebc2b45132da646e10c6620219a5b2ca97c7e2c6.jpg)

![16d32a3081b1b117f1534a0098004c2b16f99d84b7eb6981b9ffaacfb6524101.jpg](../icml_results/824_Efficient%20Heterogeneity-Aware%20Federated%20Active%20Data%20Selection/images/16d32a3081b1b117f1534a0098004c2b16f99d84b7eb6981b9ffaacfb6524101.jpg)

![334fb97150e9703664cce461ed752eb47c33d424435ddfa7e90e95a22cb7dea8.jpg](../icml_results/824_Efficient%20Heterogeneity-Aware%20Federated%20Active%20Data%20Selection/images/334fb97150e9703664cce461ed752eb47c33d424435ddfa7e90e95a22cb7dea8.jpg)

### Tables

![0f986d7e37494595ba3630bd7118bae70820c5e29092d1d8000bd5bb68a42620.jpg](../icml_results/824_Efficient%20Heterogeneity-Aware%20Federated%20Active%20Data%20Selection/tables/0f986d7e37494595ba3630bd7118bae70820c5e29092d1d8000bd5bb68a42620.jpg)

![928ad2ec42834b155ef7c0c17b8b81a6b3953e0a1a16496077f5829b5e0ea80e.jpg](../icml_results/824_Efficient%20Heterogeneity-Aware%20Federated%20Active%20Data%20Selection/tables/928ad2ec42834b155ef7c0c17b8b81a6b3953e0a1a16496077f5829b5e0ea80e.jpg)

![cb8b62dca4ab9bd776794c4b2ae8fe2fe05bf450e2c34a65c4d553af9ddf759f.jpg](../icml_results/824_Efficient%20Heterogeneity-Aware%20Federated%20Active%20Data%20Selection/tables/cb8b62dca4ab9bd776794c4b2ae8fe2fe05bf450e2c34a65c4d553af9ddf759f.jpg)

## UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning


### Images

![0060f1fb03b1189eb478a99dfd064d3c09862b19ae82173a71108bdb2929d2ba.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/0060f1fb03b1189eb478a99dfd064d3c09862b19ae82173a71108bdb2929d2ba.jpg)

![2370b0d176b9612bbe7d1e3f4962761117e5db9767e1d1563a5ec4a0d2d39297.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/2370b0d176b9612bbe7d1e3f4962761117e5db9767e1d1563a5ec4a0d2d39297.jpg)

![2532ac5f657404da8f099ff3bce3b1beb8b51d6a4273e53ff57357ceeb7c0e15.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/2532ac5f657404da8f099ff3bce3b1beb8b51d6a4273e53ff57357ceeb7c0e15.jpg)

![379ab838b75e3d8715422ae61fc67100dafc33f969dc67a1e1fedddcfccc5d1e.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/379ab838b75e3d8715422ae61fc67100dafc33f969dc67a1e1fedddcfccc5d1e.jpg)

![41033c8834e5e60b34081d47b1e9e8d7a839066d9919e91defe4cb392436db36.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/41033c8834e5e60b34081d47b1e9e8d7a839066d9919e91defe4cb392436db36.jpg)

![54f32ea724e669436942b76ee99f19c7d09af1743abf0cdbf1ede38696133ee2.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/54f32ea724e669436942b76ee99f19c7d09af1743abf0cdbf1ede38696133ee2.jpg)

![58c18e093802c3e0069456636793073047f961a015c43e37e1ae463c622f6629.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/58c18e093802c3e0069456636793073047f961a015c43e37e1ae463c622f6629.jpg)

![729e015a2d494a44231f1ea9014dd46867f365f29599b0428ac489269dc433ed.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/729e015a2d494a44231f1ea9014dd46867f365f29599b0428ac489269dc433ed.jpg)

![808bed011423dd1ac3ecd0e3120ba7b2c233ad35dfe5214215cbd1e41b0cf232.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/808bed011423dd1ac3ecd0e3120ba7b2c233ad35dfe5214215cbd1e41b0cf232.jpg)

![9004ecea3e1481f1efc06ce0cdaa3c14382daacc67d1f81a17e1cb42ec6f915d.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/9004ecea3e1481f1efc06ce0cdaa3c14382daacc67d1f81a17e1cb42ec6f915d.jpg)

![a5c8addf8c3a544859a31ed7b5cfcb652ada3b605197f686d85e0c7af6249bcd.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/a5c8addf8c3a544859a31ed7b5cfcb652ada3b605197f686d85e0c7af6249bcd.jpg)

![c8be917cd2b7151ba3e0ad2675111ff7054bc69ae8ece1320067cf9c52073371.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/c8be917cd2b7151ba3e0ad2675111ff7054bc69ae8ece1320067cf9c52073371.jpg)

![d64ef73925d1e253e6a7142cbfc4d3030f89bf690a6831efa85d9ace4bffbc3f.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/images/d64ef73925d1e253e6a7142cbfc4d3030f89bf690a6831efa85d9ace4bffbc3f.jpg)

### Tables

![383deae81ede8ab5fb83dd98665c3f4d15bc657db082c7964744f5624309cd9e.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/tables/383deae81ede8ab5fb83dd98665c3f4d15bc657db082c7964744f5624309cd9e.jpg)

![463ff9b3dd96246b75f59c731b670429c59db41c097dd880ce15b2035d979f9f.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/tables/463ff9b3dd96246b75f59c731b670429c59db41c097dd880ce15b2035d979f9f.jpg)

![631a2656240ba3a702b5d577b48d772ff45824df01dde7bf525f6f31196157e7.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/tables/631a2656240ba3a702b5d577b48d772ff45824df01dde7bf525f6f31196157e7.jpg)

![b197502d88337d657c643d8c8ec3f99ad376268fd9a55cc19c6b60981c23bd45.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/tables/b197502d88337d657c643d8c8ec3f99ad376268fd9a55cc19c6b60981c23bd45.jpg)

![ca356e3e2edb45274a26c9fb1d8d3b11934f4fe077595edf0e9e2c7c81370adf.jpg](../icml_results/825_UDora_%20A%20Unified%20Red%20Teaming%20Framework%20against%20LLM%20Agents%20by%20Dynamically%20Hijacking%20Their%20Own%20Reasoni/tables/ca356e3e2edb45274a26c9fb1d8d3b11934f4fe077595edf0e9e2c7c81370adf.jpg)

## SPEX: Scaling Feature Interaction Explanations for LLMs


### Images

![37cb7cedc782a596d32247a1cb70458b5aa501ad48b701ad4ee7646ad193303d.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/37cb7cedc782a596d32247a1cb70458b5aa501ad48b701ad4ee7646ad193303d.jpg)

![43dd2e88d2e4c0b6bf4bf426697bd67757340b0cd71a794e47b883b936e92933.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/43dd2e88d2e4c0b6bf4bf426697bd67757340b0cd71a794e47b883b936e92933.jpg)

![50251f10dc15ab363bdba973479ec4883e8c83f70744343621209b3788633916.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/50251f10dc15ab363bdba973479ec4883e8c83f70744343621209b3788633916.jpg)

![81e9e95ab2ab29c9476dd0f94ce606afe4f95537d2654c7c039942906b1ef6f9.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/81e9e95ab2ab29c9476dd0f94ce606afe4f95537d2654c7c039942906b1ef6f9.jpg)

![9d8eca91c1193d57720fa5a81988c45ded23d39f09cb8ab6e1a1d5e2c9f2f115.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/9d8eca91c1193d57720fa5a81988c45ded23d39f09cb8ab6e1a1d5e2c9f2f115.jpg)

![e75e52c457824a9e05c243db6db7633718ce7a03570031c7ab9cae79f6d680d1.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/e75e52c457824a9e05c243db6db7633718ce7a03570031c7ab9cae79f6d680d1.jpg)

![ed9858c0258221eacc952affcc9db470cb6a4756e13efd011384ac33aade8d87.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/ed9858c0258221eacc952affcc9db470cb6a4756e13efd011384ac33aade8d87.jpg)

![ee8ec893f6c5af478f6f68a0a3e1036c511182a420ebc46c35f203320d8ea857.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/ee8ec893f6c5af478f6f68a0a3e1036c511182a420ebc46c35f203320d8ea857.jpg)

![f66d40520e3acf3a16eee4095d7a59a9dc76cbed89703d898c376a3697031fc3.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/f66d40520e3acf3a16eee4095d7a59a9dc76cbed89703d898c376a3697031fc3.jpg)

![f6fcc8a4a3f8020d37e321bbc3d80d9f9a121999aab9cce6068d921085165ca1.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/f6fcc8a4a3f8020d37e321bbc3d80d9f9a121999aab9cce6068d921085165ca1.jpg)

![f72f3d70036b561289ec2aa036ceb5a59120ca794daa3547027d515eabbf9e8f.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/images/f72f3d70036b561289ec2aa036ceb5a59120ca794daa3547027d515eabbf9e8f.jpg)

### Tables

![1c09151af8477fc64cda70d673a90e25e6ed02fe468a7222b4b94d6e51118d30.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/tables/1c09151af8477fc64cda70d673a90e25e6ed02fe468a7222b4b94d6e51118d30.jpg)

![7379a8bb4081bc222149d42d70849e2589445851f22d34b21a73e67215e40dea.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/tables/7379a8bb4081bc222149d42d70849e2589445851f22d34b21a73e67215e40dea.jpg)

![a87e3e7ae50118026d5cfe3354c01ff98b0b9fa4ae24c52eadf04ed7971ca02d.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/tables/a87e3e7ae50118026d5cfe3354c01ff98b0b9fa4ae24c52eadf04ed7971ca02d.jpg)

![bae048a87ecb32f68b0a9fe7ce9d181b637587a2db5b1777507a301978a85a79.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/tables/bae048a87ecb32f68b0a9fe7ce9d181b637587a2db5b1777507a301978a85a79.jpg)

![cebf939e387b80cc67ef2a059376a798d673ee74125e2b5fe21e11f2ee631847.jpg](../icml_results/826_SPEX_%20Scaling%20Feature%20Interaction%20Explanations%20for%20LLMs/tables/cebf939e387b80cc67ef2a059376a798d673ee74125e2b5fe21e11f2ee631847.jpg)

## Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens


### Images

![51ef424c8726575469457fb74b9a98645f1f275eace860613da1ee9fb78462ed.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/images/51ef424c8726575469457fb74b9a98645f1f275eace860613da1ee9fb78462ed.jpg)

![72518ba4a17b6b75298fbe2fa2bbf5918fe802d6efef5e3ff82a4b2ad00bd299.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/images/72518ba4a17b6b75298fbe2fa2bbf5918fe802d6efef5e3ff82a4b2ad00bd299.jpg)

![90e6ea97928a7795b97942ae6eaa213e319cf4b5163f9c172a893255ff24a3ca.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/images/90e6ea97928a7795b97942ae6eaa213e319cf4b5163f9c172a893255ff24a3ca.jpg)

![c601a2176f87b47f5563ef4535dcf79ce4de2c9cd7bcf2c8960f8a961f841f5d.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/images/c601a2176f87b47f5563ef4535dcf79ce4de2c9cd7bcf2c8960f8a961f841f5d.jpg)

![c7fc1b5aa3e45211ce05c86acf054fb367a88d7313ef064536d0c7f3925adabf.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/images/c7fc1b5aa3e45211ce05c86acf054fb367a88d7313ef064536d0c7f3925adabf.jpg)

![d4971f77ae64c654042c3d1beebd4897955a330d04be16caf348316b1c3673a7.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/images/d4971f77ae64c654042c3d1beebd4897955a330d04be16caf348316b1c3673a7.jpg)

![eeef9dd638e61676631713fb20857726eb2ee03dc541d642b759149732e67063.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/images/eeef9dd638e61676631713fb20857726eb2ee03dc541d642b759149732e67063.jpg)

### Tables

![05edf848eb1db65b830daa120cb8a8d9c0ff9126807dde4755e13038db4a34ec.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/05edf848eb1db65b830daa120cb8a8d9c0ff9126807dde4755e13038db4a34ec.jpg)

![25c64c7d17b705a2d62a18ce39741c16299a470313429c354b482dc6ed9460ed.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/25c64c7d17b705a2d62a18ce39741c16299a470313429c354b482dc6ed9460ed.jpg)

![2f0193058aaa7010d6966c03dfa2aee77e1f1573fa1957f776296821df7cdaad.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/2f0193058aaa7010d6966c03dfa2aee77e1f1573fa1957f776296821df7cdaad.jpg)

![42c033af45c26f9dfc8029c64cd9c65b30568fe8bc53cc521bfc399a3a82e8be.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/42c033af45c26f9dfc8029c64cd9c65b30568fe8bc53cc521bfc399a3a82e8be.jpg)

![4e78579d227ced4c6e3ca22c407a195677e9a592fa6bfb4823fba2725a4d99c6.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/4e78579d227ced4c6e3ca22c407a195677e9a592fa6bfb4823fba2725a4d99c6.jpg)

![52809ae32bf5a9c3c52a9a3fdc7df3500029ef68409c6af62e3733a5f825829f.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/52809ae32bf5a9c3c52a9a3fdc7df3500029ef68409c6af62e3733a5f825829f.jpg)

![5a321f47684485e4e075e7bf25e87aab3aff3570bb1b9acfc878c866574883d7.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/5a321f47684485e4e075e7bf25e87aab3aff3570bb1b9acfc878c866574883d7.jpg)

![647b311d737dbb7c325888824cb33d1f18b2a3d70d9b2b2a6375961953bcb3f4.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/647b311d737dbb7c325888824cb33d1f18b2a3d70d9b2b2a6375961953bcb3f4.jpg)

![894a312f6c4d764ed0b9b3d7e12039c5533d540e9713d8c67bde650e76a1e6b2.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/894a312f6c4d764ed0b9b3d7e12039c5533d540e9713d8c67bde650e76a1e6b2.jpg)

![8ba566eee57ed7824f8669a1c85801019a868dd49dde0dace5dd549bb9679768.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/8ba566eee57ed7824f8669a1c85801019a868dd49dde0dace5dd549bb9679768.jpg)

![af9c7d1e2beb7e9028fba189b2f69641420d109cda708e919d9cdf2b5c9df7e3.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/af9c7d1e2beb7e9028fba189b2f69641420d109cda708e919d9cdf2b5c9df7e3.jpg)

![cda6fdc8e66da0ed7681616c742719a9d3a04a8453bfbe66bef0b6f99ad4f698.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/cda6fdc8e66da0ed7681616c742719a9d3a04a8453bfbe66bef0b6f99ad4f698.jpg)

![cfb2eedf1184d112282bea8d6d601dc25e2b0bef0bed9b842b9244aa680be286.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/cfb2eedf1184d112282bea8d6d601dc25e2b0bef0bed9b842b9244aa680be286.jpg)

![d5c1ed1e1b0e81191f6273d2f2c8991effef350ba487405bd566b967b8407f0d.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/d5c1ed1e1b0e81191f6273d2f2c8991effef350ba487405bd566b967b8407f0d.jpg)

![e48a237630a6280107ef5d92233bbb947f1e1669da5837d2226516a8cc1151a4.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/e48a237630a6280107ef5d92233bbb947f1e1669da5837d2226516a8cc1151a4.jpg)

![eeda68b9cccb5af7e261b419878ec9d96b04bb4eda3df78a2b4d018d88427472.jpg](../icml_results/827_Reflect-then-Plan_%20Offline%20Model-Based%20Planning%20through%20a%20Doubly%20Bayesian%20Lens/tables/eeda68b9cccb5af7e261b419878ec9d96b04bb4eda3df78a2b4d018d88427472.jpg)

## Preconditioned Riemannian Gradient Descent Algorithm for Low-Multilinear-Rank Tensor Completion


### Images

![0347e6759b1b4973e9cbb73f183f77e1988af62f6724a106fe04654417397746.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/0347e6759b1b4973e9cbb73f183f77e1988af62f6724a106fe04654417397746.jpg)

![131b5b776b588c04f2a43a48947e4fcb2dd6fc02307ebde635ca04e4fccda7fd.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/131b5b776b588c04f2a43a48947e4fcb2dd6fc02307ebde635ca04e4fccda7fd.jpg)

![2a536aa38c436fb311130474ad99059c7c676b802d8c896de0f41e6bc7318b44.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/2a536aa38c436fb311130474ad99059c7c676b802d8c896de0f41e6bc7318b44.jpg)

![2c989bb89f089b6d665d6d59dfb117226f8b7a7724a7bb5b8375987269ec10f0.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/2c989bb89f089b6d665d6d59dfb117226f8b7a7724a7bb5b8375987269ec10f0.jpg)

![4809ef31427ef6cc66cc32c74080b85a9d6209f38d63e9534e0d9f383570c930.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/4809ef31427ef6cc66cc32c74080b85a9d6209f38d63e9534e0d9f383570c930.jpg)

![54b4c0994780165dc62c58716bec434b73c968148bb59d7d43bd1e17036d421f.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/54b4c0994780165dc62c58716bec434b73c968148bb59d7d43bd1e17036d421f.jpg)

![5a86b27badba20f1c953cf90b905d56f179429fc8cc5075e4928f9dc8b3b750f.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/5a86b27badba20f1c953cf90b905d56f179429fc8cc5075e4928f9dc8b3b750f.jpg)

![65e8d3fd81db69ec864438d83336e1d1b5e6f799ebe6739754fbf6d3cb509517.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/65e8d3fd81db69ec864438d83336e1d1b5e6f799ebe6739754fbf6d3cb509517.jpg)

![76de272a75cee2ea47820f7ad0d1e3ac8939d9ad984bfd1cb1e4632dc0ab45e6.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/76de272a75cee2ea47820f7ad0d1e3ac8939d9ad984bfd1cb1e4632dc0ab45e6.jpg)

![8357ed46bdf6f4a9c663bec3e702393178679aa1b0cf2d5ad57b27a6fcfea2ea.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/8357ed46bdf6f4a9c663bec3e702393178679aa1b0cf2d5ad57b27a6fcfea2ea.jpg)

![931905da00fba6b00b370e6c87cdeb3ab0ca921f3f863831bc2fce54ac8a2bcd.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/931905da00fba6b00b370e6c87cdeb3ab0ca921f3f863831bc2fce54ac8a2bcd.jpg)

![a623b27427732bd1ea9f18f27d1e714a31e702c214696e7db27e24822e981f77.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/a623b27427732bd1ea9f18f27d1e714a31e702c214696e7db27e24822e981f77.jpg)

![a8f7558534a3323e9b1c29fbe0e9cdff95fc2bf597e829d202704597c78f0cf2.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/a8f7558534a3323e9b1c29fbe0e9cdff95fc2bf597e829d202704597c78f0cf2.jpg)

![d432a76d62aa8f2e8b55f105f8602cfa2ea055994ee394bc92c782cb045b63d7.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/images/d432a76d62aa8f2e8b55f105f8602cfa2ea055994ee394bc92c782cb045b63d7.jpg)

### Tables

![27c94d9c47e152e14e10ffe9d8b871bbe75d3c192eae4c7c78c57d6e679f0eb0.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/tables/27c94d9c47e152e14e10ffe9d8b871bbe75d3c192eae4c7c78c57d6e679f0eb0.jpg)

![647795b5f752864f4c8447bbf896300a7f1d76bb992d2d51ca71af0c384c8427.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/tables/647795b5f752864f4c8447bbf896300a7f1d76bb992d2d51ca71af0c384c8427.jpg)

![9d5266c0aa4c2bde2c618d020a223312f3c97b0f7eedb879cadbfc8d5708f942.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/tables/9d5266c0aa4c2bde2c618d020a223312f3c97b0f7eedb879cadbfc8d5708f942.jpg)

![e18ebfc65a1a5f4938d69532e4243bb3f8a57af4dd0a6fc629b0510c67a64abe.jpg](../icml_results/828_Preconditioned%20Riemannian%20Gradient%20Descent%20Algorithm%20for%20Low-Multilinear-Rank%20Tensor%20Completion/tables/e18ebfc65a1a5f4938d69532e4243bb3f8a57af4dd0a6fc629b0510c67a64abe.jpg)

## Semantics-aware Test-time Adaptation for 3D Human Pose Estimation


### Images

![10808259f311783d7706f3f6f90196c3d3acfdc761ca5cf7dd9aab5e2336ca25.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/10808259f311783d7706f3f6f90196c3d3acfdc761ca5cf7dd9aab5e2336ca25.jpg)

![14bc724f63ee7c85ee1c0c3d5792991843a58a63b4f83b680d4500b72d5bcae9.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/14bc724f63ee7c85ee1c0c3d5792991843a58a63b4f83b680d4500b72d5bcae9.jpg)

![16b003abad2bea6f6004fe7427abe353e4958a85dabcb8fa0dbdb0c11dd45868.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/16b003abad2bea6f6004fe7427abe353e4958a85dabcb8fa0dbdb0c11dd45868.jpg)

![4a14237ece8140b71d9d87ab800f638d9008f09dd5636f72dec18606a9cc81b6.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/4a14237ece8140b71d9d87ab800f638d9008f09dd5636f72dec18606a9cc81b6.jpg)

![7174f30e10bc079b1b5575327d31f43c82073e2f10f18861ec8defd61c9a893d.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/7174f30e10bc079b1b5575327d31f43c82073e2f10f18861ec8defd61c9a893d.jpg)

![9e9465eb916b10c7cc8fc832d4975c14b11148d37ca0c517fbff48a17a58ece7.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/9e9465eb916b10c7cc8fc832d4975c14b11148d37ca0c517fbff48a17a58ece7.jpg)

![b7df9bda43910b5d3337e330c2af35099ffef0b625a543c7945fa494856f0e9d.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/b7df9bda43910b5d3337e330c2af35099ffef0b625a543c7945fa494856f0e9d.jpg)

![c9dd0de5a7052c64266a46ce5f580ddbf9b44fc6e758357ebc6eb15c5048bd85.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/c9dd0de5a7052c64266a46ce5f580ddbf9b44fc6e758357ebc6eb15c5048bd85.jpg)

![cccd7c0f44f852041c11dac72510a008584eba0c955632ab872a66210bb58f53.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/cccd7c0f44f852041c11dac72510a008584eba0c955632ab872a66210bb58f53.jpg)

![e63c503a28f07f3cf4d4c6d6647b08a8860dced387954c0550ce9f58b44d6f1e.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/e63c503a28f07f3cf4d4c6d6647b08a8860dced387954c0550ce9f58b44d6f1e.jpg)

![eececd3e91331fb105264499f1588bfdb3eb4dd443581721a63aa4404c2879b7.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/images/eececd3e91331fb105264499f1588bfdb3eb4dd443581721a63aa4404c2879b7.jpg)

### Tables

![0016bda8e6d4609b7ec7e973413e7c59cb0edbbed7674fc035932e097dd11dd7.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/0016bda8e6d4609b7ec7e973413e7c59cb0edbbed7674fc035932e097dd11dd7.jpg)

![02f5f4ce209e2aa995973f802cc740ca19dbedb6710cba1369aa6be51039e295.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/02f5f4ce209e2aa995973f802cc740ca19dbedb6710cba1369aa6be51039e295.jpg)

![4b4eb4de5d2ad5d0de453398ca35aaf82f7033f11394dc823969dd68eddc971c.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/4b4eb4de5d2ad5d0de453398ca35aaf82f7033f11394dc823969dd68eddc971c.jpg)

![4ffc09533bb77f2a105e647d6db27880bceee30ee3c50ef28fdc942d2c72712e.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/4ffc09533bb77f2a105e647d6db27880bceee30ee3c50ef28fdc942d2c72712e.jpg)

![51b6947585a61131f17834c8a73656bc8f07db5308c9f1c09ca3564d6b4ddd2c.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/51b6947585a61131f17834c8a73656bc8f07db5308c9f1c09ca3564d6b4ddd2c.jpg)

![69eb275e56bb822e1af45c80ff0c1c7d9263535dbe9c47de60124961d7eeec1e.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/69eb275e56bb822e1af45c80ff0c1c7d9263535dbe9c47de60124961d7eeec1e.jpg)

![bdd3bcf92f528cc6109c6ee5a229ae3bf566af42b18486247e6b32cedfca9956.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/bdd3bcf92f528cc6109c6ee5a229ae3bf566af42b18486247e6b32cedfca9956.jpg)

![e3170f5250054473e1441c61d9a8fefa0c75bad8093dce7a1a303edfcc3acbfe.jpg](../icml_results/829_Semantics-aware%20Test-time%20Adaptation%20for%203D%20Human%20Pose%20Estimation/tables/e3170f5250054473e1441c61d9a8fefa0c75bad8093dce7a1a303edfcc3acbfe.jpg)

## Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation


### Images

![286093c40037d56c05e149997503c7820caf733982a5ddda31dc70394869de08.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/286093c40037d56c05e149997503c7820caf733982a5ddda31dc70394869de08.jpg)

![35a87eaa383aaf97fb7e84666132def6d6a67cfbad5336aeaad84e3546174ac1.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/35a87eaa383aaf97fb7e84666132def6d6a67cfbad5336aeaad84e3546174ac1.jpg)

![3ac12b170ffbbc6c7e07c5e9067f49b9847502416fd804a4dc9b852cec7d0512.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/3ac12b170ffbbc6c7e07c5e9067f49b9847502416fd804a4dc9b852cec7d0512.jpg)

![42922bb39c52101ba07ce4e6a45fee5b067e1182f2973c288a2449a8b4d988b1.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/42922bb39c52101ba07ce4e6a45fee5b067e1182f2973c288a2449a8b4d988b1.jpg)

![79b3ed7908aaf199fb063fde9f761dcd157d56555dffa45eff51d751b74759e9.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/79b3ed7908aaf199fb063fde9f761dcd157d56555dffa45eff51d751b74759e9.jpg)

![835a14099fde4a3355cc20c27fd67813715c0a1a7123552a1129c6b559da0e4a.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/835a14099fde4a3355cc20c27fd67813715c0a1a7123552a1129c6b559da0e4a.jpg)

![b9b89b4319c0f960a4e929d084cde570887797fa40661d677db772ded3e22152.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/b9b89b4319c0f960a4e929d084cde570887797fa40661d677db772ded3e22152.jpg)

![bbf4961cbd7a0b313382d3412581f4a3b083c80062f5dd3aba7e88e59816a9c5.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/images/bbf4961cbd7a0b313382d3412581f4a3b083c80062f5dd3aba7e88e59816a9c5.jpg)

### Tables

![0a3cf20edb515907b026f0b9f906fc6260cea054e065504257d9d034f519e8de.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/0a3cf20edb515907b026f0b9f906fc6260cea054e065504257d9d034f519e8de.jpg)

![2fb7a5d9881e4e76b84bbf8bb184059c4b6ea9c317f34da54bccb5e0fb8d8646.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/2fb7a5d9881e4e76b84bbf8bb184059c4b6ea9c317f34da54bccb5e0fb8d8646.jpg)

![2ffa2fea83964fa037db062a42091406c60c5668034b2d5213546dc155e49954.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/2ffa2fea83964fa037db062a42091406c60c5668034b2d5213546dc155e49954.jpg)

![5a641601936dfa47ee8ff223aa90134352b261720a1c93b45c5c02f13afef0df.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/5a641601936dfa47ee8ff223aa90134352b261720a1c93b45c5c02f13afef0df.jpg)

![7fe6f94b4f55b3f3693900ba41b93073303bd9f758a2b7efa1ef72bc96a25500.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/7fe6f94b4f55b3f3693900ba41b93073303bd9f758a2b7efa1ef72bc96a25500.jpg)

![d8ca3302a1e1c4683f45d0eca65967f6fd4e93561c9559a764b7a048f5a8857d.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/d8ca3302a1e1c4683f45d0eca65967f6fd4e93561c9559a764b7a048f5a8857d.jpg)

![daf0cb9cbd855a3ebc5d2ed89e8870737d11e0c4e3fe7fd923ea16f84224b8b1.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/daf0cb9cbd855a3ebc5d2ed89e8870737d11e0c4e3fe7fd923ea16f84224b8b1.jpg)

![ec7048801d7452e35af94e341b9abd5fa42df03d4949679a87813c3ede228d91.jpg](../icml_results/830_Reasoning%20Through%20Execution_%20Unifying%20Process%20and%20Outcome%20Rewards%20for%20Code%20Generation/tables/ec7048801d7452e35af94e341b9abd5fa42df03d4949679a87813c3ede228d91.jpg)

## Robust Reward Alignment via Hypothesis Space Batch Cutting


### Images

![0d83bba55844f7a75e8a55f84fe386986a5d7c301dcf99b3f21c18d790c8647b.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/0d83bba55844f7a75e8a55f84fe386986a5d7c301dcf99b3f21c18d790c8647b.jpg)

![29fb21c86db15cb568e21516307eb536f940dcc763ac47f3c1085bdd6cac1011.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/29fb21c86db15cb568e21516307eb536f940dcc763ac47f3c1085bdd6cac1011.jpg)

![471cc7f20f7425d92f00adc59cbf73ed48a7d9e39aa7da7a3a0bff2c0b1ac033.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/471cc7f20f7425d92f00adc59cbf73ed48a7d9e39aa7da7a3a0bff2c0b1ac033.jpg)

![5645f9b27a05ba1f77d8d7546c27fe741e62f1af07bfbdc1ea357b7509534e0b.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/5645f9b27a05ba1f77d8d7546c27fe741e62f1af07bfbdc1ea357b7509534e0b.jpg)

![5aa52aea23769565336244ef4426ea9365944722e21f88ee64b1a91db1c00781.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/5aa52aea23769565336244ef4426ea9365944722e21f88ee64b1a91db1c00781.jpg)

![857938169da198345626bbaa4b1167984dace2b4e68072c8686e1de21de8370b.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/857938169da198345626bbaa4b1167984dace2b4e68072c8686e1de21de8370b.jpg)

![902a7053c8ce823ca571aa5b1e3703afc5991f4315fe441046160835e61716c5.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/902a7053c8ce823ca571aa5b1e3703afc5991f4315fe441046160835e61716c5.jpg)

![d993c9c7970b904822e4836292979a2dd28d03e6be18378a19a70212fe4250b1.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/d993c9c7970b904822e4836292979a2dd28d03e6be18378a19a70212fe4250b1.jpg)

![ecbf9e4e460916238674c0ba8638ae5f064b37c6f26de035c8b76e11e8ed49ac.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/images/ecbf9e4e460916238674c0ba8638ae5f064b37c6f26de035c8b76e11e8ed49ac.jpg)

### Tables

![0ada4ed9918a00b04d18cec002af76727d933ff6bf4d91fea03e91d64d80e2bf.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/tables/0ada4ed9918a00b04d18cec002af76727d933ff6bf4d91fea03e91d64d80e2bf.jpg)

![456b9b1f30cfa526822c3e55ce1544a66d599dd439356cee367781c144ed8941.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/tables/456b9b1f30cfa526822c3e55ce1544a66d599dd439356cee367781c144ed8941.jpg)

![89d6d2ce19a5eb1e3d7eeabf07aa4459875b533663736f3a9b07f7070f7eff06.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/tables/89d6d2ce19a5eb1e3d7eeabf07aa4459875b533663736f3a9b07f7070f7eff06.jpg)

![a2237493f94b3fca9ffebaab483ccb009d5dbe5f962ffa645297f767d3f1f9be.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/tables/a2237493f94b3fca9ffebaab483ccb009d5dbe5f962ffa645297f767d3f1f9be.jpg)

![d3b1153521ad351bbb53cc7b8b667a97bb107f51efa81e035d3a046db2fd5aec.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/tables/d3b1153521ad351bbb53cc7b8b667a97bb107f51efa81e035d3a046db2fd5aec.jpg)

![e824e496ff62a77351cda712ddd7e548b38e1fa29739c56ee8376c50baa37d8f.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/tables/e824e496ff62a77351cda712ddd7e548b38e1fa29739c56ee8376c50baa37d8f.jpg)

![f554bbc14af665fa89980eb03224505366998aa49040346a094ddc392bf0d415.jpg](../icml_results/831_Robust%20Reward%20Alignment%20via%20Hypothesis%20Space%20Batch%20Cutting/tables/f554bbc14af665fa89980eb03224505366998aa49040346a094ddc392bf0d415.jpg)

## Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction

### Images

![01d3af1f855d5977e4f4c3ea9b925ff1cae92870b4a477143467a790bb320a9c.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/images/01d3af1f855d5977e4f4c3ea9b925ff1cae92870b4a477143467a790bb320a9c.jpg)

![3e80935bb134b3626ff517ea1beb28c9d3404c6402f66f52e7dd5260d582c7aa.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/images/3e80935bb134b3626ff517ea1beb28c9d3404c6402f66f52e7dd5260d582c7aa.jpg)

![964e84fa0d5420ca46ce2e5ca57f8ba3cf4345e311c5c516bd3b060908e89a04.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/images/964e84fa0d5420ca46ce2e5ca57f8ba3cf4345e311c5c516bd3b060908e89a04.jpg)

![9ecd1c608e16d0e61c5be101c08800ccb20bcdf3545e50f0ea21709f781c0c7a.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/images/9ecd1c608e16d0e61c5be101c08800ccb20bcdf3545e50f0ea21709f781c0c7a.jpg)

![b4de44470ead048da36eae330d807505d1652af1dfdb85b0472db793adb89335.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/images/b4de44470ead048da36eae330d807505d1652af1dfdb85b0472db793adb89335.jpg)

![c4a4d238dc91f2d6379f0a0c5b5d9d875cc9a15c6bf339793716da5e4fa308d1.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/images/c4a4d238dc91f2d6379f0a0c5b5d9d875cc9a15c6bf339793716da5e4fa308d1.jpg)

![dea19caa851a1b62515d929f564f7e7dffaa74b7910eba67a439e1f8dfb4ce20.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/images/dea19caa851a1b62515d929f564f7e7dffaa74b7910eba67a439e1f8dfb4ce20.jpg)

### Tables

![1fde34cacfb85423c323db8cbec8e1f48de42a0965da7f8cfe928ea8d92e767e.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/tables/1fde34cacfb85423c323db8cbec8e1f48de42a0965da7f8cfe928ea8d92e767e.jpg)

![345efb66e728d39c5d73782583d82fcce42164aa60233fe303fe5fc2eca0a824.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/tables/345efb66e728d39c5d73782583d82fcce42164aa60233fe303fe5fc2eca0a824.jpg)

![708824c2a52f9a439ab3598deaad5f896d0530b713579787163b8fcce9b9edf2.jpg](../icml_results/832_Sample%20Complexity%20of%20Distributionally%20Robust%20Off-Dynamics%20Reinforcement%20Learning%20with%20Online%20Interac/tables/708824c2a52f9a439ab3598deaad5f896d0530b713579787163b8fcce9b9edf2.jpg)
