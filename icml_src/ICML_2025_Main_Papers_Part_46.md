# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 46 of 100

## 目录 (Table of Contents)

1. [Ladder-Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference with Communication Overlapping](#Ladder-Residual-Parallelism-Aware-Architecture-for-Accelerating-Large-Model-Inference-with-Communication-Overlapping)
2. [The Price of Linear Time: Error Analysis of Structured Kernel Interpolation](#The-Price-of-Linear-Time-Error-Analysis-of-Structured-Kernel-Interpolation)
3. [Generalizing Causal Effects from Randomized Controlled Trials to Target Populations across Diverse Environments](#Generalizing-Causal-Effects-from-Randomized-Controlled-Trials-to-Target-Populations-across-Diverse-Environments)
4. [Multi-Session Budget Optimization for Forward Auction-based Federated Learning](#Multi-Session-Budget-Optimization-for-Forward-Auction-based-Federated-Learning)
5. [Language Models as Implicit Tree Search](#Language-Models-as-Implicit-Tree-Search)
6. [Automated Red Teaming with GOAT: the Generative Offensive Agent Tester](#Automated-Red-Teaming-with-GOAT-the-Generative-Offensive-Agent-Tester)
7. [L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning](#L3A-Label-Augmented-Analytic-Adaptation-for-Multi-Label-Class-Incremental-Learning)
8. [Dynamic Sparse Training of Diagonally Sparse Networks](#Dynamic-Sparse-Training-of-Diagonally-Sparse-Networks)
9. [AssistanceZero: Scalably Solving Assistance Games](#AssistanceZero-Scalably-Solving-Assistance-Games)
10. [Learning Survival Distributions with the Asymmetric Laplace Distribution](#Learning-Survival-Distributions-with-the-Asymmetric-Laplace-Distribution)
11. [The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training](#The-Surprising-Agreement-Between-Convex-Optimization-Theory-and-Learning-Rate-Scheduling-for-Large-Model-Training)
12. [DragLoRA: Online Optimization of LoRA Adapters for Drag-based Image Editing in Diffusion Model](#DragLoRA-Online-Optimization-of-LoRA-Adapters-for-Drag-based-Image-Editing-in-Diffusion-Model)
13. [Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule](#Piloting-Structure-Based-Drug-Design-via-Modality-Specific-Optimal-Schedule)
14. [Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting](#Time-VLM-Exploring-Multimodal-Vision-Language-Models-for-Augmented-Time-Series-Forecasting)
15. [Efficiently Access Diffusion Fisher: Within the Outer Product Span Space](#Efficiently-Access-Diffusion-Fisher-Within-the-Outer-Product-Span-Space)
16. [IRBridge: Solving Image Restoration Bridge with Pre-trained Generative Diffusion Models](#IRBridge-Solving-Image-Restoration-Bridge-with-Pre-trained-Generative-Diffusion-Models)
17. [Splitting & Integrating: Out-of-Distribution Detection via Adversarial Gradient Attribution](#Splitting-Integrating-Out-of-Distribution-Detection-via-Adversarial-Gradient-Attribution)
18. [The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](#The-Devil-Is-in-the-Details-Tackling-Unimodal-Spurious-Correlations-for-Generalizable-Multimodal-Reward-Models)
19. [Otter: Generating Tests from Issues to Validate SWE Patches](#Otter-Generating-Tests-from-Issues-to-Validate-SWE-Patches)
20. [Geometric Resampling in Nearly Linear Time for Follow-the-Perturbed-Leader with Best-of-Both-Worlds Guarantee in Bandit Problems](#Geometric-Resampling-in-Nearly-Linear-Time-for-Follow-the-Perturbed-Leader-with-Best-of-Both-Worlds-Guarantee-in-Bandit-Problems)
21. [Compositional Risk Minimization](#Compositional-Risk-Minimization)
22. [Continuous Visual Autoregressive Generation via Score Maximization](#Continuous-Visual-Autoregressive-Generation-via-Score-Maximization)
23. [Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](#Learning-Augmented-Algorithms-for-MTS-with-Bandit-Access-to-Multiple-Predictors)
24. [On Zero-Initialized Attention: Optimal Prompt and Gating Factor Estimation](#On-Zero-Initialized-Attention-Optimal-Prompt-and-Gating-Factor-Estimation)
25. [The Surprising Effectiveness of Test-Time Training for Few-Shot Learning](#The-Surprising-Effectiveness-of-Test-Time-Training-for-Few-Shot-Learning)
26. [(How) Can Transformers Predict Pseudo-Random Numbers?](#How-Can-Transformers-Predict-Pseudo-Random-Numbers)
27. [Time to Spike? Understanding the Representational Power of Spiking Neural Networks in Discrete Time](#Time-to-Spike-Understanding-the-Representational-Power-of-Spiking-Neural-Networks-in-Discrete-Time)
28. [Discriminative Policy Optimization for Token-Level Reward Models](#Discriminative-Policy-Optimization-for-Token-Level-Reward-Models)
29. [The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](#The-Panaceas-for-Improving-Low-Rank-Decomposition-in-Communication-Efficient-Federated-Learning)
30. [BCE vs. CE in Deep Feature Learning](#BCE-vs-CE-in-Deep-Feature-Learning)
31. [SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation](#SAM2Act-Integrating-Visual-Foundation-Model-with-A-Memory-Architecture-for-Robotic-Manipulation)
32. [Programming Every Example: Lifting Pre-training Data Quality Like Experts at Scale](#Programming-Every-Example-Lifting-Pre-training-Data-Quality-Like-Experts-at-Scale)
33. [ROME is Forged in Adversity: Robust Distilled Datasets via Information Bottleneck](#ROME-is-Forged-in-Adversity-Robust-Distilled-Datasets-via-Information-Bottleneck)

---


## Ladder-Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference with Communication Overlapping

### Images

![20734259f7a3cd2790a43777dbfdd4a5385ef2affc9e2a39a91bbe47161f5c24.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/20734259f7a3cd2790a43777dbfdd4a5385ef2affc9e2a39a91bbe47161f5c24.jpg)

![25e8885165be37ae6693837c8ee8925a9e280c8136988301637c69c6460608eb.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/25e8885165be37ae6693837c8ee8925a9e280c8136988301637c69c6460608eb.jpg)

![3b5de74b6ffe5e5645d89a3aff616ee8a5aaa432337689d3bad4627c5d19ce8b.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/3b5de74b6ffe5e5645d89a3aff616ee8a5aaa432337689d3bad4627c5d19ce8b.jpg)

![629f4369e311d2a227ee691ff94499d7440b77738089043dcad1a403cfb6f250.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/629f4369e311d2a227ee691ff94499d7440b77738089043dcad1a403cfb6f250.jpg)

![a60634a3d6276707f44deb4adc4342ea1aa6459afce85c9b8764cb15e8f061a0.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/a60634a3d6276707f44deb4adc4342ea1aa6459afce85c9b8764cb15e8f061a0.jpg)

![d9266ff88e6cd5841c659d4b9d8ad51604c688988579515f174c460695017ee6.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/d9266ff88e6cd5841c659d4b9d8ad51604c688988579515f174c460695017ee6.jpg)

![e1f1b2d5a6e9baa8df5e99ba384686d3b00fb58a95b3a9bc943fa1f7211c27ae.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/e1f1b2d5a6e9baa8df5e99ba384686d3b00fb58a95b3a9bc943fa1f7211c27ae.jpg)

![e8ebab11af5aee17612bb7f41560ef1c36302de1a2659994cac6723d60dd0b8e.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/images/e8ebab11af5aee17612bb7f41560ef1c36302de1a2659994cac6723d60dd0b8e.jpg)

### Tables

![161428944674d63edd9b358f2cef73eb62edd532fa75a7a42d19ee73065bcc1a.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/161428944674d63edd9b358f2cef73eb62edd532fa75a7a42d19ee73065bcc1a.jpg)

![57e672faaf3d135b0bd34e7c43dbbebc2d4daa82724bfc38c28f8a59d0f90586.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/57e672faaf3d135b0bd34e7c43dbbebc2d4daa82724bfc38c28f8a59d0f90586.jpg)

![6bcf230cf5a634aedd80f820396d30f17e7f371a36af9c0114681e2fcaaff74b.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/6bcf230cf5a634aedd80f820396d30f17e7f371a36af9c0114681e2fcaaff74b.jpg)

![9f8d963ee44441c56c2dfcda4a9f7ba75dba00e5af5ca5d2e03f6fd1afaddb3a.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/9f8d963ee44441c56c2dfcda4a9f7ba75dba00e5af5ca5d2e03f6fd1afaddb3a.jpg)

![aae4d251a4310613cc565fb5a0a4902618f50216ee569aeb21e85d9d6f3aa0f6.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/aae4d251a4310613cc565fb5a0a4902618f50216ee569aeb21e85d9d6f3aa0f6.jpg)

![ab7e079ecca7b1027d178b629db69954165fe7d30094f2d86fa0df0508a49084.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/ab7e079ecca7b1027d178b629db69954165fe7d30094f2d86fa0df0508a49084.jpg)

![bc4e91d6aca0ab1acaab2171e7552d46d284a6f13f8e05ef5cf57302ef837eb2.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/bc4e91d6aca0ab1acaab2171e7552d46d284a6f13f8e05ef5cf57302ef837eb2.jpg)

![bc7544906782a10781cd984bfe974d5c22abc42b632e74b194b4621affef1d9a.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/bc7544906782a10781cd984bfe974d5c22abc42b632e74b194b4621affef1d9a.jpg)

![e10ab7888af302763012364e2776401222e9c7ca3b19a310a4cba7ae5c9a6437.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/e10ab7888af302763012364e2776401222e9c7ca3b19a310a4cba7ae5c9a6437.jpg)

![f095c93a0b5645ec176917d40f434c0a265ac45dd3dcfa75456d2c2e399223ca.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/f095c93a0b5645ec176917d40f434c0a265ac45dd3dcfa75456d2c2e399223ca.jpg)

![f62b94fd64307d27652b8a66c218628db5357ef4bacac86282f3d5e3b22dbbbe.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/f62b94fd64307d27652b8a66c218628db5357ef4bacac86282f3d5e3b22dbbbe.jpg)

![f83602e61002c18364ab883bd8b6fa56e84131d5162fcc957b3582e9e2859233.jpg](../icml_results/1500_Beyond%20Message%20Passing_%20Neural%20Graph%20Pattern%20Machine/tables/f83602e61002c18364ab883bd8b6fa56e84131d5162fcc957b3582e9e2859233.jpg)

## Ladder-Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference with Communication Overlapping


### Images

![3124f47131659182cee7e090f7ef5082ba512481d4c82462dc4c046e70160c8e.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/images/3124f47131659182cee7e090f7ef5082ba512481d4c82462dc4c046e70160c8e.jpg)

![3e0195edea67571243ae33aae59bc1cabd8f8c6c67c8c41870e5334ef3ba45f2.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/images/3e0195edea67571243ae33aae59bc1cabd8f8c6c67c8c41870e5334ef3ba45f2.jpg)

![6ca94a9833f4414f161ef807fe8ca314b8bb6a5060149f72d183e3c82ab844ea.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/images/6ca94a9833f4414f161ef807fe8ca314b8bb6a5060149f72d183e3c82ab844ea.jpg)

![c61c8ac0291bfaedca863a85f0f7f168c1c2eaa05bd56e8b7b38fa93fcd9a7e8.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/images/c61c8ac0291bfaedca863a85f0f7f168c1c2eaa05bd56e8b7b38fa93fcd9a7e8.jpg)

![cf23663d241ada0d63547c67eceaf45be71ca9466202c7a7c48e5d0f91d98fd1.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/images/cf23663d241ada0d63547c67eceaf45be71ca9466202c7a7c48e5d0f91d98fd1.jpg)

![d1a10a5396e61bbdd33793212bcacadf5a8a734074903779bdd46fd911abdf74.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/images/d1a10a5396e61bbdd33793212bcacadf5a8a734074903779bdd46fd911abdf74.jpg)

### Tables

![284fc8a59fc83d619b5765eb4571a34cafaf8c25ed76dac0ccc6d3879dd98bf5.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/tables/284fc8a59fc83d619b5765eb4571a34cafaf8c25ed76dac0ccc6d3879dd98bf5.jpg)

![5f99581c57eee58cebb25594b1bc4be62a40d7af601431505f00ba8e7b8c07c4.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/tables/5f99581c57eee58cebb25594b1bc4be62a40d7af601431505f00ba8e7b8c07c4.jpg)

![a77dabaa97b16c404c94928fda8faa94d7b297e17c58a6cd721ee5ed14281b87.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/tables/a77dabaa97b16c404c94928fda8faa94d7b297e17c58a6cd721ee5ed14281b87.jpg)

![c2be7f669e6b6cb76e595632188f1d282364a887e5500769ac39afab9e64b313.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/tables/c2be7f669e6b6cb76e595632188f1d282364a887e5500769ac39afab9e64b313.jpg)

![f88d7b31b1786e63bef6d344171ff5693b2db3aba3b3e6b94067ed2b3f2efe5a.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/tables/f88d7b31b1786e63bef6d344171ff5693b2db3aba3b3e6b94067ed2b3f2efe5a.jpg)

![ff552f5b19cc017dcc3191ef416075c3846f685f02a759c45e6e2328e1adc10d.jpg](../icml_results/1501_Ladder-Residual_%20Parallelism-Aware%20Architecture%20for%20Accelerating%20Large%20Model%20Inference%20with%20Communic/tables/ff552f5b19cc017dcc3191ef416075c3846f685f02a759c45e6e2328e1adc10d.jpg)

## The Price of Linear Time: Error Analysis of Structured Kernel Interpolation


### Images

![d5442edaf0969a3903b6e13eaad151e3a48140ba8111cc36d428d6a33e5086d8.jpg](../icml_results/1502_The%20Price%20of%20Linear%20Time_%20Error%20Analysis%20of%20Structured%20Kernel%20Interpolation/images/d5442edaf0969a3903b6e13eaad151e3a48140ba8111cc36d428d6a33e5086d8.jpg)

### Tables

![95b8962c2eaadc5f3993fa771b92d25d32084377d6433154c5beaa8647f8245f.jpg](../icml_results/1502_The%20Price%20of%20Linear%20Time_%20Error%20Analysis%20of%20Structured%20Kernel%20Interpolation/tables/95b8962c2eaadc5f3993fa771b92d25d32084377d6433154c5beaa8647f8245f.jpg)

## Generalizing Causal Effects from Randomized Controlled Trials to Target Populations across Diverse Environments


### Images

![0cc362d2ba4e6a06f5c50bd38289c49815c1a52f51c370148bed92c72785ebed.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/images/0cc362d2ba4e6a06f5c50bd38289c49815c1a52f51c370148bed92c72785ebed.jpg)

![2ba723ac0a3f5fd247d52cd23c583cd3430af6f6f0429057da1893e0905d18d4.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/images/2ba723ac0a3f5fd247d52cd23c583cd3430af6f6f0429057da1893e0905d18d4.jpg)

![4d693143c9b6aeb39d029de4506b38a6e83c67ac2a37c9e32f442266ae7062ba.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/images/4d693143c9b6aeb39d029de4506b38a6e83c67ac2a37c9e32f442266ae7062ba.jpg)

![586ccd12e4778d4a6ec8ff05fdd02e2e91b8558cbbddad93c14f3f95f8b2c0c9.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/images/586ccd12e4778d4a6ec8ff05fdd02e2e91b8558cbbddad93c14f3f95f8b2c0c9.jpg)

![69578060518460048f688ab6f922c1749cb62cdbc85ed1efe7f3a3fb3c466c82.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/images/69578060518460048f688ab6f922c1749cb62cdbc85ed1efe7f3a3fb3c466c82.jpg)

![a30d114bbcdca27f46abfe8dcad6ad3dfed06d92feae011841f6d39cf8317118.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/images/a30d114bbcdca27f46abfe8dcad6ad3dfed06d92feae011841f6d39cf8317118.jpg)

![c2d7d383115be940b5106c0c9746882e094cb55f387d63a4a3ee6e1d7e1a8604.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/images/c2d7d383115be940b5106c0c9746882e094cb55f387d63a4a3ee6e1d7e1a8604.jpg)

### Tables

![1a340b98ba05ffafd775c7c4c22846b573097120ce6a780c9c07c49dca332abb.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/tables/1a340b98ba05ffafd775c7c4c22846b573097120ce6a780c9c07c49dca332abb.jpg)

![26280c981d9d7e9a9689adbdaddfbbd9aee083bd5ed18a7a201c10ba9417a3d9.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/tables/26280c981d9d7e9a9689adbdaddfbbd9aee083bd5ed18a7a201c10ba9417a3d9.jpg)

![5cd21b8abc86a7611e2f224c41125fc32e808d6f1ba1ea4c7c935f333b1b1b63.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/tables/5cd21b8abc86a7611e2f224c41125fc32e808d6f1ba1ea4c7c935f333b1b1b63.jpg)

![925b5baeb9ae32b235fd6f73d5f839ec264949b8f48cae00eb9fbc37d08dcff3.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/tables/925b5baeb9ae32b235fd6f73d5f839ec264949b8f48cae00eb9fbc37d08dcff3.jpg)

![bc0a38e09ffe97cfbdc27f5b7e2d86df6cf4ce3b71741a7abec439f816a67d2c.jpg](../icml_results/1503_Generalizing%20Causal%20Effects%20from%20Randomized%20Controlled%20Trials%20to%20Target%20Populations%20across%20Diverse%20E/tables/bc0a38e09ffe97cfbdc27f5b7e2d86df6cf4ce3b71741a7abec439f816a67d2c.jpg)

## Multi-Session Budget Optimization for Forward Auction-based Federated Learning


### Images

![6c8a602376baa436527a827ec5d2d3cabebaa9e86d99a28528b5d152355e8f49.jpg](../icml_results/1504_Multi-Session%20Budget%20Optimization%20for%20Forward%20Auction-based%20Federated%20Learning/images/6c8a602376baa436527a827ec5d2d3cabebaa9e86d99a28528b5d152355e8f49.jpg)

![a59241e2aebab1d6a8b0cae5564d066ae06650bc93eb165237bcdb2589aeb65a.jpg](../icml_results/1504_Multi-Session%20Budget%20Optimization%20for%20Forward%20Auction-based%20Federated%20Learning/images/a59241e2aebab1d6a8b0cae5564d066ae06650bc93eb165237bcdb2589aeb65a.jpg)

![e34ceb6ce23be4dab994104e08543b093daa1a71528644773b31d891296969c7.jpg](../icml_results/1504_Multi-Session%20Budget%20Optimization%20for%20Forward%20Auction-based%20Federated%20Learning/images/e34ceb6ce23be4dab994104e08543b093daa1a71528644773b31d891296969c7.jpg)

### Tables

![4f8100dd7b5377122ccbbd6df022a0aaf37221565e8bf1119f3d574bd3f5c563.jpg](../icml_results/1504_Multi-Session%20Budget%20Optimization%20for%20Forward%20Auction-based%20Federated%20Learning/tables/4f8100dd7b5377122ccbbd6df022a0aaf37221565e8bf1119f3d574bd3f5c563.jpg)

![adc98634c79d58311c58f3bfa78bd661d16da23db0591346e86e35d07637088b.jpg](../icml_results/1504_Multi-Session%20Budget%20Optimization%20for%20Forward%20Auction-based%20Federated%20Learning/tables/adc98634c79d58311c58f3bfa78bd661d16da23db0591346e86e35d07637088b.jpg)

## Language Models as Implicit Tree Search


### Images

![2675ed9a00c2ae15e8f7095d6abcbfb1f59de4fc847fa432432cc8a825f2859a.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/images/2675ed9a00c2ae15e8f7095d6abcbfb1f59de4fc847fa432432cc8a825f2859a.jpg)

![fa7f18363aaac89b5b1296afa25c00cfa02da873244c24d2458be9dd4227e9eb.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/images/fa7f18363aaac89b5b1296afa25c00cfa02da873244c24d2458be9dd4227e9eb.jpg)

### Tables

![45720f381e2b2869f7c3127116c6bb37c8184a7fbf5ffcbd235ffbc9add98382.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/tables/45720f381e2b2869f7c3127116c6bb37c8184a7fbf5ffcbd235ffbc9add98382.jpg)

![4630351e3e8a18517ed98caf81c1067283510806a502b9870651fdbc56c4f9c2.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/tables/4630351e3e8a18517ed98caf81c1067283510806a502b9870651fdbc56c4f9c2.jpg)

![57a8ff5242c802fdf60da8ffb61e279861c7758c662b1e24ca6960585213f4d0.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/tables/57a8ff5242c802fdf60da8ffb61e279861c7758c662b1e24ca6960585213f4d0.jpg)

![5bbcd7c089243e35453f478c5aace7734311334f950f74b3b63b662ee89fbf09.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/tables/5bbcd7c089243e35453f478c5aace7734311334f950f74b3b63b662ee89fbf09.jpg)

![7d3b62d3e5930731bd27a81d7545a7abc4ca236a483b62c7dd8a9001ef1c619d.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/tables/7d3b62d3e5930731bd27a81d7545a7abc4ca236a483b62c7dd8a9001ef1c619d.jpg)

![fb62c9da7bba27503c0ae95f36b1850a48b30ad80e044c94028067aed58c67da.jpg](../icml_results/1505_Language%20Models%20as%20Implicit%20Tree%20Search/tables/fb62c9da7bba27503c0ae95f36b1850a48b30ad80e044c94028067aed58c67da.jpg)

## Automated Red Teaming with GOAT: the Generative Offensive Agent Tester


### Images

![0c7fc432b6ae23779930ca74fe9e09e3efd8d5431047155eb0d4086d1d9c8789.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/images/0c7fc432b6ae23779930ca74fe9e09e3efd8d5431047155eb0d4086d1d9c8789.jpg)

![1b3be75019bcb8a3bedf88c08eaeb577e665e0d580dcdad86882f1e9a9058d05.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/images/1b3be75019bcb8a3bedf88c08eaeb577e665e0d580dcdad86882f1e9a9058d05.jpg)

![4afae6483018b5ea5bb7190182f7d455dcecd07ba63ee43529018f970c95f933.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/images/4afae6483018b5ea5bb7190182f7d455dcecd07ba63ee43529018f970c95f933.jpg)

![858a25e4328903b862ebff295d28b1aa24f939b7fba8af2e2332bf3f91f43cee.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/images/858a25e4328903b862ebff295d28b1aa24f939b7fba8af2e2332bf3f91f43cee.jpg)

![b4aa22bda03e1b01a0868a4e015389accaa712248c48a4ae9003d94e327ef83e.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/images/b4aa22bda03e1b01a0868a4e015389accaa712248c48a4ae9003d94e327ef83e.jpg)

![d9f42e18e9b1236b6430737d01b3d87983156c22c009e6693a43e4c76b0a12f6.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/images/d9f42e18e9b1236b6430737d01b3d87983156c22c009e6693a43e4c76b0a12f6.jpg)

### Tables

![0ac493808dc6430d1854d77be1fdecea1d143f10567f1523a9e484b8a6b5848c.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/tables/0ac493808dc6430d1854d77be1fdecea1d143f10567f1523a9e484b8a6b5848c.jpg)

![344e5420ee952bcb13a17db5bcc065fc69d11d3514453ec834e6e6050380942d.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/tables/344e5420ee952bcb13a17db5bcc065fc69d11d3514453ec834e6e6050380942d.jpg)

![56e5144ba11f5be4abfb29f88d92c27c8df778cb30ed56f80a8b35876daa90fe.jpg](../icml_results/1506_Automated%20Red%20Teaming%20with%20GOAT_%20the%20Generative%20Offensive%20Agent%20Tester/tables/56e5144ba11f5be4abfb29f88d92c27c8df778cb30ed56f80a8b35876daa90fe.jpg)

## L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning


### Images

![a9fffb38cd1d5602e17a8497844133b67cc84487af7d92daf4ec5228477590bf.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/images/a9fffb38cd1d5602e17a8497844133b67cc84487af7d92daf4ec5228477590bf.jpg)

![e1849f211c13f7c5e49b44920984b394e74a9995ef088da60bebd3757bf3b684.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/images/e1849f211c13f7c5e49b44920984b394e74a9995ef088da60bebd3757bf3b684.jpg)

![f07d47ba4890b72a2e95624dad6cecee34213d65c177e8d0369ee771d57bce78.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/images/f07d47ba4890b72a2e95624dad6cecee34213d65c177e8d0369ee771d57bce78.jpg)

### Tables

![32acc1f30d7ee2629917b56bdf06e90996351a869287cd2d6fd320c2b42dabeb.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/32acc1f30d7ee2629917b56bdf06e90996351a869287cd2d6fd320c2b42dabeb.jpg)

![4b40efab13d7b50a5ab166c685e0f05a1a666e347640a738d13d7d36e1fc76f7.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/4b40efab13d7b50a5ab166c685e0f05a1a666e347640a738d13d7d36e1fc76f7.jpg)

![547dd22f56a4b885e8faf7fc1e1950a40e6f5f1c2611fe8b9b82b50915088b57.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/547dd22f56a4b885e8faf7fc1e1950a40e6f5f1c2611fe8b9b82b50915088b57.jpg)

![78c17e1f7f816bb204af4dfb9cded21cf308a6213909a41510ab2f653998ec96.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/78c17e1f7f816bb204af4dfb9cded21cf308a6213909a41510ab2f653998ec96.jpg)

![7e00ea4aa708746e5d85e47ae8bc189bd3dfe9e5ef392c55702638ebc1a30edf.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/7e00ea4aa708746e5d85e47ae8bc189bd3dfe9e5ef392c55702638ebc1a30edf.jpg)

![cb8b846eddc85d9859719c6898b2fcc9c9ffefbab79430fa5502009c6058b9c6.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/cb8b846eddc85d9859719c6898b2fcc9c9ffefbab79430fa5502009c6058b9c6.jpg)

![cde450d6cd673e38d899f460e8f567e53780e9aab08483238f29b70ce43bfe92.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/cde450d6cd673e38d899f460e8f567e53780e9aab08483238f29b70ce43bfe92.jpg)

![ed36ef30382b4a97403442112461a3b4d306777e423b6c796f5832ea05bd486b.jpg](../icml_results/1507_L3A_%20Label-Augmented%20Analytic%20Adaptation%20for%20Multi-Label%20Class%20Incremental%20Learning/tables/ed36ef30382b4a97403442112461a3b4d306777e423b6c796f5832ea05bd486b.jpg)

## Dynamic Sparse Training of Diagonally Sparse Networks


### Images

![387c860a51268764c45e5bbf31d8e213c2854155c50c189d36d5c9adaed7c1cc.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/387c860a51268764c45e5bbf31d8e213c2854155c50c189d36d5c9adaed7c1cc.jpg)

![3a9816246aeaf65afc54be128454822868ea410329d968e69d9679d92393c738.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/3a9816246aeaf65afc54be128454822868ea410329d968e69d9679d92393c738.jpg)

![75bf8ff96e40cb2f91d1e156a3bd97fc394be72385b9b967295ceca3ba4fd1a4.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/75bf8ff96e40cb2f91d1e156a3bd97fc394be72385b9b967295ceca3ba4fd1a4.jpg)

![7892fffcd9b2ce5fb7e5cc45c03e3914ea0a8d7b4d989cc480651029eef279c2.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/7892fffcd9b2ce5fb7e5cc45c03e3914ea0a8d7b4d989cc480651029eef279c2.jpg)

![7e3844a181d2df4dc885f4452927e3d93bdc50ba2bbd53054b92d6ef3dae62ea.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/7e3844a181d2df4dc885f4452927e3d93bdc50ba2bbd53054b92d6ef3dae62ea.jpg)

![bcd0fd53fc6a7e3a20065eb236d3ca841489d9097149b9cf092e11708d284699.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/bcd0fd53fc6a7e3a20065eb236d3ca841489d9097149b9cf092e11708d284699.jpg)

![bdd77dca645940b5ccec1a749b2eb6b8551b71c138aed6bbfeb02de73bb3cf80.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/bdd77dca645940b5ccec1a749b2eb6b8551b71c138aed6bbfeb02de73bb3cf80.jpg)

![e14b9bbe3d597121ea064885b3064c417fab637d06ddb4e69ae9e54e01046612.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/e14b9bbe3d597121ea064885b3064c417fab637d06ddb4e69ae9e54e01046612.jpg)

![f9a87dbb7c08cc17a54c992d6f7c72a4c00df75e45acba224ac53ed6b4e96618.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/images/f9a87dbb7c08cc17a54c992d6f7c72a4c00df75e45acba224ac53ed6b4e96618.jpg)

### Tables

![07516f349a4118c6e757f2747aa43f7f126f9c7717ffd6210c33042e9c7efb59.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/07516f349a4118c6e757f2747aa43f7f126f9c7717ffd6210c33042e9c7efb59.jpg)

![26e83ebb8aed131666894b25a5e60046c32330a2e56bd220eec090ce59aa7ada.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/26e83ebb8aed131666894b25a5e60046c32330a2e56bd220eec090ce59aa7ada.jpg)

![26eea72ac40577ea6bd322bacc7a5f35b118e439630c08729bf25c7c46cdf20a.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/26eea72ac40577ea6bd322bacc7a5f35b118e439630c08729bf25c7c46cdf20a.jpg)

![2eb84b4d11bae69db11ba9e0415648c67f51955bb65f98cf97e3be6d18bd7907.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/2eb84b4d11bae69db11ba9e0415648c67f51955bb65f98cf97e3be6d18bd7907.jpg)

![3391b2a59d3829b65c5b062bfd7b5c8edf78b5c02af2a64433db6e6c53a86aec.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/3391b2a59d3829b65c5b062bfd7b5c8edf78b5c02af2a64433db6e6c53a86aec.jpg)

![578b3ca15b6631fe7d202e9e712883b52653bb8aaa998f3cff808ba5570f2ac4.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/578b3ca15b6631fe7d202e9e712883b52653bb8aaa998f3cff808ba5570f2ac4.jpg)

![6d7350a22675195b691270a23839e8c5ec71405b3d1cf87f4a932d0103c8b7d2.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/6d7350a22675195b691270a23839e8c5ec71405b3d1cf87f4a932d0103c8b7d2.jpg)

![751502cd9eab489e237c61fdf22ad6592ea20a9eb374b193b56fa375810a1306.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/751502cd9eab489e237c61fdf22ad6592ea20a9eb374b193b56fa375810a1306.jpg)

![8d1b6ebfb2a97dc55abd00d10bbbd099d08804ed59e1bd75e848c8b3bd63fa24.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/8d1b6ebfb2a97dc55abd00d10bbbd099d08804ed59e1bd75e848c8b3bd63fa24.jpg)

![a193951f705236acf27a1306d401bcbf29eedaabfb23a55a774e9ef345132f45.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/a193951f705236acf27a1306d401bcbf29eedaabfb23a55a774e9ef345132f45.jpg)

![aefb7ef2e51c1b3fdc02a73ed845b483aa950776ba6f0eea824917e2dae79f7f.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/aefb7ef2e51c1b3fdc02a73ed845b483aa950776ba6f0eea824917e2dae79f7f.jpg)

![af6dadb812a735465cb56352c12b01b2c9816cdd16b5afea3ab5df7e86980fc4.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/af6dadb812a735465cb56352c12b01b2c9816cdd16b5afea3ab5df7e86980fc4.jpg)

![c0f20587b99c0e28d820f93742d904475121b917ebedeb42a6d8d3c4ff1224a8.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/c0f20587b99c0e28d820f93742d904475121b917ebedeb42a6d8d3c4ff1224a8.jpg)

![c66d099a1f2cda12713fe6935050f284172510af6053a92894999b6cb68e809f.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/c66d099a1f2cda12713fe6935050f284172510af6053a92894999b6cb68e809f.jpg)

![d356f14672dec659b1677bcfd6a54af3fe236a1a77205006e36233fe0b64175e.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/d356f14672dec659b1677bcfd6a54af3fe236a1a77205006e36233fe0b64175e.jpg)

![e32a17e6b3589365cda4c2ac986e994c0be75bb1f66175b832213c6416bf515c.jpg](../icml_results/1508_Dynamic%20Sparse%20Training%20of%20Diagonally%20Sparse%20Networks/tables/e32a17e6b3589365cda4c2ac986e994c0be75bb1f66175b832213c6416bf515c.jpg)

## AssistanceZero: Scalably Solving Assistance Games


### Images

![03f8f099315c0f08a4a5ca1401e48c549883087ab0a404a3bd6a29726717cd8b.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/03f8f099315c0f08a4a5ca1401e48c549883087ab0a404a3bd6a29726717cd8b.jpg)

![06231a60a3bff580a25d475b372f16ebf484f71c86e83e5f944bf2f36090e92f.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/06231a60a3bff580a25d475b372f16ebf484f71c86e83e5f944bf2f36090e92f.jpg)

![443f00ad6e122e80a2569acad8ad30a49cb894b83f9b9694a72e40dc9a33a03d.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/443f00ad6e122e80a2569acad8ad30a49cb894b83f9b9694a72e40dc9a33a03d.jpg)

![52f3352b66c741c3a2671e77afca3d86900d609c7efee30591683a368c64442e.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/52f3352b66c741c3a2671e77afca3d86900d609c7efee30591683a368c64442e.jpg)

![5ad4dc0a4a9047c8511e0dab52a66e3bc1ad417f204eb8cb813a2fd3732781d2.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/5ad4dc0a4a9047c8511e0dab52a66e3bc1ad417f204eb8cb813a2fd3732781d2.jpg)

![5eca742ffb243dc5fbc7edc96be5906ef659209b386a2614144b74c1b6bb1f15.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/5eca742ffb243dc5fbc7edc96be5906ef659209b386a2614144b74c1b6bb1f15.jpg)

![667a6d6b6c9f0a38a176a5995bf70c5a8b88c08cd116ffb84363137972a40801.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/667a6d6b6c9f0a38a176a5995bf70c5a8b88c08cd116ffb84363137972a40801.jpg)

![a96c5e69ea17bb567de7aa204df635e32ab83c86f5aa24dfd186fef7d28287cf.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/a96c5e69ea17bb567de7aa204df635e32ab83c86f5aa24dfd186fef7d28287cf.jpg)

![adb510f69a8f5dbf0f8041e913642c17c0920ac57c8708a2ec332013ce20dbea.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/adb510f69a8f5dbf0f8041e913642c17c0920ac57c8708a2ec332013ce20dbea.jpg)

![bd8a07e01ed419325562e1f6c6a6dd19e0bee783474aba4b1d3434037f1226fd.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/bd8a07e01ed419325562e1f6c6a6dd19e0bee783474aba4b1d3434037f1226fd.jpg)

![c36e904471051d1d5cbe316f66b2d12a43e964fcbeef91da9e8042c6e334049b.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/c36e904471051d1d5cbe316f66b2d12a43e964fcbeef91da9e8042c6e334049b.jpg)

![f051062aa5d8c0967ff3d030555b7074ba0ed12f10d0761739f3d6ad98ec170c.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/images/f051062aa5d8c0967ff3d030555b7074ba0ed12f10d0761739f3d6ad98ec170c.jpg)

### Tables

![09429431e45ecfb811d3b296b25c548d1ff03984953b34ecbbdd9bc56fefd3e9.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/09429431e45ecfb811d3b296b25c548d1ff03984953b34ecbbdd9bc56fefd3e9.jpg)

![2df27311db7e684ab31ba7b8fcc6b5613a323ec5697376537ef911462f06eb5d.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/2df27311db7e684ab31ba7b8fcc6b5613a323ec5697376537ef911462f06eb5d.jpg)

![32e10d951cf3510635f969315b43f70f72565856b6c401478f9e9d58594a2815.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/32e10d951cf3510635f969315b43f70f72565856b6c401478f9e9d58594a2815.jpg)

![3ba8ca94eb7a04a43b988c9bcb1d7e09896e6b44991bd31f24e7a34eaef73527.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/3ba8ca94eb7a04a43b988c9bcb1d7e09896e6b44991bd31f24e7a34eaef73527.jpg)

![428c085a3924edcc6d63657b054ee70a6d122d08bc9d12369c4554a81313a5d2.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/428c085a3924edcc6d63657b054ee70a6d122d08bc9d12369c4554a81313a5d2.jpg)

![653b56a0ac7696f0be562ee60e5b965c83befb337af6eef0622f8c31cd67f346.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/653b56a0ac7696f0be562ee60e5b965c83befb337af6eef0622f8c31cd67f346.jpg)

![8eb62eab82840fc60b2f1b6aafae0a8b56784a2fdd7dfd32389a54affeffeda5.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/8eb62eab82840fc60b2f1b6aafae0a8b56784a2fdd7dfd32389a54affeffeda5.jpg)

![a9ea509b7cc640822d752cb41dfc7637344a45c0cc0c9a12d3efe49179df853f.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/a9ea509b7cc640822d752cb41dfc7637344a45c0cc0c9a12d3efe49179df853f.jpg)

![b3997373460bfc8f88fc0b2ea76309ad340119c64ca1217853d3b9725dd634fe.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/b3997373460bfc8f88fc0b2ea76309ad340119c64ca1217853d3b9725dd634fe.jpg)

![bf65e20a178a1b60882dadab59518712a3ee6892a9baefca15a584e986434f8f.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/bf65e20a178a1b60882dadab59518712a3ee6892a9baefca15a584e986434f8f.jpg)

![c4f873a1d13d53fad78c633c8aa97eea4b13e311ea8d54256529d5afed7f5a13.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/c4f873a1d13d53fad78c633c8aa97eea4b13e311ea8d54256529d5afed7f5a13.jpg)

![d0be81d9b27a3d5927a3a9b2a021653d0d24805d573418859f9b0329721356e7.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/d0be81d9b27a3d5927a3a9b2a021653d0d24805d573418859f9b0329721356e7.jpg)

![dea99d1c34ebcafe0162866c739be4e84ff263fd46782c5c7e7dd9e29318476a.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/dea99d1c34ebcafe0162866c739be4e84ff263fd46782c5c7e7dd9e29318476a.jpg)

![ef1ed637857e5f97de48a45458f9f111e775fac48b613952413c8575244bd64b.jpg](../icml_results/1509_AssistanceZero_%20Scalably%20Solving%20Assistance%20Games/tables/ef1ed637857e5f97de48a45458f9f111e775fac48b613952413c8575244bd64b.jpg)

## Learning Survival Distributions with the Asymmetric Laplace Distribution


### Images

![02c7b34f70de03a7b60c30ced30ad80e449fe03c844dd6c55df42218ac476710.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/02c7b34f70de03a7b60c30ced30ad80e449fe03c844dd6c55df42218ac476710.jpg)

![0e0fc3241be5e1173850a097aefd7552981863e2f9388cb59f6f919ad19a33fe.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/0e0fc3241be5e1173850a097aefd7552981863e2f9388cb59f6f919ad19a33fe.jpg)

![1081a6b31aaed96391a99a358387e2c0f2dc1344c47417f1eb8efb2d78ebbd54.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/1081a6b31aaed96391a99a358387e2c0f2dc1344c47417f1eb8efb2d78ebbd54.jpg)

![144fb6aeafd623d63adc6b3d9ce4311b1b6738391eec81688db49870a4da845f.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/144fb6aeafd623d63adc6b3d9ce4311b1b6738391eec81688db49870a4da845f.jpg)

![17cef03ac5e53a90121f7ef62860e227e5adaef5f83df1a4feae2a9309d72e16.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/17cef03ac5e53a90121f7ef62860e227e5adaef5f83df1a4feae2a9309d72e16.jpg)

![1c218c3dc5b4f3a9f5877bb206becd82529d785dee2e895f563f86cf4756bc53.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/1c218c3dc5b4f3a9f5877bb206becd82529d785dee2e895f563f86cf4756bc53.jpg)

![2972546049f28b313b9079757bd1e2b98e86cce4f6000325b41b0f4943714460.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/2972546049f28b313b9079757bd1e2b98e86cce4f6000325b41b0f4943714460.jpg)

![56c58d6fa626ab20e2db570f854e0ab95a7c53fa4e4a850eb648614728fe313d.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/56c58d6fa626ab20e2db570f854e0ab95a7c53fa4e4a850eb648614728fe313d.jpg)

![654ca0687cacf5bc2e1aa7420102c730db79956d2ee6553672affeb030ac3393.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/654ca0687cacf5bc2e1aa7420102c730db79956d2ee6553672affeb030ac3393.jpg)

![6dddb5d07a974c26bf67e0b7b9323588dc076bf2aff4adfc010998a4d0fbd085.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/6dddb5d07a974c26bf67e0b7b9323588dc076bf2aff4adfc010998a4d0fbd085.jpg)

![7d2379a2ff41a2460b3af069801a59ef30a4c83655f9baf3659944a6ea4eb947.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/7d2379a2ff41a2460b3af069801a59ef30a4c83655f9baf3659944a6ea4eb947.jpg)

![8f2a666512bc9a2db4281a0f8c61761044584e8a75874c5cc75f73a05b4662ca.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/8f2a666512bc9a2db4281a0f8c61761044584e8a75874c5cc75f73a05b4662ca.jpg)

![91b5f30f7e670fa17e3efad575510a4607bd1544b3046e630bfba6da6a1aac10.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/91b5f30f7e670fa17e3efad575510a4607bd1544b3046e630bfba6da6a1aac10.jpg)

![951bb88afe47f13ea4dd62ead91abfbad2bd071a41acef0223e6cd34529c0f88.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/951bb88afe47f13ea4dd62ead91abfbad2bd071a41acef0223e6cd34529c0f88.jpg)

![9aca2e85d057cb985e4b584e2e1255933ea0f1e13b21e4f200f3864f0f94e159.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/9aca2e85d057cb985e4b584e2e1255933ea0f1e13b21e4f200f3864f0f94e159.jpg)

![9b1d984f4590d251fec7d7972b04cc5875fc7d2d9aa8531eac0ef1b4b9d8c122.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/9b1d984f4590d251fec7d7972b04cc5875fc7d2d9aa8531eac0ef1b4b9d8c122.jpg)

![9d9720a835261edf4ddb8190df69feb47004f7acd790e1eae982dad5e5137e40.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/9d9720a835261edf4ddb8190df69feb47004f7acd790e1eae982dad5e5137e40.jpg)

![a47ea64bf87492f8ac59b87d7618f761654bf81085b228c2da5bef11c287a3f4.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/a47ea64bf87492f8ac59b87d7618f761654bf81085b228c2da5bef11c287a3f4.jpg)

![a57054ea274d32452f691b610b1a3d4b5172c4d59a20bcb27fe2bebc6297e233.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/a57054ea274d32452f691b610b1a3d4b5172c4d59a20bcb27fe2bebc6297e233.jpg)

![ac66a5b0efe51787d4763d8f4863bb4ac79d8a87a6c393b5168849984a076e83.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/ac66a5b0efe51787d4763d8f4863bb4ac79d8a87a6c393b5168849984a076e83.jpg)

![b16b3a915449e33b4dc497b8d632f8926517862e5d98d0e006219fca54e76eaa.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/b16b3a915449e33b4dc497b8d632f8926517862e5d98d0e006219fca54e76eaa.jpg)

![b4dc2fa4e6be31bb169d68c0879294a9a518d26ed6ca800cf0389ce93f5f7bc5.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/b4dc2fa4e6be31bb169d68c0879294a9a518d26ed6ca800cf0389ce93f5f7bc5.jpg)

![b65e434db023f7ca2391b76005c751af5339e63b5cadc75d77ade440aab7402e.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/b65e434db023f7ca2391b76005c751af5339e63b5cadc75d77ade440aab7402e.jpg)

![b6eaa7a19102efcf23a88a0fa4dfd55ba4735b5ef94b911adbbeb6fe6f6181e8.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/b6eaa7a19102efcf23a88a0fa4dfd55ba4735b5ef94b911adbbeb6fe6f6181e8.jpg)

![cf9005e5ae8a5cf9020711eba6618803b207cf6871ad5b5dda619766273d46c1.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/cf9005e5ae8a5cf9020711eba6618803b207cf6871ad5b5dda619766273d46c1.jpg)

![f79a86e50d527b6a75f15fb3d60c4f1454bfd481ca7dd4dd364e73f53402d536.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/images/f79a86e50d527b6a75f15fb3d60c4f1454bfd481ca7dd4dd364e73f53402d536.jpg)

### Tables

![1c36bb38cebca9d69cf16c5a63425bc183b90161361588388634d761bfa4b0ea.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/1c36bb38cebca9d69cf16c5a63425bc183b90161361588388634d761bfa4b0ea.jpg)

![2058e02593f456c4c39aea8066b771d8dc57ae409997fd3a1f910eb407a6597d.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/2058e02593f456c4c39aea8066b771d8dc57ae409997fd3a1f910eb407a6597d.jpg)

![2a8cf25f8edbbfdedf98a349339a00de6936a0bbdeee71965f5f41017a9dfb37.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/2a8cf25f8edbbfdedf98a349339a00de6936a0bbdeee71965f5f41017a9dfb37.jpg)

![6692c89567b44a50a2966afa3e41289091228857382d3844ed8c6943d8cbc7a3.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/6692c89567b44a50a2966afa3e41289091228857382d3844ed8c6943d8cbc7a3.jpg)

![875ade51d2c3484a11f467de592cf309acd7f5ebf5037fa4519527f9d5b71b80.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/875ade51d2c3484a11f467de592cf309acd7f5ebf5037fa4519527f9d5b71b80.jpg)

![95b1518ee12e209c68a86b588896b79a96c60fdf31b7e1e25463d16cf4c97a67.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/95b1518ee12e209c68a86b588896b79a96c60fdf31b7e1e25463d16cf4c97a67.jpg)

![ae17808e043c1565b4f70dfbfa2b6538774b4262bbfdbd5c0eabce691a0b323b.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/ae17808e043c1565b4f70dfbfa2b6538774b4262bbfdbd5c0eabce691a0b323b.jpg)

![d5c4f4f9bc4ef3caeaf06c8fddef7e6ea394a418c1fc9da0c66f0b217604c06b.jpg](../icml_results/1510_Learning%20Survival%20Distributions%20with%20the%20Asymmetric%20Laplace%20Distribution/tables/d5c4f4f9bc4ef3caeaf06c8fddef7e6ea394a418c1fc9da0c66f0b217604c06b.jpg)

## The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training


### Images

![0a2a0924ed9951366bee701a705a9a040c77d06a9396dadffc85f743e6fa6b64.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/0a2a0924ed9951366bee701a705a9a040c77d06a9396dadffc85f743e6fa6b64.jpg)

![126e82ba316e8c8070123021a09adf47b13261ed5f829bd97c2ab50b34575c9b.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/126e82ba316e8c8070123021a09adf47b13261ed5f829bd97c2ab50b34575c9b.jpg)

![1bb4e882e845b73cfdabf756965ffb48ef8def6759c191c62788b599c7425287.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/1bb4e882e845b73cfdabf756965ffb48ef8def6759c191c62788b599c7425287.jpg)

![254166a34e66ab9207c894b94c34fb8feec6c7c4309ecea2c0e06829a4a27b19.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/254166a34e66ab9207c894b94c34fb8feec6c7c4309ecea2c0e06829a4a27b19.jpg)

![26b009dd59573356703d1a003328a2ad3f67d7e1a616af61b6fac89ffa2e6142.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/26b009dd59573356703d1a003328a2ad3f67d7e1a616af61b6fac89ffa2e6142.jpg)

![274b5c9fad2424534a4d6000e9d925f79446b774561f5a55ed6aa817b5052884.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/274b5c9fad2424534a4d6000e9d925f79446b774561f5a55ed6aa817b5052884.jpg)

![278d5f15ac92052cc9aa9db1a81a34a3f1c0f695363b1eacd264ce2c13d28c64.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/278d5f15ac92052cc9aa9db1a81a34a3f1c0f695363b1eacd264ce2c13d28c64.jpg)

![35115e83d531c4cc7bc970cdd50306362eb1fcefaada494b3feb803b8d39b4ae.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/35115e83d531c4cc7bc970cdd50306362eb1fcefaada494b3feb803b8d39b4ae.jpg)

![3d8fd0097acfd9d2082b39b4d274aebbf9eb612705dbc2b9ae0fb17a141905c0.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/3d8fd0097acfd9d2082b39b4d274aebbf9eb612705dbc2b9ae0fb17a141905c0.jpg)

![3f8725a8a2e9a1388eb95e264a4ee8933cab9337de127a87332cc707dfdb6e59.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/3f8725a8a2e9a1388eb95e264a4ee8933cab9337de127a87332cc707dfdb6e59.jpg)

![424fd363af0f201ad67ac41a2f5eb62caa6c1981ea7a55c5d122b48b8e5ad2e0.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/424fd363af0f201ad67ac41a2f5eb62caa6c1981ea7a55c5d122b48b8e5ad2e0.jpg)

![51451ff65f8b6cc23042f0f4d9ad5891dbbe1276c0c9d9c060f2cbf42f48d111.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/51451ff65f8b6cc23042f0f4d9ad5891dbbe1276c0c9d9c060f2cbf42f48d111.jpg)

![5bfcd28f049212ad0f38707edd1c7075d9bff20ec3aefa6d016d8efb668213be.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/5bfcd28f049212ad0f38707edd1c7075d9bff20ec3aefa6d016d8efb668213be.jpg)

![614ad919f7a42bac5ca60776cedb75789521b910138322e31f34c5a64c34dde1.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/614ad919f7a42bac5ca60776cedb75789521b910138322e31f34c5a64c34dde1.jpg)

![641a7ab1decf21fd6c2cac382a97dec8f23751ab21d11d834e67ef161c03e4e4.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/641a7ab1decf21fd6c2cac382a97dec8f23751ab21d11d834e67ef161c03e4e4.jpg)

![8c178a2041b800b42a7580a420465f9cf59f01fa241a4e78ceb6368b24c75b2f.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/8c178a2041b800b42a7580a420465f9cf59f01fa241a4e78ceb6368b24c75b2f.jpg)

![92b6571dbc963f2854f59bcd2020992aae730fad3e4ec83ae55b82b0b39f25fc.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/92b6571dbc963f2854f59bcd2020992aae730fad3e4ec83ae55b82b0b39f25fc.jpg)

![9b477a6c718a3bff5fe0ebfe6e3a6e1d1f233445d89a591b2c3e1d6339a57773.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/9b477a6c718a3bff5fe0ebfe6e3a6e1d1f233445d89a591b2c3e1d6339a57773.jpg)

![a0c1df5e88976313a0c4186c315a191b91fa2b117c63887efba4a3d359eb68ca.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/a0c1df5e88976313a0c4186c315a191b91fa2b117c63887efba4a3d359eb68ca.jpg)

![af7e504e3adaf6e4a9b19f9afcd6f3c9b9d639a4410593af8c0f011a115d24a5.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/af7e504e3adaf6e4a9b19f9afcd6f3c9b9d639a4410593af8c0f011a115d24a5.jpg)

![c001c446eca421e94a938815d1c024ef78cb95957f012f4697f8cbd124cd07fc.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/c001c446eca421e94a938815d1c024ef78cb95957f012f4697f8cbd124cd07fc.jpg)

![cf02023daae03f1709308817a3d9b1fe51c4fc95c2fae500408777fabafa110b.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/cf02023daae03f1709308817a3d9b1fe51c4fc95c2fae500408777fabafa110b.jpg)

![d02992c56b2385ea5b006530094d2ab43344051f2de3d66bb084014b366aa858.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/d02992c56b2385ea5b006530094d2ab43344051f2de3d66bb084014b366aa858.jpg)

![dc0ebfdd47293279d9f38c260571cb78012dc7795dbaffadccfcf3107a8c9404.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/dc0ebfdd47293279d9f38c260571cb78012dc7795dbaffadccfcf3107a8c9404.jpg)

![e9ca0a168d064c7e821fdad69eefb01cdc20d3bdf5fbb6fe2ca72368361f687b.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/e9ca0a168d064c7e821fdad69eefb01cdc20d3bdf5fbb6fe2ca72368361f687b.jpg)

![f20b168bec85150d2fb317a9e54f35906f4ca11aa7f5c12c925c900172cc8bf0.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/f20b168bec85150d2fb317a9e54f35906f4ca11aa7f5c12c925c900172cc8bf0.jpg)

![f5a074a9410764ba63732086ee25f7ecdc534dbcb1386b35faf09e55e2861366.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/f5a074a9410764ba63732086ee25f7ecdc534dbcb1386b35faf09e55e2861366.jpg)

![ff7e5557b96d6193a8727273834af3781a8eb61b88a72b7d7a2dc0a9d7d7169c.jpg](../icml_results/1511_The%20Surprising%20Agreement%20Between%20Convex%20Optimization%20Theory%20and%20Learning-Rate%20Scheduling%20for%20Large%20M/images/ff7e5557b96d6193a8727273834af3781a8eb61b88a72b7d7a2dc0a9d7d7169c.jpg)

## DragLoRA: Online Optimization of LoRA Adapters for Drag-based Image Editing in Diffusion Model


### Images

![10ceea72eed0ef43fb067849fa5ce4b046273c60935e442086feaa7b095c8e1a.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/10ceea72eed0ef43fb067849fa5ce4b046273c60935e442086feaa7b095c8e1a.jpg)

![1f4946b0fb62fc12b47cf2d66be8df412da2a78a7dd97457b2a669a79ca179d8.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/1f4946b0fb62fc12b47cf2d66be8df412da2a78a7dd97457b2a669a79ca179d8.jpg)

![33f12465f7993304a279c51bb8c73b5ee447186a7e27435b2bbe33eecaf7a41b.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/33f12465f7993304a279c51bb8c73b5ee447186a7e27435b2bbe33eecaf7a41b.jpg)

![911d0465b9f5096dbe99503bdb8f35c5a2749abf1de30cddf0f5fe029764442b.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/911d0465b9f5096dbe99503bdb8f35c5a2749abf1de30cddf0f5fe029764442b.jpg)

![94a83ccac61018f5505518ab155b03498fd3ebb3a6afdfc2636ce5c33df5ebee.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/94a83ccac61018f5505518ab155b03498fd3ebb3a6afdfc2636ce5c33df5ebee.jpg)

![99264973acc47fd3881234d627cda2df05f87ad3e4ce289d0fdcea253c87b6ca.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/99264973acc47fd3881234d627cda2df05f87ad3e4ce289d0fdcea253c87b6ca.jpg)

![a861cf4d4f22c6421b4d01242e4c7c9ecbd68b69d8b56661a5fb4676e0dc30f5.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/a861cf4d4f22c6421b4d01242e4c7c9ecbd68b69d8b56661a5fb4676e0dc30f5.jpg)

![c0d9b8b2e5404998a344ce32267b6f1821e448a72f1e5e0c1ab9137dffef22c6.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/c0d9b8b2e5404998a344ce32267b6f1821e448a72f1e5e0c1ab9137dffef22c6.jpg)

![d55ece0eaf400a792b151ab4fa8f0c8839a900aa61b6b5d733f586bbb43dbcc9.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/d55ece0eaf400a792b151ab4fa8f0c8839a900aa61b6b5d733f586bbb43dbcc9.jpg)

![f2fa2f448f365ca47c1c57ee848f5017b80519e753c929301c103918cdffc252.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/images/f2fa2f448f365ca47c1c57ee848f5017b80519e753c929301c103918cdffc252.jpg)

### Tables

![3003700de8a1a530ebd5d01b453dd8b3870db5df84b4289a9584d8cdd0805a33.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/tables/3003700de8a1a530ebd5d01b453dd8b3870db5df84b4289a9584d8cdd0805a33.jpg)

![37d4eace7bedab139bfbd2e1e4392d920ceca09265a593f1bbbdf2ba756d5b9b.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/tables/37d4eace7bedab139bfbd2e1e4392d920ceca09265a593f1bbbdf2ba756d5b9b.jpg)

![49d4b2797f390d8ebf64eda40d4944d230f9f429a7f6f9f90af0defb2d4b314e.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/tables/49d4b2797f390d8ebf64eda40d4944d230f9f429a7f6f9f90af0defb2d4b314e.jpg)

![4f27bd076b04facc257d13f55bf9c40a6a08043faf1a0048be3d585cf5e47020.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/tables/4f27bd076b04facc257d13f55bf9c40a6a08043faf1a0048be3d585cf5e47020.jpg)

![66cc9c0111da962c4b926e9adb447379d7bb8a52f30708a01e0481f15d54b386.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/tables/66cc9c0111da962c4b926e9adb447379d7bb8a52f30708a01e0481f15d54b386.jpg)

![9dec9897932bdf5285ecac4c38269d5ea1310dd8954fab92a822ef5dfa1d5f7b.jpg](../icml_results/1512_DragLoRA_%20Online%20Optimization%20of%20LoRA%20Adapters%20for%20Drag-based%20Image%20Editing%20in%20Diffusion%20Model/tables/9dec9897932bdf5285ecac4c38269d5ea1310dd8954fab92a822ef5dfa1d5f7b.jpg)

## Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule


### Images

![0816f5d6ffa8cb7534483680050365c644cf4cb201df8f40cfd7dfda807544c7.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/0816f5d6ffa8cb7534483680050365c644cf4cb201df8f40cfd7dfda807544c7.jpg)

![1165acea8faf20d2d3f5d17a69644376510b6a29e7fd69135c718380f09a153d.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/1165acea8faf20d2d3f5d17a69644376510b6a29e7fd69135c718380f09a153d.jpg)

![1f7ce70d31b34a0e3a4755c6fec96f81f691196e7d11f1161a4f33f876c481ff.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/1f7ce70d31b34a0e3a4755c6fec96f81f691196e7d11f1161a4f33f876c481ff.jpg)

![220d6b96258963efbad7c2bd1d6d9e91c6b01663d357349f3d3023b78cc458b0.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/220d6b96258963efbad7c2bd1d6d9e91c6b01663d357349f3d3023b78cc458b0.jpg)

![2b8074a18a0f0f06c2bd6b7458741781b0d1b611256f36b869760d5e7a071ad0.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/2b8074a18a0f0f06c2bd6b7458741781b0d1b611256f36b869760d5e7a071ad0.jpg)

![64541d65ed38c3a267869f10d836ee986450ec9f46e671e7d9697a13bd261912.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/64541d65ed38c3a267869f10d836ee986450ec9f46e671e7d9697a13bd261912.jpg)

![82e630cb510c63f62410908352f0288006e87273741d4951d271a8fb0e85f4ab.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/82e630cb510c63f62410908352f0288006e87273741d4951d271a8fb0e85f4ab.jpg)

![a5f01600cebbadc5daa1560b5bd5e32319394aa76bb8ca4c7a3fb81bfebbe24b.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/a5f01600cebbadc5daa1560b5bd5e32319394aa76bb8ca4c7a3fb81bfebbe24b.jpg)

![a68a2ece38ae33020d5a8f0f788ed18aa76015ff019dfff5ccf42a3a02264bda.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/a68a2ece38ae33020d5a8f0f788ed18aa76015ff019dfff5ccf42a3a02264bda.jpg)

![af6b847bcff7c2dfc4a97e42976a1b179be7f6a261a153805cca80b94a330a2a.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/af6b847bcff7c2dfc4a97e42976a1b179be7f6a261a153805cca80b94a330a2a.jpg)

![b7df761350244cc17fc722a25912cd280bbfba0dd49128b09b6733d7294231f8.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/b7df761350244cc17fc722a25912cd280bbfba0dd49128b09b6733d7294231f8.jpg)

![ccbde53d6756bb4390b12e9317dc52e637708f2acac209054a45d73ee964cf9c.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/ccbde53d6756bb4390b12e9317dc52e637708f2acac209054a45d73ee964cf9c.jpg)

![cd21335eb3d3a96d329bff2f7c582ed6ec1b5cf62952cb7ba2f96a7287930645.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/cd21335eb3d3a96d329bff2f7c582ed6ec1b5cf62952cb7ba2f96a7287930645.jpg)

![d0d35e84eb43ab214f7f22ddd4e6739cc249e696d8df8b10a8d7fdf1cdc47d35.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/d0d35e84eb43ab214f7f22ddd4e6739cc249e696d8df8b10a8d7fdf1cdc47d35.jpg)

![dabefc868e9ac1d23f78182f602fac1a3fc48d3ed7d35a0b81ae9c5aa8bb3fa7.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/dabefc868e9ac1d23f78182f602fac1a3fc48d3ed7d35a0b81ae9c5aa8bb3fa7.jpg)

![f47d4e4090d051e5c47c8737e46d3983f59af38fb2dbc10c24afb2b32242ca3b.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/images/f47d4e4090d051e5c47c8737e46d3983f59af38fb2dbc10c24afb2b32242ca3b.jpg)

### Tables

![083dcc8e45c9c7640c40686baa86c37c21737b2cfe134c0d5fbb1e2ba83006e7.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/083dcc8e45c9c7640c40686baa86c37c21737b2cfe134c0d5fbb1e2ba83006e7.jpg)

![0d9e0f436cc63713b474d14ccec008ae37053c5c61114951f3d14c06d143b8e5.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/0d9e0f436cc63713b474d14ccec008ae37053c5c61114951f3d14c06d143b8e5.jpg)

![45e13cb7daf7285225b2b885a4e93d4c077ea492605f6c9163b928c438c32d36.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/45e13cb7daf7285225b2b885a4e93d4c077ea492605f6c9163b928c438c32d36.jpg)

![4f5749e04eb3cfd36db8fbd9442d2f847ca263a75aa88a0c6cde5b9a824ad092.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/4f5749e04eb3cfd36db8fbd9442d2f847ca263a75aa88a0c6cde5b9a824ad092.jpg)

![871ecf8fb74a7451f648a900af684a4c5380aa8e71445d94413dcc397920062a.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/871ecf8fb74a7451f648a900af684a4c5380aa8e71445d94413dcc397920062a.jpg)

![955cfc660f494bb7ba1ef5e5b84531b0a9bff118b8ceb20b2998a4327beb7b9b.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/955cfc660f494bb7ba1ef5e5b84531b0a9bff118b8ceb20b2998a4327beb7b9b.jpg)

![b11935a60ed9431539af97fb7c77436a294ab0ac18b4ff378a75a1999740708f.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/b11935a60ed9431539af97fb7c77436a294ab0ac18b4ff378a75a1999740708f.jpg)

![b1fdb1918f065016bbcf19912e97736c6f7d99e0e96ef3f2d98a149f895a3e1f.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/b1fdb1918f065016bbcf19912e97736c6f7d99e0e96ef3f2d98a149f895a3e1f.jpg)

![dd823011e3b4acb499ca9eea096f1c97c79d2cd3fa8357bc926881bf7f214ab3.jpg](../icml_results/1513_Piloting%20Structure-Based%20Drug%20Design%20via%20Modality-Specific%20Optimal%20Schedule/tables/dd823011e3b4acb499ca9eea096f1c97c79d2cd3fa8357bc926881bf7f214ab3.jpg)

## Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting


### Images

![57b13fddb1c524dd80c73fe857a471f60579d5856346d215275d79076ec6216d.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/57b13fddb1c524dd80c73fe857a471f60579d5856346d215275d79076ec6216d.jpg)

![715840cc0d78d0915ee3588626abb53ad7542b86d7f18ca5b2b8faad81ab98ca.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/715840cc0d78d0915ee3588626abb53ad7542b86d7f18ca5b2b8faad81ab98ca.jpg)

![934884f14bf27d7f126a75e27a575d9b660087a728b1b1183de42d9daffe9060.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/934884f14bf27d7f126a75e27a575d9b660087a728b1b1183de42d9daffe9060.jpg)

![a751bc75a3cb8241e5084fda2db5de33a5edafe6a0cb148ed20d9ddcb5c852c2.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/a751bc75a3cb8241e5084fda2db5de33a5edafe6a0cb148ed20d9ddcb5c852c2.jpg)

![d67fa1fb6c1e610cce2a6f53112d1d160456094c03dd9f203d03c41a36599e3e.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/d67fa1fb6c1e610cce2a6f53112d1d160456094c03dd9f203d03c41a36599e3e.jpg)

![d85bde4cdc0e78b2177777ca404980a1e4da278928733e78601209764ee54b0c.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/d85bde4cdc0e78b2177777ca404980a1e4da278928733e78601209764ee54b0c.jpg)

![e465aef11e609e8822b3933448f7747344de801a7db7d1126d73f72d1af4d6b9.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/e465aef11e609e8822b3933448f7747344de801a7db7d1126d73f72d1af4d6b9.jpg)

![e58a08acdafeb1a5df7900f29f6a589206c986971bb02b08bdfc1c70531f652e.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/e58a08acdafeb1a5df7900f29f6a589206c986971bb02b08bdfc1c70531f652e.jpg)

![f5b5610f3212ae11f0e91d70f361153d646f57e907927c45dcf3b4558b2ee424.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/f5b5610f3212ae11f0e91d70f361153d646f57e907927c45dcf3b4558b2ee424.jpg)

![f6bc4a256879d7e30f5dd8d3eb78d958cdf4f75a46b2a53d84ffd8bff56e6cfb.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/images/f6bc4a256879d7e30f5dd8d3eb78d958cdf4f75a46b2a53d84ffd8bff56e6cfb.jpg)

### Tables

![098d9d9cafa98dbf3db42b3734401eb270bd250fd993f96284518457ef9e9f0d.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/098d9d9cafa98dbf3db42b3734401eb270bd250fd993f96284518457ef9e9f0d.jpg)

![1495b791c3551c4c3a324d174cbc455a9b6f3f448daaf616ac82e8bd98e48d32.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/1495b791c3551c4c3a324d174cbc455a9b6f3f448daaf616ac82e8bd98e48d32.jpg)

![16efc4e3a5a0e647281fbde3628f33393632324b4017aa84db20b207bd996534.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/16efc4e3a5a0e647281fbde3628f33393632324b4017aa84db20b207bd996534.jpg)

![5579407903251c2a79ba5c237ab7093ea0d3739a29b21ee49c4a47c938c29e4f.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/5579407903251c2a79ba5c237ab7093ea0d3739a29b21ee49c4a47c938c29e4f.jpg)

![5978baa1990f4e134685ceca2a31d0e137c2910700235aef3f451ea982d82546.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/5978baa1990f4e134685ceca2a31d0e137c2910700235aef3f451ea982d82546.jpg)

![839bdf44c41d618c61347fad27f47b22d07107d9fab630997629b6aaf5ac3465.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/839bdf44c41d618c61347fad27f47b22d07107d9fab630997629b6aaf5ac3465.jpg)

![8a85aa6d15ad2a2195635f26805bb70167ee5aa03eb2bb9849c6abe5b6fc2000.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/8a85aa6d15ad2a2195635f26805bb70167ee5aa03eb2bb9849c6abe5b6fc2000.jpg)

![8b5ec81fdccb3c3275965e5fb6f153fe1e061a93b07ec7bab0ef35f9abe29e47.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/8b5ec81fdccb3c3275965e5fb6f153fe1e061a93b07ec7bab0ef35f9abe29e47.jpg)

![a383e91efc636858f6790a52b4610c2f2b618d160df7b02659b506657f92dfc8.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/a383e91efc636858f6790a52b4610c2f2b618d160df7b02659b506657f92dfc8.jpg)

![a4223b049dea4b7807c90879ecb6eae2c3b851f8bd1a369d0879b77bd0f21344.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/a4223b049dea4b7807c90879ecb6eae2c3b851f8bd1a369d0879b77bd0f21344.jpg)

![a99ee709551b04433c7d0eb28292ffbac30508b12bd191f7d1945a03a8254271.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/a99ee709551b04433c7d0eb28292ffbac30508b12bd191f7d1945a03a8254271.jpg)

![c8d4575f7485145d0cae7c83aea4d3e79482be01271a50e650aa0889a6c09b58.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/c8d4575f7485145d0cae7c83aea4d3e79482be01271a50e650aa0889a6c09b58.jpg)

![db9dfb5585a1837016c9f9dbb6ec50ba27f604141a02b3f15d7212ae9f799447.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/db9dfb5585a1837016c9f9dbb6ec50ba27f604141a02b3f15d7212ae9f799447.jpg)

![de2665947910447e0ad0b7d16d4fab49530a22ef416719080c3fdc861d3c5ce8.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/de2665947910447e0ad0b7d16d4fab49530a22ef416719080c3fdc861d3c5ce8.jpg)

![e0f4edf218eec23520eeda67c028be55bceae3f70ba9a4d368cd8f40052b443f.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/e0f4edf218eec23520eeda67c028be55bceae3f70ba9a4d368cd8f40052b443f.jpg)

![f4b6b2fd7c038b52a5b9ecd248be8a3f50b8c7d84c7c2cb60c3b98560f4dcb95.jpg](../icml_results/1514_Time-VLM_%20Exploring%20Multimodal%20Vision-Language%20Models%20for%20Augmented%20Time%20Series%20Forecasting/tables/f4b6b2fd7c038b52a5b9ecd248be8a3f50b8c7d84c7c2cb60c3b98560f4dcb95.jpg)

## Efficiently Access Diffusion Fisher: Within the Outer Product Span Space


### Images

![03889f129940f62ba1eeaa4ecc0d830ad2386593635d804d0a44464147a70415.jpg](../icml_results/1515_Efficiently%20Access%20Diffusion%20Fisher_%20Within%20the%20Outer%20Product%20Span%20Space/images/03889f129940f62ba1eeaa4ecc0d830ad2386593635d804d0a44464147a70415.jpg)

![38132e82c3547ff8993febc0021b5a2df2e6bb416a80fffe8909a895267654e8.jpg](../icml_results/1515_Efficiently%20Access%20Diffusion%20Fisher_%20Within%20the%20Outer%20Product%20Span%20Space/images/38132e82c3547ff8993febc0021b5a2df2e6bb416a80fffe8909a895267654e8.jpg)

![c99d31bf2ecf04b1f1765602e7b93dd6d13c311d65ca079075896a35226ac24f.jpg](../icml_results/1515_Efficiently%20Access%20Diffusion%20Fisher_%20Within%20the%20Outer%20Product%20Span%20Space/images/c99d31bf2ecf04b1f1765602e7b93dd6d13c311d65ca079075896a35226ac24f.jpg)

![ef5cc9e51bc491f1dbe033ae6ff88273b4f786d575d47142de557048414a29b3.jpg](../icml_results/1515_Efficiently%20Access%20Diffusion%20Fisher_%20Within%20the%20Outer%20Product%20Span%20Space/images/ef5cc9e51bc491f1dbe033ae6ff88273b4f786d575d47142de557048414a29b3.jpg)

### Tables

![9b95ee2f8489e0aa36af0d3aa1d854cc5a567f8f6904a360a4bce0815fae53d3.jpg](../icml_results/1515_Efficiently%20Access%20Diffusion%20Fisher_%20Within%20the%20Outer%20Product%20Span%20Space/tables/9b95ee2f8489e0aa36af0d3aa1d854cc5a567f8f6904a360a4bce0815fae53d3.jpg)

![9be745ad626619463a3d1c01885a6059a890ff3e6e38172196fd707ca5f8a614.jpg](../icml_results/1515_Efficiently%20Access%20Diffusion%20Fisher_%20Within%20the%20Outer%20Product%20Span%20Space/tables/9be745ad626619463a3d1c01885a6059a890ff3e6e38172196fd707ca5f8a614.jpg)

![d8c95ec6c94ea174b1dd4769b338b95b6ad49ff19d211929528277ed9bbf8ea4.jpg](../icml_results/1515_Efficiently%20Access%20Diffusion%20Fisher_%20Within%20the%20Outer%20Product%20Span%20Space/tables/d8c95ec6c94ea174b1dd4769b338b95b6ad49ff19d211929528277ed9bbf8ea4.jpg)

## IRBridge: Solving Image Restoration Bridge with Pre-trained Generative Diffusion Models


### Images

![01f687cc4735b2fb7fa0f0d00d2519850994d67fa42b818dd99bc0fc58db4e6a.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/01f687cc4735b2fb7fa0f0d00d2519850994d67fa42b818dd99bc0fc58db4e6a.jpg)

![0f23cb2de9d07bd3e6fac598c03a69da2ef57a5c9dc41587d4eedf01f9461c04.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/0f23cb2de9d07bd3e6fac598c03a69da2ef57a5c9dc41587d4eedf01f9461c04.jpg)

![18f5153edf1cab7d7e33329758bd93ce2550ef95e4fc65479e5ae11a53d97997.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/18f5153edf1cab7d7e33329758bd93ce2550ef95e4fc65479e5ae11a53d97997.jpg)

![2c6dce2f1f7385901d217575c156d362ed96d8aacdd06d56f473996bbbf22c9f.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/2c6dce2f1f7385901d217575c156d362ed96d8aacdd06d56f473996bbbf22c9f.jpg)

![3a576809575b947913f48bb4367007ab4033c6d11d27d81d862b9c6210d84310.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/3a576809575b947913f48bb4367007ab4033c6d11d27d81d862b9c6210d84310.jpg)

![40457460cf21113b7e7f2a0557cc438a7c93313f5efa741184d340b41af8502b.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/40457460cf21113b7e7f2a0557cc438a7c93313f5efa741184d340b41af8502b.jpg)

![466c1692d73233ada735ba8e4cded6cb7477d183d2465da03c475f9a905289f5.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/466c1692d73233ada735ba8e4cded6cb7477d183d2465da03c475f9a905289f5.jpg)

![4adcc13a473082eab33e2fa07e7f33b038c040fd401dbd02e8a8020cdc1e714c.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/4adcc13a473082eab33e2fa07e7f33b038c040fd401dbd02e8a8020cdc1e714c.jpg)

![4ec487ab7412b11ceb0a59d934f87aa7b488ad591d0c92f8fb06900606351c9e.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/4ec487ab7412b11ceb0a59d934f87aa7b488ad591d0c92f8fb06900606351c9e.jpg)

![5ba0e8273cdfd15e4c4f4ae8d118bf7e754151a14afd7f68a1707a25233daf09.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/5ba0e8273cdfd15e4c4f4ae8d118bf7e754151a14afd7f68a1707a25233daf09.jpg)

![5c0d36f64c2ae67a02b4e6912366a26c1ce2556b8ad729ca82e96548be6fdc59.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/5c0d36f64c2ae67a02b4e6912366a26c1ce2556b8ad729ca82e96548be6fdc59.jpg)

![63a6a3d9557c40765f6594a483ee2541f654e77b13b154d53eab6f240b10c68f.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/63a6a3d9557c40765f6594a483ee2541f654e77b13b154d53eab6f240b10c68f.jpg)

![65035e5a8ad8b61b71eea6d60c48bf8f57ddcf651dc681172cd8edb379209e97.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/65035e5a8ad8b61b71eea6d60c48bf8f57ddcf651dc681172cd8edb379209e97.jpg)

![651b3816c676f3a9752aff39519b692d57595ad2b49ece4c35746394fe9f1d21.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/651b3816c676f3a9752aff39519b692d57595ad2b49ece4c35746394fe9f1d21.jpg)

![678aa3bd3f96fef04fa8195814e2192bf9f3d98af757a748c036874b829cd7a9.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/678aa3bd3f96fef04fa8195814e2192bf9f3d98af757a748c036874b829cd7a9.jpg)

![72ad0d442785317320a6161b9ed3b661de83317d8675ec6753639d226077b262.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/72ad0d442785317320a6161b9ed3b661de83317d8675ec6753639d226077b262.jpg)

![8afa658378a4d11240638ada1295773fe24d2f1a52ef8de8f207d8b0e1174f3c.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/8afa658378a4d11240638ada1295773fe24d2f1a52ef8de8f207d8b0e1174f3c.jpg)

![924e3de2d935b60a7b46aa82015f7b89192eb418d8d83902182ad888c0462eed.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/924e3de2d935b60a7b46aa82015f7b89192eb418d8d83902182ad888c0462eed.jpg)

![982a4da16265c4dee443ebbec8ab83896f10ef002e0c3c6695d7a9c52ddda4bf.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/982a4da16265c4dee443ebbec8ab83896f10ef002e0c3c6695d7a9c52ddda4bf.jpg)

![a5fbd01aada7acb4c80ad4fbed35cdd90cb4858e0076e586df7bf48893790576.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/a5fbd01aada7acb4c80ad4fbed35cdd90cb4858e0076e586df7bf48893790576.jpg)

![b1e6c83e159d7a2abdc49f8cba553f0f257183f1b021696d6caa22d372a8ed1e.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/b1e6c83e159d7a2abdc49f8cba553f0f257183f1b021696d6caa22d372a8ed1e.jpg)

![b83904ab91e2987c90b8b54094930cbac749272f8ae6ff568f8eadad115ef0c7.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/b83904ab91e2987c90b8b54094930cbac749272f8ae6ff568f8eadad115ef0c7.jpg)

![d0a8e9d5ae76560999fbfbf4d98a0d4c663cd8c9bf94c2a0e810c7294284dca8.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/d0a8e9d5ae76560999fbfbf4d98a0d4c663cd8c9bf94c2a0e810c7294284dca8.jpg)

![d54f0654d3bc9e4071a4d83a08754c6d4e52a358401bd3543693bc331ea41f63.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/d54f0654d3bc9e4071a4d83a08754c6d4e52a358401bd3543693bc331ea41f63.jpg)

![da7d65379052a507d0416b0a9fc5d84e4c52a6118bdfb6be95463855353a7cea.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/da7d65379052a507d0416b0a9fc5d84e4c52a6118bdfb6be95463855353a7cea.jpg)

![f3b2d251ddaa0c5f6f0744b6461cf9ad5839e4bd834d1e0abe29f785164f0a45.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/images/f3b2d251ddaa0c5f6f0744b6461cf9ad5839e4bd834d1e0abe29f785164f0a45.jpg)

### Tables

![0ffa36ac8864e3529523176cca803e268798f8d1c545102ffc013c131eb218da.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/0ffa36ac8864e3529523176cca803e268798f8d1c545102ffc013c131eb218da.jpg)

![0fff757d5464be9aeef44851da2f9bb1b16fccf75002f3af4cd4a63254df02ec.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/0fff757d5464be9aeef44851da2f9bb1b16fccf75002f3af4cd4a63254df02ec.jpg)

![17520637be0bd56c86a47801a50981f39337111380a237652f22ac77ed20e474.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/17520637be0bd56c86a47801a50981f39337111380a237652f22ac77ed20e474.jpg)

![1aa8b97f9bafdabb94109c7b9942a84576aa68e11950f419cac8a3f9b28aac08.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/1aa8b97f9bafdabb94109c7b9942a84576aa68e11950f419cac8a3f9b28aac08.jpg)

![2617feb2ed5cd08b50d3de9e5296c22af5278ccf9c7fc680a3dc1689a5c8ee97.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/2617feb2ed5cd08b50d3de9e5296c22af5278ccf9c7fc680a3dc1689a5c8ee97.jpg)

![318c5add9e65a7fda554cd6fbc14bef61bbef956611fdee697570886924d5b1c.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/318c5add9e65a7fda554cd6fbc14bef61bbef956611fdee697570886924d5b1c.jpg)

![4ef7373426f2d77bb4de7464d278fe531be9fae41792f58f6d2593fb581bc43e.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/4ef7373426f2d77bb4de7464d278fe531be9fae41792f58f6d2593fb581bc43e.jpg)

![790b8816c77c2a4d469876159d0a420ce076387b713af82aedc3bba278f499ee.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/790b8816c77c2a4d469876159d0a420ce076387b713af82aedc3bba278f499ee.jpg)

![ab081dd3c7215fba0a623ec2ff3fdd8ea250d5fc9e92a3c8c532547c8ccd8fc0.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/ab081dd3c7215fba0a623ec2ff3fdd8ea250d5fc9e92a3c8c532547c8ccd8fc0.jpg)

![f3571349690db376a5f6edf5616c1131b382f9ad571dc87de72641de50c490ef.jpg](../icml_results/1516_IRBridge_%20Solving%20Image%20Restoration%20Bridge%20with%20Pre-trained%20Generative%20Diffusion%20Models/tables/f3571349690db376a5f6edf5616c1131b382f9ad571dc87de72641de50c490ef.jpg)

## Splitting & Integrating: Out-of-Distribution Detection via Adversarial Gradient Attribution


### Images

![30a634665df0a74108e055835e526d39c1c73cbab634bd4ee9062d7a1af5f871.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/images/30a634665df0a74108e055835e526d39c1c73cbab634bd4ee9062d7a1af5f871.jpg)

![5b375ddcd9f06c45819547207eff7a48047b033d97920aa1bd929eea823176ca.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/images/5b375ddcd9f06c45819547207eff7a48047b033d97920aa1bd929eea823176ca.jpg)

### Tables

![026a1461bab7adfbc796040d6983dee7e351045937f1eae5765fb35d6d44374a.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/tables/026a1461bab7adfbc796040d6983dee7e351045937f1eae5765fb35d6d44374a.jpg)

![2be3f36bc6a2adc2a1b79ebf0769e2f2263237ef4b9e88ef393e902162bc153f.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/tables/2be3f36bc6a2adc2a1b79ebf0769e2f2263237ef4b9e88ef393e902162bc153f.jpg)

![56479c87a92c4b1bd1912ccf16ab6d96caab60036b0bd84a1970d3210b3cd384.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/tables/56479c87a92c4b1bd1912ccf16ab6d96caab60036b0bd84a1970d3210b3cd384.jpg)

![6769e7193463f1c1ac342a65a14be418df3698730b2cd6b79992c217af9889fc.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/tables/6769e7193463f1c1ac342a65a14be418df3698730b2cd6b79992c217af9889fc.jpg)

![7702b411237e0e94b993710afb63fcb7fadea392faef3a06b80801d099410740.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/tables/7702b411237e0e94b993710afb63fcb7fadea392faef3a06b80801d099410740.jpg)

![9a409d638deda25cc4ef313227f2cc68582430d85f5e876bccd2ed8ff0bc9e2a.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/tables/9a409d638deda25cc4ef313227f2cc68582430d85f5e876bccd2ed8ff0bc9e2a.jpg)

![be18cf10db4d6d7cb743d6add720eb85d79d32113ed1feab353c2a195782004f.jpg](../icml_results/1517_Splitting%20%26%20Integrating_%20Out-of-Distribution%20Detection%20via%20Adversarial%20Gradient%20Attribution/tables/be18cf10db4d6d7cb743d6add720eb85d79d32113ed1feab353c2a195782004f.jpg)

## The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models


### Images

![39a9ee30d18e271330046a2de39fe6b85592aab5b3e6a0792f74548c177034c9.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/39a9ee30d18e271330046a2de39fe6b85592aab5b3e6a0792f74548c177034c9.jpg)

![4695636ca0aaebd9dd5c87710b966a8f15f5f6047251ffa4f50897c84daef06f.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/4695636ca0aaebd9dd5c87710b966a8f15f5f6047251ffa4f50897c84daef06f.jpg)

![580e89dc83bb5691a65a44f7595044fd871d0035bbe246032a6c2f5e1b69eff0.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/580e89dc83bb5691a65a44f7595044fd871d0035bbe246032a6c2f5e1b69eff0.jpg)

![5b17499a605d3c082d23be3429f1de72e7949767ea6c72ecc62d966831cd29e5.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/5b17499a605d3c082d23be3429f1de72e7949767ea6c72ecc62d966831cd29e5.jpg)

![5c3b45f979fe6ae293e5a11c89af8e2d151757479598cff42c1b07a94f4bbb5e.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/5c3b45f979fe6ae293e5a11c89af8e2d151757479598cff42c1b07a94f4bbb5e.jpg)

![8a3cc8f7e69aaa64e48a851eb0f96f2e3aa2d4a97ba3d8a1d1a0e7ec91e2c701.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/8a3cc8f7e69aaa64e48a851eb0f96f2e3aa2d4a97ba3d8a1d1a0e7ec91e2c701.jpg)

![8c1ebc2904f1f87a4672540cb3d3362c8dccb8de5a32139320860302eef74e1a.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/8c1ebc2904f1f87a4672540cb3d3362c8dccb8de5a32139320860302eef74e1a.jpg)

![a28fff94fca737f5c1b3a8e11c77968488e06ebbe5e6d5baa39ca4e236eb9704.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/a28fff94fca737f5c1b3a8e11c77968488e06ebbe5e6d5baa39ca4e236eb9704.jpg)

![e5c44d8cb872fb0427ab278c97c2f9467ce3dade3348cbcad6006bf543d2a4f5.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/e5c44d8cb872fb0427ab278c97c2f9467ce3dade3348cbcad6006bf543d2a4f5.jpg)

![ee9423e2aeeaa23714bcdd218835d31334bb75622e0b1a276e277043466ee6d6.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/ee9423e2aeeaa23714bcdd218835d31334bb75622e0b1a276e277043466ee6d6.jpg)

![f85eaa426fe93b5fd6d57e5fb58cbd9be6c6bf6885f3e340c5a0e6c8c9518597.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/images/f85eaa426fe93b5fd6d57e5fb58cbd9be6c6bf6885f3e340c5a0e6c8c9518597.jpg)

### Tables

![1db50946be3b109c05bc81675cc03ed44b16c409584514a4ea41fc2c77f2bd21.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/tables/1db50946be3b109c05bc81675cc03ed44b16c409584514a4ea41fc2c77f2bd21.jpg)

![a11c555941acf37d22122f339743420f95f4dbb215a3c6353773d88f88e46bde.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/tables/a11c555941acf37d22122f339743420f95f4dbb215a3c6353773d88f88e46bde.jpg)

![b1b5f658e2a5f7fc76ffd94ff02f31e0147c62482bff22fa54beed1f9e45e0e5.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/tables/b1b5f658e2a5f7fc76ffd94ff02f31e0147c62482bff22fa54beed1f9e45e0e5.jpg)

![df1a2bc3c8edd28774401ea10f0c614ceae00ec8c8c80ce38389264da3d18a85.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/tables/df1a2bc3c8edd28774401ea10f0c614ceae00ec8c8c80ce38389264da3d18a85.jpg)

![e473da121f2998f1696c2ea87cf4a61b53ccbd915efda97028bd8339dedfe4a2.jpg](../icml_results/1518_The%20Devil%20Is%20in%20the%20Details_%20Tackling%20Unimodal%20Spurious%20Correlations%20for%20Generalizable%20Multimodal%20Re/tables/e473da121f2998f1696c2ea87cf4a61b53ccbd915efda97028bd8339dedfe4a2.jpg)

## Otter: Generating Tests from Issues to Validate SWE Patches


### Images

![05b86a6a34348f478ef41db745a457cbb490bcb85711028cb809ccaa7e26c5c1.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/05b86a6a34348f478ef41db745a457cbb490bcb85711028cb809ccaa7e26c5c1.jpg)

![2904f0ae3f683f18e5682ec6a0981e866580c69ab29785efd408c1ee1c39f269.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/2904f0ae3f683f18e5682ec6a0981e866580c69ab29785efd408c1ee1c39f269.jpg)

![5aa898b845c7dd4a3b18217eaeed117aad6229b4fb00e3bb3598764c2931a6ab.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/5aa898b845c7dd4a3b18217eaeed117aad6229b4fb00e3bb3598764c2931a6ab.jpg)

![612c428d08bfa63f9ffb478b0f16ade57790385897f2c433e40917c1b82fbd19.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/612c428d08bfa63f9ffb478b0f16ade57790385897f2c433e40917c1b82fbd19.jpg)

![640129aac6611d956cc97cfbd135e07980cf42ee778fa2c1cfe49a6d061e7f52.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/640129aac6611d956cc97cfbd135e07980cf42ee778fa2c1cfe49a6d061e7f52.jpg)

![7391d70ccc81cfe3a4d152908f020dcb153b08f2675080a852e119affcf1ca33.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/7391d70ccc81cfe3a4d152908f020dcb153b08f2675080a852e119affcf1ca33.jpg)

![7a6b607f7571bf98a8359efcb5bbf6aa5dd8ccff7e049ab79a9447bd02639d73.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/7a6b607f7571bf98a8359efcb5bbf6aa5dd8ccff7e049ab79a9447bd02639d73.jpg)

![7b6b4468f348341b11d16dfefddcc4f1c90c3dd052d144119c7c262a8f160594.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/7b6b4468f348341b11d16dfefddcc4f1c90c3dd052d144119c7c262a8f160594.jpg)

![81e5a15be1d6d2258ef02bc187c6f55fe05ae50d2f680609bac7ddd3c8d2f12d.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/81e5a15be1d6d2258ef02bc187c6f55fe05ae50d2f680609bac7ddd3c8d2f12d.jpg)

![8206507d0c15a5b233f98c53e061bf74f364952381f100b75a2e2f9e338ef915.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/8206507d0c15a5b233f98c53e061bf74f364952381f100b75a2e2f9e338ef915.jpg)

![8309a007175ca0d8413235dd37d03859d414471a86c0c5f683856cc3129e4145.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/8309a007175ca0d8413235dd37d03859d414471a86c0c5f683856cc3129e4145.jpg)

![8720b69de76a952f7ce4a636c2b0395c707aff3b833aa623d579cfe3d295e8ee.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/8720b69de76a952f7ce4a636c2b0395c707aff3b833aa623d579cfe3d295e8ee.jpg)

![f6790751997d998f79b379b8d48eb258ac09054e1ae8c498d8d4405a9826e787.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/f6790751997d998f79b379b8d48eb258ac09054e1ae8c498d8d4405a9826e787.jpg)

![fe341c042a2fceb16b52e003fcd32f23b45191818374911bc421e880582eb46b.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/fe341c042a2fceb16b52e003fcd32f23b45191818374911bc421e880582eb46b.jpg)

![ff642339738ead3e87839471063a7ec5bd0667954ff719c8b9bc7e419f81e680.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/images/ff642339738ead3e87839471063a7ec5bd0667954ff719c8b9bc7e419f81e680.jpg)

### Tables

![0883d65a8a0ca6405ce5cc06d6c3f37d55b7abf13d401929244f6dde6ce17963.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/0883d65a8a0ca6405ce5cc06d6c3f37d55b7abf13d401929244f6dde6ce17963.jpg)

![1b59a8a447d171d0b5037bc9e5cbed2d1f56950bbfb86bb5bf831a7ab93b0086.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/1b59a8a447d171d0b5037bc9e5cbed2d1f56950bbfb86bb5bf831a7ab93b0086.jpg)

![20b5df54226b0c4a8ea7103952ddc5453a1420a3f79ae06c29f711f53e615a20.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/20b5df54226b0c4a8ea7103952ddc5453a1420a3f79ae06c29f711f53e615a20.jpg)

![24020547cd0aa26a6a25315087595cc43b793d130672896e72a223c517c2370f.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/24020547cd0aa26a6a25315087595cc43b793d130672896e72a223c517c2370f.jpg)

![263d4512f9e75987d82744eb7f35071f923cbb33966873b367872da529e7efbb.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/263d4512f9e75987d82744eb7f35071f923cbb33966873b367872da529e7efbb.jpg)

![748ca19ade0c35f86b13c7887321da05b6d43944a00606410db25f072e41b195.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/748ca19ade0c35f86b13c7887321da05b6d43944a00606410db25f072e41b195.jpg)

![76763529b67c42554c6e64925a188d4a2479a889be228d4ccfed4cf3884642ed.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/76763529b67c42554c6e64925a188d4a2479a889be228d4ccfed4cf3884642ed.jpg)

![7ed3aa5f2384d52c82821189149cccb0019d51b0e8a7cddbf5b7485f4c335949.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/7ed3aa5f2384d52c82821189149cccb0019d51b0e8a7cddbf5b7485f4c335949.jpg)

![a7a4738d467cde356c1ad45812ac2538314fd7e6fda9906fdc87b5df56d1db71.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/a7a4738d467cde356c1ad45812ac2538314fd7e6fda9906fdc87b5df56d1db71.jpg)

![bbe0aff7cb566a5b8c5d618c9a6496b005207af81a49928ac118b04cc344ea03.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/bbe0aff7cb566a5b8c5d618c9a6496b005207af81a49928ac118b04cc344ea03.jpg)

![ddb5d3032bde29d1b5d5d31a039b30b470318c5dbae76fcb9befb03a48d889ce.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/ddb5d3032bde29d1b5d5d31a039b30b470318c5dbae76fcb9befb03a48d889ce.jpg)

![eff9a8ca5afbb4e360e6ad63710d54ba4671cffaed3808f9dc3dd66ddba1f49d.jpg](../icml_results/1519_Otter_%20Generating%20Tests%20from%20Issues%20to%20Validate%20SWE%20Patches/tables/eff9a8ca5afbb4e360e6ad63710d54ba4671cffaed3808f9dc3dd66ddba1f49d.jpg)

## Geometric Resampling in Nearly Linear Time for Follow-the-Perturbed-Leader with Best-of-Both-Worlds Guarantee in Bandit Problems


### Images

![12980807e43072d4a3f595c421ae638aed5c735a11b3be4c8279b5f7bf66f869.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/12980807e43072d4a3f595c421ae638aed5c735a11b3be4c8279b5f7bf66f869.jpg)

![303391442afef359381d4e907f0ea9fbe556d57ce75102dfb0c675b456a1932a.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/303391442afef359381d4e907f0ea9fbe556d57ce75102dfb0c675b456a1932a.jpg)

![4fd0fec9bacd1c4126c2d94fd10c704760ba8ccdb0c48701c8497ed9949319bd.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/4fd0fec9bacd1c4126c2d94fd10c704760ba8ccdb0c48701c8497ed9949319bd.jpg)

![5946fa296abc46b214b41fd6e502d3593d804994afa434fbd5c18245e8a04b66.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/5946fa296abc46b214b41fd6e502d3593d804994afa434fbd5c18245e8a04b66.jpg)

![5a221f013fa894920696824285fbf2287da5c2c991db5409e9c3e509d26245b1.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/5a221f013fa894920696824285fbf2287da5c2c991db5409e9c3e509d26245b1.jpg)

![77624b80591273eb477953ca4cc927c6af2a0c5e44e81713354adb1feb634645.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/77624b80591273eb477953ca4cc927c6af2a0c5e44e81713354adb1feb634645.jpg)

![78262aee9f9c6621cc1ec5311ad18c9539168fcad76382e4f47182018d7cc81b.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/78262aee9f9c6621cc1ec5311ad18c9539168fcad76382e4f47182018d7cc81b.jpg)

![cfc4d5838c97176b6ab9f147c5036658dcbc64eb922d4b040ea0a450add862a6.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/images/cfc4d5838c97176b6ab9f147c5036658dcbc64eb922d4b040ea0a450add862a6.jpg)

### Tables

![3610b1a3f2fc1658f8f683e297a2d05a34128076a3f95bf144ffd1e8ca732d34.jpg](../icml_results/1520_Geometric%20Resampling%20in%20Nearly%20Linear%20Time%20for%20Follow-the-Perturbed-Leader%20with%20Best-of-Both-Worlds%20/tables/3610b1a3f2fc1658f8f683e297a2d05a34128076a3f95bf144ffd1e8ca732d34.jpg)

## Compositional Risk Minimization


### Images

![06a349e8fb7a3249f8737ec8f20da6a833eefab70fa16ad5a34839738d210873.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/06a349e8fb7a3249f8737ec8f20da6a833eefab70fa16ad5a34839738d210873.jpg)

![0beeb69e9195320f94ffeb9adec5d4c9e2fe7710d4f5dfe3af5f984e9fe10281.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/0beeb69e9195320f94ffeb9adec5d4c9e2fe7710d4f5dfe3af5f984e9fe10281.jpg)

![0f6c2e03cde91c98df96393c01bc877521a1b5a3311ef3058a0922a844a53cf8.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/0f6c2e03cde91c98df96393c01bc877521a1b5a3311ef3058a0922a844a53cf8.jpg)

![22db090c60477a53713829858c320a34f0ff3119acf70dc550d2c6e1f645475b.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/22db090c60477a53713829858c320a34f0ff3119acf70dc550d2c6e1f645475b.jpg)

![4752dbde6be679459fd9291ded5d09c09ae2e76d3cae59a53c8db745fbe87fcd.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/4752dbde6be679459fd9291ded5d09c09ae2e76d3cae59a53c8db745fbe87fcd.jpg)

![5d7bac68662cd11db26c26afb5a4e1db1dc0d51bf7bf9c0c92aef0d61dd05d06.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/5d7bac68662cd11db26c26afb5a4e1db1dc0d51bf7bf9c0c92aef0d61dd05d06.jpg)

![6badf66f873c852c92486533472aedfa9efe10dcf0e00f52d6c9bc12ac2733fd.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/6badf66f873c852c92486533472aedfa9efe10dcf0e00f52d6c9bc12ac2733fd.jpg)

![74fbae417ce3f9c1f9d7bc176e808715798f314ffc52b74b7ef2dc1982b12c71.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/74fbae417ce3f9c1f9d7bc176e808715798f314ffc52b74b7ef2dc1982b12c71.jpg)

![77557ca5336db8cbd727844f3c2a7369834f85a8822f6529a75871fee212e080.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/77557ca5336db8cbd727844f3c2a7369834f85a8822f6529a75871fee212e080.jpg)

![a0bc32802e54312f0908b609acdd2600fa983cc15682d0bd46520e0b2d4dfd05.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/a0bc32802e54312f0908b609acdd2600fa983cc15682d0bd46520e0b2d4dfd05.jpg)

![e956aedc72bcb48c35c3db98a16dde243abf42ead7acb4adf6d1df8c5edfdeaa.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/images/e956aedc72bcb48c35c3db98a16dde243abf42ead7acb4adf6d1df8c5edfdeaa.jpg)

### Tables

![2c8483467e159b22cb53a4e732f567b198dc41237fb601f4f4f17961be61dc4d.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/2c8483467e159b22cb53a4e732f567b198dc41237fb601f4f4f17961be61dc4d.jpg)

![3a107cdf65166f6fb2ca8529cb6990afa1720afb8a3bb2d22f4ab9e09ad3a510.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/3a107cdf65166f6fb2ca8529cb6990afa1720afb8a3bb2d22f4ab9e09ad3a510.jpg)

![47936eba44732313803f3997246ee8cb31ae9b607072f799108cddd2825fdd4b.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/47936eba44732313803f3997246ee8cb31ae9b607072f799108cddd2825fdd4b.jpg)

![5b475bedebdca6776366f398c5ffa853c81ed32c90fb4d6b732f6db8a45311ed.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/5b475bedebdca6776366f398c5ffa853c81ed32c90fb4d6b732f6db8a45311ed.jpg)

![5c4e3e941bb3f6393f8026d93d9e9e2be09d4da62dfb39e8b077d3e85355b6a7.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/5c4e3e941bb3f6393f8026d93d9e9e2be09d4da62dfb39e8b077d3e85355b6a7.jpg)

![86cb7da99af0dae8d61202a9a355272e51665f7d2291afabff02380ca82d0977.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/86cb7da99af0dae8d61202a9a355272e51665f7d2291afabff02380ca82d0977.jpg)

![9cb416295ac698e4116d22668a8b0a68acd215c12ec129cda49d684efcaa651c.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/9cb416295ac698e4116d22668a8b0a68acd215c12ec129cda49d684efcaa651c.jpg)

![9e2e9a0f733d1adf2ca1bb01df4abf4adf2f24242d58245dfeb52b97a2030dbb.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/9e2e9a0f733d1adf2ca1bb01df4abf4adf2f24242d58245dfeb52b97a2030dbb.jpg)

![a7943ff906788d86c21ba2f01a4d300bec7cdd2f1624f944858de820d39b0950.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/a7943ff906788d86c21ba2f01a4d300bec7cdd2f1624f944858de820d39b0950.jpg)

![b94b5fb70c61f414678fb9b06c53acfdb7392c18b8b83a16ede323b84d97c55f.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/b94b5fb70c61f414678fb9b06c53acfdb7392c18b8b83a16ede323b84d97c55f.jpg)

![bd3aec617d876d2e3a413e7d16f3a4c0759c10588dc2a05b148e5142313b9a04.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/bd3aec617d876d2e3a413e7d16f3a4c0759c10588dc2a05b148e5142313b9a04.jpg)

![dbe0bf127f50c2a04df2e949d33bc98977e68302358ca7a09af0388f98afae82.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/dbe0bf127f50c2a04df2e949d33bc98977e68302358ca7a09af0388f98afae82.jpg)

![f55cdf2b4813c1f90b02af17593140a592ba0737e09692297ed7d0fec2bf8728.jpg](../icml_results/1521_Compositional%20Risk%20Minimization/tables/f55cdf2b4813c1f90b02af17593140a592ba0737e09692297ed7d0fec2bf8728.jpg)

## Continuous Visual Autoregressive Generation via Score Maximization


### Images

![1553b4afcea94159700c364d33de94480068b5c2f42af89093b4e88d3d119c4e.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/1553b4afcea94159700c364d33de94480068b5c2f42af89093b4e88d3d119c4e.jpg)

![35aa195b8aaa32f1173369b8db2078da1beb0641ec0788c30dcca9976e0981fc.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/35aa195b8aaa32f1173369b8db2078da1beb0641ec0788c30dcca9976e0981fc.jpg)

![47b09fa2d050762a9fdb428eb6659c03e28355938ed1735689e35df287d981dd.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/47b09fa2d050762a9fdb428eb6659c03e28355938ed1735689e35df287d981dd.jpg)

![7c78855050f153faebb8e4e85838ea9495d629c696e3f5047be3c2271c5f4a2c.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/7c78855050f153faebb8e4e85838ea9495d629c696e3f5047be3c2271c5f4a2c.jpg)

![aa2ab47e02a3fcc650047cb4a7a63326448af6253f99e214cfc5970f8c1f4b32.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/aa2ab47e02a3fcc650047cb4a7a63326448af6253f99e214cfc5970f8c1f4b32.jpg)

![bdfc1e70fee2f8a4f1b201126c3c3b0f386b86db570b9e439731a7c7f755aaf1.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/bdfc1e70fee2f8a4f1b201126c3c3b0f386b86db570b9e439731a7c7f755aaf1.jpg)

![e57374d8879f3d1ef01c276fc5adf6e6db429cce8b9458cc665af9d8f8303f21.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/e57374d8879f3d1ef01c276fc5adf6e6db429cce8b9458cc665af9d8f8303f21.jpg)

![e9b27c701fc043c2a726a9e88962985568554991158e0f487818f41041eee942.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/images/e9b27c701fc043c2a726a9e88962985568554991158e0f487818f41041eee942.jpg)

### Tables

![168b9ed699971d0493302eb3dc54b2d27e1f63ab80a48ea9df2f8c99a6143bfc.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/tables/168b9ed699971d0493302eb3dc54b2d27e1f63ab80a48ea9df2f8c99a6143bfc.jpg)

![1756bec68c9b8b46d05185b685dc26a2557057e6b1cfbb3e5e01258c1687ef97.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/tables/1756bec68c9b8b46d05185b685dc26a2557057e6b1cfbb3e5e01258c1687ef97.jpg)

![2ccbfc5e475d473398e2cb1f531c98c4bbaa48b9ed44129b2e5e94737041dd43.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/tables/2ccbfc5e475d473398e2cb1f531c98c4bbaa48b9ed44129b2e5e94737041dd43.jpg)

![83e92cd2bc2b8daadeb3684e9ae376d6e1d308d52077c0663f9471b56805d124.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/tables/83e92cd2bc2b8daadeb3684e9ae376d6e1d308d52077c0663f9471b56805d124.jpg)

![a7a4de73a229c1821ef9b684ae08d5fbf5cb3ba5c7756c9f67f2f89b25a3adee.jpg](../icml_results/1522_Continuous%20Visual%20Autoregressive%20Generation%20via%20Score%20Maximization/tables/a7a4de73a229c1821ef9b684ae08d5fbf5cb3ba5c7756c9f67f2f89b25a3adee.jpg)

## Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors


### Images

![34853f325564982ec3266b302f2cd973813d1e2f698a4ff301951c5e4d5a6aae.jpg](../icml_results/1523_Learning-Augmented%20Algorithms%20for%20MTS%20with%20Bandit%20Access%20to%20Multiple%20Predictors/images/34853f325564982ec3266b302f2cd973813d1e2f698a4ff301951c5e4d5a6aae.jpg)

### Tables

![15a4a93b098990736cb81e61eab002ea8f70f5aedbf5b90c6489c3175832c57c.jpg](../icml_results/1523_Learning-Augmented%20Algorithms%20for%20MTS%20with%20Bandit%20Access%20to%20Multiple%20Predictors/tables/15a4a93b098990736cb81e61eab002ea8f70f5aedbf5b90c6489c3175832c57c.jpg)

![61f2e2abca8fb72ca54eed47c12df39af6b0157ef46de3dc02db3874b0539fcb.jpg](../icml_results/1523_Learning-Augmented%20Algorithms%20for%20MTS%20with%20Bandit%20Access%20to%20Multiple%20Predictors/tables/61f2e2abca8fb72ca54eed47c12df39af6b0157ef46de3dc02db3874b0539fcb.jpg)

![a23559d6ba675c32fbca15dd3fe098fab0cebfb53a5fa0557412c6825e2999ba.jpg](../icml_results/1523_Learning-Augmented%20Algorithms%20for%20MTS%20with%20Bandit%20Access%20to%20Multiple%20Predictors/tables/a23559d6ba675c32fbca15dd3fe098fab0cebfb53a5fa0557412c6825e2999ba.jpg)

![c0dbb4ff85bc71bfb872f94d8912af9618a5c2291b1ddc9a211d980b56cd2b6b.jpg](../icml_results/1523_Learning-Augmented%20Algorithms%20for%20MTS%20with%20Bandit%20Access%20to%20Multiple%20Predictors/tables/c0dbb4ff85bc71bfb872f94d8912af9618a5c2291b1ddc9a211d980b56cd2b6b.jpg)

![f9fd3d9554fd1dde88492d5da5d2fa1fd0285a4eda283b0ea8e100284a8a7a28.jpg](../icml_results/1523_Learning-Augmented%20Algorithms%20for%20MTS%20with%20Bandit%20Access%20to%20Multiple%20Predictors/tables/f9fd3d9554fd1dde88492d5da5d2fa1fd0285a4eda283b0ea8e100284a8a7a28.jpg)

## On Zero-Initialized Attention: Optimal Prompt and Gating Factor Estimation


### Images

![385e50c503a318df6a4398ab8e7346c1be48450ad954f6217350a82fa394a96e.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/images/385e50c503a318df6a4398ab8e7346c1be48450ad954f6217350a82fa394a96e.jpg)

![73246310313015828b9ec97579ded6ba22f4ef023359d70edef9dbdf707ea4cd.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/images/73246310313015828b9ec97579ded6ba22f4ef023359d70edef9dbdf707ea4cd.jpg)

![7bba25836335b5b30598e229fa9aca0f71d83a2dc2eaf14408002ac258a6ac30.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/images/7bba25836335b5b30598e229fa9aca0f71d83a2dc2eaf14408002ac258a6ac30.jpg)

![8de8d28182d61f36751ca0e153bf55e4181753577a55d4927d1613b4ce973f40.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/images/8de8d28182d61f36751ca0e153bf55e4181753577a55d4927d1613b4ce973f40.jpg)

![d6819e66c9fa03ee7080ff21ab053cf56f1e50955bc5b375c1b9b0dbf1eabc4d.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/images/d6819e66c9fa03ee7080ff21ab053cf56f1e50955bc5b375c1b9b0dbf1eabc4d.jpg)

### Tables

![0c51b86e415e55a8282f6c56a422e8b8b0e9af80bee05d861ce3cdafba3f9214.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/tables/0c51b86e415e55a8282f6c56a422e8b8b0e9af80bee05d861ce3cdafba3f9214.jpg)

![29ae56f8f53c834527abd35ede61d1e3e056e24fdaec9e4aaf66e59b0800d6b4.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/tables/29ae56f8f53c834527abd35ede61d1e3e056e24fdaec9e4aaf66e59b0800d6b4.jpg)

![2cf41adf6be7353fcb9e837ab50a485ec779e6fced1e334bee9f0bcac3ebc89d.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/tables/2cf41adf6be7353fcb9e837ab50a485ec779e6fced1e334bee9f0bcac3ebc89d.jpg)

![4a836d2204d2966f99a7e5882f84d93a750a5539de59384290fce6e76d1fde36.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/tables/4a836d2204d2966f99a7e5882f84d93a750a5539de59384290fce6e76d1fde36.jpg)

![5f249ef348261bc619ee769179cca1b7b708615cd61d771aa547635fca17c186.jpg](../icml_results/1524_On%20Zero-Initialized%20Attention_%20Optimal%20Prompt%20and%20Gating%20Factor%20Estimation/tables/5f249ef348261bc619ee769179cca1b7b708615cd61d771aa547635fca17c186.jpg)

## The Surprising Effectiveness of Test-Time Training for Few-Shot Learning


### Images

![00dec7a6080f5b0f23de269e92d59db36d988c85c1dfec4501994171e62cc020.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/00dec7a6080f5b0f23de269e92d59db36d988c85c1dfec4501994171e62cc020.jpg)

![067af84a1cc0df36e0ab841e5e2404409d8bc20a0536a2ba27141538d9a8a3f8.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/067af84a1cc0df36e0ab841e5e2404409d8bc20a0536a2ba27141538d9a8a3f8.jpg)

![20138b3429c5aba61a384d59b6386248ac288ff42aed91a4e89c2bf26582403c.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/20138b3429c5aba61a384d59b6386248ac288ff42aed91a4e89c2bf26582403c.jpg)

![2d73e6344811b704165502950f05821c362e353aa5bfa916ec193336e31d3291.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/2d73e6344811b704165502950f05821c362e353aa5bfa916ec193336e31d3291.jpg)

![33e82ca438022a23d03f7bb916a2658bd0ddd9f59a909367de18ff33e542ba1c.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/33e82ca438022a23d03f7bb916a2658bd0ddd9f59a909367de18ff33e542ba1c.jpg)

![45efef2e9bf566241f46ae9ee5f6c7064c47bd62125d65bcbcde8a4dce5e59e5.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/45efef2e9bf566241f46ae9ee5f6c7064c47bd62125d65bcbcde8a4dce5e59e5.jpg)

![5e5eec6f7f349f4c2e713c73701cb4ea7b7880dd32ce3af63abfeac27f6cafeb.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/5e5eec6f7f349f4c2e713c73701cb4ea7b7880dd32ce3af63abfeac27f6cafeb.jpg)

![721705a2613a42bbf9eaa6ecad6314731e2dfe6e875bdc105ff86de8359d80c6.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/721705a2613a42bbf9eaa6ecad6314731e2dfe6e875bdc105ff86de8359d80c6.jpg)

![8e26cfdbd1d1146e4906c45ce1622f0a51f480e75c59148533fdd0a39f9bfefd.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/8e26cfdbd1d1146e4906c45ce1622f0a51f480e75c59148533fdd0a39f9bfefd.jpg)

![a3ecaebc33c4e659d445c2448494ee2e7329fa370261e621c377a9fd10c3a26e.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/a3ecaebc33c4e659d445c2448494ee2e7329fa370261e621c377a9fd10c3a26e.jpg)

![ac11315ade1b58782ea9c9c7092153fc42af737a881084c8ef9052668e339990.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/ac11315ade1b58782ea9c9c7092153fc42af737a881084c8ef9052668e339990.jpg)

![c7729b5ccc5355175d1c3d3b2e6a22e89b95cf61a7a6a5c1d7417ade216333e5.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/c7729b5ccc5355175d1c3d3b2e6a22e89b95cf61a7a6a5c1d7417ade216333e5.jpg)

![c99496d44c259ffa751ef8434aeb42a331c0604413b76941f55aafe4b01ff67e.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/c99496d44c259ffa751ef8434aeb42a331c0604413b76941f55aafe4b01ff67e.jpg)

![d8edce6c7dd2a5eb414026f541f6f4b65fbc24d88ab2258577d57f87cf9e0f66.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/d8edce6c7dd2a5eb414026f541f6f4b65fbc24d88ab2258577d57f87cf9e0f66.jpg)

![e7475abdfd290234d3f5047af60b71313edf9512e502c3218101fe7959abedc6.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/e7475abdfd290234d3f5047af60b71313edf9512e502c3218101fe7959abedc6.jpg)

![e9ed738b64b6ba1a158825d0bc71f3e0b326799ae6671f27f4365d2c5717932b.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/e9ed738b64b6ba1a158825d0bc71f3e0b326799ae6671f27f4365d2c5717932b.jpg)

![ee1cfb422a11237c05ce8d30a40c873a4386b437468e1366d60661cfcbe48e83.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/images/ee1cfb422a11237c05ce8d30a40c873a4386b437468e1366d60661cfcbe48e83.jpg)

### Tables

![15e47b9757db52c73cdfdd9b9205bfe641a64a3bf4e2bfd0e770b38dddfccf98.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/15e47b9757db52c73cdfdd9b9205bfe641a64a3bf4e2bfd0e770b38dddfccf98.jpg)

![3eaeccb815876c64a831afbf1c37e29d3095c17cd8d4151eed0f97d4f836f90d.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/3eaeccb815876c64a831afbf1c37e29d3095c17cd8d4151eed0f97d4f836f90d.jpg)

![4ee5a87db15839c10fe592d22fd205d0b46dc96cde185411853594de3f205a94.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/4ee5a87db15839c10fe592d22fd205d0b46dc96cde185411853594de3f205a94.jpg)

![57904cfb16b47f1d83045c6c13b5d26bf3f74ac97073bb408b11e1e009912d78.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/57904cfb16b47f1d83045c6c13b5d26bf3f74ac97073bb408b11e1e009912d78.jpg)

![bea179f2f1f17b52282c3d7b7891e22f67c1e99d2bcffcefd1f957e84b07cf79.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/bea179f2f1f17b52282c3d7b7891e22f67c1e99d2bcffcefd1f957e84b07cf79.jpg)

![c855bd222b42d4225f0e274cd95f57cdaae400371fec50efc12c03dd3da22366.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/c855bd222b42d4225f0e274cd95f57cdaae400371fec50efc12c03dd3da22366.jpg)

![c8dcd6ebc85023d04cc095094d4ce79b3a1a39c6022240d9bc3e9b2fd121047e.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/c8dcd6ebc85023d04cc095094d4ce79b3a1a39c6022240d9bc3e9b2fd121047e.jpg)

![cbe4ba6a1a0edde1c985f8d3eddb2559358dcbe6dd006e0bfafcbde8de4c7fb3.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/cbe4ba6a1a0edde1c985f8d3eddb2559358dcbe6dd006e0bfafcbde8de4c7fb3.jpg)

![d89af7eeb4f5e3db7f48575478543c444a15e3b14621c2ab1d65a95ab2e00d48.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/d89af7eeb4f5e3db7f48575478543c444a15e3b14621c2ab1d65a95ab2e00d48.jpg)

![f7bc4ee9d88549781de526efc1870f8f37772e457690654db051463b1712b1e2.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/f7bc4ee9d88549781de526efc1870f8f37772e457690654db051463b1712b1e2.jpg)

![faa745f6bb6223ccda383a493034485fba6d98f8cc5682933786ab66775e3c2f.jpg](../icml_results/1525_The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning/tables/faa745f6bb6223ccda383a493034485fba6d98f8cc5682933786ab66775e3c2f.jpg)

## (How) Can Transformers Predict Pseudo-Random Numbers?


### Images

![10035c1b9a689ffbfb9a7c3abf94f20d2f393b3635bb65ac1ee76f3fa1dea756.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/10035c1b9a689ffbfb9a7c3abf94f20d2f393b3635bb65ac1ee76f3fa1dea756.jpg)

![266c08dfb35560e38d931f7fbf5f446d2b1e7c88ccfab6c37c51841b12d5c8b1.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/266c08dfb35560e38d931f7fbf5f446d2b1e7c88ccfab6c37c51841b12d5c8b1.jpg)

![275e2dae58a5c59aee35306eba9dfbdfd056aaecc1c01328795adb7d072977f8.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/275e2dae58a5c59aee35306eba9dfbdfd056aaecc1c01328795adb7d072977f8.jpg)

![28511bb9e3590359a3eb0759c1dee0d364aaf98bfcdda61d7cace80c7a0c9a80.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/28511bb9e3590359a3eb0759c1dee0d364aaf98bfcdda61d7cace80c7a0c9a80.jpg)

![2ae44e47a047d29c80d36ba0ff2d96a80259d90dfd97a568b08dce3fb3c8e6ce.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/2ae44e47a047d29c80d36ba0ff2d96a80259d90dfd97a568b08dce3fb3c8e6ce.jpg)

![39017aacc48b679ae842c5cb988215aa76c6291142586289a2a8b91f30fb55d6.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/39017aacc48b679ae842c5cb988215aa76c6291142586289a2a8b91f30fb55d6.jpg)

![4566179597938a86547a31b78084383f6a5c0ae8f8c40e5deefc5ee3792b6291.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/4566179597938a86547a31b78084383f6a5c0ae8f8c40e5deefc5ee3792b6291.jpg)

![4a5d56308a5ebb23d883e6204114450d990f6be969310292c033c6c8a6d2891e.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/4a5d56308a5ebb23d883e6204114450d990f6be969310292c033c6c8a6d2891e.jpg)

![548540d2038f6b8fb8fa1d55a8a5ee9c4851afd1401391c49a1c632491c0c8f3.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/548540d2038f6b8fb8fa1d55a8a5ee9c4851afd1401391c49a1c632491c0c8f3.jpg)

![58df909e967fa4d60ee5f27c1e3b5c1237cb02dac6fbe58f90f252fe5217c6e4.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/58df909e967fa4d60ee5f27c1e3b5c1237cb02dac6fbe58f90f252fe5217c6e4.jpg)

![5a711e8503a9914a8d27c3a656f68668d503520d8f5414a1640c081593c43235.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/5a711e8503a9914a8d27c3a656f68668d503520d8f5414a1640c081593c43235.jpg)

![61439abf4dcd4b1e5054cabc32a691a26e7531d1003d2a6e1b521a000cf9f657.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/61439abf4dcd4b1e5054cabc32a691a26e7531d1003d2a6e1b521a000cf9f657.jpg)

![68b8c6bb58af323c4bab085e5e53c3a54a71b68f954a493d2a896fb5b2d982fe.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/68b8c6bb58af323c4bab085e5e53c3a54a71b68f954a493d2a896fb5b2d982fe.jpg)

![69eb3e3c6a5e5c0bd820c1003d13409a8a28bc482a7e51e12ade314c29bb80f2.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/69eb3e3c6a5e5c0bd820c1003d13409a8a28bc482a7e51e12ade314c29bb80f2.jpg)

![73ffe1719900d42cd9e13b8f84130557ae7133bb74e5a05f8b884995d63a7bee.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/73ffe1719900d42cd9e13b8f84130557ae7133bb74e5a05f8b884995d63a7bee.jpg)

![776a5117f72c2001131cda017396b9ac36338e0eed7edf2110472ee0490ecb94.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/776a5117f72c2001131cda017396b9ac36338e0eed7edf2110472ee0490ecb94.jpg)

![78ffc95624b7786e967c0281e07a257f174e0d5469dce5ee80c6531fa9f29d39.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/78ffc95624b7786e967c0281e07a257f174e0d5469dce5ee80c6531fa9f29d39.jpg)

![7bbdbe87e208cdcb7e9471149e7c5a9accfe38bded590ca746ee1698c91b2f00.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/7bbdbe87e208cdcb7e9471149e7c5a9accfe38bded590ca746ee1698c91b2f00.jpg)

![7e3e884beda11fe75e3772867578c3b7c517304f609c31eabf4c79b881a821f7.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/7e3e884beda11fe75e3772867578c3b7c517304f609c31eabf4c79b881a821f7.jpg)

![85cd8e570170238c2262f8b569d2ef881987c9280c70a48fde7b34b5b62a1558.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/85cd8e570170238c2262f8b569d2ef881987c9280c70a48fde7b34b5b62a1558.jpg)

![86805d37f606be6757da31f98b7bd0e3e8adda88543f7877b3553fec3fd05ae4.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/86805d37f606be6757da31f98b7bd0e3e8adda88543f7877b3553fec3fd05ae4.jpg)

![8d147884d867d5cec6ab370b2bbbd4f9e7d178ba12325ea8f056af4315f5b0b7.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/8d147884d867d5cec6ab370b2bbbd4f9e7d178ba12325ea8f056af4315f5b0b7.jpg)

![9257496a31dd3bb3e628265524bd0ca562615b0171e65a5d66a327a76a8781a3.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/9257496a31dd3bb3e628265524bd0ca562615b0171e65a5d66a327a76a8781a3.jpg)

![9f106b5d9efef9d2938d64e85d1de2c8f4d5b1a6ec577667e39868343f1dc6c6.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/9f106b5d9efef9d2938d64e85d1de2c8f4d5b1a6ec577667e39868343f1dc6c6.jpg)

![a8771eee95a0b96bee9d3de5b85f011c8d7b46ef47b123aa23cca47410d8f653.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/a8771eee95a0b96bee9d3de5b85f011c8d7b46ef47b123aa23cca47410d8f653.jpg)

![b9469d5b742c4d9176b8160b646ba59fb6ccf89bb96fb5f10d9448642cb2907b.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/b9469d5b742c4d9176b8160b646ba59fb6ccf89bb96fb5f10d9448642cb2907b.jpg)

![bd9b6121597e5440e86a5c816f3ae43a6521301141222e23591fecdfe64d1445.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/bd9b6121597e5440e86a5c816f3ae43a6521301141222e23591fecdfe64d1445.jpg)

![d2a21c26a6635116a37ff73415ab991859ff3b597f975052d4fdb544667f757a.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/d2a21c26a6635116a37ff73415ab991859ff3b597f975052d4fdb544667f757a.jpg)

![d429dc6b9d0d614b9e2cb0d1884d2a6299b979e57babb5d554748bd9f2d5b81e.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/d429dc6b9d0d614b9e2cb0d1884d2a6299b979e57babb5d554748bd9f2d5b81e.jpg)

![daa9692e78a1b42bdf565e6db24e7ff6a5e406e448cdaa542aaf11060a49df2e.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/daa9692e78a1b42bdf565e6db24e7ff6a5e406e448cdaa542aaf11060a49df2e.jpg)

![dce121a3e50e79d28b9fb72710622ad747452baf981feded97d0eb607673a07c.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/dce121a3e50e79d28b9fb72710622ad747452baf981feded97d0eb607673a07c.jpg)

![ef7ce4e694e2ec7cae44548e9bbb58625f0a03c7d69c08d6d696af61cac2658f.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/ef7ce4e694e2ec7cae44548e9bbb58625f0a03c7d69c08d6d696af61cac2658f.jpg)

![f0423274d38bdee7345e92a8dbe9fb53a15ec90196dacec007084a812423e342.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/f0423274d38bdee7345e92a8dbe9fb53a15ec90196dacec007084a812423e342.jpg)

![f16e58498a007a3ae06405876a50de9c1eefff1710f3384b47790f88efdd4200.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/f16e58498a007a3ae06405876a50de9c1eefff1710f3384b47790f88efdd4200.jpg)

![f7aea5bae44483850755f85489613d73e2f0a827693fb2b44ba5eb81e156c9de.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/f7aea5bae44483850755f85489613d73e2f0a827693fb2b44ba5eb81e156c9de.jpg)

![feeb513c0a05d4757c32e8eac5a834c1f51525d43cf96a8ad04e1928a41918d1.jpg](../icml_results/1526_%28How) Can Transformers Predict Pseudo-Random Numbers_/images/feeb513c0a05d4757c32e8eac5a834c1f51525d43cf96a8ad04e1928a41918d1.jpg)

## Time to Spike? Understanding the Representational Power of Spiking Neural Networks in Discrete Time


### Images

![010338fb240bc2a0fd17de86e12f68908372b26bc7c31a257fd446f362dff535.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/010338fb240bc2a0fd17de86e12f68908372b26bc7c31a257fd446f362dff535.jpg)

![17c65b11befbf3ee0160d9b969d5a4b2afe059a225754e1fbe904370a2c4268a.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/17c65b11befbf3ee0160d9b969d5a4b2afe059a225754e1fbe904370a2c4268a.jpg)

![403cd600a18741decccd82445d8edc51bf6db85fad0e1cae72102accae9952c4.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/403cd600a18741decccd82445d8edc51bf6db85fad0e1cae72102accae9952c4.jpg)

![4c589a7ef14218840985ee9542a406e77104150d1b3273e74ee771d84dec99f6.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/4c589a7ef14218840985ee9542a406e77104150d1b3273e74ee771d84dec99f6.jpg)

![5830149b9e269a993f161e0f8ce38a0d1a2a4501a5ae805a139de8b25f09b13e.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/5830149b9e269a993f161e0f8ce38a0d1a2a4501a5ae805a139de8b25f09b13e.jpg)

![63bc2ed4991da1e4ee5f29db134f3853a2e3e3f6bd4dcb0d314a1a0f72689ad1.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/63bc2ed4991da1e4ee5f29db134f3853a2e3e3f6bd4dcb0d314a1a0f72689ad1.jpg)

![769a36d3b610e519150a2f9c2969bf588b4230d4b437ef1fea33cefd7f67d126.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/769a36d3b610e519150a2f9c2969bf588b4230d4b437ef1fea33cefd7f67d126.jpg)

![82a0aa6a51a4101061cdb1e69a0d47bd33ce6d8423eee5110fdf441492f0a337.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/82a0aa6a51a4101061cdb1e69a0d47bd33ce6d8423eee5110fdf441492f0a337.jpg)

![8dd3e4cf10d820cbc7bc7cf372e1adbf8a0d58b2d6884d487258d676019b8d90.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/8dd3e4cf10d820cbc7bc7cf372e1adbf8a0d58b2d6884d487258d676019b8d90.jpg)

![b232cc7388f3cfad2522075827b1696aef9a9a09c6aee75115903634c182d5de.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/b232cc7388f3cfad2522075827b1696aef9a9a09c6aee75115903634c182d5de.jpg)

![c2393ae368a93af6ee50e07247e7c9e24e29364e31372e8de824f59426e6ee46.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/c2393ae368a93af6ee50e07247e7c9e24e29364e31372e8de824f59426e6ee46.jpg)

![ff673363656b2e61f5ae4740a86b378381eeb1ed44eb73f3b939909192114570.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/images/ff673363656b2e61f5ae4740a86b378381eeb1ed44eb73f3b939909192114570.jpg)

### Tables

![2d91cf6ec639b13295008df57d16d5673276de920568e46dbbd214cd0a0d933b.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/tables/2d91cf6ec639b13295008df57d16d5673276de920568e46dbbd214cd0a0d933b.jpg)

![b6431ffa34e061acc55e1c9846edc811ce7f394dca2c25cc1ae95e341429259b.jpg](../icml_results/1527_Time%20to%20Spike_%20Understanding%20the%20Representational%20Power%20of%20Spiking%20Neural%20Networks%20in%20Discrete%20Time/tables/b6431ffa34e061acc55e1c9846edc811ce7f394dca2c25cc1ae95e341429259b.jpg)

## Discriminative Policy Optimization for Token-Level Reward Models


### Images

![14c85ff4cad9e7cf6f4988e53bb329568742c3e657ee22d5021d3eee36411bc4.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/14c85ff4cad9e7cf6f4988e53bb329568742c3e657ee22d5021d3eee36411bc4.jpg)

![339d5962ca747214078bc30b97e231575c1fe7db80d60a253a5a1b7283c8221b.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/339d5962ca747214078bc30b97e231575c1fe7db80d60a253a5a1b7283c8221b.jpg)

![59692eb2f8ba0a6618b2a2c723da07d7422e535f9d2fe3f525606c89bccafd7c.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/59692eb2f8ba0a6618b2a2c723da07d7422e535f9d2fe3f525606c89bccafd7c.jpg)

![6b4bc4935fc8a06571d0580931b10b452552c4b5a7b074a6142b0904dd349946.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/6b4bc4935fc8a06571d0580931b10b452552c4b5a7b074a6142b0904dd349946.jpg)

![6e80cca1446da38bb0a3d99adc9ff9f239afc3c96cd27a6ccdfd695e4a03dd26.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/6e80cca1446da38bb0a3d99adc9ff9f239afc3c96cd27a6ccdfd695e4a03dd26.jpg)

![737ee5ec9765e1c9b2bb7d405965ea0563180a1173d42091b465e5fac1f59e41.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/737ee5ec9765e1c9b2bb7d405965ea0563180a1173d42091b465e5fac1f59e41.jpg)

![9888ea4e4cb9f702f120c7a80a6d2cd196e5b09fb7b269d5cf74c0e4252cf172.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/9888ea4e4cb9f702f120c7a80a6d2cd196e5b09fb7b269d5cf74c0e4252cf172.jpg)

![9efd5db4ae55d85e9e5bc9c0e55a2c88f85bb7713b770f95223afe0267bcd2de.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/9efd5db4ae55d85e9e5bc9c0e55a2c88f85bb7713b770f95223afe0267bcd2de.jpg)

![ab27fd8484a5527cbec0d0fd0d7eda15d5d1f17bf7c3a1dc86c12e39e15478ba.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/ab27fd8484a5527cbec0d0fd0d7eda15d5d1f17bf7c3a1dc86c12e39e15478ba.jpg)

![b3ae9a1bcffb74390cb541e6f95c6d5f83b68c8defbbc1cbcedb8a3a0f4a6ed7.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/b3ae9a1bcffb74390cb541e6f95c6d5f83b68c8defbbc1cbcedb8a3a0f4a6ed7.jpg)

![b5b767443bd763d0978b9b4a1bb7cd24dcec5437d132876b43711570fd3cdd52.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/b5b767443bd763d0978b9b4a1bb7cd24dcec5437d132876b43711570fd3cdd52.jpg)

![c25e839d423dd96b3f7e43dd03baaf5140981744d650eae5405ec85fe1f01fc6.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/c25e839d423dd96b3f7e43dd03baaf5140981744d650eae5405ec85fe1f01fc6.jpg)

![d6e5a95d51e8f42216562c29238a468fdf92b277b00b1f0eae1fe064686d7921.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/d6e5a95d51e8f42216562c29238a468fdf92b277b00b1f0eae1fe064686d7921.jpg)

![e067e9bc57fdebd769eebcc80dcd81104dd36d0f6abe718e2ae1c0ad1f12d2b5.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/e067e9bc57fdebd769eebcc80dcd81104dd36d0f6abe718e2ae1c0ad1f12d2b5.jpg)

![e85bf9c858aec06e57b8a4ecf06bfc33509c00a6f148038b0e6af4fc6cd69e5e.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/e85bf9c858aec06e57b8a4ecf06bfc33509c00a6f148038b0e6af4fc6cd69e5e.jpg)

![eea64c6aad028f9426200d43b941c995ddc2e6b20a798408f2dbaad7ce3e4358.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/images/eea64c6aad028f9426200d43b941c995ddc2e6b20a798408f2dbaad7ce3e4358.jpg)

### Tables

![17a3740f2927c5c5abc7ab163f82abac9bf34ac27bfd46e818c64d8b7c1b264b.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/tables/17a3740f2927c5c5abc7ab163f82abac9bf34ac27bfd46e818c64d8b7c1b264b.jpg)

![4ccfe916de510fef3d27cddd7d63b52e0e48930fa66bd57f1066288deacf4619.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/tables/4ccfe916de510fef3d27cddd7d63b52e0e48930fa66bd57f1066288deacf4619.jpg)

![5bbb8a099e35b7052b33c4cf862cfe6b23dc1d2e77ba355a72db8505f984c64c.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/tables/5bbb8a099e35b7052b33c4cf862cfe6b23dc1d2e77ba355a72db8505f984c64c.jpg)

![5f200ce7da0e6aa44757190a88cd016cdec5543559b4645f813042ed595c1d52.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/tables/5f200ce7da0e6aa44757190a88cd016cdec5543559b4645f813042ed595c1d52.jpg)

![88d6a0042ea68ae02659295ac5f7ecb2eacf09200a9fcfb98af40c758eb6dac6.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/tables/88d6a0042ea68ae02659295ac5f7ecb2eacf09200a9fcfb98af40c758eb6dac6.jpg)

![ada23def1b73e90ada6576d6e1b8e0b2efc29b74cefd3dd886107674afc3533d.jpg](../icml_results/1528_Discriminative%20Policy%20Optimization%20for%20Token-Level%20Reward%20Models/tables/ada23def1b73e90ada6576d6e1b8e0b2efc29b74cefd3dd886107674afc3533d.jpg)

## The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning


### Images

![04dca6487b767e0a26a167a7292166ea97095d0389fc66a78ae2b8811444036e.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/images/04dca6487b767e0a26a167a7292166ea97095d0389fc66a78ae2b8811444036e.jpg)

![1832b85aa82025eff5aa882d53e747d480ffcc3d5b3fff61240b6d6b8462f203.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/images/1832b85aa82025eff5aa882d53e747d480ffcc3d5b3fff61240b6d6b8462f203.jpg)

![25593f6a079734ad4c9a83fc5b9c209490454027411a4930516a5b7d106db704.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/images/25593f6a079734ad4c9a83fc5b9c209490454027411a4930516a5b7d106db704.jpg)

![4a6085ab888853627055dc408e4f215e3a912c4ed2b67be6439bc08a2dd15bce.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/images/4a6085ab888853627055dc408e4f215e3a912c4ed2b67be6439bc08a2dd15bce.jpg)

![58e171320102d975f0b1e1225cd34f0f3cd5e17c613fc70c9f7bfdf4b59b4400.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/images/58e171320102d975f0b1e1225cd34f0f3cd5e17c613fc70c9f7bfdf4b59b4400.jpg)

![9f281f0c618501e1fc169197c65082589a43307a09841fbffb2e3594c93fbffe.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/images/9f281f0c618501e1fc169197c65082589a43307a09841fbffb2e3594c93fbffe.jpg)

![a482b11f4870736df3124300afd821eb41338bfea0570a74bec20dd62d8c321a.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/images/a482b11f4870736df3124300afd821eb41338bfea0570a74bec20dd62d8c321a.jpg)

### Tables

![0a9a2b1dee4b73feca8c52c37a96b25d1f399fc47aa770e7e38e1f4f2503b7eb.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/tables/0a9a2b1dee4b73feca8c52c37a96b25d1f399fc47aa770e7e38e1f4f2503b7eb.jpg)

![504aeedd9cfe038ad6cbd73a5f2a99161d8ae054dcf05a002694ec1ec2ab8683.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/tables/504aeedd9cfe038ad6cbd73a5f2a99161d8ae054dcf05a002694ec1ec2ab8683.jpg)

![81484aab86b2ba5082b7e0c1e377d4a7d9a8cfb0474816c0cf5b789f4eee5ffa.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/tables/81484aab86b2ba5082b7e0c1e377d4a7d9a8cfb0474816c0cf5b789f4eee5ffa.jpg)

![e387aef6f108c8f1f7ad0befa9d93b6ea288c639a8fe1cb5ff3778129915b2b9.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/tables/e387aef6f108c8f1f7ad0befa9d93b6ea288c639a8fe1cb5ff3778129915b2b9.jpg)

![f21c54bf848963c3c6780e95d85773980e3a942ef616c90848d9dd34c3d99f16.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/tables/f21c54bf848963c3c6780e95d85773980e3a942ef616c90848d9dd34c3d99f16.jpg)

![f3613b1ff241d508f3c451f1aa759cded571f3e541fdf7483bd556e39bcad4c2.jpg](../icml_results/1529_The%20Panaceas%20for%20Improving%20Low-Rank%20Decomposition%20in%20Communication-Efficient%20Federated%20Learning/tables/f3613b1ff241d508f3c451f1aa759cded571f3e541fdf7483bd556e39bcad4c2.jpg)

## BCE vs. CE in Deep Feature Learning


### Images

![11a4cb275de40f1ad47e081e2f304c16aa85193fa9f1fb609f0676f4fc2efdd7.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/images/11a4cb275de40f1ad47e081e2f304c16aa85193fa9f1fb609f0676f4fc2efdd7.jpg)

![1db7719a016b7b49def0410d87f5946ed27c967581350a88ac174510a36d45fe.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/images/1db7719a016b7b49def0410d87f5946ed27c967581350a88ac174510a36d45fe.jpg)

![476271be46f00b02a0dbccb5ea8c2c77f8bd309f70915b93169d3ac6318b330a.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/images/476271be46f00b02a0dbccb5ea8c2c77f8bd309f70915b93169d3ac6318b330a.jpg)

![96cce2303425c90edfee175913eee363a60e28ef1289c8a1a7ecc7c60a2ba0f5.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/images/96cce2303425c90edfee175913eee363a60e28ef1289c8a1a7ecc7c60a2ba0f5.jpg)

![b32d138888b624e6fb7870bbf00dabdfa5976384961e744cf716a976b7fc0f7e.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/images/b32d138888b624e6fb7870bbf00dabdfa5976384961e744cf716a976b7fc0f7e.jpg)

![c98cd96f11659968b6b717e8bd4d37ddc145910b894e80364d962ae9730ca26c.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/images/c98cd96f11659968b6b717e8bd4d37ddc145910b894e80364d962ae9730ca26c.jpg)

![fffe02d08e540cdc5535a45d8197eab9d943eb58b7fe52b91ce6c47bd25d5b99.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/images/fffe02d08e540cdc5535a45d8197eab9d943eb58b7fe52b91ce6c47bd25d5b99.jpg)

### Tables

![5046fe23311d19bec286e39d662a130ae1db0f7f45d74d0fd2a9e9ed37d990d3.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/5046fe23311d19bec286e39d662a130ae1db0f7f45d74d0fd2a9e9ed37d990d3.jpg)

![514a98f87abc281fbae09491bd471bed3150b907f165e7c73c95553c454f7cdb.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/514a98f87abc281fbae09491bd471bed3150b907f165e7c73c95553c454f7cdb.jpg)

![537ce3174ecfca0907a69c3fe1399b9a1c7c0d10dee33db94b2d6b02bab65e77.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/537ce3174ecfca0907a69c3fe1399b9a1c7c0d10dee33db94b2d6b02bab65e77.jpg)

![619682dfbca8f9d3df0fdf70b538df1aa784968e12094687513ed5683ecf93a4.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/619682dfbca8f9d3df0fdf70b538df1aa784968e12094687513ed5683ecf93a4.jpg)

![6711e80e33e3c6aadf42c8e888e79446e8bd6728ce6ef708a727fa5c88598bcd.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/6711e80e33e3c6aadf42c8e888e79446e8bd6728ce6ef708a727fa5c88598bcd.jpg)

![7e871210655cc08be1ae83704c8c715beaf0116c0f196a253628b102a4a165ca.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/7e871210655cc08be1ae83704c8c715beaf0116c0f196a253628b102a4a165ca.jpg)

![85a337e725a475b62674b0ea3b9bc6a7507e35beb42ed06e7ef1db93001ca2ba.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/85a337e725a475b62674b0ea3b9bc6a7507e35beb42ed06e7ef1db93001ca2ba.jpg)

![87695c1504db52c3fe3423edee3c8754aa484a0d7ab2ef191ff8ad29ddce315d.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/87695c1504db52c3fe3423edee3c8754aa484a0d7ab2ef191ff8ad29ddce315d.jpg)

![8a65553d62be1a14c93922c29b8f99702945e6058b7716b9d0e69bd422980496.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/8a65553d62be1a14c93922c29b8f99702945e6058b7716b9d0e69bd422980496.jpg)

![ad8986d8a7738339b603380722e9a19a6f51ac29fdccc5686da4e7d3091b9ad5.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/ad8986d8a7738339b603380722e9a19a6f51ac29fdccc5686da4e7d3091b9ad5.jpg)

![b9695267cf413f107234c9c5fcc508cb2bd73b6237a755b0d7db582821cadcdd.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/b9695267cf413f107234c9c5fcc508cb2bd73b6237a755b0d7db582821cadcdd.jpg)

![bf55704240ae9f84ea2bdc6e99f250f2753f488a4cddaa62af31bc98e55f0122.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/bf55704240ae9f84ea2bdc6e99f250f2753f488a4cddaa62af31bc98e55f0122.jpg)

![d9d70412ca16269cd72f0195e6c76362db595b0118b3f8abd16b4313cdb344d8.jpg](../icml_results/1530_BCE%20vs.%20CE%20in%20Deep%20Feature%20Learning/tables/d9d70412ca16269cd72f0195e6c76362db595b0118b3f8abd16b4313cdb344d8.jpg)

## SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation


### Images

![2272f8182f4c3d6eb6fb93327d31972bd9c60035e396b9734a7b15f895e721f7.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/images/2272f8182f4c3d6eb6fb93327d31972bd9c60035e396b9734a7b15f895e721f7.jpg)

![8731f8261d92dddd209664b981df75cfd92e2d79d2c1f291a042a47a9ecbba94.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/images/8731f8261d92dddd209664b981df75cfd92e2d79d2c1f291a042a47a9ecbba94.jpg)

![bd85da1e4410ef09cbc520cf2b01787a14ba0044bd809fba8ef92164f0fd19fe.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/images/bd85da1e4410ef09cbc520cf2b01787a14ba0044bd809fba8ef92164f0fd19fe.jpg)

![ced79e4f19b9c07c63fe299733481e20395dafba91213346cc9beded871b9da1.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/images/ced79e4f19b9c07c63fe299733481e20395dafba91213346cc9beded871b9da1.jpg)

![fed74c45b48318d411b17c5f494c83421276a35e372c38eff890bbc4eaeb6fe3.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/images/fed74c45b48318d411b17c5f494c83421276a35e372c38eff890bbc4eaeb6fe3.jpg)

### Tables

![0bbdbc087b0e8c38102f3133192aba71b928c4803a0dab9092a765187c2f6c53.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/0bbdbc087b0e8c38102f3133192aba71b928c4803a0dab9092a765187c2f6c53.jpg)

![11ee2cce89793526dc22ed7ccc0f18168997f943f533c4b436c40c5d90315240.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/11ee2cce89793526dc22ed7ccc0f18168997f943f533c4b436c40c5d90315240.jpg)

![15bd493b049617143f9cd5a157af8b42fb1c93aeb670c9f431dbdd05d21a615d.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/15bd493b049617143f9cd5a157af8b42fb1c93aeb670c9f431dbdd05d21a615d.jpg)

![4c426f0a97de81a8f0904df9d14087f607adf991a3d70bbaab18da8f4d58fe38.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/4c426f0a97de81a8f0904df9d14087f607adf991a3d70bbaab18da8f4d58fe38.jpg)

![5adc8e2765bc83b003165e20275894227296cf5d2a4f40394869b43af0b195d0.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/5adc8e2765bc83b003165e20275894227296cf5d2a4f40394869b43af0b195d0.jpg)

![7a77a8c3926a0e509bc649be3aa1a52319297d62244b37f2297019dcca6e7541.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/7a77a8c3926a0e509bc649be3aa1a52319297d62244b37f2297019dcca6e7541.jpg)

![7f97f732860449c489d46601f3d13a406b67f897dc3ac4a3a44958ae762550b6.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/7f97f732860449c489d46601f3d13a406b67f897dc3ac4a3a44958ae762550b6.jpg)

![8493080558f8808030dc09de75d01e103441fa1903b0d4b5c1b0c6f942499d6a.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/8493080558f8808030dc09de75d01e103441fa1903b0d4b5c1b0c6f942499d6a.jpg)

![ec15a348925c03a900cbefe4e64d62dde81cb848ddff1cadbc8a090ff15349c3.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/ec15a348925c03a900cbefe4e64d62dde81cb848ddff1cadbc8a090ff15349c3.jpg)

![f43281fcc5f3a34c0da5b50e015453fafb45d69a1e7d38fc5d9008da5ab6f74b.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/f43281fcc5f3a34c0da5b50e015453fafb45d69a1e7d38fc5d9008da5ab6f74b.jpg)

![fd17bc438c3d3dcfb02452644491d4b44fc9438ba5575cfd025398720b0f0428.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/fd17bc438c3d3dcfb02452644491d4b44fc9438ba5575cfd025398720b0f0428.jpg)

![fe2a24d3397e9c36b2d2fe91dc563119072baffe1ba8aea78f5bfc5b26f19a62.jpg](../icml_results/1531_SAM2Act_%20Integrating%20Visual%20Foundation%20Model%20with%20A%20Memory%20Architecture%20for%20Robotic%20Manipulation/tables/fe2a24d3397e9c36b2d2fe91dc563119072baffe1ba8aea78f5bfc5b26f19a62.jpg)

## Programming Every Example: Lifting Pre-training Data Quality Like Experts at Scale


### Images

![14a170bb00280a27e080684eadabb7c993b1d0a7aa89ea07cd9e6a6b83ad8f50.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/14a170bb00280a27e080684eadabb7c993b1d0a7aa89ea07cd9e6a6b83ad8f50.jpg)

![2bc86de1a3799d575600d1c742f42d316204dcff87d54def235a1c761fde8a11.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/2bc86de1a3799d575600d1c742f42d316204dcff87d54def235a1c761fde8a11.jpg)

![35826ddd5322422a0b5e27f42d046bfaa8e62b6c52a573a3d99608337b8904af.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/35826ddd5322422a0b5e27f42d046bfaa8e62b6c52a573a3d99608337b8904af.jpg)

![3ee6cddae24c1a8418c72fc6489ce00f237fc04e1525ae4bce5c6f353f228d66.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/3ee6cddae24c1a8418c72fc6489ce00f237fc04e1525ae4bce5c6f353f228d66.jpg)

![5ac5adef6cf9301218a55686292a799457a9d218b9a1b92c37b5d52c1d4df081.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/5ac5adef6cf9301218a55686292a799457a9d218b9a1b92c37b5d52c1d4df081.jpg)

![b2e418710e073e543e34aba5b67609f4fda315718ebae77f72808e7f3a6fe7fa.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/b2e418710e073e543e34aba5b67609f4fda315718ebae77f72808e7f3a6fe7fa.jpg)

![b51cfa15a50fd4ce3b28f58a35ee850344fe874d51c2e27ff655fe24a975de63.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/b51cfa15a50fd4ce3b28f58a35ee850344fe874d51c2e27ff655fe24a975de63.jpg)

![db64ccf79a0bdadee1f761a0c40e79e0c4d9678521338ad523ea676ce1d6899d.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/db64ccf79a0bdadee1f761a0c40e79e0c4d9678521338ad523ea676ce1d6899d.jpg)

![faf129a1e03d8c48347fca0f325302ac13f8791f202c4b7bb3187ff978aa523b.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/images/faf129a1e03d8c48347fca0f325302ac13f8791f202c4b7bb3187ff978aa523b.jpg)

### Tables

![0f947304701fd785b766896b0aa5bf1b0fc553174625fe7bc900a782cf426ed1.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/0f947304701fd785b766896b0aa5bf1b0fc553174625fe7bc900a782cf426ed1.jpg)

![10632e6bceb57090ebb89e36f8c2e3d05dc04fad85b4e13231b8b2e191796a90.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/10632e6bceb57090ebb89e36f8c2e3d05dc04fad85b4e13231b8b2e191796a90.jpg)

![12eaab4c03a9dcdbd4a6529668799f69dd19e393a9b6c1556b9b2cf0e14e0a2f.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/12eaab4c03a9dcdbd4a6529668799f69dd19e393a9b6c1556b9b2cf0e14e0a2f.jpg)

![131856b351262a9985527db1700ca369a6be9025d68ec60a781fcfaf71f0490a.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/131856b351262a9985527db1700ca369a6be9025d68ec60a781fcfaf71f0490a.jpg)

![2fae410a23adfdc2f713f09f24bf48fff03f051a54ae6fda56f6d26631e556ef.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/2fae410a23adfdc2f713f09f24bf48fff03f051a54ae6fda56f6d26631e556ef.jpg)

![4ed150295920ea058fcde9825f1f9baeddf20eb42dc77d5ad8ef7e8109a22988.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/4ed150295920ea058fcde9825f1f9baeddf20eb42dc77d5ad8ef7e8109a22988.jpg)

![5b1f7e12afa16671afb9cfd8a5d0964254bc213345fd858ad32ad224444ec59d.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/5b1f7e12afa16671afb9cfd8a5d0964254bc213345fd858ad32ad224444ec59d.jpg)

![5db266d8b307d90bc7b4612a1991cdc751edec58f7c841b2a73379e99d92be9e.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/5db266d8b307d90bc7b4612a1991cdc751edec58f7c841b2a73379e99d92be9e.jpg)

![6669e0ebd2a3f592e1068822c911c53f810848df5fdde6ed08e8075da9fad698.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/6669e0ebd2a3f592e1068822c911c53f810848df5fdde6ed08e8075da9fad698.jpg)

![6839e193633eb8f6e45012b0b828e81da4577594db4977a50b84df7c0211bc42.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/6839e193633eb8f6e45012b0b828e81da4577594db4977a50b84df7c0211bc42.jpg)

![69bee217d7f7f97325573990c4a79215b591a230db1b85ba419348b947f79500.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/69bee217d7f7f97325573990c4a79215b591a230db1b85ba419348b947f79500.jpg)

![6fd731727ac82665b618d457bd9ab295f54f28e7e0972189a1d6d704dd9d2c65.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/6fd731727ac82665b618d457bd9ab295f54f28e7e0972189a1d6d704dd9d2c65.jpg)

![727c435dc226f34bb2d130b2860af8a01844dc7819f16483f7a657ecb55a118b.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/727c435dc226f34bb2d130b2860af8a01844dc7819f16483f7a657ecb55a118b.jpg)

![7577445e91fa3a8dece6864a3a4ec26785da6d1bd17e0713a0c2dc60ed22e852.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/7577445e91fa3a8dece6864a3a4ec26785da6d1bd17e0713a0c2dc60ed22e852.jpg)

![769acd1ee5eaeefd52e38a94bb9d56db7368b4252743156b5b1dfe68fc44b614.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/769acd1ee5eaeefd52e38a94bb9d56db7368b4252743156b5b1dfe68fc44b614.jpg)

![76aa7da658d6f1df1d9e3c5083461dc8d069e84295349cdd2385d289812e1ac9.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/76aa7da658d6f1df1d9e3c5083461dc8d069e84295349cdd2385d289812e1ac9.jpg)

![77627db13c6f340c591bd99ddc63314a61cce611bf81fecf4377485b29504ca9.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/77627db13c6f340c591bd99ddc63314a61cce611bf81fecf4377485b29504ca9.jpg)

![7d9dd4cc10459c7637165d3ea19f27e64d5e344f2b8f5bb7753febd1c56aa125.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/7d9dd4cc10459c7637165d3ea19f27e64d5e344f2b8f5bb7753febd1c56aa125.jpg)

![932129611b20ce37fd166e3cc2d89f2065f7332176e129f7814d2d3258ec43fe.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/932129611b20ce37fd166e3cc2d89f2065f7332176e129f7814d2d3258ec43fe.jpg)

![9bb48f880ac73bfcff13006e5d792a0700c0db281ba3b90c5579e2f7ee7122b7.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/9bb48f880ac73bfcff13006e5d792a0700c0db281ba3b90c5579e2f7ee7122b7.jpg)

![a1bef33036cb64f2a308596bfa19451da01bd4f168a9064d6980da3118f33950.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/a1bef33036cb64f2a308596bfa19451da01bd4f168a9064d6980da3118f33950.jpg)

![a3f0c99629f628020c986cfcea6eb868242212e4e73f14742f0618f4136c5223.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/a3f0c99629f628020c986cfcea6eb868242212e4e73f14742f0618f4136c5223.jpg)

![a4c42a2033186ed8675b13ed55868c4ad56d84a0dc4377939afbef0c25990ced.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/a4c42a2033186ed8675b13ed55868c4ad56d84a0dc4377939afbef0c25990ced.jpg)

![a5f833a35065e45eb18c18b2c516cd3db6d350b6598892e2be8027f68b36803e.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/a5f833a35065e45eb18c18b2c516cd3db6d350b6598892e2be8027f68b36803e.jpg)

![a89a10125e0374103f256d91f60337d89a42b70f5760d921de0c387a12a4b956.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/a89a10125e0374103f256d91f60337d89a42b70f5760d921de0c387a12a4b956.jpg)

![a8e0784156e284c092a02a9c653ac2e510ef908f7f74e1860f86a287904dcc9e.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/a8e0784156e284c092a02a9c653ac2e510ef908f7f74e1860f86a287904dcc9e.jpg)

![b9643269663f56de76f5a0f8957059d1c6920d2ce4ec00ff57b67b04e28cd0dd.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/b9643269663f56de76f5a0f8957059d1c6920d2ce4ec00ff57b67b04e28cd0dd.jpg)

![bbf011b3fc4419ee7b391894efeddc24868e25cf4bbeb575197932ce43e20888.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/bbf011b3fc4419ee7b391894efeddc24868e25cf4bbeb575197932ce43e20888.jpg)

![bc59a5a3a5aef3d3807fe49eb592a87b762fddaf495f08286cb90bce38541faa.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/bc59a5a3a5aef3d3807fe49eb592a87b762fddaf495f08286cb90bce38541faa.jpg)

![d04f6033c128e1c085d889fe26f662cea4e952341ededa44bd03ab77a84323d7.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/d04f6033c128e1c085d889fe26f662cea4e952341ededa44bd03ab77a84323d7.jpg)

![d217a84de85ddd4d284ad96f07c3f3eec5db7b389d2ed3e7c2220c4a5e8e3a31.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/d217a84de85ddd4d284ad96f07c3f3eec5db7b389d2ed3e7c2220c4a5e8e3a31.jpg)

![d7b8c2f68f122270d430dab0d70723417cba5e72109f06c4806f958c7c1d5a60.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/d7b8c2f68f122270d430dab0d70723417cba5e72109f06c4806f958c7c1d5a60.jpg)

![df2bd2f64c317838c4aa8d920d838f516dd881063a46b97cf7d0639064349a68.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/df2bd2f64c317838c4aa8d920d838f516dd881063a46b97cf7d0639064349a68.jpg)

![e30f599ad58facdf63b31f2e0058e87805f0aaafe80c428035f51414304b2067.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/e30f599ad58facdf63b31f2e0058e87805f0aaafe80c428035f51414304b2067.jpg)

![e6980fc432c627263525b65ea3e250bb8db420ca1c4b6064181a393bc54aea61.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/e6980fc432c627263525b65ea3e250bb8db420ca1c4b6064181a393bc54aea61.jpg)

![e839984f9da2c529e6199870509480e41a494d1d93642189c348cd718b02e086.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/e839984f9da2c529e6199870509480e41a494d1d93642189c348cd718b02e086.jpg)

![fc613a06a423c01cdf8c26c94527f5dd52fd88af9a708df9236478b00b3d8a96.jpg](../icml_results/1532_Programming%20Every%20Example_%20Lifting%20Pre-training%20Data%20Quality%20Like%20Experts%20at%20Scale/tables/fc613a06a423c01cdf8c26c94527f5dd52fd88af9a708df9236478b00b3d8a96.jpg)

## ROME is Forged in Adversity: Robust Distilled Datasets via Information Bottleneck

### Images

![0ae5b386eb976fad9a674fbd2f55e8179c8030aa39dca62f24118cb577c4857c.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/0ae5b386eb976fad9a674fbd2f55e8179c8030aa39dca62f24118cb577c4857c.jpg)

![0c3717aa4321cea3fbab4f6111a6a0757dd0337e89c3c9aaf90618ffe2a9cadc.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/0c3717aa4321cea3fbab4f6111a6a0757dd0337e89c3c9aaf90618ffe2a9cadc.jpg)

![303004607b303640240df776b861354f18fdff574148335b86d49c1f82a4849b.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/303004607b303640240df776b861354f18fdff574148335b86d49c1f82a4849b.jpg)

![524fb8f44858fbfcc86c7272e3a96d0c86a5eb5a7e14f9545183d8c50c1f261a.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/524fb8f44858fbfcc86c7272e3a96d0c86a5eb5a7e14f9545183d8c50c1f261a.jpg)

![66b158cab3f0575cd699cba2c924e17b7eec5b5136502cf4304d2861753be2c1.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/66b158cab3f0575cd699cba2c924e17b7eec5b5136502cf4304d2861753be2c1.jpg)

![752dc5c580f73b442f4b1e80675466b700c5a87d8f5817d6e19b15a94a5ce5b4.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/752dc5c580f73b442f4b1e80675466b700c5a87d8f5817d6e19b15a94a5ce5b4.jpg)

![789d1df70227b3aee6d9111cbb159fb20ede94279be9a3e5041b152ef8643122.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/789d1df70227b3aee6d9111cbb159fb20ede94279be9a3e5041b152ef8643122.jpg)

![8d62233934c31f5e27bbaf4cce7d2a49a62a2fe674899597b64cb6f6199ae825.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/8d62233934c31f5e27bbaf4cce7d2a49a62a2fe674899597b64cb6f6199ae825.jpg)

![b1b7153d0973bcca9e0fe86ad70960f74ddc36b713468cbe539509ef005a186a.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/b1b7153d0973bcca9e0fe86ad70960f74ddc36b713468cbe539509ef005a186a.jpg)

![d37007cbc9e0d91f9d5ff474418374a8bc1886836acccbd793dab08491a50aba.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/images/d37007cbc9e0d91f9d5ff474418374a8bc1886836acccbd793dab08491a50aba.jpg)

### Tables

![1a01dd477eeba57e24074b630827d2b964041bf480f298ef5c8877df09495f95.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/1a01dd477eeba57e24074b630827d2b964041bf480f298ef5c8877df09495f95.jpg)

![6fc993439ed3d7ebb41d303c880afce0b4ff3f8ea9efd7026b5b79af63b0c23f.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/6fc993439ed3d7ebb41d303c880afce0b4ff3f8ea9efd7026b5b79af63b0c23f.jpg)

![8c507ef8bf791ea97fed288fa6b9d1b02844042590c3660591f5f53f5ea1ea2a.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/8c507ef8bf791ea97fed288fa6b9d1b02844042590c3660591f5f53f5ea1ea2a.jpg)

![a27932802146d5cc068940d980c799eac6195cac2158da5644d986c46094673b.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/a27932802146d5cc068940d980c799eac6195cac2158da5644d986c46094673b.jpg)

![b2b61269068a8911dd26def05b2e573b41dea442e965f0167ca971a86cf05d41.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/b2b61269068a8911dd26def05b2e573b41dea442e965f0167ca971a86cf05d41.jpg)

![bcc220bca0b31646252ecb0134b154ff5e2ebc47dfd5e37dbcd80df56fa409e6.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/bcc220bca0b31646252ecb0134b154ff5e2ebc47dfd5e37dbcd80df56fa409e6.jpg)

![f1f4191db3fb92b8942c2b97687d818983c18414c421d254b985047527fdded2.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/f1f4191db3fb92b8942c2b97687d818983c18414c421d254b985047527fdded2.jpg)

![f7759da8a7d8ac494aecbac2927e6bfd76d8e8bb44f3f9d7505d6b53953ce0b6.jpg](../icml_results/1533_ROME%20is%20Forged%20in%20Adversity_%20Robust%20Distilled%20Datasets%20via%20Information%20Bottleneck/tables/f7759da8a7d8ac494aecbac2927e6bfd76d8e8bb44f3f9d7505d6b53953ce0b6.jpg)
