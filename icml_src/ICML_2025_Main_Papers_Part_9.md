# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 9 of 100

## 目录 (Table of Contents)

1. [Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data](#Penalizing-Infeasible-Actions-and-Reward-Scaling-in-Reinforcement-Learning-with-Offline-Data)
2. [LOCATE 3D: Real-World Object Localization via Self-Supervised Learning in 3D](#LOCATE-3D-Real-World-Object-Localization-via-Self-Supervised-Learning-in-3D)
3. [Is Complex Query Answering Really Complex?](#Is-Complex-Query-Answering-Really-Complex)
4. [Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models](#Layer-wise-Alignment-Examining-Safety-Alignment-Across-Image-Encoder-Layers-in-Vision-Language-Models)
5. [Identifying Causal Direction via Variational Bayesian Compression](#Identifying-Causal-Direction-via-Variational-Bayesian-Compression)
6. [Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](#Understanding-and-Mitigating-Memorization-in-Generative-Models-via-Sharpness-of-Probability-Landscapes)
7. [am-ELO: A Stable Framework for Arena-based LLM Evaluation](#am-ELO-A-Stable-Framework-for-Arena-based-LLM-Evaluation)
8. [Flopping for FLOPs: Leveraging Equivariance for Computational Efficiency](#Flopping-for-FLOPs-Leveraging-Equivariance-for-Computational-Efficiency)
9. [Enforcing Latent Euclidean Geometry in Single-Cell VAEs for Manifold Interpolation](#Enforcing-Latent-Euclidean-Geometry-in-Single-Cell-VAEs-for-Manifold-Interpolation)
10. [Monte-Carlo Tree Search with Uncertainty Propagation via Optimal Transport](#Monte-Carlo-Tree-Search-with-Uncertainty-Propagation-via-Optimal-Transport)
11. [Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger](#Re-ranking-Reasoning-Context-with-Tree-Search-Makes-Large-Vision-Language-Models-Stronger)
12. [Robust Automatic Modulation Classification with Fuzzy Regularization](#Robust-Automatic-Modulation-Classification-with-Fuzzy-Regularization)
13. [Diffusion-based Adversarial Purification from the Perspective of the Frequency Domain](#Diffusion-based-Adversarial-Purification-from-the-Perspective-of-the-Frequency-Domain)
14. [Not All Wrong is Bad: Using Adversarial Examples for Unlearning](#Not-All-Wrong-is-Bad-Using-Adversarial-Examples-for-Unlearning)
15. [TLLC: Transfer Learning-based Label Completion for Crowdsourcing](#TLLC-Transfer-Learning-based-Label-Completion-for-Crowdsourcing)
16. [Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspective](#Distribution-aware-Fairness-Learning-in-Medical-Image-Segmentation-From-A-Control-Theoretic-Perspective)
17. [Learning Soft Sparse Shapes for Efficient Time-Series Classification](#Learning-Soft-Sparse-Shapes-for-Efficient-Time-Series-Classification)
18. [Optimizing Adaptive Attacks against Watermarks for Language Models](#Optimizing-Adaptive-Attacks-against-Watermarks-for-Language-Models)
19. [Exogenous Isomorphism for Counterfactual Identifiability](#Exogenous-Isomorphism-for-Counterfactual-Identifiability)
20. [Robust ML Auditing using Prior Knowledge](#Robust-ML-Auditing-using-Prior-Knowledge)
21. [Adjusting Model Size in Continual Gaussian Processes: How Big is Big Enough?](#Adjusting-Model-Size-in-Continual-Gaussian-Processes-How-Big-is-Big-Enough)
22. [LotteryCodec: Searching the Implicit Representation in a Random Network for Low-Complexity Image Compression](#LotteryCodec-Searching-the-Implicit-Representation-in-a-Random-Network-for-Low-Complexity-Image-Compression)
23. [MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding](#MODA-MOdular-Duplex-Attention-for-Multimodal-Perception-Cognition-and-Emotion-Understanding)
24. [FedSSI: Rehearsal-Free Continual Federated Learning with Synergistic  Synaptic Intelligence](#FedSSI-Rehearsal-Free-Continual-Federated-Learning-with-Synergistic-Synaptic-Intelligence)
25. [Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization](#Mechanistic-Unlearning-Robust-Knowledge-Unlearning-and-Editing-via-Mechanistic-Localization)
26. [Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](#Robust-Noise-Attenuation-via-Adaptive-Pooling-of-Transformer-Outputs)
27. [Geometric Representation Condition Improves Equivariant Molecule Generation](#Geometric-Representation-Condition-Improves-Equivariant-Molecule-Generation)
28. [RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding](#RAPID-Long-Context-Inference-with-Retrieval-Augmented-Speculative-Decoding)
29. [$K^2$VAE: A Koopman-Kalman Enhanced Variational AutoEncoder for Probabilistic Time Series Forecasting](#K2VAE-A-Koopman-Kalman-Enhanced-Variational-AutoEncoder-for-Probabilistic-Time-Series-Forecasting)
30. [Self-supervised Masked Graph Autoencoder via Structure-aware Curriculum](#Self-supervised-Masked-Graph-Autoencoder-via-Structure-aware-Curriculum)
31. [SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](#SDP-CROWN-Efficient-Bound-Propagation-for-Neural-Network-Verification-with-Tightness-of-Semidefinite-Programming)
32. [The Synergy of LLMs & RL Unlocks Offline Learning of Generalizable Language-Conditioned Policies with Low-fidelity Data](#The-Synergy-of-LLMs-RL-Unlocks-Offline-Learning-of-Generalizable-Language-Conditioned-Policies-with-Low-fidelity-Data)
33. [Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks in Superposition](#Everything-Everywhere-All-at-Once-LLMs-can-In-Context-Learn-Multiple-Tasks-in-Superposition)

---


## Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data

### Images

![5a25078f81eb593f2e37b3a0ea247ca842bacf518260fc5aa841a7631a295518.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/5a25078f81eb593f2e37b3a0ea247ca842bacf518260fc5aa841a7631a295518.jpg)

![68472ca40d8294d43563c76b0e48e801821898f56d89de21c31f1e10aa98d12d.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/68472ca40d8294d43563c76b0e48e801821898f56d89de21c31f1e10aa98d12d.jpg)

![72b4ce1d21c55a978f7f27c6e568a1d2ce50defeef75cd0641742ec8401d93d1.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/72b4ce1d21c55a978f7f27c6e568a1d2ce50defeef75cd0641742ec8401d93d1.jpg)

![8b24060d5bbf9cbdab8c1e0e6760a7129a59fbb25628402711172e49f5728cd1.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/8b24060d5bbf9cbdab8c1e0e6760a7129a59fbb25628402711172e49f5728cd1.jpg)

![a29f102544e87d65a5e29e0d0390b9d593d337a81a298d255181ec27de888a61.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/a29f102544e87d65a5e29e0d0390b9d593d337a81a298d255181ec27de888a61.jpg)

![a6d93d0758a43a666f485b96aeaa8d10e607e94cbb0db2491300b32f7d62d2c7.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/a6d93d0758a43a666f485b96aeaa8d10e607e94cbb0db2491300b32f7d62d2c7.jpg)

![c136834b96273356fdb1d296cf15c2720992a68164a0951df0c38b7c02449eac.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/c136834b96273356fdb1d296cf15c2720992a68164a0951df0c38b7c02449eac.jpg)

![ff754f42a30a16a4fee8afbd1940d7e8b2ec90b1b7ce21984cd134f76df2fb4f.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/images/ff754f42a30a16a4fee8afbd1940d7e8b2ec90b1b7ce21984cd134f76df2fb4f.jpg)

### Tables

![0899f53690decc18c38b0a6b0da0dc7f75361425bbd5c2cbf2d71d1e030a39fa.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/0899f53690decc18c38b0a6b0da0dc7f75361425bbd5c2cbf2d71d1e030a39fa.jpg)

![1fba7e60b88e93a5117973e1e27917708440dc93394f5a59e9affe4714f7743c.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/1fba7e60b88e93a5117973e1e27917708440dc93394f5a59e9affe4714f7743c.jpg)

![5448b356f81c460f1df2d2c19b204a649e76f1fa416861951bce6165173378f7.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/5448b356f81c460f1df2d2c19b204a649e76f1fa416861951bce6165173378f7.jpg)

![6a0bcaad93f53a96f649d10fc6df9ef30fa337d4ec94969bd3cc4b1763c29dd9.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/6a0bcaad93f53a96f649d10fc6df9ef30fa337d4ec94969bd3cc4b1763c29dd9.jpg)

![77ec07918f73583d146e759d6228b7acb04dddc35d9abd36d81fcc5eef47bc55.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/77ec07918f73583d146e759d6228b7acb04dddc35d9abd36d81fcc5eef47bc55.jpg)

![840434a60d2666c6f18635d61d80e3b0a85078c7e1eaba5aa0ce0b3b102f79ed.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/840434a60d2666c6f18635d61d80e3b0a85078c7e1eaba5aa0ce0b3b102f79ed.jpg)

![88c7e3f6be8b6bf19c2c1a5a439ea91023688b9e66ff27da28b964928ac9d4bb.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/88c7e3f6be8b6bf19c2c1a5a439ea91023688b9e66ff27da28b964928ac9d4bb.jpg)

![8c239718fb53e299a85f1e1f9480a0103f5a2436e03eeebd6c89faa33c5bd38f.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/8c239718fb53e299a85f1e1f9480a0103f5a2436e03eeebd6c89faa33c5bd38f.jpg)

![8dc38be8bcc89eed22b6236a37cfc0cdc910d0ea84b60d39cc181fa4ce15cbdc.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/8dc38be8bcc89eed22b6236a37cfc0cdc910d0ea84b60d39cc181fa4ce15cbdc.jpg)

![9a8674fcadecbfd50ca7cf219fbf2cb640f61e47103e35af63c0e4d19d94b807.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/9a8674fcadecbfd50ca7cf219fbf2cb640f61e47103e35af63c0e4d19d94b807.jpg)

![b1cb41e5d739201a4706ba1db2731a2502bc6114886441b7daa8efbbe86c5b35.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/b1cb41e5d739201a4706ba1db2731a2502bc6114886441b7daa8efbbe86c5b35.jpg)

![b3f612dd1f672a562edd0fce16309124989b8edd85ac1d89cb96a3f3d8f5a975.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/b3f612dd1f672a562edd0fce16309124989b8edd85ac1d89cb96a3f3d8f5a975.jpg)

![c2928d2f7e28059706ae4ade63a3298a125a05995bb3062c73a6fbac7c8e74fc.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/c2928d2f7e28059706ae4ade63a3298a125a05995bb3062c73a6fbac7c8e74fc.jpg)

![c62dd38caad1b11124a9861e9e00472571716372535bc334fb25e1b121568b6d.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/c62dd38caad1b11124a9861e9e00472571716372535bc334fb25e1b121568b6d.jpg)

![db44231521b9fc291f161bc13d6a9adfe97872bd3d7e6141688c5b0fb0006875.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/db44231521b9fc291f161bc13d6a9adfe97872bd3d7e6141688c5b0fb0006875.jpg)

![e4796e7824866a79a234260ef3b168b2f339f3d878a1d5e68f9b015072b1c93b.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/e4796e7824866a79a234260ef3b168b2f339f3d878a1d5e68f9b015072b1c93b.jpg)

![fc036a6a916aacb3c97fff2f558bf30bf6e80bf84378403f0ae7652829dace7b.jpg](../icml_results/266_Learning%20Safety%20Constraints%20for%20Large%20Language%20Models/tables/fc036a6a916aacb3c97fff2f558bf30bf6e80bf84378403f0ae7652829dace7b.jpg)

## Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data


### Images

![0d5c3152c8e2567609a0eec40a1ae42a191a099adf1417c2630cd20fca7bed0e.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/0d5c3152c8e2567609a0eec40a1ae42a191a099adf1417c2630cd20fca7bed0e.jpg)

![2630df09dbb9d55f141da957f9a293a348d06490f80f2181b46c55422769b933.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/2630df09dbb9d55f141da957f9a293a348d06490f80f2181b46c55422769b933.jpg)

![2b780ba01075d39b81225d6ae5ebcdb8e1269de378444531eab6f835994a92b4.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/2b780ba01075d39b81225d6ae5ebcdb8e1269de378444531eab6f835994a92b4.jpg)

![4479e414bf46a437b5b3eed3e411f1415707b8868ca89d1b9e41005c6c10b033.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/4479e414bf46a437b5b3eed3e411f1415707b8868ca89d1b9e41005c6c10b033.jpg)

![447caf66b44db19f9c95b1fec79fb741b3e047e2468d04fdfb9e6560414a52d7.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/447caf66b44db19f9c95b1fec79fb741b3e047e2468d04fdfb9e6560414a52d7.jpg)

![45e54c13c8e30dda5e0c5ba52fe5e2f90e0a88e8c4afe945986a8f886c51feb5.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/45e54c13c8e30dda5e0c5ba52fe5e2f90e0a88e8c4afe945986a8f886c51feb5.jpg)

![4c92573d3f09f30b20895e863074e56a254872a09009cec744fca888ce15d5c7.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/4c92573d3f09f30b20895e863074e56a254872a09009cec744fca888ce15d5c7.jpg)

![5ccfb19a3be8e0c0db49b79ef18fd2d3eb9e59da4e7c839b0781a6b0a7c32b4f.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/5ccfb19a3be8e0c0db49b79ef18fd2d3eb9e59da4e7c839b0781a6b0a7c32b4f.jpg)

![5d814f58b278d708d72d8d28711e209b86ba33242e9173408153a4f1ed3e4d32.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/5d814f58b278d708d72d8d28711e209b86ba33242e9173408153a4f1ed3e4d32.jpg)

![683bd3c67c266d19afaf95fe5e45a1555a179c5efee1a254762b027d43d2131e.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/683bd3c67c266d19afaf95fe5e45a1555a179c5efee1a254762b027d43d2131e.jpg)

![7132248d6bbe3cc792e3abf44bfee0ef4f297fe4a473811cdf890867b89f0188.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/7132248d6bbe3cc792e3abf44bfee0ef4f297fe4a473811cdf890867b89f0188.jpg)

![71cdc0f433aace43c37b09db4ec75de5defbf5bcf972afe0bb9bbc57af3dd520.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/71cdc0f433aace43c37b09db4ec75de5defbf5bcf972afe0bb9bbc57af3dd520.jpg)

![8ad0d18d96bc2d98efb76b0b6d3b5ad0129560429b9b3e32110f704614fae875.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/8ad0d18d96bc2d98efb76b0b6d3b5ad0129560429b9b3e32110f704614fae875.jpg)

![8ea6cd7a3aedcf12e5e10f876c92bc076c001f80c9e35dffc149203392ee78af.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/8ea6cd7a3aedcf12e5e10f876c92bc076c001f80c9e35dffc149203392ee78af.jpg)

![b71ccc010764786e214d0d6e7ca2a618178f0256346f594f00143c8d4c19d64f.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/b71ccc010764786e214d0d6e7ca2a618178f0256346f594f00143c8d4c19d64f.jpg)

![bd5e165526e9ee818442ee76ed459f40aa1c01da6749bbef6b4088eee097e78f.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/bd5e165526e9ee818442ee76ed459f40aa1c01da6749bbef6b4088eee097e78f.jpg)

![be7d79b2a0dabd0fcede64f127904db23c1eda654adf845c8d6d9c84efde8333.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/be7d79b2a0dabd0fcede64f127904db23c1eda654adf845c8d6d9c84efde8333.jpg)

![dc7d1a757369fa6e586c9bc536f2006a9961bd1ddc5aa13ab3aa177a59091d65.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/dc7d1a757369fa6e586c9bc536f2006a9961bd1ddc5aa13ab3aa177a59091d65.jpg)

![e61236394f92bb8e2144475133364e894d668f540bc25393025ecb096e4de80d.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/e61236394f92bb8e2144475133364e894d668f540bc25393025ecb096e4de80d.jpg)

![f296ed48e4bd90b00e113f13c95ba9c0343cdfae4d3bba070adcd5b731fe9d7d.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/f296ed48e4bd90b00e113f13c95ba9c0343cdfae4d3bba070adcd5b731fe9d7d.jpg)

![fa98c68fbba0778beb096bc24de0f27b77b3fcb849ba15baaeccdfed0e775f05.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/fa98c68fbba0778beb096bc24de0f27b77b3fcb849ba15baaeccdfed0e775f05.jpg)

![fe6a6c672fc55c35d09b7e7226afb4a33a51690b0639ab96a8260430269a47f6.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/images/fe6a6c672fc55c35d09b7e7226afb4a33a51690b0639ab96a8260430269a47f6.jpg)

### Tables

![256f03cd6fa8fc2c4b47c4cb126418e39d24420a843f47d1287bfec0c623b005.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/256f03cd6fa8fc2c4b47c4cb126418e39d24420a843f47d1287bfec0c623b005.jpg)

![2743ad3c7a099463e0370cb042dc5bd261404b4cd85b17af6d4ea326812e10a6.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/2743ad3c7a099463e0370cb042dc5bd261404b4cd85b17af6d4ea326812e10a6.jpg)

![2bfb0ea021e725f195bfbdf55b5491564c6a0076bfc8af6d1bc5a8773113ce69.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/2bfb0ea021e725f195bfbdf55b5491564c6a0076bfc8af6d1bc5a8773113ce69.jpg)

![58f2a3b51d4722e61e0d9f981c00be8cdcfaad954145f3758c1759d37ad3319f.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/58f2a3b51d4722e61e0d9f981c00be8cdcfaad954145f3758c1759d37ad3319f.jpg)

![60de90d78e2f06fd25d84807a4910661e1b4b0419df2a6e62aa36c6773fbcbfc.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/60de90d78e2f06fd25d84807a4910661e1b4b0419df2a6e62aa36c6773fbcbfc.jpg)

![65ddf0e7be6fa807161b44b02ced155c7c1b88341ae168d7737656f6d2857656.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/65ddf0e7be6fa807161b44b02ced155c7c1b88341ae168d7737656f6d2857656.jpg)

![6bed967ad6ace1020963c615c87c62b16fdd71a9d9f9c6f0090375cf3ce138ac.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/6bed967ad6ace1020963c615c87c62b16fdd71a9d9f9c6f0090375cf3ce138ac.jpg)

![6f8e6d61d4df2eacd16f95f57c050005cb348389cdc9b77a6b0003a964ee5bac.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/6f8e6d61d4df2eacd16f95f57c050005cb348389cdc9b77a6b0003a964ee5bac.jpg)

![79b8a3ae4fb65fa081d2b56f85bdec25a962fd2f0edb13625299ac2d44695fb6.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/79b8a3ae4fb65fa081d2b56f85bdec25a962fd2f0edb13625299ac2d44695fb6.jpg)

![7e01a517939e27809f2b397714835977c576263f3ffc50e7340a605adce4bd9e.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/7e01a517939e27809f2b397714835977c576263f3ffc50e7340a605adce4bd9e.jpg)

![9b6841a1fa8d1d141d83989bb7b74e923c1545f7c6d0ff751e0e86a0dcec4dfb.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/9b6841a1fa8d1d141d83989bb7b74e923c1545f7c6d0ff751e0e86a0dcec4dfb.jpg)

![bc56906e715970fa78f664b11f686257197301c7826f2ead6dc70e0e43e32195.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/bc56906e715970fa78f664b11f686257197301c7826f2ead6dc70e0e43e32195.jpg)

![be21fb6edb11e8adea3ae08bf7cf416a8a04234de3c7d04305302c5b749b234b.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/be21fb6edb11e8adea3ae08bf7cf416a8a04234de3c7d04305302c5b749b234b.jpg)

![be75519ddc629943e02c19912f358bc093e83f3d277d2cb4be253184ca34ee98.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/be75519ddc629943e02c19912f358bc093e83f3d277d2cb4be253184ca34ee98.jpg)

![d63285a141aae9f887de57654507bb69f9871294f507c68e5f13fd331da06ab0.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/d63285a141aae9f887de57654507bb69f9871294f507c68e5f13fd331da06ab0.jpg)

![e4049a4482de1c2812433201286fef7b58aa54b745d5d26572ffeb6b96c86179.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/e4049a4482de1c2812433201286fef7b58aa54b745d5d26572ffeb6b96c86179.jpg)

![e52eca5655ae23d4919a2770fc175c9ed2091fee473164dccde43a6caa219d95.jpg](../icml_results/267_Penalizing%20Infeasible%20Actions%20and%20Reward%20Scaling%20in%20Reinforcement%20Learning%20with%20Offline%20Data/tables/e52eca5655ae23d4919a2770fc175c9ed2091fee473164dccde43a6caa219d95.jpg)

## LOCATE 3D: Real-World Object Localization via Self-Supervised Learning in 3D


### Images

![02e03581618883ca8660ad05fc4e1d9b2e8bcc6e7bd708aae9392d43baa1e2c6.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/02e03581618883ca8660ad05fc4e1d9b2e8bcc6e7bd708aae9392d43baa1e2c6.jpg)

![02f8eb524e79da989807a9c8ebfe2798201650ad2e750e0dfd13a79a7a1a54b3.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/02f8eb524e79da989807a9c8ebfe2798201650ad2e750e0dfd13a79a7a1a54b3.jpg)

![09662aea23571fbc38d11090e34eaa507fafe1fb206ababaa26a11d7ff9afb1b.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/09662aea23571fbc38d11090e34eaa507fafe1fb206ababaa26a11d7ff9afb1b.jpg)

![2401d915569df4bed7b35cf8cf1a2bf9609a28289d83c7540de7deac3d024a14.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/2401d915569df4bed7b35cf8cf1a2bf9609a28289d83c7540de7deac3d024a14.jpg)

![283806655830efd273d9430c4b8072965b94c289d12b699e82dbbf0792b8ccb3.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/283806655830efd273d9430c4b8072965b94c289d12b699e82dbbf0792b8ccb3.jpg)

![7d864259fcc206364ba6006f427cfc8d0014a361f7d9c2631567212521835008.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/7d864259fcc206364ba6006f427cfc8d0014a361f7d9c2631567212521835008.jpg)

![8bc5dc6a71fce63aa6367fe376e1adf5f7cabaaa3a55c0de8db0beb24b10a2c0.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/8bc5dc6a71fce63aa6367fe376e1adf5f7cabaaa3a55c0de8db0beb24b10a2c0.jpg)

![913d9e09240794e0832e70f8707f0536b7db79fa78e8ee66389bfe402076816c.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/913d9e09240794e0832e70f8707f0536b7db79fa78e8ee66389bfe402076816c.jpg)

![9beca94ab36ce69102e759dcc9dc505196ea83fd4bcdd585b3cd5db22855f286.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/9beca94ab36ce69102e759dcc9dc505196ea83fd4bcdd585b3cd5db22855f286.jpg)

![9eba51bec7d1a623a310e31cf23dfe4d621f55a3db570fd993330ff792a7facf.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/9eba51bec7d1a623a310e31cf23dfe4d621f55a3db570fd993330ff792a7facf.jpg)

![e3e397bc378620721720311936551b3108e2b3eedc46b5a966beb66a84712acb.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/e3e397bc378620721720311936551b3108e2b3eedc46b5a966beb66a84712acb.jpg)

![f547f3315e603c7dadeada3c155c6f118538db5caed448a004192ebaeb79e6ab.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/images/f547f3315e603c7dadeada3c155c6f118538db5caed448a004192ebaeb79e6ab.jpg)

### Tables

![154bd82d827f37c8c192e8c8d2282fc5628703f9da933e3d80ee23d4795351bb.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/154bd82d827f37c8c192e8c8d2282fc5628703f9da933e3d80ee23d4795351bb.jpg)

![19be18ff50c39924942ecc0bc3948f38ffebf7fcd3fc446324f14fa3b3928213.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/19be18ff50c39924942ecc0bc3948f38ffebf7fcd3fc446324f14fa3b3928213.jpg)

![20133209c9d1ccadc652cbd6e5be618e9d138f22c75b466d72d63407ae579bf6.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/20133209c9d1ccadc652cbd6e5be618e9d138f22c75b466d72d63407ae579bf6.jpg)

![24299e92d4a709093918eec04050f87a7607a5adf8a92eb0ebecc40a248ad357.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/24299e92d4a709093918eec04050f87a7607a5adf8a92eb0ebecc40a248ad357.jpg)

![2b2545e16e4bc7808478e5ea82ea595e0a701befd77d287338ddd87b85edbb9e.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/2b2545e16e4bc7808478e5ea82ea595e0a701befd77d287338ddd87b85edbb9e.jpg)

![53e7fda38a3b6e988d46bc77a3134e296a6367f9aea32981837669351147eec6.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/53e7fda38a3b6e988d46bc77a3134e296a6367f9aea32981837669351147eec6.jpg)

![5b939170162d08363ca92b81c60b2bbe4481347c26586c29d1c93506a1cb7c21.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/5b939170162d08363ca92b81c60b2bbe4481347c26586c29d1c93506a1cb7c21.jpg)

![6a1c4207a14a326b9f68ee7863f30cad4b84d154bb026995dea7620549db9d94.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/6a1c4207a14a326b9f68ee7863f30cad4b84d154bb026995dea7620549db9d94.jpg)

![c26807fcbc478afff410116b96589be42a86efa03bd0a13462d727a040d00ba1.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/c26807fcbc478afff410116b96589be42a86efa03bd0a13462d727a040d00ba1.jpg)

![dd9dda3ceb523de5c310b1e549e38d55d3a49a9fba8a5580a6f9e807f731d8ae.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/dd9dda3ceb523de5c310b1e549e38d55d3a49a9fba8a5580a6f9e807f731d8ae.jpg)

![f8ad6c31a872643894ba9254e0a49025a9a6ddf3683cc6740fad19a35e28e9dc.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/f8ad6c31a872643894ba9254e0a49025a9a6ddf3683cc6740fad19a35e28e9dc.jpg)

![fb27ab92fe633d16413b79ee096d0670952cffda13702e3c0d382677e47eb362.jpg](../icml_results/268_LOCATE%203D_%20Real-World%20Object%20Localization%20via%20Self-Supervised%20Learning%20in%203D/tables/fb27ab92fe633d16413b79ee096d0670952cffda13702e3c0d382677e47eb362.jpg)

## Is Complex Query Answering Really Complex?


### Images

![0e4e7b8c1737400a9a1ed4c17c0e11a0fde4717b6831dcc8765c460f801441ec.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/0e4e7b8c1737400a9a1ed4c17c0e11a0fde4717b6831dcc8765c460f801441ec.jpg)

![1be393c52bbba58f598e9d02b088e1a28b948da16d3bd259a9105e721270f0be.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/1be393c52bbba58f598e9d02b088e1a28b948da16d3bd259a9105e721270f0be.jpg)

![2a54f3d3fee3ef76dd57ac5ae4f8e3970052882c5b398e5065b86cddecfb1643.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/2a54f3d3fee3ef76dd57ac5ae4f8e3970052882c5b398e5065b86cddecfb1643.jpg)

![359f72eb7f55272734c1463073bf0a1228e850d5907b438b331425c7ce164e54.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/359f72eb7f55272734c1463073bf0a1228e850d5907b438b331425c7ce164e54.jpg)

![509fe0f998a4da9c33f5b7caff41580db64fe88342f5c89368cc7da8d929142a.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/509fe0f998a4da9c33f5b7caff41580db64fe88342f5c89368cc7da8d929142a.jpg)

![53330b5d3b2b7c26f5d2e4a05af3aa92e7bfba4cee7253f530928bfe2860484b.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/53330b5d3b2b7c26f5d2e4a05af3aa92e7bfba4cee7253f530928bfe2860484b.jpg)

![e5fa7e0d30a0a8d3a29017c93e892969e8ee26f1c98394a779785c00f454d950.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/e5fa7e0d30a0a8d3a29017c93e892969e8ee26f1c98394a779785c00f454d950.jpg)

![ecfa4b724b58fe9a022716cb0f3c6a25d5d91a60baf630b9508e8aa6ddaf2399.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/ecfa4b724b58fe9a022716cb0f3c6a25d5d91a60baf630b9508e8aa6ddaf2399.jpg)

![fd52df24c5003e3e6f0d44eab96d3078235169fe9d113bd684f50dea3958a2f3.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/images/fd52df24c5003e3e6f0d44eab96d3078235169fe9d113bd684f50dea3958a2f3.jpg)

### Tables

![12e2c5679d494f0c0e38085352e824f827e25d69698e2e8f4e298c176a2998f0.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/12e2c5679d494f0c0e38085352e824f827e25d69698e2e8f4e298c176a2998f0.jpg)

![1a99d83cbf96be3d45567e13708abd7ae6d0dc2f9a5f27eb00bf8ee6d5d2dd19.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/1a99d83cbf96be3d45567e13708abd7ae6d0dc2f9a5f27eb00bf8ee6d5d2dd19.jpg)

![20509ffdcd67a1f9cdca62e74549ab53e5ec7652cfb3716beab5009f2bf77aeb.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/20509ffdcd67a1f9cdca62e74549ab53e5ec7652cfb3716beab5009f2bf77aeb.jpg)

![25ec591dde97f2ac18ff975383aa75be5641e168006e8b772f992798efdec44c.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/25ec591dde97f2ac18ff975383aa75be5641e168006e8b772f992798efdec44c.jpg)

![45e4902410b4b77b14bd3f8fad58cc15ffae965a78318d205e75e92abf0fcf3e.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/45e4902410b4b77b14bd3f8fad58cc15ffae965a78318d205e75e92abf0fcf3e.jpg)

![69f6bc43825daea1cd46780ca7c2463ab8b8588cfffacaca8204928685d4ef85.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/69f6bc43825daea1cd46780ca7c2463ab8b8588cfffacaca8204928685d4ef85.jpg)

![72ce370556f88e577e9e0553f97392c53e5b1c6f625efc5d532fbe34762d0be8.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/72ce370556f88e577e9e0553f97392c53e5b1c6f625efc5d532fbe34762d0be8.jpg)

![7315de504f9d12b82a6a8e9e42d789c62aafd7e350f3d8323897e096c652bf4d.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/7315de504f9d12b82a6a8e9e42d789c62aafd7e350f3d8323897e096c652bf4d.jpg)

![7efd352d94d33dc3af2b61cdf53b53935b37e792395fb1317cabc74a7d561c79.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/7efd352d94d33dc3af2b61cdf53b53935b37e792395fb1317cabc74a7d561c79.jpg)

![85752b2c3f08319803a6883e5cb1a7e1448030eeab8dd4ce076b84093225af6a.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/85752b2c3f08319803a6883e5cb1a7e1448030eeab8dd4ce076b84093225af6a.jpg)

![8bc476152ff85584856fae25adaee60e62397315a692c87938c90facf7f77f7e.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/8bc476152ff85584856fae25adaee60e62397315a692c87938c90facf7f77f7e.jpg)

![97d5291129e5184991cdb44dd4e821eab97947b00ea94d962a3af1b9cc9b0b20.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/97d5291129e5184991cdb44dd4e821eab97947b00ea94d962a3af1b9cc9b0b20.jpg)

![b96f158ee29db12071ce2b1d4098ee00dbd743fecfb434b9cd41b22e73c52098.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/b96f158ee29db12071ce2b1d4098ee00dbd743fecfb434b9cd41b22e73c52098.jpg)

![c352de74f525ca4f6c09a7fce349c77f15964fbb6fced1b86892a46967f0f3ce.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/c352de74f525ca4f6c09a7fce349c77f15964fbb6fced1b86892a46967f0f3ce.jpg)

![cc9e4cff50352ddaec1548d2b8f9a114247f85581f73e5d7f6521a6599e5e58d.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/cc9e4cff50352ddaec1548d2b8f9a114247f85581f73e5d7f6521a6599e5e58d.jpg)

![cdd3c3c23834e04d2bdb4d02bdff6169401749d85b5700c5d2136b10b2c38506.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/cdd3c3c23834e04d2bdb4d02bdff6169401749d85b5700c5d2136b10b2c38506.jpg)

![d80065eb00c83fc0f5d1caa573687eb33ac769f3b4c37a5e8880f7ea275420c9.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/d80065eb00c83fc0f5d1caa573687eb33ac769f3b4c37a5e8880f7ea275420c9.jpg)

![df3d61ac2ccb7a270f81f5f2db93c6e7e9db33865305bd548ba421871a474a6e.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/df3d61ac2ccb7a270f81f5f2db93c6e7e9db33865305bd548ba421871a474a6e.jpg)

![e74db42b7ca9de20d79a833957aa1b9dffd9f2afac6e6be432f171bf3b04ae7c.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/e74db42b7ca9de20d79a833957aa1b9dffd9f2afac6e6be432f171bf3b04ae7c.jpg)

![fbace084cd357a88cfd0fdccf86ace42589ef7767665b0b8ad6739ff0d87fab6.jpg](../icml_results/269_Is%20Complex%20Query%20Answering%20Really%20Complex_/tables/fbace084cd357a88cfd0fdccf86ace42589ef7767665b0b8ad6739ff0d87fab6.jpg)

## Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models


### Images

![22d0520c67b72bfa7d853a5593f45209ff016eb11a2591a27f73aa474b2ada87.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/22d0520c67b72bfa7d853a5593f45209ff016eb11a2591a27f73aa474b2ada87.jpg)

![2e679f8fc4743d7b1ae1c48348220856036843123824a9c52775e360b93739f1.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/2e679f8fc4743d7b1ae1c48348220856036843123824a9c52775e360b93739f1.jpg)

![40ace3ee64ec16265ab383226082bcd4df240f7fc5eac21d4e4d91727f750f90.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/40ace3ee64ec16265ab383226082bcd4df240f7fc5eac21d4e4d91727f750f90.jpg)

![7436f7fe1886ff0e02cad106db39f34c9e86bac8a12b854c8255e0f3b5088554.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/7436f7fe1886ff0e02cad106db39f34c9e86bac8a12b854c8255e0f3b5088554.jpg)

![7af4d8b61b50cda476cfcde43d8f638972591cc99909b1660460b248a8467e1e.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/7af4d8b61b50cda476cfcde43d8f638972591cc99909b1660460b248a8467e1e.jpg)

![83e10235f53c9e13112eac33600a599e3aa7fa260baa9620951e808c9fc14359.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/83e10235f53c9e13112eac33600a599e3aa7fa260baa9620951e808c9fc14359.jpg)

![83fefc6dbb613317bc0c18075c727fb2ace39cffc2a59168645718c93a9ab483.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/83fefc6dbb613317bc0c18075c727fb2ace39cffc2a59168645718c93a9ab483.jpg)

![9b048726c3c5b72eb8397772c66162eb048dbd2a96951f22ab03c8bc25c71cde.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/9b048726c3c5b72eb8397772c66162eb048dbd2a96951f22ab03c8bc25c71cde.jpg)

![cfdd8a41f040055f08b83ab18dc0c056f69083b17235df74a84c7157090bd8b9.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/cfdd8a41f040055f08b83ab18dc0c056f69083b17235df74a84c7157090bd8b9.jpg)

![ee6b809862cfa713471087515d171503d0840be476718f7744fb749eb46df6eb.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/images/ee6b809862cfa713471087515d171503d0840be476718f7744fb749eb46df6eb.jpg)

### Tables

![0a6c2f8051cb955b865b69fdde110b86e28cf0a92321d3f257ed3afe159c3406.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/0a6c2f8051cb955b865b69fdde110b86e28cf0a92321d3f257ed3afe159c3406.jpg)

![0b1686377915975c28862406f979a034523d5bb194be5ac1dd4dc9a2e7a55e8c.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/0b1686377915975c28862406f979a034523d5bb194be5ac1dd4dc9a2e7a55e8c.jpg)

![327e3e5933ecd397c775980f4d98465bc1883515025f57c08a72b3f32021e133.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/327e3e5933ecd397c775980f4d98465bc1883515025f57c08a72b3f32021e133.jpg)

![346239fb1c66fa5bba6ad086575ca5546542b0f00aa7193c7ee86edc7c65c8f0.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/346239fb1c66fa5bba6ad086575ca5546542b0f00aa7193c7ee86edc7c65c8f0.jpg)

![4825b54f2353565ac43707b12acd3c7732bf7f14f13b3a5c046904712fd41ef5.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/4825b54f2353565ac43707b12acd3c7732bf7f14f13b3a5c046904712fd41ef5.jpg)

![527b099dd9ac3ee5e1d70c872f51456962787fff5c10851e5a84300dadbc0b4e.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/527b099dd9ac3ee5e1d70c872f51456962787fff5c10851e5a84300dadbc0b4e.jpg)

![6360cf3e9b50c108ffddd2fe880c4774f102e19acde1b8a7f5cd0b7a52d38c34.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/6360cf3e9b50c108ffddd2fe880c4774f102e19acde1b8a7f5cd0b7a52d38c34.jpg)

![68289caaae2f3148f27b4eaa54890d764ba58e3778e7a0a85f0eda4ab347f3e8.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/68289caaae2f3148f27b4eaa54890d764ba58e3778e7a0a85f0eda4ab347f3e8.jpg)

![68f1f9225137bbea3b056c68342e1b8b74d72b377c9f20daf134ae0c34ca996a.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/68f1f9225137bbea3b056c68342e1b8b74d72b377c9f20daf134ae0c34ca996a.jpg)

![7bb0882e48c205ea99939df15208c63950eb76a90664cbc4f460184385a3a059.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/7bb0882e48c205ea99939df15208c63950eb76a90664cbc4f460184385a3a059.jpg)

![7cab5fa34733d622308d3ea61e48da0f951d0a297a131e681fea23e81344c85e.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/7cab5fa34733d622308d3ea61e48da0f951d0a297a131e681fea23e81344c85e.jpg)

![8378b3155eedf17441a6449c561948e6e28045688ce7191848deccab1702cd3e.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/8378b3155eedf17441a6449c561948e6e28045688ce7191848deccab1702cd3e.jpg)

![8a2e7a3bbb77614ecf8b2663abb2c89415add3fe2820376e030ea763aed15f3b.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/8a2e7a3bbb77614ecf8b2663abb2c89415add3fe2820376e030ea763aed15f3b.jpg)

![8c7fd35cd1760a33df0b659d7594967a3ec0062869ac4dbe5f5ceffd868349b9.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/8c7fd35cd1760a33df0b659d7594967a3ec0062869ac4dbe5f5ceffd868349b9.jpg)

![a81240ea8f70dbd9f30ab1d3b1e95b96747325c7dbb02b7ceeed5d7ede856417.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/a81240ea8f70dbd9f30ab1d3b1e95b96747325c7dbb02b7ceeed5d7ede856417.jpg)

![a9744e7fd4a3e9381683f28353d362effa620334bea3232378768e476abd0620.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/a9744e7fd4a3e9381683f28353d362effa620334bea3232378768e476abd0620.jpg)

![bbfdc95bd3968ea9945fd34e5e59d88a029830cff9d3bfec191da6b8e1c0db01.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/bbfdc95bd3968ea9945fd34e5e59d88a029830cff9d3bfec191da6b8e1c0db01.jpg)

![d710e265522709b3b532b5ac40b7d963020026c0b58a667d7fdfccfa761d0b0c.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/d710e265522709b3b532b5ac40b7d963020026c0b58a667d7fdfccfa761d0b0c.jpg)

![da2112c5746362434955e7a3e3eb8e32fcc103ffa8d10b378e11e5117ad2fbed.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/da2112c5746362434955e7a3e3eb8e32fcc103ffa8d10b378e11e5117ad2fbed.jpg)

![de9cfe829218cc4be61f6218df34461e252beca0dc542f3550c6e0ee34fad3c9.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/de9cfe829218cc4be61f6218df34461e252beca0dc542f3550c6e0ee34fad3c9.jpg)

![f2460a814f001d80e274a73ae2ba6473473f3c09b940e063680d541aa706714c.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/f2460a814f001d80e274a73ae2ba6473473f3c09b940e063680d541aa706714c.jpg)

![f8f87c2d6da8539d7fe0eba03087de84e1c5253c47f5e8613b5f6482a77a3c20.jpg](../icml_results/270_Layer-wise%20Alignment_%20Examining%20Safety%20Alignment%20Across%20Image%20Encoder%20Layers%20in%20Vision%20Language%20Mode/tables/f8f87c2d6da8539d7fe0eba03087de84e1c5253c47f5e8613b5f6482a77a3c20.jpg)

## Identifying Causal Direction via Variational Bayesian Compression


### Images

![6d65ee47cbde5d27064305076a2dd906b547fb691e4b8241ea68520b25a25515.jpg](../icml_results/271_Identifying%20Causal%20Direction%20via%20Variational%20Bayesian%20Compression/images/6d65ee47cbde5d27064305076a2dd906b547fb691e4b8241ea68520b25a25515.jpg)

![c18394af43a7c1788248e11bd6a7870567e3ba624137f3511fa6fc3282519c82.jpg](../icml_results/271_Identifying%20Causal%20Direction%20via%20Variational%20Bayesian%20Compression/images/c18394af43a7c1788248e11bd6a7870567e3ba624137f3511fa6fc3282519c82.jpg)

### Tables

![18b7a25d23010681d31ee0649d3a2011a18ed6028bddb3e341ef07d13f0bc53f.jpg](../icml_results/271_Identifying%20Causal%20Direction%20via%20Variational%20Bayesian%20Compression/tables/18b7a25d23010681d31ee0649d3a2011a18ed6028bddb3e341ef07d13f0bc53f.jpg)

![6184cbe5078fa914989d2402aa8ba1a33338ebeea09a1741b9d360ecb28ddf31.jpg](../icml_results/271_Identifying%20Causal%20Direction%20via%20Variational%20Bayesian%20Compression/tables/6184cbe5078fa914989d2402aa8ba1a33338ebeea09a1741b9d360ecb28ddf31.jpg)

![be7f2a4ea3f8fabc299a1aa8a7083cd8dbb736d3d7865dab3ceed0eb77e45324.jpg](../icml_results/271_Identifying%20Causal%20Direction%20via%20Variational%20Bayesian%20Compression/tables/be7f2a4ea3f8fabc299a1aa8a7083cd8dbb736d3d7865dab3ceed0eb77e45324.jpg)

![eecc19b0a30b7d5ba546565b8036b32be030a89febd044f101b4e25c1540d102.jpg](../icml_results/271_Identifying%20Causal%20Direction%20via%20Variational%20Bayesian%20Compression/tables/eecc19b0a30b7d5ba546565b8036b32be030a89febd044f101b4e25c1540d102.jpg)

## Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes


### Images

![6228212804509c63324fbd49c60d9ed4119781bb9218ae657a647bd774b90b7c.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/6228212804509c63324fbd49c60d9ed4119781bb9218ae657a647bd774b90b7c.jpg)

![66b27dba21aaadb104efe056f0ab8204029f075fa60790ab6cb7bcb7091f6e67.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/66b27dba21aaadb104efe056f0ab8204029f075fa60790ab6cb7bcb7091f6e67.jpg)

![74c99f19ac0dea149a956e760dd50341b17be7e74462616077431187af5ae012.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/74c99f19ac0dea149a956e760dd50341b17be7e74462616077431187af5ae012.jpg)

![89aaa29436020665977b3b273b82ff8cbf6bce70e9ab4da61bf9631c25b53a95.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/89aaa29436020665977b3b273b82ff8cbf6bce70e9ab4da61bf9631c25b53a95.jpg)

![90d96b6613e74257395fb34bbbe2c1190c29b501b49baf9d3a1080b898da1552.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/90d96b6613e74257395fb34bbbe2c1190c29b501b49baf9d3a1080b898da1552.jpg)

![b968dfe91ac1bda0306b5e9bb7f19cee8db2465876417670d0463e2c0702b13a.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/b968dfe91ac1bda0306b5e9bb7f19cee8db2465876417670d0463e2c0702b13a.jpg)

![d606d3d4f6b8669c41f6246ea703e9524ffe2797f69705c0bd68677f70a9de1e.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/d606d3d4f6b8669c41f6246ea703e9524ffe2797f69705c0bd68677f70a9de1e.jpg)

![db4ff4409be67f8d5e663aad5304d587a5dafb13abf7238410c27bb1d6880719.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/images/db4ff4409be67f8d5e663aad5304d587a5dafb13abf7238410c27bb1d6880719.jpg)

### Tables

![6715b612e09aca2f5bbd753a72d977b6c1e9d85c582cb3b03bac232fc0329d86.jpg](../icml_results/272_Understanding%20and%20Mitigating%20Memorization%20in%20Generative%20Models%20via%20Sharpness%20of%20Probability%20Landscap/tables/6715b612e09aca2f5bbd753a72d977b6c1e9d85c582cb3b03bac232fc0329d86.jpg)

## am-ELO: A Stable Framework for Arena-based LLM Evaluation


### Images

![00a9265cda449e74dcb0a63fc61bdd2da593b5f469ee73a005797fa61278bcf3.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/images/00a9265cda449e74dcb0a63fc61bdd2da593b5f469ee73a005797fa61278bcf3.jpg)

![31ee2bf6516d18a3a3b14ddc4ac8b8150704fcc8bd54c395a7231a32330b1414.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/images/31ee2bf6516d18a3a3b14ddc4ac8b8150704fcc8bd54c395a7231a32330b1414.jpg)

![5904c3850b3657988beaf8fff651d21f3a226a01d2d6036dfc57e0181a3e1ac7.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/images/5904c3850b3657988beaf8fff651d21f3a226a01d2d6036dfc57e0181a3e1ac7.jpg)

![67beb8a4c878063c27e1edd35708a92bbefd70bde03322669eb1f9455e86f9ba.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/images/67beb8a4c878063c27e1edd35708a92bbefd70bde03322669eb1f9455e86f9ba.jpg)

![81d3c369d15dae5859596b77bcf0b626b2ddd5be9913219114ad79d1f6cb9250.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/images/81d3c369d15dae5859596b77bcf0b626b2ddd5be9913219114ad79d1f6cb9250.jpg)

![93e9ccb0632240c71cad3d20a5e8e426bf428fcb6733fe7bc03ad794a5363452.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/images/93e9ccb0632240c71cad3d20a5e8e426bf428fcb6733fe7bc03ad794a5363452.jpg)

![bb7f71100ee3e1cc8d0906c628d683b0d421f1954711f524c639a2b3e6f6482c.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/images/bb7f71100ee3e1cc8d0906c628d683b0d421f1954711f524c639a2b3e6f6482c.jpg)

### Tables

![19ce0cf6d154069837d48501d7b34f4689c99643a0248d65c21f4b7ef4002a3b.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/tables/19ce0cf6d154069837d48501d7b34f4689c99643a0248d65c21f4b7ef4002a3b.jpg)

![2c544b1097015c407fcc565cc474b5c4d6701cffca9b6cd2c6f7294e490f6ea0.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/tables/2c544b1097015c407fcc565cc474b5c4d6701cffca9b6cd2c6f7294e490f6ea0.jpg)

![5eb4231e7671ed7de5badbd5b2329c690089d492fc450cf5cdda1ff36854bfd6.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/tables/5eb4231e7671ed7de5badbd5b2329c690089d492fc450cf5cdda1ff36854bfd6.jpg)

![b94f675239d3c43089249229ecfe9c27b556397f4ad9b9c0b10cb8c6c03d7d81.jpg](../icml_results/273_am-ELO_%20A%20Stable%20Framework%20for%20Arena-based%20LLM%20Evaluation/tables/b94f675239d3c43089249229ecfe9c27b556397f4ad9b9c0b10cb8c6c03d7d81.jpg)

## Flopping for FLOPs: Leveraging Equivariance for Computational Efficiency


### Images

![01ac11fe7c85c686d4c3d63a0e84a49ee3dcdd36c411a1258982a7660d3fdf9f.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/images/01ac11fe7c85c686d4c3d63a0e84a49ee3dcdd36c411a1258982a7660d3fdf9f.jpg)

![41abd8ebb349250df9ad77598b517b5545d4429b9abf41dd004a1178eb8df9db.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/images/41abd8ebb349250df9ad77598b517b5545d4429b9abf41dd004a1178eb8df9db.jpg)

![9f323ecbf72b01c0c9b754bb9a1ee46e023d5f5fe7100fc1833c7a5b0c93ea07.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/images/9f323ecbf72b01c0c9b754bb9a1ee46e023d5f5fe7100fc1833c7a5b0c93ea07.jpg)

![c2a769738565f1ce804ead3a692d086fb9c7305a541cbf8f1164b7a18c554ebc.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/images/c2a769738565f1ce804ead3a692d086fb9c7305a541cbf8f1164b7a18c554ebc.jpg)

![c7b310b28c171176423e0621694a621648807607a447ff7135713734d8495423.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/images/c7b310b28c171176423e0621694a621648807607a447ff7135713734d8495423.jpg)

![f3936c535addc35a6b22517e9f54f73ffaadadebca135d7a204cd4687f713cf4.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/images/f3936c535addc35a6b22517e9f54f73ffaadadebca135d7a204cd4687f713cf4.jpg)

### Tables

![107daabf93eb390127b282fb8bb6123fb0410336f629e715334c36a681bfaf8d.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/tables/107daabf93eb390127b282fb8bb6123fb0410336f629e715334c36a681bfaf8d.jpg)

![b799201fda6a8f7aa414d712f086442e1329a53451bc7bca010bd082a9048a8d.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/tables/b799201fda6a8f7aa414d712f086442e1329a53451bc7bca010bd082a9048a8d.jpg)

![c321652d4eaa5db34341b6c1b904d0ac90c275b5c0102855c96b737a6efec132.jpg](../icml_results/274_Flopping%20for%20FLOPs_%20Leveraging%20Equivariance%20for%20Computational%20Efficiency/tables/c321652d4eaa5db34341b6c1b904d0ac90c275b5c0102855c96b737a6efec132.jpg)

## Enforcing Latent Euclidean Geometry in Single-Cell VAEs for Manifold Interpolation


### Images

![03dd7eeb2b6e71694d4b9ba031b7ca6827b652f28144f5241987cb775a9d9003.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/03dd7eeb2b6e71694d4b9ba031b7ca6827b652f28144f5241987cb775a9d9003.jpg)

![0aa4bc0daec1cc735055d1b16075a0a1f76fceda52cedb27cc90be2365c743f3.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/0aa4bc0daec1cc735055d1b16075a0a1f76fceda52cedb27cc90be2365c743f3.jpg)

![0da4fc89361d7863e60fc007a38cb66208e1c6510fc4894a31d58ad752688f8f.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/0da4fc89361d7863e60fc007a38cb66208e1c6510fc4894a31d58ad752688f8f.jpg)

![23307e16001a35bd7a4dd2f3596ab422bb483f87a26d6e65347547826facdb4d.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/23307e16001a35bd7a4dd2f3596ab422bb483f87a26d6e65347547826facdb4d.jpg)

![3ec8fab8fac67ca22643b5ac6dfcb4eaabc8877ed88b970fc0266d985798d383.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/3ec8fab8fac67ca22643b5ac6dfcb4eaabc8877ed88b970fc0266d985798d383.jpg)

![5f379f616ff141860350dd72735bf400b9b9a687e996fbc9eaf7b0dcfa4c7c05.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/5f379f616ff141860350dd72735bf400b9b9a687e996fbc9eaf7b0dcfa4c7c05.jpg)

![7a7e3404965483ef4591619382cb3ca0b5620f1b9f5ab3b88142cc8561a68119.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/7a7e3404965483ef4591619382cb3ca0b5620f1b9f5ab3b88142cc8561a68119.jpg)

![80fc277b53791925b0dcd4457980c674b4a06938b89529b19f7374b747f0a5e7.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/80fc277b53791925b0dcd4457980c674b4a06938b89529b19f7374b747f0a5e7.jpg)

![82fed81e96ea6906dcf40d00b543ad6ef251076c519ec9e8e959949708e38019.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/82fed81e96ea6906dcf40d00b543ad6ef251076c519ec9e8e959949708e38019.jpg)

![8b77c2eda53de70f02f854c9eaaa9195b0fa182e1d604bbb7aebb22b0579fd90.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/8b77c2eda53de70f02f854c9eaaa9195b0fa182e1d604bbb7aebb22b0579fd90.jpg)

![9ca19b5d92578e75210026078daef09c039122b91d05e0edc0183c6b6b59924b.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/9ca19b5d92578e75210026078daef09c039122b91d05e0edc0183c6b6b59924b.jpg)

![c64922af58938004acb63b5875402852775e6e2b165f907c708e8dd07c1700f5.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/c64922af58938004acb63b5875402852775e6e2b165f907c708e8dd07c1700f5.jpg)

![f7efd850363283b1d919922704a781ca038e6868bc39d1fabb2bf297099b4fda.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/f7efd850363283b1d919922704a781ca038e6868bc39d1fabb2bf297099b4fda.jpg)

![fbf0a426a79cc5a36f930980cf5190a93d3a01a923c5e080d72e0bd1a53e43d3.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/images/fbf0a426a79cc5a36f930980cf5190a93d3a01a923c5e080d72e0bd1a53e43d3.jpg)

### Tables

![05baf944f543ec6ea71a2e6eb59e9f1680638af531acdfa73cf259c79e10e305.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/05baf944f543ec6ea71a2e6eb59e9f1680638af531acdfa73cf259c79e10e305.jpg)

![18bd641fdee465e014372d9163e7b53d7c0e728792d8c23cf297a30e0d3bff65.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/18bd641fdee465e014372d9163e7b53d7c0e728792d8c23cf297a30e0d3bff65.jpg)

![1ebf1990dc93a8172bcf6ab2ac058ee5356eb4c292e498a40cf7e5caa7d0131c.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/1ebf1990dc93a8172bcf6ab2ac058ee5356eb4c292e498a40cf7e5caa7d0131c.jpg)

![2311a6552fdfa2f25ad0409fba89837c11c6cb1e80132efaa1a15fbcd64b56b7.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/2311a6552fdfa2f25ad0409fba89837c11c6cb1e80132efaa1a15fbcd64b56b7.jpg)

![5c7ed289fb512a0ff79696c4f299c31cfd590da91b684053adeb48b43d21ab3c.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/5c7ed289fb512a0ff79696c4f299c31cfd590da91b684053adeb48b43d21ab3c.jpg)

![682a651f765b21e2e057feec3f50268b21b6b9c91ae6835a97338bb4a7f32422.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/682a651f765b21e2e057feec3f50268b21b6b9c91ae6835a97338bb4a7f32422.jpg)

![841b53c8471f9cc64c3e05e3ba868ca73b95085782abb5d8a3a8e18636888ed6.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/841b53c8471f9cc64c3e05e3ba868ca73b95085782abb5d8a3a8e18636888ed6.jpg)

![dd94ef34c64dccc0da2a6c5a368cb235d6a86a82cbe682bac111b48a27606571.jpg](../icml_results/275_Enforcing%20Latent%20Euclidean%20Geometry%20in%20Single-Cell%20VAEs%20for%20Manifold%20Interpolation/tables/dd94ef34c64dccc0da2a6c5a368cb235d6a86a82cbe682bac111b48a27606571.jpg)

## Monte-Carlo Tree Search with Uncertainty Propagation via Optimal Transport


### Images

![d767e7af32325c2e32e487893c70051b9ef47d796fb547542f9846b3cd460022.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/images/d767e7af32325c2e32e487893c70051b9ef47d796fb547542f9846b3cd460022.jpg)

![f28e8be4128583cbbdd5ed4965b314dab2c9625f54129e16072fa16d4b9c2e97.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/images/f28e8be4128583cbbdd5ed4965b314dab2c9625f54129e16072fa16d4b9c2e97.jpg)

### Tables

![0f0f075b0dc03b0d73645b729f90a8095131d5889050401165c330f4e5543a1f.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/tables/0f0f075b0dc03b0d73645b729f90a8095131d5889050401165c330f4e5543a1f.jpg)

![4639f5a0c80a34d954a5d068517c4dd78309fb47668f7cb96ef6f669ff25d6cb.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/tables/4639f5a0c80a34d954a5d068517c4dd78309fb47668f7cb96ef6f669ff25d6cb.jpg)

![47bf4e9a0bf19d3bfcbe3db959cdc9c367db7b7e8748aa45204ea87d70dfb898.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/tables/47bf4e9a0bf19d3bfcbe3db959cdc9c367db7b7e8748aa45204ea87d70dfb898.jpg)

![5b9233cf2e7653de7d5e791269139c8af420abb428942db9907ffdcc0ca7825d.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/tables/5b9233cf2e7653de7d5e791269139c8af420abb428942db9907ffdcc0ca7825d.jpg)

![92e671e46a1540a37bc0823df026f05872f2758dc77bacaf61340b90c98bbd38.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/tables/92e671e46a1540a37bc0823df026f05872f2758dc77bacaf61340b90c98bbd38.jpg)

![fb43ceb8bc7f45e60fc42f810617f143d848628ccf6b0d77061f63ca83db3002.jpg](../icml_results/276_Monte-Carlo%20Tree%20Search%20with%20Uncertainty%20Propagation%20via%20Optimal%20Transport/tables/fb43ceb8bc7f45e60fc42f810617f143d848628ccf6b0d77061f63ca83db3002.jpg)

## Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger


### Images

![009c9ad0fe022781f64217a4f1e1e833c9537f8fa55ffde52b40f33c71d1d168.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/009c9ad0fe022781f64217a4f1e1e833c9537f8fa55ffde52b40f33c71d1d168.jpg)

![037b0ee36c06a91be794b9f621ce7e38eff60a0fc7f61d5075f2ecbd8b453171.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/037b0ee36c06a91be794b9f621ce7e38eff60a0fc7f61d5075f2ecbd8b453171.jpg)

![061d6c25a7f46c9ac28225f2b25c2736a165e547b02aac8e051acb0cdd16a8e2.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/061d6c25a7f46c9ac28225f2b25c2736a165e547b02aac8e051acb0cdd16a8e2.jpg)

![0c51493a473d426aaab44533414095862d5e8a23b3b141b9bec109e14761b460.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/0c51493a473d426aaab44533414095862d5e8a23b3b141b9bec109e14761b460.jpg)

![103dd80c1280c9edd9c51dd7c2348b483e849c8ef6e13ae5cc487d89b8fe7a96.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/103dd80c1280c9edd9c51dd7c2348b483e849c8ef6e13ae5cc487d89b8fe7a96.jpg)

![171dcaf2b2f3561ed54485412f7f6be414456f165c792c83f08d6a8f6a776c15.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/171dcaf2b2f3561ed54485412f7f6be414456f165c792c83f08d6a8f6a776c15.jpg)

![185c68f8fe6e87114cc2c679fd007cf9f08634fcf7cf02ab4d0c352c9753915b.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/185c68f8fe6e87114cc2c679fd007cf9f08634fcf7cf02ab4d0c352c9753915b.jpg)

![20d3a1aca961c676ced680179fa3dd5b98bb41e7046c3f0c89b4cf8e9b4ce08e.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/20d3a1aca961c676ced680179fa3dd5b98bb41e7046c3f0c89b4cf8e9b4ce08e.jpg)

![2aa5ac52a47d970914e837c9725677b66335e48c7e07a5ec07d88dcfdf34334a.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/2aa5ac52a47d970914e837c9725677b66335e48c7e07a5ec07d88dcfdf34334a.jpg)

![43133ab6d722b52f8e5134722b92e8168764af0a0ffdcac759a0b52921c97196.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/43133ab6d722b52f8e5134722b92e8168764af0a0ffdcac759a0b52921c97196.jpg)

![4e2c9bf9010980d9b696b6d07e03badb51ad0c2d1fd2302fc735542821475037.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/4e2c9bf9010980d9b696b6d07e03badb51ad0c2d1fd2302fc735542821475037.jpg)

![6ceb0421f66bf7bf249b929a70f811014581f3051821928f77147a3e0c779221.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/6ceb0421f66bf7bf249b929a70f811014581f3051821928f77147a3e0c779221.jpg)

![826371c49d6e389eae1af5269e35be4b0721a2ee3dc3ab2c142adc7a886d24fa.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/826371c49d6e389eae1af5269e35be4b0721a2ee3dc3ab2c142adc7a886d24fa.jpg)

![9cb786a564c7a1ebe9f4fbe25efa5dbff03825974c192f5aa77103be290f98bd.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/9cb786a564c7a1ebe9f4fbe25efa5dbff03825974c192f5aa77103be290f98bd.jpg)

![a068551a0dc5cf9a86c602600a73498161181aa03cf27b0209c1245b06aa579d.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/a068551a0dc5cf9a86c602600a73498161181aa03cf27b0209c1245b06aa579d.jpg)

![c205178672c1425abab33a7a1ba611ca15d625c45aa1c67ee90fba0ff8f8faf7.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/c205178672c1425abab33a7a1ba611ca15d625c45aa1c67ee90fba0ff8f8faf7.jpg)

![f2ff6fb7062d627df596369c78f892ab186ada26b38c80d1477f4222347bf90a.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/f2ff6fb7062d627df596369c78f892ab186ada26b38c80d1477f4222347bf90a.jpg)

![fee1a1d275023661c2f1652d4dc139c1b77fce7f7365bd39cd912c948f1a1d76.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/images/fee1a1d275023661c2f1652d4dc139c1b77fce7f7365bd39cd912c948f1a1d76.jpg)

### Tables

![0c78ef087464aa68255c858b67df2bdd519c0dce0063135345cdc68c9eb1bf78.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/0c78ef087464aa68255c858b67df2bdd519c0dce0063135345cdc68c9eb1bf78.jpg)

![74d9ea48d444fcec53092e83128dade0ca81b9b49e27cf3605c8ca4b1fe97432.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/74d9ea48d444fcec53092e83128dade0ca81b9b49e27cf3605c8ca4b1fe97432.jpg)

![7c4cd05b066420fde8c3af57c6703765906ebfe3e13bdf11c3f159cd89e79fdb.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/7c4cd05b066420fde8c3af57c6703765906ebfe3e13bdf11c3f159cd89e79fdb.jpg)

![91d97b7c109e270b8303d75d05e309701247869c57b32aa4f8c725001d0ee597.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/91d97b7c109e270b8303d75d05e309701247869c57b32aa4f8c725001d0ee597.jpg)

![9ca4b66b8c7d6cdd34e727b872626e3449a80f68ef58dac38a2d690edcd3eff0.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/9ca4b66b8c7d6cdd34e727b872626e3449a80f68ef58dac38a2d690edcd3eff0.jpg)

![a501c2987c2f985a62ed1a26476cc629f870304dc07ddadae836e825475c1e97.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/a501c2987c2f985a62ed1a26476cc629f870304dc07ddadae836e825475c1e97.jpg)

![af7314501c78611a518cb09bad0e71bafc50875e57d17b9dd9ab45f39524263f.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/af7314501c78611a518cb09bad0e71bafc50875e57d17b9dd9ab45f39524263f.jpg)

![f402f6f491bf571beed7e8fdc8f685d68d903dd6090504b13a1626172a4d58e9.jpg](../icml_results/277_Re-ranking%20Reasoning%20Context%20with%20Tree%20Search%20Makes%20Large%20Vision-Language%20Models%20Stronger/tables/f402f6f491bf571beed7e8fdc8f685d68d903dd6090504b13a1626172a4d58e9.jpg)

## Robust Automatic Modulation Classification with Fuzzy Regularization


### Images

![3986b3013b460a1cf7ee8592bc78b51b9d6c25e8059ff7fa45801386f6afef20.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/images/3986b3013b460a1cf7ee8592bc78b51b9d6c25e8059ff7fa45801386f6afef20.jpg)

![4fa558bf0feaad990ed6d9f86b44fc789d899a0fef72fdcf1d6f300ccabfd640.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/images/4fa558bf0feaad990ed6d9f86b44fc789d899a0fef72fdcf1d6f300ccabfd640.jpg)

![56f14767d5a60d5962c21c0159b97dc843027655ff96956b5d366e7582135978.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/images/56f14767d5a60d5962c21c0159b97dc843027655ff96956b5d366e7582135978.jpg)

![874322633d2a96660a2baa94b1cb093872b22b3dc68fb383053dec181ca03ab6.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/images/874322633d2a96660a2baa94b1cb093872b22b3dc68fb383053dec181ca03ab6.jpg)

![b832a58dc6185def333aec8cf07f8f367d8b066972e55c57ed711d8510c18347.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/images/b832a58dc6185def333aec8cf07f8f367d8b066972e55c57ed711d8510c18347.jpg)

![ba1fe6492417a39d396de1a1e5084a7032106d268d73c99baec20daaf2737a98.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/images/ba1fe6492417a39d396de1a1e5084a7032106d268d73c99baec20daaf2737a98.jpg)

![f353ae0dfca84c7d251f3232d7e309298e24195aaf6492936af3afddd9c6e6e5.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/images/f353ae0dfca84c7d251f3232d7e309298e24195aaf6492936af3afddd9c6e6e5.jpg)

### Tables

![4f2458b1a9e1a3568098920a21d6571fbdf7bab7d0637bb42a2da44ee16150ee.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/tables/4f2458b1a9e1a3568098920a21d6571fbdf7bab7d0637bb42a2da44ee16150ee.jpg)

![cab58b6dda82c28b1c9f245cedbc36a3cd4f7e71091729d27f4b4533b81c40da.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/tables/cab58b6dda82c28b1c9f245cedbc36a3cd4f7e71091729d27f4b4533b81c40da.jpg)

![e3951c5086b2b0a52c89061a356b2c08aa9f5b1ab7f96cca8b8f16eb4cc0ab61.jpg](../icml_results/278_Robust%20Automatic%20Modulation%20Classification%20with%20Fuzzy%20Regularization/tables/e3951c5086b2b0a52c89061a356b2c08aa9f5b1ab7f96cca8b8f16eb4cc0ab61.jpg)

## Diffusion-based Adversarial Purification from the Perspective of the Frequency Domain


### Images

![15817d6376e1a458a441e8b386440872a4a89b26bd9e59b7352b474666e0e45d.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/15817d6376e1a458a441e8b386440872a4a89b26bd9e59b7352b474666e0e45d.jpg)

![1a621edcaa69317fd982bba6252a6b990a1ed36d84daa32b37f39c069656e9a9.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/1a621edcaa69317fd982bba6252a6b990a1ed36d84daa32b37f39c069656e9a9.jpg)

![1e657fbfa6458c03fdfb321946fd3b5ec50aef0b31e706f15ef86bb32c83d647.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/1e657fbfa6458c03fdfb321946fd3b5ec50aef0b31e706f15ef86bb32c83d647.jpg)

![460eeae3b48599586018414433ba147aa3efcdbaf3ae292bd0f511ddb3e3df9a.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/460eeae3b48599586018414433ba147aa3efcdbaf3ae292bd0f511ddb3e3df9a.jpg)

![7519201c4bd79b8daf4677a7b04eecd673635e1c31978c47822c2c3458e06d31.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/7519201c4bd79b8daf4677a7b04eecd673635e1c31978c47822c2c3458e06d31.jpg)

![7f5a8606475658b9f181e2cada6fcc8b433195b7659bbb9778de3800b32fde4e.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/7f5a8606475658b9f181e2cada6fcc8b433195b7659bbb9778de3800b32fde4e.jpg)

![9a45cceb439b326ad7aabbd0d695108dc3424ee60b2291e1256c2756caa7cf98.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/9a45cceb439b326ad7aabbd0d695108dc3424ee60b2291e1256c2756caa7cf98.jpg)

![c937e9f62919e4129542a1be7f3a81b5d0f0cac1bc44ad3b0fab2ae5ced3babe.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/c937e9f62919e4129542a1be7f3a81b5d0f0cac1bc44ad3b0fab2ae5ced3babe.jpg)

![d1f1e0610933193b20fbdc5447b71e3f0da90dc990b3551258ce6c3124ac1d5c.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/d1f1e0610933193b20fbdc5447b71e3f0da90dc990b3551258ce6c3124ac1d5c.jpg)

![de226cfc6403232349f1f77ba98808d9ab5d451e13739ee051065ea98526f83f.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/de226cfc6403232349f1f77ba98808d9ab5d451e13739ee051065ea98526f83f.jpg)

![e6465e7a6106c93a4186f6c359f33f9f4de2eb8bc87371f25d35f4a967324f98.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/e6465e7a6106c93a4186f6c359f33f9f4de2eb8bc87371f25d35f4a967324f98.jpg)

![f9bb4ea6a22f8700efb74ce55f9407102adff61fb5e20183b1943632a1f730b4.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/f9bb4ea6a22f8700efb74ce55f9407102adff61fb5e20183b1943632a1f730b4.jpg)

![fa86691ac1e9e58d1f030301f92e6cdf35328a4e4f7cf97a68e360b7db713258.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/images/fa86691ac1e9e58d1f030301f92e6cdf35328a4e4f7cf97a68e360b7db713258.jpg)

### Tables

![02131e1237c534db24e6a2a27e47a1faed5d094540bacf978634f25eda88ac8c.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/02131e1237c534db24e6a2a27e47a1faed5d094540bacf978634f25eda88ac8c.jpg)

![3fd7a403caa78cfde0cc9d3b0cadb114b59328d4006b33257c18ea5e5af5cad1.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/3fd7a403caa78cfde0cc9d3b0cadb114b59328d4006b33257c18ea5e5af5cad1.jpg)

![4af7b742e53e52d0393ce65f6bfbe5d17f1b76b5a670093c474f35c7e5e1f1d9.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/4af7b742e53e52d0393ce65f6bfbe5d17f1b76b5a670093c474f35c7e5e1f1d9.jpg)

![54603f7a6564c7090be491bc28ce6d2499b1218b75b9785e33c6ee12ae09a690.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/54603f7a6564c7090be491bc28ce6d2499b1218b75b9785e33c6ee12ae09a690.jpg)

![6f71d5e2d9171bac3a256675f9461893dea0a46a44c591f49562137ff29e5ce1.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/6f71d5e2d9171bac3a256675f9461893dea0a46a44c591f49562137ff29e5ce1.jpg)

![9230e1b010b4ecbfc26ea496ef7d2b9502f2fadc834e804e6d8d8101003d8d03.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/9230e1b010b4ecbfc26ea496ef7d2b9502f2fadc834e804e6d8d8101003d8d03.jpg)

![ab3fd895acdbca33a79911765916208b3becb42d0abf64742ce4a57fd5b82e70.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/ab3fd895acdbca33a79911765916208b3becb42d0abf64742ce4a57fd5b82e70.jpg)

![baa58f1823e8d0f7ed2130a8deb2129f7f2a86562a61d139659ed1f615270b92.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/baa58f1823e8d0f7ed2130a8deb2129f7f2a86562a61d139659ed1f615270b92.jpg)

![cad1d344a527d2564ccafc015951be923b953a2d9aeabf6ab34c8af79c804dde.jpg](../icml_results/279_Diffusion-based%20Adversarial%20Purification%20from%20the%20Perspective%20of%20the%20Frequency%20Domain/tables/cad1d344a527d2564ccafc015951be923b953a2d9aeabf6ab34c8af79c804dde.jpg)

## Not All Wrong is Bad: Using Adversarial Examples for Unlearning


### Images

![0131d1aa7ffbb82a1f3e5ce74c15b80fa57fe94dc798ade413e98d8e4e3c2c59.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/0131d1aa7ffbb82a1f3e5ce74c15b80fa57fe94dc798ade413e98d8e4e3c2c59.jpg)

![4002a033517de991e9bf21cf9242aa8ae7cff730a5b056e8d15e39aac5fb01ff.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/4002a033517de991e9bf21cf9242aa8ae7cff730a5b056e8d15e39aac5fb01ff.jpg)

![72286362e26c1f5e28ea94f8129b4151a2f76d68580ab4337e7688d9c18759cf.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/72286362e26c1f5e28ea94f8129b4151a2f76d68580ab4337e7688d9c18759cf.jpg)

![8f463499251a1c60de4864183902d6aff8e45e41a514557ee07a6a55148511b9.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/8f463499251a1c60de4864183902d6aff8e45e41a514557ee07a6a55148511b9.jpg)

![cd7714baf0a17dc613e017449e0f09f13bc9af4bd5beb93d72ef9cdc228c860b.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/cd7714baf0a17dc613e017449e0f09f13bc9af4bd5beb93d72ef9cdc228c860b.jpg)

![da969f307ea83dfb66197c9ed33e2d7356c23281763a4828ccc6c23f325cfca8.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/da969f307ea83dfb66197c9ed33e2d7356c23281763a4828ccc6c23f325cfca8.jpg)

![e2cf5f921896d8aadb365749b6ae273dc6aa6cc318c073e213f1a515285de641.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/e2cf5f921896d8aadb365749b6ae273dc6aa6cc318c073e213f1a515285de641.jpg)

![f120457e5b6b47c161d05efab2a22ec6db992b3e8b719b2cb018b79868189894.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/f120457e5b6b47c161d05efab2a22ec6db992b3e8b719b2cb018b79868189894.jpg)

![f5e695c80542983b109282ce577953cdef31295c05a0383665ef73fba5700861.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/images/f5e695c80542983b109282ce577953cdef31295c05a0383665ef73fba5700861.jpg)

### Tables

![10c907ee834f39d87cbbb6183da46316aaed756a0d0ca2065260ebebd8adc5f7.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/10c907ee834f39d87cbbb6183da46316aaed756a0d0ca2065260ebebd8adc5f7.jpg)

![1901e994e8d0fa19d0d1fa904ca071c38403a7c82f929ce5e0ae898e899a5f29.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/1901e994e8d0fa19d0d1fa904ca071c38403a7c82f929ce5e0ae898e899a5f29.jpg)

![337255b5bb57e75b44159fcf8866d9cfad920429e2d4dd8ec409b5a3846cfeca.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/337255b5bb57e75b44159fcf8866d9cfad920429e2d4dd8ec409b5a3846cfeca.jpg)

![4dc4defc88cef29d84da8c1feb088c4b8ba3c186dcc299adfc7639e198438e61.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/4dc4defc88cef29d84da8c1feb088c4b8ba3c186dcc299adfc7639e198438e61.jpg)

![674eaf6d209ba8eb0769722124c3304281b08764fe3c22595f3c34ada5c3be8f.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/674eaf6d209ba8eb0769722124c3304281b08764fe3c22595f3c34ada5c3be8f.jpg)

![6785c7a30a4a2c4014293a7e71284047d2f7b3977497d50c4a7466bbaff9a366.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/6785c7a30a4a2c4014293a7e71284047d2f7b3977497d50c4a7466bbaff9a366.jpg)

![7d6a07e8e488305be066a35132a01f07b8e557201d99b605a09e45a8f32c917c.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/7d6a07e8e488305be066a35132a01f07b8e557201d99b605a09e45a8f32c917c.jpg)

![86125ae523b7ceb525c9606715fe9aa0e67a04eafa15e2a4eb8fa0d9688bdfd9.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/86125ae523b7ceb525c9606715fe9aa0e67a04eafa15e2a4eb8fa0d9688bdfd9.jpg)

![92144273edbe83cbf41b740e2282798a48b3e08e8e50d2e52115bafaa1a7aaed.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/92144273edbe83cbf41b740e2282798a48b3e08e8e50d2e52115bafaa1a7aaed.jpg)

![a2978a9c1324ee6c7b0c2926a9569a334c33b007171294f8b689134f5928dd95.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/a2978a9c1324ee6c7b0c2926a9569a334c33b007171294f8b689134f5928dd95.jpg)

![e0904a6df07dcb44423296fc6e3937ffa68b93b9311f8e8e590e527ae8f50b78.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/e0904a6df07dcb44423296fc6e3937ffa68b93b9311f8e8e590e527ae8f50b78.jpg)

![f24abf1cce92093f6987b37102e78bcf6a1e5f0a0c44e12b567adb36d1928f22.jpg](../icml_results/280_Not%20All%20Wrong%20is%20Bad_%20Using%20Adversarial%20Examples%20for%20Unlearning/tables/f24abf1cce92093f6987b37102e78bcf6a1e5f0a0c44e12b567adb36d1928f22.jpg)

## TLLC: Transfer Learning-based Label Completion for Crowdsourcing


### Images

![1805348b7b8c92fa899bf8b25a03480a0e3653d797097c3b5a236c643c8f6117.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/1805348b7b8c92fa899bf8b25a03480a0e3653d797097c3b5a236c643c8f6117.jpg)

![1d3b7c5eae885f1a00ee7b9971d1ebecc73b6b383a81451949f121b4d7143f2a.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/1d3b7c5eae885f1a00ee7b9971d1ebecc73b6b383a81451949f121b4d7143f2a.jpg)

![45e1efca34a05c28bcd610bca6a60ace75681cf31e18cac77c0d6fcea0b04550.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/45e1efca34a05c28bcd610bca6a60ace75681cf31e18cac77c0d6fcea0b04550.jpg)

![875b8b37f0f260cc530b990ee938332bd7dd2d7160139be763ef26cdd0d05ffb.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/875b8b37f0f260cc530b990ee938332bd7dd2d7160139be763ef26cdd0d05ffb.jpg)

![879c3f82c2b91cf5ab44cc5f2eebc20a12748d62db8c6b02555ddec6e1df48c4.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/879c3f82c2b91cf5ab44cc5f2eebc20a12748d62db8c6b02555ddec6e1df48c4.jpg)

![9fc799edcad10d20b2f9e083cf63e8f2a3e06c9794cebab22491f4c5a5096849.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/9fc799edcad10d20b2f9e083cf63e8f2a3e06c9794cebab22491f4c5a5096849.jpg)

![acbbef2f31d1f7ea54f8af3205b67f3eb442f3253a272de2d425be4ce5e9a7d1.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/acbbef2f31d1f7ea54f8af3205b67f3eb442f3253a272de2d425be4ce5e9a7d1.jpg)

![b558ed0ea94de42742650cd8cb2807746e906691c7841806081fa0d83f4f12f2.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/b558ed0ea94de42742650cd8cb2807746e906691c7841806081fa0d83f4f12f2.jpg)

![d2d1bec23244bbc159d8bb3f32ee335e8fce4fc522b92110c14037e724f0ad2d.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/d2d1bec23244bbc159d8bb3f32ee335e8fce4fc522b92110c14037e724f0ad2d.jpg)

![f986c82d691386e0c9c39e80b7b8ff534653726299c560b73026ee651b21ba64.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/images/f986c82d691386e0c9c39e80b7b8ff534653726299c560b73026ee651b21ba64.jpg)

### Tables

![1b1a20e59d902f386ece9df326f7cab7f6b9416dbb7c98505d2559152bf4e7d8.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/tables/1b1a20e59d902f386ece9df326f7cab7f6b9416dbb7c98505d2559152bf4e7d8.jpg)

![725bc7e7be3b750f720e051824dc3aadd674cd8d924298b0bab2bde0c31f9943.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/tables/725bc7e7be3b750f720e051824dc3aadd674cd8d924298b0bab2bde0c31f9943.jpg)

![735c2b6442471b9d2b16aa834559151800dd230eab9bc88b8d6e44e0b8eae48c.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/tables/735c2b6442471b9d2b16aa834559151800dd230eab9bc88b8d6e44e0b8eae48c.jpg)

![925b43408fae37d32b88afb454a98e951c38e510de17a6424c70c38027fb233c.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/tables/925b43408fae37d32b88afb454a98e951c38e510de17a6424c70c38027fb233c.jpg)

![b14af6158122afb628a0c2780e3127438db5437e36f8d0cb984c71d1b8be727b.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/tables/b14af6158122afb628a0c2780e3127438db5437e36f8d0cb984c71d1b8be727b.jpg)

![d83a3884df4d94ee80c9b55018c420afe592f29a420aacc1bf6a6c884e8fd40e.jpg](../icml_results/281_TLLC_%20Transfer%20Learning-based%20Label%20Completion%20for%20Crowdsourcing/tables/d83a3884df4d94ee80c9b55018c420afe592f29a420aacc1bf6a6c884e8fd40e.jpg)

## Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspective


### Images

![030579ec71427e2194c7256e9d0f58e0d91dbf6c0baf20607cdea7fdbdc374d5.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/images/030579ec71427e2194c7256e9d0f58e0d91dbf6c0baf20607cdea7fdbdc374d5.jpg)

![28490441f23ec39f61963d465acee43f3775fb3bc2067906fd9f81c9a1b1d31c.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/images/28490441f23ec39f61963d465acee43f3775fb3bc2067906fd9f81c9a1b1d31c.jpg)

![63acd73d2f1cb9e9241fcf746abe0b7a78146c4807abee81b99c13e3c6a12437.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/images/63acd73d2f1cb9e9241fcf746abe0b7a78146c4807abee81b99c13e3c6a12437.jpg)

### Tables

![0f53bba079ec2f8facb20baf499079b060a7a2f48865405c65c0851fd1d4039b.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/0f53bba079ec2f8facb20baf499079b060a7a2f48865405c65c0851fd1d4039b.jpg)

![196715b6ed977990fbc74607f395d2efc5ab44d9c4c5d84f35223a076c459230.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/196715b6ed977990fbc74607f395d2efc5ab44d9c4c5d84f35223a076c459230.jpg)

![2b6e5b8ca281fa80bfd79690700ed4a921347516196e10c101ab7d2134428efc.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/2b6e5b8ca281fa80bfd79690700ed4a921347516196e10c101ab7d2134428efc.jpg)

![3bf1e027cce1c289558cc4581c8c6479aca114ac3a3171f1608e2fa356c9c879.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/3bf1e027cce1c289558cc4581c8c6479aca114ac3a3171f1608e2fa356c9c879.jpg)

![5160bb01bd13b5cc11741f2be12954e6513c4d624aab646529e7c4cf1efecb6f.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/5160bb01bd13b5cc11741f2be12954e6513c4d624aab646529e7c4cf1efecb6f.jpg)

![6442d78223821348dd36f746804326ccaa517384cb2b1fa759ec886aceaa2f0b.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/6442d78223821348dd36f746804326ccaa517384cb2b1fa759ec886aceaa2f0b.jpg)

![76bebc60028f5c587690b966d090800499a011610a11abf4d1d274d419a70109.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/76bebc60028f5c587690b966d090800499a011610a11abf4d1d274d419a70109.jpg)

![9e26ac8da7e1c8449184f144be98a330a35dd9dab16195cc405a9ba85c2ffdc6.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/9e26ac8da7e1c8449184f144be98a330a35dd9dab16195cc405a9ba85c2ffdc6.jpg)

![a78bd89637b4248d5b9eb79dec4246b958e28b7fcd442a678742c97e2fdd29f8.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/a78bd89637b4248d5b9eb79dec4246b958e28b7fcd442a678742c97e2fdd29f8.jpg)

![a8283f581cd2873d19455991832c12c1cbe0346d7d206ca9140034ba6e4c6078.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/a8283f581cd2873d19455991832c12c1cbe0346d7d206ca9140034ba6e4c6078.jpg)

![aa7c65c24c99c2b3c151c8cde71151192eb388fd5e7384a26167f43191b8047f.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/aa7c65c24c99c2b3c151c8cde71151192eb388fd5e7384a26167f43191b8047f.jpg)

![e48d0d6dc28ad701fd5f46224a935f70f191681ae301b524a451c342cc502ef5.jpg](../icml_results/282_Distribution-aware%20Fairness%20Learning%20in%20Medical%20Image%20Segmentation%20From%20A%20Control-Theoretic%20Perspect/tables/e48d0d6dc28ad701fd5f46224a935f70f191681ae301b524a451c342cc502ef5.jpg)

## Learning Soft Sparse Shapes for Efficient Time-Series Classification


### Images

![366244a3a4c5cf0f9639c0e1a0d4ead176a6789871110f3227b0700e68157d98.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/366244a3a4c5cf0f9639c0e1a0d4ead176a6789871110f3227b0700e68157d98.jpg)

![367c032d2107c460e9d8cfb2a4fe7d8d63bb0927b6b9217488a9e9738024ef5e.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/367c032d2107c460e9d8cfb2a4fe7d8d63bb0927b6b9217488a9e9738024ef5e.jpg)

![37f7576551d37acad9956cc8483a3059f906596869173027845489ebde3e2b88.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/37f7576551d37acad9956cc8483a3059f906596869173027845489ebde3e2b88.jpg)

![5b54ce89b3a440643cde46e1bed1817586a36716fc3c5721b2e4bb0b0c14c50a.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/5b54ce89b3a440643cde46e1bed1817586a36716fc3c5721b2e4bb0b0c14c50a.jpg)

![5d03ad69203591d9fa7df82ccadb7c8672e3105480f6ce0c37993f588650e2b7.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/5d03ad69203591d9fa7df82ccadb7c8672e3105480f6ce0c37993f588650e2b7.jpg)

![6a1fa0682d20ff5ac3c2a39cdd7f4c809c1a048614033f1ed716d61cb2aa0fa6.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/6a1fa0682d20ff5ac3c2a39cdd7f4c809c1a048614033f1ed716d61cb2aa0fa6.jpg)

![6a352b3fb36db0f558f135846bcf04c8e14187ee9c0f1c5b38117b3aa6db97e1.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/6a352b3fb36db0f558f135846bcf04c8e14187ee9c0f1c5b38117b3aa6db97e1.jpg)

![7614844413b689d5e2da82329b54bca99ca49764f4852affdbdc68f9c887baef.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/7614844413b689d5e2da82329b54bca99ca49764f4852affdbdc68f9c887baef.jpg)

![832b83763ab1853afb21ef65843563eb81ddd29aa095f16151dafcf444547148.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/832b83763ab1853afb21ef65843563eb81ddd29aa095f16151dafcf444547148.jpg)

![e00965a0591bd1a5b59dd8f734f3e8c86d0dece2870e71eb49fb17aea4263d05.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/images/e00965a0591bd1a5b59dd8f734f3e8c86d0dece2870e71eb49fb17aea4263d05.jpg)

### Tables

![014becb45a489ced70804f3ba73f9c081c7304ea386a653ba62ae89d5832bfea.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/014becb45a489ced70804f3ba73f9c081c7304ea386a653ba62ae89d5832bfea.jpg)

![08117d95e6210d22ccf906feb90a0da79d412a745052fd26c25e7f584be5a13b.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/08117d95e6210d22ccf906feb90a0da79d412a745052fd26c25e7f584be5a13b.jpg)

![107963ab5d3eac9e8a37958350bd0dfe002352ac3a6949bb7af9fd0a37d768cd.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/107963ab5d3eac9e8a37958350bd0dfe002352ac3a6949bb7af9fd0a37d768cd.jpg)

![2f0a96be80852bce829f1dbabb7eca96f58a69a4056e6a1a789062fd547c01de.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/2f0a96be80852bce829f1dbabb7eca96f58a69a4056e6a1a789062fd547c01de.jpg)

![3f6e000596556d0e9356c6379752b887c7c13beb261cec7826deb9a369e52c59.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/3f6e000596556d0e9356c6379752b887c7c13beb261cec7826deb9a369e52c59.jpg)

![484b90ee253b3892752889b684b09bb5587694d5e6fd548ea69154212e367e8e.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/484b90ee253b3892752889b684b09bb5587694d5e6fd548ea69154212e367e8e.jpg)

![65b0286fdfafe67b1ecb73006963f757e996fedaf906dc2b9eaa56cde5930d13.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/65b0286fdfafe67b1ecb73006963f757e996fedaf906dc2b9eaa56cde5930d13.jpg)

![67b9ff49025d6d2061c0c12c0569269810f86361336f88171237d71cfa9e0259.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/67b9ff49025d6d2061c0c12c0569269810f86361336f88171237d71cfa9e0259.jpg)

![70bbc795fdacaad7f6d01f46266dc6f3096a53cd6a214c25f33402b873ba74dc.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/70bbc795fdacaad7f6d01f46266dc6f3096a53cd6a214c25f33402b873ba74dc.jpg)

![7a1ce955ee5ef892e228698ccfbd1870d8cd8b5120f388b847c10a5029a53f79.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/7a1ce955ee5ef892e228698ccfbd1870d8cd8b5120f388b847c10a5029a53f79.jpg)

![88290c1119b988c56387f4e707b8e41b4715cded02b5e05d0c2c69cbf5241e32.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/88290c1119b988c56387f4e707b8e41b4715cded02b5e05d0c2c69cbf5241e32.jpg)

![8da6e9080ccbdd41cfb41176889ae9682d1bbe83110a8e261bcf0c11ca3230d9.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/8da6e9080ccbdd41cfb41176889ae9682d1bbe83110a8e261bcf0c11ca3230d9.jpg)

![8fb1ce7e0aeda524aaf51165d576a3179dc12ab2fadc11565d277831d5e639f4.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/8fb1ce7e0aeda524aaf51165d576a3179dc12ab2fadc11565d277831d5e639f4.jpg)

![9152cf6a68afb2bef4a9a8d3f546801908b1f8216c915b6c8c40d6933620b130.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/9152cf6a68afb2bef4a9a8d3f546801908b1f8216c915b6c8c40d6933620b130.jpg)

![94c8d9642ad788f660b4af6c02870a274824e18357f4dd14d6820a949cd481ee.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/94c8d9642ad788f660b4af6c02870a274824e18357f4dd14d6820a949cd481ee.jpg)

![9a614cb3a9f98ed69d2a5c812aa878fc23cbacbfafd0b5a1d06ddf269cacd2ae.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/9a614cb3a9f98ed69d2a5c812aa878fc23cbacbfafd0b5a1d06ddf269cacd2ae.jpg)

![b5f37aa742c852f4890d2e0f541bef2ee6bd7d92a9570765865c5d561c1a3391.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/b5f37aa742c852f4890d2e0f541bef2ee6bd7d92a9570765865c5d561c1a3391.jpg)

![beecf6ea209819b3f6dc38ae0176cce7a5c935353fd561b8dcebd24c63ffaeb2.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/beecf6ea209819b3f6dc38ae0176cce7a5c935353fd561b8dcebd24c63ffaeb2.jpg)

![dad336e31416ce0c6c9f1dc68c7d329d247a7e3c4bc240064396b2acdb48651b.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/dad336e31416ce0c6c9f1dc68c7d329d247a7e3c4bc240064396b2acdb48651b.jpg)

![e1b64c4d6cdaeff830d53752e9d82420b552e95594c26d05e0e2cd82c6d94285.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/e1b64c4d6cdaeff830d53752e9d82420b552e95594c26d05e0e2cd82c6d94285.jpg)

![e23d52f640883233539fef07746ff1b3d8b93113d477a02a9f17eb6b118cea14.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/e23d52f640883233539fef07746ff1b3d8b93113d477a02a9f17eb6b118cea14.jpg)

![e987716dd054affff8c8cc5eb710ab71a8e5afe87afc5b6533390a8de4be14ef.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/e987716dd054affff8c8cc5eb710ab71a8e5afe87afc5b6533390a8de4be14ef.jpg)

![ec1990aa862d35994f3a35ad85e7396369be7c7e1372737d95c1d64e0b06712f.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/ec1990aa862d35994f3a35ad85e7396369be7c7e1372737d95c1d64e0b06712f.jpg)

![ed6642d208c2153f2d97a80eb53ec171158e004960253bdcd3c3b43cab66eb5f.jpg](../icml_results/283_Learning%20Soft%20Sparse%20Shapes%20for%20Efficient%20Time-Series%20Classification/tables/ed6642d208c2153f2d97a80eb53ec171158e004960253bdcd3c3b43cab66eb5f.jpg)

## Optimizing Adaptive Attacks against Watermarks for Language Models


### Images

![03a15ceb15756f426c2e3de85bfc90e49c439ac52e5fa3cee1ed488f036e4d8d.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/03a15ceb15756f426c2e3de85bfc90e49c439ac52e5fa3cee1ed488f036e4d8d.jpg)

![05ebd6cb072ae6a33960a2828e3eb87dfcb4972dae6c038e291fce5816435874.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/05ebd6cb072ae6a33960a2828e3eb87dfcb4972dae6c038e291fce5816435874.jpg)

![18a25378dede89d47cd4c7f536aaebca758b6697c4ecb68c51870cfe8da1ece8.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/18a25378dede89d47cd4c7f536aaebca758b6697c4ecb68c51870cfe8da1ece8.jpg)

![242fb0a99b41290428bdcaa734058e895f7479fd7546c2e17738177eba62ad5d.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/242fb0a99b41290428bdcaa734058e895f7479fd7546c2e17738177eba62ad5d.jpg)

![2a19ba41f2b21fb33998d6e19c07b1171aa90de5361695da1d8b096863152880.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/2a19ba41f2b21fb33998d6e19c07b1171aa90de5361695da1d8b096863152880.jpg)

![428ded4e01ef765fe525e68260c82d96dffa4bf40ed7d9e4d3ffbe94eb331d5f.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/428ded4e01ef765fe525e68260c82d96dffa4bf40ed7d9e4d3ffbe94eb331d5f.jpg)

![4542517d4eedcd9466b628e3e70db7235da8a9eac4d0569060f7e19c41edb8da.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/4542517d4eedcd9466b628e3e70db7235da8a9eac4d0569060f7e19c41edb8da.jpg)

![4fd1c03101d410451cfa6353dcd18163325e444da3cae40fbccdd8b60acc62d8.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/4fd1c03101d410451cfa6353dcd18163325e444da3cae40fbccdd8b60acc62d8.jpg)

![6620c24566b5c598ac1c0a58cc25711345aecfa269371b604133d89b21f012d5.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/6620c24566b5c598ac1c0a58cc25711345aecfa269371b604133d89b21f012d5.jpg)

![aafcfc2ef6a0c44be0beb3de404e1e3e12129c1c4fe700ad21b53adefeb451b3.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/aafcfc2ef6a0c44be0beb3de404e1e3e12129c1c4fe700ad21b53adefeb451b3.jpg)

![ac8f7b5102095de8bbd6e2ba7fef04a6c3a361a10eb7aacb8103bcf6ca94d6cd.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/ac8f7b5102095de8bbd6e2ba7fef04a6c3a361a10eb7aacb8103bcf6ca94d6cd.jpg)

![b70aac7ddb201628f565a8ae8cb1a372988d25ba04942701c95a56b700035746.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/b70aac7ddb201628f565a8ae8cb1a372988d25ba04942701c95a56b700035746.jpg)

![ca4781d64f1602fbc5be16fbaed28660a1e8c57b89004603377bf1f4f54ca035.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/ca4781d64f1602fbc5be16fbaed28660a1e8c57b89004603377bf1f4f54ca035.jpg)

![ce4ff1609abcc9c955433da7b11bcac3f76bb37e4af9f9b5ed88121d0d12092e.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/ce4ff1609abcc9c955433da7b11bcac3f76bb37e4af9f9b5ed88121d0d12092e.jpg)

![dbc7d87c0bc0a25319d9ab096048f05abe9607243378d7e3052487a1285504c2.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/dbc7d87c0bc0a25319d9ab096048f05abe9607243378d7e3052487a1285504c2.jpg)

![e4c99ea34e1e43bddb266aefbe856494ef405348cb04c8f79c8052fecbd8e1ea.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/e4c99ea34e1e43bddb266aefbe856494ef405348cb04c8f79c8052fecbd8e1ea.jpg)

![fee2410872c10c3d4b5407880bbf663ad84c77d7c3997081fd8260e1bad487ef.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/images/fee2410872c10c3d4b5407880bbf663ad84c77d7c3997081fd8260e1bad487ef.jpg)

### Tables

![3073cb9918589f9cdc1484b581b2c2f52b7e9c6495ba12540d6c6502a7b3fd5a.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/tables/3073cb9918589f9cdc1484b581b2c2f52b7e9c6495ba12540d6c6502a7b3fd5a.jpg)

![53ee262725577d7de70a2508fcc4f716391c7d1dbe73a75a9104563271aecfe0.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/tables/53ee262725577d7de70a2508fcc4f716391c7d1dbe73a75a9104563271aecfe0.jpg)

![90731e27931c756a70f9c3c9af4de8f5c61448ec2ce83271987d3174057dac90.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/tables/90731e27931c756a70f9c3c9af4de8f5c61448ec2ce83271987d3174057dac90.jpg)

![b9142a38aee1766e39655e94fc8c0bbd55fa758e69ef9eeafe276da21dade0ef.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/tables/b9142a38aee1766e39655e94fc8c0bbd55fa758e69ef9eeafe276da21dade0ef.jpg)

![e5457ac4f274e5db1fc4193d68adef312c41aebae68024eb502a8fb90fdf0c70.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/tables/e5457ac4f274e5db1fc4193d68adef312c41aebae68024eb502a8fb90fdf0c70.jpg)

![f87fd0c2756212c7d8abcc4c2cbbf26ab5a343ccf31856254c7c89339929be9e.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/tables/f87fd0c2756212c7d8abcc4c2cbbf26ab5a343ccf31856254c7c89339929be9e.jpg)

![fa0ff89c7e2c5f455dba2bc8fef8bf9f528b0b88aaa27b840222f3f8c3cbaf0f.jpg](../icml_results/284_Optimizing%20Adaptive%20Attacks%20against%20Watermarks%20for%20Language%20Models/tables/fa0ff89c7e2c5f455dba2bc8fef8bf9f528b0b88aaa27b840222f3f8c3cbaf0f.jpg)

## Exogenous Isomorphism for Counterfactual Identifiability


### Images

![3e9f9fe5e49436c489f45812ce03720e098619d62f4bf97bedbeb159845f3641.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/images/3e9f9fe5e49436c489f45812ce03720e098619d62f4bf97bedbeb159845f3641.jpg)

![708efb0bb20bb0419fde2e769f7c6c1ac98471c80d105e810d9b46716f926566.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/images/708efb0bb20bb0419fde2e769f7c6c1ac98471c80d105e810d9b46716f926566.jpg)

![88790735b01b4d9d71f69e9cd6c97e56ad38dce3ef1fb910dfb44c45d1ae24ba.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/images/88790735b01b4d9d71f69e9cd6c97e56ad38dce3ef1fb910dfb44c45d1ae24ba.jpg)

### Tables

![12a10d27f5f869a52bd4dd4fd60875f14e31148c5f732862ff604302d7e3c17e.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/tables/12a10d27f5f869a52bd4dd4fd60875f14e31148c5f732862ff604302d7e3c17e.jpg)

![4b698b99b396a3edcdda55f9e15389805a51bdf920a891eaab7c1dd3bb1a515c.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/tables/4b698b99b396a3edcdda55f9e15389805a51bdf920a891eaab7c1dd3bb1a515c.jpg)

![518c6cb093531f8d15dad574224f4096c5781626f76630c1b9d6a0c18b00d56c.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/tables/518c6cb093531f8d15dad574224f4096c5781626f76630c1b9d6a0c18b00d56c.jpg)

![5d9f918e06961e5da2054d276d0f7cc66d4074eb1f6a46173504edde0fa83ef8.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/tables/5d9f918e06961e5da2054d276d0f7cc66d4074eb1f6a46173504edde0fa83ef8.jpg)

![7ab7b914a48d6c8cf36c5c7e204a48907444ba28bb751f9e7cfe0ed5b17ecdfa.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/tables/7ab7b914a48d6c8cf36c5c7e204a48907444ba28bb751f9e7cfe0ed5b17ecdfa.jpg)

![b8da8e780eed6f8b30830dbb5ee5ae716946708aa31804cc60c638bed2dd73b5.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/tables/b8da8e780eed6f8b30830dbb5ee5ae716946708aa31804cc60c638bed2dd73b5.jpg)

![fff3ccd341547e19df873b056c81f5ae4a5818200b4bcd4d08dbf40e3800d7e4.jpg](../icml_results/285_Exogenous%20Isomorphism%20for%20Counterfactual%20Identifiability/tables/fff3ccd341547e19df873b056c81f5ae4a5818200b4bcd4d08dbf40e3800d7e4.jpg)

## Robust ML Auditing using Prior Knowledge


### Images

![52464bcfeab9473555b19a6cc8caa7dab0d3a2ed223bc390d4bda0104e9670f0.jpg](../icml_results/286_Robust%20ML%20Auditing%20using%20Prior%20Knowledge/images/52464bcfeab9473555b19a6cc8caa7dab0d3a2ed223bc390d4bda0104e9670f0.jpg)

![97bd586707d2ad1cfac9b50ab977ed1142c22b31e381d6fbb602af1e007db5d4.jpg](../icml_results/286_Robust%20ML%20Auditing%20using%20Prior%20Knowledge/images/97bd586707d2ad1cfac9b50ab977ed1142c22b31e381d6fbb602af1e007db5d4.jpg)

![a5aaf33ce0d9c3a7f35a3d6a60421314de32283461830f59b0ab1ec6623c98db.jpg](../icml_results/286_Robust%20ML%20Auditing%20using%20Prior%20Knowledge/images/a5aaf33ce0d9c3a7f35a3d6a60421314de32283461830f59b0ab1ec6623c98db.jpg)

![f94c076c727e4765c813d62f9c822ae91b57d2c86b8d42cc49606072df67ddd6.jpg](../icml_results/286_Robust%20ML%20Auditing%20using%20Prior%20Knowledge/images/f94c076c727e4765c813d62f9c822ae91b57d2c86b8d42cc49606072df67ddd6.jpg)

### Tables

![fa281e50a3060f56b0c60e6b4518a11a83a99bc6d77409a5f66e5c8a6b6df64c.jpg](../icml_results/286_Robust%20ML%20Auditing%20using%20Prior%20Knowledge/tables/fa281e50a3060f56b0c60e6b4518a11a83a99bc6d77409a5f66e5c8a6b6df64c.jpg)

## Adjusting Model Size in Continual Gaussian Processes: How Big is Big Enough?


### Images

![342d44cffda0fd065c0a79914df2c41dfd88b48ddaba790e953961e7bd045410.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/342d44cffda0fd065c0a79914df2c41dfd88b48ddaba790e953961e7bd045410.jpg)

![41b45d6f84c3b37d7c139a31a5a4ea4bb3f32a88caaebae1b3dac739ce0df4e3.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/41b45d6f84c3b37d7c139a31a5a4ea4bb3f32a88caaebae1b3dac739ce0df4e3.jpg)

![42cacddb55e9918c97e5916c959aceca43b02422fc616a83f294e0c44b3992ec.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/42cacddb55e9918c97e5916c959aceca43b02422fc616a83f294e0c44b3992ec.jpg)

![533f1404da55793767a87a78e2ac5aa698f7d59bbf23bc2bae1d8497286ad792.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/533f1404da55793767a87a78e2ac5aa698f7d59bbf23bc2bae1d8497286ad792.jpg)

![752405d592de68f682e2553684b1764884855c20b22d7910aa4f8e0fcfb050b3.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/752405d592de68f682e2553684b1764884855c20b22d7910aa4f8e0fcfb050b3.jpg)

![9dff3429c4406f42df13e121d9db6f534ead26cacd9bd46cc6a03760f415aad4.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/9dff3429c4406f42df13e121d9db6f534ead26cacd9bd46cc6a03760f415aad4.jpg)

![a43bd3070e4089d22e7991f77a6749614085189ea46c84c97db285c9f1fcf710.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/a43bd3070e4089d22e7991f77a6749614085189ea46c84c97db285c9f1fcf710.jpg)

![abbdfa930196ba9a1d740e3ef5ffc0da1cbe3f4b3ff011c6d1360451123a37ec.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/abbdfa930196ba9a1d740e3ef5ffc0da1cbe3f4b3ff011c6d1360451123a37ec.jpg)

![bc495b3feff64b306f4c9ff5cb9c96a6965dbf3780fdc5d04f2ee273f420f762.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/bc495b3feff64b306f4c9ff5cb9c96a6965dbf3780fdc5d04f2ee273f420f762.jpg)

![c9d3c1025c5b29516d44a1e8c7a53e98271516411f534cbc71fd0bc13a080f47.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/c9d3c1025c5b29516d44a1e8c7a53e98271516411f534cbc71fd0bc13a080f47.jpg)

![d0e1417922d953b6e72ba86ef04a0d025a4201cd6f4870d5ef9dc2a2d5d29f4a.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/d0e1417922d953b6e72ba86ef04a0d025a4201cd6f4870d5ef9dc2a2d5d29f4a.jpg)

![da22f07c800f54684fe29f6c344c95391511ea91ac497163a57a154215e0c3aa.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/da22f07c800f54684fe29f6c344c95391511ea91ac497163a57a154215e0c3aa.jpg)

![f4fe2ef7d12225c858c0828e18bdf1b0198f5a920cccd5b81c59e701d75d1865.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/f4fe2ef7d12225c858c0828e18bdf1b0198f5a920cccd5b81c59e701d75d1865.jpg)

![fc13671e098f12240a4c6c804bd01bc665d06b4c5173131d72253c29d7220ed9.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/images/fc13671e098f12240a4c6c804bd01bc665d06b4c5173131d72253c29d7220ed9.jpg)

### Tables

![024372cdfc0b6a977033af91e392d14c796ab5956797536eacc9d8040f0fd718.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/024372cdfc0b6a977033af91e392d14c796ab5956797536eacc9d8040f0fd718.jpg)

![2282f8297a2a1b6bb236f6858ceee2269b5dd0bc8271cb4f31285d2a2efee3e1.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/2282f8297a2a1b6bb236f6858ceee2269b5dd0bc8271cb4f31285d2a2efee3e1.jpg)

![622a13b6b18b6d7ab936623005554b28e02d80b47195e2c83abddb3bcfd6ed3f.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/622a13b6b18b6d7ab936623005554b28e02d80b47195e2c83abddb3bcfd6ed3f.jpg)

![a5d7ae5f02c12c62a1b5ecfd2f2796b1a1cccff087f3e2e93e830dd33996b8a5.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/a5d7ae5f02c12c62a1b5ecfd2f2796b1a1cccff087f3e2e93e830dd33996b8a5.jpg)

![a97b396ae7feacd3594453abdcc8907dca45e2cce8c2dba45b4b4864f59ab065.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/a97b396ae7feacd3594453abdcc8907dca45e2cce8c2dba45b4b4864f59ab065.jpg)

![b6ec9a7bc8afe5c740db7404a0debf3c99472e2c0a5508bbbb72c317d536f461.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/b6ec9a7bc8afe5c740db7404a0debf3c99472e2c0a5508bbbb72c317d536f461.jpg)

![ec45fb202387cd4709c5d6e48a0b686d22ec884d1d962ce86bd2f0863497c319.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/ec45fb202387cd4709c5d6e48a0b686d22ec884d1d962ce86bd2f0863497c319.jpg)

![ef6288e68ffd48dc37c32699872c9f2143ef242e7721504235a68afb70b7722a.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/ef6288e68ffd48dc37c32699872c9f2143ef242e7721504235a68afb70b7722a.jpg)

![f245d1a997150def79156b917ad3aea07fae77ebadb9f5f9dbc0b15261913633.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/f245d1a997150def79156b917ad3aea07fae77ebadb9f5f9dbc0b15261913633.jpg)

![f72f91de72dcee9e056a42a70663e81c71483b007385aa3ed7fb09ca9f85c5a5.jpg](../icml_results/287_Adjusting%20Model%20Size%20in%20Continual%20Gaussian%20Processes_%20How%20Big%20is%20Big%20Enough_/tables/f72f91de72dcee9e056a42a70663e81c71483b007385aa3ed7fb09ca9f85c5a5.jpg)

## LotteryCodec: Searching the Implicit Representation in a Random Network for Low-Complexity Image Compression


### Images

![11f8b8f9a0462041a71b53c557ec0d1c9e0ef0cd95649fac13774120d61a6743.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/11f8b8f9a0462041a71b53c557ec0d1c9e0ef0cd95649fac13774120d61a6743.jpg)

![16f887a940e80273b331efaed13d43a63293c8cf5c23692ef73bdba72fa6a40d.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/16f887a940e80273b331efaed13d43a63293c8cf5c23692ef73bdba72fa6a40d.jpg)

![2bc6f555821995cd02060d1864c70d299457035f317e7e5ed7ec580ab533365c.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/2bc6f555821995cd02060d1864c70d299457035f317e7e5ed7ec580ab533365c.jpg)

![3000b18ffd85d082b63fae8a271c54057522a34f329d23ed014ebdf65344745a.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/3000b18ffd85d082b63fae8a271c54057522a34f329d23ed014ebdf65344745a.jpg)

![370a7a3030ff8303edf6ddde27c2418fea41923352d5add2fc92a860d0eb89f3.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/370a7a3030ff8303edf6ddde27c2418fea41923352d5add2fc92a860d0eb89f3.jpg)

![41e4acb416ee78e21cd1959131530f606cc9245daf98296f60a721f6519eb8e7.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/41e4acb416ee78e21cd1959131530f606cc9245daf98296f60a721f6519eb8e7.jpg)

![439cdd96f164bafb65caa97d0cbcc8933ef16b444a220a57afc75931d1a097ea.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/439cdd96f164bafb65caa97d0cbcc8933ef16b444a220a57afc75931d1a097ea.jpg)

![4e3fe1be0f238d9313528f4db884e278accc5b03adf352d622c0822692d30489.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/4e3fe1be0f238d9313528f4db884e278accc5b03adf352d622c0822692d30489.jpg)

![6a92f35a9e3e0733173fd93572deb48fb7bc55ae66c63e7d1c588d08db7453dc.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/6a92f35a9e3e0733173fd93572deb48fb7bc55ae66c63e7d1c588d08db7453dc.jpg)

![6c596d09148e6968d5c4719ddb1934f95d9bf4160137a5a8b08161fbf3b4c143.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/6c596d09148e6968d5c4719ddb1934f95d9bf4160137a5a8b08161fbf3b4c143.jpg)

![7e4458fe6ce520bcd376de219cacab483b663b6e6cc36f2cf2349efdb2ad4d21.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/7e4458fe6ce520bcd376de219cacab483b663b6e6cc36f2cf2349efdb2ad4d21.jpg)

![81637c799e9e9867fdb2308efe8a3197611c9ae89b1360111fe5b1c97aa03ce9.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/81637c799e9e9867fdb2308efe8a3197611c9ae89b1360111fe5b1c97aa03ce9.jpg)

![9b6648e03c9533abcfa26def7427021b04306569db9689aac2c29f10cea8ed15.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/9b6648e03c9533abcfa26def7427021b04306569db9689aac2c29f10cea8ed15.jpg)

![cbe0cc42c59a87f4ddf229256de83159efb8e9aec4a76b6b99a33492fee0e4ad.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/cbe0cc42c59a87f4ddf229256de83159efb8e9aec4a76b6b99a33492fee0e4ad.jpg)

![dc21d1075523ac3c87b52c28d88a3031f1fc9c40a572a418ccd6e223ce4a79a9.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/dc21d1075523ac3c87b52c28d88a3031f1fc9c40a572a418ccd6e223ce4a79a9.jpg)

![dec2bf6d958bdefd47eafdfadaca6d9ffea4a1486121562b737dba20b666ea49.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/dec2bf6d958bdefd47eafdfadaca6d9ffea4a1486121562b737dba20b666ea49.jpg)

![ef5a1beddd5978bb526fca7e085a2c4eb085c77fa88e1278b08108ad0101c625.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/ef5a1beddd5978bb526fca7e085a2c4eb085c77fa88e1278b08108ad0101c625.jpg)

![f1786dd0d4d233070d3706eb5314eb74b77d9fc52ecdc55c2c00407407adb665.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/images/f1786dd0d4d233070d3706eb5314eb74b77d9fc52ecdc55c2c00407407adb665.jpg)

### Tables

![04557042aae53406ae04383a70a3663b4d4837dcf8dc951d4e62c2e6cc4b5a71.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/04557042aae53406ae04383a70a3663b4d4837dcf8dc951d4e62c2e6cc4b5a71.jpg)

![0f88318fa94b8a445fbd592d474726a248a3946a27efb80c647929b2576af471.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/0f88318fa94b8a445fbd592d474726a248a3946a27efb80c647929b2576af471.jpg)

![204d298f1461ad46fd95b97db78dda8d2bc4c164297987b4741ea4e31b7d4bf5.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/204d298f1461ad46fd95b97db78dda8d2bc4c164297987b4741ea4e31b7d4bf5.jpg)

![32b14288792aedfa806a887966ba06e25b7df7b2097b25bbd8fb97753038bffe.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/32b14288792aedfa806a887966ba06e25b7df7b2097b25bbd8fb97753038bffe.jpg)

![61109cf9bb80c358f5e05d338d1aa317659c047d92838aeacc7eb2f9751ee471.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/61109cf9bb80c358f5e05d338d1aa317659c047d92838aeacc7eb2f9751ee471.jpg)

![6dcaa4197704ec5218d4d947557b75a79c070d91f6307addcde37a2c3786b278.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/6dcaa4197704ec5218d4d947557b75a79c070d91f6307addcde37a2c3786b278.jpg)

![7a862a84fee8110a5177440cf71abd9f6cd34634019f47a52f5cd30948d86f78.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/7a862a84fee8110a5177440cf71abd9f6cd34634019f47a52f5cd30948d86f78.jpg)

![a892cd9f1684cabf02f469fbedeb26699f6e84fbc7f67219581432d5be9d6561.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/a892cd9f1684cabf02f469fbedeb26699f6e84fbc7f67219581432d5be9d6561.jpg)

![f5b3e6a3b37032e1efa2e24702828b600d3c9912ea1bc58cccd04af376073673.jpg](../icml_results/288_LotteryCodec_%20Searching%20the%20Implicit%20Representation%20in%20a%20Random%20Network%20for%20Low-Complexity%20Image%20Com/tables/f5b3e6a3b37032e1efa2e24702828b600d3c9912ea1bc58cccd04af376073673.jpg)

## MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding


### Images

![00245bdaac7acd46112afb500b578301dc0f672e460da51f3916358468a1add0.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/00245bdaac7acd46112afb500b578301dc0f672e460da51f3916358468a1add0.jpg)

![12aad94229bd55748eafa3754c8f1d937c9fcfc7f7cb677428cc1f27d9aeb6ba.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/12aad94229bd55748eafa3754c8f1d937c9fcfc7f7cb677428cc1f27d9aeb6ba.jpg)

![87c1fa3ed5bec26603fff13bd4bc26c640b34517c675b1a85a6cb29faa79456c.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/87c1fa3ed5bec26603fff13bd4bc26c640b34517c675b1a85a6cb29faa79456c.jpg)

![89b867ed22b6478bcca57bc47753d6ad36300fee5e751544ee7eb6e47473fca3.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/89b867ed22b6478bcca57bc47753d6ad36300fee5e751544ee7eb6e47473fca3.jpg)

![91d61158c4c25da872da3e6640f0a6b68d925eff23d7be70da57811276a46fc1.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/91d61158c4c25da872da3e6640f0a6b68d925eff23d7be70da57811276a46fc1.jpg)

![ad85d0fa725c80c2b0ee8bf011f8896a30c1a542764bea958540f777b353b490.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/ad85d0fa725c80c2b0ee8bf011f8896a30c1a542764bea958540f777b353b490.jpg)

![c71ec2c5a5e4a8f391c21caead93c52540883d1072727905a3b2dcd13301e3ba.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/c71ec2c5a5e4a8f391c21caead93c52540883d1072727905a3b2dcd13301e3ba.jpg)

![f30a96843d01b223a3d804b5b470a81e214c93be6feb12536ed63a8c2397456c.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/f30a96843d01b223a3d804b5b470a81e214c93be6feb12536ed63a8c2397456c.jpg)

![f782998f798f1573cb2b15d6c74c4d4d4d18685fc4dca8555b8d32363db74c55.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/images/f782998f798f1573cb2b15d6c74c4d4d4d18685fc4dca8555b8d32363db74c55.jpg)

### Tables

![50f45a7814b3cd8aa1df71090e9d69394e8f289fe85b9408d068f09c5321ef93.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/tables/50f45a7814b3cd8aa1df71090e9d69394e8f289fe85b9408d068f09c5321ef93.jpg)

![61d87f80ebbf32077660c6e1faffffb54f7a2b078325d80a3ef1de3d42b9c7d6.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/tables/61d87f80ebbf32077660c6e1faffffb54f7a2b078325d80a3ef1de3d42b9c7d6.jpg)

![a06f619d5bbeaabd9cbb9776699c823780e89c9635bf05a40fc842a381f3f6bb.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/tables/a06f619d5bbeaabd9cbb9776699c823780e89c9635bf05a40fc842a381f3f6bb.jpg)

![f2ca92c76615d6bc20aa3bfd8086182e94b5da640ed19b716f35b846ee5b14a5.jpg](../icml_results/289_MODA_%20MOdular%20Duplex%20Attention%20for%20Multimodal%20Perception%2C%20Cognition%2C%20and%20Emotion%20Understanding/tables/f2ca92c76615d6bc20aa3bfd8086182e94b5da640ed19b716f35b846ee5b14a5.jpg)

## FedSSI: Rehearsal-Free Continual Federated Learning with Synergistic  Synaptic Intelligence


### Images

![2e403c1fa7cdbb103775fdd84ace04f7e4f9c7db6df9ae63624e2c4f5467c17c.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/images/2e403c1fa7cdbb103775fdd84ace04f7e4f9c7db6df9ae63624e2c4f5467c17c.jpg)

![2e89b7e92de4c2d19f01ca8a8bc90891106516183ad922edfc535943f36fe73b.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/images/2e89b7e92de4c2d19f01ca8a8bc90891106516183ad922edfc535943f36fe73b.jpg)

![a54b292f1b0db1ea58f030872d530194b754654710bdc171ee7ba070a2a97621.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/images/a54b292f1b0db1ea58f030872d530194b754654710bdc171ee7ba070a2a97621.jpg)

![c96d5465ca1f9340ae6a16bd4074006000abf3aec84aafc4ea2c79329bd1e625.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/images/c96d5465ca1f9340ae6a16bd4074006000abf3aec84aafc4ea2c79329bd1e625.jpg)

### Tables

![03973203e1bb019976242842b2b0b22f65ca305bd50354bdfac7e117fd7ee7cf.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/03973203e1bb019976242842b2b0b22f65ca305bd50354bdfac7e117fd7ee7cf.jpg)

![0417648073f72d76aca628eb4ff0d0fd46aadfa14efd119b59dde8bfefeab41b.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/0417648073f72d76aca628eb4ff0d0fd46aadfa14efd119b59dde8bfefeab41b.jpg)

![38b2f7d3797f412fbf1490ff2cdd86efbc2b6d58c8488849629c75883e8ddd62.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/38b2f7d3797f412fbf1490ff2cdd86efbc2b6d58c8488849629c75883e8ddd62.jpg)

![3df41ce96046120aae6e50bd87200ed4b594ed58ebbbadfb1bfd31a4dd6db1b7.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/3df41ce96046120aae6e50bd87200ed4b594ed58ebbbadfb1bfd31a4dd6db1b7.jpg)

![631f9bc7b2ed34b221114a00aeb928c3a638ac75e9a05531a48e0d9f18c0d9d9.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/631f9bc7b2ed34b221114a00aeb928c3a638ac75e9a05531a48e0d9f18c0d9d9.jpg)

![8c19a921bd1fadd310b6d1b9711b28ddcb7f554898f17cbdd51b536332085ad7.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/8c19a921bd1fadd310b6d1b9711b28ddcb7f554898f17cbdd51b536332085ad7.jpg)

![9aacd9a3cf619eb5559eb45707edf8bf4de10f208ee1e9548ec5143a32534d8f.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/9aacd9a3cf619eb5559eb45707edf8bf4de10f208ee1e9548ec5143a32534d8f.jpg)

![9e7cf414a947f659718e64ffda5f89b01f4d8c07fd1ea6e518866ced3f3027f1.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/9e7cf414a947f659718e64ffda5f89b01f4d8c07fd1ea6e518866ced3f3027f1.jpg)

![9ee62eda2b834498a8231a2665b70df4f8ac6823d9af094a7f96b5e1bc644b48.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/9ee62eda2b834498a8231a2665b70df4f8ac6823d9af094a7f96b5e1bc644b48.jpg)

![a34aa1e5888f8b9d759864203dbc052661ceddc0ccc1fd53d963a62f6ed81243.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/a34aa1e5888f8b9d759864203dbc052661ceddc0ccc1fd53d963a62f6ed81243.jpg)

![b07fccf7b80c931440ab16a3fa3da79a03dd1accc0c50ea11588a35567cfeb38.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/b07fccf7b80c931440ab16a3fa3da79a03dd1accc0c50ea11588a35567cfeb38.jpg)

![f1b15b39787d893183e390f8f8f0d5c5d99949fa39d5752a12f93b612fb572cf.jpg](../icml_results/290_FedSSI_%20Rehearsal-Free%20Continual%20Federated%20Learning%20with%20Synergistic%20%20Synaptic%20Intelligence/tables/f1b15b39787d893183e390f8f8f0d5c5d99949fa39d5752a12f93b612fb572cf.jpg)

## Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization


### Images

![00a45ce5de68f129fa871dc0f46d1811430a8a588010b00165fc0bbd6f982282.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/00a45ce5de68f129fa871dc0f46d1811430a8a588010b00165fc0bbd6f982282.jpg)

![032ec63240bcaaea4eb702f3657a6cbcaeec038c6195c2c008cb7df555567513.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/032ec63240bcaaea4eb702f3657a6cbcaeec038c6195c2c008cb7df555567513.jpg)

![0ba32dd7e8100f1bc0eb8a8dcf71e15d35c6dcd08f953a336445ab1b3c512e32.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/0ba32dd7e8100f1bc0eb8a8dcf71e15d35c6dcd08f953a336445ab1b3c512e32.jpg)

![0d9e41e03470780cc3ef646d2c899261189b23085efe607f8943a294a5d0b3b2.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/0d9e41e03470780cc3ef646d2c899261189b23085efe607f8943a294a5d0b3b2.jpg)

![15f8fda81017964f6120537814a80ff96485faaae5e02ab83704c77f835200cd.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/15f8fda81017964f6120537814a80ff96485faaae5e02ab83704c77f835200cd.jpg)

![204aeef8c5143b63b122c19904fc2968fa284d564cffa07a8044dc8e30109157.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/204aeef8c5143b63b122c19904fc2968fa284d564cffa07a8044dc8e30109157.jpg)

![2a068d5c04da7087cbe95c985a9ed6046913a32ced9457c368ad926e3e66a4fb.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/2a068d5c04da7087cbe95c985a9ed6046913a32ced9457c368ad926e3e66a4fb.jpg)

![2b4f536b46d65a71990605b535538fe68c582161ba9014addbe4827d59f7488f.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/2b4f536b46d65a71990605b535538fe68c582161ba9014addbe4827d59f7488f.jpg)

![2d847cfeddbadee81b9f2afc2259625f56747c5525e7d1763713f3360a27f042.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/2d847cfeddbadee81b9f2afc2259625f56747c5525e7d1763713f3360a27f042.jpg)

![32698d0ebfcfc7eae9085a7382466d59bd6f1b0db304512bd06a631fe69c8586.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/32698d0ebfcfc7eae9085a7382466d59bd6f1b0db304512bd06a631fe69c8586.jpg)

![352326c107275b56e10bb5ea15944ee77f141ccbdb63a07197e897fbb9f4889c.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/352326c107275b56e10bb5ea15944ee77f141ccbdb63a07197e897fbb9f4889c.jpg)

![40ce2e828f3c5d29992b6fa6a0e78eb46e7ece643a7fc03243a5f69184e36e42.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/40ce2e828f3c5d29992b6fa6a0e78eb46e7ece643a7fc03243a5f69184e36e42.jpg)

![594a118e26e1bb448118275da328669359c0e9cc2a68dff1b4e7bade7df3f3af.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/594a118e26e1bb448118275da328669359c0e9cc2a68dff1b4e7bade7df3f3af.jpg)

![5c3ae83c985b9796d4e5f1ff622dc8c0be365ab724645a809ba7c4adf6253d80.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/5c3ae83c985b9796d4e5f1ff622dc8c0be365ab724645a809ba7c4adf6253d80.jpg)

![61074167515b5961b1b3d8ed2097628bbc3a86f3b3ad0de567f0c42407f91e34.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/61074167515b5961b1b3d8ed2097628bbc3a86f3b3ad0de567f0c42407f91e34.jpg)

![6670ab7646e818ae922598240019950204845e51995e2f64b26d274a1b601f2b.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/6670ab7646e818ae922598240019950204845e51995e2f64b26d274a1b601f2b.jpg)

![6b2bc0de8a0c39da610e5788e0d24655c041e124f10fb735cd1f3347e0f19404.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/6b2bc0de8a0c39da610e5788e0d24655c041e124f10fb735cd1f3347e0f19404.jpg)

![7074d646a28a93a949883d1d862dd1451b520fc40e4a81a111b43ab369c1897f.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/7074d646a28a93a949883d1d862dd1451b520fc40e4a81a111b43ab369c1897f.jpg)

![79bea0d70606b9f15f7b730450e05188230986a8e24b809585aa437ceeb599a7.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/79bea0d70606b9f15f7b730450e05188230986a8e24b809585aa437ceeb599a7.jpg)

![808fa5f2c76b43a5f3edd41abea177ea83e45e6882fc1209cbc93fc6ee5af26f.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/808fa5f2c76b43a5f3edd41abea177ea83e45e6882fc1209cbc93fc6ee5af26f.jpg)

![83f81aceb1018367a0890af7d649bb61ae92b81c611b39eb1e0bb51d01355bbd.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/83f81aceb1018367a0890af7d649bb61ae92b81c611b39eb1e0bb51d01355bbd.jpg)

![845cb55408abbb9430cb51243e927e009c35ce88225a412d9b9e81491ce3c60a.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/845cb55408abbb9430cb51243e927e009c35ce88225a412d9b9e81491ce3c60a.jpg)

![89d008dda4b2c6fecc86ea50ee68a3bb6e52effb161fe87775a8426f7ac5bf91.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/89d008dda4b2c6fecc86ea50ee68a3bb6e52effb161fe87775a8426f7ac5bf91.jpg)

![8af961aff552c3010e599c379b6f776cc00cc99dc48899935ceb305e0b6184b5.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/8af961aff552c3010e599c379b6f776cc00cc99dc48899935ceb305e0b6184b5.jpg)

![8bafdb55a262d537151667b48ae6a53fe7fb1552cf7ab61681ebcc97ec5a2155.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/8bafdb55a262d537151667b48ae6a53fe7fb1552cf7ab61681ebcc97ec5a2155.jpg)

![99087a0a342c0554c16954064973a216cd6e7bcb6f30cf52512c57f909640f70.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/99087a0a342c0554c16954064973a216cd6e7bcb6f30cf52512c57f909640f70.jpg)

![9edf177d6362df59b904a27bfe8681688dfe03748134da6ea6d75cfc2d7e1bfb.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/9edf177d6362df59b904a27bfe8681688dfe03748134da6ea6d75cfc2d7e1bfb.jpg)

![a3c933b80f5680c04dc033c8283f1d99d38443435c5c4f80472c5cbaa0dce50c.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/a3c933b80f5680c04dc033c8283f1d99d38443435c5c4f80472c5cbaa0dce50c.jpg)

![aa87ca8f3e9ccbd6013dadb730db6119eb64bb61730c5bb7377e200fae91b6b2.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/aa87ca8f3e9ccbd6013dadb730db6119eb64bb61730c5bb7377e200fae91b6b2.jpg)

![ae8e059ea9641031bff619002c50faf58a8365d877c80af867eb54158b919159.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/ae8e059ea9641031bff619002c50faf58a8365d877c80af867eb54158b919159.jpg)

![b6368c9f196d438d4526957ae9fdc1533dfdfa4e9b7642f09e21f1043fca3e2c.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/b6368c9f196d438d4526957ae9fdc1533dfdfa4e9b7642f09e21f1043fca3e2c.jpg)

![b716a65ae06ff9b6215c1d3fb61cc8e0b8f6c8e2025025571f34a3e7550fb6b6.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/b716a65ae06ff9b6215c1d3fb61cc8e0b8f6c8e2025025571f34a3e7550fb6b6.jpg)

![b7c3ee8ae97b85ed3e46b5002c85beaa05b8701f156bd2b5d6f6b88e5cc5348f.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/b7c3ee8ae97b85ed3e46b5002c85beaa05b8701f156bd2b5d6f6b88e5cc5348f.jpg)

![b9b2d603249d03a052d7ac87d8f685267c65036f599e7290272b2a4f53927e0f.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/b9b2d603249d03a052d7ac87d8f685267c65036f599e7290272b2a4f53927e0f.jpg)

![bec9712a879a2b25d3dedf37e9405aed58f30cb0cc480eda8273ceaa25f31fbf.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/bec9712a879a2b25d3dedf37e9405aed58f30cb0cc480eda8273ceaa25f31fbf.jpg)

![bee65afa91359ece4f13d710849f42300f8e7a938a0cd6b4bd4cc4d670e3942f.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/bee65afa91359ece4f13d710849f42300f8e7a938a0cd6b4bd4cc4d670e3942f.jpg)

![c0ffa5151791cca171300278893a13d0d2603abe4620a7082def089adeaac54c.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/c0ffa5151791cca171300278893a13d0d2603abe4620a7082def089adeaac54c.jpg)

![c4fe0124776037558ac9c1a90a14aca5ba60be3297c26e9950594ce2b8dff7fc.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/c4fe0124776037558ac9c1a90a14aca5ba60be3297c26e9950594ce2b8dff7fc.jpg)

![cb3fa7d550aa04f595d85dc1d4db3b4bb17ee158909bf21d5f650be77514d910.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/cb3fa7d550aa04f595d85dc1d4db3b4bb17ee158909bf21d5f650be77514d910.jpg)

![d161a5d5e99f18db064a6f96d868a53b025023c5c196923e3635998786a8996a.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/d161a5d5e99f18db064a6f96d868a53b025023c5c196923e3635998786a8996a.jpg)

![d450499f3f83952842f2a64833369101476342c734806dcf2d0005d9b6865df6.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/d450499f3f83952842f2a64833369101476342c734806dcf2d0005d9b6865df6.jpg)

![d68abd69ff39553b520bc8caa2ccb3b0534a9c8b1162be2dc53711acc86804f6.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/d68abd69ff39553b520bc8caa2ccb3b0534a9c8b1162be2dc53711acc86804f6.jpg)

![e5a6d39718ed0255ddd18afb32364cf82db721eb3c351aa6e32061fd793b81b1.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/e5a6d39718ed0255ddd18afb32364cf82db721eb3c351aa6e32061fd793b81b1.jpg)

![e98903f76444fd5ab55bf9f5b082918becf613177419eba0dee89ea4efd677c0.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/e98903f76444fd5ab55bf9f5b082918becf613177419eba0dee89ea4efd677c0.jpg)

![f0d832c28c0fdec776267ca4ae414eeb842674b7796319fe7f42465b641dd32c.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/f0d832c28c0fdec776267ca4ae414eeb842674b7796319fe7f42465b641dd32c.jpg)

![f3b92793ebe69890b3c69b97c82c47a2745428e5f43f9c31a306506006476a72.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/f3b92793ebe69890b3c69b97c82c47a2745428e5f43f9c31a306506006476a72.jpg)

![f40c0c47849c05fc7267a74eb6cb9696bfd409e3fbf36649b4a119cf4fd5ffad.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/f40c0c47849c05fc7267a74eb6cb9696bfd409e3fbf36649b4a119cf4fd5ffad.jpg)

![f48765c6e55c30443376ef1b498119edb943a880b829e0f3862ea4febffa0302.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/f48765c6e55c30443376ef1b498119edb943a880b829e0f3862ea4febffa0302.jpg)

![f863f8222ea7f1f1d2794ab4bf9728a21e0f223f644669a9fe7504f7cce58c9e.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/f863f8222ea7f1f1d2794ab4bf9728a21e0f223f644669a9fe7504f7cce58c9e.jpg)

![f96d7f70aad7db5dcce1b6db20e0f279050665b672d745584a2e371149ee8e4c.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/f96d7f70aad7db5dcce1b6db20e0f279050665b672d745584a2e371149ee8e4c.jpg)

![fc77648778af62293c39b6ce9123c86d50d113ad755b9c55369a2daed97a116c.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/fc77648778af62293c39b6ce9123c86d50d113ad755b9c55369a2daed97a116c.jpg)

![fdfe095cf3c822c4bdda807095d671355800eb4e02d4e91826c3c581a4119832.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/images/fdfe095cf3c822c4bdda807095d671355800eb4e02d4e91826c3c581a4119832.jpg)

### Tables

![1891e6e54725e1823826004705f673c006f39b5ef3af47e7b7dae5d6bea25a04.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/tables/1891e6e54725e1823826004705f673c006f39b5ef3af47e7b7dae5d6bea25a04.jpg)

![40357861b5e605e0355d1f759921e6e962fa35137460e4c3eb6d3d259f3decec.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/tables/40357861b5e605e0355d1f759921e6e962fa35137460e4c3eb6d3d259f3decec.jpg)

![8aa922955c203299502f71325822611f94070c2c6cd9badb96e1069d4592049a.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/tables/8aa922955c203299502f71325822611f94070c2c6cd9badb96e1069d4592049a.jpg)

![93cc4f865fa806f195ba6251f868a5c1ff3021ed0cf5a73ba1ddb528ce93f308.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/tables/93cc4f865fa806f195ba6251f868a5c1ff3021ed0cf5a73ba1ddb528ce93f308.jpg)

![9d6c4440a8aa52674d6054ae758736538db4ff02bdef373758424ce8291acf8e.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/tables/9d6c4440a8aa52674d6054ae758736538db4ff02bdef373758424ce8291acf8e.jpg)

![dba036a698a14ac409e6119aca15f5cef9b934484d48516ef9b8f93ab5187a05.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/tables/dba036a698a14ac409e6119aca15f5cef9b934484d48516ef9b8f93ab5187a05.jpg)

![e5850fe3a4b46e7c9166bda7188d720fa5b5de280e7567f2f82daf012bcc4fc2.jpg](../icml_results/291_Mechanistic%20Unlearning_%20Robust%20Knowledge%20Unlearning%20and%20Editing%20via%20Mechanistic%20Localization/tables/e5850fe3a4b46e7c9166bda7188d720fa5b5de280e7567f2f82daf012bcc4fc2.jpg)

## Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs


### Images

![05b0a93940336260c44ae322a5db1fb9251f703e8b3c7ab778910c2b1d9c2ce5.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/05b0a93940336260c44ae322a5db1fb9251f703e8b3c7ab778910c2b1d9c2ce5.jpg)

![204a2f00d5bd7537a0e22f24ec0b0751bb9c742bffdcc318a322794e32a646a6.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/204a2f00d5bd7537a0e22f24ec0b0751bb9c742bffdcc318a322794e32a646a6.jpg)

![28b26443050aee43741c7aeaed1ba285993059a7d8ba2999c84131bbcaf51dd0.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/28b26443050aee43741c7aeaed1ba285993059a7d8ba2999c84131bbcaf51dd0.jpg)

![57e4dcc20f280d8727c62f939378d3b1c741f69c7e3a9f2db41e59875eee6846.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/57e4dcc20f280d8727c62f939378d3b1c741f69c7e3a9f2db41e59875eee6846.jpg)

![67425d30f712d9a09614e4c40cd0fa34c6254979fb1e231efec7e9e9ead1b06b.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/67425d30f712d9a09614e4c40cd0fa34c6254979fb1e231efec7e9e9ead1b06b.jpg)

![87dd847b658e79479035891727f61ae06c809258b56d539e234391284e37811c.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/87dd847b658e79479035891727f61ae06c809258b56d539e234391284e37811c.jpg)

![b7e928b04964e025651c571384a60c4628c227a2d251e24ffb0962378c5d753e.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/b7e928b04964e025651c571384a60c4628c227a2d251e24ffb0962378c5d753e.jpg)

![cd8fa3faeb3ea23d2442f61edc469892a9efb39991b7ba93a5b6d1d2b239739a.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/cd8fa3faeb3ea23d2442f61edc469892a9efb39991b7ba93a5b6d1d2b239739a.jpg)

![d2593e18b883940329e89ac2546c4279c7a4f9be2f7cdc3c3109d7e8bd4391cd.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/d2593e18b883940329e89ac2546c4279c7a4f9be2f7cdc3c3109d7e8bd4391cd.jpg)

![f06eb471f0dbe32b49b7182ffd82dcc8ebacb9e4e269ec960fed3ee66032ba34.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/images/f06eb471f0dbe32b49b7182ffd82dcc8ebacb9e4e269ec960fed3ee66032ba34.jpg)

### Tables

![0a311ffa85323ec7779500927b589cd88f9dfb91a782c8d64a97fa8015b0d62c.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/0a311ffa85323ec7779500927b589cd88f9dfb91a782c8d64a97fa8015b0d62c.jpg)

![14ca6f975ec95bbb2fb3313da4d810290418df979f241f79b9e4baff2b8543de.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/14ca6f975ec95bbb2fb3313da4d810290418df979f241f79b9e4baff2b8543de.jpg)

![29873e52c8ca625979cceb48d205d5d3246ca94069a56fefd054c169526e32e0.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/29873e52c8ca625979cceb48d205d5d3246ca94069a56fefd054c169526e32e0.jpg)

![4f6b74cea550e83ba949f62dcb5a183d306c86ba32d22921ef8faa56ec349659.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/4f6b74cea550e83ba949f62dcb5a183d306c86ba32d22921ef8faa56ec349659.jpg)

![55d1a4ffc39002a294d79b4317a95b53d332abae9c4d9729de2973798cf2ac87.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/55d1a4ffc39002a294d79b4317a95b53d332abae9c4d9729de2973798cf2ac87.jpg)

![6902f06a31c6f579d84cea07c8b203e71102be548cb1f53d0d005b75f2ad7422.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/6902f06a31c6f579d84cea07c8b203e71102be548cb1f53d0d005b75f2ad7422.jpg)

![7cb4b8f4990977cb06b8d397989f893057fca00ce69971b603904eb9c20da953.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/7cb4b8f4990977cb06b8d397989f893057fca00ce69971b603904eb9c20da953.jpg)

![9bdb0af2978b787d71bd86b4dd7373e765893adec1c1746453d8ce70dcd1fa60.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/9bdb0af2978b787d71bd86b4dd7373e765893adec1c1746453d8ce70dcd1fa60.jpg)

![9f7f472f522592fad547fb4119f4d0becfe4ea4fc5e27ec51db5778731c8b47d.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/9f7f472f522592fad547fb4119f4d0becfe4ea4fc5e27ec51db5778731c8b47d.jpg)

![a5ea2a746a69d041fcdf0b1e5c1ecbc6d15ffc00cbd061c2c1af8ba978446992.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/a5ea2a746a69d041fcdf0b1e5c1ecbc6d15ffc00cbd061c2c1af8ba978446992.jpg)

![c884d0d34bf4aabedebf3fcbd90f1d0675ffb9605fad6c609ccc5b53540f19e7.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/c884d0d34bf4aabedebf3fcbd90f1d0675ffb9605fad6c609ccc5b53540f19e7.jpg)

![e292c970cbade957bf631ef1d89aa36f332418d618daeb43f3cccbeb909e71ba.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/e292c970cbade957bf631ef1d89aa36f332418d618daeb43f3cccbeb909e71ba.jpg)

![e411a810df974bf173de2a14f2cfbf020f6fc5d90972b878474f5bc1f8b9225a.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/e411a810df974bf173de2a14f2cfbf020f6fc5d90972b878474f5bc1f8b9225a.jpg)

![e464a84566c131d2a23ef02baead074584c197285d9087306062a1aa1f23935b.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/e464a84566c131d2a23ef02baead074584c197285d9087306062a1aa1f23935b.jpg)

![e937e14f60c504197e9ca4849cde9a0b607e9734f38d4361603ee0e2fa093305.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/e937e14f60c504197e9ca4849cde9a0b607e9734f38d4361603ee0e2fa093305.jpg)

![f5099630f654e21d536f47e69f9ee59fed5f3c0261b17bb9bfa8f906e6c02765.jpg](../icml_results/292_Robust%20Noise%20Attenuation%20via%20Adaptive%20Pooling%20of%20Transformer%20Outputs/tables/f5099630f654e21d536f47e69f9ee59fed5f3c0261b17bb9bfa8f906e6c02765.jpg)

## Geometric Representation Condition Improves Equivariant Molecule Generation


### Images

![229f687913da2b69f56ec019f99e197cd2bb61f5bc7795a1b9eccf1f73fcd5f9.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/229f687913da2b69f56ec019f99e197cd2bb61f5bc7795a1b9eccf1f73fcd5f9.jpg)

![22b7498ba372bc05353a9164ed62081e5593f08fe3ac69a11ee978ccc8f9a133.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/22b7498ba372bc05353a9164ed62081e5593f08fe3ac69a11ee978ccc8f9a133.jpg)

![29e865466cddb6030503f49fe9b12ef54318ea723e764d3d2509a793a4119a43.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/29e865466cddb6030503f49fe9b12ef54318ea723e764d3d2509a793a4119a43.jpg)

![2c19b9748ebc120bdc2c50357b191aad9c7d0c310283871de4e2d46330192aff.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/2c19b9748ebc120bdc2c50357b191aad9c7d0c310283871de4e2d46330192aff.jpg)

![34202af124bbb26f0f4344cf5359d06eadce529077058659d8060d1268df7325.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/34202af124bbb26f0f4344cf5359d06eadce529077058659d8060d1268df7325.jpg)

![5bd19d05ac3daceacaa8831d88f3e1580f50f1d545811d0487ddb67d16f6f3b7.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/5bd19d05ac3daceacaa8831d88f3e1580f50f1d545811d0487ddb67d16f6f3b7.jpg)

![6f437ad3adc2520c154b87ece2be1a4fe070e648f558208d18cc63dba60c935f.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/6f437ad3adc2520c154b87ece2be1a4fe070e648f558208d18cc63dba60c935f.jpg)

![820659783cb99911dad1fed80d991a9ab0d41c8c81da88f08d51a7ec9b503328.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/820659783cb99911dad1fed80d991a9ab0d41c8c81da88f08d51a7ec9b503328.jpg)

![8256604c744db328377f9ba1e19969d28f293617eecf714cede07db81d092fa4.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/8256604c744db328377f9ba1e19969d28f293617eecf714cede07db81d092fa4.jpg)

![9e4116389e0ebac72e71573f2e827c3a4d70c817762e9c58575c9adb0ab59060.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/9e4116389e0ebac72e71573f2e827c3a4d70c817762e9c58575c9adb0ab59060.jpg)

![c610dbdda7f21a46e88c7b12a4e5852420c2da70d947465b10fc858d5fe3e52d.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/c610dbdda7f21a46e88c7b12a4e5852420c2da70d947465b10fc858d5fe3e52d.jpg)

![cdd7d8580b29258eda6c346e6041994ae35d077cd1bda90458c3a3d29680fb23.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/images/cdd7d8580b29258eda6c346e6041994ae35d077cd1bda90458c3a3d29680fb23.jpg)

### Tables

![1382fcc49d0c77d3b55592746339f4d78ed8e09c2238a2af0085a3230ef634bf.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/1382fcc49d0c77d3b55592746339f4d78ed8e09c2238a2af0085a3230ef634bf.jpg)

![1fb9d5cc81deba5036486ee3bc02bf98c8b47aac82e3fcdf3977c1a66e82dae9.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/1fb9d5cc81deba5036486ee3bc02bf98c8b47aac82e3fcdf3977c1a66e82dae9.jpg)

![3e6031930fa62c9dc96d6e520044301dddf668bf01ce702c4fac3c1c63d0f6de.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/3e6031930fa62c9dc96d6e520044301dddf668bf01ce702c4fac3c1c63d0f6de.jpg)

![4b157fd0c4b4b32eeff0f099f5bd9ac4d2497b0cdaf65887a504a26ad8cf69ef.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/4b157fd0c4b4b32eeff0f099f5bd9ac4d2497b0cdaf65887a504a26ad8cf69ef.jpg)

![5a76e73f5085e0ddefeb0712806f1ccc0852d485b895c62a1ae5c110455aa85c.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/5a76e73f5085e0ddefeb0712806f1ccc0852d485b895c62a1ae5c110455aa85c.jpg)

![81232e3d49f599feef70d8a1486678d3bdfe507c5257741cd24f0f85fa4a4b98.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/81232e3d49f599feef70d8a1486678d3bdfe507c5257741cd24f0f85fa4a4b98.jpg)

![8af181987605f59a0be340d99e17d1758060c3ce1656dff2f887fe111a86f728.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/8af181987605f59a0be340d99e17d1758060c3ce1656dff2f887fe111a86f728.jpg)

![d36fe3eee6c9231652804b0502982bff3c4dcf400bdd7d454a1e725497ff2820.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/d36fe3eee6c9231652804b0502982bff3c4dcf400bdd7d454a1e725497ff2820.jpg)

![d8ef7d6b51b7cdccdcd60acff6594e3ef34fd8d7592390264c30b7e7006bdf81.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/d8ef7d6b51b7cdccdcd60acff6594e3ef34fd8d7592390264c30b7e7006bdf81.jpg)

![e1a937338146638bac68e0fe32e61a91503f2a884203a847ae75bb0a491321b2.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/e1a937338146638bac68e0fe32e61a91503f2a884203a847ae75bb0a491321b2.jpg)

![f2298732df7865b6807f19e8be6e5c47c8e03c1e7ad791c43ff90d73eb8e87d3.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/f2298732df7865b6807f19e8be6e5c47c8e03c1e7ad791c43ff90d73eb8e87d3.jpg)

![fb0c094191fa363406f879d4055243a3bf05c4f38665d654310f3a30cb750d77.jpg](../icml_results/293_Geometric%20Representation%20Condition%20Improves%20Equivariant%20Molecule%20Generation/tables/fb0c094191fa363406f879d4055243a3bf05c4f38665d654310f3a30cb750d77.jpg)

## RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding


### Images

![0a8365bf27abf1590d5b3e173d1d866e7721b4e7d020909f5b009c6c31f53ef6.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/images/0a8365bf27abf1590d5b3e173d1d866e7721b4e7d020909f5b009c6c31f53ef6.jpg)

![5fa325626d838945be1488d2b7202d80c90b429a3d3e25ce6054dd2f9ce38274.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/images/5fa325626d838945be1488d2b7202d80c90b429a3d3e25ce6054dd2f9ce38274.jpg)

![f203c39f305f7db015f629ca6134c866a72f45a87bcb0e65373cc135f7c3ef79.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/images/f203c39f305f7db015f629ca6134c866a72f45a87bcb0e65373cc135f7c3ef79.jpg)

### Tables

![30c0b55371a6123ebbfd9ca20996c5848ed64961d84bffa67b1908ef72c0899f.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/tables/30c0b55371a6123ebbfd9ca20996c5848ed64961d84bffa67b1908ef72c0899f.jpg)

![45fa37fd4d81058cbf341756061990d1880df896267dde74fdee70463733b8ba.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/tables/45fa37fd4d81058cbf341756061990d1880df896267dde74fdee70463733b8ba.jpg)

![4628ef45f5d67138a543a8b4e1a49f4fd921b5306c0149fc39664ae144544c9a.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/tables/4628ef45f5d67138a543a8b4e1a49f4fd921b5306c0149fc39664ae144544c9a.jpg)

![830685bd0bff603b227bcc4659f5badb8cd7c7953708d0d48a723c224ece6832.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/tables/830685bd0bff603b227bcc4659f5badb8cd7c7953708d0d48a723c224ece6832.jpg)

![927979c8f4db2abc8a797693bde5b0de3fcce1660270dc3f51c8f957cc1c6cf3.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/tables/927979c8f4db2abc8a797693bde5b0de3fcce1660270dc3f51c8f957cc1c6cf3.jpg)

![ae10d192053f4d21772436d5bfaf8774e6ace711c11626f3db14894151bdb04b.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/tables/ae10d192053f4d21772436d5bfaf8774e6ace711c11626f3db14894151bdb04b.jpg)

![dc1805dd0c5b01b388f76c4aca3563ca7bdc5d47d6659ddfe76ce4675e022e40.jpg](../icml_results/294_RAPID_%20Long-Context%20Inference%20with%20Retrieval-Augmented%20Speculative%20Decoding/tables/dc1805dd0c5b01b388f76c4aca3563ca7bdc5d47d6659ddfe76ce4675e022e40.jpg)

## $K^2$VAE: A Koopman-Kalman Enhanced Variational AutoEncoder for Probabilistic Time Series Forecasting


### Images

![23b3c8a6eb2eec08813827c00c0c9fad01de497559ac6bb2c338223d82711d01.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/images/23b3c8a6eb2eec08813827c00c0c9fad01de497559ac6bb2c338223d82711d01.jpg)

![6cd59909edbd522595bd2d41ff419e03ad78e7afa93dbf53ebf856ec462afb5a.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/images/6cd59909edbd522595bd2d41ff419e03ad78e7afa93dbf53ebf856ec462afb5a.jpg)

![a39f89c74d72f58c1e5f9cc32363d7034d681204a5a77bed139860505fd349b1.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/images/a39f89c74d72f58c1e5f9cc32363d7034d681204a5a77bed139860505fd349b1.jpg)

![bac08db8a3cf1db903859857a67b76e546f2d8345e6bd262f10c4fda55a8ab27.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/images/bac08db8a3cf1db903859857a67b76e546f2d8345e6bd262f10c4fda55a8ab27.jpg)

![d1771b3aa2613bcc55dbfe20fb0ca50bf38f8ab391cfd3e74dfc60cd529aff17.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/images/d1771b3aa2613bcc55dbfe20fb0ca50bf38f8ab391cfd3e74dfc60cd529aff17.jpg)

![f1769be299822e60ed29353bd9822780a600c5984f31ddf2d6fbe3dfed6e24ef.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/images/f1769be299822e60ed29353bd9822780a600c5984f31ddf2d6fbe3dfed6e24ef.jpg)

![f2448cbe042b6079db299640031f2b81f0bc5790ab8bd5b9ab4cc89a80ca7929.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/images/f2448cbe042b6079db299640031f2b81f0bc5790ab8bd5b9ab4cc89a80ca7929.jpg)

### Tables

![0a2cca2c33bd459b68b712fbf117476696c216b63854c2fd625a6300e5a430d4.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/0a2cca2c33bd459b68b712fbf117476696c216b63854c2fd625a6300e5a430d4.jpg)

![2e492e4d9f0b204cf0c4af1444245788d3411d390adda36051533ec04db5671f.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/2e492e4d9f0b204cf0c4af1444245788d3411d390adda36051533ec04db5671f.jpg)

![39312d5701941908bee7bc0b69c9aca4b9d59fd06835b81463598564e0fd1691.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/39312d5701941908bee7bc0b69c9aca4b9d59fd06835b81463598564e0fd1691.jpg)

![56c7e3bcc1d62c95ee08518755820e9a7ba5b2ea7db54bf61c88d873f8f061ef.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/56c7e3bcc1d62c95ee08518755820e9a7ba5b2ea7db54bf61c88d873f8f061ef.jpg)

![6646dc12359f46e0a40901058cafdab6f8e05a3d95a2aa048461e3d50d184a6f.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/6646dc12359f46e0a40901058cafdab6f8e05a3d95a2aa048461e3d50d184a6f.jpg)

![8bc30fa7ebbaa3a8bbc399b673498d51b6d8505e16bf7d7c8caf7900381a2226.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/8bc30fa7ebbaa3a8bbc399b673498d51b6d8505e16bf7d7c8caf7900381a2226.jpg)

![8ec32bc7107123ed29ded6783ccda92fe7d10b17b75ccdc58a8467153e335f81.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/8ec32bc7107123ed29ded6783ccda92fe7d10b17b75ccdc58a8467153e335f81.jpg)

![a4ed89315a533de8ff9c8c123bba54a66d64896096d8ba988cdc862c7822c8e0.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/a4ed89315a533de8ff9c8c123bba54a66d64896096d8ba988cdc862c7822c8e0.jpg)

![bf8cde5bfc2b8ad5229a9f9fb86f19ac62b4ece2baaa0fac5266d6521d15fb35.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/bf8cde5bfc2b8ad5229a9f9fb86f19ac62b4ece2baaa0fac5266d6521d15fb35.jpg)

![cb75b67074726c40001af1faa2a31308a86304c8f2c8050ad0895cfb8c6cd734.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/cb75b67074726c40001af1faa2a31308a86304c8f2c8050ad0895cfb8c6cd734.jpg)

![d0c577a5da1415b2fd86dcaea21179b919a11a3bbf141860e4bcfbefdde0affa.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/d0c577a5da1415b2fd86dcaea21179b919a11a3bbf141860e4bcfbefdde0affa.jpg)

![dc323a8429e3bc92eb80e2ea50c4006b4cdd8631412e1867e0edac0519fc50c9.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/dc323a8429e3bc92eb80e2ea50c4006b4cdd8631412e1867e0edac0519fc50c9.jpg)

![ec726069daa4ea510b340253bb82efad2534135e3d6e3bd1665a5a58be94521e.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/ec726069daa4ea510b340253bb82efad2534135e3d6e3bd1665a5a58be94521e.jpg)

![f7cd50fe3d7305a52cb110a9dd8a1e4d3d8ff3b303160df1da8b2331acde6a99.jpg](../icml_results/295_%24K%5E2%24VAE_%20A%20Koopman-Kalman%20Enhanced%20Variational%20AutoEncoder%20for%20Probabilistic%20Time%20Series%20Forecastin/tables/f7cd50fe3d7305a52cb110a9dd8a1e4d3d8ff3b303160df1da8b2331acde6a99.jpg)

## Self-supervised Masked Graph Autoencoder via Structure-aware Curriculum


### Images

![01e11b2f4463ed9c238b5212b27dfb4980848fd433765fc93136cb2ef521378b.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/images/01e11b2f4463ed9c238b5212b27dfb4980848fd433765fc93136cb2ef521378b.jpg)

![6a14a8569760a79506b886df82e5ca0c86a76e5ac62f4d1b717f6446890c45e1.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/images/6a14a8569760a79506b886df82e5ca0c86a76e5ac62f4d1b717f6446890c45e1.jpg)

![b26962fd7e28ff1ea75a3953a77775255087280efc86298e3836eb55938c7b5d.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/images/b26962fd7e28ff1ea75a3953a77775255087280efc86298e3836eb55938c7b5d.jpg)

![eae45ca97ab3ce4c64379b666a68c3a8bf08c8931677be38c308879a98adb07e.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/images/eae45ca97ab3ce4c64379b666a68c3a8bf08c8931677be38c308879a98adb07e.jpg)

### Tables

![5d3c9d49a44e4554a20b563249b4751cca54b2a2d97b81bca925ebbecc08074a.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/5d3c9d49a44e4554a20b563249b4751cca54b2a2d97b81bca925ebbecc08074a.jpg)

![68baaf408d6e4219361961097a57035141634ce4737d805c05f7affe330dbb56.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/68baaf408d6e4219361961097a57035141634ce4737d805c05f7affe330dbb56.jpg)

![a0b6802fcf48fa9522194a98d7d3edac73c2d5d98f8d0cfb2d2013fdbfb9e2a3.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/a0b6802fcf48fa9522194a98d7d3edac73c2d5d98f8d0cfb2d2013fdbfb9e2a3.jpg)

![bc6bab053b2c2205061f5cec218b8e3fa12da8266ec0b19d520a58681443147a.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/bc6bab053b2c2205061f5cec218b8e3fa12da8266ec0b19d520a58681443147a.jpg)

![bdd29a4b22bf11a2f40a43372aaa6af39fd557b0c8ddefdae42981e7a9241ede.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/bdd29a4b22bf11a2f40a43372aaa6af39fd557b0c8ddefdae42981e7a9241ede.jpg)

![db1a0a93519aae0101b1355af8ffba6877eb1227fe88f1b24f29807c34f0f376.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/db1a0a93519aae0101b1355af8ffba6877eb1227fe88f1b24f29807c34f0f376.jpg)

![e802ec98142ced9b3e1100269b61b7f8106f0ec999e0dbf027f440c770df8f22.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/e802ec98142ced9b3e1100269b61b7f8106f0ec999e0dbf027f440c770df8f22.jpg)

![e85e2f8e1b3d1a61227eb9ea04701c4ef978dc3c992c68c23953219c5f61c269.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/e85e2f8e1b3d1a61227eb9ea04701c4ef978dc3c992c68c23953219c5f61c269.jpg)

![f46ce5f33e138d1bdacd2ed870f27189e26c6ccbf604f07cbac19a5847830c92.jpg](../icml_results/296_Self-supervised%20Masked%20Graph%20Autoencoder%20via%20Structure-aware%20Curriculum/tables/f46ce5f33e138d1bdacd2ed870f27189e26c6ccbf604f07cbac19a5847830c92.jpg)

## SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming


### Images

![55858dadb2105bc97ea2276bced02396bc40c80b3256e649c711b69a29b7bca8.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/images/55858dadb2105bc97ea2276bced02396bc40c80b3256e649c711b69a29b7bca8.jpg)

![5acb89eb2d5864755ed0baba668cad38673d41dcf13c4f39f4276040ae1ecde8.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/images/5acb89eb2d5864755ed0baba668cad38673d41dcf13c4f39f4276040ae1ecde8.jpg)

![5e0f5a8acd289d07d7daa24e9fd6b1159cc1122050b6fa57c601f74e21ad177c.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/images/5e0f5a8acd289d07d7daa24e9fd6b1159cc1122050b6fa57c601f74e21ad177c.jpg)

![5f5dc3d7d7654c72f6a7aaa7548e0e4abdb72ca49e0679c7767ed8cdb6c29e93.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/images/5f5dc3d7d7654c72f6a7aaa7548e0e4abdb72ca49e0679c7767ed8cdb6c29e93.jpg)

![67ecfd6a94b87d27e3efa49992a9d006bde9829099f7c136664551a866d683d6.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/images/67ecfd6a94b87d27e3efa49992a9d006bde9829099f7c136664551a866d683d6.jpg)

![687a88d6ee2bb5e9cae0e25c2c0feb7ee658c29eb8c8abd05460433f689182dc.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/images/687a88d6ee2bb5e9cae0e25c2c0feb7ee658c29eb8c8abd05460433f689182dc.jpg)

![dd66b7c69b65eae659a0693fc7bee87d5d4f6dcf715427d713305170eb384833.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/images/dd66b7c69b65eae659a0693fc7bee87d5d4f6dcf715427d713305170eb384833.jpg)

### Tables

![bbe9cae2a213e9f7b0dd0b17dc4f285842ba4db79a9b98e0ba8f303ce6b2f201.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/tables/bbe9cae2a213e9f7b0dd0b17dc4f285842ba4db79a9b98e0ba8f303ce6b2f201.jpg)

![c6ab6b46c98a0e791da04d6a3ce48da29071837bbc6726660388faf7c059af1b.jpg](../icml_results/297_SDP-CROWN_%20Efficient%20Bound%20Propagation%20for%20Neural%20Network%20Verification%20with%20Tightness%20of%20Semidefinit/tables/c6ab6b46c98a0e791da04d6a3ce48da29071837bbc6726660388faf7c059af1b.jpg)

## The Synergy of LLMs & RL Unlocks Offline Learning of Generalizable Language-Conditioned Policies with Low-fidelity Data


### Images

![11dfc1add87ba8fdd58e34f999baf1e66ec908b6f7620a0cef68821b10eb3df5.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/11dfc1add87ba8fdd58e34f999baf1e66ec908b6f7620a0cef68821b10eb3df5.jpg)

![519fed2bcdfba5b847a490baf2c170087374a722d3d1f7f276465aee6cb9db63.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/519fed2bcdfba5b847a490baf2c170087374a722d3d1f7f276465aee6cb9db63.jpg)

![56a9bdf3b652953bebdc0e63f05f4d25d2a0c8d1e9370f916c15146ab8932425.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/56a9bdf3b652953bebdc0e63f05f4d25d2a0c8d1e9370f916c15146ab8932425.jpg)

![7d47364ad573f9d2844b8b0b27fc9a63eecb8f9c55b068cb85ac64fe5b54b066.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/7d47364ad573f9d2844b8b0b27fc9a63eecb8f9c55b068cb85ac64fe5b54b066.jpg)

![860ab57e19e425dd6710a153f6b5ecb85a5569ae2446873211925e53ab3af92b.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/860ab57e19e425dd6710a153f6b5ecb85a5569ae2446873211925e53ab3af92b.jpg)

![9f3589ce1f9b69539b6b946c048fdde2aa37b0a953b3a3d5a57a914bed134dda.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/9f3589ce1f9b69539b6b946c048fdde2aa37b0a953b3a3d5a57a914bed134dda.jpg)

![a7de18f34a685a2d96ee4ceffba4f8cdb9257f709d0a03cf06f4cf9e3a56cc4b.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/a7de18f34a685a2d96ee4ceffba4f8cdb9257f709d0a03cf06f4cf9e3a56cc4b.jpg)

![cb5d1a3cf09f9fcce30f399e882029375fe3031185bd4ad808092e0a094744f9.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/cb5d1a3cf09f9fcce30f399e882029375fe3031185bd4ad808092e0a094744f9.jpg)

![cdefbf834bf7739437c867356eb796db72052fbb71cf8b04d29590411deda348.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/cdefbf834bf7739437c867356eb796db72052fbb71cf8b04d29590411deda348.jpg)

![d599c21be0e9e0deebeef81d6663793ed6a45da817c889687c6a828e91cd7e88.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/d599c21be0e9e0deebeef81d6663793ed6a45da817c889687c6a828e91cd7e88.jpg)

![dfd9a2b4ac41e634bcf85aa73fc375846b7f5d7cf413676e5d67c865f108839c.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/dfd9a2b4ac41e634bcf85aa73fc375846b7f5d7cf413676e5d67c865f108839c.jpg)

![fd84e08c47a2a2bcc1cd0880d500314ca0f835624efa162ae9f80ef5947b67af.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/fd84e08c47a2a2bcc1cd0880d500314ca0f835624efa162ae9f80ef5947b67af.jpg)

![fec8a81083929f4a11722a33b16aa6784b2a17cc5d8e46ab5688617b9e1c7fa7.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/images/fec8a81083929f4a11722a33b16aa6784b2a17cc5d8e46ab5688617b9e1c7fa7.jpg)

### Tables

![0f2b3c6d06c4698ea1d55900d33881ec0ffcc422a73507f51c34a2c3b7072cf1.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/0f2b3c6d06c4698ea1d55900d33881ec0ffcc422a73507f51c34a2c3b7072cf1.jpg)

![1f0c0892ee25936176849666832b7e808ef00102e9d6042a4701e82d5e0e15b8.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/1f0c0892ee25936176849666832b7e808ef00102e9d6042a4701e82d5e0e15b8.jpg)

![3b743dac62b158c49174ea4a402dd3240203636d2f775142f6be240dd704cd11.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/3b743dac62b158c49174ea4a402dd3240203636d2f775142f6be240dd704cd11.jpg)

![5cca00ecf95e1c62ed418b444bf30e8c6a3c7e8c3174f540dc82334c9c39ab89.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/5cca00ecf95e1c62ed418b444bf30e8c6a3c7e8c3174f540dc82334c9c39ab89.jpg)

![732845ba8af38d77d3fc3fe1de454fa0ba26fb200adcf958f54b9afda06da1f8.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/732845ba8af38d77d3fc3fe1de454fa0ba26fb200adcf958f54b9afda06da1f8.jpg)

![9c106152306ad035f0aabdfc10671589dafc410f23317fb8805b924c9e2d2920.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/9c106152306ad035f0aabdfc10671589dafc410f23317fb8805b924c9e2d2920.jpg)

![ab11ebe1200dcb989b9650d8600608e9864f9f17948a568cd7e1c9b82eaf795e.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/ab11ebe1200dcb989b9650d8600608e9864f9f17948a568cd7e1c9b82eaf795e.jpg)

![b0a13ac1cdf10e5bdba72844b021e9805f84bdeb89aa792e4f7fc6b00fb7e346.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/b0a13ac1cdf10e5bdba72844b021e9805f84bdeb89aa792e4f7fc6b00fb7e346.jpg)

![b7da6074d001e78ea0d057e7af1610fcccc52419378225c567d5286ef70248da.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/b7da6074d001e78ea0d057e7af1610fcccc52419378225c567d5286ef70248da.jpg)

![ba4e590e3e613bd6d6f0bdc8d47a8e19997c398186e47dc713a01db43f223cc0.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/ba4e590e3e613bd6d6f0bdc8d47a8e19997c398186e47dc713a01db43f223cc0.jpg)

![c1ec3394515ca9e29e7eee2da95257ff0c36d02d493dea88a82ba4f3f2fac5d2.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/c1ec3394515ca9e29e7eee2da95257ff0c36d02d493dea88a82ba4f3f2fac5d2.jpg)

![d83e98aa934c7318a6585a071b67243964d801c72f652d9ca262e0d88b768ada.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/d83e98aa934c7318a6585a071b67243964d801c72f652d9ca262e0d88b768ada.jpg)

![dd3a892fc0873363a0311623d2692eac6b647fc91588b801542db78bf9999b90.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/dd3a892fc0873363a0311623d2692eac6b647fc91588b801542db78bf9999b90.jpg)

![de4230ba76f97cb3bc04f1568e71de9a99cd4b462496c54312d86323211e5c09.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/de4230ba76f97cb3bc04f1568e71de9a99cd4b462496c54312d86323211e5c09.jpg)

![f334d9d8f55f1dbfda28c7f11d16903627f0e112f9313f4bf53c0fff9a0979f1.jpg](../icml_results/298_The%20Synergy%20of%20LLMs%20%26%20RL%20Unlocks%20Offline%20Learning%20of%20Generalizable%20Language-Conditioned%20Policies%20wit/tables/f334d9d8f55f1dbfda28c7f11d16903627f0e112f9313f4bf53c0fff9a0979f1.jpg)

## Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks in Superposition

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
