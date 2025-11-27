# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 57 of 100

## 目录 (Table of Contents)

1. [Just Enough Shifts: Mitigating Over-Refusal in Aligned Language Models with Targeted Representation Fine-Tuning](#Just-Enough-Shifts-Mitigating-Over-Refusal-in-Aligned-Language-Models-with-Targeted-Representation-Fine-Tuning)
2. [TSP: A Two-Sided Smoothed Primal-Dual Method for Nonconvex Bilevel Optimization](#TSP-A-Two-Sided-Smoothed-Primal-Dual-Method-for-Nonconvex-Bilevel-Optimization)
3. [SADA: Stability-guided Adaptive Diffusion Acceleration](#SADA-Stability-guided-Adaptive-Diffusion-Acceleration)
4. [Adapting Precomputed Features for Efficient Graph Condensation](#Adapting-Precomputed-Features-for-Efficient-Graph-Condensation)
5. [Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting](#Proxy-FDA-Proxy-based-Feature-Distribution-Alignment-for-Fine-tuning-Vision-Foundation-Models-without-Forgetting)
6. [On the Robustness of Reward Models for Language Model Alignment](#On-the-Robustness-of-Reward-Models-for-Language-Model-Alignment)
7. [Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](#Non-stationary-Online-Learning-for-Curved-Losses-Improved-Dynamic-Regret-via-Mixability)
8. [Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization](#Exploiting-Similarity-for-Computation-and-Communication-Efficient-Decentralized-Optimization)
9. [CLIMB: Data Foundations for Large Scale Multimodal Clinical Foundation Models](#CLIMB-Data-Foundations-for-Large-Scale-Multimodal-Clinical-Foundation-Models)
10. [Stacey: Promoting Stochastic Steepest Descent via Accelerated $\ell_p$-Smooth Nonconvex Optimization](#Stacey-Promoting-Stochastic-Steepest-Descent-via-Accelerated-ell_p-Smooth-Nonconvex-Optimization)
11. [A Physics-Augmented Deep Learning Framework for Classifying Single Molecule Force Spectroscopy Data](#A-Physics-Augmented-Deep-Learning-Framework-for-Classifying-Single-Molecule-Force-Spectroscopy-Data)
12. [RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing](#RepoAudit-An-Autonomous-LLM-Agent-for-Repository-Level-Code-Auditing)
13. [Distributionally Robust Policy Learning under Concept Drifts](#Distributionally-Robust-Policy-Learning-under-Concept-Drifts)
14. [Mind the Gap: A Practical Attack on GGUF Quantization](#Mind-the-Gap-A-Practical-Attack-on-GGUF-Quantization)
15. [Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](#Federated-In-Context-Learning-Iterative-Refinement-for-Improved-Answer-Quality)
16. [Unlocking the Power of SAM 2 for Few-Shot Segmentation](#Unlocking-the-Power-of-SAM-2-for-Few-Shot-Segmentation)
17. [Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations](#Jacobian-Sparse-Autoencoders-Sparsify-Computations-Not-Just-Activations)
18. [Privacy Amplification Through Synthetic Data: Insights from Linear Regression](#Privacy-Amplification-Through-Synthetic-Data-Insights-from-Linear-Regression)
19. [Scaling Large Motion Models with Million-Level Human Motions](#Scaling-Large-Motion-Models-with-Million-Level-Human-Motions)
20. [Locality Preserving Markovian Transition for Instance Retrieval](#Locality-Preserving-Markovian-Transition-for-Instance-Retrieval)
21. [Learning Latent Graph Structures and their Uncertainty](#Learning-Latent-Graph-Structures-and-their-Uncertainty)
22. [Homophily Enhanced Graph Domain Adaptation](#Homophily-Enhanced-Graph-Domain-Adaptation)
23. [TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](#TGDPO-Harnessing-Token-Level-Reward-Guidance-for-Enhancing-Direct-Preference-Optimization)
24. [ROS: A GNN-based Relax-Optimize-and-Sample Framework for Max-$k$-Cut Problems](#ROS-A-GNN-based-Relax-Optimize-and-Sample-Framework-for-Max-k-Cut-Problems)
25. [Retrieval-Augmented Language Model for Knowledge-aware Protein Encoding](#Retrieval-Augmented-Language-Model-for-Knowledge-aware-Protein-Encoding)
26. [Optimal Fair Learning Robust to Adversarial Distribution Shift](#Optimal-Fair-Learning-Robust-to-Adversarial-Distribution-Shift)
27. [Tokenized Bandit for LLM Decoding and Alignment](#Tokenized-Bandit-for-LLM-Decoding-and-Alignment)
28. [Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](#Enhancing-Cooperative-Multi-Agent-Reinforcement-Learning-with-State-Modelling-and-Adversarial-Exploration)
29. [Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism](#Robot-Gated-Interactive-Imitation-Learning-with-Adaptive-Intervention-Mechanism)
30. [A Rescaling-Invariant Lipschitz Bound Based on Path-Metrics for Modern ReLU Network Parameterizations](#A-Rescaling-Invariant-Lipschitz-Bound-Based-on-Path-Metrics-for-Modern-ReLU-Network-Parameterizations)
31. [PTTA: Purifying Malicious Samples for Test-Time Model Adaptation](#PTTA-Purifying-Malicious-Samples-for-Test-Time-Model-Adaptation)
32. [Linear Transformers as VAR Models:  Aligning Autoregressive Attention Mechanisms with Autoregressive Forecasting](#Linear-Transformers-as-VAR-Models-Aligning-Autoregressive-Attention-Mechanisms-with-Autoregressive-Forecasting)
33. [Tree-Sliced Wasserstein Distance with Nonlinear Projection](#Tree-Sliced-Wasserstein-Distance-with-Nonlinear-Projection)

---


## Just Enough Shifts: Mitigating Over-Refusal in Aligned Language Models with Targeted Representation Fine-Tuning

### Images

![0b348bba3660a8eeafd3aadb988a220dd4658877261aa819fff68aed605e4bc1.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/images/0b348bba3660a8eeafd3aadb988a220dd4658877261aa819fff68aed605e4bc1.jpg)

![11e629d69461ad288ad3787b8b77be2e72f3c99e3e747e94132e548def2556ee.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/images/11e629d69461ad288ad3787b8b77be2e72f3c99e3e747e94132e548def2556ee.jpg)

![386bfa2026382244efec0c51b1ae7ae5132c68bc673e34ff2d7d017b1dcd8be7.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/images/386bfa2026382244efec0c51b1ae7ae5132c68bc673e34ff2d7d017b1dcd8be7.jpg)

![b3702153cd66acae70c9d1279b55cc282ba24ebdb2879f6032e8e055fda905d0.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/images/b3702153cd66acae70c9d1279b55cc282ba24ebdb2879f6032e8e055fda905d0.jpg)

![e8504ea7c3b56fd123c8f1086db09689fbd58ebe8b5641950873ccf183788bd5.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/images/e8504ea7c3b56fd123c8f1086db09689fbd58ebe8b5641950873ccf183788bd5.jpg)

### Tables

![537329780c404623f5eba1c791d04f439b1be396921ad26ea5c746b9f0ed8616.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/tables/537329780c404623f5eba1c791d04f439b1be396921ad26ea5c746b9f0ed8616.jpg)

![539e660c963d2f7e64a74bb0eb372bf4cd34010780ea8f93701e9ef9838c13e9.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/tables/539e660c963d2f7e64a74bb0eb372bf4cd34010780ea8f93701e9ef9838c13e9.jpg)

![741861d321492ab52782f508c0d494f824a5c60ec5055c9de6eb1a57543000e8.jpg](../icml_results/1867_AutoAL_%20Automated%20Active%20Learning%20with%20Differentiable%20Query%20Strategy%20Search/tables/741861d321492ab52782f508c0d494f824a5c60ec5055c9de6eb1a57543000e8.jpg)

## Just Enough Shifts: Mitigating Over-Refusal in Aligned Language Models with Targeted Representation Fine-Tuning


### Images

![28e33424e13b0dcf02c5bba9c5c503a5cad816e0e893317e6283ac3f31c8cdf2.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/images/28e33424e13b0dcf02c5bba9c5c503a5cad816e0e893317e6283ac3f31c8cdf2.jpg)

![6b0a72ebf572a5005d4affcd6e92c91c74800c638590047cc2311f5afdc88e05.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/images/6b0a72ebf572a5005d4affcd6e92c91c74800c638590047cc2311f5afdc88e05.jpg)

![8887287d1f4676f9b5f90a0f5bdf37a88bebbf3bfcba0776758e2392d4bf6086.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/images/8887287d1f4676f9b5f90a0f5bdf37a88bebbf3bfcba0776758e2392d4bf6086.jpg)

![9baf493df0463d94075d14ef4f909f311f99379233b8d9e3a66c190455f9f8e0.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/images/9baf493df0463d94075d14ef4f909f311f99379233b8d9e3a66c190455f9f8e0.jpg)

![9e573c0eac8107f166aa90f6461a45467e21fefed5eb76c0ca847c4a93cf4fc6.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/images/9e573c0eac8107f166aa90f6461a45467e21fefed5eb76c0ca847c4a93cf4fc6.jpg)

![e3027461e407b91066db32c0e70c8790e2820be3061e9539b2f7c93ebf6dbb75.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/images/e3027461e407b91066db32c0e70c8790e2820be3061e9539b2f7c93ebf6dbb75.jpg)

![f8ea6df657490d6e791b5446e1b968b06bee87032f168318b20abf4920c71a29.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/images/f8ea6df657490d6e791b5446e1b968b06bee87032f168318b20abf4920c71a29.jpg)

### Tables

![0ce3b3e6e2626ac430d10b10ab42daafff7c9d2cdb0e8960e2f688758ce40595.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/0ce3b3e6e2626ac430d10b10ab42daafff7c9d2cdb0e8960e2f688758ce40595.jpg)

![17d68cff6c1af6c4c841f1049c00ca464e89ab0a7a8c21a104acc8877cf114c8.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/17d68cff6c1af6c4c841f1049c00ca464e89ab0a7a8c21a104acc8877cf114c8.jpg)

![463b24252e6b2f9206e169e9b06dc0c2ab55f1c3d30c10affc72bced222581f4.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/463b24252e6b2f9206e169e9b06dc0c2ab55f1c3d30c10affc72bced222581f4.jpg)

![5942f110fbd7a31be972a8624779f432d5a7a0bb56468429f532e567c55b3c36.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/5942f110fbd7a31be972a8624779f432d5a7a0bb56468429f532e567c55b3c36.jpg)

![6625189078298e606c3cda3f2c307cc433a3c92bb2cc8c5ccbb29ed1927bb4ea.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/6625189078298e606c3cda3f2c307cc433a3c92bb2cc8c5ccbb29ed1927bb4ea.jpg)

![71f714875e7292885587aa4456323111d69ea048c74b8e25231b7db3d952c360.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/71f714875e7292885587aa4456323111d69ea048c74b8e25231b7db3d952c360.jpg)

![9b897daee79e42b1abf110bd3f7e21b08f07c414612a526e08a0475518dbb9e2.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/9b897daee79e42b1abf110bd3f7e21b08f07c414612a526e08a0475518dbb9e2.jpg)

![c2cc98fe8db0635068e960a65aecd3261742968122007ff46e0613fed6089381.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/c2cc98fe8db0635068e960a65aecd3261742968122007ff46e0613fed6089381.jpg)

![ff3449df354e92a6373fffe4ddc3a6a14f04e6d623b6fe028dee8e4d2ff8b8b3.jpg](../icml_results/1868_Just%20Enough%20Shifts_%20Mitigating%20Over-Refusal%20in%20Aligned%20Language%20Models%20with%20Targeted%20Representation%20/tables/ff3449df354e92a6373fffe4ddc3a6a14f04e6d623b6fe028dee8e4d2ff8b8b3.jpg)

## TSP: A Two-Sided Smoothed Primal-Dual Method for Nonconvex Bilevel Optimization


### Images

![199bdbaed4cbafdba4432eaa7c2cd951ac26d24cfcc84b3c8342ad303b59706c.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/images/199bdbaed4cbafdba4432eaa7c2cd951ac26d24cfcc84b3c8342ad303b59706c.jpg)

![5379ae7d59c7b70950d6bf4d0df3f5adb2c1ade02711e6c1a09a9f3ee933a112.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/images/5379ae7d59c7b70950d6bf4d0df3f5adb2c1ade02711e6c1a09a9f3ee933a112.jpg)

![b0e9009392559e64aaccaafbec7feae4be04272c807cebf146af5e513b3480f6.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/images/b0e9009392559e64aaccaafbec7feae4be04272c807cebf146af5e513b3480f6.jpg)

![d4b0e52b972f475b340826b317a7d09b72f6f7e8bd3d73fd920bc500bbe4892d.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/images/d4b0e52b972f475b340826b317a7d09b72f6f7e8bd3d73fd920bc500bbe4892d.jpg)

![e90c23b323020449e40c98cd33457ea17a635429101d8799a9c6973f637bd61a.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/images/e90c23b323020449e40c98cd33457ea17a635429101d8799a9c6973f637bd61a.jpg)

### Tables

![0ebe1f17d6e5cfac4f3590ac2919754a72d753ee8ef02c484ea927c7bf088c2d.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/tables/0ebe1f17d6e5cfac4f3590ac2919754a72d753ee8ef02c484ea927c7bf088c2d.jpg)

![105ec08720b7bb39292b53eb49373243d981fcfbff44f5fe99d09947c96d7298.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/tables/105ec08720b7bb39292b53eb49373243d981fcfbff44f5fe99d09947c96d7298.jpg)

![9cea279d1612c62199a77e95625e999f161a9830935a566f6b12cff9c1a368ef.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/tables/9cea279d1612c62199a77e95625e999f161a9830935a566f6b12cff9c1a368ef.jpg)

![ccabf79ab043d0f2abb842295ce758e21faf58f25353fffcf0b7021c1309fe2f.jpg](../icml_results/1869_TSP_%20A%20Two-Sided%20Smoothed%20Primal-Dual%20Method%20for%20Nonconvex%20Bilevel%20Optimization/tables/ccabf79ab043d0f2abb842295ce758e21faf58f25353fffcf0b7021c1309fe2f.jpg)

## SADA: Stability-guided Adaptive Diffusion Acceleration


### Images

![08b7ba7f3635fd859cdda19e5f27adca606d29e85eaab94091eded5a1f321197.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/08b7ba7f3635fd859cdda19e5f27adca606d29e85eaab94091eded5a1f321197.jpg)

![0ad9bf0346c096b3eab49cf2cb710fdb5d8f5a9c64caac93ec0ca040a757b9b6.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/0ad9bf0346c096b3eab49cf2cb710fdb5d8f5a9c64caac93ec0ca040a757b9b6.jpg)

![1c0677bef5aa3c21dd230ab1fe562f01febaa752eb2a75fae98cf3598d1c2d91.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/1c0677bef5aa3c21dd230ab1fe562f01febaa752eb2a75fae98cf3598d1c2d91.jpg)

![2ba3a44ad036ca3da942daf4c524ee9ab63f9c39ad871de9ffbf72ea19e399f7.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/2ba3a44ad036ca3da942daf4c524ee9ab63f9c39ad871de9ffbf72ea19e399f7.jpg)

![3713e633914503f204cf2cb532b85104bd700e1399e78a8cccfcb545ae7c299c.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/3713e633914503f204cf2cb532b85104bd700e1399e78a8cccfcb545ae7c299c.jpg)

![3a1b3d583bc1c407c98c253324b0d03a63017040d27ca41ac75c20de58c52fb3.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/3a1b3d583bc1c407c98c253324b0d03a63017040d27ca41ac75c20de58c52fb3.jpg)

![4d3f3e8a771736af1e0cbc66feb6d22dc12d433160c6dad7b595960236637b51.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/4d3f3e8a771736af1e0cbc66feb6d22dc12d433160c6dad7b595960236637b51.jpg)

![72133f2ea686ae39688f0c460f593814e822685968ed9500d588913673937b30.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/72133f2ea686ae39688f0c460f593814e822685968ed9500d588913673937b30.jpg)

![745958f330399a9f4c031b3a2de6492bf4aa92adf017338248897ebb140dcbe6.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/745958f330399a9f4c031b3a2de6492bf4aa92adf017338248897ebb140dcbe6.jpg)

![e5d2d2e51c01c4993a42d3022b78fee37cdf257482ccabec4d3fdf3d4f22191a.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/images/e5d2d2e51c01c4993a42d3022b78fee37cdf257482ccabec4d3fdf3d4f22191a.jpg)

### Tables

![91075849dd8138eb2e5a58662564bfa018b99259fb7a314b2cb99c7205160365.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/tables/91075849dd8138eb2e5a58662564bfa018b99259fb7a314b2cb99c7205160365.jpg)

![cb239867af4dd2153fac07daf5150f18110b0f93931199d26c4aa4fad58b0075.jpg](../icml_results/1870_SADA_%20Stability-guided%20Adaptive%20Diffusion%20Acceleration/tables/cb239867af4dd2153fac07daf5150f18110b0f93931199d26c4aa4fad58b0075.jpg)

## Adapting Precomputed Features for Efficient Graph Condensation


### Images

![21bfe15230f74a736662845903724842d1eaccabfd2b061344bcb53df72b23de.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/images/21bfe15230f74a736662845903724842d1eaccabfd2b061344bcb53df72b23de.jpg)

![313f6d073d2f4f2a4812cd78a6107008276065db2384700489b128ae6d8518e4.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/images/313f6d073d2f4f2a4812cd78a6107008276065db2384700489b128ae6d8518e4.jpg)

![6aa88f7858cb35fb70ab0ea815967061ff26d0fefb75bcc38314c45607e450cf.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/images/6aa88f7858cb35fb70ab0ea815967061ff26d0fefb75bcc38314c45607e450cf.jpg)

![8344a74cbab3a9589681e8e9bdbc7cb626d8d1f8cb089dab47ae89209de3df97.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/images/8344a74cbab3a9589681e8e9bdbc7cb626d8d1f8cb089dab47ae89209de3df97.jpg)

![c0b960ae9f02d754cf956db119f31917c573127195c395bd8a7b2f3e8ada0986.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/images/c0b960ae9f02d754cf956db119f31917c573127195c395bd8a7b2f3e8ada0986.jpg)

![dd218190cc733c82b5f35cb19d792cf25b4058fab45a10071909ce240ff75d6b.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/images/dd218190cc733c82b5f35cb19d792cf25b4058fab45a10071909ce240ff75d6b.jpg)

![fe6b6d29080c4c3388bb44e70783fdc07b08b0662bd6ec53078dc1bc150c93cf.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/images/fe6b6d29080c4c3388bb44e70783fdc07b08b0662bd6ec53078dc1bc150c93cf.jpg)

### Tables

![0d95eb3b4e339f3c42f675b5b2f6da7abe201d14e43b39d79a51ab148a949417.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/tables/0d95eb3b4e339f3c42f675b5b2f6da7abe201d14e43b39d79a51ab148a949417.jpg)

![35977d96a03cf9fa80971adefb3c4b2ecdc057a3f8c857a8723351c433213213.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/tables/35977d96a03cf9fa80971adefb3c4b2ecdc057a3f8c857a8723351c433213213.jpg)

![3fdafbf6532f6c555184f8ac340113517674bd56015ffd57d26c7eedefd8db0d.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/tables/3fdafbf6532f6c555184f8ac340113517674bd56015ffd57d26c7eedefd8db0d.jpg)

![5641fcaec2ee89380f1abada2cb0165ae008cadfe0fd262cabf1569fe0d2f567.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/tables/5641fcaec2ee89380f1abada2cb0165ae008cadfe0fd262cabf1569fe0d2f567.jpg)

![7aacc6f21082848a53ce14be800ad2fc66fcf16ec8a5328ca713f7f76d163709.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/tables/7aacc6f21082848a53ce14be800ad2fc66fcf16ec8a5328ca713f7f76d163709.jpg)

![d4bb1e5c69ba470c68a2a4f874ce6edf7a5e832cb32c35231196975aafe43d1b.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/tables/d4bb1e5c69ba470c68a2a4f874ce6edf7a5e832cb32c35231196975aafe43d1b.jpg)

![e7b58c858b3fe19b661b485aac5c123e3fc9ca99a804a10168b546d644fd5dc0.jpg](../icml_results/1871_Adapting%20Precomputed%20Features%20for%20Efficient%20Graph%20Condensation/tables/e7b58c858b3fe19b661b485aac5c123e3fc9ca99a804a10168b546d644fd5dc0.jpg)

## Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting


### Images

![3cd0b6acb004c57796a5831f33a8f0160b8ef4ea83f2e5634a6237abc9156175.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/3cd0b6acb004c57796a5831f33a8f0160b8ef4ea83f2e5634a6237abc9156175.jpg)

![3f04dfd3be26cea34d7beb231ea075a6558ea3a550b5ce3051b9599df12b094d.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/3f04dfd3be26cea34d7beb231ea075a6558ea3a550b5ce3051b9599df12b094d.jpg)

![461c36a03e77c000e804a5f5816a0b1d6e887c573485db2fdcfc404492463d2a.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/461c36a03e77c000e804a5f5816a0b1d6e887c573485db2fdcfc404492463d2a.jpg)

![95b7632e60452437515db6429af2af5cc4826e0e69fe2790c28673d67714ddeb.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/95b7632e60452437515db6429af2af5cc4826e0e69fe2790c28673d67714ddeb.jpg)

![99335d384a263bf1c89ecf8092678c390dcbf080c1d37d5f754a9c053e4cfe11.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/99335d384a263bf1c89ecf8092678c390dcbf080c1d37d5f754a9c053e4cfe11.jpg)

![b8f78037e8ffb1e6815fbc1d2d0bb6cc063d664589d728feb6fb94dcc58a8c53.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/b8f78037e8ffb1e6815fbc1d2d0bb6cc063d664589d728feb6fb94dcc58a8c53.jpg)

![e66a33faf4b0656cfa3f9e1e1a29aff958159f5a87de66d121bf2a78c87dfab7.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/e66a33faf4b0656cfa3f9e1e1a29aff958159f5a87de66d121bf2a78c87dfab7.jpg)

![ef0db968464e1cbc18792e1ebb91f41d963450bc592fd2671af0e79110aa5499.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/images/ef0db968464e1cbc18792e1ebb91f41d963450bc592fd2671af0e79110aa5499.jpg)

### Tables

![0820c4a5c6d8cc9126382fe2f2e23c26e57f903fe2de34ff7590d93430e9c935.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/0820c4a5c6d8cc9126382fe2f2e23c26e57f903fe2de34ff7590d93430e9c935.jpg)

![095d715fe3af8cabd243607db8347ffe268479bea9ebabc70718babb323f9a0d.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/095d715fe3af8cabd243607db8347ffe268479bea9ebabc70718babb323f9a0d.jpg)

![297317456e6cc3904da3ae46b3b3f3b4a1a9fb97e965dae9bb996766c6c124fe.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/297317456e6cc3904da3ae46b3b3f3b4a1a9fb97e965dae9bb996766c6c124fe.jpg)

![341e1072ca6ed710dcecfbdf8f03ab0903103e90f88181c52569c6a148f83643.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/341e1072ca6ed710dcecfbdf8f03ab0903103e90f88181c52569c6a148f83643.jpg)

![743209e2dca87ca3027d83fef0fc10c6c4bf8ff05082715fdbef01be6c6b22ea.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/743209e2dca87ca3027d83fef0fc10c6c4bf8ff05082715fdbef01be6c6b22ea.jpg)

![79f34fc696bb5350fa1ce26f3e100a44867c33b1cfc70198714742d996f5e16f.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/79f34fc696bb5350fa1ce26f3e100a44867c33b1cfc70198714742d996f5e16f.jpg)

![81b750473b0399c84ea89c799b333d1c180a6e050b9aa052389f679522520561.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/81b750473b0399c84ea89c799b333d1c180a6e050b9aa052389f679522520561.jpg)

![8318033943783d2ea6a2b52e6b8b5d8fa5d4c773f7d58a075132cf7aec43f97a.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/8318033943783d2ea6a2b52e6b8b5d8fa5d4c773f7d58a075132cf7aec43f97a.jpg)

![a2e1f0d5f3e00e0ef8b0c1160f46ade1625e3c38c69cdf9ca67f8125df71e2c7.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/a2e1f0d5f3e00e0ef8b0c1160f46ade1625e3c38c69cdf9ca67f8125df71e2c7.jpg)

![c6893b76588e5852826853228324847efad1717cbf235f8185f5132c1754558d.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/c6893b76588e5852826853228324847efad1717cbf235f8185f5132c1754558d.jpg)

![c9e25f9d6cae60706c3da9f3845059a37f6bd731cb1b6e1ed3d7055fcd19846f.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/c9e25f9d6cae60706c3da9f3845059a37f6bd731cb1b6e1ed3d7055fcd19846f.jpg)

![f36729c4b40c0bc55c92653e571e678236ca0dfbe3eee7cd9be4b8efa617aafe.jpg](../icml_results/1872_Proxy-FDA_%20Proxy-based%20Feature%20Distribution%20Alignment%20for%20Fine-tuning%20Vision%20Foundation%20Models%20witho/tables/f36729c4b40c0bc55c92653e571e678236ca0dfbe3eee7cd9be4b8efa617aafe.jpg)

## On the Robustness of Reward Models for Language Model Alignment


### Images

![13620289ec189e80bc9d5c3fdd4d979274178d4810cecf692ebfd3ba69a9b7f7.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/images/13620289ec189e80bc9d5c3fdd4d979274178d4810cecf692ebfd3ba69a9b7f7.jpg)

![16bb7d01b43a01baf6d177163cb8efc0bbb68f781cc36c4b0d1e8d71205da4df.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/images/16bb7d01b43a01baf6d177163cb8efc0bbb68f781cc36c4b0d1e8d71205da4df.jpg)

![3ae88e5448d6f94deb66c0748aebbb081f8783637d8b90f67251c3f4dd4579a8.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/images/3ae88e5448d6f94deb66c0748aebbb081f8783637d8b90f67251c3f4dd4579a8.jpg)

![8e4c1590d966eb90ca88b0c020309d8f12a8fde5cda5f01bfa4f98e918d0b71e.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/images/8e4c1590d966eb90ca88b0c020309d8f12a8fde5cda5f01bfa4f98e918d0b71e.jpg)

![99e43461e2cbd8a82de13b2a34bb7335e4ec7b56f5b3aafedafb143c54ee2a52.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/images/99e43461e2cbd8a82de13b2a34bb7335e4ec7b56f5b3aafedafb143c54ee2a52.jpg)

![cebbeeb48faa4cd17a8bdfd18ccd7911e1c9ea60dc6fae45a512fe1362f634ef.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/images/cebbeeb48faa4cd17a8bdfd18ccd7911e1c9ea60dc6fae45a512fe1362f634ef.jpg)

### Tables

![2f1b9a7e7c6cd50c5b1862c571686f19d87cabf4d6bcab27fde33b988b6f7467.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/tables/2f1b9a7e7c6cd50c5b1862c571686f19d87cabf4d6bcab27fde33b988b6f7467.jpg)

![3feecfdd411aca941ba056ebda96a25bf1d94bcc160fc678c7529b1ffefe9791.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/tables/3feecfdd411aca941ba056ebda96a25bf1d94bcc160fc678c7529b1ffefe9791.jpg)

![652a75d492eb0d08cb226d7c421928d3f9bb8d2ce359b40dd329f749d990dec6.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/tables/652a75d492eb0d08cb226d7c421928d3f9bb8d2ce359b40dd329f749d990dec6.jpg)

![8d71a85ad6994bcbcafbd6b06e23c86d9e4153982bed80392d9f0dd30036ee63.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/tables/8d71a85ad6994bcbcafbd6b06e23c86d9e4153982bed80392d9f0dd30036ee63.jpg)

![ba6307b163b715c10d7cdb71b6c6e5c702df8d6a9c6277d98607987b61633ffa.jpg](../icml_results/1873_On%20the%20Robustness%20of%20Reward%20Models%20for%20Language%20Model%20Alignment/tables/ba6307b163b715c10d7cdb71b6c6e5c702df8d6a9c6277d98607987b61633ffa.jpg)

## Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability


### Images

![39acae398e3eee33fc8cecc9d012353383f872c5ca7370f2c94ad53d53d57fb8.jpg](../icml_results/1874_Non-stationary%20Online%20Learning%20for%20Curved%20Losses_%20Improved%20Dynamic%20Regret%20via%20Mixability/images/39acae398e3eee33fc8cecc9d012353383f872c5ca7370f2c94ad53d53d57fb8.jpg)

### Tables

![318d819870d37726fc3057bd87abc9bbeb2d998b6b6977b9d88d59bed2b6c98f.jpg](../icml_results/1874_Non-stationary%20Online%20Learning%20for%20Curved%20Losses_%20Improved%20Dynamic%20Regret%20via%20Mixability/tables/318d819870d37726fc3057bd87abc9bbeb2d998b6b6977b9d88d59bed2b6c98f.jpg)

## Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization


### Images

![0aad0d8c6b5c3bcafa308b07779feba036f44353bff9e9eb7845a0dd87ec1a8d.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/images/0aad0d8c6b5c3bcafa308b07779feba036f44353bff9e9eb7845a0dd87ec1a8d.jpg)

![0b949f17f825c896d175cdb944228171b86957c040fa7d87ae8722db3d4d8b9a.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/images/0b949f17f825c896d175cdb944228171b86957c040fa7d87ae8722db3d4d8b9a.jpg)

![5e78baecfab3505eed31a3c0bfe76237b15895547f448cfe3ce9d5694b37e1fb.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/images/5e78baecfab3505eed31a3c0bfe76237b15895547f448cfe3ce9d5694b37e1fb.jpg)

![852cbdd9766f5149a2c42dc3b9b8957635942aaa7a4877aa82ea07d6559d6343.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/images/852cbdd9766f5149a2c42dc3b9b8957635942aaa7a4877aa82ea07d6559d6343.jpg)

### Tables

![27aa36277c2da7023304bd47502f371b07f3bb4eb8f7377c077742ebb3e3d61b.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/tables/27aa36277c2da7023304bd47502f371b07f3bb4eb8f7377c077742ebb3e3d61b.jpg)

![2cc43bd822e374acb96418b2d71da2d45d44fe4fb6d3d165e845e0dd7225664c.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/tables/2cc43bd822e374acb96418b2d71da2d45d44fe4fb6d3d165e845e0dd7225664c.jpg)

![465049da665080a027d90c5e5d0d1d16f4e3060ee7fcc62c3d2660e36d3dedf9.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/tables/465049da665080a027d90c5e5d0d1d16f4e3060ee7fcc62c3d2660e36d3dedf9.jpg)

![68268b78301dbf915609f767acc96d69b59fc638355fc149913f117e5f04d118.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/tables/68268b78301dbf915609f767acc96d69b59fc638355fc149913f117e5f04d118.jpg)

![6a50da0533ce7e8d9f4cebc7609f4d15fc216e38b0992ce43c12aa32ac319cfe.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/tables/6a50da0533ce7e8d9f4cebc7609f4d15fc216e38b0992ce43c12aa32ac319cfe.jpg)

![9ca60a5d55c19071b474b04cf7021449005fe3408e870db2b8beceb532896bd2.jpg](../icml_results/1875_Exploiting%20Similarity%20for%20Computation%20and%20Communication-Efficient%20Decentralized%20Optimization/tables/9ca60a5d55c19071b474b04cf7021449005fe3408e870db2b8beceb532896bd2.jpg)

## CLIMB: Data Foundations for Large Scale Multimodal Clinical Foundation Models


### Images

![286852a89574435f9f6952a77444498887fb500334078a43a663aa68bcca4240.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/images/286852a89574435f9f6952a77444498887fb500334078a43a663aa68bcca4240.jpg)

![33aa6290cc749a2c3303f874de52a62494f20f436d96613fece10b59e8d0315d.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/images/33aa6290cc749a2c3303f874de52a62494f20f436d96613fece10b59e8d0315d.jpg)

![590d6608efb4c76991f3d997a85ee73e8b02804b4ad59f2649ebeeafd02bc46d.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/images/590d6608efb4c76991f3d997a85ee73e8b02804b4ad59f2649ebeeafd02bc46d.jpg)

![6a26256d9489d484ff420b3165c14e3cf27c59ebe8089e8c48b8bdb3d601672a.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/images/6a26256d9489d484ff420b3165c14e3cf27c59ebe8089e8c48b8bdb3d601672a.jpg)

![6be05a27089a622749e79ebf7957fc4ff636747ba7cedff45274b2f3a8331599.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/images/6be05a27089a622749e79ebf7957fc4ff636747ba7cedff45274b2f3a8331599.jpg)

![be0c299827598df45909d5b99a8ffbeeb23a19536b65cde0a9c510ce67b8b8c5.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/images/be0c299827598df45909d5b99a8ffbeeb23a19536b65cde0a9c510ce67b8b8c5.jpg)

![e994c39e82f7072bab5051eaeb215e7dafc0413565c8fd5c54b8e3d7eac70fc7.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/images/e994c39e82f7072bab5051eaeb215e7dafc0413565c8fd5c54b8e3d7eac70fc7.jpg)

### Tables

![02a29bedde6ea50b24c2adff5d308ad3d032891477827ca2cdb0deb0538fd363.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/02a29bedde6ea50b24c2adff5d308ad3d032891477827ca2cdb0deb0538fd363.jpg)

![02c77e39e83507a244788c3299a82efe14a896c25529b59364c9e81ea794a711.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/02c77e39e83507a244788c3299a82efe14a896c25529b59364c9e81ea794a711.jpg)

![03d346ae8f45e684d4a727f0bdb0bf5634d0b1ead6445a1d08ce42e2a1025c03.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/03d346ae8f45e684d4a727f0bdb0bf5634d0b1ead6445a1d08ce42e2a1025c03.jpg)

![0f2e681849a3ad2f07e16435a0dd83958945a7c5cd6d8b83cab005d77207712f.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/0f2e681849a3ad2f07e16435a0dd83958945a7c5cd6d8b83cab005d77207712f.jpg)

![16b0372be8976642eca602799a445cdcdb7ba7ad0dcbe7d4055ebfb6a6b5b266.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/16b0372be8976642eca602799a445cdcdb7ba7ad0dcbe7d4055ebfb6a6b5b266.jpg)

![2392dae147e3cea772cdb028d06d157b6e69684bbebc107d6a989fa01754e412.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/2392dae147e3cea772cdb028d06d157b6e69684bbebc107d6a989fa01754e412.jpg)

![274003728dc6c588483d3b454cf1add7385d16fadcb68702a522e35f48973ef2.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/274003728dc6c588483d3b454cf1add7385d16fadcb68702a522e35f48973ef2.jpg)

![2d80091511ac019a18dea6b7a928e6e59591a0ee83a42aee77619a3ae8d9af74.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/2d80091511ac019a18dea6b7a928e6e59591a0ee83a42aee77619a3ae8d9af74.jpg)

![305d4e7d2eeb8f89fc84749932bbbd69c95ea8fe87fe92057b219a2196c3cf78.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/305d4e7d2eeb8f89fc84749932bbbd69c95ea8fe87fe92057b219a2196c3cf78.jpg)

![45a3bc26456697e00d2755e86fa66c41d2b979b19a087a1f8172a654924434dd.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/45a3bc26456697e00d2755e86fa66c41d2b979b19a087a1f8172a654924434dd.jpg)

![46276136bd6c655c00e879a5b63766d82969c63b85f480044c261afc1c65ccdc.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/46276136bd6c655c00e879a5b63766d82969c63b85f480044c261afc1c65ccdc.jpg)

![4e388e16fed46e69c377fa15b24c106928d5a052029a0a450baecac17346483d.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/4e388e16fed46e69c377fa15b24c106928d5a052029a0a450baecac17346483d.jpg)

![51bd28c21ad4da00c78126190f7e4556c8e9c6ac0de9df23cf0bbab8ae1eca0d.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/51bd28c21ad4da00c78126190f7e4556c8e9c6ac0de9df23cf0bbab8ae1eca0d.jpg)

![52808f6593a07d4180e6af33de31b6c39d6e2c150ba89096297b689f05a9d849.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/52808f6593a07d4180e6af33de31b6c39d6e2c150ba89096297b689f05a9d849.jpg)

![5d637b25ab9501329eab02628f253a15564fe4c0714a1a1beb912e7cb6e0850c.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/5d637b25ab9501329eab02628f253a15564fe4c0714a1a1beb912e7cb6e0850c.jpg)

![641100f99ce88f05172f7f2af0c2ee66377186280d46ea80474a4c24d5429914.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/641100f99ce88f05172f7f2af0c2ee66377186280d46ea80474a4c24d5429914.jpg)

![699de28d80e718dbef833cf51a55d037a206fc14828193b4d195db71afdc45e8.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/699de28d80e718dbef833cf51a55d037a206fc14828193b4d195db71afdc45e8.jpg)

![6d53372cf49cdab5e9912aaee30329b046528af6ca7620599f3768f58265828e.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/6d53372cf49cdab5e9912aaee30329b046528af6ca7620599f3768f58265828e.jpg)

![6e22a87c5cbc503e2ff083d691148759344b0a0baa5b5b0a3a1b4aa4287c34c7.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/6e22a87c5cbc503e2ff083d691148759344b0a0baa5b5b0a3a1b4aa4287c34c7.jpg)

![6ed5ecd857e4dc8bb2aceae1ec0b950b135c9557ad9f4b62abf488ac8f2db8e5.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/6ed5ecd857e4dc8bb2aceae1ec0b950b135c9557ad9f4b62abf488ac8f2db8e5.jpg)

![76e884d9366dd36cceaca5df0ffb9499545157e4175bfb6c3c9707018604fe6c.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/76e884d9366dd36cceaca5df0ffb9499545157e4175bfb6c3c9707018604fe6c.jpg)

![82b003e20bee35d7fdcf37bef856361f20e9231ee6e7cee24629bacd40b4d5de.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/82b003e20bee35d7fdcf37bef856361f20e9231ee6e7cee24629bacd40b4d5de.jpg)

![919c6aaf71dd2757e2bc59363eab107c981b7c5c0dd8bf2abe45e4c6d9fc87ba.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/919c6aaf71dd2757e2bc59363eab107c981b7c5c0dd8bf2abe45e4c6d9fc87ba.jpg)

![9deb46c94ed2a344ebe864b13f2cf3156620670c8b886c2184a58e2b0f8a350c.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/9deb46c94ed2a344ebe864b13f2cf3156620670c8b886c2184a58e2b0f8a350c.jpg)

![a540215fb0f675ab43b5259b20aca7c134b13e7c216be93e69e4051b73aa3b33.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/a540215fb0f675ab43b5259b20aca7c134b13e7c216be93e69e4051b73aa3b33.jpg)

![b671d937a0af6167d849b89458226bd42165282eb23f2484ec82f4513de4190e.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/b671d937a0af6167d849b89458226bd42165282eb23f2484ec82f4513de4190e.jpg)

![b8d0c1363e87c996b3cf7c939f63f1460fefd2899276227a9bf573a2b6925109.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/b8d0c1363e87c996b3cf7c939f63f1460fefd2899276227a9bf573a2b6925109.jpg)

![bfdf3b7306790f28d5758c9263615c7034bb4d50f463db369e7f7cd9fece3c30.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/bfdf3b7306790f28d5758c9263615c7034bb4d50f463db369e7f7cd9fece3c30.jpg)

![c175bf1fc912f222bee6709b62c8a8ffd6d6358c2fdc321d308102caed5fcbbe.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/c175bf1fc912f222bee6709b62c8a8ffd6d6358c2fdc321d308102caed5fcbbe.jpg)

![c19e4bddee938062a519cce29dc34a91f0cbcc4bf24de86b8db224fb779ee48d.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/c19e4bddee938062a519cce29dc34a91f0cbcc4bf24de86b8db224fb779ee48d.jpg)

![c27c277cba7d258aff683c893b2663f61f4bd18106501f00e035acd9416b4acc.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/c27c277cba7d258aff683c893b2663f61f4bd18106501f00e035acd9416b4acc.jpg)

![d71335ea18f181c0d26d2ca2263a2736121788e21696d631970e0eee72178d89.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/d71335ea18f181c0d26d2ca2263a2736121788e21696d631970e0eee72178d89.jpg)

![f76916a3390832883d0914aad84afc183e96b386f06be2a661891c6a73e8e363.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/f76916a3390832883d0914aad84afc183e96b386f06be2a661891c6a73e8e363.jpg)

![f893170a073438145658ff865833959db486b295f3ad2564895e3b827ff1d4a1.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/f893170a073438145658ff865833959db486b295f3ad2564895e3b827ff1d4a1.jpg)

![fa97fbe91a990a089c51c0d4352c0a10236c3c14880fbc469c6f7893c1621aa0.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/fa97fbe91a990a089c51c0d4352c0a10236c3c14880fbc469c6f7893c1621aa0.jpg)

![fb007e614b3184d1ff8ea9bf495b67ec52785f718fe11f846509b67e8a5d2ffe.jpg](../icml_results/1876_CLIMB_%20Data%20Foundations%20for%20Large%20Scale%20Multimodal%20Clinical%20Foundation%20Models/tables/fb007e614b3184d1ff8ea9bf495b67ec52785f718fe11f846509b67e8a5d2ffe.jpg)

## Stacey: Promoting Stochastic Steepest Descent via Accelerated $\ell_p$-Smooth Nonconvex Optimization


### Images

![5cb79b4419cdaacaa3d295a12691aef3ca454470e95280ae345361f884f2313d.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/images/5cb79b4419cdaacaa3d295a12691aef3ca454470e95280ae345361f884f2313d.jpg)

![7c6d17b6e44cbac7a2d03ab81288d67a01199023250faa3cbc84e1755fde50d6.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/images/7c6d17b6e44cbac7a2d03ab81288d67a01199023250faa3cbc84e1755fde50d6.jpg)

### Tables

![1cc229c8ec5760da0ec7276b586ee863ddf72a22caf4db2e9395545acb6b68ce.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/tables/1cc229c8ec5760da0ec7276b586ee863ddf72a22caf4db2e9395545acb6b68ce.jpg)

![3eb34b1c1bc192f369fe0f1c1329734beb04540b8c4e8d4088d928eba07b7ce3.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/tables/3eb34b1c1bc192f369fe0f1c1329734beb04540b8c4e8d4088d928eba07b7ce3.jpg)

![62943430ff99072e801116a1eaf0763d53a532e484a915a4324bc0e1158611fd.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/tables/62943430ff99072e801116a1eaf0763d53a532e484a915a4324bc0e1158611fd.jpg)

![839d03f25f3ca912f2e90cdfbafe75391494aa7c5d8f6003b5892f51b6c93c94.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/tables/839d03f25f3ca912f2e90cdfbafe75391494aa7c5d8f6003b5892f51b6c93c94.jpg)

![998b13cbedf9116775e62eda08e0ec68428594899cff0ec1e5ab76effa5c896c.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/tables/998b13cbedf9116775e62eda08e0ec68428594899cff0ec1e5ab76effa5c896c.jpg)

![f4d0d3cf8b05a45c981594eaa344fbcb95552ba61ba86def0c6bda3b66a18f7f.jpg](../icml_results/1877_Stacey_%20Promoting%20Stochastic%20Steepest%20Descent%20via%20Accelerated%20%24_ell_p%24-Smooth%20Nonconvex%20Optimization/tables/f4d0d3cf8b05a45c981594eaa344fbcb95552ba61ba86def0c6bda3b66a18f7f.jpg)

## A Physics-Augmented Deep Learning Framework for Classifying Single Molecule Force Spectroscopy Data


### Images

![0abf357e6c6c155ce9dd9c3b84a764b1387012fd150da580b56b148c803c0b32.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/0abf357e6c6c155ce9dd9c3b84a764b1387012fd150da580b56b148c803c0b32.jpg)

![112caaf62b9cb0b836de7c64108d5243c7a73539bf7c40f5d04a52ebe1ec66cb.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/112caaf62b9cb0b836de7c64108d5243c7a73539bf7c40f5d04a52ebe1ec66cb.jpg)

![1470a7f1562460adc49d537b5d900047f4f2d5cf4bbbfa30117ae87a2cd8113a.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/1470a7f1562460adc49d537b5d900047f4f2d5cf4bbbfa30117ae87a2cd8113a.jpg)

![248e7d8d4094833e20ec50e32ec8d852c1a4c05647b76e8d8c95cbe24bceaf78.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/248e7d8d4094833e20ec50e32ec8d852c1a4c05647b76e8d8c95cbe24bceaf78.jpg)

![29411c155ada4d877ec2780d837075f6e38362a8f040ad70b8214e9dc66853fa.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/29411c155ada4d877ec2780d837075f6e38362a8f040ad70b8214e9dc66853fa.jpg)

![305e96d65f08413e345276d26f873e5fda3b84077bb6111e5d2078180b25257a.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/305e96d65f08413e345276d26f873e5fda3b84077bb6111e5d2078180b25257a.jpg)

![474ee0b75744ec9889eae5c688be1d359cc5b70c4fb006f6c91b978d8810f8bf.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/474ee0b75744ec9889eae5c688be1d359cc5b70c4fb006f6c91b978d8810f8bf.jpg)

![8a702e305cc666b588d96c98ef4ba91b1c5847124aeebbc503630746e5321e0d.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/8a702e305cc666b588d96c98ef4ba91b1c5847124aeebbc503630746e5321e0d.jpg)

![95833cf201ca4eb55a4b795942ea093a6064cb6cd51cdc9f39e3e849aa8055d8.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/95833cf201ca4eb55a4b795942ea093a6064cb6cd51cdc9f39e3e849aa8055d8.jpg)

![a16c15f714cecc575d46073e6d6cc508e1b61178d6e0ebe0227464691f6b2254.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/a16c15f714cecc575d46073e6d6cc508e1b61178d6e0ebe0227464691f6b2254.jpg)

![a584475cd48f485523e2d95d821ce3762352dda7de06e409787ea516e9d3b4e0.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/a584475cd48f485523e2d95d821ce3762352dda7de06e409787ea516e9d3b4e0.jpg)

![acdad7ba9473409900e83738b1b942a58e64d72395d9dc7ae3d791b30c293257.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/acdad7ba9473409900e83738b1b942a58e64d72395d9dc7ae3d791b30c293257.jpg)

![b09c50317c12d0060ea2d49c2e90f50d83284807fdc1b5544e8bd120af06c21e.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/b09c50317c12d0060ea2d49c2e90f50d83284807fdc1b5544e8bd120af06c21e.jpg)

![b4c1961f8e8c3b3a26c21f821b9c77b03c06967c8f1cdd2a5333c69e848d310c.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/b4c1961f8e8c3b3a26c21f821b9c77b03c06967c8f1cdd2a5333c69e848d310c.jpg)

![e91bbb6d890b117df5224457cff79d4f5f1b335d0f3abc0bf61e8de889533f76.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/images/e91bbb6d890b117df5224457cff79d4f5f1b335d0f3abc0bf61e8de889533f76.jpg)

### Tables

![56de4ec984cb870c4f3b7d7feaeb0ad478c283fe4d8499a2f7c60eef07fd0f9f.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/tables/56de4ec984cb870c4f3b7d7feaeb0ad478c283fe4d8499a2f7c60eef07fd0f9f.jpg)

![5c2659688014e4775de9d8df66c60bc431854551b47d5f0d3fbe8af64edf9e92.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/tables/5c2659688014e4775de9d8df66c60bc431854551b47d5f0d3fbe8af64edf9e92.jpg)

![6c807a160327119e6fb02912d767ef33ab842cbaa53155cffc8d09d14c421739.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/tables/6c807a160327119e6fb02912d767ef33ab842cbaa53155cffc8d09d14c421739.jpg)

![823d4bf51ce33d142b126dff515686261cb5c235b1316a4aefea4496378743c0.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/tables/823d4bf51ce33d142b126dff515686261cb5c235b1316a4aefea4496378743c0.jpg)

![8ef04874ee3a5122669828f9a3e911f4a63ede37a26b9021da3bd88bd7fd81e8.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/tables/8ef04874ee3a5122669828f9a3e911f4a63ede37a26b9021da3bd88bd7fd81e8.jpg)

![a68cb199dc19e0a3eb395fb37963415073ce611ee6b1d7b312b0e10daf29f72a.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/tables/a68cb199dc19e0a3eb395fb37963415073ce611ee6b1d7b312b0e10daf29f72a.jpg)

![ff8d25337bcfa364922571d2c603b037d33c28e04137ac80af3bce0bede3894e.jpg](../icml_results/1878_A%20Physics-Augmented%20Deep%20Learning%20Framework%20for%20Classifying%20Single%20Molecule%20Force%20Spectroscopy%20Data/tables/ff8d25337bcfa364922571d2c603b037d33c28e04137ac80af3bce0bede3894e.jpg)

## RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing


### Images

![1f50051fc721dd16f0cb02684c6092fa8dfe304ab2997bc6c834199a617db26a.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/images/1f50051fc721dd16f0cb02684c6092fa8dfe304ab2997bc6c834199a617db26a.jpg)

![681e5aef54824d035727b559289d1b31f1b29a9357ab7938e3bad3695f2aadba.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/images/681e5aef54824d035727b559289d1b31f1b29a9357ab7938e3bad3695f2aadba.jpg)

![824dac55027216a95d6d74521c13e4404d40d562db512cdc30adc39541670951.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/images/824dac55027216a95d6d74521c13e4404d40d562db512cdc30adc39541670951.jpg)

![94e81f5693bc88deb917d9ea5fab5263cd80e5f7f7d0637a6fc4c984f64ab9f6.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/images/94e81f5693bc88deb917d9ea5fab5263cd80e5f7f7d0637a6fc4c984f64ab9f6.jpg)

![ba36a133955c72759a40517d4f4b68ddec007a60aeda7d7a4e4ee81eea980e06.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/images/ba36a133955c72759a40517d4f4b68ddec007a60aeda7d7a4e4ee81eea980e06.jpg)

![ff12b23dd6c2504e722f21b0508dce77c7b6e2cd15ae8493dbcff91c4767f730.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/images/ff12b23dd6c2504e722f21b0508dce77c7b6e2cd15ae8493dbcff91c4767f730.jpg)

### Tables

![19718274f14390889b4f17ca0c2d978d2f1457a41fd5d0bad656adf7d6d5c3ec.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/19718274f14390889b4f17ca0c2d978d2f1457a41fd5d0bad656adf7d6d5c3ec.jpg)

![4053af9482c21ad6f7a1d34e9edeb9549c45bea66de68c876884f7d5ff36e953.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/4053af9482c21ad6f7a1d34e9edeb9549c45bea66de68c876884f7d5ff36e953.jpg)

![6f25f0f9af7fa34d9700ddb2ed1632be36fa6379769e965dc841037c24d83883.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/6f25f0f9af7fa34d9700ddb2ed1632be36fa6379769e965dc841037c24d83883.jpg)

![77c93ee8ed058fe8a59ddea49833c88ce77d5f1434484054dfb5c60be96c6fe2.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/77c93ee8ed058fe8a59ddea49833c88ce77d5f1434484054dfb5c60be96c6fe2.jpg)

![7ea5a1f83b0c76f2a2a291af5a1014d27ece0b91b7cc0fa0a54f96d8b8f40a54.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/7ea5a1f83b0c76f2a2a291af5a1014d27ece0b91b7cc0fa0a54f96d8b8f40a54.jpg)

![82914578bdd155da62c15371b12d683806f0695a05754493768987a7bbbc57b7.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/82914578bdd155da62c15371b12d683806f0695a05754493768987a7bbbc57b7.jpg)

![8906be9144c3a13aec302a0f0fca05290c525f966d64c66e87d2397e2d2dbaab.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/8906be9144c3a13aec302a0f0fca05290c525f966d64c66e87d2397e2d2dbaab.jpg)

![a4f8fe5d6ad5575a4b143006598ad530c74f78a9a426beed087d097bda8523d9.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/a4f8fe5d6ad5575a4b143006598ad530c74f78a9a426beed087d097bda8523d9.jpg)

![cd0e2fd4c4c26258006f5366e3c50c9335e0c8a541b94a2737936ce504734510.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/cd0e2fd4c4c26258006f5366e3c50c9335e0c8a541b94a2737936ce504734510.jpg)

![cdab8281244a227e940cda3b5bb9d26b45207bccf3c787bcc3092627125a16af.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/cdab8281244a227e940cda3b5bb9d26b45207bccf3c787bcc3092627125a16af.jpg)

![e836cd6347fd882472b57b21202bde6d200aeb226d30e13439812b4d04ccafb6.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/e836cd6347fd882472b57b21202bde6d200aeb226d30e13439812b4d04ccafb6.jpg)

![fd69622f794396008613e68b2e9beaedf985b6389b4d236443531e042b397808.jpg](../icml_results/1879_RepoAudit_%20An%20Autonomous%20LLM-Agent%20for%20Repository-Level%20Code%20Auditing/tables/fd69622f794396008613e68b2e9beaedf985b6389b4d236443531e042b397808.jpg)

## Distributionally Robust Policy Learning under Concept Drifts


### Images

![1720ce391e63ad7833a92f1ba45b7fd8adfa0c5293eeadef44634a24df05d9dc.jpg](../icml_results/1880_Distributionally%20Robust%20Policy%20Learning%20under%20Concept%20Drifts/images/1720ce391e63ad7833a92f1ba45b7fd8adfa0c5293eeadef44634a24df05d9dc.jpg)

![c1da3e7c3f1e9422e307db1013c6238bab2cb04f6aff7d93d0cbeba8133d090d.jpg](../icml_results/1880_Distributionally%20Robust%20Policy%20Learning%20under%20Concept%20Drifts/images/c1da3e7c3f1e9422e307db1013c6238bab2cb04f6aff7d93d0cbeba8133d090d.jpg)

### Tables

![0ef492ef0ab024dcd4a77d5389bc55c56fe6e1de4ee0aaaece8911740236daa3.jpg](../icml_results/1880_Distributionally%20Robust%20Policy%20Learning%20under%20Concept%20Drifts/tables/0ef492ef0ab024dcd4a77d5389bc55c56fe6e1de4ee0aaaece8911740236daa3.jpg)

![4b662677ed6a031d0283491487c66294bb8169d15bb84960f5f027a9a8f7a16f.jpg](../icml_results/1880_Distributionally%20Robust%20Policy%20Learning%20under%20Concept%20Drifts/tables/4b662677ed6a031d0283491487c66294bb8169d15bb84960f5f027a9a8f7a16f.jpg)

![aa67c0668a2f5a64b640b715e56e7dc1edb8a8a9d98527cfb2dbce85916979d2.jpg](../icml_results/1880_Distributionally%20Robust%20Policy%20Learning%20under%20Concept%20Drifts/tables/aa67c0668a2f5a64b640b715e56e7dc1edb8a8a9d98527cfb2dbce85916979d2.jpg)

## Mind the Gap: A Practical Attack on GGUF Quantization


### Images

![2acf88c9c77b8983da483158d686708e939cfef3df506de85148c06b7dfd35a6.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/images/2acf88c9c77b8983da483158d686708e939cfef3df506de85148c06b7dfd35a6.jpg)

![465fe7266959da529d7e6c121fc1fd14b4dfca01b892738348134edc766b6813.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/images/465fe7266959da529d7e6c121fc1fd14b4dfca01b892738348134edc766b6813.jpg)

![653d28da24817ace7e6074edb86cb380018ba3d655a69eb90e57aed730d92067.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/images/653d28da24817ace7e6074edb86cb380018ba3d655a69eb90e57aed730d92067.jpg)

![ba6822612aa7f27b1ce84c1430a500a6def0349b956568ddbe56eb6fdc0424fd.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/images/ba6822612aa7f27b1ce84c1430a500a6def0349b956568ddbe56eb6fdc0424fd.jpg)

### Tables

![194f9c94535b5771a1e37365444277efbd103fcc28a133b06e2f01600eb4f6f6.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/194f9c94535b5771a1e37365444277efbd103fcc28a133b06e2f01600eb4f6f6.jpg)

![24c071b5c671b30d2b03b201272fb0321a285eaaf6b4c52133cb2ab0da308280.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/24c071b5c671b30d2b03b201272fb0321a285eaaf6b4c52133cb2ab0da308280.jpg)

![267cad1a0965111f43c54ac0c790be14dc1257353d21403c3778a3e20c890cad.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/267cad1a0965111f43c54ac0c790be14dc1257353d21403c3778a3e20c890cad.jpg)

![373436df092fa643b3368a816109b641e6f48c3ea77bd8ed96d1dc915e1f56ef.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/373436df092fa643b3368a816109b641e6f48c3ea77bd8ed96d1dc915e1f56ef.jpg)

![428e03b00ba867274c895372a1e6029520a2d6ad8f9a1a891f399b28d4062594.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/428e03b00ba867274c895372a1e6029520a2d6ad8f9a1a891f399b28d4062594.jpg)

![5061147992b47403028f2c21199ba0421e553249ae7893c7469c7609ce57a92b.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/5061147992b47403028f2c21199ba0421e553249ae7893c7469c7609ce57a92b.jpg)

![62540f5710052cc890714bb5e2302eb22d9196ad0415f444ef844be579f59800.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/62540f5710052cc890714bb5e2302eb22d9196ad0415f444ef844be579f59800.jpg)

![68ece7f1c94be95b7a409a95a530da9ab9c846c27883865322556c7e1e7f01d8.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/68ece7f1c94be95b7a409a95a530da9ab9c846c27883865322556c7e1e7f01d8.jpg)

![69d10b19048378c5d45f2bcb1f3cf4686772f4fce11bc9a022d5fad0e48a6c7f.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/69d10b19048378c5d45f2bcb1f3cf4686772f4fce11bc9a022d5fad0e48a6c7f.jpg)

![70f210ec2713e83a3da35ad5bfe1240a24514492b426da110efd67e0aaadf608.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/70f210ec2713e83a3da35ad5bfe1240a24514492b426da110efd67e0aaadf608.jpg)

![73b128dc6104df8655f0b967a4dbe5f8c75624523efeb2e2954250583771c83f.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/73b128dc6104df8655f0b967a4dbe5f8c75624523efeb2e2954250583771c83f.jpg)

![8bc2ad3fa3fd55123767d7a79e420d73b3291398cd9462a009deed42669eefb5.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/8bc2ad3fa3fd55123767d7a79e420d73b3291398cd9462a009deed42669eefb5.jpg)

![97a0680d2f21398146dd06e29f226fcb8f63276ade32de220920b84647f9d92b.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/97a0680d2f21398146dd06e29f226fcb8f63276ade32de220920b84647f9d92b.jpg)

![98833e058cafbee1c1426e47489c20cea8eb877aad49f666037300442b5d71fc.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/98833e058cafbee1c1426e47489c20cea8eb877aad49f666037300442b5d71fc.jpg)

![a9a8f667ad93f0e2ac66df0c249db5d00fe609f68c54f13a82f6885f614e63ab.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/a9a8f667ad93f0e2ac66df0c249db5d00fe609f68c54f13a82f6885f614e63ab.jpg)

![acb50c6126c5d9d5c8c72345df7992f0973afe7a935757af2fc65181a4cb1a8d.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/acb50c6126c5d9d5c8c72345df7992f0973afe7a935757af2fc65181a4cb1a8d.jpg)

![aeea6798fe73768403df134166923e44e0601a1a5ceb9193154833b06540263b.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/aeea6798fe73768403df134166923e44e0601a1a5ceb9193154833b06540263b.jpg)

![be406fcac6d662218c92cd77c74f3d0c74f881ed6f0c6b5f663dd5077316d364.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/be406fcac6d662218c92cd77c74f3d0c74f881ed6f0c6b5f663dd5077316d364.jpg)

![dea50cbea7953b58575d5558f15ec81c3ba3a6de6f27aedec37a75e223db7ad5.jpg](../icml_results/1881_Mind%20the%20Gap_%20A%20Practical%20Attack%20on%20GGUF%20Quantization/tables/dea50cbea7953b58575d5558f15ec81c3ba3a6de6f27aedec37a75e223db7ad5.jpg)

## Federated In-Context Learning: Iterative Refinement for Improved Answer Quality


### Images

![02343b8970aa5d7c819f61175c7d2f86b19166de9db79d0898166a82aaf73245.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/02343b8970aa5d7c819f61175c7d2f86b19166de9db79d0898166a82aaf73245.jpg)

![16e55742828c52c847a17b7e15d2325afcb84084c3f6bf4b2adcad05bbdbf4e8.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/16e55742828c52c847a17b7e15d2325afcb84084c3f6bf4b2adcad05bbdbf4e8.jpg)

![176fbe5febe34659c880b57cf61bdc3e37d78844eaffd1f693d331d3009b2a8f.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/176fbe5febe34659c880b57cf61bdc3e37d78844eaffd1f693d331d3009b2a8f.jpg)

![195e118cf5b862f690e9c369135725cb48b010588c69ca0caf7541978092a056.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/195e118cf5b862f690e9c369135725cb48b010588c69ca0caf7541978092a056.jpg)

![44a67373040639cfa1b050ce3a926168a4495a375c3910b2205763e0f294fe1c.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/44a67373040639cfa1b050ce3a926168a4495a375c3910b2205763e0f294fe1c.jpg)

![48b86fce466d35cc4878c8b261ba02af6aa21381be282f15a51ddf9308373d47.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/48b86fce466d35cc4878c8b261ba02af6aa21381be282f15a51ddf9308373d47.jpg)

![4956031d3db13898d37bcec2fa53c0a23150e6c48009ef5b64a2df24e7dc7cbe.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/4956031d3db13898d37bcec2fa53c0a23150e6c48009ef5b64a2df24e7dc7cbe.jpg)

![5eae10a83d3f23fe5037bfa8143b3b9a6dfa29735cd3440e85ac332363a1a8e7.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/5eae10a83d3f23fe5037bfa8143b3b9a6dfa29735cd3440e85ac332363a1a8e7.jpg)

![602a14ff48a42bb3536578dba3649a1bd32ba265590b53a149c63caa92845ac9.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/602a14ff48a42bb3536578dba3649a1bd32ba265590b53a149c63caa92845ac9.jpg)

![8cb072833dd93d2a822122f7a60caef89cfdeb9a4a6c0620542413ca3ea8566f.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/8cb072833dd93d2a822122f7a60caef89cfdeb9a4a6c0620542413ca3ea8566f.jpg)

![9501ced825d2ae5e5a4a8d2b183e18ec69f4a3ebf00715155010b29a618005b3.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/9501ced825d2ae5e5a4a8d2b183e18ec69f4a3ebf00715155010b29a618005b3.jpg)

![db46d30d6aca4b2f14dcc2dc24041396b1686741a21c7d9f5330612481bbdced.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/db46d30d6aca4b2f14dcc2dc24041396b1686741a21c7d9f5330612481bbdced.jpg)

![eb1b8a98471ff68f0f33a224f2c7ddf583b1e983ab2c270c25f264fdde026b58.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/eb1b8a98471ff68f0f33a224f2c7ddf583b1e983ab2c270c25f264fdde026b58.jpg)

![f9bb23d6efa352d196ca58ae63e033388529e7ce4961717f9243e4508c3f5618.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/images/f9bb23d6efa352d196ca58ae63e033388529e7ce4961717f9243e4508c3f5618.jpg)

### Tables

![1094a1cd3b984d5ae99ca10142a2d86bcc5b8c19aab2588102e3accd54658bcf.jpg](../icml_results/1882_Federated%20In-Context%20Learning_%20Iterative%20Refinement%20for%20Improved%20Answer%20Quality/tables/1094a1cd3b984d5ae99ca10142a2d86bcc5b8c19aab2588102e3accd54658bcf.jpg)

## Unlocking the Power of SAM 2 for Few-Shot Segmentation


### Images

![1f9fcb40e12d4f23c3b19a1faf94df559230822ceed0fe327085925a4d230c1c.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/1f9fcb40e12d4f23c3b19a1faf94df559230822ceed0fe327085925a4d230c1c.jpg)

![2202cb32c764f19a9e0ff114c893821b8524a574b88264d46a3d9b5151009db8.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/2202cb32c764f19a9e0ff114c893821b8524a574b88264d46a3d9b5151009db8.jpg)

![3041eacc0edddb3576a5150248cde4a28e4a67118b7ec2a4b39fbc603c90e6c0.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/3041eacc0edddb3576a5150248cde4a28e4a67118b7ec2a4b39fbc603c90e6c0.jpg)

![7072f334f2495817e103b7df6f3d393c0ad6916c0c734711178668349826a2ac.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/7072f334f2495817e103b7df6f3d393c0ad6916c0c734711178668349826a2ac.jpg)

![86cf3733dc40441f0b11efcd83aa13f163f55b56c9ff936b404b00a25afbd83f.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/86cf3733dc40441f0b11efcd83aa13f163f55b56c9ff936b404b00a25afbd83f.jpg)

![b18fe2ce1709386f5a100522d1785d03e57fe204f06e1f9268364331c6e91d2d.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/b18fe2ce1709386f5a100522d1785d03e57fe204f06e1f9268364331c6e91d2d.jpg)

![c6b69f776b7b9968771594872f48abf03e5c8e4268b115a7ca4c5396272a7089.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/c6b69f776b7b9968771594872f48abf03e5c8e4268b115a7ca4c5396272a7089.jpg)

![ceeccfb2545d90ddafc19274e1a6818803a2ed4a56a36254748f2749e2b53abc.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/ceeccfb2545d90ddafc19274e1a6818803a2ed4a56a36254748f2749e2b53abc.jpg)

![d51917d55d21dfed70db8130e5ca2e98a4576d2235a899649420bad76244b21e.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/d51917d55d21dfed70db8130e5ca2e98a4576d2235a899649420bad76244b21e.jpg)

![ed0d35d307366a821ec86655cb0ffa909065ce7d4fc8ab842e5b953681167276.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/ed0d35d307366a821ec86655cb0ffa909065ce7d4fc8ab842e5b953681167276.jpg)

![eff398af356934ab7625c44be578290faf6d19cca44d4c4da7638dc0c37ebc7c.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/images/eff398af356934ab7625c44be578290faf6d19cca44d4c4da7638dc0c37ebc7c.jpg)

### Tables

![1b25ed4d0c100116c37a398f7cd182f0c13e01b6420ab66481e85cae8b5929a4.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/1b25ed4d0c100116c37a398f7cd182f0c13e01b6420ab66481e85cae8b5929a4.jpg)

![1c1983dfb552d5c911fc32f21b5258e19b5856b0481f9a1928b1652b9039f60c.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/1c1983dfb552d5c911fc32f21b5258e19b5856b0481f9a1928b1652b9039f60c.jpg)

![482f43abe1b3c7062f1d813dfb325f93a3c3e6cd0261d0ab6a9fab547fc69320.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/482f43abe1b3c7062f1d813dfb325f93a3c3e6cd0261d0ab6a9fab547fc69320.jpg)

![930b9adc75f6331bb7b7bf4b4fc3b4012e75162277fd63ae9643b5860424c079.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/930b9adc75f6331bb7b7bf4b4fc3b4012e75162277fd63ae9643b5860424c079.jpg)

![a128c1d4905d6d8abb6d8b09b69fc2298455861b02354614da692c15d9a21ff6.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/a128c1d4905d6d8abb6d8b09b69fc2298455861b02354614da692c15d9a21ff6.jpg)

![b303493605252d1a66b8cd73ecd3887adeb351db5ba506faf626c9d5fa0503fd.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/b303493605252d1a66b8cd73ecd3887adeb351db5ba506faf626c9d5fa0503fd.jpg)

![c0513ba9cde8e5427f6badd3bf8d83b5589f2016d31fa53edfc851c98eb1b8ae.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/c0513ba9cde8e5427f6badd3bf8d83b5589f2016d31fa53edfc851c98eb1b8ae.jpg)

![c18a2b971bce155944def5169fe7788f56b03952e377fde87922012e8231bffe.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/c18a2b971bce155944def5169fe7788f56b03952e377fde87922012e8231bffe.jpg)

![ebc6bfae6611837b02b45c4c68079afb25999361bb13503a3ad7ed418bc803d5.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/ebc6bfae6611837b02b45c4c68079afb25999361bb13503a3ad7ed418bc803d5.jpg)

![f4b12c3359742284d224feb8dbe46ed273549fb152d53f824ffe16c28f6dc3ef.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/f4b12c3359742284d224feb8dbe46ed273549fb152d53f824ffe16c28f6dc3ef.jpg)

![f4d90cd5841d1c9b34f4e2258d3a9841d3bb8ffcf8fe26a75cde687afc8e4241.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/f4d90cd5841d1c9b34f4e2258d3a9841d3bb8ffcf8fe26a75cde687afc8e4241.jpg)

![f766bcacd6f51083f2e065781c39fd554bc33a538dfb2832b7bea1b2babc6ffd.jpg](../icml_results/1883_Unlocking%20the%20Power%20of%20SAM%202%20for%20Few-Shot%20Segmentation/tables/f766bcacd6f51083f2e065781c39fd554bc33a538dfb2832b7bea1b2babc6ffd.jpg)

## Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations


### Images

![039036aecd574179a124a4eace3ec2736c7b2d904a3003cfdc39f5fcc9a241b1.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/039036aecd574179a124a4eace3ec2736c7b2d904a3003cfdc39f5fcc9a241b1.jpg)

![0d71e4cb4c82f037f9b3ed6278a7551215349084d6cef25361deeb7006e259dd.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/0d71e4cb4c82f037f9b3ed6278a7551215349084d6cef25361deeb7006e259dd.jpg)

![1b11d84bed2dd5da349409e6c2679bf365a568f2893acc89ba6047081e5eec9f.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/1b11d84bed2dd5da349409e6c2679bf365a568f2893acc89ba6047081e5eec9f.jpg)

![1ff12f52fabb4acab0fa21654ad944ea15f7d2af7f11730f32cc13cdbc1498fe.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/1ff12f52fabb4acab0fa21654ad944ea15f7d2af7f11730f32cc13cdbc1498fe.jpg)

![26d7671f503cbea80ed452d5937d7cb41d707bdf1486ed8711a92a31b3f84a1c.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/26d7671f503cbea80ed452d5937d7cb41d707bdf1486ed8711a92a31b3f84a1c.jpg)

![2a6940ff3563eb1942945a2583696798739d8c85573efcce2e02d2fd239f4651.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/2a6940ff3563eb1942945a2583696798739d8c85573efcce2e02d2fd239f4651.jpg)

![2c56cb1229f71165a4e219be9ce0fd6f4de77efda9fe969f63629fb3a1ce389a.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/2c56cb1229f71165a4e219be9ce0fd6f4de77efda9fe969f63629fb3a1ce389a.jpg)

![313ba1fd718455e3b275e82e4053679425b33c7c9f0e54349302da6f64c25443.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/313ba1fd718455e3b275e82e4053679425b33c7c9f0e54349302da6f64c25443.jpg)

![329c06be8a9e1852a211429a1c6b1ca491d8a2656e4908b8cebaaa06332015f5.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/329c06be8a9e1852a211429a1c6b1ca491d8a2656e4908b8cebaaa06332015f5.jpg)

![357480f3d1b68a977a8ad63e698a230db1f9eba9a455ce05a73ea6ee777858b9.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/357480f3d1b68a977a8ad63e698a230db1f9eba9a455ce05a73ea6ee777858b9.jpg)

![370e3f0d38b24a802f91b66649417c5c73f73e7416c502fc8045b84bee8e38d5.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/370e3f0d38b24a802f91b66649417c5c73f73e7416c502fc8045b84bee8e38d5.jpg)

![429188d6ebeeba822d9cf62520e471756256503bebb7a41218a7a83fc6b4e9a3.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/429188d6ebeeba822d9cf62520e471756256503bebb7a41218a7a83fc6b4e9a3.jpg)

![456adcb43c05c269890d3f90ed9b30c948be4124a0be99d00d8d1ad0fc64defd.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/456adcb43c05c269890d3f90ed9b30c948be4124a0be99d00d8d1ad0fc64defd.jpg)

![5adb83b8a96316730549696fec23f43899d3ee409a9c7b80f9bc963ce2dde7bc.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/5adb83b8a96316730549696fec23f43899d3ee409a9c7b80f9bc963ce2dde7bc.jpg)

![5c5d5e5b203c888bad5a3b07fe5c97837718ecfd2f48376baf8475c57d3104a9.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/5c5d5e5b203c888bad5a3b07fe5c97837718ecfd2f48376baf8475c57d3104a9.jpg)

![5d1d96540416ae36e29737a8cd6e7b35a08a603c1caceb1dc922470609277906.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/5d1d96540416ae36e29737a8cd6e7b35a08a603c1caceb1dc922470609277906.jpg)

![5ef01765f81d9d703d0b7048239cf75995883e8297571c0239481f367d7491a3.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/5ef01765f81d9d703d0b7048239cf75995883e8297571c0239481f367d7491a3.jpg)

![62c918fde78985e11da0c7a42fc134562fc4d016dde40b3e4649aee71a8ccd48.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/62c918fde78985e11da0c7a42fc134562fc4d016dde40b3e4649aee71a8ccd48.jpg)

![6845f0cb3c4a7cd95158a3cf1902ed54aa09eea52f4575ebe62dd26f6782dd63.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/6845f0cb3c4a7cd95158a3cf1902ed54aa09eea52f4575ebe62dd26f6782dd63.jpg)

![692b6d9b0133a9d66ace83be6a0f69fd54176fbb527b854dde9489a2ce98456d.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/692b6d9b0133a9d66ace83be6a0f69fd54176fbb527b854dde9489a2ce98456d.jpg)

![6a690954ec14465d2c981c66c2c61b690ee42eb57a1b2bfa84c24b8f1ba19e88.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/6a690954ec14465d2c981c66c2c61b690ee42eb57a1b2bfa84c24b8f1ba19e88.jpg)

![81dd8703d11f1285aea86e486d7ac007a2f9f15c82f93ed2b469f5ceff46e24f.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/81dd8703d11f1285aea86e486d7ac007a2f9f15c82f93ed2b469f5ceff46e24f.jpg)

![8699f87bbf8c3d440bcd7cdc8d6564897e20b7ff565b87fe04d490c9461e64b8.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/8699f87bbf8c3d440bcd7cdc8d6564897e20b7ff565b87fe04d490c9461e64b8.jpg)

![87cdcc9307a0838e7b0e28d5f3279a6fdceb5cc5c811882e8537f46ad79e6810.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/87cdcc9307a0838e7b0e28d5f3279a6fdceb5cc5c811882e8537f46ad79e6810.jpg)

![92f2298f1f75bb1e9c8829f253bf2b96f19c75a71f4ef1b2990991617486e3a9.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/92f2298f1f75bb1e9c8829f253bf2b96f19c75a71f4ef1b2990991617486e3a9.jpg)

![a0c45ea5640f051a13d14b8a5b26a42da907186b1ca697a51a0da0f0b5d9293d.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/a0c45ea5640f051a13d14b8a5b26a42da907186b1ca697a51a0da0f0b5d9293d.jpg)

![a6758e9bc3ca68cbb1e2ee23a0932390daaee1726e7436b9ec9e497656cb593c.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/a6758e9bc3ca68cbb1e2ee23a0932390daaee1726e7436b9ec9e497656cb593c.jpg)

![a7bfd469a368ce1164fbbf49b375be2d266069dab94d8444a2439a68e080d39f.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/a7bfd469a368ce1164fbbf49b375be2d266069dab94d8444a2439a68e080d39f.jpg)

![af37523d0fba5b8b3b60817f6b370fa1ee3a1b82fd2a1fc798b0e39e83a04785.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/af37523d0fba5b8b3b60817f6b370fa1ee3a1b82fd2a1fc798b0e39e83a04785.jpg)

![b096b871994bf306736fe7f9de28129b53754bb305c16c677592ad2991d455bd.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/b096b871994bf306736fe7f9de28129b53754bb305c16c677592ad2991d455bd.jpg)

![b3cc1833355a6b0580e95e185be0ff4fe676e71acabf9c521f039dde9669d4c0.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/b3cc1833355a6b0580e95e185be0ff4fe676e71acabf9c521f039dde9669d4c0.jpg)

![b3f6c5e15158b64b3889bf80fbfef47a5c15c0821ab7d3dd7c3fd499553bc87c.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/b3f6c5e15158b64b3889bf80fbfef47a5c15c0821ab7d3dd7c3fd499553bc87c.jpg)

![b8d2636f24e4d80c1c06a5e35eba6e19509f75188c4b9fc3c6dbfe4c9f430e88.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/b8d2636f24e4d80c1c06a5e35eba6e19509f75188c4b9fc3c6dbfe4c9f430e88.jpg)

![bd4c4271586fa06a9605ce8e786f70a2dc4b6fc3311b36e2ad8642ff616df51e.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/bd4c4271586fa06a9605ce8e786f70a2dc4b6fc3311b36e2ad8642ff616df51e.jpg)

![c0b7526a07cfddf36b61ad343b8544d60920ab174d6384274fe13d9350009484.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/c0b7526a07cfddf36b61ad343b8544d60920ab174d6384274fe13d9350009484.jpg)

![c1d53321950cbb89f354d1f1fb1938433aa5765d7d0b19fb4301e6e87ef8ce15.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/c1d53321950cbb89f354d1f1fb1938433aa5765d7d0b19fb4301e6e87ef8ce15.jpg)

![c74ca257e452b8ab951b08f9580ac56558793315afddfe104f9541efdab04fee.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/c74ca257e452b8ab951b08f9580ac56558793315afddfe104f9541efdab04fee.jpg)

![cb58ad90bf8bd9d563c15a1e2a4263c487b31169e73414f7c197a0bfd8963f8c.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/cb58ad90bf8bd9d563c15a1e2a4263c487b31169e73414f7c197a0bfd8963f8c.jpg)

![ccb3176fe99c3871d2d1ec88ee1e4bfff8a739124593f39658abdc0bf7fce9cf.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/ccb3176fe99c3871d2d1ec88ee1e4bfff8a739124593f39658abdc0bf7fce9cf.jpg)

![ccc3897ecefe1a84f392152ccfd7dceb2291081057d907014857f593984ca1fc.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/ccc3897ecefe1a84f392152ccfd7dceb2291081057d907014857f593984ca1fc.jpg)

![dc4caa72698e2b891fe9ef6c692b3cdb6c0863fa84cbd3d1f6ceb5e50722a686.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/dc4caa72698e2b891fe9ef6c692b3cdb6c0863fa84cbd3d1f6ceb5e50722a686.jpg)

![df576e9628726e6d3a961df577b4a37b33c6889097e44f6a04a8f71f8d70a282.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/df576e9628726e6d3a961df577b4a37b33c6889097e44f6a04a8f71f8d70a282.jpg)

![e9383ea65b77473388e510543ace8ec4c9901179b441539c79122ad35e0eec26.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/e9383ea65b77473388e510543ace8ec4c9901179b441539c79122ad35e0eec26.jpg)

![e9d928630be93d72478aaed0ece4e69ec6f0e9841738089379b08e4a023aad39.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/e9d928630be93d72478aaed0ece4e69ec6f0e9841738089379b08e4a023aad39.jpg)

![ea9c66b5adeb5b6116a7e211033db18216616ec3ee13ac4c692ddf6ab787b24d.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/ea9c66b5adeb5b6116a7e211033db18216616ec3ee13ac4c692ddf6ab787b24d.jpg)

![efca015316a1359415e0bfcaab1bb243fd507c7a9e79dfb3188bcf39e9830bbd.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/efca015316a1359415e0bfcaab1bb243fd507c7a9e79dfb3188bcf39e9830bbd.jpg)

![f543b0acf8a6a21370843bd54efe6dc94f2e5efb6573f4d4283adb8a70332ed6.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/images/f543b0acf8a6a21370843bd54efe6dc94f2e5efb6573f4d4283adb8a70332ed6.jpg)

### Tables

![0058270cffd7de2deedb2ae40423f0dbda47509782184dcca80458cb4493c65e.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/0058270cffd7de2deedb2ae40423f0dbda47509782184dcca80458cb4493c65e.jpg)

![0347b18ce42664b0841bdd479915998ffff9aae9208b83186607277c0d9354f4.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/0347b18ce42664b0841bdd479915998ffff9aae9208b83186607277c0d9354f4.jpg)

![081e5543b37ba25dfa302b67d5ce3a24d56a98665164a2294ef158efb5d9f071.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/081e5543b37ba25dfa302b67d5ce3a24d56a98665164a2294ef158efb5d9f071.jpg)

![0fabfe446d0f3fc52a961893ce8a1be9e2a9556b4f32b376c2972cef4e1e2ec7.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/0fabfe446d0f3fc52a961893ce8a1be9e2a9556b4f32b376c2972cef4e1e2ec7.jpg)

![10522563e6ab0b313c53b50fafc80131dff3c96bf529a7121896d647565db98c.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/10522563e6ab0b313c53b50fafc80131dff3c96bf529a7121896d647565db98c.jpg)

![19dfd25fc8703f904ece27c2cd171a879404b57fd520a4bf8279ff355f99e4b9.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/19dfd25fc8703f904ece27c2cd171a879404b57fd520a4bf8279ff355f99e4b9.jpg)

![25f9f83fbeedc07719911dde80613865e1e62ab244a6c67043a010078e483379.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/25f9f83fbeedc07719911dde80613865e1e62ab244a6c67043a010078e483379.jpg)

![29a0ee35f5fca595132bd56467f435f4c77c9ef8473ccb933eaa577ca3f492a4.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/29a0ee35f5fca595132bd56467f435f4c77c9ef8473ccb933eaa577ca3f492a4.jpg)

![30b8fbf1cf228e0aa9b5cd641334fea6d8289de8feeda5564493cae296eeac95.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/30b8fbf1cf228e0aa9b5cd641334fea6d8289de8feeda5564493cae296eeac95.jpg)

![33c37d6d28ddc6b3c1aa5cd864a09c9010486dfbb21a7534c19193400720a6d5.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/33c37d6d28ddc6b3c1aa5cd864a09c9010486dfbb21a7534c19193400720a6d5.jpg)

![3f715b68a4fbe43cdfc93eb9b6b25b2b88f241004a07c7533a677d2a689c0ab7.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/3f715b68a4fbe43cdfc93eb9b6b25b2b88f241004a07c7533a677d2a689c0ab7.jpg)

![40ac9a6c54d48c6e7f5ebf2017c503959dbd5bc11aa984b507ad19c821e0d830.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/40ac9a6c54d48c6e7f5ebf2017c503959dbd5bc11aa984b507ad19c821e0d830.jpg)

![45645da8d30caad77a9ba03fb5a122a4691012f8b4733fccddbd5b2f475dc5cd.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/45645da8d30caad77a9ba03fb5a122a4691012f8b4733fccddbd5b2f475dc5cd.jpg)

![575039026634c0de9dbd22b183c5ded426aa18353ec29b0c12b5438e86c81489.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/575039026634c0de9dbd22b183c5ded426aa18353ec29b0c12b5438e86c81489.jpg)

![5bb1391ea8b9939db072b360f4ee564a1f95bf78f4c10d7bdbc4276817532181.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/5bb1391ea8b9939db072b360f4ee564a1f95bf78f4c10d7bdbc4276817532181.jpg)

![6295eb35471042d46a68939c468c8170457a81f90cbecac7d7dcab145676b5fe.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/6295eb35471042d46a68939c468c8170457a81f90cbecac7d7dcab145676b5fe.jpg)

![63d076ab4ea0d9db0a730ffe1d9cc323e3623592aad88d1842d1afc8a84e8a5e.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/63d076ab4ea0d9db0a730ffe1d9cc323e3623592aad88d1842d1afc8a84e8a5e.jpg)

![6d242da74606ce8e2730f65751cf43c3d530baa3c94e93890e1218c41f758fa8.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/6d242da74606ce8e2730f65751cf43c3d530baa3c94e93890e1218c41f758fa8.jpg)

![7fb4645d8aa2a9165caa2bff1fa28e7601edeca3835e8eea241b8e13595fd231.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/7fb4645d8aa2a9165caa2bff1fa28e7601edeca3835e8eea241b8e13595fd231.jpg)

![8e96f3bf00afcc57e99474dea5655dce7ade4dc4ae4b15de480ea8038bd458ff.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/8e96f3bf00afcc57e99474dea5655dce7ade4dc4ae4b15de480ea8038bd458ff.jpg)

![9552bd5c355b59b66a07a60595e16057c7c011c385d7729ec0cc74c900c3655a.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/9552bd5c355b59b66a07a60595e16057c7c011c385d7729ec0cc74c900c3655a.jpg)

![9610f6665c921ad84f2b4b5e6955d943c6a8593d9d8fcab01b666f0fbe2b597e.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/9610f6665c921ad84f2b4b5e6955d943c6a8593d9d8fcab01b666f0fbe2b597e.jpg)

![98d3e51cbdb8d506b7b10c12f7c4ca7127dc1ec365ed7850a27c473bedf420f1.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/98d3e51cbdb8d506b7b10c12f7c4ca7127dc1ec365ed7850a27c473bedf420f1.jpg)

![9ca189318753a6d5e6788d58b4c3137c837b994efd9aadd383faad0e42ece78c.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/9ca189318753a6d5e6788d58b4c3137c837b994efd9aadd383faad0e42ece78c.jpg)

![a31de233a9512bfe2823467f3a2d1ac907f62b01270e85f803314b50e78a43e8.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/a31de233a9512bfe2823467f3a2d1ac907f62b01270e85f803314b50e78a43e8.jpg)

![a3c57ff884e762c22b7ebe69a63e39364a810c11bec60f3ce9235c6026699e31.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/a3c57ff884e762c22b7ebe69a63e39364a810c11bec60f3ce9235c6026699e31.jpg)

![b0ab43fe3217e65988b263d92e63a05e9c80a193896ff77bebe20c50d0fb111b.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/b0ab43fe3217e65988b263d92e63a05e9c80a193896ff77bebe20c50d0fb111b.jpg)

![be79188d2b6cbc388d6e4750543789c4886f683704648829900475818d2c4557.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/be79188d2b6cbc388d6e4750543789c4886f683704648829900475818d2c4557.jpg)

![ce12e1e061e198692dec4e7219d03846af7e504e0888049a3cffab5dd80e5be7.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/ce12e1e061e198692dec4e7219d03846af7e504e0888049a3cffab5dd80e5be7.jpg)

![d75faa1674b4e0416fdf8bb97fa0fd4263abf13400de20d188545cec55dc4acb.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/d75faa1674b4e0416fdf8bb97fa0fd4263abf13400de20d188545cec55dc4acb.jpg)

![d928dca500621102f5aab224dd4eeaa3c0652ba42cc968e25c87f36560b7fa03.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/d928dca500621102f5aab224dd4eeaa3c0652ba42cc968e25c87f36560b7fa03.jpg)

![ddef80d7a8f79c0ab9d2e628936ee4bf05b8c2dc4a48d537d7af200ab3a81bb1.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/ddef80d7a8f79c0ab9d2e628936ee4bf05b8c2dc4a48d537d7af200ab3a81bb1.jpg)

![e133eca5a632039a581d347a5ae5c03a6f53ecd42e2bf51135d9b11788c02f85.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/e133eca5a632039a581d347a5ae5c03a6f53ecd42e2bf51135d9b11788c02f85.jpg)

![e9386e8779ec46cb20d2c9126ae34cfbd1bc9fbfcc3647d5939a496cb47e8f55.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/e9386e8779ec46cb20d2c9126ae34cfbd1bc9fbfcc3647d5939a496cb47e8f55.jpg)

![e996499608ee3605ac1fbf6f99572098b8756dd1933c6af6f784af681b0332c2.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/e996499608ee3605ac1fbf6f99572098b8756dd1933c6af6f784af681b0332c2.jpg)

![edbeb4abe7a07dbea42a03de00dab5e629298b5a1d7bba971cabb8a6f9253507.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/edbeb4abe7a07dbea42a03de00dab5e629298b5a1d7bba971cabb8a6f9253507.jpg)

![effff08537e2a3101cfde8236efc105a49c468ced9fbfc22f13a242cf583849e.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/effff08537e2a3101cfde8236efc105a49c468ced9fbfc22f13a242cf583849e.jpg)

![f402cf5d5c96cec4aa454becd393af89cd5742fb8465bbdcdd127413a9d5f78e.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/f402cf5d5c96cec4aa454becd393af89cd5742fb8465bbdcdd127413a9d5f78e.jpg)

![f8bcecef8c550de2d2a6a2121a48c520465d942df0caa6769373b7f244e71edf.jpg](../icml_results/1884_Jacobian%20Sparse%20Autoencoders_%20Sparsify%20Computations%2C%20Not%20Just%20Activations/tables/f8bcecef8c550de2d2a6a2121a48c520465d942df0caa6769373b7f244e71edf.jpg)

## Privacy Amplification Through Synthetic Data: Insights from Linear Regression


### Images

![8c95d45a7d5178ea7515d45f2323baa2327fbdb2ec24fb9b5704ff935ebd8a29.jpg](../icml_results/1885_Privacy%20Amplification%20Through%20Synthetic%20Data_%20Insights%20from%20Linear%20Regression/images/8c95d45a7d5178ea7515d45f2323baa2327fbdb2ec24fb9b5704ff935ebd8a29.jpg)

![a78ee4f97fdbac3269f15c8714d01f2c07648741d8807262d9ea903448c9035a.jpg](../icml_results/1885_Privacy%20Amplification%20Through%20Synthetic%20Data_%20Insights%20from%20Linear%20Regression/images/a78ee4f97fdbac3269f15c8714d01f2c07648741d8807262d9ea903448c9035a.jpg)

![cdc730302b56d84c3c57c995238debf90a8c4809693abdb123324e8c0c93de4a.jpg](../icml_results/1885_Privacy%20Amplification%20Through%20Synthetic%20Data_%20Insights%20from%20Linear%20Regression/images/cdc730302b56d84c3c57c995238debf90a8c4809693abdb123324e8c0c93de4a.jpg)

## Scaling Large Motion Models with Million-Level Human Motions


### Images

![128565c8afa0180cef7cf8055e578a3f7c74527a79f5bf4782dcf1a51ceebdbd.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/128565c8afa0180cef7cf8055e578a3f7c74527a79f5bf4782dcf1a51ceebdbd.jpg)

![1f565c40fafd4503696ef4f2bfb3118e0f4c7732db071053502ace2f6deb17ef.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/1f565c40fafd4503696ef4f2bfb3118e0f4c7732db071053502ace2f6deb17ef.jpg)

![6183d04aff4a48a844ab48828571a1bab2f28ec0083f33112c967726c73e8922.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/6183d04aff4a48a844ab48828571a1bab2f28ec0083f33112c967726c73e8922.jpg)

![73939c14555d0fceb93f606f19be38c3578f8a8276f99291a85227a540a44c58.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/73939c14555d0fceb93f606f19be38c3578f8a8276f99291a85227a540a44c58.jpg)

![7fb0c3631b57b36211b72f581d9b9b3ad2f506d80b62d13c230df9f2d8cc7830.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/7fb0c3631b57b36211b72f581d9b9b3ad2f506d80b62d13c230df9f2d8cc7830.jpg)

![8535bfe3aaee886325f5f57a6b1a565227106148a487afd8b1450d05f609af02.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/8535bfe3aaee886325f5f57a6b1a565227106148a487afd8b1450d05f609af02.jpg)

![8f290fa8c21670622969c4c9c19580fdec1030dbbb87e8582fba1df9445f0a9b.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/8f290fa8c21670622969c4c9c19580fdec1030dbbb87e8582fba1df9445f0a9b.jpg)

![92ab4e34be02c4e34112bf3397be7f62b2699d1f5f4e5f754db5bf4f71aa5f93.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/92ab4e34be02c4e34112bf3397be7f62b2699d1f5f4e5f754db5bf4f71aa5f93.jpg)

![99a4c4342cf22dff3f91112b279c9fa7b5ee502bcf89194642096d695a9d6ce3.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/99a4c4342cf22dff3f91112b279c9fa7b5ee502bcf89194642096d695a9d6ce3.jpg)

![9d5cde10ca892da9b5d52452c0d23da9f59b78512e442110b6c6751e01393c63.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/9d5cde10ca892da9b5d52452c0d23da9f59b78512e442110b6c6751e01393c63.jpg)

![a70e16759567129b3e363ae6883fb92e7a9c3321cebd2354e39b523b6f6ce38c.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/a70e16759567129b3e363ae6883fb92e7a9c3321cebd2354e39b523b6f6ce38c.jpg)

![ae796a635e0cf6b6da849eea63e57e462bf654e2bb54a4609ad1926650a7e18a.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/ae796a635e0cf6b6da849eea63e57e462bf654e2bb54a4609ad1926650a7e18a.jpg)

![e0ab136a9b1db8aa3c2ecbe784d1f2518e293756f73f0af78b828f3ab65316f8.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/images/e0ab136a9b1db8aa3c2ecbe784d1f2518e293756f73f0af78b828f3ab65316f8.jpg)

### Tables

![18827f2752f8c94f2dad06afb2fb0a35197eb918408110c395a40a61ab31c3ec.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/18827f2752f8c94f2dad06afb2fb0a35197eb918408110c395a40a61ab31c3ec.jpg)

![1a3ebd5e8cf4e3f15f72291f2848d70a9f8c1f63a6e38fc287ed43b886e13eba.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/1a3ebd5e8cf4e3f15f72291f2848d70a9f8c1f63a6e38fc287ed43b886e13eba.jpg)

![2762835e3de51b6fc762927d69cc2d35187fea25bd3f7b24817f8cbb7758143e.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/2762835e3de51b6fc762927d69cc2d35187fea25bd3f7b24817f8cbb7758143e.jpg)

![2894ae1bf8242b909557e45c0cfe6010073b8a4313797a2c1fa15edc3f80bdf3.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/2894ae1bf8242b909557e45c0cfe6010073b8a4313797a2c1fa15edc3f80bdf3.jpg)

![425d59e9115b07862134171c70a0f9c264bcfc3aa5db07830208fdd94065a5c6.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/425d59e9115b07862134171c70a0f9c264bcfc3aa5db07830208fdd94065a5c6.jpg)

![54c279f61037403a304c6e758a40428e7bcdd68f38d4f9dd3127813d17d1ee9c.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/54c279f61037403a304c6e758a40428e7bcdd68f38d4f9dd3127813d17d1ee9c.jpg)

![58933acd1f0b3df9f893c2c80c4a981cc4fb8ced1947aa5cfd525646fe331ff9.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/58933acd1f0b3df9f893c2c80c4a981cc4fb8ced1947aa5cfd525646fe331ff9.jpg)

![5d32ea2c3015612a4086a6e08f65b0157a4bc9d50c6eb20b78af7dc1c27210cd.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/5d32ea2c3015612a4086a6e08f65b0157a4bc9d50c6eb20b78af7dc1c27210cd.jpg)

![8efdf3603bc483ed47287071b0d0e9c56c2c9933451454cb2c2bdaa3c92f2e6f.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/8efdf3603bc483ed47287071b0d0e9c56c2c9933451454cb2c2bdaa3c92f2e6f.jpg)

![90e3375432fd240d44db2648f923bb7a91308f59a1bd7294fb5fe5659b0f7b94.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/90e3375432fd240d44db2648f923bb7a91308f59a1bd7294fb5fe5659b0f7b94.jpg)

![9e40fbdf472efcccce2029c3900f26eef86cf4959455b4ed0202786512f61d5e.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/9e40fbdf472efcccce2029c3900f26eef86cf4959455b4ed0202786512f61d5e.jpg)

![da3cc9d76083cd021308b98b4ce47dbeba7171e71aafc03782f813f6aa460ea7.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/da3cc9d76083cd021308b98b4ce47dbeba7171e71aafc03782f813f6aa460ea7.jpg)

![e3ab47d9197f4d2ae7446b4296cd8e3885dafe832073ae254167f5b259b9c2c8.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/e3ab47d9197f4d2ae7446b4296cd8e3885dafe832073ae254167f5b259b9c2c8.jpg)

![e5acf73775ed2c513db142e33bdaad0fc589567ddebe3941c0ebc0978105577e.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/e5acf73775ed2c513db142e33bdaad0fc589567ddebe3941c0ebc0978105577e.jpg)

![f46b302d7d36a5647e6ad42f6709e51cdeca804f23e9f12361d124f00f9313a4.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/f46b302d7d36a5647e6ad42f6709e51cdeca804f23e9f12361d124f00f9313a4.jpg)

![f7419be417949e9f670cf40d40a6e36d1e7672b41d51c69cc9cdb47a1f1b27e3.jpg](../icml_results/1886_Scaling%20Large%20Motion%20Models%20with%20Million-Level%20Human%20Motions/tables/f7419be417949e9f670cf40d40a6e36d1e7672b41d51c69cc9cdb47a1f1b27e3.jpg)

## Locality Preserving Markovian Transition for Instance Retrieval


### Images

![1999f5f36d42ff5d133312c6b847170dce61d9b77c76e8fdcd47b275738a6930.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/images/1999f5f36d42ff5d133312c6b847170dce61d9b77c76e8fdcd47b275738a6930.jpg)

![497d9e3df3702e51604b354ff9a74d8700d4e239f84004f1708d8e2fff520bfb.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/images/497d9e3df3702e51604b354ff9a74d8700d4e239f84004f1708d8e2fff520bfb.jpg)

![9dc6b2c18a7a60f12b60f680bcb90edc730915262e91e2f6077550fba1ec686f.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/images/9dc6b2c18a7a60f12b60f680bcb90edc730915262e91e2f6077550fba1ec686f.jpg)

![ad70060c7666d92b74dca99266463dcd67fbd551abb05423a24c3c9db6357a5d.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/images/ad70060c7666d92b74dca99266463dcd67fbd551abb05423a24c3c9db6357a5d.jpg)

![b8b62c16c83724cf80c6948b5f7d90a92b83e47a81d78c91594af162547a1028.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/images/b8b62c16c83724cf80c6948b5f7d90a92b83e47a81d78c91594af162547a1028.jpg)

![bc81e53633b3fee4167bdf7868f8096c041703e7f20ee4b1612851cdb60457f0.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/images/bc81e53633b3fee4167bdf7868f8096c041703e7f20ee4b1612851cdb60457f0.jpg)

![cebb702a9da98782cf6ac5ad3d17c0e76adbbfc8065f371133925298440e8759.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/images/cebb702a9da98782cf6ac5ad3d17c0e76adbbfc8065f371133925298440e8759.jpg)

### Tables

![0202a5e8de38ba96b0d01debbbff7e649c754f6b07d5fe93279c6400194efc09.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/0202a5e8de38ba96b0d01debbbff7e649c754f6b07d5fe93279c6400194efc09.jpg)

![195141ca0f08cce7053f4c22aedafe8452ac91fd47bffb356828c56264a46a52.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/195141ca0f08cce7053f4c22aedafe8452ac91fd47bffb356828c56264a46a52.jpg)

![30ff63ffffe7334454ad1ee94ce212328ed60179f78366b4a6421eafbc5c6640.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/30ff63ffffe7334454ad1ee94ce212328ed60179f78366b4a6421eafbc5c6640.jpg)

![354e19a3e031417105160bbbb54fda7fdd0b0aa126590712da960416664afa59.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/354e19a3e031417105160bbbb54fda7fdd0b0aa126590712da960416664afa59.jpg)

![366e3b4e0eb0b5401716d9c94ff331a41ad26d4ceab612d6af93b655b7537ee2.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/366e3b4e0eb0b5401716d9c94ff331a41ad26d4ceab612d6af93b655b7537ee2.jpg)

![6eb1326a8db01ac91089796acbb366a4dcb60d944478022c54bfc3de656d2eb2.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/6eb1326a8db01ac91089796acbb366a4dcb60d944478022c54bfc3de656d2eb2.jpg)

![8f10a422f69638a600834348f2d4a58664d1d3e03595be3765f22a0d3367333a.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/8f10a422f69638a600834348f2d4a58664d1d3e03595be3765f22a0d3367333a.jpg)

![a71fdb0c912f350baf58ab2cafe3b3d3db39d124513af99620e0b4474a71ae83.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/a71fdb0c912f350baf58ab2cafe3b3d3db39d124513af99620e0b4474a71ae83.jpg)

![c5f6095a93f72d1c0d27c95581700150944af88b1c09ee461782b32348322f0c.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/c5f6095a93f72d1c0d27c95581700150944af88b1c09ee461782b32348322f0c.jpg)

![d72e1b9a53844bddda9fd576234acb1f4b591089b8a4822e9a1a122c8442f4a6.jpg](../icml_results/1887_Locality%20Preserving%20Markovian%20Transition%20for%20Instance%20Retrieval/tables/d72e1b9a53844bddda9fd576234acb1f4b591089b8a4822e9a1a122c8442f4a6.jpg)

## Learning Latent Graph Structures and their Uncertainty


### Images

![3900f1024acb055034f73391cb5598f3c538ee91aa90e3aa9e6f93ab90cf14cd.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/3900f1024acb055034f73391cb5598f3c538ee91aa90e3aa9e6f93ab90cf14cd.jpg)

![3d15b13b18233daf6b750103e7c2625cd8a3789edb547c79ed0cbebdff2631f8.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/3d15b13b18233daf6b750103e7c2625cd8a3789edb547c79ed0cbebdff2631f8.jpg)

![59a96f77fcbbb165dffc9864f20e4ae28405094f6b79dd996bdd77c5d91c8222.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/59a96f77fcbbb165dffc9864f20e4ae28405094f6b79dd996bdd77c5d91c8222.jpg)

![6bdb80e5178d9cc5e10d034e4ca1f417374b4faaa99954c58b9936ec712e41b9.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/6bdb80e5178d9cc5e10d034e4ca1f417374b4faaa99954c58b9936ec712e41b9.jpg)

![a1c25101ea8c2fe22726b4a463d6f880cf70db85330a9fa33667b77814d35a1f.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/a1c25101ea8c2fe22726b4a463d6f880cf70db85330a9fa33667b77814d35a1f.jpg)

![ba3df2da297225b9d963aaf3fada311625e09143d3ff2f5be17ce4d659ef9e68.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/ba3df2da297225b9d963aaf3fada311625e09143d3ff2f5be17ce4d659ef9e68.jpg)

![ba93aaef4ddfd6524a654efa47dc907ac2adaae988d93e706fc31ee6ad00dd77.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/ba93aaef4ddfd6524a654efa47dc907ac2adaae988d93e706fc31ee6ad00dd77.jpg)

![bdb351d2e67e307de273d4a345da3c2f54d3263fa11ffc7c9a28c5dce139955c.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/bdb351d2e67e307de273d4a345da3c2f54d3263fa11ffc7c9a28c5dce139955c.jpg)

![cf708754a749f7c91ce0d12555a443c5ba9b82159cccfe664c4b22c159be90f7.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/cf708754a749f7c91ce0d12555a443c5ba9b82159cccfe664c4b22c159be90f7.jpg)

![d50f198b680c0ed25d8c5ea6dd8d48f269b03e5b6feafdb80908d8d7dc1a12a8.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/d50f198b680c0ed25d8c5ea6dd8d48f269b03e5b6feafdb80908d8d7dc1a12a8.jpg)

![e4039143961347a8d7f23702088c13ce9a7100b36f919033f8d2800b4cdce798.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/e4039143961347a8d7f23702088c13ce9a7100b36f919033f8d2800b4cdce798.jpg)

![e969d495fd0cbf016030d8b7eee1a2621cff91e2453847227aeb34796a0c375a.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/e969d495fd0cbf016030d8b7eee1a2621cff91e2453847227aeb34796a0c375a.jpg)

![ec5a0bf61221ac1283810c18a8b940743d9f690d5430bc535f24f3563fc06b4b.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/ec5a0bf61221ac1283810c18a8b940743d9f690d5430bc535f24f3563fc06b4b.jpg)

![f0dd20f9795c43b34182ad3a101628226d7a6d09d94877c6c8eca67e9909a0d9.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/images/f0dd20f9795c43b34182ad3a101628226d7a6d09d94877c6c8eca67e9909a0d9.jpg)

### Tables

![0b879bf29f1c22c2b4cf49446d82b4906cfe32b8cdcce488bda5efb9895b1b49.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/tables/0b879bf29f1c22c2b4cf49446d82b4906cfe32b8cdcce488bda5efb9895b1b49.jpg)

![20362680a005794022dd7636f086b2b5e77cd1f045a873e8ab42e1cfce7a4f04.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/tables/20362680a005794022dd7636f086b2b5e77cd1f045a873e8ab42e1cfce7a4f04.jpg)

![32d0e3fec066734599d52b970c1d47c44f7176ff865aec3e24e8907e2494f822.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/tables/32d0e3fec066734599d52b970c1d47c44f7176ff865aec3e24e8907e2494f822.jpg)

![737d2a722dd48cdc052e4bd4a44cc1eb142f3d0462ee6dbfb62dc11121a2f70c.jpg](../icml_results/1888_Learning%20Latent%20Graph%20Structures%20and%20their%20Uncertainty/tables/737d2a722dd48cdc052e4bd4a44cc1eb142f3d0462ee6dbfb62dc11121a2f70c.jpg)

## Homophily Enhanced Graph Domain Adaptation


### Images

![85ee5ae3a1c139002eda1c9542ebe917a26d21a07c85a06e1f237af149545844.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/85ee5ae3a1c139002eda1c9542ebe917a26d21a07c85a06e1f237af149545844.jpg)

![871d0ec982d4e8e46d5d9d7179a208d260980940994718519fb8017b681055ad.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/871d0ec982d4e8e46d5d9d7179a208d260980940994718519fb8017b681055ad.jpg)

![8da84abaca8a466a02c5825640433920d6b52bb6b9876a6b645d8383175856c8.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/8da84abaca8a466a02c5825640433920d6b52bb6b9876a6b645d8383175856c8.jpg)

![9c99a9208a6e032600deed75f7cd2adbaf812358beafbd2f6be61bbadd9a4c03.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/9c99a9208a6e032600deed75f7cd2adbaf812358beafbd2f6be61bbadd9a4c03.jpg)

![9f87c9e7b983cb83f55b05c83dec11f4d570ad7664c5a7fbd2d663b0026be01b.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/9f87c9e7b983cb83f55b05c83dec11f4d570ad7664c5a7fbd2d663b0026be01b.jpg)

![a23c84d3ba8c58c036f082e82fe8c03bed5cc95b424a6f1949698db816639b58.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/a23c84d3ba8c58c036f082e82fe8c03bed5cc95b424a6f1949698db816639b58.jpg)

![dd3de3929285573fe0e418f7c35860988ed46c70daf40a23a521f0f5d59d7dd0.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/dd3de3929285573fe0e418f7c35860988ed46c70daf40a23a521f0f5d59d7dd0.jpg)

![f22ee11415ef9cf3c869ce3dd7c51647cb48e7b3a6b1c451917cfb5f8981f555.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/f22ee11415ef9cf3c869ce3dd7c51647cb48e7b3a6b1c451917cfb5f8981f555.jpg)

![f5ddf9660e25d55c5cf43d1b24d7e67eccb8e16d6d62495464b9d7183db8785a.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/images/f5ddf9660e25d55c5cf43d1b24d7e67eccb8e16d6d62495464b9d7183db8785a.jpg)

### Tables

![005caec6f5e5ccd997aeffc9234cddc7020c3a035ab80e0da8cbed8bf41da695.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/tables/005caec6f5e5ccd997aeffc9234cddc7020c3a035ab80e0da8cbed8bf41da695.jpg)

![0436d0a438e580bfaf1d0119d24f944386921fa95f4f5262a3426a326b668b64.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/tables/0436d0a438e580bfaf1d0119d24f944386921fa95f4f5262a3426a326b668b64.jpg)

![41f0fa26de0a29c442604e4fce4536ff7fd3851013e92d17be38d5cda73dc067.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/tables/41f0fa26de0a29c442604e4fce4536ff7fd3851013e92d17be38d5cda73dc067.jpg)

![76cdf8d614b0852f8e60eac42199b062251c362786adad00955b13f832f2dc2c.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/tables/76cdf8d614b0852f8e60eac42199b062251c362786adad00955b13f832f2dc2c.jpg)

![8102d5903d05faebf2964a618609e8c5432fe31ef38604ff3060a2253e6f8a60.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/tables/8102d5903d05faebf2964a618609e8c5432fe31ef38604ff3060a2253e6f8a60.jpg)

![e244c42ad8c9ff9ecea666020a4fc8b09391dea172c853c79bac67a255a74291.jpg](../icml_results/1889_Homophily%20Enhanced%20Graph%20Domain%20Adaptation/tables/e244c42ad8c9ff9ecea666020a4fc8b09391dea172c853c79bac67a255a74291.jpg)

## TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization


### Images

![f1347df5b9fe865ca673dbe030378b08e5d7ea0f56ab4745a691a260bb003a90.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/images/f1347df5b9fe865ca673dbe030378b08e5d7ea0f56ab4745a691a260bb003a90.jpg)

### Tables

![037ec5ae9b5bb2370d8983a781ef06a3e6e7916ce6577574d2beada27ddc5182.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/037ec5ae9b5bb2370d8983a781ef06a3e6e7916ce6577574d2beada27ddc5182.jpg)

![3660a1987607f45b5ca59c7f0d8f7b9344d3cc783d68f7a8bc166ee452adc885.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/3660a1987607f45b5ca59c7f0d8f7b9344d3cc783d68f7a8bc166ee452adc885.jpg)

![5eb5cb908c170e0d4d763d26c0c02d17ef8ccc94a695d14d08e648e6fef8f63a.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/5eb5cb908c170e0d4d763d26c0c02d17ef8ccc94a695d14d08e648e6fef8f63a.jpg)

![659bc43ac90f1b79aa122bade96718d5b07edc9ff069ec6791f4f53635c86f72.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/659bc43ac90f1b79aa122bade96718d5b07edc9ff069ec6791f4f53635c86f72.jpg)

![76e0dc1256ec24b0be0107a8226b3fe7e1e00d497548742eb99b1bbcf1de5058.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/76e0dc1256ec24b0be0107a8226b3fe7e1e00d497548742eb99b1bbcf1de5058.jpg)

![8286b69d615eee30e81a62a6358a386879109e9a283625e42243543c8aafc7cf.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/8286b69d615eee30e81a62a6358a386879109e9a283625e42243543c8aafc7cf.jpg)

![b48019188c0b6aa9f3a3860c461120acfb7d16cad5a76e17c18c86c401f899ee.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/b48019188c0b6aa9f3a3860c461120acfb7d16cad5a76e17c18c86c401f899ee.jpg)

![d2d9e0c6a3575bdec2a50c4a96d47c8db65fa36eef01eabf328d8e118abc39c8.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/d2d9e0c6a3575bdec2a50c4a96d47c8db65fa36eef01eabf328d8e118abc39c8.jpg)

![f5d7cbaaa50f9b3cfc6a45e677513788100d453a280637b6534b9d571d44a6b2.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/f5d7cbaaa50f9b3cfc6a45e677513788100d453a280637b6534b9d571d44a6b2.jpg)

![fa40e48f3b070699168edca6551374bdeba9e60fd2c7dd09584723cb296d797b.jpg](../icml_results/1890_TGDPO_%20Harnessing%20Token-Level%20Reward%20Guidance%20for%20Enhancing%20Direct%20Preference%20Optimization/tables/fa40e48f3b070699168edca6551374bdeba9e60fd2c7dd09584723cb296d797b.jpg)

## ROS: A GNN-based Relax-Optimize-and-Sample Framework for Max-$k$-Cut Problems


### Images

![21f8a830d20fd1d49319621481160f249b9856599d833b90e72dc2fe8f2f2d52.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/images/21f8a830d20fd1d49319621481160f249b9856599d833b90e72dc2fe8f2f2d52.jpg)

![5262818b1603ccf2827977e7b8d11dd0cc7852827577f0e7595289d0bede7977.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/images/5262818b1603ccf2827977e7b8d11dd0cc7852827577f0e7595289d0bede7977.jpg)

![531e8a96dfb79114a4be1d937d5b0d8b280043a413755a0bd6f1ecd08ef93b37.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/images/531e8a96dfb79114a4be1d937d5b0d8b280043a413755a0bd6f1ecd08ef93b37.jpg)

![aa0756234104f69cefae3c25b1195539337e093bf47d4a7a4e282b16e59b6faf.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/images/aa0756234104f69cefae3c25b1195539337e093bf47d4a7a4e282b16e59b6faf.jpg)

### Tables

![360119254fd69f0e1e29c04dc756ac910150a4d4c075863d6fcbbd3116e9053e.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/360119254fd69f0e1e29c04dc756ac910150a4d4c075863d6fcbbd3116e9053e.jpg)

![3c4e2172561b54ffd8a62a64cf1fcdb09d14d1598996a1f1f3ff825bc0677c33.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/3c4e2172561b54ffd8a62a64cf1fcdb09d14d1598996a1f1f3ff825bc0677c33.jpg)

![3e10897d61b09d7970e5d3010587d4ad106615ca8d93eb1b450b74b3b90eccce.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/3e10897d61b09d7970e5d3010587d4ad106615ca8d93eb1b450b74b3b90eccce.jpg)

![583c476d5bff331578bf56d8120e9554fb0139fb5ecdff1fe85fa30ba609b112.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/583c476d5bff331578bf56d8120e9554fb0139fb5ecdff1fe85fa30ba609b112.jpg)

![5fe72cab429dc9b3a4e22c9e409eeb25407cced021f6ad0e04b2a86fb46892c1.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/5fe72cab429dc9b3a4e22c9e409eeb25407cced021f6ad0e04b2a86fb46892c1.jpg)

![60fee2b36376d3b82ae395c5c4a9cb1336db982ea70963ca888a1c1be76b0bd1.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/60fee2b36376d3b82ae395c5c4a9cb1336db982ea70963ca888a1c1be76b0bd1.jpg)

![71df1deb456eeb7812cb036e2679e4c7da6e84f1e34b2d0b29ef8de530af5e7f.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/71df1deb456eeb7812cb036e2679e4c7da6e84f1e34b2d0b29ef8de530af5e7f.jpg)

![742106fba4c376fdeed50d0e5d584fe464a8881f7dd5ef213ef1989ced3afad8.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/742106fba4c376fdeed50d0e5d584fe464a8881f7dd5ef213ef1989ced3afad8.jpg)

![7e570ebab76d39a65740fd54282593624a2f0a73a93bc1efadab95963698b292.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/7e570ebab76d39a65740fd54282593624a2f0a73a93bc1efadab95963698b292.jpg)

![8b2fe4e18f52ab32fef13eb6120de0b7fb813a937fbba45a762ab5658f4df668.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/8b2fe4e18f52ab32fef13eb6120de0b7fb813a937fbba45a762ab5658f4df668.jpg)

![92b46e893d7ed0662197955c4647a4c72c0896af3c689ff2ac182e7be17a9cde.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/92b46e893d7ed0662197955c4647a4c72c0896af3c689ff2ac182e7be17a9cde.jpg)

![a17af656839d31effbbd5137a821ce5b8385c81f279024627df8617c0234ffe2.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/a17af656839d31effbbd5137a821ce5b8385c81f279024627df8617c0234ffe2.jpg)

![a5b87eabea517ecb38fa3dc4b94d224312cbeeb1b4dd3ac1dd166ca1e2d70bb5.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/a5b87eabea517ecb38fa3dc4b94d224312cbeeb1b4dd3ac1dd166ca1e2d70bb5.jpg)

![bd0a69e22ae62698bf19a97b8a117158a30564da8ea538f0fbd97e3f9fb76242.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/bd0a69e22ae62698bf19a97b8a117158a30564da8ea538f0fbd97e3f9fb76242.jpg)

![d0243972d014f68ca5761a1290cf3ac180726fcc077b81391ce51840a7de5843.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/d0243972d014f68ca5761a1290cf3ac180726fcc077b81391ce51840a7de5843.jpg)

![e776fc2586dcf9ff179a48baa0946d20af65f8789de3d29119e60240fa848b6d.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/e776fc2586dcf9ff179a48baa0946d20af65f8789de3d29119e60240fa848b6d.jpg)

![ec86987680eaa521b9ee8de70d5efbf577f4a19e7b979f76a29ba260adbbbcbe.jpg](../icml_results/1891_ROS_%20A%20GNN-based%20Relax-Optimize-and-Sample%20Framework%20for%20Max-%24k%24-Cut%20Problems/tables/ec86987680eaa521b9ee8de70d5efbf577f4a19e7b979f76a29ba260adbbbcbe.jpg)

## Retrieval-Augmented Language Model for Knowledge-aware Protein Encoding


### Images

![1d317915b755b3d1c3321e141ce6bbcd31e6c243fba54e225f727daa3199119d.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/images/1d317915b755b3d1c3321e141ce6bbcd31e6c243fba54e225f727daa3199119d.jpg)

![ae567766eb4ff14ed4d6318355b264525068e4355efbdef72472479e6ea5ab82.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/images/ae567766eb4ff14ed4d6318355b264525068e4355efbdef72472479e6ea5ab82.jpg)

![bcd3791628b95d207962b8240654e8efdf76580d390be428ec735fe61dab1e04.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/images/bcd3791628b95d207962b8240654e8efdf76580d390be428ec735fe61dab1e04.jpg)

![c3a7908b6069cd20984991df8c0b1a92ffc88775eb52437d33a37689b97065e5.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/images/c3a7908b6069cd20984991df8c0b1a92ffc88775eb52437d33a37689b97065e5.jpg)

![e5160fa7c8444be9283afaa9e7ce7bddd4b01cf842e689977bd68ea7d9b360d7.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/images/e5160fa7c8444be9283afaa9e7ce7bddd4b01cf842e689977bd68ea7d9b360d7.jpg)

### Tables

![0e309cdd1449200e45cd4392df6dfd6e774c4b7e016eff59b60af990dcec9202.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/0e309cdd1449200e45cd4392df6dfd6e774c4b7e016eff59b60af990dcec9202.jpg)

![11aa2f419b5f389fc7563f5ec2415e411c9b4786d2df32d244e6023a66ffe434.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/11aa2f419b5f389fc7563f5ec2415e411c9b4786d2df32d244e6023a66ffe434.jpg)

![1642259b0b65e792cab9afcf02adc526e7e195d3df07398274680db4c2523e9d.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/1642259b0b65e792cab9afcf02adc526e7e195d3df07398274680db4c2523e9d.jpg)

![250386d600da2309a63ef4d6f12b55cbb5ef58795b0ddbbeba21023fd82bbb60.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/250386d600da2309a63ef4d6f12b55cbb5ef58795b0ddbbeba21023fd82bbb60.jpg)

![286ae0ae19fbc27a411c729a40cf84a9c23982cf3c2d29e21a6c7be9c44733fd.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/286ae0ae19fbc27a411c729a40cf84a9c23982cf3c2d29e21a6c7be9c44733fd.jpg)

![469c99b1759e576bb8ce4828e31b49ca09a4d5b555ccbf4acbb36dac41a89f85.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/469c99b1759e576bb8ce4828e31b49ca09a4d5b555ccbf4acbb36dac41a89f85.jpg)

![5d22f0532ca7b183cce2819521ea41fd38996f62ccc78f697862b83ed591f001.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/5d22f0532ca7b183cce2819521ea41fd38996f62ccc78f697862b83ed591f001.jpg)

![82fd2250dc63101c2de32777b7d101c2085ebddaa3b49d0d3ad25b804fbfc457.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/82fd2250dc63101c2de32777b7d101c2085ebddaa3b49d0d3ad25b804fbfc457.jpg)

![8b4c070fc9947aab1049249107d551cc885bf565a1be6d14edfc86bd582034bf.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/8b4c070fc9947aab1049249107d551cc885bf565a1be6d14edfc86bd582034bf.jpg)

![a0354c6a27e351c95e1dbf522cff93f0b05f7748b1dbe8da65232d3a9583a92f.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/a0354c6a27e351c95e1dbf522cff93f0b05f7748b1dbe8da65232d3a9583a92f.jpg)

![a507e5fdfbd1a2eb5c5da0538c6517c235716630133730fda9e2036c0cb940d4.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/a507e5fdfbd1a2eb5c5da0538c6517c235716630133730fda9e2036c0cb940d4.jpg)

![f69e9855e15eb4e20f7a1785111b126128cdfa7f87cfd61dec33bc39bde83ecf.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/f69e9855e15eb4e20f7a1785111b126128cdfa7f87cfd61dec33bc39bde83ecf.jpg)

![f8d4a37aec40c592f3ff35c23c3fc1385e7ed5e8b4c92ff55814818df5fd976c.jpg](../icml_results/1892_Retrieval-Augmented%20Language%20Model%20for%20Knowledge-aware%20Protein%20Encoding/tables/f8d4a37aec40c592f3ff35c23c3fc1385e7ed5e8b4c92ff55814818df5fd976c.jpg)

## Optimal Fair Learning Robust to Adversarial Distribution Shift


### Images

![47ca0eb455b9a042cd113f3df414e1ca21bb04ec9d5b0e182b36ea2d7940cbd1.jpg](../icml_results/1893_Optimal%20Fair%20Learning%20Robust%20to%20Adversarial%20Distribution%20Shift/images/47ca0eb455b9a042cd113f3df414e1ca21bb04ec9d5b0e182b36ea2d7940cbd1.jpg)

## Tokenized Bandit for LLM Decoding and Alignment


### Images

![1d5083988bb064fad95ece025a76f1e964769720544780e55d0741b439a27b3c.jpg](../icml_results/1894_Tokenized%20Bandit%20for%20LLM%20Decoding%20and%20Alignment/images/1d5083988bb064fad95ece025a76f1e964769720544780e55d0741b439a27b3c.jpg)

![23b774297a90d8964ab663a91d356a2084dde311cdd526d9ef24295cf321b134.jpg](../icml_results/1894_Tokenized%20Bandit%20for%20LLM%20Decoding%20and%20Alignment/images/23b774297a90d8964ab663a91d356a2084dde311cdd526d9ef24295cf321b134.jpg)

![6ba0dd821908403613c70d3c9110a0f11c3d6e53e69fdd322e9efb5ff192bcc4.jpg](../icml_results/1894_Tokenized%20Bandit%20for%20LLM%20Decoding%20and%20Alignment/images/6ba0dd821908403613c70d3c9110a0f11c3d6e53e69fdd322e9efb5ff192bcc4.jpg)

![7e7b0f287f74acdb3b957b005f8626311b3fc310c9059bd20c67bf1f627f26c6.jpg](../icml_results/1894_Tokenized%20Bandit%20for%20LLM%20Decoding%20and%20Alignment/images/7e7b0f287f74acdb3b957b005f8626311b3fc310c9059bd20c67bf1f627f26c6.jpg)

![d12a2c591a78a1d75ccbd943727bb96535c4a85e5e4d1f951dca7085fd296a0b.jpg](../icml_results/1894_Tokenized%20Bandit%20for%20LLM%20Decoding%20and%20Alignment/images/d12a2c591a78a1d75ccbd943727bb96535c4a85e5e4d1f951dca7085fd296a0b.jpg)

### Tables

![a3337e18ce194ce020ce821bf312b71a27ff687852f95ab55e4e931af0b9a0f6.jpg](../icml_results/1894_Tokenized%20Bandit%20for%20LLM%20Decoding%20and%20Alignment/tables/a3337e18ce194ce020ce821bf312b71a27ff687852f95ab55e4e931af0b9a0f6.jpg)

## Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration


### Images

![03cc02587ec58cfc2badf14435baef93696a8d555bed5701277d3e11d7cc9ba9.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/03cc02587ec58cfc2badf14435baef93696a8d555bed5701277d3e11d7cc9ba9.jpg)

![0c077583b6938a24a7f95103048c476e55795d337e4e149086874b21f35048b5.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/0c077583b6938a24a7f95103048c476e55795d337e4e149086874b21f35048b5.jpg)

![0c3ce5ce2fa6671d7e7c58d4e85f889903fce4793ba719685a9b6737ffeb4452.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/0c3ce5ce2fa6671d7e7c58d4e85f889903fce4793ba719685a9b6737ffeb4452.jpg)

![0d72786829d49d2414636c90531aa309df601f3f4dc61e4342fd490bdb19ea03.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/0d72786829d49d2414636c90531aa309df601f3f4dc61e4342fd490bdb19ea03.jpg)

![137dcde0a544a5d96a43843a966cc3a8d69587bae476e572ea09c1976e92ef76.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/137dcde0a544a5d96a43843a966cc3a8d69587bae476e572ea09c1976e92ef76.jpg)

![228d80d4ec3d6edef420abaaf3fa074a43bbc7fd348004011e0ce1fbea8dbc1e.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/228d80d4ec3d6edef420abaaf3fa074a43bbc7fd348004011e0ce1fbea8dbc1e.jpg)

![271254a2641720d13d99e2ae6fc2f64d70e753673622954bf0d02a34a442e2af.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/271254a2641720d13d99e2ae6fc2f64d70e753673622954bf0d02a34a442e2af.jpg)

![368f28e2e424b1d9ce1d21ac5df8cb255c73db1f47a6972bbeb4aa39ab668489.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/368f28e2e424b1d9ce1d21ac5df8cb255c73db1f47a6972bbeb4aa39ab668489.jpg)

![37da5a3365a3c2376c6634f3a50da5a489347a4caf350ef21df5e3b6dcebabc9.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/37da5a3365a3c2376c6634f3a50da5a489347a4caf350ef21df5e3b6dcebabc9.jpg)

![37f1aba4fa47a8a8aaa184992f1ec242e362da054d5a8f968ed49dc68c96ac41.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/37f1aba4fa47a8a8aaa184992f1ec242e362da054d5a8f968ed49dc68c96ac41.jpg)

![3be9cce03b9d84c47edce60b7db2881c5937b3f56b46ea6ee95fffbc6e360f15.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/3be9cce03b9d84c47edce60b7db2881c5937b3f56b46ea6ee95fffbc6e360f15.jpg)

![69f231c9b575cbc7925bf7835f7962aa3fa835490eff0a4bfe10c8cd9d6e3e10.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/69f231c9b575cbc7925bf7835f7962aa3fa835490eff0a4bfe10c8cd9d6e3e10.jpg)

![701b893935c621daeceda4b2b2a6e19fe6815558a98879299e39bf2215e5c5d9.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/701b893935c621daeceda4b2b2a6e19fe6815558a98879299e39bf2215e5c5d9.jpg)

![82553682178da6c858734a41089bc75a8f77fb648a8e75399bc9c27a6c59eaff.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/82553682178da6c858734a41089bc75a8f77fb648a8e75399bc9c27a6c59eaff.jpg)

![9b0739d50e60b3dfe774c7a54e00926b38468ea623200bea6ef9da14909a9036.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/9b0739d50e60b3dfe774c7a54e00926b38468ea623200bea6ef9da14909a9036.jpg)

![aa71152a7f67f5ed917f50c159b8bc34b54c13b183ebeec643dd9e174f5e4b57.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/aa71152a7f67f5ed917f50c159b8bc34b54c13b183ebeec643dd9e174f5e4b57.jpg)

![b6be80e8094d999b819005bfac388f5344cfe396412ee624bc9d43102ce2a78b.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/b6be80e8094d999b819005bfac388f5344cfe396412ee624bc9d43102ce2a78b.jpg)

![b852f7419d42f8b3081338e9a43c7247c725094b6cd5b88059fdadc313d8bf22.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/b852f7419d42f8b3081338e9a43c7247c725094b6cd5b88059fdadc313d8bf22.jpg)

![b9f665cf6934c0c6754f1c4fdfe1a472345a15f1e6ead49a799f8bb45756b44d.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/b9f665cf6934c0c6754f1c4fdfe1a472345a15f1e6ead49a799f8bb45756b44d.jpg)

![c407dc2f8912a1fdd78996c0df1acdad78b084927033b8ba20cc83f177b72428.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/c407dc2f8912a1fdd78996c0df1acdad78b084927033b8ba20cc83f177b72428.jpg)

![c43a3bad599665c095283d16945a39a106ce4a93dd01fdb3ba068226cb131c0c.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/c43a3bad599665c095283d16945a39a106ce4a93dd01fdb3ba068226cb131c0c.jpg)

![cb05fa9a905f975d2f1c85fc76ecfb8785f2c0ece8aecba95c21fbf0b11d1039.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/cb05fa9a905f975d2f1c85fc76ecfb8785f2c0ece8aecba95c21fbf0b11d1039.jpg)

![dca6e741ce5f9e10977e08aeac14a4a204dda90ec435109ee74b832fd6b5e4c0.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/dca6e741ce5f9e10977e08aeac14a4a204dda90ec435109ee74b832fd6b5e4c0.jpg)

![e6559e93d27d5be8d53c806e1296912ee873fbff2d4f49861ddab0fbb63e6211.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/e6559e93d27d5be8d53c806e1296912ee873fbff2d4f49861ddab0fbb63e6211.jpg)

![eef563af37b43ed19e9b7697552d2bb00b4ecc0af45a06d4c8b3df481037a3bf.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/images/eef563af37b43ed19e9b7697552d2bb00b4ecc0af45a06d4c8b3df481037a3bf.jpg)

### Tables

![0a89a814b180e8605906acf67cf18575a88b2340e200f8b7726b57a3050a051f.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/tables/0a89a814b180e8605906acf67cf18575a88b2340e200f8b7726b57a3050a051f.jpg)

![1c806ef6f98e2ed6c5b49223db36df0735bb7466ad80675d2d9e4b501f94447f.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/tables/1c806ef6f98e2ed6c5b49223db36df0735bb7466ad80675d2d9e4b501f94447f.jpg)

![20731ee3d9ac3db5a8fa4c72cce0d2a9f879f0a46da869c2c68361d6e841a84f.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/tables/20731ee3d9ac3db5a8fa4c72cce0d2a9f879f0a46da869c2c68361d6e841a84f.jpg)

![54865534f62fd0e2e9934acec21ce70c84af2faef7c78bbaae16c0c4ebc89dc0.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/tables/54865534f62fd0e2e9934acec21ce70c84af2faef7c78bbaae16c0c4ebc89dc0.jpg)

![5e094379925ec39dada5e94b676b6663112c4abdd34d34463b2eadfa3b93ec2c.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/tables/5e094379925ec39dada5e94b676b6663112c4abdd34d34463b2eadfa3b93ec2c.jpg)

![6f2c92d597165b8092dec115724d6660f366d49524c280b8b41ffff2137721d2.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/tables/6f2c92d597165b8092dec115724d6660f366d49524c280b8b41ffff2137721d2.jpg)

![81d7ee7c3d200b35c353797405b22979f0b1900a79f255e7d2aad01fb647f1dc.jpg](../icml_results/1895_Enhancing%20Cooperative%20Multi-Agent%20Reinforcement%20Learning%20with%20State%20Modelling%20and%20Adversarial%20Explor/tables/81d7ee7c3d200b35c353797405b22979f0b1900a79f255e7d2aad01fb647f1dc.jpg)

## Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism


### Images

![032a8d0dc6353fa35e6df9b7150fa41e7ed11ae56d783619aaabfb5ffe1ce7f4.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/032a8d0dc6353fa35e6df9b7150fa41e7ed11ae56d783619aaabfb5ffe1ce7f4.jpg)

![0fe5625a38d2961c2c47ec38d3c4fde337644b01042afff544ccf7a8a3d8e93d.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/0fe5625a38d2961c2c47ec38d3c4fde337644b01042afff544ccf7a8a3d8e93d.jpg)

![2479a64a22a90c20a88d734cfd40cef8f8d9335eec89808c4a94ad9e0e7d6663.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/2479a64a22a90c20a88d734cfd40cef8f8d9335eec89808c4a94ad9e0e7d6663.jpg)

![3c97395119bb8731ce60a8df5e672b6c65a8aeb59df3c085146e92ec2de9fc78.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/3c97395119bb8731ce60a8df5e672b6c65a8aeb59df3c085146e92ec2de9fc78.jpg)

![48853b5080501e27d5a7111a4d8b9572e8f380ac3527d12c614bfa104b700d89.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/48853b5080501e27d5a7111a4d8b9572e8f380ac3527d12c614bfa104b700d89.jpg)

![6e6b684e8bccafe1d2e7bbb78fc3377f3bb86468702ea2baf61fd7a9153160b7.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/6e6b684e8bccafe1d2e7bbb78fc3377f3bb86468702ea2baf61fd7a9153160b7.jpg)

![894587a10a73cc98d3b00108b3200190f0156207d7ebe7c2c819e1a1dd50e8c4.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/894587a10a73cc98d3b00108b3200190f0156207d7ebe7c2c819e1a1dd50e8c4.jpg)

![92b9818ddf15ea0e7924617cde43003953f7fb4a6218f8986df8b78013de857a.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/92b9818ddf15ea0e7924617cde43003953f7fb4a6218f8986df8b78013de857a.jpg)

![b3034ce2fefd847cb153b50dd0314e0d3beca73ed57e479b77c9e12b625285c4.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/b3034ce2fefd847cb153b50dd0314e0d3beca73ed57e479b77c9e12b625285c4.jpg)

![f5aa37a099048422485ce50e2a9b16437830821285e2d873e04360551603d493.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/images/f5aa37a099048422485ce50e2a9b16437830821285e2d873e04360551603d493.jpg)

### Tables

![159639bfb1ec96e32d8c72521b9455c884966097102065533a508c999bb18352.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/159639bfb1ec96e32d8c72521b9455c884966097102065533a508c999bb18352.jpg)

![3655cfd234f63a26012d0523a87c62f1c0d14c0e56d3518b0780aff9b8933e89.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/3655cfd234f63a26012d0523a87c62f1c0d14c0e56d3518b0780aff9b8933e89.jpg)

![39d38736f93ae8cc957c0c9b4c949b5407f54823cf73c7441657bad67ea38abf.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/39d38736f93ae8cc957c0c9b4c949b5407f54823cf73c7441657bad67ea38abf.jpg)

![4e403673cc17f8baff1cb8b8c4ca81a7385fa6499e7485d27288a0146cd7f110.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/4e403673cc17f8baff1cb8b8c4ca81a7385fa6499e7485d27288a0146cd7f110.jpg)

![589aae678eb89c1c09f5724be21cdf516e0e01f5cd7a4a2de2fcb2259266d622.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/589aae678eb89c1c09f5724be21cdf516e0e01f5cd7a4a2de2fcb2259266d622.jpg)

![66466ff0f2d6f94d89da71b322e504b93a3583610f88a4467be18a5a95c78f13.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/66466ff0f2d6f94d89da71b322e504b93a3583610f88a4467be18a5a95c78f13.jpg)

![6942646836711d23e40ca52dd43b0e36924755d32745be275872636361c77911.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/6942646836711d23e40ca52dd43b0e36924755d32745be275872636361c77911.jpg)

![90fe52b4af76c0b90b3136ffcaa2682472be923bcfe76aa1f3329ce9b7fe925f.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/90fe52b4af76c0b90b3136ffcaa2682472be923bcfe76aa1f3329ce9b7fe925f.jpg)

![c0d2effea1d21118f845522f0335a266ee586539048949db4a44dd2184aa9d57.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/c0d2effea1d21118f845522f0335a266ee586539048949db4a44dd2184aa9d57.jpg)

![cf50cec88227a7e42a3df2b2233f5e27ff87ba5bbc554dce7f9dc8bb8f1744ae.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/cf50cec88227a7e42a3df2b2233f5e27ff87ba5bbc554dce7f9dc8bb8f1744ae.jpg)

![e58f88af26c33a45c0b83903526232c18e0bfa1c29f154d861315f607e8ef8d1.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/e58f88af26c33a45c0b83903526232c18e0bfa1c29f154d861315f607e8ef8d1.jpg)

![f5428f0676ddfa055f67b85282cd54b5c272fa82750fa1c2d546af56eb159d6b.jpg](../icml_results/1896_Robot-Gated%20Interactive%20Imitation%20Learning%20with%20Adaptive%20Intervention%20Mechanism/tables/f5428f0676ddfa055f67b85282cd54b5c272fa82750fa1c2d546af56eb159d6b.jpg)

## A Rescaling-Invariant Lipschitz Bound Based on Path-Metrics for Modern ReLU Network Parameterizations


### Images

![33fdc7bc257d862bdce72f0f71c9411f9572d6ff916c6cc3064030a7d9e13abb.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/images/33fdc7bc257d862bdce72f0f71c9411f9572d6ff916c6cc3064030a7d9e13abb.jpg)

![4863c3a9cff329e08fef174ac37678aa3fd63fbfa2013538158b616926512ecb.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/images/4863c3a9cff329e08fef174ac37678aa3fd63fbfa2013538158b616926512ecb.jpg)

![581986e397cb8eae4092dcefe55d5ae110d158b27f3c97ef16e6a37d411301ef.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/images/581986e397cb8eae4092dcefe55d5ae110d158b27f3c97ef16e6a37d411301ef.jpg)

![64d70a78682dd7bdeda53809240721c9d48fad487c90d16fc60f602471375db0.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/images/64d70a78682dd7bdeda53809240721c9d48fad487c90d16fc60f602471375db0.jpg)

![6cd8fdace96163b32f10738b47946a34463130f00cac632edc9234b755a50bb0.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/images/6cd8fdace96163b32f10738b47946a34463130f00cac632edc9234b755a50bb0.jpg)

![c56b5a8fa88d6b8f7f51df1e099f8e8335abd1f6aa32e0061112ac3e8c353f53.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/images/c56b5a8fa88d6b8f7f51df1e099f8e8335abd1f6aa32e0061112ac3e8c353f53.jpg)

![f2b4ce41515898ce3dcace8b90269f8f6c960f3d605f7b9face8753602cc184d.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/images/f2b4ce41515898ce3dcace8b90269f8f6c960f3d605f7b9face8753602cc184d.jpg)

### Tables

![09f63e55e91c0030f151fba914c533b498685853ccffefb56faf85104801b87c.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/tables/09f63e55e91c0030f151fba914c533b498685853ccffefb56faf85104801b87c.jpg)

![35cf9aef1507dd5bd2ae62d3d54e954e6729a8e860ce40a2de84f4623308de10.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/tables/35cf9aef1507dd5bd2ae62d3d54e954e6729a8e860ce40a2de84f4623308de10.jpg)

![562db5af56f28320a23e5983891797bb079c2f3f941946dd9363b96e86b197d0.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/tables/562db5af56f28320a23e5983891797bb079c2f3f941946dd9363b96e86b197d0.jpg)

![739107689c5769c5d261d3a7d13554fd260978276cf55620b0f22ca183abcfb4.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/tables/739107689c5769c5d261d3a7d13554fd260978276cf55620b0f22ca183abcfb4.jpg)

![a91b7b059e991f018c1205c2569b1b9589ad178a5a3ebdddde495558e83afab6.jpg](../icml_results/1897_A%20Rescaling-Invariant%20Lipschitz%20Bound%20Based%20on%20Path-Metrics%20for%20Modern%20ReLU%20Network%20Parameterization/tables/a91b7b059e991f018c1205c2569b1b9589ad178a5a3ebdddde495558e83afab6.jpg)

## PTTA: Purifying Malicious Samples for Test-Time Model Adaptation


### Images

![005917bcf0bc814c422f5fd648c8c30d149f3b0fca060c7560b8eae37d19b200.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/005917bcf0bc814c422f5fd648c8c30d149f3b0fca060c7560b8eae37d19b200.jpg)

![036c49eaa1ae370ed55b292666c8c3be868c167c5b1d60e3489698af53f770d6.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/036c49eaa1ae370ed55b292666c8c3be868c167c5b1d60e3489698af53f770d6.jpg)

![0ca39617c24bb60ddd1ff86a36cac8db5ac9e25317995201a7e1cf6882d360c5.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/0ca39617c24bb60ddd1ff86a36cac8db5ac9e25317995201a7e1cf6882d360c5.jpg)

![32a95c6ac6816221dcfe3575505a35c9b22a60e03e8c998e078d5a181fb44e37.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/32a95c6ac6816221dcfe3575505a35c9b22a60e03e8c998e078d5a181fb44e37.jpg)

![342686bdb99b84436f60df9b27c8b43d1f286273c47ee43cabad0a52096da56a.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/342686bdb99b84436f60df9b27c8b43d1f286273c47ee43cabad0a52096da56a.jpg)

![41716de8570e2e427a0dce8230780eac1772b06a8f921849a8be7bc125f84326.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/41716de8570e2e427a0dce8230780eac1772b06a8f921849a8be7bc125f84326.jpg)

![452a51b7b955c79f6c7a1af312c4e02703d5d20c58f9a557163fc7eece54ec45.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/452a51b7b955c79f6c7a1af312c4e02703d5d20c58f9a557163fc7eece54ec45.jpg)

![965b51da5008255cee39411d1f583c32b41f6513f1a844ccd6e4ab28345cdbc7.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/965b51da5008255cee39411d1f583c32b41f6513f1a844ccd6e4ab28345cdbc7.jpg)

![a0361a7ae76a590839bfffdc0876250bb4e9b6de2f5f65de81974d2277bc63dd.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/a0361a7ae76a590839bfffdc0876250bb4e9b6de2f5f65de81974d2277bc63dd.jpg)

![e50ad749ab7e5da3293a68c8fa722053ceda64edf601efac0554421274a2ab1e.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/e50ad749ab7e5da3293a68c8fa722053ceda64edf601efac0554421274a2ab1e.jpg)

![e82eb7d7cca80a2f74149acca7a5e371a7444db0cf28312cbb454d78896f473a.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/e82eb7d7cca80a2f74149acca7a5e371a7444db0cf28312cbb454d78896f473a.jpg)

![fbe9d5d4daf14bdf3547ec2bd0905631f0b4157b759721008bfee835e73a12bb.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/fbe9d5d4daf14bdf3547ec2bd0905631f0b4157b759721008bfee835e73a12bb.jpg)

![fd0b4a1c20da0f2d6a8b4b52159561c5d079a611d6daa168ade76d4b7ac914f2.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/images/fd0b4a1c20da0f2d6a8b4b52159561c5d079a611d6daa168ade76d4b7ac914f2.jpg)

### Tables

![003e2aee7cc6c3c77eacad1ce13bf40fc8c65d758dcff73545bcd7ad45261f91.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/003e2aee7cc6c3c77eacad1ce13bf40fc8c65d758dcff73545bcd7ad45261f91.jpg)

![185f0e165863f61d5450bfc8cb15c694767279b10441eea92f14f6b89cb07faa.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/185f0e165863f61d5450bfc8cb15c694767279b10441eea92f14f6b89cb07faa.jpg)

![3a5f0dd2cde6bb1cffa974d5eb88df2d3c52d4024b216608091b8bb5bf9bd2e2.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/3a5f0dd2cde6bb1cffa974d5eb88df2d3c52d4024b216608091b8bb5bf9bd2e2.jpg)

![5333dad3bbadd784151232b45e6d3ceaaf6548804513312e6e1d0d4d39c1caec.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/5333dad3bbadd784151232b45e6d3ceaaf6548804513312e6e1d0d4d39c1caec.jpg)

![56e361ec6510ff64e260b64ed15e0828352e396acfa8ba2ed81114941174046f.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/56e361ec6510ff64e260b64ed15e0828352e396acfa8ba2ed81114941174046f.jpg)

![607395e62b37f175abfe5f418a83be68389d7ca28374d314319554af9f245144.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/607395e62b37f175abfe5f418a83be68389d7ca28374d314319554af9f245144.jpg)

![70b2c2ececaccb3e4ab4d91064b83b0916f5e3c0b5ba74557d01bbed7be2a5a2.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/70b2c2ececaccb3e4ab4d91064b83b0916f5e3c0b5ba74557d01bbed7be2a5a2.jpg)

![7fbbc14191d65975e63af0ab8791123c939ca04094d3763507967962ef7ba460.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/7fbbc14191d65975e63af0ab8791123c939ca04094d3763507967962ef7ba460.jpg)

![96a148da2d6380d9b2bab0cb31caf9d3821f670cd914a31dc8a05c6c49d1335f.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/96a148da2d6380d9b2bab0cb31caf9d3821f670cd914a31dc8a05c6c49d1335f.jpg)

![97f528b5cf55ead1a304b3c877f672173e0c5ea3169606871e59e709168080fb.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/97f528b5cf55ead1a304b3c877f672173e0c5ea3169606871e59e709168080fb.jpg)

![a7b1a421cd1d8bca7508a6edd3c6d83b3640be1d6b9e9bf6a35f88fc882954ee.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/a7b1a421cd1d8bca7508a6edd3c6d83b3640be1d6b9e9bf6a35f88fc882954ee.jpg)

![a816a8f8e5624d84ca3fce6cdc6915a4e8b187fa38de74c5bc330944c93a7254.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/a816a8f8e5624d84ca3fce6cdc6915a4e8b187fa38de74c5bc330944c93a7254.jpg)

![a8c5bce32e909a6a080f0c6e5f3403c293d7c7e0b08c3ad377dc8a0dd7116c93.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/a8c5bce32e909a6a080f0c6e5f3403c293d7c7e0b08c3ad377dc8a0dd7116c93.jpg)

![b014ecba9d5126c6d978266888fda3966b9bf9090c9b856ae28479e350a66e08.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/b014ecba9d5126c6d978266888fda3966b9bf9090c9b856ae28479e350a66e08.jpg)

![b69c7f8eceb7ea402464d1808ff8f6435397723aa59b5412aa180987c77c9895.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/b69c7f8eceb7ea402464d1808ff8f6435397723aa59b5412aa180987c77c9895.jpg)

![bdd95a5911b512780f90582f078c40aa0154ecf84d692d748d8d5168d29b7347.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/bdd95a5911b512780f90582f078c40aa0154ecf84d692d748d8d5168d29b7347.jpg)

![c00a70857b42b79f17e58cf77b64e3e20ad52d78a9f5970372d2e01951f78765.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/c00a70857b42b79f17e58cf77b64e3e20ad52d78a9f5970372d2e01951f78765.jpg)

![cb8045926be7a1b2a0bbdfa908a3a4344a0c328c00771fc523605ae370ef4178.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/cb8045926be7a1b2a0bbdfa908a3a4344a0c328c00771fc523605ae370ef4178.jpg)

![d513f69394487744d91fa1abf0f128845a53b7ba093c844058c67e74583a1ac1.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/d513f69394487744d91fa1abf0f128845a53b7ba093c844058c67e74583a1ac1.jpg)

![d90a9664693077c0be3b449c3038a7b06c5ae2f0a05d2c98acd00593b97ead72.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/d90a9664693077c0be3b449c3038a7b06c5ae2f0a05d2c98acd00593b97ead72.jpg)

![e4dbf61f0d22048a15f538e568fca57b24dd93b918c70672b4e002c21a7b3386.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/e4dbf61f0d22048a15f538e568fca57b24dd93b918c70672b4e002c21a7b3386.jpg)

![ede694b55c4897ed508f7f85bd114718875a0b676a6290dc104b26f62267cda8.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/ede694b55c4897ed508f7f85bd114718875a0b676a6290dc104b26f62267cda8.jpg)

![f9ae7ae7eefb64c8c06f740686ebde269cde374d5d798c80702ee93df7bad4ac.jpg](../icml_results/1898_PTTA_%20Purifying%20Malicious%20Samples%20for%20Test-Time%20Model%20Adaptation/tables/f9ae7ae7eefb64c8c06f740686ebde269cde374d5d798c80702ee93df7bad4ac.jpg)

## Linear Transformers as VAR Models:  Aligning Autoregressive Attention Mechanisms with Autoregressive Forecasting


### Images

![0c31ab740ea170039394eee688eaf3707c11da7fb7841b8cb1b39ab824e87c96.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/0c31ab740ea170039394eee688eaf3707c11da7fb7841b8cb1b39ab824e87c96.jpg)

![26253fcd0383f355cc50aa70e30be540b813604adca95e978a2d363c6fa464d3.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/26253fcd0383f355cc50aa70e30be540b813604adca95e978a2d363c6fa464d3.jpg)

![2a5b5ab130abe726eb60f722c6cf295153bcb02f1ba1467ec161fbdb56ffaf84.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/2a5b5ab130abe726eb60f722c6cf295153bcb02f1ba1467ec161fbdb56ffaf84.jpg)

![2ac3c65e483d1526f11e48c1498256a75309b34fc77e798ca49a2290a8b1049d.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/2ac3c65e483d1526f11e48c1498256a75309b34fc77e798ca49a2290a8b1049d.jpg)

![2e1fbaf0771c875e2153ee8cb137930badf4efd7053818718bfc3e12f3d667f7.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/2e1fbaf0771c875e2153ee8cb137930badf4efd7053818718bfc3e12f3d667f7.jpg)

![6a119df969910e10b1f6a91dcd81e565be184d9441d0c01f78d77f50cd387b0c.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/6a119df969910e10b1f6a91dcd81e565be184d9441d0c01f78d77f50cd387b0c.jpg)

![99e86af82aa42e7e88dcbf93eb6cc672b76b7f70261eb6284736e0c6f72ea526.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/99e86af82aa42e7e88dcbf93eb6cc672b76b7f70261eb6284736e0c6f72ea526.jpg)

![bb6c98d601752481936fd9b86b8cae219459b994facedb3183fad48b2e841a17.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/bb6c98d601752481936fd9b86b8cae219459b994facedb3183fad48b2e841a17.jpg)

![d48c74d0854b5d0613e4dca967b5c35539a175d2a817f7dee609f8ff9bf767d0.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/d48c74d0854b5d0613e4dca967b5c35539a175d2a817f7dee609f8ff9bf767d0.jpg)

![dc4a2bffd430f3595f882ac39798ea9b6fc6afd0a38fe270dc678c37ad1767ae.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/dc4a2bffd430f3595f882ac39798ea9b6fc6afd0a38fe270dc678c37ad1767ae.jpg)

![dd50de4e5953beec7455d114d2ad5fdae305bb4fb574cd6268632d801ac04e8d.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/dd50de4e5953beec7455d114d2ad5fdae305bb4fb574cd6268632d801ac04e8d.jpg)

![e44b90d481b1d8114d0e1e4740032b81573dda078909982267b25f2dd5711686.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/e44b90d481b1d8114d0e1e4740032b81573dda078909982267b25f2dd5711686.jpg)

![e74cce1a3aed481258981cdaa95dcfc92a28db8cfc7d78ed61d5cbb340fa69a4.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/images/e74cce1a3aed481258981cdaa95dcfc92a28db8cfc7d78ed61d5cbb340fa69a4.jpg)

### Tables

![0115f68abeb9b6038a632530c113f65c625e828853769bb7909024089919f3a1.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/tables/0115f68abeb9b6038a632530c113f65c625e828853769bb7909024089919f3a1.jpg)

![28eb9129784ecf6afda8caa87c473895a2d8d674c0f0f8ef622ad0a4ee2d095f.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/tables/28eb9129784ecf6afda8caa87c473895a2d8d674c0f0f8ef622ad0a4ee2d095f.jpg)

![4d0bde0c6c7dd7d085b6796a16646210baaf90cd577f95b1e0faae12fbe56cc2.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/tables/4d0bde0c6c7dd7d085b6796a16646210baaf90cd577f95b1e0faae12fbe56cc2.jpg)

![8038072131a1a81d9f47855097a79d626d6f7da7e3d7d584a93d50183c441d12.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/tables/8038072131a1a81d9f47855097a79d626d6f7da7e3d7d584a93d50183c441d12.jpg)

![e1ba61cb8a0e7f45f6060fb139cfb6539ef4537b98d6e8c9dfaf6b777845682c.jpg](../icml_results/1899_Linear%20Transformers%20as%20VAR%20Models_%20%20Aligning%20Autoregressive%20Attention%20Mechanisms%20with%20Autoregressive/tables/e1ba61cb8a0e7f45f6060fb139cfb6539ef4537b98d6e8c9dfaf6b777845682c.jpg)

## Tree-Sliced Wasserstein Distance with Nonlinear Projection

### Images

![0e7ce1407aa06c4b4b0e075c410e22d6526a7df2a4b5fc9ea9f61a51275b8cbf.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/0e7ce1407aa06c4b4b0e075c410e22d6526a7df2a4b5fc9ea9f61a51275b8cbf.jpg)

![24579406d1a9dae39ff78619b7ee6c652ff93a9b038e85482073c95a3842c2aa.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/24579406d1a9dae39ff78619b7ee6c652ff93a9b038e85482073c95a3842c2aa.jpg)

![4dfaa7cf6f83bc69c9d552d723ee7830a18d06b278b3b25032af084ad16ce081.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/4dfaa7cf6f83bc69c9d552d723ee7830a18d06b278b3b25032af084ad16ce081.jpg)

![559a199f6e3869304f62ae972ad901cfcc3981acab0c1e9d322a5bb0aba082af.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/559a199f6e3869304f62ae972ad901cfcc3981acab0c1e9d322a5bb0aba082af.jpg)

![67214d6c024b320f020e09640be49e3304ed499136fc2030c4e8a3c2ffe60b71.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/67214d6c024b320f020e09640be49e3304ed499136fc2030c4e8a3c2ffe60b71.jpg)

![b1ef6d8d0bd54e832f321b03f58529d20c0428e25c4b0c58912811e38bff3f2b.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/b1ef6d8d0bd54e832f321b03f58529d20c0428e25c4b0c58912811e38bff3f2b.jpg)

![c1ad9d73503bf32415637cb585cf0592fbcf06fbc1d7da7036f989f01b91d03a.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/c1ad9d73503bf32415637cb585cf0592fbcf06fbc1d7da7036f989f01b91d03a.jpg)

![fad2db4eb505ecb1d04b228d2df12f0f8b06be597a444205d93d1646da86cb77.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/images/fad2db4eb505ecb1d04b228d2df12f0f8b06be597a444205d93d1646da86cb77.jpg)

### Tables

![0a2fcad2f6dc21919209941270ec493cf98956692e35896d65688aea3241beec.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/0a2fcad2f6dc21919209941270ec493cf98956692e35896d65688aea3241beec.jpg)

![40b092c7f1b6b4146c3b29e012ebe03cc4e881a929f326a8c33776bc2addece8.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/40b092c7f1b6b4146c3b29e012ebe03cc4e881a929f326a8c33776bc2addece8.jpg)

![48b0833187843f669599b025e2912bfc1db2f1123459a69bc7a30d106c14ca81.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/48b0833187843f669599b025e2912bfc1db2f1123459a69bc7a30d106c14ca81.jpg)

![630d57ce91c316eadfbc97864d7af1be0d8f75c3261635ac8a4af9be73f6bc78.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/630d57ce91c316eadfbc97864d7af1be0d8f75c3261635ac8a4af9be73f6bc78.jpg)

![751050851f34874ed3b99f8f8b95c4dcc0722d9442cdf7971d233b2fbe41de1a.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/751050851f34874ed3b99f8f8b95c4dcc0722d9442cdf7971d233b2fbe41de1a.jpg)

![7bf39590d34b113b34e11f363aff6864dd96d6c7607509885f5fc521173c0abd.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/7bf39590d34b113b34e11f363aff6864dd96d6c7607509885f5fc521173c0abd.jpg)

![b13dfc318e8f62823f944ed881fcafe068ca02bf83536b19e59572ef0984dfff.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/b13dfc318e8f62823f944ed881fcafe068ca02bf83536b19e59572ef0984dfff.jpg)

![c6d0702f43e2435b47bfdce178a16c4a6c3028d1b60a526877fa3de53ebcff2d.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/c6d0702f43e2435b47bfdce178a16c4a6c3028d1b60a526877fa3de53ebcff2d.jpg)

![d4c808e224c35044f8a4171f138080d1373a8bd01a6b878f8fe545477d3c5960.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/d4c808e224c35044f8a4171f138080d1373a8bd01a6b878f8fe545477d3c5960.jpg)

![f758910e535f1e302249b93a83b22042c10d8426dbb48bf7bc0af406f466434c.jpg](../icml_results/1900_Tree-Sliced%20Wasserstein%20Distance%20with%20Nonlinear%20Projection/tables/f758910e535f1e302249b93a83b22042c10d8426dbb48bf7bc0af406f466434c.jpg)
