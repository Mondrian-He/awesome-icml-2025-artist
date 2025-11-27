# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 58 of 100

## 目录 (Table of Contents)

1. [Tightening Causal Bounds via Covariate-Aware Optimal Transport](#Tightening-Causal-Bounds-via-Covariate-Aware-Optimal-Transport)
2. [Generative Audio Language Modeling with Continuous-valued Tokens and Masked Next-Token Prediction](#Generative-Audio-Language-Modeling-with-Continuous-valued-Tokens-and-Masked-Next-Token-Prediction)
3. [ResKoopNet: Learning Koopman Representations for Complex Dynamics with Spectral Residuals](#ResKoopNet-Learning-Koopman-Representations-for-Complex-Dynamics-with-Spectral-Residuals)
4. [Dialogue Without Limits: Constant-Sized KV Caches for Extended Response in LLMs](#Dialogue-Without-Limits-Constant-Sized-KV-Caches-for-Extended-Response-in-LLMs)
5. [Tree-Sliced Wasserstein Distance: A Geometric Perspective](#Tree-Sliced-Wasserstein-Distance-A-Geometric-Perspective)
6. [When Bad Data Leads to Good Models](#When-Bad-Data-Leads-to-Good-Models)
7. [A Physics-Informed Machine Learning Framework for Safe and Optimal Control of Autonomous Systems](#A-Physics-Informed-Machine-Learning-Framework-for-Safe-and-Optimal-Control-of-Autonomous-Systems)
8. [Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts](#Moirai-MoE-Empowering-Time-Series-Foundation-Models-with-Sparse-Mixture-of-Experts)
9. [Towards Theoretical Understanding of Sequential Decision Making with Preference Feedback](#Towards-Theoretical-Understanding-of-Sequential-Decision-Making-with-Preference-Feedback)
10. [Regret-Free Reinforcement Learning for Temporal Logic Specifications](#Regret-Free-Reinforcement-Learning-for-Temporal-Logic-Specifications)
11. [LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces](#LIVS-A-Pluralistic-Alignment-Dataset-for-Inclusive-Public-Spaces)
12. [Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models](#Simultaneous-Multi-Robot-Motion-Planning-with-Projected-Diffusion-Models)
13. [Off-Policy Evaluation under Nonignorable Missing Data](#Off-Policy-Evaluation-under-Nonignorable-Missing-Data)
14. [The Perils of Optimizing Learned Reward Functions: Low Training Error Does Not Guarantee Low Regret](#The-Perils-of-Optimizing-Learned-Reward-Functions-Low-Training-Error-Does-Not-Guarantee-Low-Regret)
15. [Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis](#Does-Graph-Prompt-Work-A-Data-Operation-Perspective-with-Theoretical-Analysis)
16. [Propagation of Chaos for Mean-Field Langevin Dynamics and its Application to Model Ensemble](#Propagation-of-Chaos-for-Mean-Field-Langevin-Dynamics-and-its-Application-to-Model-Ensemble)
17. [Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability](#Synthesizing-Images-on-Perceptual-Boundaries-of-ANNs-for-Uncovering-and-Manipulating-Human-Perceptual-Variability)
18. [Boosting Multi-Domain Fine-Tuning of Large Language Models through Evolving Interactions between Samples](#Boosting-Multi-Domain-Fine-Tuning-of-Large-Language-Models-through-Evolving-Interactions-between-Samples)
19. [Differential Privacy Under Class Imbalance: Methods and Empirical Insights](#Differential-Privacy-Under-Class-Imbalance-Methods-and-Empirical-Insights)
20. [RepLoRA: Reparameterizing Low-rank Adaptation via the Perspective of Mixture of Experts](#RepLoRA-Reparameterizing-Low-rank-Adaptation-via-the-Perspective-of-Mixture-of-Experts)
21. [SECOND: Mitigating Perceptual Hallucination in Vision-Language Models via Selective and Contrastive Decoding](#SECOND-Mitigating-Perceptual-Hallucination-in-Vision-Language-Models-via-Selective-and-Contrastive-Decoding)
22. [Time Series Representations with Hard-Coded Invariances](#Time-Series-Representations-with-Hard-Coded-Invariances)
23. [Finding Wasserstein Ball Center: Efficient Algorithm and The Applications in Fairness](#Finding-Wasserstein-Ball-Center-Efficient-Algorithm-and-The-Applications-in-Fairness)
24. [Average Certified Radius is a Poor Metric for Randomized Smoothing](#Average-Certified-Radius-is-a-Poor-Metric-for-Randomized-Smoothing)
25. [The Case for Learned Provenance-based System Behavior Baseline](#The-Case-for-Learned-Provenance-based-System-Behavior-Baseline)
26. [DocKS-RAG: Optimizing Document-Level Relation Extraction through LLM-Enhanced Hybrid Prompt Tuning](#DocKS-RAG-Optimizing-Document-Level-Relation-Extraction-through-LLM-Enhanced-Hybrid-Prompt-Tuning)
27. [Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment](#Make-LoRA-Great-Again-Boosting-LoRA-with-Adaptive-Singular-Values-and-Mixture-of-Experts-Optimization-Alignment)
28. [Stochastic Poisson Surface Reconstruction with One Solve using Geometric Gaussian Processes](#Stochastic-Poisson-Surface-Reconstruction-with-One-Solve-using-Geometric-Gaussian-Processes)
29. [Rectifying Conformity Scores for Better Conditional Coverage](#Rectifying-Conformity-Scores-for-Better-Conditional-Coverage)
30. [Robust Multimodal Large Language Models Against Modality Conflict](#Robust-Multimodal-Large-Language-Models-Against-Modality-Conflict)
31. [Learning Distances from Data with Normalizing Flows and Score Matching](#Learning-Distances-from-Data-with-Normalizing-Flows-and-Score-Matching)
32. [On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains](#On-the-Vulnerability-of-Applying-Retrieval-Augmented-Generation-within-Knowledge-Intensive-Application-Domains)
33. [DANCE: Dual Unbiased Expansion with Group-acquired Alignment for Out-of-distribution Graph Fairness Learning](#DANCE-Dual-Unbiased-Expansion-with-Group-acquired-Alignment-for-Out-of-distribution-Graph-Fairness-Learning)

---


## Tightening Causal Bounds via Covariate-Aware Optimal Transport

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

## Tightening Causal Bounds via Covariate-Aware Optimal Transport


### Images

![1836a6f96a8ca457d58b84fe023ff46664afbc46515f2ed51fe9ceab908a4da8.jpg](../icml_results/1901_Tightening%20Causal%20Bounds%20via%20Covariate-Aware%20Optimal%20Transport/images/1836a6f96a8ca457d58b84fe023ff46664afbc46515f2ed51fe9ceab908a4da8.jpg)

![2707fa0f900f452ed0f839e1a9662011ec635357d257827b60ff9e10ec7cb801.jpg](../icml_results/1901_Tightening%20Causal%20Bounds%20via%20Covariate-Aware%20Optimal%20Transport/images/2707fa0f900f452ed0f839e1a9662011ec635357d257827b60ff9e10ec7cb801.jpg)

![37baabfc0a5ed491463d08f32468d5b675c8c7f099c22ad95e17ae549ae7cda5.jpg](../icml_results/1901_Tightening%20Causal%20Bounds%20via%20Covariate-Aware%20Optimal%20Transport/images/37baabfc0a5ed491463d08f32468d5b675c8c7f099c22ad95e17ae549ae7cda5.jpg)

![6b5d39fa33daa2de25ae3b5f355b9f9a8f244b91b821375fb84fea0b0c45de48.jpg](../icml_results/1901_Tightening%20Causal%20Bounds%20via%20Covariate-Aware%20Optimal%20Transport/images/6b5d39fa33daa2de25ae3b5f355b9f9a8f244b91b821375fb84fea0b0c45de48.jpg)

### Tables

![596f8c84d7cfab5ba55a9e322c770839d751f77ad15e8d331682f4d00d61590c.jpg](../icml_results/1901_Tightening%20Causal%20Bounds%20via%20Covariate-Aware%20Optimal%20Transport/tables/596f8c84d7cfab5ba55a9e322c770839d751f77ad15e8d331682f4d00d61590c.jpg)

![bfd51196840fe0d8bb571af74ff6b3fb99e0d5f7059623f7fd5c816630761171.jpg](../icml_results/1901_Tightening%20Causal%20Bounds%20via%20Covariate-Aware%20Optimal%20Transport/tables/bfd51196840fe0d8bb571af74ff6b3fb99e0d5f7059623f7fd5c816630761171.jpg)

![fced4a5a437e31b34f3d201349f5ae04deae7e67dd26989103da8be708196a2c.jpg](../icml_results/1901_Tightening%20Causal%20Bounds%20via%20Covariate-Aware%20Optimal%20Transport/tables/fced4a5a437e31b34f3d201349f5ae04deae7e67dd26989103da8be708196a2c.jpg)

## Generative Audio Language Modeling with Continuous-valued Tokens and Masked Next-Token Prediction


### Images

![333618c713fa7c2d3f6e24c706000dd73a4c8773d3fd6a42c71aae7b2afe05c6.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/333618c713fa7c2d3f6e24c706000dd73a4c8773d3fd6a42c71aae7b2afe05c6.jpg)

![4a59915d0402afb85ab0c41163de8c94748fb16e0cfbd5c49dd9fc678f826934.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/4a59915d0402afb85ab0c41163de8c94748fb16e0cfbd5c49dd9fc678f826934.jpg)

![5f3a76d94c3a738ff374dc985264409e0057d0c3c584baf4a2ae7166ecf12c26.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/5f3a76d94c3a738ff374dc985264409e0057d0c3c584baf4a2ae7166ecf12c26.jpg)

![66be66d4c5d6772618b8243fd0696785065aa88a3f85027662e5164a9b6a409b.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/66be66d4c5d6772618b8243fd0696785065aa88a3f85027662e5164a9b6a409b.jpg)

![a3f898196f3f3e1d69805fcaab726e24577fd85b86ffe9309bb4f34cd0f7b82c.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/a3f898196f3f3e1d69805fcaab726e24577fd85b86ffe9309bb4f34cd0f7b82c.jpg)

![a622f268b6a7ef91e8ce36aef5517646a3376234256d440dddc4bb3bcf2ac080.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/a622f268b6a7ef91e8ce36aef5517646a3376234256d440dddc4bb3bcf2ac080.jpg)

![e5ba238ce009dc856cefafbf767d79415ab3e21fd587c2399dfe6022fefb7750.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/e5ba238ce009dc856cefafbf767d79415ab3e21fd587c2399dfe6022fefb7750.jpg)

![e820c72474b0f7486c1fce366392b09b85264f3c050e1bccfdce8a88f7950af0.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/images/e820c72474b0f7486c1fce366392b09b85264f3c050e1bccfdce8a88f7950af0.jpg)

### Tables

![03a25e8f3fab934cafd97df455ee7c7c22a72797aacd87b5cc2d167f05d5d439.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/03a25e8f3fab934cafd97df455ee7c7c22a72797aacd87b5cc2d167f05d5d439.jpg)

![03ed2dede9603476aa5ecc9ff05e0220a18bf98eef05fec0befd7150d1727793.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/03ed2dede9603476aa5ecc9ff05e0220a18bf98eef05fec0befd7150d1727793.jpg)

![34afcd6ad002a7d3bc209c7e24a96c2034ad06684279469ba117c2c68720b09e.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/34afcd6ad002a7d3bc209c7e24a96c2034ad06684279469ba117c2c68720b09e.jpg)

![5af44412320c34ad072b67fc249c612d7df2736c77259b0ce82a38ab815e1a1e.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/5af44412320c34ad072b67fc249c612d7df2736c77259b0ce82a38ab815e1a1e.jpg)

![6470d09666e7cfb4bf466db502de8e494e9a7023c57322930653d42b13c91220.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/6470d09666e7cfb4bf466db502de8e494e9a7023c57322930653d42b13c91220.jpg)

![8e0d27c32368c956d533abbf1f28c148f9366a2373607876a498d60c2da1b53f.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/8e0d27c32368c956d533abbf1f28c148f9366a2373607876a498d60c2da1b53f.jpg)

![b29bf4be7761fff8c6821faa13d10cc6eb1619dc92aaf3357275b9106e3ef837.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/b29bf4be7761fff8c6821faa13d10cc6eb1619dc92aaf3357275b9106e3ef837.jpg)

![ed393b677adec7522f9751ea2b5d3dd171f3407ac9aa76b22fdcd448f071f49d.jpg](../icml_results/1902_Generative%20Audio%20Language%20Modeling%20with%20Continuous-valued%20Tokens%20and%20Masked%20Next-Token%20Prediction/tables/ed393b677adec7522f9751ea2b5d3dd171f3407ac9aa76b22fdcd448f071f49d.jpg)

## ResKoopNet: Learning Koopman Representations for Complex Dynamics with Spectral Residuals


### Images

![07dfb8aa1f0f777d4ef86099665a18fdb1c302e5483eb5ebda213aef6cf4b0ef.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/07dfb8aa1f0f777d4ef86099665a18fdb1c302e5483eb5ebda213aef6cf4b0ef.jpg)

![0977fbc86da220501364a20adf910ed4d19a3a1ebabaa15a1a46b687ab1b9401.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/0977fbc86da220501364a20adf910ed4d19a3a1ebabaa15a1a46b687ab1b9401.jpg)

![12381d9ced7e8b76bc15b604db33e87186a42272059abbb9afd5c6df3d1347ae.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/12381d9ced7e8b76bc15b604db33e87186a42272059abbb9afd5c6df3d1347ae.jpg)

![199e8edc969e1298d919aea7164dd79be597ab1bdc37f104a129f7dfb57b1d2a.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/199e8edc969e1298d919aea7164dd79be597ab1bdc37f104a129f7dfb57b1d2a.jpg)

![48f1c97756b73140f1fe5b0d92f03335420b5ff08c0b2762f3aa1fc1d426e228.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/48f1c97756b73140f1fe5b0d92f03335420b5ff08c0b2762f3aa1fc1d426e228.jpg)

![4b158bb663331fda3dda2fc3f724249b6f38a297ae6b34ed21b4f01426caf780.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/4b158bb663331fda3dda2fc3f724249b6f38a297ae6b34ed21b4f01426caf780.jpg)

![4d53b433214be22956935ba9d09a097eb29ce5b14caa30c966506a888202ceda.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/4d53b433214be22956935ba9d09a097eb29ce5b14caa30c966506a888202ceda.jpg)

![57e41980f5c8b62785827ef16c6a730a0b673a46c37a814249d7de1d131f2f9f.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/57e41980f5c8b62785827ef16c6a730a0b673a46c37a814249d7de1d131f2f9f.jpg)

![674d10c97890b0e512d4b2229b1d04aca4b0d842df8ff82e08b41229aec51794.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/674d10c97890b0e512d4b2229b1d04aca4b0d842df8ff82e08b41229aec51794.jpg)

![6cf605cfa17478fe29a1d79a41d3d9a336f150df20fda23d53c854d574452784.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/6cf605cfa17478fe29a1d79a41d3d9a336f150df20fda23d53c854d574452784.jpg)

![7909f82665fd7917a9b8cc97a47d4b625869ee255ca229dd23001ec4c51cc6fe.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/7909f82665fd7917a9b8cc97a47d4b625869ee255ca229dd23001ec4c51cc6fe.jpg)

![7efe687b290cf15792219f968dd6e90319487a9524584ac24c44110f0c01e0b8.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/7efe687b290cf15792219f968dd6e90319487a9524584ac24c44110f0c01e0b8.jpg)

![8db0d1c595804a72d09edf8f4a6fea00f78321e2de770b609b04a418f27c747b.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/8db0d1c595804a72d09edf8f4a6fea00f78321e2de770b609b04a418f27c747b.jpg)

![9c5480860d9f92adcb05bbbbe8b4ecf5a83bf78bfc4a54871353a8483a727990.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/9c5480860d9f92adcb05bbbbe8b4ecf5a83bf78bfc4a54871353a8483a727990.jpg)

![a12ebd6c1b235a6407af3e9da13acf669a32358b2aa34af71c629fafc33ef120.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/a12ebd6c1b235a6407af3e9da13acf669a32358b2aa34af71c629fafc33ef120.jpg)

![c13d7b7249bf096de55e7a5f2f3daf7825f7a704deb8321c59b53ef3e79530a5.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/c13d7b7249bf096de55e7a5f2f3daf7825f7a704deb8321c59b53ef3e79530a5.jpg)

![dc5054dd171e5b1904329d4bb7eabc3732fb79d0257eb802ea92afb5c07e9a66.jpg](../icml_results/1903_ResKoopNet_%20Learning%20Koopman%20Representations%20for%20Complex%20Dynamics%20with%20Spectral%20Residuals/images/dc5054dd171e5b1904329d4bb7eabc3732fb79d0257eb802ea92afb5c07e9a66.jpg)

## Dialogue Without Limits: Constant-Sized KV Caches for Extended Response in LLMs


### Images

![1a2f509667dd18d3fcd18ea3bc543c46d4220526a353a2ea1b4bacc9e356ab2c.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/1a2f509667dd18d3fcd18ea3bc543c46d4220526a353a2ea1b4bacc9e356ab2c.jpg)

![1ed495d1fe581a69de737247a614cac4e0d8bacf2c0913f84b6926c85d1b1205.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/1ed495d1fe581a69de737247a614cac4e0d8bacf2c0913f84b6926c85d1b1205.jpg)

![5ff322b670c5872da2578cfe973c837551e5edcb7026f94469a38b199f709a2f.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/5ff322b670c5872da2578cfe973c837551e5edcb7026f94469a38b199f709a2f.jpg)

![7134ec464a3cfc42b549713c54007f6f97acabe3b4659e206f56703a741afc4d.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/7134ec464a3cfc42b549713c54007f6f97acabe3b4659e206f56703a741afc4d.jpg)

![80291dfd9e165a332e34d2ed9375de31b15fa5dfe8197e93847a8fe1103043b5.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/80291dfd9e165a332e34d2ed9375de31b15fa5dfe8197e93847a8fe1103043b5.jpg)

![86eb836719b35206e39475eb9d1b3ccdc7758eea20602063802cb25b0ae1db92.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/86eb836719b35206e39475eb9d1b3ccdc7758eea20602063802cb25b0ae1db92.jpg)

![94d72b28e14ac512b20bad8005cd4cac0625262329909d969a0e03d9cb8766b7.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/94d72b28e14ac512b20bad8005cd4cac0625262329909d969a0e03d9cb8766b7.jpg)

![986aeaa9f13ac281dd17736f3b26895c1d2501bbd3f8891f2fd1af087334db7c.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/986aeaa9f13ac281dd17736f3b26895c1d2501bbd3f8891f2fd1af087334db7c.jpg)

![b1265c43b701cc1c35325e63563d9e6d804c6781666b31f765431e74b1027f18.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/b1265c43b701cc1c35325e63563d9e6d804c6781666b31f765431e74b1027f18.jpg)

![cf10cf2b8d24a9147a3d0cd4da13e0db1bcca489ff2a50e5f48ecc0b76221387.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/images/cf10cf2b8d24a9147a3d0cd4da13e0db1bcca489ff2a50e5f48ecc0b76221387.jpg)

### Tables

![0adb15813a2183a4d0086b209304b2bf39f6e8d5f5c8b7e8069312363467bc97.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/0adb15813a2183a4d0086b209304b2bf39f6e8d5f5c8b7e8069312363467bc97.jpg)

![0cd1692ffbc95f99c6600c43eda8447aa15b327204a12c468ec2e5a77759b34c.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/0cd1692ffbc95f99c6600c43eda8447aa15b327204a12c468ec2e5a77759b34c.jpg)

![5d175ad101eb4f4cada31763711e8664acf22107832188d7aeda3e5c4a946f9c.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/5d175ad101eb4f4cada31763711e8664acf22107832188d7aeda3e5c4a946f9c.jpg)

![84f6c922e07c3fa539945733ac635718973d62409718380ee67fe79a9640420d.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/84f6c922e07c3fa539945733ac635718973d62409718380ee67fe79a9640420d.jpg)

![88c862168706f6e3790dd0e4c4e418d7c0900f8df317875b999a3b3776b1752c.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/88c862168706f6e3790dd0e4c4e418d7c0900f8df317875b999a3b3776b1752c.jpg)

![8a72ea1dc967acc42aa7d355599e0ab3cd9006c89dd0b6991b9ba5683498dca1.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/8a72ea1dc967acc42aa7d355599e0ab3cd9006c89dd0b6991b9ba5683498dca1.jpg)

![8f0d066c8eb09b2e9dd1c61e7d49d12d3e0980c481971c94fd150d46b417ff45.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/8f0d066c8eb09b2e9dd1c61e7d49d12d3e0980c481971c94fd150d46b417ff45.jpg)

![a30c8a013bc0d8af6cc6d7519ae6ba14d683332e5524d43e85670d61085a387f.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/a30c8a013bc0d8af6cc6d7519ae6ba14d683332e5524d43e85670d61085a387f.jpg)

![a77b3a7aadccde4ef610d432153386a03958e56cfe8bf8b1a02904df4ec5a802.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/a77b3a7aadccde4ef610d432153386a03958e56cfe8bf8b1a02904df4ec5a802.jpg)

![ea8019b2fcdbe1e5565633b7db75df900d8982e85f4e9a0603196a54763e10eb.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/ea8019b2fcdbe1e5565633b7db75df900d8982e85f4e9a0603196a54763e10eb.jpg)

![f5bc5b47db379af9c2ea8e95c014eadb7c70c74d8a222993b2aed9e2354dce8f.jpg](../icml_results/1904_Dialogue%20Without%20Limits_%20Constant-Sized%20KV%20Caches%20for%20Extended%20Response%20in%20LLMs/tables/f5bc5b47db379af9c2ea8e95c014eadb7c70c74d8a222993b2aed9e2354dce8f.jpg)

## Tree-Sliced Wasserstein Distance: A Geometric Perspective


### Images

![1a4b62917ba5cdc92acacc08384dddbc91909bfeea68cc2451619488466b877b.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/1a4b62917ba5cdc92acacc08384dddbc91909bfeea68cc2451619488466b877b.jpg)

![3defd2d9e38bb3fc6240b0cd11b26a29c925578ead98202139728a6def81d534.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/3defd2d9e38bb3fc6240b0cd11b26a29c925578ead98202139728a6def81d534.jpg)

![4b3cf0bf7b7798dfe8076c95ed1d720c14b6eb64bece6c4986c05163dc983379.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/4b3cf0bf7b7798dfe8076c95ed1d720c14b6eb64bece6c4986c05163dc983379.jpg)

![58cf2518b2bcfedbbdfb54e5d496c58c13a728dec445756fd4dcce5f7a9a4eaa.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/58cf2518b2bcfedbbdfb54e5d496c58c13a728dec445756fd4dcce5f7a9a4eaa.jpg)

![66c89570f0548c0fea4c4eca6e04cefff414fe9b9a7202181d9c3e52f3952bbf.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/66c89570f0548c0fea4c4eca6e04cefff414fe9b9a7202181d9c3e52f3952bbf.jpg)

![75bae7b510216679bb4e24970d735dd04e89c628e4494c3a16a064c6b2323fd8.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/75bae7b510216679bb4e24970d735dd04e89c628e4494c3a16a064c6b2323fd8.jpg)

![770ed46cb2b2fdf10e114f9367162d3a0c2726c42277c3467cd67d14b39e1619.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/770ed46cb2b2fdf10e114f9367162d3a0c2726c42277c3467cd67d14b39e1619.jpg)

![8c0c13512a3e4a8ad4a11857896fa294cd003587c5feedc043de0fd0c6d3dbd3.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/8c0c13512a3e4a8ad4a11857896fa294cd003587c5feedc043de0fd0c6d3dbd3.jpg)

![a1448f9897f26c90d381171511886750b1800e743e0a3a87faa42da14d52044d.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/a1448f9897f26c90d381171511886750b1800e743e0a3a87faa42da14d52044d.jpg)

![b1d0bb6888b041081c7bdede84247b669c874bf9a70c648a93b72c57f2f65f7f.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/b1d0bb6888b041081c7bdede84247b669c874bf9a70c648a93b72c57f2f65f7f.jpg)

![d2dfc2be77fd8d9e05a3ccd06bc7769246a30a3702244549425cd145b93424c4.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/d2dfc2be77fd8d9e05a3ccd06bc7769246a30a3702244549425cd145b93424c4.jpg)

![dafc9f93fcd3b0c8ecba17c36d59567e6b9f5a34300e07fde31d01fd4ce27a14.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/images/dafc9f93fcd3b0c8ecba17c36d59567e6b9f5a34300e07fde31d01fd4ce27a14.jpg)

### Tables

![2299dc0670fb9c9dd90d424693099dc1fcd4e4bd290b6e24618afc46623f3583.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/tables/2299dc0670fb9c9dd90d424693099dc1fcd4e4bd290b6e24618afc46623f3583.jpg)

![4bbeec09ec5f86b2f957608b5099ee778640212c982d3e15602be6586e56728e.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/tables/4bbeec09ec5f86b2f957608b5099ee778640212c982d3e15602be6586e56728e.jpg)

![8b357ca460ea5b852c471fd8ebdea097a7a932541920c8574231a11c6638ad6d.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/tables/8b357ca460ea5b852c471fd8ebdea097a7a932541920c8574231a11c6638ad6d.jpg)

![910a95be0594bb6a10d45bec81c8da854671aff274b55bf6105d1602c360f8f6.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/tables/910a95be0594bb6a10d45bec81c8da854671aff274b55bf6105d1602c360f8f6.jpg)

![b445664215ddc0b4ff9475ebbd76a5b9c38ffe96249a43c24bedc7f6375803be.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/tables/b445664215ddc0b4ff9475ebbd76a5b9c38ffe96249a43c24bedc7f6375803be.jpg)

![b7f115433bf77a8521cfee44c58fa55a3e830d6ffdf31f2d77e8aa83ad5bea53.jpg](../icml_results/1905_Tree-Sliced%20Wasserstein%20Distance_%20A%20Geometric%20Perspective/tables/b7f115433bf77a8521cfee44c58fa55a3e830d6ffdf31f2d77e8aa83ad5bea53.jpg)

## When Bad Data Leads to Good Models


### Images

![019c4922a821f93db3bd9520fdc1896b8ddd4bef2d060ed64a05b29e1ef1da6e.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/019c4922a821f93db3bd9520fdc1896b8ddd4bef2d060ed64a05b29e1ef1da6e.jpg)

![0d5c49e914271d62b7ae6202c57cce692b2210b62113a17ff918b38eebeb2c38.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/0d5c49e914271d62b7ae6202c57cce692b2210b62113a17ff918b38eebeb2c38.jpg)

![1937fcce4bcf1acc0ae1a3c2e634fb1c1a4a515adb87f6b8202dd908a8022f70.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/1937fcce4bcf1acc0ae1a3c2e634fb1c1a4a515adb87f6b8202dd908a8022f70.jpg)

![6e2113b57f6b2975d9eed7cc2e4adb295fab3ae2cce5149fce2ae5d1ed40a4e0.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/6e2113b57f6b2975d9eed7cc2e4adb295fab3ae2cce5149fce2ae5d1ed40a4e0.jpg)

![7e634fdf50d876c0d74e9ba75b8de9f7971c9e65204f1e3497a16c1cd5b40fc2.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/7e634fdf50d876c0d74e9ba75b8de9f7971c9e65204f1e3497a16c1cd5b40fc2.jpg)

![8fcdb56dced9666f3b3b0277ddcef2bd100dabbaba50864f42baf18c14e8c7a0.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/8fcdb56dced9666f3b3b0277ddcef2bd100dabbaba50864f42baf18c14e8c7a0.jpg)

![cd306c5354246255e8d93f3e51a2a20db6b49808a1e8188de3547ade87472fda.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/cd306c5354246255e8d93f3e51a2a20db6b49808a1e8188de3547ade87472fda.jpg)

![f6594fee861a17a94e9eddd6209e15214c1cb0e1352366f73d7777d0d96318fe.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/images/f6594fee861a17a94e9eddd6209e15214c1cb0e1352366f73d7777d0d96318fe.jpg)

### Tables

![0e96c2eee1db2e5cd51c4249459d5a9b845d0cf19c5fad4f30730220ac1cb4e9.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/tables/0e96c2eee1db2e5cd51c4249459d5a9b845d0cf19c5fad4f30730220ac1cb4e9.jpg)

![406b3fd99746325a31e36d52b64fb00a70cc31518aaae3b8918752da93cb0a52.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/tables/406b3fd99746325a31e36d52b64fb00a70cc31518aaae3b8918752da93cb0a52.jpg)

![9e7f947e0948a08ffc138fb8882983bfc481a1b8b38f2eeb84a05a4f8e6dccce.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/tables/9e7f947e0948a08ffc138fb8882983bfc481a1b8b38f2eeb84a05a4f8e6dccce.jpg)

![e706ee8733960c8e842f53ad12b7dff5b4b407045c23768937a801ecbefc1e64.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/tables/e706ee8733960c8e842f53ad12b7dff5b4b407045c23768937a801ecbefc1e64.jpg)

![ea4216bbb80c6c7fbca45fd44342475e85445f86d4f1f89d4e51e21690acafcd.jpg](../icml_results/1906_When%20Bad%20Data%20Leads%20to%20Good%20Models/tables/ea4216bbb80c6c7fbca45fd44342475e85445f86d4f1f89d4e51e21690acafcd.jpg)

## A Physics-Informed Machine Learning Framework for Safe and Optimal Control of Autonomous Systems


### Images

![168163c3c2997064d8c1fb2665debdbe49bcc90fe14fcd07ea3f2099caca9d43.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/168163c3c2997064d8c1fb2665debdbe49bcc90fe14fcd07ea3f2099caca9d43.jpg)

![1c6f1754c84f229cc38258e9a4859bb097c768d2d1b36f7b12990823a46f8132.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/1c6f1754c84f229cc38258e9a4859bb097c768d2d1b36f7b12990823a46f8132.jpg)

![3ba4be5b5f5f899dff11c28aa0105895df7740ee760ae1cbbff3ad6a5e818abf.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/3ba4be5b5f5f899dff11c28aa0105895df7740ee760ae1cbbff3ad6a5e818abf.jpg)

![417afc9177ca183f50aabe244e174ab201b00eca25d1d83aee8bf4de381a010d.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/417afc9177ca183f50aabe244e174ab201b00eca25d1d83aee8bf4de381a010d.jpg)

![58f54dfb2de51489baadc9aca4fc44100c3715af3f7a55d1e185a009de51c7a8.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/58f54dfb2de51489baadc9aca4fc44100c3715af3f7a55d1e185a009de51c7a8.jpg)

![79c53136517570be612b481952245f90976af15b9f985c4c82db0f8f85aa9fb7.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/79c53136517570be612b481952245f90976af15b9f985c4c82db0f8f85aa9fb7.jpg)

![a2577c51a7af1094c138bd2795fdf16c090f27b58ae7f2b250e451561fa7fe53.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/a2577c51a7af1094c138bd2795fdf16c090f27b58ae7f2b250e451561fa7fe53.jpg)

![aaa68a3fdb19a8d4055e0072d4699d6bf293ec64a6bb7f372bd22afd28fd2e3e.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/aaa68a3fdb19a8d4055e0072d4699d6bf293ec64a6bb7f372bd22afd28fd2e3e.jpg)

![c95b0f647011a8b7ebeb98ae7772c882cb6ce60b960e590aaa224b22dd779eb7.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/c95b0f647011a8b7ebeb98ae7772c882cb6ce60b960e590aaa224b22dd779eb7.jpg)

![cb2aaa64cf9960204593632c43c0f253e9dd9193b845a9011525462f3a961d02.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/cb2aaa64cf9960204593632c43c0f253e9dd9193b845a9011525462f3a961d02.jpg)

![cf62950d5b4bd1f940dd0cd4444b5206439d75b949833ded91d83a113fa7fde6.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/cf62950d5b4bd1f940dd0cd4444b5206439d75b949833ded91d83a113fa7fde6.jpg)

![da0598f7919a04ea861993442099024d90ae6a2bb32a2e68a6cb0158fc6e64a1.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/images/da0598f7919a04ea861993442099024d90ae6a2bb32a2e68a6cb0158fc6e64a1.jpg)

### Tables

![0cee057cb04d20f425a45e1ad0ec562d4d11a1828b45039b2766bf9841b29ced.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/tables/0cee057cb04d20f425a45e1ad0ec562d4d11a1828b45039b2766bf9841b29ced.jpg)

![242642fabaf3b64c051d07e8fb72f52361bc7c0fad601be2576d05eed818d787.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/tables/242642fabaf3b64c051d07e8fb72f52361bc7c0fad601be2576d05eed818d787.jpg)

![8c11ae4f52161fb8942c42656a50a071fe2908d916d0b48106b0c571847bceaf.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/tables/8c11ae4f52161fb8942c42656a50a071fe2908d916d0b48106b0c571847bceaf.jpg)

![c28058807b622c1b133260e0d119a644df0669bd097048edf953af5885bb9a06.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/tables/c28058807b622c1b133260e0d119a644df0669bd097048edf953af5885bb9a06.jpg)

![febee23fcd574669f85d5cef04234b6e2d0ac7a4aa8a4f49242de23c5656a17b.jpg](../icml_results/1907_A%20Physics-Informed%20Machine%20Learning%20Framework%20for%20Safe%20and%20Optimal%20Control%20of%20Autonomous%20Systems/tables/febee23fcd574669f85d5cef04234b6e2d0ac7a4aa8a4f49242de23c5656a17b.jpg)

## Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts


### Images

![41b8acc1d6a4dcfb30da62da603b0f75d8da4d2ea7be7732a5208cd4efdb6219.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/41b8acc1d6a4dcfb30da62da603b0f75d8da4d2ea7be7732a5208cd4efdb6219.jpg)

![554957b90d1f7507ef52f38f7edd00b7d25b2a65850e85ae0014b4b58744a89f.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/554957b90d1f7507ef52f38f7edd00b7d25b2a65850e85ae0014b4b58744a89f.jpg)

![6dcbc9ce81d12d4e272157e53a26a9c7d56e51c0d936c34461382a5bfdfa8366.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/6dcbc9ce81d12d4e272157e53a26a9c7d56e51c0d936c34461382a5bfdfa8366.jpg)

![7aedea68040d7f1d144a54c8929865cf23e3e2be72b687443e4adc536fb6c960.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/7aedea68040d7f1d144a54c8929865cf23e3e2be72b687443e4adc536fb6c960.jpg)

![8f555c07bd590ec41f532ba5fdbcaa6f2679c13ede0d6f5b2c846b2aac009236.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/8f555c07bd590ec41f532ba5fdbcaa6f2679c13ede0d6f5b2c846b2aac009236.jpg)

![940139688818834745cd5ddb08ee101318614413fad0e25360626f2732eed802.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/940139688818834745cd5ddb08ee101318614413fad0e25360626f2732eed802.jpg)

![a0f0110aaf78b757bdacaa6ba991352c43f198cce495074ba7e623b7daa1c446.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/a0f0110aaf78b757bdacaa6ba991352c43f198cce495074ba7e623b7daa1c446.jpg)

![bce32e3ae0f0b56c1d6f2b4614e19e51d727350649203137d42fbf316b6619c4.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/bce32e3ae0f0b56c1d6f2b4614e19e51d727350649203137d42fbf316b6619c4.jpg)

![c45051e83abb064486bf8c1d6798c82c5b65aebcee696505f835f2fb53413584.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/c45051e83abb064486bf8c1d6798c82c5b65aebcee696505f835f2fb53413584.jpg)

![c6c6e9f077ac531ac4267e39af3556684fea76b6d7a3fb05b477f2f0e2b38169.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/c6c6e9f077ac531ac4267e39af3556684fea76b6d7a3fb05b477f2f0e2b38169.jpg)

![c998509acef7176f4285280493cef033eb175af7c29af83677433efcf0611bb2.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/c998509acef7176f4285280493cef033eb175af7c29af83677433efcf0611bb2.jpg)

![dc082c9f33151a71e4f5939d46391b5fa7f0b8ce3cb2315685f1220a288a9506.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/dc082c9f33151a71e4f5939d46391b5fa7f0b8ce3cb2315685f1220a288a9506.jpg)

![e4ca721138c0cd000b6a2519266aef6e5b1f5f9f0bff1b3006e54e7e5040c187.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/images/e4ca721138c0cd000b6a2519266aef6e5b1f5f9f0bff1b3006e54e7e5040c187.jpg)

### Tables

![5f414c9a5d897c97b8880758a938a654dc8faf06536a6bcac20fb95aa0e54a3b.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/5f414c9a5d897c97b8880758a938a654dc8faf06536a6bcac20fb95aa0e54a3b.jpg)

![6236f1595f8b70e97e61ce936d0ac614ef2d5c52185a6e9dea18e751936b2dbe.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/6236f1595f8b70e97e61ce936d0ac614ef2d5c52185a6e9dea18e751936b2dbe.jpg)

![866daa3541b0643087052faae57e0ca40a4be65b62734bfef8221767be954d24.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/866daa3541b0643087052faae57e0ca40a4be65b62734bfef8221767be954d24.jpg)

![a138063d01a9d7f3ada4ec18d5ebb6e1b867d7f5d1852fc1742dab33de183a83.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/a138063d01a9d7f3ada4ec18d5ebb6e1b867d7f5d1852fc1742dab33de183a83.jpg)

![a30c8d7f9d0ec6f66fc6b63279661ee2556af4db5825ea3909b28e7d9ad1ebab.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/a30c8d7f9d0ec6f66fc6b63279661ee2556af4db5825ea3909b28e7d9ad1ebab.jpg)

![a65520704efa30bf1b1366fd90bcc594037b655f115792ba9b3e985662bd619f.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/a65520704efa30bf1b1366fd90bcc594037b655f115792ba9b3e985662bd619f.jpg)

![aa29dc0e4272a5428543c7c3d89cd5f7ffb2e98d365919b9853d80f94151a4fa.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/aa29dc0e4272a5428543c7c3d89cd5f7ffb2e98d365919b9853d80f94151a4fa.jpg)

![b3a2bfdc6917b01f056d1a4f8a1c0084a93924329b29c7ea70ea1c3c30637cbe.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/b3a2bfdc6917b01f056d1a4f8a1c0084a93924329b29c7ea70ea1c3c30637cbe.jpg)

![d09a48b9bcd5bedba591693ed0aa86adb94ac5c9bc81a679840d0441d53ec516.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/d09a48b9bcd5bedba591693ed0aa86adb94ac5c9bc81a679840d0441d53ec516.jpg)

![d24f2e096d7bd7fe74fa0233ff22cb28fba621212781ce72eb24d036fcb1a666.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/d24f2e096d7bd7fe74fa0233ff22cb28fba621212781ce72eb24d036fcb1a666.jpg)

![db052cb67cec5154b6306d6b34dc64ed33eeb3066052b73b4ec1e090aa34d3e3.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/db052cb67cec5154b6306d6b34dc64ed33eeb3066052b73b4ec1e090aa34d3e3.jpg)

![e4c26ad3ec9d558ccd069162e44ab88878feff47ac2db515f1e22d04cb55ad87.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/e4c26ad3ec9d558ccd069162e44ab88878feff47ac2db515f1e22d04cb55ad87.jpg)

![e56f6c79b6ae5401f51c4d1fb3500110cf8821c83489d673bdaee7468327da1a.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/e56f6c79b6ae5401f51c4d1fb3500110cf8821c83489d673bdaee7468327da1a.jpg)

![f03099c451a2a71750b6f544e4bd15356f7b704d902d50925a83aa535a78e79d.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/f03099c451a2a71750b6f544e4bd15356f7b704d902d50925a83aa535a78e79d.jpg)

![ff31a18d6648460290b19a697e3b0b0c441da5f6507c7568f66a76f4e2546fde.jpg](../icml_results/1908_Moirai-MoE_%20Empowering%20Time%20Series%20Foundation%20Models%20with%20Sparse%20Mixture%20of%20Experts/tables/ff31a18d6648460290b19a697e3b0b0c441da5f6507c7568f66a76f4e2546fde.jpg)

## Towards Theoretical Understanding of Sequential Decision Making with Preference Feedback


### Images

![5fa9e3abcd2b0a70b5f2ec964ba6f1edb2cd9eea623f7e4cddaba7319e624da7.jpg](../icml_results/1909_Towards%20Theoretical%20Understanding%20of%20Sequential%20Decision%20Making%20with%20Preference%20Feedback/images/5fa9e3abcd2b0a70b5f2ec964ba6f1edb2cd9eea623f7e4cddaba7319e624da7.jpg)

## Regret-Free Reinforcement Learning for Temporal Logic Specifications


### Images

![0cbe517d9405f1d8889281dab857ee364b7cd769923e07551d04370cba0e7974.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/images/0cbe517d9405f1d8889281dab857ee364b7cd769923e07551d04370cba0e7974.jpg)

![12986b0025b855f9690ffcf642544b7e8f5641a4355f99dfad562fa96241f74b.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/images/12986b0025b855f9690ffcf642544b7e8f5641a4355f99dfad562fa96241f74b.jpg)

![237e62c76380054cccf06544f223a5ce2959b2327efc7509dcbea39422b1ed54.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/images/237e62c76380054cccf06544f223a5ce2959b2327efc7509dcbea39422b1ed54.jpg)

![5f362b44c53f660af0313208da02d7e0a527ab0a652844618d70b2d852302522.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/images/5f362b44c53f660af0313208da02d7e0a527ab0a652844618d70b2d852302522.jpg)

![7d389a538cc15a951f054ed7c339d1f0f31166bf9c5d2b586fd49eb32bef329c.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/images/7d389a538cc15a951f054ed7c339d1f0f31166bf9c5d2b586fd49eb32bef329c.jpg)

![9511a5496859f2870443ca85c7b94e07688baa158c8f079e1e7424075be17d43.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/images/9511a5496859f2870443ca85c7b94e07688baa158c8f079e1e7424075be17d43.jpg)

### Tables

![0e965f3c6a8d6d2e662057c2b78d8e6a357ef30efad5da7d024c2c1e74adcc8a.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/tables/0e965f3c6a8d6d2e662057c2b78d8e6a357ef30efad5da7d024c2c1e74adcc8a.jpg)

![4ca3a01548ea5043ce5ec0fc747c1a4fc51fbc7bfc0b45b3e4b026f116589d65.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/tables/4ca3a01548ea5043ce5ec0fc747c1a4fc51fbc7bfc0b45b3e4b026f116589d65.jpg)

![74309b5e738f49c0a448236879ff0f739961d81cb9de49c52d56450ef45d20ee.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/tables/74309b5e738f49c0a448236879ff0f739961d81cb9de49c52d56450ef45d20ee.jpg)

![a78f331247473075c6d5e360ada14f27934921083255f1c364a5553eadf64538.jpg](../icml_results/1910_Regret-Free%20Reinforcement%20Learning%20for%20Temporal%20Logic%20Specifications/tables/a78f331247473075c6d5e360ada14f27934921083255f1c364a5553eadf64538.jpg)

## LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces


### Images

![0ca1780e117f680462594b053e61a65879be5e44d92406399a07d3608d32093a.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/0ca1780e117f680462594b053e61a65879be5e44d92406399a07d3608d32093a.jpg)

![0d61f5a54bee209073a4d1d20e163ad4f63d3d78484c956f89b356100d66d9d6.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/0d61f5a54bee209073a4d1d20e163ad4f63d3d78484c956f89b356100d66d9d6.jpg)

![152e8b7342f9ecccea09ac4b463aece0d542b7119c5d0eeec3fb2ca784f3238c.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/152e8b7342f9ecccea09ac4b463aece0d542b7119c5d0eeec3fb2ca784f3238c.jpg)

![197d8785f64927eeedbf6049a2f853afbeb29720091dabba4a3e12bab420459a.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/197d8785f64927eeedbf6049a2f853afbeb29720091dabba4a3e12bab420459a.jpg)

![43360f0737804f0488ea1ba6405d73117d42e96d2687d8ffcff96c34d5c63f9d.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/43360f0737804f0488ea1ba6405d73117d42e96d2687d8ffcff96c34d5c63f9d.jpg)

![4f5cfaca504b027fb66f66643afa37f2857d3ab27f661dfa48addcc48a517259.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/4f5cfaca504b027fb66f66643afa37f2857d3ab27f661dfa48addcc48a517259.jpg)

![4fba0a1cc929e4e629034cb77bf2544f9935e5591ad73c14d548ab1eddd0eaab.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/4fba0a1cc929e4e629034cb77bf2544f9935e5591ad73c14d548ab1eddd0eaab.jpg)

![537d078e4fe1455fb29d947f4bd0fccc7a9d502a07e3b41ec657191cf11c2db6.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/537d078e4fe1455fb29d947f4bd0fccc7a9d502a07e3b41ec657191cf11c2db6.jpg)

![67eb1bf74eb0443f4e4ac7015fc0b515a15dd92c27cc115a4a16235812ba64e6.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/67eb1bf74eb0443f4e4ac7015fc0b515a15dd92c27cc115a4a16235812ba64e6.jpg)

![7aa0ccd9791a68e953eeb2e956fe0159808d45ee0277a17ba587162e51aa4fc3.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/7aa0ccd9791a68e953eeb2e956fe0159808d45ee0277a17ba587162e51aa4fc3.jpg)

![8747320bdeeb24942bd6db6920f0c2e79936ed8b1a7dfe9105423ccda288c3f1.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/8747320bdeeb24942bd6db6920f0c2e79936ed8b1a7dfe9105423ccda288c3f1.jpg)

![8ed9216e675ab4caf3d6bf381155947d898df9610c7d8ad0f13cd88abce91c3c.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/8ed9216e675ab4caf3d6bf381155947d898df9610c7d8ad0f13cd88abce91c3c.jpg)

![8f06c4b07ff9597b289288013c67aff649bad65649a4500c2c9cafc6346b5a0f.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/8f06c4b07ff9597b289288013c67aff649bad65649a4500c2c9cafc6346b5a0f.jpg)

![9bf646df29d1919fbe1e608e65c0d4fbab372e1c68b5a658b4c8b7d3db0db2b4.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/9bf646df29d1919fbe1e608e65c0d4fbab372e1c68b5a658b4c8b7d3db0db2b4.jpg)

![c486e6845a95de9db3c19024723c2a1610e5907245a446a92acac8a02ad3285b.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/c486e6845a95de9db3c19024723c2a1610e5907245a446a92acac8a02ad3285b.jpg)

![c60281b002335dfa9a9140d088b6c419d4896122702ed8fc112906e6f19ef439.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/c60281b002335dfa9a9140d088b6c419d4896122702ed8fc112906e6f19ef439.jpg)

![d90daa50ab41399d721291e4a86f614bf90d1a29a63026a2a02383c9a5d66046.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/d90daa50ab41399d721291e4a86f614bf90d1a29a63026a2a02383c9a5d66046.jpg)

![dc60a112c5a4ea8af5d6ccc5c0e9ac2f6319be3085e4c8baa855b81987436302.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/dc60a112c5a4ea8af5d6ccc5c0e9ac2f6319be3085e4c8baa855b81987436302.jpg)

![ed5b3fadbb859fa2bdc4b9dbb7a13934c84a21a8a645e52f9e2358b3d5999abf.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/ed5b3fadbb859fa2bdc4b9dbb7a13934c84a21a8a645e52f9e2358b3d5999abf.jpg)

![f747a69d1866be16acd18122837a7e1a1033a98f63fa652345cf0f8b3fd07974.jpg](../icml_results/1911_LIVS_%20A%20Pluralistic%20Alignment%20Dataset%20for%20Inclusive%20Public%20Spaces/images/f747a69d1866be16acd18122837a7e1a1033a98f63fa652345cf0f8b3fd07974.jpg)

## Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models


### Images

![2368efcaef34ccabd6d35f8d5b865422f81819def105e774831a779cc14de85c.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/2368efcaef34ccabd6d35f8d5b865422f81819def105e774831a779cc14de85c.jpg)

![29b20a8c50b1c5092a5aba57bade5a6c465055e463ef173efd75e8728c2eb279.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/29b20a8c50b1c5092a5aba57bade5a6c465055e463ef173efd75e8728c2eb279.jpg)

![2d3ce133d9fb8c210eb216cef1b26d64c4819ada8fd83698ffaa8a4216c5136d.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/2d3ce133d9fb8c210eb216cef1b26d64c4819ada8fd83698ffaa8a4216c5136d.jpg)

![552072c22b0f81a7a19ddbd932772c91e0d9e368db0809cad86d569c3244e293.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/552072c22b0f81a7a19ddbd932772c91e0d9e368db0809cad86d569c3244e293.jpg)

![69a11c7fd981fd8c33cf3aca546604d8d614af5cca1dd3d0cb9b42ff4a06d551.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/69a11c7fd981fd8c33cf3aca546604d8d614af5cca1dd3d0cb9b42ff4a06d551.jpg)

![6dc5b35f3e3ab468e22ea4ca4a66c7fe0afa7e1685d4b99dc741ead953cdf4e1.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/6dc5b35f3e3ab468e22ea4ca4a66c7fe0afa7e1685d4b99dc741ead953cdf4e1.jpg)

![a05b2150f732acfeef3681f07d7795d1a5d47a066b4d4a92ebd35bceb8173a79.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/a05b2150f732acfeef3681f07d7795d1a5d47a066b4d4a92ebd35bceb8173a79.jpg)

![d03f13af9af8a64251fceee9c527493ddbaaacb9b1cee2b2bdac929bf358c651.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/d03f13af9af8a64251fceee9c527493ddbaaacb9b1cee2b2bdac929bf358c651.jpg)

![f29991ff6975ce50c69ba4b836ad4a265df87caa8ce032adaeb641188376f4e7.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/f29991ff6975ce50c69ba4b836ad4a265df87caa8ce032adaeb641188376f4e7.jpg)

![fded9f9cf62358a13e9162e7eb3b3dee6ccddd8b9f7460a66c05373fd857d062.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/images/fded9f9cf62358a13e9162e7eb3b3dee6ccddd8b9f7460a66c05373fd857d062.jpg)

### Tables

![1cc1d245de6f9e7bdc10bcb313d89d058cdff2791db174611bb1e5dad8ef42d5.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/tables/1cc1d245de6f9e7bdc10bcb313d89d058cdff2791db174611bb1e5dad8ef42d5.jpg)

![4b55bf4d0fceb3aebfeff858606a50d0b5550913e8f14069970100e3ce59acab.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/tables/4b55bf4d0fceb3aebfeff858606a50d0b5550913e8f14069970100e3ce59acab.jpg)

![6aa1c91036ae353501193833e412d2e8d5c73985de431178e868f4c69ecb728f.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/tables/6aa1c91036ae353501193833e412d2e8d5c73985de431178e868f4c69ecb728f.jpg)

![744286ab14156a9b45e0c75b8faaf1845907800f89d10a165295b3c580da96c5.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/tables/744286ab14156a9b45e0c75b8faaf1845907800f89d10a165295b3c580da96c5.jpg)

![a500c4c6ec92cc026212b311f7a779e74e8885e3fb30aeee3d86748ef10c9e1e.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/tables/a500c4c6ec92cc026212b311f7a779e74e8885e3fb30aeee3d86748ef10c9e1e.jpg)

![f534476c2d8295775468e88fa4906de9ad2fc829e261cb33618befab6befb102.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/tables/f534476c2d8295775468e88fa4906de9ad2fc829e261cb33618befab6befb102.jpg)

![f59f15f53c0e60cecc946248bac31356d69018552c12c4774d2a14a7fe86c217.jpg](../icml_results/1912_Simultaneous%20Multi-Robot%20Motion%20Planning%20with%20Projected%20Diffusion%20Models/tables/f59f15f53c0e60cecc946248bac31356d69018552c12c4774d2a14a7fe86c217.jpg)

## Off-Policy Evaluation under Nonignorable Missing Data


### Images

![1b075b9a153d29b35f0e0b0bbff1d2535be89e87bb12db33fa8e6f90a4be2eca.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/images/1b075b9a153d29b35f0e0b0bbff1d2535be89e87bb12db33fa8e6f90a4be2eca.jpg)

![78e02db1efa1a4444d75e6fcc2e06640e9857a03a620e7e611679c63d2e493d1.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/images/78e02db1efa1a4444d75e6fcc2e06640e9857a03a620e7e611679c63d2e493d1.jpg)

![a9fd2945b5d6fcecee3ee862711cf349eb8dcc0b62511116e0781fd664a2a3a5.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/images/a9fd2945b5d6fcecee3ee862711cf349eb8dcc0b62511116e0781fd664a2a3a5.jpg)

### Tables

![0716da612847ea6cdb695fe67d14c5a0f7023238e4e1dd222182bf4f2b314a11.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/tables/0716da612847ea6cdb695fe67d14c5a0f7023238e4e1dd222182bf4f2b314a11.jpg)

![2959f4cacecc277cd521a2da2372aa673e930f628335da26d87f88f517cfe153.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/tables/2959f4cacecc277cd521a2da2372aa673e930f628335da26d87f88f517cfe153.jpg)

![c980713bc7f05fdaf882f741fe7a18819e507d739ba6f33a0564af15d0c83a74.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/tables/c980713bc7f05fdaf882f741fe7a18819e507d739ba6f33a0564af15d0c83a74.jpg)

![e02328381dc11578082e526fc1eb9975e507279ccfe01c28815f8a14ebf41daf.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/tables/e02328381dc11578082e526fc1eb9975e507279ccfe01c28815f8a14ebf41daf.jpg)

![fd513b87c11de8b3ea9f49f3b4f7459a7ea1277d86ca8ca261d12439b47818a8.jpg](../icml_results/1913_Off-Policy%20Evaluation%20under%20Nonignorable%20Missing%20Data/tables/fd513b87c11de8b3ea9f49f3b4f7459a7ea1277d86ca8ca261d12439b47818a8.jpg)

## The Perils of Optimizing Learned Reward Functions: Low Training Error Does Not Guarantee Low Regret


### Images

![8209d6784e4bab29d9d0ab7563ba9dad3b0ed2482b0fef242f561ece651b8218.jpg](../icml_results/1914_The%20Perils%20of%20Optimizing%20Learned%20Reward%20Functions_%20Low%20Training%20Error%20Does%20Not%20Guarantee%20Low%20Regret/images/8209d6784e4bab29d9d0ab7563ba9dad3b0ed2482b0fef242f561ece651b8218.jpg)

![a0e0b48c9ec9ee36b9c923001956f92f955661d13b99db0f51677ccac139e8b6.jpg](../icml_results/1914_The%20Perils%20of%20Optimizing%20Learned%20Reward%20Functions_%20Low%20Training%20Error%20Does%20Not%20Guarantee%20Low%20Regret/images/a0e0b48c9ec9ee36b9c923001956f92f955661d13b99db0f51677ccac139e8b6.jpg)

![a4776eedc5c2cb726b653819e81b29a210f3c8c8abecb4358698df58217adb3c.jpg](../icml_results/1914_The%20Perils%20of%20Optimizing%20Learned%20Reward%20Functions_%20Low%20Training%20Error%20Does%20Not%20Guarantee%20Low%20Regret/images/a4776eedc5c2cb726b653819e81b29a210f3c8c8abecb4358698df58217adb3c.jpg)

![b802f415ed21661e65a84453dd50277e341949047ea050accfc9c87b32b14d0f.jpg](../icml_results/1914_The%20Perils%20of%20Optimizing%20Learned%20Reward%20Functions_%20Low%20Training%20Error%20Does%20Not%20Guarantee%20Low%20Regret/images/b802f415ed21661e65a84453dd50277e341949047ea050accfc9c87b32b14d0f.jpg)

## Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis


### Images

![5d315122dbcf05c9464ad17944c153b94c757c7c47a19006df108a26dc3641c2.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/5d315122dbcf05c9464ad17944c153b94c757c7c47a19006df108a26dc3641c2.jpg)

![615d2f034623788d5a13708c354e551caeb6703f84140f9324e5765aecc04e05.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/615d2f034623788d5a13708c354e551caeb6703f84140f9324e5765aecc04e05.jpg)

![673e6caaf53223fb8c6317b82543ad79e67e7c45b840867893aa592e70734edc.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/673e6caaf53223fb8c6317b82543ad79e67e7c45b840867893aa592e70734edc.jpg)

![88f6dacfdca6da5f4e466f4dd07c84fd141810f4ee8692acfdd4f7b780bcf1fc.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/88f6dacfdca6da5f4e466f4dd07c84fd141810f4ee8692acfdd4f7b780bcf1fc.jpg)

![8b816f53d682cb9e59cce57e5d297d5a175fdba8fb71969eaa46a4bc19cbc419.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/8b816f53d682cb9e59cce57e5d297d5a175fdba8fb71969eaa46a4bc19cbc419.jpg)

![b41cb0082494872c7b0baf4eabf05286d337c775406596c17a359d068f28909b.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/b41cb0082494872c7b0baf4eabf05286d337c775406596c17a359d068f28909b.jpg)

![ce9ce94db3088ba4411a07558c33426da5a12036071fad0e0d734ff888c509a6.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/ce9ce94db3088ba4411a07558c33426da5a12036071fad0e0d734ff888c509a6.jpg)

![d5d78e11bfca7b1af7751f768a9f6064a707d6c59c591e5c8c64aebbdfc1eadc.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/d5d78e11bfca7b1af7751f768a9f6064a707d6c59c591e5c8c64aebbdfc1eadc.jpg)

![e519f76ad42ce5323a72a166ab4f2f40f18f73e93a0c7037b719956cd6cad9f2.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/images/e519f76ad42ce5323a72a166ab4f2f40f18f73e93a0c7037b719956cd6cad9f2.jpg)

### Tables

![053624489d203e50efb01b49c53e54c6af9cc275fda5fd44d8c33fcade1cd6ae.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/tables/053624489d203e50efb01b49c53e54c6af9cc275fda5fd44d8c33fcade1cd6ae.jpg)

![793adaf498ef2dbbb0e8429130c94c657ad0daf0bb7c55c11bae7468ad9c1d92.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/tables/793adaf498ef2dbbb0e8429130c94c657ad0daf0bb7c55c11bae7468ad9c1d92.jpg)

![e7177410aceb2059c5d091720c41482fbd231af0da409daba2a0c3253b17a859.jpg](../icml_results/1915_Does%20Graph%20Prompt%20Work_%20A%20Data%20Operation%20Perspective%20with%20Theoretical%20Analysis/tables/e7177410aceb2059c5d091720c41482fbd231af0da409daba2a0c3253b17a859.jpg)

## Propagation of Chaos for Mean-Field Langevin Dynamics and its Application to Model Ensemble


### Images

![6fcca1949c8a8cd5e5708b1f7148537d42d783468c473674cc1997c9b79b2f8e.jpg](../icml_results/1916_Propagation%20of%20Chaos%20for%20Mean-Field%20Langevin%20Dynamics%20and%20its%20Application%20to%20Model%20Ensemble/images/6fcca1949c8a8cd5e5708b1f7148537d42d783468c473674cc1997c9b79b2f8e.jpg)

![c112b5e87b1c4843166fcfbdebb7ca529a39bf138c751b09d13babc78256b058.jpg](../icml_results/1916_Propagation%20of%20Chaos%20for%20Mean-Field%20Langevin%20Dynamics%20and%20its%20Application%20to%20Model%20Ensemble/images/c112b5e87b1c4843166fcfbdebb7ca529a39bf138c751b09d13babc78256b058.jpg)

### Tables

![37775b375c4ab15786739c90813a1b372b26a12697ec00759399d164aed6dc1a.jpg](../icml_results/1916_Propagation%20of%20Chaos%20for%20Mean-Field%20Langevin%20Dynamics%20and%20its%20Application%20to%20Model%20Ensemble/tables/37775b375c4ab15786739c90813a1b372b26a12697ec00759399d164aed6dc1a.jpg)

![75c1136a259cae8745fdab5959fc012028423a0e157d35dc216550bb568d2d9b.jpg](../icml_results/1916_Propagation%20of%20Chaos%20for%20Mean-Field%20Langevin%20Dynamics%20and%20its%20Application%20to%20Model%20Ensemble/tables/75c1136a259cae8745fdab5959fc012028423a0e157d35dc216550bb568d2d9b.jpg)

![7639b5e65330359bf7723f68b20de19049e9c33e139a2d03966b2e7af15b24a0.jpg](../icml_results/1916_Propagation%20of%20Chaos%20for%20Mean-Field%20Langevin%20Dynamics%20and%20its%20Application%20to%20Model%20Ensemble/tables/7639b5e65330359bf7723f68b20de19049e9c33e139a2d03966b2e7af15b24a0.jpg)

## Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability


### Images

![097bd9f5a97cdcd912e525e67e92f495923555f589058d48c072923f8180de65.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/097bd9f5a97cdcd912e525e67e92f495923555f589058d48c072923f8180de65.jpg)

![0c37354210ce3b53ee79fec8f9068ffb0be9f33b361ebf05af192d24480a43ef.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/0c37354210ce3b53ee79fec8f9068ffb0be9f33b361ebf05af192d24480a43ef.jpg)

![1ce44740bb4b2f673473baa7adac90453900159a23af7965229b2913d901ead7.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/1ce44740bb4b2f673473baa7adac90453900159a23af7965229b2913d901ead7.jpg)

![1eed9dca1036936c511466876ae0fa71dc0aefacb2d30bab928647e05905cdb3.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/1eed9dca1036936c511466876ae0fa71dc0aefacb2d30bab928647e05905cdb3.jpg)

![305932a360d0a293f997abbdc7c29955895a0547d0d336e07a450e43dd8bb3fa.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/305932a360d0a293f997abbdc7c29955895a0547d0d336e07a450e43dd8bb3fa.jpg)

![3149ceb2e4f8105a343b816be3babe8d6c0b801801640b3cdd995f9f8eb9c292.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/3149ceb2e4f8105a343b816be3babe8d6c0b801801640b3cdd995f9f8eb9c292.jpg)

![3229d8fcbee306a1ea508951c0b16fede7e0d40881e47830e5ba82e45f891d25.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/3229d8fcbee306a1ea508951c0b16fede7e0d40881e47830e5ba82e45f891d25.jpg)

![379b6f4c6a1aa14dffc6b18c6420cdaf43742228ecd462c48f6a7f51e68231e1.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/379b6f4c6a1aa14dffc6b18c6420cdaf43742228ecd462c48f6a7f51e68231e1.jpg)

![3d069be54a9a3bcbd1060a4b1c159453a925bebce4ddf26fab309fb8f7e27ad2.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/3d069be54a9a3bcbd1060a4b1c159453a925bebce4ddf26fab309fb8f7e27ad2.jpg)

![4144e93a4aad24ab915391fbb0fe6eed0506aad15df25a0bad33e2c23d1946e8.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/4144e93a4aad24ab915391fbb0fe6eed0506aad15df25a0bad33e2c23d1946e8.jpg)

![42916b443a36e16b8b3c135b86416c67ed243a4ba2689e08ba047f9b4481abad.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/42916b443a36e16b8b3c135b86416c67ed243a4ba2689e08ba047f9b4481abad.jpg)

![4da1cfc9e69e9c4579062c476528a274dbd21bcc16dbac05be634339d6e102ed.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/4da1cfc9e69e9c4579062c476528a274dbd21bcc16dbac05be634339d6e102ed.jpg)

![515f14e7a559ad2efacb77bb0ce79ffa2bf7a69f3c04d1198395401d3b492279.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/515f14e7a559ad2efacb77bb0ce79ffa2bf7a69f3c04d1198395401d3b492279.jpg)

![542b46d92bfd4cb72e84dce9524a660172943ba639d5736f77c577d1fbdad41d.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/542b46d92bfd4cb72e84dce9524a660172943ba639d5736f77c577d1fbdad41d.jpg)

![59412306245d19a6b70ca4bd2bd6bdbe726b4b23be8fe9ba5ebdabf011b1c4fd.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/59412306245d19a6b70ca4bd2bd6bdbe726b4b23be8fe9ba5ebdabf011b1c4fd.jpg)

![5ce0daf62d1d00d7384d9d9d7518e6052c7370a65629a9a9f00faf2377392f86.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/5ce0daf62d1d00d7384d9d9d7518e6052c7370a65629a9a9f00faf2377392f86.jpg)

![5df0bda8501ceed19d499c5a7a3f9c5a87ffd90df82edc07dbf09affcbe62667.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/5df0bda8501ceed19d499c5a7a3f9c5a87ffd90df82edc07dbf09affcbe62667.jpg)

![61cc070bebdce9af13c927180509332112dfd28f72731d372b4577d19e7310b2.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/61cc070bebdce9af13c927180509332112dfd28f72731d372b4577d19e7310b2.jpg)

![640eb132fdbfa6b027019414dbbe3ad3cc3ddfa9e48516130155b4bed02e414c.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/640eb132fdbfa6b027019414dbbe3ad3cc3ddfa9e48516130155b4bed02e414c.jpg)

![649f0dbd2864762f26d8007edf9d9b96f1f11a39861f131837c93569726ae626.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/649f0dbd2864762f26d8007edf9d9b96f1f11a39861f131837c93569726ae626.jpg)

![64c94bb82396f51998db35dd9d4a786fe98c23231adbe474ff1dea4fbb07f2bb.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/64c94bb82396f51998db35dd9d4a786fe98c23231adbe474ff1dea4fbb07f2bb.jpg)

![6abc602350ffc473e643e919b03ea8d57a626d16d69c86257f96d27ad47c27d1.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/6abc602350ffc473e643e919b03ea8d57a626d16d69c86257f96d27ad47c27d1.jpg)

![6f99eb6e2c55af5c2c44541fb1b4af4be04bc2bb1188f0fce9163bee4f5d57f7.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/6f99eb6e2c55af5c2c44541fb1b4af4be04bc2bb1188f0fce9163bee4f5d57f7.jpg)

![7f30041bcbf0073a5343b17bc0f758089b3a707ef4857b5d62add3992675f151.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/7f30041bcbf0073a5343b17bc0f758089b3a707ef4857b5d62add3992675f151.jpg)

![946d4f3b8899e269314f599870a10a254fb7ecfadcfd819f16dccb57b5680b98.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/946d4f3b8899e269314f599870a10a254fb7ecfadcfd819f16dccb57b5680b98.jpg)

![9c0100d171e44cf7f9584ff3de5f70ba8ba9804c072e2b89fee39e347daa0ac1.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/9c0100d171e44cf7f9584ff3de5f70ba8ba9804c072e2b89fee39e347daa0ac1.jpg)

![9dce25ca1daef5ac7b0b1eca98ae90f5733a8381eb71bfe3e7743d241bc963e1.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/9dce25ca1daef5ac7b0b1eca98ae90f5733a8381eb71bfe3e7743d241bc963e1.jpg)

![af2cf3eda2565d622db910ee99a2eadb2c9ebecbed6631e147d46942975ab5d1.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/af2cf3eda2565d622db910ee99a2eadb2c9ebecbed6631e147d46942975ab5d1.jpg)

![b25d0ee7381bbe45ef467970fabb6f375de02b8d4c6d099203613aaf553ac7ca.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/b25d0ee7381bbe45ef467970fabb6f375de02b8d4c6d099203613aaf553ac7ca.jpg)

![b5533df5d297d404cef65fa42b574ccb2a70fea65bc9d3c9262a2acae4233f9b.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/b5533df5d297d404cef65fa42b574ccb2a70fea65bc9d3c9262a2acae4233f9b.jpg)

![c503eeba9f87ec3c8eb05b87bd9f62dc71903baa2dc2cb4f17da71df5768c2f6.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/c503eeba9f87ec3c8eb05b87bd9f62dc71903baa2dc2cb4f17da71df5768c2f6.jpg)

![cfd7ccb63c324109f150f6ced3637f84718f8b3c381b55a2833b0d498da9dfdb.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/cfd7ccb63c324109f150f6ced3637f84718f8b3c381b55a2833b0d498da9dfdb.jpg)

![d3275d4c0ff640b1df0c6f9199674e1df302cc0898272c90b0204f1e037dd1e0.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/d3275d4c0ff640b1df0c6f9199674e1df302cc0898272c90b0204f1e037dd1e0.jpg)

![da4d9ea876ec439d3436b1e41f8c3b444d5638813aae51d5a5523b9a61089276.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/da4d9ea876ec439d3436b1e41f8c3b444d5638813aae51d5a5523b9a61089276.jpg)

![dcb2ae495f5223ac7a6d94350e7d6b968181fee651bb62561e6d421d2ebf796f.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/dcb2ae495f5223ac7a6d94350e7d6b968181fee651bb62561e6d421d2ebf796f.jpg)

![e16e22506447eaa1f3fc0a31a3c6b964148b8bf72747b75a30b1790521470491.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/images/e16e22506447eaa1f3fc0a31a3c6b964148b8bf72747b75a30b1790521470491.jpg)

### Tables

![2c254f8aafa4f23b7fba609de17d7f5c911b23b431e25791a1ea9d9e13cce1b1.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/tables/2c254f8aafa4f23b7fba609de17d7f5c911b23b431e25791a1ea9d9e13cce1b1.jpg)

![a4672ff24ee4ed5d2176493045d75a98fed54f114c08351f4656c62af2fdf9ec.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/tables/a4672ff24ee4ed5d2176493045d75a98fed54f114c08351f4656c62af2fdf9ec.jpg)

![b6d62655133b137513ed247f6f4190b2ae7757f74ec4872b6788c5d7a1f18334.jpg](../icml_results/1917_Synthesizing%20Images%20on%20Perceptual%20Boundaries%20of%20ANNs%20for%20Uncovering%20and%20Manipulating%20Human%20Perceptua/tables/b6d62655133b137513ed247f6f4190b2ae7757f74ec4872b6788c5d7a1f18334.jpg)

## Boosting Multi-Domain Fine-Tuning of Large Language Models through Evolving Interactions between Samples


### Images

![02f2dbc97da4c1e7fb9d9a455c494f4ff0898cb4670166875de7781a399ce74c.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/images/02f2dbc97da4c1e7fb9d9a455c494f4ff0898cb4670166875de7781a399ce74c.jpg)

![68659ee93ea5d6351e602dbf76964fa3857500af31aa0d40102a5e995bc6e805.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/images/68659ee93ea5d6351e602dbf76964fa3857500af31aa0d40102a5e995bc6e805.jpg)

![804bd75113efdca228dd1b453b5c745ac9a1368961dd2fb52f50acb74570a105.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/images/804bd75113efdca228dd1b453b5c745ac9a1368961dd2fb52f50acb74570a105.jpg)

![c7cbd9f838ee9f0a0058daa16208e31ab8b504f1e1405b7dfb96df5bba990c2e.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/images/c7cbd9f838ee9f0a0058daa16208e31ab8b504f1e1405b7dfb96df5bba990c2e.jpg)

### Tables

![4a2f4f00177059b28a236e77e321ecd769962cd21d00be8cc7e01881abfea555.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/4a2f4f00177059b28a236e77e321ecd769962cd21d00be8cc7e01881abfea555.jpg)

![5b36c713ff69d5d0bb234cba4736d959d1596a4a20cc726e930429f18c67732e.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/5b36c713ff69d5d0bb234cba4736d959d1596a4a20cc726e930429f18c67732e.jpg)

![6e55f22aade5fed7bf0ad54e2b030cc1b4ee0b3a0296d72c9e699b4ec09a7ea1.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/6e55f22aade5fed7bf0ad54e2b030cc1b4ee0b3a0296d72c9e699b4ec09a7ea1.jpg)

![7c8f776a745e019d2a9e2988627e2b9f84d1f3e6269e91f6a15943ae7a3042c6.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/7c8f776a745e019d2a9e2988627e2b9f84d1f3e6269e91f6a15943ae7a3042c6.jpg)

![ad1c96124893a5c15ea282b402f6cba0187a23eb668ec9f22799d58efe05e97b.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/ad1c96124893a5c15ea282b402f6cba0187a23eb668ec9f22799d58efe05e97b.jpg)

![b929e27339a2b5daf21a3ca89cc336fd9a89e1d708f6483e2e54b1ad25176afa.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/b929e27339a2b5daf21a3ca89cc336fd9a89e1d708f6483e2e54b1ad25176afa.jpg)

![cb3bfa301a2394635d30a77b8a2adec1137f484c098609de68043b2ba25646fa.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/cb3bfa301a2394635d30a77b8a2adec1137f484c098609de68043b2ba25646fa.jpg)

![fb05316d400db05f8b4dbc45009f69b7dc21f7c3cc405e2b067571c0ce3ee926.jpg](../icml_results/1918_Boosting%20Multi-Domain%20Fine-Tuning%20of%20Large%20Language%20Models%20through%20Evolving%20Interactions%20between%20Sam/tables/fb05316d400db05f8b4dbc45009f69b7dc21f7c3cc405e2b067571c0ce3ee926.jpg)

## Differential Privacy Under Class Imbalance: Methods and Empirical Insights


### Images

![1a4790c9354575380482b84decdc1c9130208e70ab84bca13d400b6896a3e0a8.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/1a4790c9354575380482b84decdc1c9130208e70ab84bca13d400b6896a3e0a8.jpg)

![29a0c3e916d8d24fbb8563d1c85614b5c55116c6bdc086bdadc2ce2500ae893b.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/29a0c3e916d8d24fbb8563d1c85614b5c55116c6bdc086bdadc2ce2500ae893b.jpg)

![3ff4b84af339074ee3ff8ad616a50cf8067d819041e2762758ff68f5b6e4cff1.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/3ff4b84af339074ee3ff8ad616a50cf8067d819041e2762758ff68f5b6e4cff1.jpg)

![40df3b605cbce18baa6e6fbd1852cac5f54241ec846cd4ebbbd27522566c792a.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/40df3b605cbce18baa6e6fbd1852cac5f54241ec846cd4ebbbd27522566c792a.jpg)

![5914ba354330860b3ae7c45252d755de4ec9e36edc717d943300eba8e1ad8fa1.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/5914ba354330860b3ae7c45252d755de4ec9e36edc717d943300eba8e1ad8fa1.jpg)

![6c2127b43fbaadb566cd7ac810206d40743505c681be4cb666bde09508af73dd.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/6c2127b43fbaadb566cd7ac810206d40743505c681be4cb666bde09508af73dd.jpg)

![6eed759505ae6879c0073e26f063261e917545f758587fce579cec31c268a4fb.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/6eed759505ae6879c0073e26f063261e917545f758587fce579cec31c268a4fb.jpg)

![741e2907858a2d6ac4c3bf96383d729b64b99a7b05f6f6e91f8b0d452cdb1521.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/741e2907858a2d6ac4c3bf96383d729b64b99a7b05f6f6e91f8b0d452cdb1521.jpg)

![b3c5f96edd842f518a34437501e79a45ccd26282365dcda6f66cca85e2e0db9f.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/b3c5f96edd842f518a34437501e79a45ccd26282365dcda6f66cca85e2e0db9f.jpg)

![ea290a2c2593c0337d63f92dec88a2d6567c0cb8aab2591119e3dfeab6eea3d4.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/ea290a2c2593c0337d63f92dec88a2d6567c0cb8aab2591119e3dfeab6eea3d4.jpg)

![ec5b40851a7f0cc657f6de3bfba6378b09bbd251e43bdbe6661be3899aa1aa92.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/ec5b40851a7f0cc657f6de3bfba6378b09bbd251e43bdbe6661be3899aa1aa92.jpg)

![ee8902bf1ec3d6129c62f99fa5b13986e4458ba755e55f725942d27fc88e98a2.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/ee8902bf1ec3d6129c62f99fa5b13986e4458ba755e55f725942d27fc88e98a2.jpg)

![fed59a7741cf9144308c39557d7398e81c099b84b9851b094e3fd1f2614d3e5d.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/images/fed59a7741cf9144308c39557d7398e81c099b84b9851b094e3fd1f2614d3e5d.jpg)

### Tables

![0ff377634cf2936a78e872ea4adc3fe48101edb27bf402058bdecd328f01a58c.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/0ff377634cf2936a78e872ea4adc3fe48101edb27bf402058bdecd328f01a58c.jpg)

![138644860b7279836501fba92f3fa3f573297cfa9090b0449ba93124a71e0ff5.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/138644860b7279836501fba92f3fa3f573297cfa9090b0449ba93124a71e0ff5.jpg)

![1763ab91ee7a57525699770f201eb09c2d4b1936f67c8bbbdef72d5e9d59ac35.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/1763ab91ee7a57525699770f201eb09c2d4b1936f67c8bbbdef72d5e9d59ac35.jpg)

![30f384421b16af1879809de7e4d14c847fa77a5d885264fd072e145e70f7764a.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/30f384421b16af1879809de7e4d14c847fa77a5d885264fd072e145e70f7764a.jpg)

![32568fa481f37b7a525ca5cc144a370577bc17e9aff15700337237da83531dde.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/32568fa481f37b7a525ca5cc144a370577bc17e9aff15700337237da83531dde.jpg)

![65fc542f9799505cad7fc5ae0ef76e6e4207ce217349311cfedb395f29cfdd5d.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/65fc542f9799505cad7fc5ae0ef76e6e4207ce217349311cfedb395f29cfdd5d.jpg)

![6a69c5e3250019468bc45c04a1935fa0ca53b242159f3c90cb0f757aad8251df.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/6a69c5e3250019468bc45c04a1935fa0ca53b242159f3c90cb0f757aad8251df.jpg)

![85b91fff40a8a4fcc978a096db06e2a95af3b51c2371b0a2a9ab7d5c7ff4b594.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/85b91fff40a8a4fcc978a096db06e2a95af3b51c2371b0a2a9ab7d5c7ff4b594.jpg)

![870098e8792edf007dc4cd363c08eaa2290bc3d274c5dcbdda263a47170ec396.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/870098e8792edf007dc4cd363c08eaa2290bc3d274c5dcbdda263a47170ec396.jpg)

![93529cb9e8c61023a0835fc9136ef3620b4a59778f769c1d6fb028150aff2f0f.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/93529cb9e8c61023a0835fc9136ef3620b4a59778f769c1d6fb028150aff2f0f.jpg)

![b8aab763f50deaa6c49de93063910cf787f3cbdf2ca67dceabcd9f7d89b66a6d.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/b8aab763f50deaa6c49de93063910cf787f3cbdf2ca67dceabcd9f7d89b66a6d.jpg)

![bb3c9fa81bad02e4041a325057c2f9739c413d37fec86b10f12af55dc5415418.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/bb3c9fa81bad02e4041a325057c2f9739c413d37fec86b10f12af55dc5415418.jpg)

![bd3074b5deaf253d431ff1dbe44a97f8134e8501a394f2d26af4a30ab122fda6.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/bd3074b5deaf253d431ff1dbe44a97f8134e8501a394f2d26af4a30ab122fda6.jpg)

![c81f73af748b216655c808d5b39c706c233cc82a991656d66ca5b03b45a40b66.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/c81f73af748b216655c808d5b39c706c233cc82a991656d66ca5b03b45a40b66.jpg)

![cec620b92b3dc251d2ff70d5948ec6494ba4962ed6d6070ad7454f7d4379ef97.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/cec620b92b3dc251d2ff70d5948ec6494ba4962ed6d6070ad7454f7d4379ef97.jpg)

![e5ca0cd1bc8903693622d7a3799df358bc280e49f10badd04dc2aa78343a3112.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/e5ca0cd1bc8903693622d7a3799df358bc280e49f10badd04dc2aa78343a3112.jpg)

![e7207b14f5cc91c5c90553fefcb5ee0b7ecec715f3551e78866924f2943a088d.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/e7207b14f5cc91c5c90553fefcb5ee0b7ecec715f3551e78866924f2943a088d.jpg)

![e7dc194b52f1cb01c4bbcd4641981e08a64b11c386a4fce3b359a93b2cb230ee.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/e7dc194b52f1cb01c4bbcd4641981e08a64b11c386a4fce3b359a93b2cb230ee.jpg)

![e880b9a30e9218147231676b22d6666166dbb1ccf56d9f95665b236a27c1d5e0.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/e880b9a30e9218147231676b22d6666166dbb1ccf56d9f95665b236a27c1d5e0.jpg)

![eae1670f12cb86d6e2313dbffce5d6ff8571b38c47aad662047bf8baf2a81c5a.jpg](../icml_results/1919_Differential%20Privacy%20Under%20Class%20Imbalance_%20Methods%20and%20Empirical%20Insights/tables/eae1670f12cb86d6e2313dbffce5d6ff8571b38c47aad662047bf8baf2a81c5a.jpg)

## RepLoRA: Reparameterizing Low-rank Adaptation via the Perspective of Mixture of Experts


### Images

![465b31fda41b58eefd2eb287ecf2e0638e623a63e97ab9957154944125c3f1e0.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/images/465b31fda41b58eefd2eb287ecf2e0638e623a63e97ab9957154944125c3f1e0.jpg)

![89595d23bf5b3975c38d5ada10986b7a43678ae1b9ff5230d05837e69ec18752.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/images/89595d23bf5b3975c38d5ada10986b7a43678ae1b9ff5230d05837e69ec18752.jpg)

![a7259a352dc7b05c4a335be1762d4bb5ecbf691fa5f47d940aea9f9d538a5664.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/images/a7259a352dc7b05c4a335be1762d4bb5ecbf691fa5f47d940aea9f9d538a5664.jpg)

![bc02d1fd295e30e9cfa45b4199f615764cb82d2159f400f4a255bf2a29d18c8e.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/images/bc02d1fd295e30e9cfa45b4199f615764cb82d2159f400f4a255bf2a29d18c8e.jpg)

### Tables

![0417441457f597f480927489f5a31e295355bcb7f350db11faaecae2ce7500d1.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/0417441457f597f480927489f5a31e295355bcb7f350db11faaecae2ce7500d1.jpg)

![1822aa81f56d3b8260a98589ec3b3e4072cbe203970e811321b12e2817ec5e96.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/1822aa81f56d3b8260a98589ec3b3e4072cbe203970e811321b12e2817ec5e96.jpg)

![20b3bf511b26bb501dc50e9da4801e3595e93ee1cc513e8cf108298fcf7c51ef.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/20b3bf511b26bb501dc50e9da4801e3595e93ee1cc513e8cf108298fcf7c51ef.jpg)

![4cece0c12f2f107a8382279c163d6e77852f75050398b2a309f6d32ab7d5d9b7.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/4cece0c12f2f107a8382279c163d6e77852f75050398b2a309f6d32ab7d5d9b7.jpg)

![576aed803b42bc0b35a092576f64895dffda84286f391895f166f432b7cd6dc8.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/576aed803b42bc0b35a092576f64895dffda84286f391895f166f432b7cd6dc8.jpg)

![6d9ca542b41e6bc91f7b823b7f3ded03de7246f760e66229c97f4c50cb0d2957.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/6d9ca542b41e6bc91f7b823b7f3ded03de7246f760e66229c97f4c50cb0d2957.jpg)

![9fa7664e6b9c9d5c3a16fd38f74f9a050782ed6f5a9fea922d0c73a7b037f942.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/9fa7664e6b9c9d5c3a16fd38f74f9a050782ed6f5a9fea922d0c73a7b037f942.jpg)

![a1dcc5570a836baaed49d4dfa5046886d7c868b9c3101dbddaf210419ae3eb7d.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/a1dcc5570a836baaed49d4dfa5046886d7c868b9c3101dbddaf210419ae3eb7d.jpg)

![ab4efae6e54c8a7fc39eb64eb2a259770afadfff8ad1a6e84a2628b2af38009c.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/ab4efae6e54c8a7fc39eb64eb2a259770afadfff8ad1a6e84a2628b2af38009c.jpg)

![baecb4e63251964f8434d0489ccf973a8105b2667a8cb0edfbbea0711ab74d03.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/baecb4e63251964f8434d0489ccf973a8105b2667a8cb0edfbbea0711ab74d03.jpg)

![c430c13f51caa437c3c8a27136e26167fee6e80f47ae0a17bbc65ca2c8d966b5.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/c430c13f51caa437c3c8a27136e26167fee6e80f47ae0a17bbc65ca2c8d966b5.jpg)

![cb722d0fe7d86eba586af306590c53939c410ff783cb40a0b8ff04ce458cbc40.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/cb722d0fe7d86eba586af306590c53939c410ff783cb40a0b8ff04ce458cbc40.jpg)

![cc759dcaab6581c5a6a722e9e660fd76c96f9442e9df91d87576c7115a660029.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/cc759dcaab6581c5a6a722e9e660fd76c96f9442e9df91d87576c7115a660029.jpg)

![d6c8ac1eee24d2d1f341c397207aa2680a9444add859f69045ba2548daf90d25.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/d6c8ac1eee24d2d1f341c397207aa2680a9444add859f69045ba2548daf90d25.jpg)

![e1159c9d920f80ea7ea2088f319bcef536e39b14a953496c7a65c44096c3800a.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/e1159c9d920f80ea7ea2088f319bcef536e39b14a953496c7a65c44096c3800a.jpg)

![ebb1e1a9c1c321aa3728475c20dde5673de3320704f2ef750e064f196d8c8649.jpg](../icml_results/1920_RepLoRA_%20Reparameterizing%20Low-rank%20Adaptation%20via%20the%20Perspective%20of%20Mixture%20of%20Experts/tables/ebb1e1a9c1c321aa3728475c20dde5673de3320704f2ef750e064f196d8c8649.jpg)

## SECOND: Mitigating Perceptual Hallucination in Vision-Language Models via Selective and Contrastive Decoding


### Images

![083b9bf616e1d1ddf9228dcf7d45589e4032fb148104a493f4dd51325d123af8.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/083b9bf616e1d1ddf9228dcf7d45589e4032fb148104a493f4dd51325d123af8.jpg)

![18e5ca5185240c5b37d177943eec18510186c6f6496f61a2d4d278a965b2c199.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/18e5ca5185240c5b37d177943eec18510186c6f6496f61a2d4d278a965b2c199.jpg)

![4877b158f1f7de07ce99060231da3798c52ec5f603fae8fff7231ea8c0678dc6.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/4877b158f1f7de07ce99060231da3798c52ec5f603fae8fff7231ea8c0678dc6.jpg)

![783068bdd04649ef9f1418f6e4cb3a60f4053bc24d62f620a6fdf87ab97cc4c6.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/783068bdd04649ef9f1418f6e4cb3a60f4053bc24d62f620a6fdf87ab97cc4c6.jpg)

![a1b9d7c98f445028789cb06658fa80c7b958e5776574699c3f7eb341cd830229.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/a1b9d7c98f445028789cb06658fa80c7b958e5776574699c3f7eb341cd830229.jpg)

![b40b1376c9acf81c43fcf655d834a6a62bf71fc97d2b4579e0d1678a9143d0bf.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/b40b1376c9acf81c43fcf655d834a6a62bf71fc97d2b4579e0d1678a9143d0bf.jpg)

![b4d47bbb89084814265434b45a89c26b77a9875ced719b6e187171dd11e28d3a.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/b4d47bbb89084814265434b45a89c26b77a9875ced719b6e187171dd11e28d3a.jpg)

![ba70cdc7c44830842177e1610ed1f4def32720afdd53cd4ab082900724185b48.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/ba70cdc7c44830842177e1610ed1f4def32720afdd53cd4ab082900724185b48.jpg)

![f0ef70b0214470c77558294883d426aee65eaae22f33672afa29d9ef907ac8ce.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/images/f0ef70b0214470c77558294883d426aee65eaae22f33672afa29d9ef907ac8ce.jpg)

### Tables

![07376c365784f8b7b0d3f58e10534747edadf999ec59b3a429b5961d173714ba.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/07376c365784f8b7b0d3f58e10534747edadf999ec59b3a429b5961d173714ba.jpg)

![083e4444bb904d5d3803a2aebecf1a469b2c421063800468a3548b2eff5350dc.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/083e4444bb904d5d3803a2aebecf1a469b2c421063800468a3548b2eff5350dc.jpg)

![2e41432dd4aee0eeb606c3d50e2aa3ecb11ae81c6ca043dc37749c25e62e83dc.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/2e41432dd4aee0eeb606c3d50e2aa3ecb11ae81c6ca043dc37749c25e62e83dc.jpg)

![40a2250fd0ebc6de95b439910b1adb5113ba025a84b496cd4f6a0949384b0b39.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/40a2250fd0ebc6de95b439910b1adb5113ba025a84b496cd4f6a0949384b0b39.jpg)

![6a0883790a1a8b906eed5b8faf8ebb785460f5546f2be0d4f8e9f615e7c54d25.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/6a0883790a1a8b906eed5b8faf8ebb785460f5546f2be0d4f8e9f615e7c54d25.jpg)

![7276c56f78a5c4e8e7e091b822337f9f10a9d6311aaad54e0e3a04d13f6b448a.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/7276c56f78a5c4e8e7e091b822337f9f10a9d6311aaad54e0e3a04d13f6b448a.jpg)

![ddf35d8dc3308c254947b880290074d5d901154dc73633e59e86629eb00f42ad.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/ddf35d8dc3308c254947b880290074d5d901154dc73633e59e86629eb00f42ad.jpg)

![f7d0f56d275074112cff5fe88391eda49901c3edc7710d00b8fa58bde7c3fee3.jpg](../icml_results/1921_SECOND_%20Mitigating%20Perceptual%20Hallucination%20in%20Vision-Language%20Models%20via%20Selective%20and%20Contrastive%20/tables/f7d0f56d275074112cff5fe88391eda49901c3edc7710d00b8fa58bde7c3fee3.jpg)

## Time Series Representations with Hard-Coded Invariances


### Images

![36e9bef563d3f6bf7022bdddb8354630f904a5027bedf52a428b25aeae1f3b82.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/36e9bef563d3f6bf7022bdddb8354630f904a5027bedf52a428b25aeae1f3b82.jpg)

![ac262391572c55e5bbc691ead1be2afa02776dbe56fdd81759f6c037bfdf1176.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/ac262391572c55e5bbc691ead1be2afa02776dbe56fdd81759f6c037bfdf1176.jpg)

![c995bc3823033e9432e2c07f0af333ba8bcd476c947c6fbecb8e2c4f6b943b12.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/c995bc3823033e9432e2c07f0af333ba8bcd476c947c6fbecb8e2c4f6b943b12.jpg)

![cccaf8d2c0385f780057cabe507f7cd2f6f3097168a8245e0029507080590eef.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/cccaf8d2c0385f780057cabe507f7cd2f6f3097168a8245e0029507080590eef.jpg)

![cda2d917e9a4c1392778da4b9aa1162f6e775ed061fc18f140cc71f54bb49996.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/cda2d917e9a4c1392778da4b9aa1162f6e775ed061fc18f140cc71f54bb49996.jpg)

![d56702f6eef169536a71a6ac2b7f1b4e1dfbcf4cf636e14087d6ab30174df412.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/d56702f6eef169536a71a6ac2b7f1b4e1dfbcf4cf636e14087d6ab30174df412.jpg)

![d73b765a122e39b5e3f38eea30452417f9ec4341f4764b0479da8ea9ebe79f99.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/d73b765a122e39b5e3f38eea30452417f9ec4341f4764b0479da8ea9ebe79f99.jpg)

![d7f056cd5e487f07e2361da5dfdeef69cf31cda5cf3d32bf4f78cbda9b63768e.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/d7f056cd5e487f07e2361da5dfdeef69cf31cda5cf3d32bf4f78cbda9b63768e.jpg)

![e2ca09457298069d4cc0f29446de300bd5d30ceae4b53bee58d5292f446c2db4.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/e2ca09457298069d4cc0f29446de300bd5d30ceae4b53bee58d5292f446c2db4.jpg)

![fb7af092b9adc0108bc4a0c8f1fec9f1419d78efe21e3389c72ee008a98a0792.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/images/fb7af092b9adc0108bc4a0c8f1fec9f1419d78efe21e3389c72ee008a98a0792.jpg)

### Tables

![0ff30ec39798d0e65ba7012d673f758d026f2e67c5fe7cb38cacadee0b94246b.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/0ff30ec39798d0e65ba7012d673f758d026f2e67c5fe7cb38cacadee0b94246b.jpg)

![1c305e8454e9574838b21d2caa7d5c773f1b8364e3c3e7c14d009964f91d65f8.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/1c305e8454e9574838b21d2caa7d5c773f1b8364e3c3e7c14d009964f91d65f8.jpg)

![1f8e325bd60a8cd7793bf67d28960ae815c6d3689a8afdb86cab51fa451c67fd.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/1f8e325bd60a8cd7793bf67d28960ae815c6d3689a8afdb86cab51fa451c67fd.jpg)

![311927f5e7dba2a1af7519a92e1ba951bf8dda7ef44c4b9cadb5248da7acd248.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/311927f5e7dba2a1af7519a92e1ba951bf8dda7ef44c4b9cadb5248da7acd248.jpg)

![34a43b6992c43fcd14d7f7d9868f46bdcebc778729c92593d36b39506bdc165b.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/34a43b6992c43fcd14d7f7d9868f46bdcebc778729c92593d36b39506bdc165b.jpg)

![3f477fb06d1a717862090e9f4d88ecc5a455086ff7945105f893d100c05e79a9.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/3f477fb06d1a717862090e9f4d88ecc5a455086ff7945105f893d100c05e79a9.jpg)

![47ef597c612fef42820e483beb9c73b3aa95c949ea5864d491425f28ed1fe0bb.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/47ef597c612fef42820e483beb9c73b3aa95c949ea5864d491425f28ed1fe0bb.jpg)

![60bc5e6d2cb954e10ba7d5213d321b7461a595311e1e26f731384e66d4c247a8.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/60bc5e6d2cb954e10ba7d5213d321b7461a595311e1e26f731384e66d4c247a8.jpg)

![83bb6aa4d4309f67dd19a881247eb8a147c30b7a5147d031802db0b06e69baef.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/83bb6aa4d4309f67dd19a881247eb8a147c30b7a5147d031802db0b06e69baef.jpg)

![9188e0eebc41055a6a492ce6b9268a977a57665c1444caf2a85fb8de1fb411ea.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/9188e0eebc41055a6a492ce6b9268a977a57665c1444caf2a85fb8de1fb411ea.jpg)

![92f6f24cb6860700e1cd83e408137f65af1e3054ad81505b567189d0033a9506.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/92f6f24cb6860700e1cd83e408137f65af1e3054ad81505b567189d0033a9506.jpg)

![cbc84ba4b34159e9c4cf1d6f3b8a94cdcb633bcb4b6ee64531daa97c97570fa3.jpg](../icml_results/1922_Time%20Series%20Representations%20with%20Hard-Coded%20Invariances/tables/cbc84ba4b34159e9c4cf1d6f3b8a94cdcb633bcb4b6ee64531daa97c97570fa3.jpg)

## Finding Wasserstein Ball Center: Efficient Algorithm and The Applications in Fairness


### Images

![1a7892c8f68326cab2458c43bf57527033173a5f4838076297b9fab705f4d8ea.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/1a7892c8f68326cab2458c43bf57527033173a5f4838076297b9fab705f4d8ea.jpg)

![44c23924be14d1ee035a392fab0c10f606131359ba5ee98d0be2f36b76ab46a9.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/44c23924be14d1ee035a392fab0c10f606131359ba5ee98d0be2f36b76ab46a9.jpg)

![4cd388085a72a00d38cf4514492c53d164978c8575722a3f764798dec972eefe.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/4cd388085a72a00d38cf4514492c53d164978c8575722a3f764798dec972eefe.jpg)

![5c5d020c12e47186cab82864321e6ffdf81a5191228314b72d762098ce4bd2c6.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/5c5d020c12e47186cab82864321e6ffdf81a5191228314b72d762098ce4bd2c6.jpg)

![5d782a1bb03204426a79e5f68efd2e9cbe6dd6b3a24dd72f02828183c010d271.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/5d782a1bb03204426a79e5f68efd2e9cbe6dd6b3a24dd72f02828183c010d271.jpg)

![7b121ae604fcb79a9677e1a3ae160b910be8361ed560670c0be3b047dcc56709.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/7b121ae604fcb79a9677e1a3ae160b910be8361ed560670c0be3b047dcc56709.jpg)

![8ebc47cb155d3c6a80054ed4537d75a0198bf26fa6b711c51a741631669f634c.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/8ebc47cb155d3c6a80054ed4537d75a0198bf26fa6b711c51a741631669f634c.jpg)

![991bf75164cc300428312a5bd1bcf3e3524073b74f755c2512b061be0c03125c.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/991bf75164cc300428312a5bd1bcf3e3524073b74f755c2512b061be0c03125c.jpg)

![9a8dd85adde68d925ec98331eb78b2e91b5e665e3d51bd3ca9dd75b8677757c8.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/9a8dd85adde68d925ec98331eb78b2e91b5e665e3d51bd3ca9dd75b8677757c8.jpg)

![b61a4dab34562ea26f37595f08822dc520fe403e0177f4a816c71734c6be536e.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/b61a4dab34562ea26f37595f08822dc520fe403e0177f4a816c71734c6be536e.jpg)

![d6d5a7e7950e5d67e7ffa03dfe134ef31e1f79307a78739dd87eed5092a4f150.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/d6d5a7e7950e5d67e7ffa03dfe134ef31e1f79307a78739dd87eed5092a4f150.jpg)

![efad45c070e6203257a3ab36555d530c2328fd34cf582fc5be3a4a821f73f32a.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/efad45c070e6203257a3ab36555d530c2328fd34cf582fc5be3a4a821f73f32a.jpg)

![fa841e27b6f8b4e862086241eefb21318a53f7d210e7427d4eb9c2bea9f5e10e.jpg](../icml_results/1923_Finding%20Wasserstein%20Ball%20Center_%20Efficient%20Algorithm%20and%20The%20Applications%20in%20Fairness/images/fa841e27b6f8b4e862086241eefb21318a53f7d210e7427d4eb9c2bea9f5e10e.jpg)

## Average Certified Radius is a Poor Metric for Randomized Smoothing


### Images

![070e97197d4653368c038b7eab524875361010f1455562029f434a7b371fb55b.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/070e97197d4653368c038b7eab524875361010f1455562029f434a7b371fb55b.jpg)

![41742306601c8e011ea2e5c30182ac0c353db44c5af4460f43d940ed9d85cea8.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/41742306601c8e011ea2e5c30182ac0c353db44c5af4460f43d940ed9d85cea8.jpg)

![5237e56e7a0af9a1b23b5de21b9a98d2461073d8ff25345a5dc478476da906fd.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/5237e56e7a0af9a1b23b5de21b9a98d2461073d8ff25345a5dc478476da906fd.jpg)

![61e38e8ada94a8f35877079f472bbb10da3bddf35dee05082472bd051e388b9f.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/61e38e8ada94a8f35877079f472bbb10da3bddf35dee05082472bd051e388b9f.jpg)

![63b09de01f45501c0f5d788d708f55652a683688685a42d932f80a9af2728f5b.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/63b09de01f45501c0f5d788d708f55652a683688685a42d932f80a9af2728f5b.jpg)

![6c3e307638dd027b92a98039314e367ac473b1c9e9aa154bb3c9e7bf773549fc.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/6c3e307638dd027b92a98039314e367ac473b1c9e9aa154bb3c9e7bf773549fc.jpg)

![6c7aa9e4e83d47511e09cbc646c5e9e515839b5103d1f7dddc781fc30b8b2be3.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/6c7aa9e4e83d47511e09cbc646c5e9e515839b5103d1f7dddc781fc30b8b2be3.jpg)

![abb211748db4c59cc826297875874a5deac5e26b46ce2d5b2adc2f3a0bceb7d6.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/abb211748db4c59cc826297875874a5deac5e26b46ce2d5b2adc2f3a0bceb7d6.jpg)

![b792cab40610a1256a45c3a25028835ae53c7b0dc2c782902afd8e8071d3602e.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/b792cab40610a1256a45c3a25028835ae53c7b0dc2c782902afd8e8071d3602e.jpg)

![bec92cd915a4153884fdb7b0a416f1f796aa5fd2e7ae8af9ece73e530ed0047a.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/bec92cd915a4153884fdb7b0a416f1f796aa5fd2e7ae8af9ece73e530ed0047a.jpg)

![f7a90a001e122789c11e23d64c42fa1d5930604fb649ecfca65572ab2274e7fe.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/images/f7a90a001e122789c11e23d64c42fa1d5930604fb649ecfca65572ab2274e7fe.jpg)

### Tables

![0f7290bfedc212d2b52afcd91c93502ca705a530d8a1a0e357f6b374751e8e31.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/0f7290bfedc212d2b52afcd91c93502ca705a530d8a1a0e357f6b374751e8e31.jpg)

![1bd316f940feedbb91251e819312eec10045ca5a8dd5b09571e79e396c4b0085.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/1bd316f940feedbb91251e819312eec10045ca5a8dd5b09571e79e396c4b0085.jpg)

![25e071a7b0e5df03aff3bbaaf4700521d63f340a67043ee57f57295a54db27ee.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/25e071a7b0e5df03aff3bbaaf4700521d63f340a67043ee57f57295a54db27ee.jpg)

![33fdf846bcfebe6213f3ef051b65ef0ab380e6b341d0ab0f6ca9a5b840254045.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/33fdf846bcfebe6213f3ef051b65ef0ab380e6b341d0ab0f6ca9a5b840254045.jpg)

![3484643f1eee4626982988bfbf8bc6a455019d55588392d9c2419573e13d5630.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/3484643f1eee4626982988bfbf8bc6a455019d55588392d9c2419573e13d5630.jpg)

![56de7e347fb1cae58af7753d54c80b1c462328c215ed096062789e3d8382683b.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/56de7e347fb1cae58af7753d54c80b1c462328c215ed096062789e3d8382683b.jpg)

![6640d6f45bd9933c0541282c7a7d6f3b897951830b99abca5665fab98d976c3e.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/6640d6f45bd9933c0541282c7a7d6f3b897951830b99abca5665fab98d976c3e.jpg)

![68141399f5f0cf648baa7a3c038b9d153f305e70a53aacab38168ebcaaf6b4a1.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/68141399f5f0cf648baa7a3c038b9d153f305e70a53aacab38168ebcaaf6b4a1.jpg)

![8f466d497cdddb1f0895bf0cd4997ab0fe735ab85c87471f8e4f9d7754724ac4.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/8f466d497cdddb1f0895bf0cd4997ab0fe735ab85c87471f8e4f9d7754724ac4.jpg)

![94256bd8aad3f4712967657759d340c3526773278bdcd44e36b1bf0b3317b7e1.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/94256bd8aad3f4712967657759d340c3526773278bdcd44e36b1bf0b3317b7e1.jpg)

![9f681f9705381eae33ce77c26545e39ece42f90a17547483c8fd1ab73bf0af85.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/9f681f9705381eae33ce77c26545e39ece42f90a17547483c8fd1ab73bf0af85.jpg)

![a0f015c85392e1ef344be69ce71dc15d32444ed2de957f80cc406231045f6f2c.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/a0f015c85392e1ef344be69ce71dc15d32444ed2de957f80cc406231045f6f2c.jpg)

![beebfdbcbfa19c73bec5bffb53230b0505f60e629ec6688d9e962cf1b75ba2e3.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/beebfdbcbfa19c73bec5bffb53230b0505f60e629ec6688d9e962cf1b75ba2e3.jpg)

![c0b432f6ce7b1375b99bc73524e8812ed00d59d33bb87e328180e8a69750bfe2.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/c0b432f6ce7b1375b99bc73524e8812ed00d59d33bb87e328180e8a69750bfe2.jpg)

![d3284bf3b528c78fb4193885124cefb357d7169d6d16e9917174baced7a905c5.jpg](../icml_results/1924_Average%20Certified%20Radius%20is%20a%20Poor%20Metric%20for%20Randomized%20Smoothing/tables/d3284bf3b528c78fb4193885124cefb357d7169d6d16e9917174baced7a905c5.jpg)

## The Case for Learned Provenance-based System Behavior Baseline


### Images

![1cb02889f5f573f755d3ae3fa4cbe4caa6baeb3f6c3113b41ef4669993134cb4.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/images/1cb02889f5f573f755d3ae3fa4cbe4caa6baeb3f6c3113b41ef4669993134cb4.jpg)

![537249b15dd59c80d6957a8134cc98b4de9d752529fd178c186df5d836d9ced2.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/images/537249b15dd59c80d6957a8134cc98b4de9d752529fd178c186df5d836d9ced2.jpg)

![87a7ac5879532fb6c94dd67efe4d7d3eebf72c43e6240bd584bd540a766fc380.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/images/87a7ac5879532fb6c94dd67efe4d7d3eebf72c43e6240bd584bd540a766fc380.jpg)

![c568e0fa4f48dac5123b9b12e12612b2513a8874d7033f3f028e3999feb42dab.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/images/c568e0fa4f48dac5123b9b12e12612b2513a8874d7033f3f028e3999feb42dab.jpg)

### Tables

![19cfa40d4c324aa3a7c38cec1aaba658209ba084bed72f56e7ed5c3019cb9035.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/19cfa40d4c324aa3a7c38cec1aaba658209ba084bed72f56e7ed5c3019cb9035.jpg)

![340d4eb70994f20d41848dfe3be0af43abfa8c611f5236f9ff1dfbd775b081b1.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/340d4eb70994f20d41848dfe3be0af43abfa8c611f5236f9ff1dfbd775b081b1.jpg)

![6e007d73741fca7878967418eec11c0ad8bfb5517bc0e83d799e1ce7ee9a1396.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/6e007d73741fca7878967418eec11c0ad8bfb5517bc0e83d799e1ce7ee9a1396.jpg)

![796978a39bd15751e5e3a1007aea7ebf99eef3355433edc83d84821057695c5f.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/796978a39bd15751e5e3a1007aea7ebf99eef3355433edc83d84821057695c5f.jpg)

![8d05a6c7248adaf70b47551699d1162fae119cde3ca4fe74c80a9d718421675c.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/8d05a6c7248adaf70b47551699d1162fae119cde3ca4fe74c80a9d718421675c.jpg)

![8fdafcea49a62552c32644456de902d52f09b2a4066291ecf9ee09b0333cc21e.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/8fdafcea49a62552c32644456de902d52f09b2a4066291ecf9ee09b0333cc21e.jpg)

![aa3e1c077478cd20790daed4dbb5ded102f9d0962be0b1ff5252140f8f73e15c.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/aa3e1c077478cd20790daed4dbb5ded102f9d0962be0b1ff5252140f8f73e15c.jpg)

![fd3a3f64ce68d2335a0a765a649f6bca8be8d714041179354ce021023c8b4421.jpg](../icml_results/1925_The%20Case%20for%20Learned%20Provenance-based%20System%20Behavior%20Baseline/tables/fd3a3f64ce68d2335a0a765a649f6bca8be8d714041179354ce021023c8b4421.jpg)

## DocKS-RAG: Optimizing Document-Level Relation Extraction through LLM-Enhanced Hybrid Prompt Tuning


### Images

![1356710796fdfd77a694a287c0d60a80888f7925ca307243cab43ba0f28c41cf.jpg](../icml_results/1926_DocKS-RAG_%20Optimizing%20Document-Level%20Relation%20Extraction%20through%20LLM-Enhanced%20Hybrid%20Prompt%20Tuning/images/1356710796fdfd77a694a287c0d60a80888f7925ca307243cab43ba0f28c41cf.jpg)

![29bcdc24779eabbf5ce88d4d6c8cbe5666009b2e4750c206bb04191a06501474.jpg](../icml_results/1926_DocKS-RAG_%20Optimizing%20Document-Level%20Relation%20Extraction%20through%20LLM-Enhanced%20Hybrid%20Prompt%20Tuning/images/29bcdc24779eabbf5ce88d4d6c8cbe5666009b2e4750c206bb04191a06501474.jpg)

### Tables

![48b79233b05d97758afcee8fb5683b0fdbfc7910b7b39c69ac85a1423969e7d1.jpg](../icml_results/1926_DocKS-RAG_%20Optimizing%20Document-Level%20Relation%20Extraction%20through%20LLM-Enhanced%20Hybrid%20Prompt%20Tuning/tables/48b79233b05d97758afcee8fb5683b0fdbfc7910b7b39c69ac85a1423969e7d1.jpg)

![57094cce0f80665d091f0e0fd0a6b2ea85c77702f2306ca90a547e097dc0cac9.jpg](../icml_results/1926_DocKS-RAG_%20Optimizing%20Document-Level%20Relation%20Extraction%20through%20LLM-Enhanced%20Hybrid%20Prompt%20Tuning/tables/57094cce0f80665d091f0e0fd0a6b2ea85c77702f2306ca90a547e097dc0cac9.jpg)

![575d23be8327e8f29d780acc256194dc3bbee2b685abab2b72fd16f80b71b787.jpg](../icml_results/1926_DocKS-RAG_%20Optimizing%20Document-Level%20Relation%20Extraction%20through%20LLM-Enhanced%20Hybrid%20Prompt%20Tuning/tables/575d23be8327e8f29d780acc256194dc3bbee2b685abab2b72fd16f80b71b787.jpg)

![78f30b51fc0d0fc45721f61a0b7db0dc3b7802d73bb83bf7667bb82d5597b51d.jpg](../icml_results/1926_DocKS-RAG_%20Optimizing%20Document-Level%20Relation%20Extraction%20through%20LLM-Enhanced%20Hybrid%20Prompt%20Tuning/tables/78f30b51fc0d0fc45721f61a0b7db0dc3b7802d73bb83bf7667bb82d5597b51d.jpg)

## Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment


### Images

![1ad6174369f398e38f216e7b12995db0331cff3dd7690cee515fbbc3397e2266.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/images/1ad6174369f398e38f216e7b12995db0331cff3dd7690cee515fbbc3397e2266.jpg)

![29c39743b9f19182bcd1dcbba673c2e0f756616ef7db0bce42c6df15ef91a096.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/images/29c39743b9f19182bcd1dcbba673c2e0f756616ef7db0bce42c6df15ef91a096.jpg)

![692f78737b3feccf491a018bd46fec92a9117aad98d83f5e60c30c8d7e72786e.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/images/692f78737b3feccf491a018bd46fec92a9117aad98d83f5e60c30c8d7e72786e.jpg)

![9487be9f4a7174e34a75828ba190d1ededbce926f7d26b5077d2b99163c62767.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/images/9487be9f4a7174e34a75828ba190d1ededbce926f7d26b5077d2b99163c62767.jpg)

![b0295e4b130b38457af44974d5a10dd217805120569bf043db16692ebccb80da.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/images/b0295e4b130b38457af44974d5a10dd217805120569bf043db16692ebccb80da.jpg)

![df4be29f74f94d1c74433fa5c72fa777d6431ca4b349e7a09d7ee8d932d7b7a5.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/images/df4be29f74f94d1c74433fa5c72fa777d6431ca4b349e7a09d7ee8d932d7b7a5.jpg)

![e1563bfb22a3df4ef0c7830980f14b11c72234cec7a1fdf69b30e41d7817ed40.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/images/e1563bfb22a3df4ef0c7830980f14b11c72234cec7a1fdf69b30e41d7817ed40.jpg)

### Tables

![049d6cf743c6fe469ff21a94078c0a9319b7021e7e586cce0ae25306d2fb86e7.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/049d6cf743c6fe469ff21a94078c0a9319b7021e7e586cce0ae25306d2fb86e7.jpg)

![2c846bcc10b77e080011ee58a56c16512f7d82e6962b5e149c90024745727957.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/2c846bcc10b77e080011ee58a56c16512f7d82e6962b5e149c90024745727957.jpg)

![4c169610b48ff93c52e4df99a6a20d7ce6adc42f38b92414c369f7166ec0370c.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/4c169610b48ff93c52e4df99a6a20d7ce6adc42f38b92414c369f7166ec0370c.jpg)

![6e7de75c47a55339290a9ed110b14e0496e7e61fc49474ed517c8e3722033603.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/6e7de75c47a55339290a9ed110b14e0496e7e61fc49474ed517c8e3722033603.jpg)

![8f287d42bdd7c9de87222f84e7ce6742d1dafd8a54921ebd81e30d398f82b9cf.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/8f287d42bdd7c9de87222f84e7ce6742d1dafd8a54921ebd81e30d398f82b9cf.jpg)

![a7e11556fc8c9225def9938e768a14b6bdefec9100618f1be3fd64363ace98fc.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/a7e11556fc8c9225def9938e768a14b6bdefec9100618f1be3fd64363ace98fc.jpg)

![aa7ce8993d8becc39fd6a084837be03cb1f63984f89b96818e1ccc00303f6baf.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/aa7ce8993d8becc39fd6a084837be03cb1f63984f89b96818e1ccc00303f6baf.jpg)

![b27427b18ce94b9c00d3b8ca875e88d37fcaac4cd9bab9cabd0ebe859448fa42.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/b27427b18ce94b9c00d3b8ca875e88d37fcaac4cd9bab9cabd0ebe859448fa42.jpg)

![bb2f84c0190fb9177cf40f5c5bb4021b600f4635ec927d842eb5a35a0b630d6d.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/bb2f84c0190fb9177cf40f5c5bb4021b600f4635ec927d842eb5a35a0b630d6d.jpg)

![c3e2c727bdb3fa6f7af4b4b4ab3e349078dc3b42de0e89cfe5efd63f0c60e2e7.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/c3e2c727bdb3fa6f7af4b4b4ab3e349078dc3b42de0e89cfe5efd63f0c60e2e7.jpg)

![d863360fd1564793f9a9488d4a953c6604a3fd564b443e6dcc579489a3753151.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/d863360fd1564793f9a9488d4a953c6604a3fd564b443e6dcc579489a3753151.jpg)

![d9df3eff7ee02444443e92c7cbf24f229cda0b5b01f1681b6c84a57c95f3445c.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/d9df3eff7ee02444443e92c7cbf24f229cda0b5b01f1681b6c84a57c95f3445c.jpg)

![e240e285cc97fd59f9509b0e6f67edae6466120eba418df3cdf6fb27bf464fa5.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/e240e285cc97fd59f9509b0e6f67edae6466120eba418df3cdf6fb27bf464fa5.jpg)

![fa9ec3ed40228a2cf8006232566304710ead2b123a02b19dbec03091a9482ee2.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/fa9ec3ed40228a2cf8006232566304710ead2b123a02b19dbec03091a9482ee2.jpg)

![fb5a02c47e3a2268910058a04ca100553730f49e39ef96bdd5b4aeb371474e9d.jpg](../icml_results/1927_Make%20LoRA%20Great%20Again_%20Boosting%20LoRA%20with%20Adaptive%20Singular%20Values%20and%20Mixture-of-Experts%20Optimizati/tables/fb5a02c47e3a2268910058a04ca100553730f49e39ef96bdd5b4aeb371474e9d.jpg)

## Stochastic Poisson Surface Reconstruction with One Solve using Geometric Gaussian Processes


### Images

![2f2c8517134c88a56020c92e0d906acf232e10cb2a25712c16e657e6990d9eb9.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/2f2c8517134c88a56020c92e0d906acf232e10cb2a25712c16e657e6990d9eb9.jpg)

![31001aac367053b920320f02043badbe8412ff9d296dfae890ebc3d1071b9332.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/31001aac367053b920320f02043badbe8412ff9d296dfae890ebc3d1071b9332.jpg)

![33f7ea9447a526e844da2860f85b3b5702f02fa7ac70892b20ce23501404be47.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/33f7ea9447a526e844da2860f85b3b5702f02fa7ac70892b20ce23501404be47.jpg)

![3739fa98357dc2dd0e071733ac210f6d27f2e6c011cfd57421eb01ddee06bb6f.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/3739fa98357dc2dd0e071733ac210f6d27f2e6c011cfd57421eb01ddee06bb6f.jpg)

![4028ff2942d20f5fd48996cedfedea3668021714c73ee524442bd1e77ef95369.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/4028ff2942d20f5fd48996cedfedea3668021714c73ee524442bd1e77ef95369.jpg)

![4c8bc1b21c0e35b514f60d7cde3ac0410412cb2bb7fcb536dcb4bbdb36736ada.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/4c8bc1b21c0e35b514f60d7cde3ac0410412cb2bb7fcb536dcb4bbdb36736ada.jpg)

![4cd06d1b06867ffbc745135c19bc684c8768dc7691fd235c24ed59891a93d564.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/4cd06d1b06867ffbc745135c19bc684c8768dc7691fd235c24ed59891a93d564.jpg)

![50ebd3adabf9baa4ba70c83daea8a229110958d11b1fa5f8a13706a7f3cb74a6.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/50ebd3adabf9baa4ba70c83daea8a229110958d11b1fa5f8a13706a7f3cb74a6.jpg)

![606d38d72d134ba8bae4b988fed757ef3e307e1018ba2fed576acdc529331f18.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/606d38d72d134ba8bae4b988fed757ef3e307e1018ba2fed576acdc529331f18.jpg)

![84c33174d63dc2dcd08852d4f9cb3ebde444c60742b4e0fa756160daa6e219dd.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/84c33174d63dc2dcd08852d4f9cb3ebde444c60742b4e0fa756160daa6e219dd.jpg)

![86d5a4e20c42ea222dc2c761331d4c6909084e4a4260fa8c57a2b747594ce94e.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/86d5a4e20c42ea222dc2c761331d4c6909084e4a4260fa8c57a2b747594ce94e.jpg)

![9c1dd7f8c7887dfa6a396f87593f516facd7adb01b6f024a60a63ef890b8880c.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/9c1dd7f8c7887dfa6a396f87593f516facd7adb01b6f024a60a63ef890b8880c.jpg)

![a0ecf3ba03050b92392e14ed0a09076afc29e1b6c86104c05404b49fe3ea2b1c.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/a0ecf3ba03050b92392e14ed0a09076afc29e1b6c86104c05404b49fe3ea2b1c.jpg)

![b2327ff3af6dfaabadc65f9a8a3cea837be8800f1766a183e40f3024ce9c002c.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/b2327ff3af6dfaabadc65f9a8a3cea837be8800f1766a183e40f3024ce9c002c.jpg)

![b978f85564fb31d64f9df1b61dcbef9d5acd55e061017517bb251e471feed338.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/b978f85564fb31d64f9df1b61dcbef9d5acd55e061017517bb251e471feed338.jpg)

![ca1509bca70a59c0b9452336dbf48132c085a68ad99210578e0a9b3336344dc1.jpg](../icml_results/1928_Stochastic%20Poisson%20Surface%20Reconstruction%20with%20One%20Solve%20using%20Geometric%20Gaussian%20Processes/images/ca1509bca70a59c0b9452336dbf48132c085a68ad99210578e0a9b3336344dc1.jpg)

## Rectifying Conformity Scores for Better Conditional Coverage


### Images

![4eede9aa6c2cbf362f02dc565c9278f0805443ae57174dcb33d805d40a99ffa2.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/4eede9aa6c2cbf362f02dc565c9278f0805443ae57174dcb33d805d40a99ffa2.jpg)

![54d6526de5bea934782fee936fae35f1f0eab64144e04fbe3332675583ed21de.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/54d6526de5bea934782fee936fae35f1f0eab64144e04fbe3332675583ed21de.jpg)

![578ac897f2aeb1cff583167843fff9084f35e3117fcc0789c4bc8f68f87a3478.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/578ac897f2aeb1cff583167843fff9084f35e3117fcc0789c4bc8f68f87a3478.jpg)

![59776d3ba531dadb95e15148902a05cd8236f8e74b339ed38c28b06acd0ce7e4.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/59776d3ba531dadb95e15148902a05cd8236f8e74b339ed38c28b06acd0ce7e4.jpg)

![76f8fb7c99d0d84dbdbaf916333b50012d51a3b3a0c03b5e06d7ddaa6e2d7df3.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/76f8fb7c99d0d84dbdbaf916333b50012d51a3b3a0c03b5e06d7ddaa6e2d7df3.jpg)

![77f0ed772f41a13e6cc385d67e49f73871f9b38d5d78814406aa601579167675.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/77f0ed772f41a13e6cc385d67e49f73871f9b38d5d78814406aa601579167675.jpg)

![7c2b604cab7a42dc8df3fb46c2cf89ef61904597f0a2755f58c8c9be29a6ecc5.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/7c2b604cab7a42dc8df3fb46c2cf89ef61904597f0a2755f58c8c9be29a6ecc5.jpg)

![979cc652612a848a8c065e9eb7256c9db4defe0be197efa0275523a55a385ae4.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/979cc652612a848a8c065e9eb7256c9db4defe0be197efa0275523a55a385ae4.jpg)

![ba75ffcbe22d9f2adff4afbf2f4f1c5b1c1f4185da5a592f6b7c7bcda7b39743.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/ba75ffcbe22d9f2adff4afbf2f4f1c5b1c1f4185da5a592f6b7c7bcda7b39743.jpg)

![badd13880447f2177421a2bfe5db18a77dc5c5231fafdfc3bf49c1251a476822.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/badd13880447f2177421a2bfe5db18a77dc5c5231fafdfc3bf49c1251a476822.jpg)

![d3c6109e862ce09a65339c744c2fea549205014a04f3c28a6ab74d7303a5c875.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/d3c6109e862ce09a65339c744c2fea549205014a04f3c28a6ab74d7303a5c875.jpg)

![d847afa0108a0491bfa4f46ec6540e76fd9bba23057c89c9aee62c0998ac4c8e.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/images/d847afa0108a0491bfa4f46ec6540e76fd9bba23057c89c9aee62c0998ac4c8e.jpg)

### Tables

![174a0b9d7f8227aaa1e30eee56ec715780fd4f3a142eb2bc0af238af217c9966.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/tables/174a0b9d7f8227aaa1e30eee56ec715780fd4f3a142eb2bc0af238af217c9966.jpg)

![1772ae018ab6e5701e9594a44a813104c7d63919181c5ffead7ac368fa692ea1.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/tables/1772ae018ab6e5701e9594a44a813104c7d63919181c5ffead7ac368fa692ea1.jpg)

![6aa84fd055430dfe183e4abb3754ad15b9b6d74ed32896546618d0271cb42f01.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/tables/6aa84fd055430dfe183e4abb3754ad15b9b6d74ed32896546618d0271cb42f01.jpg)

![d705afeb79be3719eb9b56c92f365bd2bf0eb7a93a26e6c7800421a19fd1f25f.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/tables/d705afeb79be3719eb9b56c92f365bd2bf0eb7a93a26e6c7800421a19fd1f25f.jpg)

![e063c13ad5e32cea120b8bd584ff2708d97a626382dbe3c24e0c82b81c3f41a7.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/tables/e063c13ad5e32cea120b8bd584ff2708d97a626382dbe3c24e0c82b81c3f41a7.jpg)

![f2ee50ce5e0cd14352001b38c83c261ef6f44c2fe86aecbb1337ba904088ceb4.jpg](../icml_results/1929_Rectifying%20Conformity%20Scores%20for%20Better%20Conditional%20Coverage/tables/f2ee50ce5e0cd14352001b38c83c261ef6f44c2fe86aecbb1337ba904088ceb4.jpg)

## Robust Multimodal Large Language Models Against Modality Conflict


### Images

![286442b55f74a2043dacfdbcd76a0d68c714cbd4598d8f190873eb5e61d8a165.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/286442b55f74a2043dacfdbcd76a0d68c714cbd4598d8f190873eb5e61d8a165.jpg)

![2f8fb428da2d739f38490f6f64908e64233185a2c2467dc615d7947898b8a033.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/2f8fb428da2d739f38490f6f64908e64233185a2c2467dc615d7947898b8a033.jpg)

![3a879ebcd0efab67eda0affab8c4fedc75b3411007e21afb59158e391abc8e53.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/3a879ebcd0efab67eda0affab8c4fedc75b3411007e21afb59158e391abc8e53.jpg)

![4def03fcd16608def062766af8d4c2e585fba4f7d9f15fff13a1e7ec9c169462.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/4def03fcd16608def062766af8d4c2e585fba4f7d9f15fff13a1e7ec9c169462.jpg)

![4e35496a2fd8f342667ebfe7ab6ac1d9457e05f0581fa9e5e4dcf6bf34bf4f37.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/4e35496a2fd8f342667ebfe7ab6ac1d9457e05f0581fa9e5e4dcf6bf34bf4f37.jpg)

![5873ac73fcac01eb27846f9fac47a20073a6d5193647941af6b1004c421dcffa.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/5873ac73fcac01eb27846f9fac47a20073a6d5193647941af6b1004c421dcffa.jpg)

![975e116f028938fabfa49f0fcf4954574f067c2e356612f0f0e0f483cd480c85.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/975e116f028938fabfa49f0fcf4954574f067c2e356612f0f0e0f483cd480c85.jpg)

![a23885a25133e241ea66e0b98a06f4b7f103a50c4e534297480e2c01f8f840ea.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/a23885a25133e241ea66e0b98a06f4b7f103a50c4e534297480e2c01f8f840ea.jpg)

![ae97d4e60bd6724a1e385179e7107f3a009029885bd88764e24a2b0483be0551.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/ae97d4e60bd6724a1e385179e7107f3a009029885bd88764e24a2b0483be0551.jpg)

![dbf0a330148046e5f52bf92754d75416e49c1e538d40b4a934a1c0727829126d.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/dbf0a330148046e5f52bf92754d75416e49c1e538d40b4a934a1c0727829126d.jpg)

![e1b51891ba306f07ba1ee31fb2482c3abfb3c1b22252595c14afb898f8c5b960.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/e1b51891ba306f07ba1ee31fb2482c3abfb3c1b22252595c14afb898f8c5b960.jpg)

![f032e961719584b675e434c619918676c26244640250f5770a2896a8e7da2683.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/f032e961719584b675e434c619918676c26244640250f5770a2896a8e7da2683.jpg)

![f5bd44fdcbc2084e6261a7560cf33a7797e3c5ff8571e96dbe3f68fa991cb58f.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/f5bd44fdcbc2084e6261a7560cf33a7797e3c5ff8571e96dbe3f68fa991cb58f.jpg)

![f7869d3337b725f8c71ded1bc109728323e54d4ea60c7c4220677115dd5d2643.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/images/f7869d3337b725f8c71ded1bc109728323e54d4ea60c7c4220677115dd5d2643.jpg)

### Tables

![1dffb0a80f1e361d9180ffd97c5c5bbaca51965f8878c8634b33b53af565bbde.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/1dffb0a80f1e361d9180ffd97c5c5bbaca51965f8878c8634b33b53af565bbde.jpg)

![4db7e3c5e7505fae1926efb780da4791cafe9561acc2ce640cf9ded346fe03f3.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/4db7e3c5e7505fae1926efb780da4791cafe9561acc2ce640cf9ded346fe03f3.jpg)

![b97faff99d62e159abdabcb37f5c1f1906466aa7f1048a3b1fd1697cdc566b3e.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/b97faff99d62e159abdabcb37f5c1f1906466aa7f1048a3b1fd1697cdc566b3e.jpg)

![bc7e89d9bb6158d801f0483ffefa791c04765f504d74237f158d85342dec2a7f.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/bc7e89d9bb6158d801f0483ffefa791c04765f504d74237f158d85342dec2a7f.jpg)

![c5ad3846dfae8da5de55ccd05ce3985abc71d54755d06a5f980feafc1518aff2.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/c5ad3846dfae8da5de55ccd05ce3985abc71d54755d06a5f980feafc1518aff2.jpg)

![de86543ed49f0257e2c7c1abdb042b3a75b07d60da94bf393769f9e5e34dbee9.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/de86543ed49f0257e2c7c1abdb042b3a75b07d60da94bf393769f9e5e34dbee9.jpg)

![e0e990a39f0f07b6ac95b4293568f8f2bb0a276d7b48939a84f6521901985122.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/e0e990a39f0f07b6ac95b4293568f8f2bb0a276d7b48939a84f6521901985122.jpg)

![e7fb0cf409cda84ffd335ead7c5e24a7a5a9b35737302280bee815ca1a25d431.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/e7fb0cf409cda84ffd335ead7c5e24a7a5a9b35737302280bee815ca1a25d431.jpg)

![f7148e3e0a8885d330ca3696e904e3931c0cb3b163c7cf1290bc11da7e032acb.jpg](../icml_results/1930_Robust%20Multimodal%20Large%20Language%20Models%20Against%20Modality%20Conflict/tables/f7148e3e0a8885d330ca3696e904e3931c0cb3b163c7cf1290bc11da7e032acb.jpg)

## Learning Distances from Data with Normalizing Flows and Score Matching


### Images

![02d3e4c2ce1d7bc29d2ef03e9fb007eedc593f9f0ba8c0e36222f225b42e9bc8.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/02d3e4c2ce1d7bc29d2ef03e9fb007eedc593f9f0ba8c0e36222f225b42e9bc8.jpg)

![1ff78f3e4c5a6c060e279200b20673bba3c5f5310ffaec878796888b206f432b.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/1ff78f3e4c5a6c060e279200b20673bba3c5f5310ffaec878796888b206f432b.jpg)

![293d754578e7e63931151f922338bd648340c3d0683c2607397f23fb39d5d964.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/293d754578e7e63931151f922338bd648340c3d0683c2607397f23fb39d5d964.jpg)

![2f702e8887fb31c2c10a5dd14d33564a2c9f373452b50b13f49dfbe837a82365.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/2f702e8887fb31c2c10a5dd14d33564a2c9f373452b50b13f49dfbe837a82365.jpg)

![4f4626ea945f5f1785a9e66bbd123144fc4c37eb8e0345bc20286fda938b7f93.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/4f4626ea945f5f1785a9e66bbd123144fc4c37eb8e0345bc20286fda938b7f93.jpg)

![735e69b031e3612239884dad240b4049462f3cd97b2e1a5e4c573e917fa0c30c.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/735e69b031e3612239884dad240b4049462f3cd97b2e1a5e4c573e917fa0c30c.jpg)

![7f3d193caa6cc5e70485dae605bd3f5a4e92882097be9baca71512dfd9677295.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/7f3d193caa6cc5e70485dae605bd3f5a4e92882097be9baca71512dfd9677295.jpg)

![b9d5b2f9c9246baa1331f6958c4bd100d5284dd60c8e8e95f4ed237fad8a101e.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/b9d5b2f9c9246baa1331f6958c4bd100d5284dd60c8e8e95f4ed237fad8a101e.jpg)

![bdf6e4e32d33c3a4f06728e31df02e9f81211bc4942f9dc219729996a8154ae5.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/bdf6e4e32d33c3a4f06728e31df02e9f81211bc4942f9dc219729996a8154ae5.jpg)

![dfc9eea39a210333002ed7c1f6e19d57fb6845e8dfc1785b108568402896bf31.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/dfc9eea39a210333002ed7c1f6e19d57fb6845e8dfc1785b108568402896bf31.jpg)

![e4a02831675766a5981553624e2504797041ff27ce893ab85543cb88954abd76.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/e4a02831675766a5981553624e2504797041ff27ce893ab85543cb88954abd76.jpg)

![f83351dc4234dedc442a03f76f97be3081acf2c00b861edc9af2101db5343db7.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/images/f83351dc4234dedc442a03f76f97be3081acf2c00b861edc9af2101db5343db7.jpg)

### Tables

![2f3128ee1366b12dee66b6ee1d8a5c7339262c6481785e6090938015b14ec591.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/tables/2f3128ee1366b12dee66b6ee1d8a5c7339262c6481785e6090938015b14ec591.jpg)

![d9817d399891030fb9180bd5f3dfeb86e9e36de595b25a2fedc0f18e3963193f.jpg](../icml_results/1931_Learning%20Distances%20from%20Data%20with%20Normalizing%20Flows%20and%20Score%20Matching/tables/d9817d399891030fb9180bd5f3dfeb86e9e36de595b25a2fedc0f18e3963193f.jpg)

## On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains


### Images

![312ff01ee9867cd2cade68b7438cb19ed020dff645436ddb80a8165af13a36bd.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/312ff01ee9867cd2cade68b7438cb19ed020dff645436ddb80a8165af13a36bd.jpg)

![840c5281e908a63588cf73b262c1f497c9c36413f7d042a1ef6e17b19463116e.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/840c5281e908a63588cf73b262c1f497c9c36413f7d042a1ef6e17b19463116e.jpg)

![8b1de2a309137623f9858b9cbd97f0e195ff5178918a4f34808e3d5c3d99edbf.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/8b1de2a309137623f9858b9cbd97f0e195ff5178918a4f34808e3d5c3d99edbf.jpg)

![ad510b8aea35c6bfd961116fae17c9ae69a0666bd24ce49c0983f93580ed6866.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/ad510b8aea35c6bfd961116fae17c9ae69a0666bd24ce49c0983f93580ed6866.jpg)

![c277aa12aa202e846313d40e0c65b9ddbd0a1f2f0873fd338047862feb8da452.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/c277aa12aa202e846313d40e0c65b9ddbd0a1f2f0873fd338047862feb8da452.jpg)

![d4950809524bde44d5368a14c9317e8202c7047f033c3a225655ed756c2f24be.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/d4950809524bde44d5368a14c9317e8202c7047f033c3a225655ed756c2f24be.jpg)

![e7208ed097809b3acf4574cac267f607890606bd4adff5743645c77f9fbe9fa6.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/e7208ed097809b3acf4574cac267f607890606bd4adff5743645c77f9fbe9fa6.jpg)

![ebb5a8f6b66cb10367b687e2485f868c3b30e51a4fd9ccbe6e8a6f00c7ce10e3.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/images/ebb5a8f6b66cb10367b687e2485f868c3b30e51a4fd9ccbe6e8a6f00c7ce10e3.jpg)

### Tables

![03bb3ae225f48361292fa728ad4bae4e5c87deb50f5f9f5aaab1d2d83b923721.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/03bb3ae225f48361292fa728ad4bae4e5c87deb50f5f9f5aaab1d2d83b923721.jpg)

![2826951283c4767ac33240ff410044637aac5fc0ff2e2a0be3ee059b2a0ad531.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/2826951283c4767ac33240ff410044637aac5fc0ff2e2a0be3ee059b2a0ad531.jpg)

![5a3f23b9390107ddd490ec458c3d023a5059eace7769ab766e0e59dfe8e35b3b.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/5a3f23b9390107ddd490ec458c3d023a5059eace7769ab766e0e59dfe8e35b3b.jpg)

![64ccbacacc11891554712c351b098a9ff57616407cea39686ed9e511bbcc088c.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/64ccbacacc11891554712c351b098a9ff57616407cea39686ed9e511bbcc088c.jpg)

![73652b0a4694de30d3432d039b512e97d77d15fa776bc0ac8eeb5953d76a57ed.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/73652b0a4694de30d3432d039b512e97d77d15fa776bc0ac8eeb5953d76a57ed.jpg)

![81b51a3fe20a84fb6f6f54e3b270b6f46a448bbabd5d14fa31bbacac756402c5.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/81b51a3fe20a84fb6f6f54e3b270b6f46a448bbabd5d14fa31bbacac756402c5.jpg)

![97fa2b76ee972decd0f5bfa9951fe578ffde824eff6e59735b921fedf5652d2b.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/97fa2b76ee972decd0f5bfa9951fe578ffde824eff6e59735b921fedf5652d2b.jpg)

![a03d44a14cfe428463eefcac6f9ac29f6c3267bcc54007de1426c12e81f21e42.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/a03d44a14cfe428463eefcac6f9ac29f6c3267bcc54007de1426c12e81f21e42.jpg)

![ca12fe90812c9ef5bf67e331d484a94717040e50861a749c824fb08b4be1ee06.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/ca12fe90812c9ef5bf67e331d484a94717040e50861a749c824fb08b4be1ee06.jpg)

![e71137320a8a0b56ad95754dd02517c569e56236882b0fea55b58d96ec042671.jpg](../icml_results/1932_On%20the%20Vulnerability%20of%20Applying%20Retrieval-Augmented%20Generation%20within%20Knowledge-Intensive%20Applicati/tables/e71137320a8a0b56ad95754dd02517c569e56236882b0fea55b58d96ec042671.jpg)

## DANCE: Dual Unbiased Expansion with Group-acquired Alignment for Out-of-distribution Graph Fairness Learning

### Images

![18073f4b20e46e6b2f2e30a54e318cbd77869efd012f778955fad1ab8855688f.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/images/18073f4b20e46e6b2f2e30a54e318cbd77869efd012f778955fad1ab8855688f.jpg)

![2548294f5b130075552f6ca090e318dbab276ba940ebd2e9ab59a68be0a0dc2a.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/images/2548294f5b130075552f6ca090e318dbab276ba940ebd2e9ab59a68be0a0dc2a.jpg)

![7036fb13c513ebee1418b9ffccaa3e3a718be89e0c5bf94b6918cd871f9246c0.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/images/7036fb13c513ebee1418b9ffccaa3e3a718be89e0c5bf94b6918cd871f9246c0.jpg)

![7a5935232dd36cec2925784576a51d52b0474d2f84555eababce90526c6126e0.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/images/7a5935232dd36cec2925784576a51d52b0474d2f84555eababce90526c6126e0.jpg)

![ba5a8dd2f730a5a7657e8ccdc2e0a5675ae9f5143b09a0917c76c4b80cdc8e54.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/images/ba5a8dd2f730a5a7657e8ccdc2e0a5675ae9f5143b09a0917c76c4b80cdc8e54.jpg)

![ee54561b605b82174f0f40917acb4936d8f2304439263f856993fd9ecc4464a4.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/images/ee54561b605b82174f0f40917acb4936d8f2304439263f856993fd9ecc4464a4.jpg)

### Tables

![0c07f576f160c84b522faeeeed142277e0ac42122b62cc389184ff15a2bd731f.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/tables/0c07f576f160c84b522faeeeed142277e0ac42122b62cc389184ff15a2bd731f.jpg)

![58ffe11c5d6cbe9c1a458938c7a33a2293a57a23a6edcfc6a38ac6c54b256802.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/tables/58ffe11c5d6cbe9c1a458938c7a33a2293a57a23a6edcfc6a38ac6c54b256802.jpg)

![8748d253e599c760f7ae81dee4a2e51256e050f739cc99c112f6a9ee0be7bd79.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/tables/8748d253e599c760f7ae81dee4a2e51256e050f739cc99c112f6a9ee0be7bd79.jpg)

![a48e5d4be9986ff5c165d28e737c57b5d5c7debb0909b85dcb50b77857360f27.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/tables/a48e5d4be9986ff5c165d28e737c57b5d5c7debb0909b85dcb50b77857360f27.jpg)

![fa599b2829e87901f34219e2610104b0c0ba8dbdc22d4aaf430015ccc86f55af.jpg](../icml_results/1933_DANCE_%20Dual%20Unbiased%20Expansion%20with%20Group-acquired%20Alignment%20for%20Out-of-distribution%20Graph%20Fairness%20/tables/fa599b2829e87901f34219e2610104b0c0ba8dbdc22d4aaf430015ccc86f55af.jpg)
