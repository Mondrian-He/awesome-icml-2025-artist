# ICML 2025 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 34934
- 📋 Total tables: 26167
- 📄 Total files: 61101

*Note: Equations have been filtered out and are not included.*

---

# ICML 2025 Main Papers - Part 65 of 100

## 目录 (Table of Contents)

1. [Learning In-context $n$-grams with Transformers: Sub-$n$-grams Are Near-Stationary Points](#Learning-In-context-n-grams-with-Transformers-Sub-n-grams-Are-Near-Stationary-Points)
2. [Demystifying Long Chain-of-Thought Reasoning](#Demystifying-Long-Chain-of-Thought-Reasoning)
3. [Nested Expectations with Kernel Quadrature](#Nested-Expectations-with-Kernel-Quadrature)
4. [Boosting Virtual Agent Learning and Reasoning: A Step-Wise, Multi-Dimensional, and Generalist Reward Model with Benchmark](#Boosting-Virtual-Agent-Learning-and-Reasoning-A-Step-Wise-Multi-Dimensional-and-Generalist-Reward-Model-with-Benchmark)
5. [Physics-informed Temporal Alignment for Auto-regressive PDE Foundation Models](#Physics-informed-Temporal-Alignment-for-Auto-regressive-PDE-Foundation-Models)
6. [Fundamental limits of learning in sequence multi-index models and deep attention networks: high-dimensional asymptotics and sharp thresholds](#Fundamental-limits-of-learning-in-sequence-multi-index-models-and-deep-attention-networks-high-dimensional-asymptotics-and-sharp-thresholds)
7. [Compositional Causal Reasoning Evaluation in Language Models](#Compositional-Causal-Reasoning-Evaluation-in-Language-Models)
8. [A Versatile Influence Function for Data Attribution with Non-Decomposable Loss](#A-Versatile-Influence-Function-for-Data-Attribution-with-Non-Decomposable-Loss)
9. [WyckoffDiff -- A Generative Diffusion Model for Crystal Symmetry](#WyckoffDiff-A-Generative-Diffusion-Model-for-Crystal-Symmetry)
10. [CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection](#CurvGAD-Leveraging-Curvature-for-Enhanced-Graph-Anomaly-Detection)
11. [When Do LLMs Help With Node Classification? A Comprehensive Analysis](#When-Do-LLMs-Help-With-Node-Classification-A-Comprehensive-Analysis)
12. [$\mathcal{V}ista\mathcal{DPO}$: Video Hierarchical Spatial-Temporal Direct Preference Optimization for Large Video Models](#mathcalVistamathcalDPO-Video-Hierarchical-Spatial-Temporal-Direct-Preference-Optimization-for-Large-Video-Models)
13. [Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization](#Optimal-Transport-Barycenter-via-Nonconvex-Concave-Minimax-Optimization)
14. [SafeMap: Robust HD Map Construction from Incomplete Observations](#SafeMap-Robust-HD-Map-Construction-from-Incomplete-Observations)
15. [Accurate Identification of Communication Between Multiple Interacting Neural Populations](#Accurate-Identification-of-Communication-Between-Multiple-Interacting-Neural-Populations)
16. [PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization](#PipeOffload-Improving-Scalability-of-Pipeline-Parallelism-with-Memory-Optimization)
17. [GRAIL: Graph Edit Distance and Node Alignment using LLM-Generated Code](#GRAIL-Graph-Edit-Distance-and-Node-Alignment-using-LLM-Generated-Code)
18. [When Model Knowledge meets Diffusion Model: Diffusion-assisted Data-free Image Synthesis with Alignment of Domain and Class](#When-Model-Knowledge-meets-Diffusion-Model-Diffusion-assisted-Data-free-Image-Synthesis-with-Alignment-of-Domain-and-Class)
19. [Bridging Fairness and Efficiency in Conformal Inference: A Surrogate-Assisted Group-Clustered Approach](#Bridging-Fairness-and-Efficiency-in-Conformal-Inference-A-Surrogate-Assisted-Group-Clustered-Approach)
20. [TinyMIG: Transferring Generalization from Vision Foundation Models to Single-Domain Medical Imaging](#TinyMIG-Transferring-Generalization-from-Vision-Foundation-Models-to-Single-Domain-Medical-Imaging)
21. [An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](#An-Efficient-Matrix-Multiplication-Algorithm-for-Accelerating-Inference-in-Binary-and-Ternary-Neural-Networks)
22. [AUTOCIRCUIT-RL: Reinforcement Learning-Driven LLM for Automated Circuit Topology Generation](#AUTOCIRCUIT-RL-Reinforcement-Learning-Driven-LLM-for-Automated-Circuit-Topology-Generation)
23. [Scalable Gaussian Processes with Latent Kronecker Structure](#Scalable-Gaussian-Processes-with-Latent-Kronecker-Structure)
24. [DMM: Distributed Matrix Mechanism for Differentially-Private Federated Learning Based on Constant-Overhead Linear Secret Resharing](#DMM-Distributed-Matrix-Mechanism-for-Differentially-Private-Federated-Learning-Based-on-Constant-Overhead-Linear-Secret-Resharing)
25. [Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning](#Underestimated-Privacy-Risks-for-Minority-Populations-in-Large-Language-Model-Unlearning)
26. [Balancing Preservation and Modification: A Region and Semantic Aware Metric for Instruction-Based Image Editing](#Balancing-Preservation-and-Modification-A-Region-and-Semantic-Aware-Metric-for-Instruction-Based-Image-Editing)
27. [MARS: Unleashing the Power of Variance Reduction for Training Large Models](#MARS-Unleashing-the-Power-of-Variance-Reduction-for-Training-Large-Models)
28. [Improving Value Estimation Critically Enhances Vanilla Policy Gradient](#Improving-Value-Estimation-Critically-Enhances-Vanilla-Policy-Gradient)
29. [Refining Adaptive Zeroth-Order Optimization at Ease](#Refining-Adaptive-Zeroth-Order-Optimization-at-Ease)
30. [Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks](#Sparse-Spectral-Training-and-Inference-on-Euclidean-and-Hyperbolic-Neural-Networks)
31. [MDDM: Practical Message-Driven Generative Image Steganography Based on Diffusion Models](#MDDM-Practical-Message-Driven-Generative-Image-Steganography-Based-on-Diffusion-Models)
32. [Pixel-level Certified Explanations via Randomized Smoothing](#Pixel-level-Certified-Explanations-via-Randomized-Smoothing)
33. [Agent-as-a-Judge: Evaluate Agents with Agents](#Agent-as-a-Judge-Evaluate-Agents-with-Agents)

---


## Learning In-context $n$-grams with Transformers: Sub-$n$-grams Are Near-Stationary Points

### Images

![420106757c34294f06fc7e77cbb7a5790279990c9ef5ae7f8e0da82a7ac7dd0d.jpg](../icml_results/2132_On%20the%20Interplay%20between%20Graph%20Structure%20and%20Learning%20Algorithms%20in%20Graph%20Neural%20Networks/images/420106757c34294f06fc7e77cbb7a5790279990c9ef5ae7f8e0da82a7ac7dd0d.jpg)

![45390bb30c84e3423ae604149daecdc091e9b7f3eb77abfe66c1c825c4dbbba5.jpg](../icml_results/2132_On%20the%20Interplay%20between%20Graph%20Structure%20and%20Learning%20Algorithms%20in%20Graph%20Neural%20Networks/images/45390bb30c84e3423ae604149daecdc091e9b7f3eb77abfe66c1c825c4dbbba5.jpg)

![d01bb650273869b20446bacedd05cad00f972fb66384a7c64d646bd801104067.jpg](../icml_results/2132_On%20the%20Interplay%20between%20Graph%20Structure%20and%20Learning%20Algorithms%20in%20Graph%20Neural%20Networks/images/d01bb650273869b20446bacedd05cad00f972fb66384a7c64d646bd801104067.jpg)

## Learning In-context $n$-grams with Transformers: Sub-$n$-grams Are Near-Stationary Points


### Images

![17924bd0739c9edc1a1e35306b57fbd747a9968243f4abad02331780c78e338e.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/17924bd0739c9edc1a1e35306b57fbd747a9968243f4abad02331780c78e338e.jpg)

![5118216765a6b72ade18e85c3d416cc2834a2979fa765782a3883c36aafab68b.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/5118216765a6b72ade18e85c3d416cc2834a2979fa765782a3883c36aafab68b.jpg)

![74c29af13919ffe7dbc71701c1396bfcce35a7e19887f02133956d53b8149c47.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/74c29af13919ffe7dbc71701c1396bfcce35a7e19887f02133956d53b8149c47.jpg)

![7cefbed0e21bf0ac5ae3c214b1a6e048a0c3700974bdca8ca4b979df48dd8b62.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/7cefbed0e21bf0ac5ae3c214b1a6e048a0c3700974bdca8ca4b979df48dd8b62.jpg)

![98c27250493db3bcefb88bbea104b96288447203e501b41d70db075ced4d349e.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/98c27250493db3bcefb88bbea104b96288447203e501b41d70db075ced4d349e.jpg)

![99f841ba2ef450e786683b2b8a08e6d6dc08c9aa315401f9128df9cfd1c0a8d6.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/99f841ba2ef450e786683b2b8a08e6d6dc08c9aa315401f9128df9cfd1c0a8d6.jpg)

![dc1ce50f82dca09249687f4095867b5e22b88987a6e4a43426a342163be85039.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/dc1ce50f82dca09249687f4095867b5e22b88987a6e4a43426a342163be85039.jpg)

![ef0eaf25c81fa4b5bf8e4a6c36473efeb8260e4f2e063b1bf3ee3ceffca89e64.jpg](../icml_results/2133_Learning%20In-context%20%24n%24-grams%20with%20Transformers_%20Sub-%24n%24-grams%20Are%20Near-Stationary%20Points/images/ef0eaf25c81fa4b5bf8e4a6c36473efeb8260e4f2e063b1bf3ee3ceffca89e64.jpg)

## Demystifying Long Chain-of-Thought Reasoning


### Images

![24bc28bb3dc7c52419810acd6d7e7dd5fbf1703492b7781489cc3f30aa226544.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/24bc28bb3dc7c52419810acd6d7e7dd5fbf1703492b7781489cc3f30aa226544.jpg)

![286175d7a7af79079e94d72278de6964a8188d4b1dc0dd00e95eca0652fb9ce3.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/286175d7a7af79079e94d72278de6964a8188d4b1dc0dd00e95eca0652fb9ce3.jpg)

![2ada72e3b6ef14ca5898b2e2cbaea11306ba64165ed0b1af6cf72a5c9fb31129.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/2ada72e3b6ef14ca5898b2e2cbaea11306ba64165ed0b1af6cf72a5c9fb31129.jpg)

![371949c251f2b8ba95d30ef8bbc547a04cf281ad7683fcb82d2c992525fe4bc8.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/371949c251f2b8ba95d30ef8bbc547a04cf281ad7683fcb82d2c992525fe4bc8.jpg)

![39cb670633cf5fbf8a0fbee554cf4699d92ab28b4424519f8c693482c4ec517e.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/39cb670633cf5fbf8a0fbee554cf4699d92ab28b4424519f8c693482c4ec517e.jpg)

![40260ca0eb34b68f844d8760bb3acbe490c570a85dab14583da3331c6e687762.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/40260ca0eb34b68f844d8760bb3acbe490c570a85dab14583da3331c6e687762.jpg)

![46ef500543d7f09ea9e1601969479f8c033490f499aed008e619db890ee7ba15.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/46ef500543d7f09ea9e1601969479f8c033490f499aed008e619db890ee7ba15.jpg)

![4bb200ee52beab386c31ada2b1b5d3f77bb1500fa072891668be1700c5329b15.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/4bb200ee52beab386c31ada2b1b5d3f77bb1500fa072891668be1700c5329b15.jpg)

![82ed41aa8f0c2419f110ef54ac7cd5c0d5ea5945b4fbc1677aa5c0f871783860.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/82ed41aa8f0c2419f110ef54ac7cd5c0d5ea5945b4fbc1677aa5c0f871783860.jpg)

![8c8580b42f74498b7701e4c9336ffc649ac015b658752bee35b6b39a87341685.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/8c8580b42f74498b7701e4c9336ffc649ac015b658752bee35b6b39a87341685.jpg)

![8d45252acdeee410690f69c3d014d169cb29ed8570b3184f7cbdb1eef7e02533.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/8d45252acdeee410690f69c3d014d169cb29ed8570b3184f7cbdb1eef7e02533.jpg)

![d6f29ce848e6c68f1dace51662c4fb91733df9486083ff72e0ef1c5f5bb5cda6.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/d6f29ce848e6c68f1dace51662c4fb91733df9486083ff72e0ef1c5f5bb5cda6.jpg)

![d9efabe74ad69cba9907f82f996b8e1106336dd7cf3944bf354a847284b39238.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/d9efabe74ad69cba9907f82f996b8e1106336dd7cf3944bf354a847284b39238.jpg)

![e11e41aa1f0dd73f2bb3de523d86f87a79c8c24eb2050c65bc417018cbbfac42.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/e11e41aa1f0dd73f2bb3de523d86f87a79c8c24eb2050c65bc417018cbbfac42.jpg)

![f52872ced47bf7480fa7eaee79aa002721bbe7a346f39e5a48467e738e425491.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/f52872ced47bf7480fa7eaee79aa002721bbe7a346f39e5a48467e738e425491.jpg)

![f675cad1fefbd74f3a27a1c41a695f46fb9c6036abb759b37b2bd7bb5be81f05.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/images/f675cad1fefbd74f3a27a1c41a695f46fb9c6036abb759b37b2bd7bb5be81f05.jpg)

### Tables

![0396d0876e9a59a47b3b596e1870dbb2d0b775ae3e0a917cb7dc9a1b6b8f03db.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/0396d0876e9a59a47b3b596e1870dbb2d0b775ae3e0a917cb7dc9a1b6b8f03db.jpg)

![044afd22860a564b44dbc47e29293f6cfb8c4e865e77bb385e15c85a3d84902f.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/044afd22860a564b44dbc47e29293f6cfb8c4e865e77bb385e15c85a3d84902f.jpg)

![088606283e59f846379fd995a3372c2a948f669878a40723ae141f4207ac7153.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/088606283e59f846379fd995a3372c2a948f669878a40723ae141f4207ac7153.jpg)

![2782d629faa98ec14f77d39b7ad12499224dccde026972e532f072e6e1e8a1a4.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/2782d629faa98ec14f77d39b7ad12499224dccde026972e532f072e6e1e8a1a4.jpg)

![28781acaf6d86fee1faf1fa42f12cf24dfb8486b1e9556e41182e729af4d43d0.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/28781acaf6d86fee1faf1fa42f12cf24dfb8486b1e9556e41182e729af4d43d0.jpg)

![32205408d07cde03a906140093d5931794af6082a82cf1d6f920338db8b72c47.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/32205408d07cde03a906140093d5931794af6082a82cf1d6f920338db8b72c47.jpg)

![5be6c194bf3d9929721e74fbd179d84c6bd50cbc533fd681cfa2905909dea4b0.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/5be6c194bf3d9929721e74fbd179d84c6bd50cbc533fd681cfa2905909dea4b0.jpg)

![6398c957d75b5a041be4849eb1d86c21684c79b30b43e4e957c3d1ed840e0412.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/6398c957d75b5a041be4849eb1d86c21684c79b30b43e4e957c3d1ed840e0412.jpg)

![6ab820f698ecc90e9847bb779a71195380cbec0b42da9e4a79be79859042c732.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/6ab820f698ecc90e9847bb779a71195380cbec0b42da9e4a79be79859042c732.jpg)

![7e1fb07df32b14806f11196983810466683970660d4d3fbefaffb4405c96ba91.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/7e1fb07df32b14806f11196983810466683970660d4d3fbefaffb4405c96ba91.jpg)

![898503c78379e128c6416945f611d1bce53a27cd988da6a7602dc0de74814ef3.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/898503c78379e128c6416945f611d1bce53a27cd988da6a7602dc0de74814ef3.jpg)

![92bcf92d29506a2b0e20245957d7a986fac37d06ebf666735be12c0ef94a2a93.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/92bcf92d29506a2b0e20245957d7a986fac37d06ebf666735be12c0ef94a2a93.jpg)

![943b326c61d266dff1f951c400439832d34e112ce8a2eeb8b9dcd98bbf37cd6a.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/943b326c61d266dff1f951c400439832d34e112ce8a2eeb8b9dcd98bbf37cd6a.jpg)

![a10e4cebd7e10aea7a6aeb7a92364a05437c49be356cf64bc88ff9c21082bf9f.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/a10e4cebd7e10aea7a6aeb7a92364a05437c49be356cf64bc88ff9c21082bf9f.jpg)

![a32a98e8a63c011d0aa100b1ff07e5ddefb33ba575c7253457d31d4118d45230.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/a32a98e8a63c011d0aa100b1ff07e5ddefb33ba575c7253457d31d4118d45230.jpg)

![a5a974739d18b55c9f583f13c2844c4ec97cf7987cac3894c9c5c86f81e45bc3.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/a5a974739d18b55c9f583f13c2844c4ec97cf7987cac3894c9c5c86f81e45bc3.jpg)

![ef4c595e07c878562e4ea6a0b8ed1449a88ed90f4baac821610020a91cfee869.jpg](../icml_results/2134_Demystifying%20Long%20Chain-of-Thought%20Reasoning/tables/ef4c595e07c878562e4ea6a0b8ed1449a88ed90f4baac821610020a91cfee869.jpg)

## Nested Expectations with Kernel Quadrature


### Images

![2884f8cd9eaec9afcdcc7b36fd56f8355858e1044c4c45f5e29ec65ef8cb22e6.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/images/2884f8cd9eaec9afcdcc7b36fd56f8355858e1044c4c45f5e29ec65ef8cb22e6.jpg)

![2c65e1da86d2376c7c8e28f606e4261c3df29942e1cf795fd3072cab6b6c503d.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/images/2c65e1da86d2376c7c8e28f606e4261c3df29942e1cf795fd3072cab6b6c503d.jpg)

![4b3e9a66824db720b3b982e8825291094e1a24ca29a5cdbe109b020ac4c739b6.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/images/4b3e9a66824db720b3b982e8825291094e1a24ca29a5cdbe109b020ac4c739b6.jpg)

![6729d19d1c52e67c86b5d45e36c7ac2e2040f465db58812b211e5a0993203c6d.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/images/6729d19d1c52e67c86b5d45e36c7ac2e2040f465db58812b211e5a0993203c6d.jpg)

![9c041f85fb05e2d00d08f35fe0087e09730647b53f0adab9cbfbf3aecd78f375.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/images/9c041f85fb05e2d00d08f35fe0087e09730647b53f0adab9cbfbf3aecd78f375.jpg)

![b884e28a41c23f58ca520ed2f177644f4d661426b4dcf63db0432e4d57925677.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/images/b884e28a41c23f58ca520ed2f177644f4d661426b4dcf63db0432e4d57925677.jpg)

### Tables

![0266f4c9fab95cb55a4affc661d41d5d7caa66151ce9a389b93ef054d8b13dc5.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/tables/0266f4c9fab95cb55a4affc661d41d5d7caa66151ce9a389b93ef054d8b13dc5.jpg)

![8586fe0f57ef55bf504974c0755aac548424030bbe274b2da79c16cb9f4f95b1.jpg](../icml_results/2135_Nested%20Expectations%20with%20Kernel%20Quadrature/tables/8586fe0f57ef55bf504974c0755aac548424030bbe274b2da79c16cb9f4f95b1.jpg)

## Boosting Virtual Agent Learning and Reasoning: A Step-Wise, Multi-Dimensional, and Generalist Reward Model with Benchmark


### Images

![0080b21abe907c15251053da3a09f7e4c53bb87cffd829ec6357c0438492f56f.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/0080b21abe907c15251053da3a09f7e4c53bb87cffd829ec6357c0438492f56f.jpg)

![08a94fee8d0e62cbc095b3409da41a87c2cf4dd45bfa3611d8047f91895e2ed4.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/08a94fee8d0e62cbc095b3409da41a87c2cf4dd45bfa3611d8047f91895e2ed4.jpg)

![1341c654ea3c7fde97222e54d861a6a318e2dc687c46ea173fa33c80092f0ae0.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/1341c654ea3c7fde97222e54d861a6a318e2dc687c46ea173fa33c80092f0ae0.jpg)

![17378768debed73ed5a70e738677978156ef101907a88468f0b5fad0f7b61e0f.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/17378768debed73ed5a70e738677978156ef101907a88468f0b5fad0f7b61e0f.jpg)

![1af2fff0be7310dae73b72594f063fe78d2c4103d24ecb494bee0822cd7d55da.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/1af2fff0be7310dae73b72594f063fe78d2c4103d24ecb494bee0822cd7d55da.jpg)

![64b0aba1f30bfd9be1a6621a2f5c9331352d5840e3bfba4bc5f62d4b90d3b6b0.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/64b0aba1f30bfd9be1a6621a2f5c9331352d5840e3bfba4bc5f62d4b90d3b6b0.jpg)

![784f70b148ad285cf1e5e7bc3d789b88250cca668c1b237ff1c41fe71243560d.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/784f70b148ad285cf1e5e7bc3d789b88250cca668c1b237ff1c41fe71243560d.jpg)

![8e5e39d9b34f343a4ccfa5a708bc399ce5cfc7377d7f96bfea672c8bfd133afb.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/8e5e39d9b34f343a4ccfa5a708bc399ce5cfc7377d7f96bfea672c8bfd133afb.jpg)

![c525717b318db2dfd2e6ebbba78f6b156efe7bf81b1ba043771d722478611db9.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/c525717b318db2dfd2e6ebbba78f6b156efe7bf81b1ba043771d722478611db9.jpg)

![ca46a1cbe74ae935ce00a1eae6c8e73229bf48a042d5fb5a1eb39958e86ce5a2.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/ca46a1cbe74ae935ce00a1eae6c8e73229bf48a042d5fb5a1eb39958e86ce5a2.jpg)

![cae5b4c3b3944a39dbca7a21f217a373300d4f64d671cacafef2f581e5797fd4.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/cae5b4c3b3944a39dbca7a21f217a373300d4f64d671cacafef2f581e5797fd4.jpg)

![e76d817c132ebd93ee9394e7f32210b56fd84d07d3f82a6922624dd4e64ad7da.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/images/e76d817c132ebd93ee9394e7f32210b56fd84d07d3f82a6922624dd4e64ad7da.jpg)

### Tables

![691ac02cd42b70666c920e30891f6c709b1d70fb1e9d3e53d6b65aee7f65bd74.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/tables/691ac02cd42b70666c920e30891f6c709b1d70fb1e9d3e53d6b65aee7f65bd74.jpg)

![6f88b0bbb00c7555eb22ad3c1f62b3c2696a21014bc936e73d133809689a8d78.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/tables/6f88b0bbb00c7555eb22ad3c1f62b3c2696a21014bc936e73d133809689a8d78.jpg)

![73e9705c89598555ded5cec21eaeffe7abbe328205127eda8848fa2e5f8c313e.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/tables/73e9705c89598555ded5cec21eaeffe7abbe328205127eda8848fa2e5f8c313e.jpg)

![95109a37192aa769b534d97149dd34cc06e1924e5f7e5fed538b6a3bf77eeda7.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/tables/95109a37192aa769b534d97149dd34cc06e1924e5f7e5fed538b6a3bf77eeda7.jpg)

![c8f61dcc9875c01f1d020b77fbdfb4f39cd8065cbe170198d1b371c430543d32.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/tables/c8f61dcc9875c01f1d020b77fbdfb4f39cd8065cbe170198d1b371c430543d32.jpg)

![ce0e35c4bcfdfdcc4282c554220eacb84a45ea91a9e0e1083eb9e3a64f8850c7.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/tables/ce0e35c4bcfdfdcc4282c554220eacb84a45ea91a9e0e1083eb9e3a64f8850c7.jpg)

![e08ce71ec5aaefcf395ce50884ec9c5de183f186043d4769a6630b41285a1da8.jpg](../icml_results/2136_Boosting%20Virtual%20Agent%20Learning%20and%20Reasoning_%20A%20Step-Wise%2C%20Multi-Dimensional%2C%20and%20Generalist%20Reward/tables/e08ce71ec5aaefcf395ce50884ec9c5de183f186043d4769a6630b41285a1da8.jpg)

## Physics-informed Temporal Alignment for Auto-regressive PDE Foundation Models


### Images

![071bcf22ed1fc219ea807d32cf32f18037853ec699243a9074e1864c53f5416e.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/071bcf22ed1fc219ea807d32cf32f18037853ec699243a9074e1864c53f5416e.jpg)

![0814aca6c56a32e990daa5f19fa5f7296d71a7d687132068515c13c9e4223226.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/0814aca6c56a32e990daa5f19fa5f7296d71a7d687132068515c13c9e4223226.jpg)

![0bfc11e5c827c0c8745f2de6d0b4470c50c679bc1034c4a281c451a06a176976.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/0bfc11e5c827c0c8745f2de6d0b4470c50c679bc1034c4a281c451a06a176976.jpg)

![12f5ee398abb59de19645ed9caf880fa98f88f5224411662679514f9d73c76d8.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/12f5ee398abb59de19645ed9caf880fa98f88f5224411662679514f9d73c76d8.jpg)

![1546304b58f63c84a53fc8a0e64d6cc39104c6e8be63e2218bae50a9d27f1534.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/1546304b58f63c84a53fc8a0e64d6cc39104c6e8be63e2218bae50a9d27f1534.jpg)

![207af7a8f8efac37b3ac20e31bf15ac586dd46ebbf77fb69c51e11daae037811.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/207af7a8f8efac37b3ac20e31bf15ac586dd46ebbf77fb69c51e11daae037811.jpg)

![324450cb850ece2550f21d4f39404bd18b96c40438ea8f248eca8e10233bda2a.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/324450cb850ece2550f21d4f39404bd18b96c40438ea8f248eca8e10233bda2a.jpg)

![332db3c469bfe15d96ca5fd289db186f5c807d99f0eceec63ca7af14112db413.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/332db3c469bfe15d96ca5fd289db186f5c807d99f0eceec63ca7af14112db413.jpg)

![47b84f9ed40d284f2436f7d7a33c182e6331a0d5b7467d16a783aa2b6a98300f.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/47b84f9ed40d284f2436f7d7a33c182e6331a0d5b7467d16a783aa2b6a98300f.jpg)

![493732199a75386cead92dcb7a0e80edc1f86cca6f8040ab129efb718f28a121.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/493732199a75386cead92dcb7a0e80edc1f86cca6f8040ab129efb718f28a121.jpg)

![4dbfd69ae0aeab0a6feb4caba0d595c0e26bf922b785163c6cae427d3a0a6a9e.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/4dbfd69ae0aeab0a6feb4caba0d595c0e26bf922b785163c6cae427d3a0a6a9e.jpg)

![5638703a28c31249d6928e29c27c44dd8d233f0258df977653d2ba2f5b787ade.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/5638703a28c31249d6928e29c27c44dd8d233f0258df977653d2ba2f5b787ade.jpg)

![569b4f0970ea88de0d83608f03fd053a67b2f77b5125ebed3547bf00934d0410.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/569b4f0970ea88de0d83608f03fd053a67b2f77b5125ebed3547bf00934d0410.jpg)

![6afed3484823dd662b8e61270826623afa1bff98173c4ff97ae0bd61aefa7d37.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/6afed3484823dd662b8e61270826623afa1bff98173c4ff97ae0bd61aefa7d37.jpg)

![735ad09f591b7c2cb28a7c0bbeb61b88a5d799bda9aa584eb5725a660a5ea2db.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/735ad09f591b7c2cb28a7c0bbeb61b88a5d799bda9aa584eb5725a660a5ea2db.jpg)

![8eca7c470467bad20408c9fc15483c127568bfbb6f645354433c2286f6c61508.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/8eca7c470467bad20408c9fc15483c127568bfbb6f645354433c2286f6c61508.jpg)

![a531570fe97e115d47e9eac00cb71da5c48c3196e8871db7a81bdf834028a322.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/a531570fe97e115d47e9eac00cb71da5c48c3196e8871db7a81bdf834028a322.jpg)

![a7eba36fabc2c67bde8f6d63acf040e6d884a421b12b13a0184acee8fbc4f6b9.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/a7eba36fabc2c67bde8f6d63acf040e6d884a421b12b13a0184acee8fbc4f6b9.jpg)

![ab3d12d645bec42ff20f73c16e83f5bfcaeedb5266b74970a94e16144d151885.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/ab3d12d645bec42ff20f73c16e83f5bfcaeedb5266b74970a94e16144d151885.jpg)

![aee93ef8edb8073ae89e39f932f679bddc477130dd2cf78ef830f013c63fc896.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/aee93ef8edb8073ae89e39f932f679bddc477130dd2cf78ef830f013c63fc896.jpg)

![c35c2dcb86d6100f264361256a51f693e233f04a7f2a793cb1a87875b115cab9.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/c35c2dcb86d6100f264361256a51f693e233f04a7f2a793cb1a87875b115cab9.jpg)

![cb5148eae2411a97077dc9af596dc671246c152e3962b2a79aa777452edc380a.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/cb5148eae2411a97077dc9af596dc671246c152e3962b2a79aa777452edc380a.jpg)

![cb7d11b4d1c89bbc3bc61b4092d44cd8ce43e16d88515d8613c33ba239ec0a22.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/cb7d11b4d1c89bbc3bc61b4092d44cd8ce43e16d88515d8613c33ba239ec0a22.jpg)

![e1b258f4fe4f457c03dfb095f79d82708ee3af467b352dcc02649e8a9e48982c.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/images/e1b258f4fe4f457c03dfb095f79d82708ee3af467b352dcc02649e8a9e48982c.jpg)

### Tables

![0f584a086188dba328d8f64d7cb5085773dfe6fd965eecb694ead3ba53b6724a.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/0f584a086188dba328d8f64d7cb5085773dfe6fd965eecb694ead3ba53b6724a.jpg)

![2365e7455a38d309a5c69bb5dc260f9bd21bca289a402f7d7d9d472b455247d1.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/2365e7455a38d309a5c69bb5dc260f9bd21bca289a402f7d7d9d472b455247d1.jpg)

![2a9645a881774c2658424640d6c8b4593d1516b5c28a598eb30a0ff83217f0d3.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/2a9645a881774c2658424640d6c8b4593d1516b5c28a598eb30a0ff83217f0d3.jpg)

![3b22879c5fcca7b1e27f7c778493c9791911b1edcdb7ea70db69bac5202a59a1.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/3b22879c5fcca7b1e27f7c778493c9791911b1edcdb7ea70db69bac5202a59a1.jpg)

![478d2114a98cbd4e87c0b57c4743bda1847e63e2c35a753d85c6eb8006337b93.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/478d2114a98cbd4e87c0b57c4743bda1847e63e2c35a753d85c6eb8006337b93.jpg)

![47ecffe013a60751ef56cefe02f8cd666b2b00667c9bc941d743ee590567bc28.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/47ecffe013a60751ef56cefe02f8cd666b2b00667c9bc941d743ee590567bc28.jpg)

![61156e8a0c5a63a25d10a6248f1065e5275c3539b79bffc447ffde4a1732c71a.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/61156e8a0c5a63a25d10a6248f1065e5275c3539b79bffc447ffde4a1732c71a.jpg)

![666f8902970379584023f9bc413e80bb86790674f3c6afcc91712ad66f388e99.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/666f8902970379584023f9bc413e80bb86790674f3c6afcc91712ad66f388e99.jpg)

![6dd2109f0883fd2f12095a4c9050bce1fdca7b8225907959919a691c6ce221dc.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/6dd2109f0883fd2f12095a4c9050bce1fdca7b8225907959919a691c6ce221dc.jpg)

![71f24464c046e056d3b2d4ec5c41b40670510b3b37a84812644edf7c46f0c832.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/71f24464c046e056d3b2d4ec5c41b40670510b3b37a84812644edf7c46f0c832.jpg)

![7f278bbc332ca9997a0443314cf875cb22937fa2c5a5af8f0934da4c1b2bb32d.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/7f278bbc332ca9997a0443314cf875cb22937fa2c5a5af8f0934da4c1b2bb32d.jpg)

![89a10f3b30cb52ff0cc80e800637fbd8b8ad96e9353c5fb65beaa4686ef06233.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/89a10f3b30cb52ff0cc80e800637fbd8b8ad96e9353c5fb65beaa4686ef06233.jpg)

![9e73d6564a222173217a87d2263864f4ebcfaf4d39aea3b7e3af7f765d3ecdc5.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/9e73d6564a222173217a87d2263864f4ebcfaf4d39aea3b7e3af7f765d3ecdc5.jpg)

![b1765a65c5bf324cd2e1922b0a57a4c13ae254ac19531a0753e55449e810b4ea.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/b1765a65c5bf324cd2e1922b0a57a4c13ae254ac19531a0753e55449e810b4ea.jpg)

![bc63a8e52a553c82ca771813897ab4040e982a551cb868184c642638dbdafb21.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/bc63a8e52a553c82ca771813897ab4040e982a551cb868184c642638dbdafb21.jpg)

![e06e32832147a9beea57324affbe9e559672623d962c6ebb79623cc2c1009de2.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/e06e32832147a9beea57324affbe9e559672623d962c6ebb79623cc2c1009de2.jpg)

![e1136e28aa4bd33d784fd0f9181e7cd4dcaf974df93de8a1506a3f142210e569.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/e1136e28aa4bd33d784fd0f9181e7cd4dcaf974df93de8a1506a3f142210e569.jpg)

![e2973c1a10ae9ff0e14c95a62b440eb02724335f5ce64e11e7b59771192177f0.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/e2973c1a10ae9ff0e14c95a62b440eb02724335f5ce64e11e7b59771192177f0.jpg)

![e9d4912504111b7d1b73c81b41ddd2be6fd8ed220c45c65057f8e54a5336f3e4.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/e9d4912504111b7d1b73c81b41ddd2be6fd8ed220c45c65057f8e54a5336f3e4.jpg)

![edfe9cd9a06be2776f975b373179e29bb62c724642d479c379ab80f16e983a9d.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/edfe9cd9a06be2776f975b373179e29bb62c724642d479c379ab80f16e983a9d.jpg)

![f246533bed0135d83d27cd12775e654a0bb93977d10d6dd5a21121debb5898a3.jpg](../icml_results/2137_Physics-informed%20Temporal%20Alignment%20for%20Auto-regressive%20PDE%20Foundation%20Models/tables/f246533bed0135d83d27cd12775e654a0bb93977d10d6dd5a21121debb5898a3.jpg)

## Fundamental limits of learning in sequence multi-index models and deep attention networks: high-dimensional asymptotics and sharp thresholds


### Images

![145333400f3f6a348b30a512d09e95a99d9476a68d7bb6143473e11caab7598f.jpg](../icml_results/2138_Fundamental%20limits%20of%20learning%20in%20sequence%20multi-index%20models%20and%20deep%20attention%20networks_%20high-dime/images/145333400f3f6a348b30a512d09e95a99d9476a68d7bb6143473e11caab7598f.jpg)

![90eb94e451ecf2fd46dd12e5d9cc75db569c9ea5222f3d11a0b3bc0c16008175.jpg](../icml_results/2138_Fundamental%20limits%20of%20learning%20in%20sequence%20multi-index%20models%20and%20deep%20attention%20networks_%20high-dime/images/90eb94e451ecf2fd46dd12e5d9cc75db569c9ea5222f3d11a0b3bc0c16008175.jpg)

![adbbd1e6518fc05e40037995d1d3dee219dc8555e7c6a42f65ceab54d0947e5f.jpg](../icml_results/2138_Fundamental%20limits%20of%20learning%20in%20sequence%20multi-index%20models%20and%20deep%20attention%20networks_%20high-dime/images/adbbd1e6518fc05e40037995d1d3dee219dc8555e7c6a42f65ceab54d0947e5f.jpg)

![b1597040a024ca7bf82f89cbc112478bbd792e67679ebfe78871747570076ca0.jpg](../icml_results/2138_Fundamental%20limits%20of%20learning%20in%20sequence%20multi-index%20models%20and%20deep%20attention%20networks_%20high-dime/images/b1597040a024ca7bf82f89cbc112478bbd792e67679ebfe78871747570076ca0.jpg)

![f82a81e60ce102d28b6411af11afebea42705d395812677456249f3728953c03.jpg](../icml_results/2138_Fundamental%20limits%20of%20learning%20in%20sequence%20multi-index%20models%20and%20deep%20attention%20networks_%20high-dime/images/f82a81e60ce102d28b6411af11afebea42705d395812677456249f3728953c03.jpg)

## Compositional Causal Reasoning Evaluation in Language Models


### Images

![15fb60c2bc9ddae5dd7726b2bb03f84bb38a69555d382e1ebc6bb4d54b7fdad3.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/15fb60c2bc9ddae5dd7726b2bb03f84bb38a69555d382e1ebc6bb4d54b7fdad3.jpg)

![1663076b6b33ab9ccc3a4d0178454cbed22ff3581fb7d751f39f9b246ace8828.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/1663076b6b33ab9ccc3a4d0178454cbed22ff3581fb7d751f39f9b246ace8828.jpg)

![2333a0f8523e6ce639e99aa0b72a3ae1f4a45417022fd3cf177eab98f8041f95.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/2333a0f8523e6ce639e99aa0b72a3ae1f4a45417022fd3cf177eab98f8041f95.jpg)

![2859ff805980c22a1f345cfaedbf25961a28608e23a677acb3f0445d484394e8.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/2859ff805980c22a1f345cfaedbf25961a28608e23a677acb3f0445d484394e8.jpg)

![296ef226989b8603fc83e5306fa922416d6c97720976754bcd9dff365405b374.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/296ef226989b8603fc83e5306fa922416d6c97720976754bcd9dff365405b374.jpg)

![2b38350b97939dedcab702899e4883b5248673eec2f52cd214d7ae1eab6a32c6.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/2b38350b97939dedcab702899e4883b5248673eec2f52cd214d7ae1eab6a32c6.jpg)

![350ebe6dc3b9c9a5cc46c02d611063dbcaab3a993588b8a489a34595a89ea867.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/350ebe6dc3b9c9a5cc46c02d611063dbcaab3a993588b8a489a34595a89ea867.jpg)

![37b3acbc45804bd8f2d8a7ef9166ce4dc627327c8ee88268178e99ee568816d3.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/37b3acbc45804bd8f2d8a7ef9166ce4dc627327c8ee88268178e99ee568816d3.jpg)

![42bea27d4fd68fb7d33134a92ab89b9114db73898dc5960211099a160bd5584c.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/42bea27d4fd68fb7d33134a92ab89b9114db73898dc5960211099a160bd5584c.jpg)

![4f7e27b71b236c2a6044c3f8cd5fff89303924cf9fb57594b422bfe0ea214f49.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/4f7e27b71b236c2a6044c3f8cd5fff89303924cf9fb57594b422bfe0ea214f49.jpg)

![5e74464c7ed8d0073aa031a0935acfba768560a139efcf3fcb920e433a120d4a.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/5e74464c7ed8d0073aa031a0935acfba768560a139efcf3fcb920e433a120d4a.jpg)

![675c2f2d761af4c25ec96d8c89232a572c151e235813d991d33a8167dac06e25.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/675c2f2d761af4c25ec96d8c89232a572c151e235813d991d33a8167dac06e25.jpg)

![6fa7b730966fd8247c2df15fffc1378e41ae2e46bbf43e781819cf5e7002955f.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/6fa7b730966fd8247c2df15fffc1378e41ae2e46bbf43e781819cf5e7002955f.jpg)

![733095ce913b16db85e129c79b1e517bbb1b51743cabf97c45381172a6b10157.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/733095ce913b16db85e129c79b1e517bbb1b51743cabf97c45381172a6b10157.jpg)

![7bcaf7937ef2b0c1bd9e56ec9fc9cab742a9750b37a16a4965d5ec4410f30acc.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/7bcaf7937ef2b0c1bd9e56ec9fc9cab742a9750b37a16a4965d5ec4410f30acc.jpg)

![867704b36aac82ae44e69f7bd2f861efc0be03591b75edfbf9f41a057d38e739.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/867704b36aac82ae44e69f7bd2f861efc0be03591b75edfbf9f41a057d38e739.jpg)

![a7625173082a37fa0c3cd2d11ccca0ca754b1ed067f476406ca9c73bd28fe57c.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/a7625173082a37fa0c3cd2d11ccca0ca754b1ed067f476406ca9c73bd28fe57c.jpg)

![b011a74e5a117005691548825c3c6ce8130ee81bad9a0da451c1350128184303.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/b011a74e5a117005691548825c3c6ce8130ee81bad9a0da451c1350128184303.jpg)

![b057418b2fb5fa6a9e33eb5ed800ca3cd66c0b207cdaa8e8cb6bc275ce203870.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/b057418b2fb5fa6a9e33eb5ed800ca3cd66c0b207cdaa8e8cb6bc275ce203870.jpg)

![b0d9332271b1ff7ac1876934460694e6195b63a0e9c6a410e51c7acede07fa96.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/b0d9332271b1ff7ac1876934460694e6195b63a0e9c6a410e51c7acede07fa96.jpg)

![b87bb105b9726ed25c2180c0924cf40c7b15d128b5f780eb8bf6e236f239cc05.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/b87bb105b9726ed25c2180c0924cf40c7b15d128b5f780eb8bf6e236f239cc05.jpg)

![c03fbe896066b462e8a879d415b0d81dddaca225525ec0f700ead2b909f9c2fb.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/c03fbe896066b462e8a879d415b0d81dddaca225525ec0f700ead2b909f9c2fb.jpg)

![c84cc7ca407b4b781acf7c9848f00c1c4613c9373dd44cb633dcce453eaf12a7.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/c84cc7ca407b4b781acf7c9848f00c1c4613c9373dd44cb633dcce453eaf12a7.jpg)

![d0f999ef71985f57ba71d07fc25fccb0a9936b1eb5d8dd15d1e119f9e214afe5.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/d0f999ef71985f57ba71d07fc25fccb0a9936b1eb5d8dd15d1e119f9e214afe5.jpg)

![d745a76322a02a12857a89bb990972dff0fdf596259c23778558c0f70356ae61.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/d745a76322a02a12857a89bb990972dff0fdf596259c23778558c0f70356ae61.jpg)

![d818cc52abb5de7e8f0f1889a03bd295e7051e1e1fb62b7fe25286652631727f.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/d818cc52abb5de7e8f0f1889a03bd295e7051e1e1fb62b7fe25286652631727f.jpg)

![d843d9f28d37960b6b9fb00e5b27637f98030335c00d61227d05ed6d39a5ffef.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/d843d9f28d37960b6b9fb00e5b27637f98030335c00d61227d05ed6d39a5ffef.jpg)

![e0d5b055f652c02cbedd2e96a5f73f014e622c047f8a2089044f05630f950e39.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/e0d5b055f652c02cbedd2e96a5f73f014e622c047f8a2089044f05630f950e39.jpg)

![e5d53ad10cf4c255b56565595ab219729a28985d56fe91d02b282174d2ee5711.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/e5d53ad10cf4c255b56565595ab219729a28985d56fe91d02b282174d2ee5711.jpg)

![f72dd2265a1b33b477d0e1dc3c8c07c843f6f7f01dc8273e7be184cc82b445c7.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/f72dd2265a1b33b477d0e1dc3c8c07c843f6f7f01dc8273e7be184cc82b445c7.jpg)

![fbec37ab052a0c5375f9a4b2363a9ebbf5d0cf8a6e33e45ce4704d760c7d5809.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/images/fbec37ab052a0c5375f9a4b2363a9ebbf5d0cf8a6e33e45ce4704d760c7d5809.jpg)

### Tables

![1cf95583277867601f3d0bb66c99216fa7414bd0ceb6ddf4b0c6a07d2147cf5e.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/tables/1cf95583277867601f3d0bb66c99216fa7414bd0ceb6ddf4b0c6a07d2147cf5e.jpg)

![718c83712ab99bbec623b5f678ef5be289a49b7da002b994fb357a740ded497c.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/tables/718c83712ab99bbec623b5f678ef5be289a49b7da002b994fb357a740ded497c.jpg)

![cb983870b41547631a2094fc27d4178d63a0ac07a65cd6f593681ba1d9e9753b.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/tables/cb983870b41547631a2094fc27d4178d63a0ac07a65cd6f593681ba1d9e9753b.jpg)

![fb21e6257aa8e387545fa163947a44d2bec834aed8ac559ba97b960e1d946145.jpg](../icml_results/2139_Compositional%20Causal%20Reasoning%20Evaluation%20in%20Language%20Models/tables/fb21e6257aa8e387545fa163947a44d2bec834aed8ac559ba97b960e1d946145.jpg)

## A Versatile Influence Function for Data Attribution with Non-Decomposable Loss


### Images

![3c73b30be28f7dd511cac32a31131ef566c3006832e9297831cd159afa3d9249.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/images/3c73b30be28f7dd511cac32a31131ef566c3006832e9297831cd159afa3d9249.jpg)

![3eabf5125cfbeedc6e7c787572d8932db48f027428929531d878d9665bb4251f.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/images/3eabf5125cfbeedc6e7c787572d8932db48f027428929531d878d9665bb4251f.jpg)

![8e18385a6ecee2aa4786e7b2df2c4f7b48014fc08a3bc3f0a77552aaf559dbf4.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/images/8e18385a6ecee2aa4786e7b2df2c4f7b48014fc08a3bc3f0a77552aaf559dbf4.jpg)

![ed2b4976c53c689a645d42a90e7fac3c48ad8a44fba7edcbc6fcaf32ddc93d8d.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/images/ed2b4976c53c689a645d42a90e7fac3c48ad8a44fba7edcbc6fcaf32ddc93d8d.jpg)

### Tables

![23a46ace8641ca5191661186dc9b26b0f34bbe1cabe4d8c1f4bf4ff8d40a9804.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/tables/23a46ace8641ca5191661186dc9b26b0f34bbe1cabe4d8c1f4bf4ff8d40a9804.jpg)

![2940a3dac9071b661f45c0bbd35b8420120513f2c6c9e462d52f1d162363d6ad.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/tables/2940a3dac9071b661f45c0bbd35b8420120513f2c6c9e462d52f1d162363d6ad.jpg)

![2e523d8de987d7072db0ac6cc9ae3ee3d7360484332ff8cf772ec52efbf3e467.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/tables/2e523d8de987d7072db0ac6cc9ae3ee3d7360484332ff8cf772ec52efbf3e467.jpg)

![5f7c02e4d01bc18065177549ee8721de7ef34b8f10ac3abf5ac6628787770d99.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/tables/5f7c02e4d01bc18065177549ee8721de7ef34b8f10ac3abf5ac6628787770d99.jpg)

![ad6b465ff0693a39975d667058b15bea957090dbff47d6178c6ce60feeae09b4.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/tables/ad6b465ff0693a39975d667058b15bea957090dbff47d6178c6ce60feeae09b4.jpg)

![cd40a431db3a400cf13796b5ea36821a53724ca70d185c672b804da9bdd831d4.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/tables/cd40a431db3a400cf13796b5ea36821a53724ca70d185c672b804da9bdd831d4.jpg)

![fd4af519f52a10ea30b9c9aa9a6bfd8f9dc392d1818f44819abc8ccbe4bdc31e.jpg](../icml_results/2140_A%20Versatile%20Influence%20Function%20for%20Data%20Attribution%20with%20Non-Decomposable%20Loss/tables/fd4af519f52a10ea30b9c9aa9a6bfd8f9dc392d1818f44819abc8ccbe4bdc31e.jpg)

## WyckoffDiff -- A Generative Diffusion Model for Crystal Symmetry


### Images

![39d24271ae28e9c91ed66dbd52d54797773f604c106f9892add7f2ec6b676c2e.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/images/39d24271ae28e9c91ed66dbd52d54797773f604c106f9892add7f2ec6b676c2e.jpg)

![666143602a2c4e2945c98cb4f40e4f791e0baca59c58f80bb198fbd43e1c47c4.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/images/666143602a2c4e2945c98cb4f40e4f791e0baca59c58f80bb198fbd43e1c47c4.jpg)

![f7928687ea56fa968549d58a34e4fdfcc367b8497e13ba74d8e49f6bf96ba305.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/images/f7928687ea56fa968549d58a34e4fdfcc367b8497e13ba74d8e49f6bf96ba305.jpg)

![f8b57be8fb9e4ed236c29468edc875fa16899a8dd52f36d6a06b150877ce3f0a.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/images/f8b57be8fb9e4ed236c29468edc875fa16899a8dd52f36d6a06b150877ce3f0a.jpg)

### Tables

![0645dec2a3b38daccd8f109a18de78e4870fd96d03e6f0e3559e966782621853.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/tables/0645dec2a3b38daccd8f109a18de78e4870fd96d03e6f0e3559e966782621853.jpg)

![1991d7a1f58fd1a1082fc0cf0ff436f431e89664cb00f3f607aaa46a681781e1.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/tables/1991d7a1f58fd1a1082fc0cf0ff436f431e89664cb00f3f607aaa46a681781e1.jpg)

![a552becd6df0880bb1e6e8a0cda63a5441bd1071a4459224486e0c6388f97309.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/tables/a552becd6df0880bb1e6e8a0cda63a5441bd1071a4459224486e0c6388f97309.jpg)

![c13e04c0de6bd91669681ed54ef3593f5a0f6f9b8008f4977aa8bd4bc15dd1ae.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/tables/c13e04c0de6bd91669681ed54ef3593f5a0f6f9b8008f4977aa8bd4bc15dd1ae.jpg)

![df781b0c1b46241a3741008b4a71cb73ffbfe468fd296a88894e22afc26ca1bf.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/tables/df781b0c1b46241a3741008b4a71cb73ffbfe468fd296a88894e22afc26ca1bf.jpg)

![e348f6d53c437d4bcccfbc51d9a8f14cdb2df818fafefc2284df3d022795c6c9.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/tables/e348f6d53c437d4bcccfbc51d9a8f14cdb2df818fafefc2284df3d022795c6c9.jpg)

![f3b45b54f2209ddf47d7c9c7f9a1f8057c521db68648263141cb2c997bfa2677.jpg](../icml_results/2141_WyckoffDiff%20--%20A%20Generative%20Diffusion%20Model%20for%20Crystal%20Symmetry/tables/f3b45b54f2209ddf47d7c9c7f9a1f8057c521db68648263141cb2c997bfa2677.jpg)

## CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection


### Images

![0af485d83fd1a6e1dc3fdcb53eefacc005b846d0570e643198c699667f773f11.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/images/0af485d83fd1a6e1dc3fdcb53eefacc005b846d0570e643198c699667f773f11.jpg)

![0cec201851c4fd4787c0fb8b6d945203e770bb106e37cc60094664116009d1e0.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/images/0cec201851c4fd4787c0fb8b6d945203e770bb106e37cc60094664116009d1e0.jpg)

![664ebad680f14b19ceed3aa2b9fc3d6b8b44dd8a9619cddea7f27b15e066cc58.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/images/664ebad680f14b19ceed3aa2b9fc3d6b8b44dd8a9619cddea7f27b15e066cc58.jpg)

![6d9f110f9df6de52bed1ac18a93749f325dbcd044fbf46149c984b670e33c567.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/images/6d9f110f9df6de52bed1ac18a93749f325dbcd044fbf46149c984b670e33c567.jpg)

### Tables

![0236da67ce61243f510e0729aa003d3ccff32675c4b15c9f832b7f9b14374ace.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/0236da67ce61243f510e0729aa003d3ccff32675c4b15c9f832b7f9b14374ace.jpg)

![09c1b0b3352ff12270035a002f5b732d3bbf4eed4f0af7f19367d761dae28445.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/09c1b0b3352ff12270035a002f5b732d3bbf4eed4f0af7f19367d761dae28445.jpg)

![30d87d2fceda948eabe3fc23c8aa2ef2ee4cc20189c186a452ed744b4c10d454.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/30d87d2fceda948eabe3fc23c8aa2ef2ee4cc20189c186a452ed744b4c10d454.jpg)

![4edfed43c730e30cffa950c5b1fe4e47da62c1d193d8e3c3d9d84dd7b1b6cf11.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/4edfed43c730e30cffa950c5b1fe4e47da62c1d193d8e3c3d9d84dd7b1b6cf11.jpg)

![54749054d714c326783baef5b7f1de52be4006154868b913f7eab0cdf841fc2a.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/54749054d714c326783baef5b7f1de52be4006154868b913f7eab0cdf841fc2a.jpg)

![63b9f44f09f11b4a909d1a059434573651df284befab77cd5560af4a4c49f270.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/63b9f44f09f11b4a909d1a059434573651df284befab77cd5560af4a4c49f270.jpg)

![ae3c463dc447601b1405b89967e2e0099beb3ff31d3e7b6ab5a139b68e6852ea.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/ae3c463dc447601b1405b89967e2e0099beb3ff31d3e7b6ab5a139b68e6852ea.jpg)

![ee5a18c1cbe92bdf8c55fd21f881439380b2298f8e6cb13f1baf97097672596c.jpg](../icml_results/2142_CurvGAD_%20Leveraging%20Curvature%20for%20Enhanced%20Graph%20Anomaly%20Detection/tables/ee5a18c1cbe92bdf8c55fd21f881439380b2298f8e6cb13f1baf97097672596c.jpg)

## When Do LLMs Help With Node Classification? A Comprehensive Analysis


### Images

![4a537d767f0e8abc2300c28d50fa62a6628ebbd9c88b40e24c38324bdaf210e4.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/images/4a537d767f0e8abc2300c28d50fa62a6628ebbd9c88b40e24c38324bdaf210e4.jpg)

![4aa68c4e90504f600e942d879804c765dd5c03670bf55c9a32cfbf8123576b69.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/images/4aa68c4e90504f600e942d879804c765dd5c03670bf55c9a32cfbf8123576b69.jpg)

![a65aef0a275bae4ebfbca0de5f7cecdc5d6b90988982a0f1d6567564cbe8670b.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/images/a65aef0a275bae4ebfbca0de5f7cecdc5d6b90988982a0f1d6567564cbe8670b.jpg)

![ea36c8c8459d67c62ae8b2e4d8b028b593c30555ceee4b0c5eeafa4fa5f5293f.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/images/ea36c8c8459d67c62ae8b2e4d8b028b593c30555ceee4b0c5eeafa4fa5f5293f.jpg)

![f02a5cf61dcfd3e4f06a995b324658b4003a102963de0414fce78cb34206c5dc.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/images/f02a5cf61dcfd3e4f06a995b324658b4003a102963de0414fce78cb34206c5dc.jpg)

![f4ec734cb5b5ecc316693f1dc5474a8f640e8654a6948ca202606fed81ad71cf.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/images/f4ec734cb5b5ecc316693f1dc5474a8f640e8654a6948ca202606fed81ad71cf.jpg)

### Tables

![10060154657e022f0a3d6894f48da30e7ff18f2b2eede6951464a8375ac30527.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/10060154657e022f0a3d6894f48da30e7ff18f2b2eede6951464a8375ac30527.jpg)

![1bc4da9f512fa9347b2ade376736b0b7481ea6438fed0e283f63af02ab9fe442.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/1bc4da9f512fa9347b2ade376736b0b7481ea6438fed0e283f63af02ab9fe442.jpg)

![2750cc57658337225800e88218aead123290dcbab248f6785d350d40fd001788.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/2750cc57658337225800e88218aead123290dcbab248f6785d350d40fd001788.jpg)

![29ce9e306b2f4dc21a0f92d5526d33f873fc1e1ec82b8cd1a00cb656b807bbef.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/29ce9e306b2f4dc21a0f92d5526d33f873fc1e1ec82b8cd1a00cb656b807bbef.jpg)

![3be79dbb147b09dc1741b50aa0d3ea02967fcf56c9dd265f2fa00a7859aaa2d2.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/3be79dbb147b09dc1741b50aa0d3ea02967fcf56c9dd265f2fa00a7859aaa2d2.jpg)

![6365857abe313f766a19d0f7f36fc1f75fa9f71bccd980bdfbc5bd52ca5ed07e.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/6365857abe313f766a19d0f7f36fc1f75fa9f71bccd980bdfbc5bd52ca5ed07e.jpg)

![6eb5f7bae3d8f0ee634bff4b52bafc3a2aa84696c5bfa0b9383e7cec2e650a60.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/6eb5f7bae3d8f0ee634bff4b52bafc3a2aa84696c5bfa0b9383e7cec2e650a60.jpg)

![73b6b195fb62e12f357d4d8ba72cbde5825044d2451bec925f11832b307d0192.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/73b6b195fb62e12f357d4d8ba72cbde5825044d2451bec925f11832b307d0192.jpg)

![7ca863bcd1df2a7b16bffe33b1b901e17f6a66d29de9baa9c056a4704aaf6876.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/7ca863bcd1df2a7b16bffe33b1b901e17f6a66d29de9baa9c056a4704aaf6876.jpg)

![7e4292ca8b55edd5dde8e9b6be80ce3c16a4c40ac0a8b63e9cfc72c6d0b5bfa7.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/7e4292ca8b55edd5dde8e9b6be80ce3c16a4c40ac0a8b63e9cfc72c6d0b5bfa7.jpg)

![a1386985a63643365fdf153701bd7955a172195762d22ca4d8349c5d32bdcf0f.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/a1386985a63643365fdf153701bd7955a172195762d22ca4d8349c5d32bdcf0f.jpg)

![a53c838de72e7ac8f4f8b6ecab7041ddfcff1d6ed26726ce4915d3d71690c613.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/a53c838de72e7ac8f4f8b6ecab7041ddfcff1d6ed26726ce4915d3d71690c613.jpg)

![a7f4a78efd7e9a886a87f34003a0a96ed37de6a3eb0de7cc4158b7c3ed5639f8.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/a7f4a78efd7e9a886a87f34003a0a96ed37de6a3eb0de7cc4158b7c3ed5639f8.jpg)

![bf2ac908d5380b615967accdf9c8e1847bacb93cd5736004b308fcfae3b4da4e.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/bf2ac908d5380b615967accdf9c8e1847bacb93cd5736004b308fcfae3b4da4e.jpg)

![c2e9fb67f261b98d659fdaf7d96bb8c3e3451bc0572c67a3e7518fb89e8393d7.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/c2e9fb67f261b98d659fdaf7d96bb8c3e3451bc0572c67a3e7518fb89e8393d7.jpg)

![d5b8a25c50ce00edfbaf1d05d4f4b8f3bd6a43eac1af32d4410e23b99b0a14c7.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/d5b8a25c50ce00edfbaf1d05d4f4b8f3bd6a43eac1af32d4410e23b99b0a14c7.jpg)

![da07bc02bb65e2d44e6562b081475e439b634bc8b9d1acc2439c42227699bae0.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/da07bc02bb65e2d44e6562b081475e439b634bc8b9d1acc2439c42227699bae0.jpg)

![da37907441b5d5cdda999af312db71ba81e6a1fe80416d6a8e44f044bc92ab16.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/da37907441b5d5cdda999af312db71ba81e6a1fe80416d6a8e44f044bc92ab16.jpg)

![f999b5b12e26f8425a5c664fb1bc9a25a02ca66319d826c9bdcae6361bf91a92.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/f999b5b12e26f8425a5c664fb1bc9a25a02ca66319d826c9bdcae6361bf91a92.jpg)

![fd795601fe43848afeb97817c9ca2399d895ec8d44f6acf4252c9edb4d40709e.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/fd795601fe43848afeb97817c9ca2399d895ec8d44f6acf4252c9edb4d40709e.jpg)

![fedb6b93f0163fc3cf5af6e937e71e8f47dc25cee87c0c7d7bb714d059468c77.jpg](../icml_results/2143_When%20Do%20LLMs%20Help%20With%20Node%20Classification_%20A%20Comprehensive%20Analysis/tables/fedb6b93f0163fc3cf5af6e937e71e8f47dc25cee87c0c7d7bb714d059468c77.jpg)

## $\mathcal{V}ista\mathcal{DPO}$: Video Hierarchical Spatial-Temporal Direct Preference Optimization for Large Video Models


### Images

![0688e23395e59faa4a08a2dacc8d6de888d2e6cac207f1dec79ffd32229fa1d1.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/0688e23395e59faa4a08a2dacc8d6de888d2e6cac207f1dec79ffd32229fa1d1.jpg)

![1ed41d79f45202c200d7fa904db9e13c4a0c8312e8423ba82f0120bc27eae94e.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/1ed41d79f45202c200d7fa904db9e13c4a0c8312e8423ba82f0120bc27eae94e.jpg)

![36a0603ebdc75f380441dac1145361c3457afd1fcd61064c4320ef03038f9d86.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/36a0603ebdc75f380441dac1145361c3457afd1fcd61064c4320ef03038f9d86.jpg)

![3d83f0dc428e25f13eccf22b6199cf12bc7cd1f4d5cf052715ba5a162548e2de.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/3d83f0dc428e25f13eccf22b6199cf12bc7cd1f4d5cf052715ba5a162548e2de.jpg)

![550e2fe2dab96f97905301a5713602e473ffd3ef1f1f5ea8e3d93aeb7721e2a8.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/550e2fe2dab96f97905301a5713602e473ffd3ef1f1f5ea8e3d93aeb7721e2a8.jpg)

![73dc60dfedcce2955a02d2bef12276dc000aefe4c91fd62a66e0e3b818310296.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/73dc60dfedcce2955a02d2bef12276dc000aefe4c91fd62a66e0e3b818310296.jpg)

![7e002114afbf1cf65ff30ab340974b6b8202866207544f5a332045606ca8db6d.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/7e002114afbf1cf65ff30ab340974b6b8202866207544f5a332045606ca8db6d.jpg)

![9051920d3557a0e912b71c0e7618b3178062d1925a49368fd8263f300d35b9b0.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/9051920d3557a0e912b71c0e7618b3178062d1925a49368fd8263f300d35b9b0.jpg)

![b5888784a02f35244979dcdbcd892cbf1c841525bc4f0e0717ae45ea35ade0c9.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/b5888784a02f35244979dcdbcd892cbf1c841525bc4f0e0717ae45ea35ade0c9.jpg)

![b68f66313a3d41b4a7538dba2dd869db9d8a68d8c08f132071a39f0fa159887a.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/b68f66313a3d41b4a7538dba2dd869db9d8a68d8c08f132071a39f0fa159887a.jpg)

![bd4f7cfde390bf40c0a830eeb036f12a3e6a2f5f7a3de68873a4a51bd6a22c1b.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/bd4f7cfde390bf40c0a830eeb036f12a3e6a2f5f7a3de68873a4a51bd6a22c1b.jpg)

![d407cc2dc8d286ba80ab150510f7fcaee5271994be686dcc7770c565008d643b.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/d407cc2dc8d286ba80ab150510f7fcaee5271994be686dcc7770c565008d643b.jpg)

![d6d5a55b3a9e8bbfb1ee218c7d4185cd80e9043a70cc5317bf12b2ffbfee629e.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/d6d5a55b3a9e8bbfb1ee218c7d4185cd80e9043a70cc5317bf12b2ffbfee629e.jpg)

![daebbc23cc794abd3a6e0b2784d3273fce62f62c7bc6736db6b01177a6a9cb8d.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/daebbc23cc794abd3a6e0b2784d3273fce62f62c7bc6736db6b01177a6a9cb8d.jpg)

![e52b6dacde7a8b35bb063e351ba96e1a959dd8e91ed3215e943af6bae2873443.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/images/e52b6dacde7a8b35bb063e351ba96e1a959dd8e91ed3215e943af6bae2873443.jpg)

### Tables

![05c67a80e6a2697c1565ff4417629f4c9e8ea2501f27232442b46affba138784.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/tables/05c67a80e6a2697c1565ff4417629f4c9e8ea2501f27232442b46affba138784.jpg)

![17edbae93b107c71dae735bc1d03d689bb49a2b2a9115e804e8819ec95b96a20.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/tables/17edbae93b107c71dae735bc1d03d689bb49a2b2a9115e804e8819ec95b96a20.jpg)

![5a7b2d2ebecae4dc7665d01d1b56928337e919f95b96ba2855ec57cf9f0a54bc.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/tables/5a7b2d2ebecae4dc7665d01d1b56928337e919f95b96ba2855ec57cf9f0a54bc.jpg)

![6c6fae55d6804c664eed3fe2b12d8062ef7a7ad9bd53d5cb3955c27812f98393.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/tables/6c6fae55d6804c664eed3fe2b12d8062ef7a7ad9bd53d5cb3955c27812f98393.jpg)

![9f775fc0644a1f9f3020b45a9992f1e27e877c37d5aeb016c181d71d000b86ca.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/tables/9f775fc0644a1f9f3020b45a9992f1e27e877c37d5aeb016c181d71d000b86ca.jpg)

![b3a388af784561a6089466b94882fc315c45e66a1936392deec9c8480ca3d516.jpg](../icml_results/2145_%24_mathcal%7BV%7Dista_mathcal%7BDPO%7D%24_%20Video%20Hierarchical%20Spatial-Temporal%20Direct%20Preference%20Optimization%20f/tables/b3a388af784561a6089466b94882fc315c45e66a1936392deec9c8480ca3d516.jpg)

## Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization


### Images

![13de6cbfb2c2652c59db7d8297dc185333d0d0294ab1b62de263b950eb87c5d9.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/images/13de6cbfb2c2652c59db7d8297dc185333d0d0294ab1b62de263b950eb87c5d9.jpg)

![230a1d44814a4a86223637bb4cbcf4885049b397af795b2d2e3519c3bb02e014.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/images/230a1d44814a4a86223637bb4cbcf4885049b397af795b2d2e3519c3bb02e014.jpg)

![5087339dc2c435efee27500ebebdc625b13c3a1849b03fe666cc23bb9b60fbbe.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/images/5087339dc2c435efee27500ebebdc625b13c3a1849b03fe666cc23bb9b60fbbe.jpg)

![a55fc36d1d78bfb09f35674eaf2cc212d98b06f0ab6c1dbfe1d776e2f64a0d73.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/images/a55fc36d1d78bfb09f35674eaf2cc212d98b06f0ab6c1dbfe1d776e2f64a0d73.jpg)

![ea0c17a6b5429c2266be7c4a40cced10ee24b6141a90b36d0b26494d9a06c213.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/images/ea0c17a6b5429c2266be7c4a40cced10ee24b6141a90b36d0b26494d9a06c213.jpg)

### Tables

![29c613a63dcda5a24b45dcde7b55bec67a42bd5751ef4f888810211d52a297f1.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/tables/29c613a63dcda5a24b45dcde7b55bec67a42bd5751ef4f888810211d52a297f1.jpg)

![3ce0cdd92989891ce44533f8d8902ef7b3d84545132e68f210e9e2044b2ad8ad.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/tables/3ce0cdd92989891ce44533f8d8902ef7b3d84545132e68f210e9e2044b2ad8ad.jpg)

![7c7565a59514e9aeb81ef008813f96d76f8396ef08506f4dde4580a2704ee68d.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/tables/7c7565a59514e9aeb81ef008813f96d76f8396ef08506f4dde4580a2704ee68d.jpg)

![8c69a667032d050c7ffbc4736a157e9f91aa1d740b72186728a971597cef5d73.jpg](../icml_results/2146_Optimal%20Transport%20Barycenter%20via%20Nonconvex-Concave%20Minimax%20Optimization/tables/8c69a667032d050c7ffbc4736a157e9f91aa1d740b72186728a971597cef5d73.jpg)

## SafeMap: Robust HD Map Construction from Incomplete Observations


### Images

![51c61daf3c42ed6a779ba95ca3edd5b7c0ab7ccb992841d85c42ea9bcb9f479b.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/images/51c61daf3c42ed6a779ba95ca3edd5b7c0ab7ccb992841d85c42ea9bcb9f479b.jpg)

![60841991400b48bcf2078c52e84117b5243497a954d53564a5e026b91356d02a.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/images/60841991400b48bcf2078c52e84117b5243497a954d53564a5e026b91356d02a.jpg)

![7dc40ea604c6e7a08818fc08856897a435f811be9ed1afc5f75c829c5351568e.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/images/7dc40ea604c6e7a08818fc08856897a435f811be9ed1afc5f75c829c5351568e.jpg)

![9d35ae7c9d37a040200bcf027bc7ac7f69259bba964a3e2018fd0b958590aee0.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/images/9d35ae7c9d37a040200bcf027bc7ac7f69259bba964a3e2018fd0b958590aee0.jpg)

![a894c03610b1aa0ab180bab296f7ca6de92bf1d34e727ea8bc452197d3ad124e.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/images/a894c03610b1aa0ab180bab296f7ca6de92bf1d34e727ea8bc452197d3ad124e.jpg)

![e0d063cc0b5f1fda9921e232b4a8e4c47df76dcb320b9023956ac81b7314f53b.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/images/e0d063cc0b5f1fda9921e232b4a8e4c47df76dcb320b9023956ac81b7314f53b.jpg)

![e174ed896ed11bfe3523cf0f6f4f350ea6351d947ed8fe5afb34194ce9e43947.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/images/e174ed896ed11bfe3523cf0f6f4f350ea6351d947ed8fe5afb34194ce9e43947.jpg)

### Tables

![05d71f58ca1748271fb7526c9e1df34bb2c0e1ee8c2b2f3e25ee4d21e42ffaac.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/05d71f58ca1748271fb7526c9e1df34bb2c0e1ee8c2b2f3e25ee4d21e42ffaac.jpg)

![4cb710c4e054521ae8e4bedbdf4c3e60088e3e9f0f0b2e6d841af455be6c0914.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/4cb710c4e054521ae8e4bedbdf4c3e60088e3e9f0f0b2e6d841af455be6c0914.jpg)

![7405c6e9c2e13732c04609a4e33d3dcf65fb073e70cacb8eba45ad728a4a0903.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/7405c6e9c2e13732c04609a4e33d3dcf65fb073e70cacb8eba45ad728a4a0903.jpg)

![74e2ae74715208fe46893e9e1df00807de6f0a41434fc446f0426e73a935b7c6.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/74e2ae74715208fe46893e9e1df00807de6f0a41434fc446f0426e73a935b7c6.jpg)

![b204d62a871e0c2628f2befa61f3f62cef77ad3c0a5f50fca4d25b7bd939e032.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/b204d62a871e0c2628f2befa61f3f62cef77ad3c0a5f50fca4d25b7bd939e032.jpg)

![c0060647b029d0a966586d10e6add747eef9450e5589d950de6dd6383b87c1a1.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/c0060647b029d0a966586d10e6add747eef9450e5589d950de6dd6383b87c1a1.jpg)

![cd92f15f979cf864274e3df22b5049f39fdd1b35c3f97b2b9d5567b128eae619.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/cd92f15f979cf864274e3df22b5049f39fdd1b35c3f97b2b9d5567b128eae619.jpg)

![d2cc50df96da7d52b39d0a3752f70bfd49375e6e0cc44b354d0eee95793acc19.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/d2cc50df96da7d52b39d0a3752f70bfd49375e6e0cc44b354d0eee95793acc19.jpg)

![e7fcf330a9e901ff6154f44c2c158fb1f1023d60ab676816a9f6eeab6b7a3f5d.jpg](../icml_results/2147_SafeMap_%20Robust%20HD%20Map%20Construction%20from%20Incomplete%20Observations/tables/e7fcf330a9e901ff6154f44c2c158fb1f1023d60ab676816a9f6eeab6b7a3f5d.jpg)

## Accurate Identification of Communication Between Multiple Interacting Neural Populations


### Images

![034822a115372edcc74b33eafd9483250c23e786daec394a67e415025d2c67e6.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/034822a115372edcc74b33eafd9483250c23e786daec394a67e415025d2c67e6.jpg)

![30e41a81b9f724dbcc7acc52e2c25432fa955ee15b19cfba18615262f2499410.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/30e41a81b9f724dbcc7acc52e2c25432fa955ee15b19cfba18615262f2499410.jpg)

![39372042ae4109f47add35a8650017756a5be00e0a7dd536802ab002bd3633bb.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/39372042ae4109f47add35a8650017756a5be00e0a7dd536802ab002bd3633bb.jpg)

![520832616257c0cc81e6e5e1699b55978b81f4a96a162e6ec30a46b763af8c68.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/520832616257c0cc81e6e5e1699b55978b81f4a96a162e6ec30a46b763af8c68.jpg)

![7c8e01fe8eec046dccadaab9c8d23479d25268f8de747977929b34b6ea2c2af8.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/7c8e01fe8eec046dccadaab9c8d23479d25268f8de747977929b34b6ea2c2af8.jpg)

![993153803837aa8b6868c5d7a55d40822efd2c301ff0fa4eb9601c63a390f315.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/993153803837aa8b6868c5d7a55d40822efd2c301ff0fa4eb9601c63a390f315.jpg)

![a3a13478d49898b93082e2abbcc9ba8d73cfaf79de7891a808b305dbca927637.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/a3a13478d49898b93082e2abbcc9ba8d73cfaf79de7891a808b305dbca927637.jpg)

![bf3fafe8620d9ba7950e50290826b7f51d52d68df785665a70d096e3a864930d.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/bf3fafe8620d9ba7950e50290826b7f51d52d68df785665a70d096e3a864930d.jpg)

![c26d5bedc568d6e432f5f2ba71849a7d060c7e6f967b51e54634d5da1eb7a903.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/c26d5bedc568d6e432f5f2ba71849a7d060c7e6f967b51e54634d5da1eb7a903.jpg)

![cf75a0778e8ad340b1cdfd80b1756aabe70ab8e2e3123ba212bb42c8205988b2.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/images/cf75a0778e8ad340b1cdfd80b1756aabe70ab8e2e3123ba212bb42c8205988b2.jpg)

### Tables

![8253ef999e6671c9cf006d5b419bc44ac561c1174aa19787ab763d5aa009ca1e.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/tables/8253ef999e6671c9cf006d5b419bc44ac561c1174aa19787ab763d5aa009ca1e.jpg)

![94429f0cada0be9dee1e55be09dd5d9943d491292b9d67d5430f950ef7b5686d.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/tables/94429f0cada0be9dee1e55be09dd5d9943d491292b9d67d5430f950ef7b5686d.jpg)

![b1aa902727c4a239a8519430a4f1a729aeb4ff4bc0d24f898cdc06a4952bdf04.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/tables/b1aa902727c4a239a8519430a4f1a729aeb4ff4bc0d24f898cdc06a4952bdf04.jpg)

![bb73ad205394962e71e5919976e521a938ed0408e63adb87c557e92211c192f6.jpg](../icml_results/2148_Accurate%20Identification%20of%20Communication%20Between%20Multiple%20Interacting%20Neural%20Populations/tables/bb73ad205394962e71e5919976e521a938ed0408e63adb87c557e92211c192f6.jpg)

## PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization


### Images

![0b88459cb2af489fe344150025fbef793d51fba29bc0865bcab8cc156c6ec542.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/0b88459cb2af489fe344150025fbef793d51fba29bc0865bcab8cc156c6ec542.jpg)

![1268ab1c773c41d484cf5368f8e4640e1337a927dc170ea1be2764d56dba893a.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/1268ab1c773c41d484cf5368f8e4640e1337a927dc170ea1be2764d56dba893a.jpg)

![2a6fd8add3672c9d8de0466f6607de1af7a4e168bd104284af62a94ebb513509.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/2a6fd8add3672c9d8de0466f6607de1af7a4e168bd104284af62a94ebb513509.jpg)

![2d76632b1327debbdfab854793a0a9041c0ffa78e029639a688b396a44827d33.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/2d76632b1327debbdfab854793a0a9041c0ffa78e029639a688b396a44827d33.jpg)

![5d622c53f35c3b0d34f1e89144013c4fdd08b232a06a3e7515fb4a11a6f2a18e.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/5d622c53f35c3b0d34f1e89144013c4fdd08b232a06a3e7515fb4a11a6f2a18e.jpg)

![6462006c8a009c956177cc2b93b1d5b0dd8ba0a4089d17f97920e59a456ac511.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/6462006c8a009c956177cc2b93b1d5b0dd8ba0a4089d17f97920e59a456ac511.jpg)

![6b8739be45fef6dd7751b392a4b00b1805301d4604b0ebb427d63cc366bcb840.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/6b8739be45fef6dd7751b392a4b00b1805301d4604b0ebb427d63cc366bcb840.jpg)

![7def26dd974b58b3d0eef9166ef857f5de860d905af952517aa23e27607afd89.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/7def26dd974b58b3d0eef9166ef857f5de860d905af952517aa23e27607afd89.jpg)

![83d86de32d39bd30c838d69e113f998b15c2b8e273837921e4093551ac3be91c.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/83d86de32d39bd30c838d69e113f998b15c2b8e273837921e4093551ac3be91c.jpg)

![89f9f34df0bcf5a4c74817e7d259d2d4f13007e1a59eb34c2afe9331c7283169.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/89f9f34df0bcf5a4c74817e7d259d2d4f13007e1a59eb34c2afe9331c7283169.jpg)

![8a993a97b2f9d167dfbde70bf1a879aa7241632d3a2618ca3c9e74338e82591d.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/8a993a97b2f9d167dfbde70bf1a879aa7241632d3a2618ca3c9e74338e82591d.jpg)

![8f35d9954b40ab1f6d369ffff948060374b67655fc8f75f516e91c6705ca39aa.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/8f35d9954b40ab1f6d369ffff948060374b67655fc8f75f516e91c6705ca39aa.jpg)

![a6554acde19b8e025e237ba5715f42796e0a606763e070774b2356d5ea9b585a.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/a6554acde19b8e025e237ba5715f42796e0a606763e070774b2356d5ea9b585a.jpg)

![afa07306ce4a2105ad0404fe7222693f9a14d69f6c504e40f32094726c520775.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/afa07306ce4a2105ad0404fe7222693f9a14d69f6c504e40f32094726c520775.jpg)

![b9cc3fda93c60785ed23afddad3bdc1bd4192068a72365738c16d31679838b69.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/b9cc3fda93c60785ed23afddad3bdc1bd4192068a72365738c16d31679838b69.jpg)

![c3f52e4395bf3daa4e371d54f457c74393c58b8773e323ce3adfc328d7e21cac.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/c3f52e4395bf3daa4e371d54f457c74393c58b8773e323ce3adfc328d7e21cac.jpg)

![d70fbab6ea9ed8407958c09a75370340be374974e11b0da833554c871636e9fa.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/d70fbab6ea9ed8407958c09a75370340be374974e11b0da833554c871636e9fa.jpg)

![d999623720294c49f141ff04b6078136703b0853ffdc2fd4d00ca9f03d339b1c.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/d999623720294c49f141ff04b6078136703b0853ffdc2fd4d00ca9f03d339b1c.jpg)

![de99e93eb7676f382625b7b803aaf1b8a68a49420d81cea92cd2126db9dce488.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/de99e93eb7676f382625b7b803aaf1b8a68a49420d81cea92cd2126db9dce488.jpg)

![edfb2e091fd72f42fc2d4bc64e78d6b9d8b6e51c0563b51803c1ec15c334e971.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/images/edfb2e091fd72f42fc2d4bc64e78d6b9d8b6e51c0563b51803c1ec15c334e971.jpg)

### Tables

![3e41a2a6616469abc36bf70c379a7e5c74507045d3fdcca09017a13a45a0551f.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/tables/3e41a2a6616469abc36bf70c379a7e5c74507045d3fdcca09017a13a45a0551f.jpg)

![b9a084e3ada21a9589e77c38ed2338fc15f34a49259df55954f810b82bbd5e2d.jpg](../icml_results/2149_PipeOffload_%20Improving%20Scalability%20of%20Pipeline%20Parallelism%20with%20Memory%20Optimization/tables/b9a084e3ada21a9589e77c38ed2338fc15f34a49259df55954f810b82bbd5e2d.jpg)

## GRAIL: Graph Edit Distance and Node Alignment using LLM-Generated Code


### Images

![03b9da3f38795a463111fd8b9545574131306ec06d51d9bea45412de65617fb0.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/03b9da3f38795a463111fd8b9545574131306ec06d51d9bea45412de65617fb0.jpg)

![57973846458d4b8991f2ee25ed74ca72c5b351bc298bf432be9ca242847133c6.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/57973846458d4b8991f2ee25ed74ca72c5b351bc298bf432be9ca242847133c6.jpg)

![5a42ef9d79cc290bf1226d1a02532f98a928ca64cb8c7d3d5c98d570843ccd44.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/5a42ef9d79cc290bf1226d1a02532f98a928ca64cb8c7d3d5c98d570843ccd44.jpg)

![7850689c646bd623db49129d72d598609046f88448c30622541ab9d2a8701b66.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/7850689c646bd623db49129d72d598609046f88448c30622541ab9d2a8701b66.jpg)

![a27659b4e22ce2086d368d9298853ba788be08447ea5d35795a204b052802fc6.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/a27659b4e22ce2086d368d9298853ba788be08447ea5d35795a204b052802fc6.jpg)

![af3ab9262b1ca4e6e49b59bd7f216bd1f2643d8b73e388d2bbabe7de56138337.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/af3ab9262b1ca4e6e49b59bd7f216bd1f2643d8b73e388d2bbabe7de56138337.jpg)

![b867f59179638332461b86048448806014d6c926e31d874d627bbec38e0277f8.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/b867f59179638332461b86048448806014d6c926e31d874d627bbec38e0277f8.jpg)

![bef7de0388b7a3611f4fe1b595837eedc7b02bbab8ff0b7e0cd25b4f9a8bbf05.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/bef7de0388b7a3611f4fe1b595837eedc7b02bbab8ff0b7e0cd25b4f9a8bbf05.jpg)

![c5f7e5b512c5e36b389b96fbd6abbee7e83976ad5366eaccadccb0594a797f10.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/c5f7e5b512c5e36b389b96fbd6abbee7e83976ad5366eaccadccb0594a797f10.jpg)

![d01e4bd6e2e76bce6b7eed05a172c32165cbd28a22163b1abf68dab91af6eed3.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/d01e4bd6e2e76bce6b7eed05a172c32165cbd28a22163b1abf68dab91af6eed3.jpg)

![fa12e0527f0ab841c5a23113d7ad64092a3b5aef889edecb777a87c359e6ad40.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/images/fa12e0527f0ab841c5a23113d7ad64092a3b5aef889edecb777a87c359e6ad40.jpg)

### Tables

![0081e9afb4a6b97764eb0e41526890d34f24bfcc7bee84fe12f363ccfe6aad7f.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/0081e9afb4a6b97764eb0e41526890d34f24bfcc7bee84fe12f363ccfe6aad7f.jpg)

![07c2a29712299cf1127a29410b4743672658394ff674fb2f6e0a3d95fd15d97d.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/07c2a29712299cf1127a29410b4743672658394ff674fb2f6e0a3d95fd15d97d.jpg)

![0d103e4e1af0d3893fb63788247567a3ff3004f04abfc9194fc9a8798cfa4fca.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/0d103e4e1af0d3893fb63788247567a3ff3004f04abfc9194fc9a8798cfa4fca.jpg)

![27ec630d95e61fbcf60a17680a20dfe1b029dd90a5496e713a5a80a2d9dc5fe4.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/27ec630d95e61fbcf60a17680a20dfe1b029dd90a5496e713a5a80a2d9dc5fe4.jpg)

![4f899325a364dbe9c6b330657e802eaa20a28b96c69a365aa5caf4561586ac3b.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/4f899325a364dbe9c6b330657e802eaa20a28b96c69a365aa5caf4561586ac3b.jpg)

![9e464fd0e0a60f0642b09553bfc0254754b6a6d270fa96040fba1edfb73ea87a.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/9e464fd0e0a60f0642b09553bfc0254754b6a6d270fa96040fba1edfb73ea87a.jpg)

![c955cb61ad2c00d215eada797a566147c59745a83aa812cf689b95886af5659c.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/c955cb61ad2c00d215eada797a566147c59745a83aa812cf689b95886af5659c.jpg)

![d92e771561925994b5b6af703370fd308d3ce6a2cfd35d975954dcd603c9013d.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/d92e771561925994b5b6af703370fd308d3ce6a2cfd35d975954dcd603c9013d.jpg)

![f7e7f8c1e5c53255e5238a360d897b780a130ce8a8bd50b631235ad57260ab3b.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/f7e7f8c1e5c53255e5238a360d897b780a130ce8a8bd50b631235ad57260ab3b.jpg)

![fd2e00c0a947fcc53baf595f9fc37bd48a3a3e1e75d09fe33aa1ec828c2ce36b.jpg](../icml_results/2150_GRAIL_%20Graph%20Edit%20Distance%20and%20Node%20Alignment%20using%20LLM-Generated%20Code/tables/fd2e00c0a947fcc53baf595f9fc37bd48a3a3e1e75d09fe33aa1ec828c2ce36b.jpg)

## When Model Knowledge meets Diffusion Model: Diffusion-assisted Data-free Image Synthesis with Alignment of Domain and Class


### Images

![0f53a8e7e79d275ca3d485a457c1021366d7fffa82256eeed78ac4b828904bbe.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/0f53a8e7e79d275ca3d485a457c1021366d7fffa82256eeed78ac4b828904bbe.jpg)

![19e3c0dc39a14c1888fb54ad48ef83e42ffe9f2a095d36eed783fa7bfee631b3.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/19e3c0dc39a14c1888fb54ad48ef83e42ffe9f2a095d36eed783fa7bfee631b3.jpg)

![29a97a42da1b8e6d96e04034ac6c4917e903753c69493ed78326518e4b3a634a.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/29a97a42da1b8e6d96e04034ac6c4917e903753c69493ed78326518e4b3a634a.jpg)

![2f3121b4f7f54fea61147b4e8752b1eebd131960d8d32f250d318a43c19fd0ef.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/2f3121b4f7f54fea61147b4e8752b1eebd131960d8d32f250d318a43c19fd0ef.jpg)

![45913653a6342b6ba9d4da14f157a31a25c88b1b55c1e29c2270821768b640fa.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/45913653a6342b6ba9d4da14f157a31a25c88b1b55c1e29c2270821768b640fa.jpg)

![62d8038cbf481814393d34ff5b3b10afa678a9cebb542f6ad316947e0e862ea9.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/62d8038cbf481814393d34ff5b3b10afa678a9cebb542f6ad316947e0e862ea9.jpg)

![6670dc35bbbd2f29305f34da9a33dfc6d364c9e0e35a4c49f06a15d2fbb22f68.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/6670dc35bbbd2f29305f34da9a33dfc6d364c9e0e35a4c49f06a15d2fbb22f68.jpg)

![6828d3cc50d1dcce3bae8f8389e62801d2bdcd253e858a606e6c7e90231ea426.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/6828d3cc50d1dcce3bae8f8389e62801d2bdcd253e858a606e6c7e90231ea426.jpg)

![7075c2ce3462e2616bd04fd694eca0c1c975e113ed8c70b9d26158086fce68b2.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/7075c2ce3462e2616bd04fd694eca0c1c975e113ed8c70b9d26158086fce68b2.jpg)

![744f776d467dca90ab7ac10bf384c2bdc870e00490d8bf9635b4d50f4dad4bf9.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/744f776d467dca90ab7ac10bf384c2bdc870e00490d8bf9635b4d50f4dad4bf9.jpg)

![8064eda424fe38cbf71fb20c904f515d979ef46095bd5ab7c60a180a8d40a6a6.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/8064eda424fe38cbf71fb20c904f515d979ef46095bd5ab7c60a180a8d40a6a6.jpg)

![959af5cfa6f7861d3df2cdf6c4bd31b50c21307a8b62d52980e1340193d0f4ed.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/959af5cfa6f7861d3df2cdf6c4bd31b50c21307a8b62d52980e1340193d0f4ed.jpg)

![9cf39e8b2cd182c246a2b2de61d32ccb821c7475f6c1ddd21089b9d2873fcf4f.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/9cf39e8b2cd182c246a2b2de61d32ccb821c7475f6c1ddd21089b9d2873fcf4f.jpg)

![a2611289789f68727471883243cbfd022db52a24d526cb27da6e5f5bc8491e1a.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/a2611289789f68727471883243cbfd022db52a24d526cb27da6e5f5bc8491e1a.jpg)

![a625a1b733fd407c2c3b1d9f07218a05e815dd89c68995c8ea5bc3c8fe1c1640.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/a625a1b733fd407c2c3b1d9f07218a05e815dd89c68995c8ea5bc3c8fe1c1640.jpg)

![b0a79a1aac1fc9c42989b8e551707b9415ca3edc0a3f5b2bbd8b11c15bbb5dd3.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/b0a79a1aac1fc9c42989b8e551707b9415ca3edc0a3f5b2bbd8b11c15bbb5dd3.jpg)

![b5557dd2b45962397389e1262a495a8b8e4e530acfd9d1264a29b528bc55ff37.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/b5557dd2b45962397389e1262a495a8b8e4e530acfd9d1264a29b528bc55ff37.jpg)

![d9051d02eac3ba97a8a0490091f30c2bad709e0aebe2cdf92d3b794c77ae978f.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/d9051d02eac3ba97a8a0490091f30c2bad709e0aebe2cdf92d3b794c77ae978f.jpg)

![e6f1daae056ed490ec32937475895b9086b0ff2d22286fbeebd9102894f55ec8.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/e6f1daae056ed490ec32937475895b9086b0ff2d22286fbeebd9102894f55ec8.jpg)

![eaf15b8075e0cf97472aed8f8059cb876d1bc23fdead3527c739ab189fdccabb.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/eaf15b8075e0cf97472aed8f8059cb876d1bc23fdead3527c739ab189fdccabb.jpg)

![ef2c92bca8325687dde0fdf3d4ad66e04baae0bee0d9ea0af5e711a846da2cf7.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/images/ef2c92bca8325687dde0fdf3d4ad66e04baae0bee0d9ea0af5e711a846da2cf7.jpg)

### Tables

![17a104072917f415aa440aec1e2b4c06fb27931dc31e03c949f461a906c1dd1a.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/tables/17a104072917f415aa440aec1e2b4c06fb27931dc31e03c949f461a906c1dd1a.jpg)

![1e09d427e976279dfd427979cfe3a0bba7906ada253b2869a14494170cc99958.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/tables/1e09d427e976279dfd427979cfe3a0bba7906ada253b2869a14494170cc99958.jpg)

![755ba5ba926b1ad8ee71a426da6189e73ce122aa45b3693c97110f056e147d3c.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/tables/755ba5ba926b1ad8ee71a426da6189e73ce122aa45b3693c97110f056e147d3c.jpg)

![7b3fc1e1281c286898f7cd76468fa20e89479afdfdc9a478d36e2d5600a6cde6.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/tables/7b3fc1e1281c286898f7cd76468fa20e89479afdfdc9a478d36e2d5600a6cde6.jpg)

![8c959f0358ae701d3687cfd1a429bc3d69511668898e51420c319bae1834a2ab.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/tables/8c959f0358ae701d3687cfd1a429bc3d69511668898e51420c319bae1834a2ab.jpg)

![948451a9d9e4c9d2be44235adfb94de370e19914e695fb15f92f8bf62e45470a.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/tables/948451a9d9e4c9d2be44235adfb94de370e19914e695fb15f92f8bf62e45470a.jpg)

![bde73dcf3c74c3660239b022cc39e8c086d9e0f495ad87ab22b59b367ca49d8b.jpg](../icml_results/2151_When%20Model%20Knowledge%20meets%20Diffusion%20Model_%20Diffusion-assisted%20Data-free%20Image%20Synthesis%20with%20Alignm/tables/bde73dcf3c74c3660239b022cc39e8c086d9e0f495ad87ab22b59b367ca49d8b.jpg)

## Bridging Fairness and Efficiency in Conformal Inference: A Surrogate-Assisted Group-Clustered Approach


### Images

![34b9f49fb7a75968ebb93d0bd27fc1617f43f6c0283642cfc7882a56ea7beda4.jpg](../icml_results/2152_Bridging%20Fairness%20and%20Efficiency%20in%20Conformal%20Inference_%20A%20Surrogate-Assisted%20Group-Clustered%20Approa/images/34b9f49fb7a75968ebb93d0bd27fc1617f43f6c0283642cfc7882a56ea7beda4.jpg)

![65d37ef097d5689206574d0e6546e0bd4fd13729ec08ff729526f45cd2856743.jpg](../icml_results/2152_Bridging%20Fairness%20and%20Efficiency%20in%20Conformal%20Inference_%20A%20Surrogate-Assisted%20Group-Clustered%20Approa/images/65d37ef097d5689206574d0e6546e0bd4fd13729ec08ff729526f45cd2856743.jpg)

![8b883fa4cc29f49cf4705a790a288c19459346240e4567f0a1b5360503d8cb52.jpg](../icml_results/2152_Bridging%20Fairness%20and%20Efficiency%20in%20Conformal%20Inference_%20A%20Surrogate-Assisted%20Group-Clustered%20Approa/images/8b883fa4cc29f49cf4705a790a288c19459346240e4567f0a1b5360503d8cb52.jpg)

![f50c9052b8e8525caadf2fa9464cfd972a880a583c540b7c887d325e868a3c82.jpg](../icml_results/2152_Bridging%20Fairness%20and%20Efficiency%20in%20Conformal%20Inference_%20A%20Surrogate-Assisted%20Group-Clustered%20Approa/images/f50c9052b8e8525caadf2fa9464cfd972a880a583c540b7c887d325e868a3c82.jpg)

### Tables

![834be347cd3e138cfcfa76e0e9b727ad5a3858d84c1c84109dd308668c4dd435.jpg](../icml_results/2152_Bridging%20Fairness%20and%20Efficiency%20in%20Conformal%20Inference_%20A%20Surrogate-Assisted%20Group-Clustered%20Approa/tables/834be347cd3e138cfcfa76e0e9b727ad5a3858d84c1c84109dd308668c4dd435.jpg)

![d064f1ff6942aa78d89e4ac0400a20cd59004dad0d212ddc611b53c51f766981.jpg](../icml_results/2152_Bridging%20Fairness%20and%20Efficiency%20in%20Conformal%20Inference_%20A%20Surrogate-Assisted%20Group-Clustered%20Approa/tables/d064f1ff6942aa78d89e4ac0400a20cd59004dad0d212ddc611b53c51f766981.jpg)

## TinyMIG: Transferring Generalization from Vision Foundation Models to Single-Domain Medical Imaging


### Images

![078602b0b8a5aa9b1d850223ae1151fc69e5b6dffc06ad2b297f12bb3ba93131.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/078602b0b8a5aa9b1d850223ae1151fc69e5b6dffc06ad2b297f12bb3ba93131.jpg)

![455aa48909addf4da40f6104947729e99d085b2f3a334e5d1aa7b476ed6f15f0.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/455aa48909addf4da40f6104947729e99d085b2f3a334e5d1aa7b476ed6f15f0.jpg)

![510e62ce173a780b5745494a0aa544ddce18906fee346303fe2933177bae296a.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/510e62ce173a780b5745494a0aa544ddce18906fee346303fe2933177bae296a.jpg)

![6ef556dc8e3e81180c289e4855984c15f990b6a69b53d8445a8255d70d757996.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/6ef556dc8e3e81180c289e4855984c15f990b6a69b53d8445a8255d70d757996.jpg)

![942fdbf44a995e9491946b540aae1fb15c0e9a4e2964f8f7aa5f8fcdd3db3845.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/942fdbf44a995e9491946b540aae1fb15c0e9a4e2964f8f7aa5f8fcdd3db3845.jpg)

![ade42bcf11f59ba68660f796e6ac68d7216e73ba5fed7d16aa9573f5e60b71a4.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/ade42bcf11f59ba68660f796e6ac68d7216e73ba5fed7d16aa9573f5e60b71a4.jpg)

![b6984673218ab4f95e0b99c91b0f9194cfa053283e174d489acb347659be8a7a.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/b6984673218ab4f95e0b99c91b0f9194cfa053283e174d489acb347659be8a7a.jpg)

![e4946c82f87a853bf35165a50f456b9cf2d43e8f6cd3db71196600966bb3f6ac.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/e4946c82f87a853bf35165a50f456b9cf2d43e8f6cd3db71196600966bb3f6ac.jpg)

![fa19715d7412f54cc1bb40928a0384b4c68a6bd104f7266f492e13c6466e490a.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/fa19715d7412f54cc1bb40928a0384b4c68a6bd104f7266f492e13c6466e490a.jpg)

![fe159fbed86f5ee429cdfecd1d0fb6749e09997189802f5520cbe676acf3738d.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/images/fe159fbed86f5ee429cdfecd1d0fb6749e09997189802f5520cbe676acf3738d.jpg)

### Tables

![0f0903d29dd85f8fb15a637c0ed7c6daa9b0eb2f9bcc8dcc1aed5757df6ea14b.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/0f0903d29dd85f8fb15a637c0ed7c6daa9b0eb2f9bcc8dcc1aed5757df6ea14b.jpg)

![461ff0011aacbd1792a0f326879e7391899f92b7f9aa94afe08e11ebe119e026.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/461ff0011aacbd1792a0f326879e7391899f92b7f9aa94afe08e11ebe119e026.jpg)

![4bd708ac139e042fa99d39725c807483c95b13bdc174a9fed9eff7d35cbd70b9.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/4bd708ac139e042fa99d39725c807483c95b13bdc174a9fed9eff7d35cbd70b9.jpg)

![5eb6cf94663ff085d5bce3056bc3688ab091b8dc32f34252ba398ef2cd19e0f5.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/5eb6cf94663ff085d5bce3056bc3688ab091b8dc32f34252ba398ef2cd19e0f5.jpg)

![82c3ca35143301a77cd1049b3c569008bf759ead0ce75fd7446febcbb5ab8295.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/82c3ca35143301a77cd1049b3c569008bf759ead0ce75fd7446febcbb5ab8295.jpg)

![9ce86c94fdad4a67763e3e7ab162888a3e6b2a31d90570bb6805b4a545815638.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/9ce86c94fdad4a67763e3e7ab162888a3e6b2a31d90570bb6805b4a545815638.jpg)

![d7ca179e7941fd04eff468a92019012412625890d3ad5c757c4d38568a842f26.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/d7ca179e7941fd04eff468a92019012412625890d3ad5c757c4d38568a842f26.jpg)

![e87bd1327074769421779b223fdfa5caab8f651585712e273592a2f8378dce9a.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/e87bd1327074769421779b223fdfa5caab8f651585712e273592a2f8378dce9a.jpg)

![f2745bfab020709c90903b45e4db9c6ef1b9a3bb4aae8a4820d2eb5f0592a1b1.jpg](../icml_results/2153_TinyMIG_%20Transferring%20Generalization%20from%20Vision%20Foundation%20Models%20to%20Single-Domain%20Medical%20Imaging/tables/f2745bfab020709c90903b45e4db9c6ef1b9a3bb4aae8a4820d2eb5f0592a1b1.jpg)

## An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks


### Images

![021825e9f3dd1021d7da4f8db8a1f2b58a0d4eaaf3ca7a70ee73e913dee18c1f.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/021825e9f3dd1021d7da4f8db8a1f2b58a0d4eaaf3ca7a70ee73e913dee18c1f.jpg)

![05e12e3bc1ddd4c4c38d195af1fe3e631cde97c7a327ea53712a7c55dd6d96ca.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/05e12e3bc1ddd4c4c38d195af1fe3e631cde97c7a327ea53712a7c55dd6d96ca.jpg)

![1dd0b51fcf49aa6dc688dc6c9bd7c85737b7fba7449802032e9922e93b174c24.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/1dd0b51fcf49aa6dc688dc6c9bd7c85737b7fba7449802032e9922e93b174c24.jpg)

![4dbaa3ea33cdf3c98935289c3335db31b317cacef94276fc136f3906292a373e.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/4dbaa3ea33cdf3c98935289c3335db31b317cacef94276fc136f3906292a373e.jpg)

![5674818a5da3d31cee321944235f8e7a298dd9c74cdd402e5346f37919ef4725.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/5674818a5da3d31cee321944235f8e7a298dd9c74cdd402e5346f37919ef4725.jpg)

![5b2d8c250d92f58a567992e58ffc29a9ebdeeff127c429457b045ff8b594a1cd.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/5b2d8c250d92f58a567992e58ffc29a9ebdeeff127c429457b045ff8b594a1cd.jpg)

![6d841bf14862811aea8b6b64790a9cc5bfa6e460d63036c05f825258cea3cdd8.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/6d841bf14862811aea8b6b64790a9cc5bfa6e460d63036c05f825258cea3cdd8.jpg)

![81557eb29a7e992732685f4355fd138f80a1385ec326c0d55c25b982ae7650e4.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/81557eb29a7e992732685f4355fd138f80a1385ec326c0d55c25b982ae7650e4.jpg)

![9de2f0ac8f25be447e59bcbb3cc33496c339e7d9bd6d4aa2bbdd12229da3c4cc.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/9de2f0ac8f25be447e59bcbb3cc33496c339e7d9bd6d4aa2bbdd12229da3c4cc.jpg)

![a61875e03de61fbf95631b53943879a29650c7361471ebbd221d33975f6d8c62.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/a61875e03de61fbf95631b53943879a29650c7361471ebbd221d33975f6d8c62.jpg)

![a7daed0ebdeee9f8ee940c24d08303a08f5b091b418fe3e628fd77ce8c82777f.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/a7daed0ebdeee9f8ee940c24d08303a08f5b091b418fe3e628fd77ce8c82777f.jpg)

![aa69c4534cc25d55bec6fd12fe68a6bc7bc756c1e90d41b5dfbc0d87ffb256ce.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/aa69c4534cc25d55bec6fd12fe68a6bc7bc756c1e90d41b5dfbc0d87ffb256ce.jpg)

![b4094d83fbdc738e4fca70635519c5d16f8ff6cc2615d7017e734cb500ebe492.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/b4094d83fbdc738e4fca70635519c5d16f8ff6cc2615d7017e734cb500ebe492.jpg)

![c21245c7f8655ee5266688cc2053694351a7ad8ea31ab603a3774f6fa3144cf0.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/c21245c7f8655ee5266688cc2053694351a7ad8ea31ab603a3774f6fa3144cf0.jpg)

![ca5b143a1bd73b4353d86803c64a9fba9695a02b259288b704f097f64e503e2b.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/images/ca5b143a1bd73b4353d86803c64a9fba9695a02b259288b704f097f64e503e2b.jpg)

### Tables

![25c11f27e468ae50234ccf4c2fe65eb30678ca88046b2c6d1aaef3dfb6211ac1.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/25c11f27e468ae50234ccf4c2fe65eb30678ca88046b2c6d1aaef3dfb6211ac1.jpg)

![696d08388c955754ecc44e3484f67d9e164f1242a1d0a0882a513b5b99106949.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/696d08388c955754ecc44e3484f67d9e164f1242a1d0a0882a513b5b99106949.jpg)

![8737c77703d2c76e45b84c723ef357b1075a0c2d5c885c5ef9252f430ec847ff.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/8737c77703d2c76e45b84c723ef357b1075a0c2d5c885c5ef9252f430ec847ff.jpg)

![b4d658b8ec57dbdfc5c8a3ee1b51a5bbdd4fdbe0a8ea7a6a11af945054fb77d7.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/b4d658b8ec57dbdfc5c8a3ee1b51a5bbdd4fdbe0a8ea7a6a11af945054fb77d7.jpg)

![bc66392d1e9e575fdd71ac9a51e3b07a1ef4fdcde93b31c1660a58e4b25a4223.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/bc66392d1e9e575fdd71ac9a51e3b07a1ef4fdcde93b31c1660a58e4b25a4223.jpg)

![c305fe6ee31e552485c188e893cf941595ac3f9c21892c58564b2cb5cb43b4cd.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/c305fe6ee31e552485c188e893cf941595ac3f9c21892c58564b2cb5cb43b4cd.jpg)

![ef75440087ef8bd8e00b582000693f89e1c32d23279fa6544b93542d6913f328.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/ef75440087ef8bd8e00b582000693f89e1c32d23279fa6544b93542d6913f328.jpg)

![f67b7654b30195d32cc9ef27f17895b5718a11ed96277a4a4f14103ae3c0add4.jpg](../icml_results/2154_An%20Efficient%20Matrix%20Multiplication%20Algorithm%20for%20Accelerating%20Inference%20in%20Binary%20and%20Ternary%20Neural/tables/f67b7654b30195d32cc9ef27f17895b5718a11ed96277a4a4f14103ae3c0add4.jpg)

## AUTOCIRCUIT-RL: Reinforcement Learning-Driven LLM for Automated Circuit Topology Generation


### Images

![14af74c7f1f6d056511f03cffa84ddd2756863f339355acc0a03d88007fa7158.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/14af74c7f1f6d056511f03cffa84ddd2756863f339355acc0a03d88007fa7158.jpg)

![1f22e9c4f0a0f63ed7f635dc36dbe87e419969ad48c8c41a10714d9e82da9d15.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/1f22e9c4f0a0f63ed7f635dc36dbe87e419969ad48c8c41a10714d9e82da9d15.jpg)

![50149600f95bf95ee8e8997397b64a5de6b2ade07690a41d7c9790dbc595949b.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/50149600f95bf95ee8e8997397b64a5de6b2ade07690a41d7c9790dbc595949b.jpg)

![6aba1ae4fcd5129695cfdb7b24bb321ad8c246be5e3068a233c25b92bf42d1d9.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/6aba1ae4fcd5129695cfdb7b24bb321ad8c246be5e3068a233c25b92bf42d1d9.jpg)

![9c84a2415e6916f2b87332d78caf80fb15ea5448923b08ca1a35afe297f8919b.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/9c84a2415e6916f2b87332d78caf80fb15ea5448923b08ca1a35afe297f8919b.jpg)

![a15ca55cc928a9ff120c7cbabddd82749aeff8a608093090bc0b57f115789b24.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/a15ca55cc928a9ff120c7cbabddd82749aeff8a608093090bc0b57f115789b24.jpg)

![f92137dbc6e90861a60229e65e855d4527df502bb324dc034221f1a3676c204c.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/f92137dbc6e90861a60229e65e855d4527df502bb324dc034221f1a3676c204c.jpg)

![fce4f2bde6cdc7a95d0261be3fd7010c00f145ca02c2622867ccee2c993159b4.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/images/fce4f2bde6cdc7a95d0261be3fd7010c00f145ca02c2622867ccee2c993159b4.jpg)

### Tables

![5513a29c914ded0e4953d012d3946014c38961b088249736cb4741a89bebcb49.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/tables/5513a29c914ded0e4953d012d3946014c38961b088249736cb4741a89bebcb49.jpg)

![99a313cbcb1c9de3d9e5f3eac758208719f813416e642b539c2670102cfd5d1a.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/tables/99a313cbcb1c9de3d9e5f3eac758208719f813416e642b539c2670102cfd5d1a.jpg)

![b696a6528f64ee953dbbddc0fc5818724108a4e3c1ab5fc6c4f13f964ab2d1c5.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/tables/b696a6528f64ee953dbbddc0fc5818724108a4e3c1ab5fc6c4f13f964ab2d1c5.jpg)

![cc0031646a64717c12ea35a698f7f8c536179b3ffc4f9e3a7f8f1e777704854f.jpg](../icml_results/2155_AUTOCIRCUIT-RL_%20Reinforcement%20Learning-Driven%20LLM%20for%20Automated%20Circuit%20Topology%20Generation/tables/cc0031646a64717c12ea35a698f7f8c536179b3ffc4f9e3a7f8f1e777704854f.jpg)

## Scalable Gaussian Processes with Latent Kronecker Structure


### Images

![1eb7825e3703ffa1687a7b28c7bb27c28501d0dd4d819845083c2bde8a394f98.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/images/1eb7825e3703ffa1687a7b28c7bb27c28501d0dd4d819845083c2bde8a394f98.jpg)

![812877e93b639ac29377a5266a5e280f6ecbc147f190a54eaf58f730024d2a3e.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/images/812877e93b639ac29377a5266a5e280f6ecbc147f190a54eaf58f730024d2a3e.jpg)

![8d24327cf847ad9731270da16acf6226f215638d681f02c3c03887cf933bd8ab.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/images/8d24327cf847ad9731270da16acf6226f215638d681f02c3c03887cf933bd8ab.jpg)

![d6374e0a42a8068e6a55b884ebdb2bd9625988350ae7a1943ce3b9a2f697bf20.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/images/d6374e0a42a8068e6a55b884ebdb2bd9625988350ae7a1943ce3b9a2f697bf20.jpg)

![dca41d064eb83cee0ce3af5ce70e2d389a1bb2c2348b51a4f1e68e17eb829cee.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/images/dca41d064eb83cee0ce3af5ce70e2d389a1bb2c2348b51a4f1e68e17eb829cee.jpg)

### Tables

![02c3fb4d319be6d5a6dd9dd8389fa7304ea5fc77c168ae45937351a6bffc761a.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/tables/02c3fb4d319be6d5a6dd9dd8389fa7304ea5fc77c168ae45937351a6bffc761a.jpg)

![363a0074f9ad1b776fe7c36e0ce6f0eda4177046fed2e0307faa450fcd22f17f.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/tables/363a0074f9ad1b776fe7c36e0ce6f0eda4177046fed2e0307faa450fcd22f17f.jpg)

![4030581c1facf6e0bf539d2fc88246376fc5be819083b87e61ae54b14c17878f.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/tables/4030581c1facf6e0bf539d2fc88246376fc5be819083b87e61ae54b14c17878f.jpg)

![5d9c8760d20e119b128411b5f05a7084972dd3bfef46c049fbdeba243572e79e.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/tables/5d9c8760d20e119b128411b5f05a7084972dd3bfef46c049fbdeba243572e79e.jpg)

![9d9124c0a41da1b24d128b0d4328ae9035a25577a312b2aa837bb9084414b887.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/tables/9d9124c0a41da1b24d128b0d4328ae9035a25577a312b2aa837bb9084414b887.jpg)

![bd98b35fa829e1a55d67b8c5a34fc334754d7282b17536b58abb128308265342.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/tables/bd98b35fa829e1a55d67b8c5a34fc334754d7282b17536b58abb128308265342.jpg)

![c36751dc5a1012c80e8b767ea6f78300eb0db78fb5f2607de38b36aa6c259ec0.jpg](../icml_results/2156_Scalable%20Gaussian%20Processes%20with%20Latent%20Kronecker%20Structure/tables/c36751dc5a1012c80e8b767ea6f78300eb0db78fb5f2607de38b36aa6c259ec0.jpg)

## DMM: Distributed Matrix Mechanism for Differentially-Private Federated Learning Based on Constant-Overhead Linear Secret Resharing


### Images

![10539abb6131b2d7246b212ddfaa23277b1d0f381575fbc3a62f2deb5ab19ccf.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/images/10539abb6131b2d7246b212ddfaa23277b1d0f381575fbc3a62f2deb5ab19ccf.jpg)

![691636f88fa47deb64ce266d3480de13b64c7b0dc6388b2a9fab0de9ca08ae84.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/images/691636f88fa47deb64ce266d3480de13b64c7b0dc6388b2a9fab0de9ca08ae84.jpg)

![8fddea212edb578e264ed56022915a04249a0ca44be3ab367737c6bd297562e0.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/images/8fddea212edb578e264ed56022915a04249a0ca44be3ab367737c6bd297562e0.jpg)

![b8302c6bae5be035c0cbfa166298ba6da3a5eab92e76997078e2a6ce19e91fb7.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/images/b8302c6bae5be035c0cbfa166298ba6da3a5eab92e76997078e2a6ce19e91fb7.jpg)

![d379b8bcb01fa62d1e37f3a0a43ef32e2e7f3c2ac16df2c738a11353675f02a1.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/images/d379b8bcb01fa62d1e37f3a0a43ef32e2e7f3c2ac16df2c738a11353675f02a1.jpg)

![fa6ad434d680b9d7499b17e6b253693992f3a4a6be605d96c2b1d284f973cb84.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/images/fa6ad434d680b9d7499b17e6b253693992f3a4a6be605d96c2b1d284f973cb84.jpg)

### Tables

![332c308fac42d685f2e851a40e133073588ebeb86fb138cc37a9045b582684a1.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/tables/332c308fac42d685f2e851a40e133073588ebeb86fb138cc37a9045b582684a1.jpg)

![fb023f3ac45fa278d37a814185cbaa0272e1889096fab36d39e69fc349bc90a4.jpg](../icml_results/2157_DMM_%20Distributed%20Matrix%20Mechanism%20for%20Differentially-Private%20Federated%20Learning%20Based%20on%20Constant-Ov/tables/fb023f3ac45fa278d37a814185cbaa0272e1889096fab36d39e69fc349bc90a4.jpg)

## Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning


### Images

![0713231cfb16ca817db3c3fa5bf4213a0c59024cfe4b2f0016ead1a3b79ab00c.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/0713231cfb16ca817db3c3fa5bf4213a0c59024cfe4b2f0016ead1a3b79ab00c.jpg)

![193e679d70c1605d3083aed9da4e5fc31d172e50166b89dbdffc9b7d168dcba3.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/193e679d70c1605d3083aed9da4e5fc31d172e50166b89dbdffc9b7d168dcba3.jpg)

![1bfc1ff565e56346ecdf733680d95ab65f16968fd34f94afcae40a27241c3289.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/1bfc1ff565e56346ecdf733680d95ab65f16968fd34f94afcae40a27241c3289.jpg)

![220de7c1478ef9b23b28acd3213e91b5b3e9bfafb4740999473c629799ed7cc4.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/220de7c1478ef9b23b28acd3213e91b5b3e9bfafb4740999473c629799ed7cc4.jpg)

![284f8e7c47316c39f8b0a246909d890f42e0ff8c0f6d1ba4f439ccdb93859288.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/284f8e7c47316c39f8b0a246909d890f42e0ff8c0f6d1ba4f439ccdb93859288.jpg)

![299343cbe72aaf69d975c31c90eb30c786d0592c36408f123191aaffc6bb4252.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/299343cbe72aaf69d975c31c90eb30c786d0592c36408f123191aaffc6bb4252.jpg)

![2b79b81455c6a183d3dfaaea3d3f2a6e4255c3192a2f4bb3e42df98269eb5215.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/2b79b81455c6a183d3dfaaea3d3f2a6e4255c3192a2f4bb3e42df98269eb5215.jpg)

![361e8e0a3d1a0f10181c9b472d575c492d1f52ce0b621b730159e7039bf98c3b.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/361e8e0a3d1a0f10181c9b472d575c492d1f52ce0b621b730159e7039bf98c3b.jpg)

![365ea4823f011eed2e69bbc8ca975d4cadb884d09bfd0921753854684e211f9f.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/365ea4823f011eed2e69bbc8ca975d4cadb884d09bfd0921753854684e211f9f.jpg)

![4a09ee70171c7d886ea797fc5410358556d446a025ed3e8ad59f246a0f4f196c.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/4a09ee70171c7d886ea797fc5410358556d446a025ed3e8ad59f246a0f4f196c.jpg)

![4b8b01683fc9e257a4d5fa60c0fde959c5d317d056b8146b27061585ea94dad1.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/4b8b01683fc9e257a4d5fa60c0fde959c5d317d056b8146b27061585ea94dad1.jpg)

![81394efdce69e5a2d40a49553ef7639402821218f1bb0e5c0eb42f869df617a1.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/81394efdce69e5a2d40a49553ef7639402821218f1bb0e5c0eb42f869df617a1.jpg)

![8d8929c3978f35c4214f922b3690cc0667e4f3606b491b251d227270ac32a516.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/8d8929c3978f35c4214f922b3690cc0667e4f3606b491b251d227270ac32a516.jpg)

![909bac9849f9df5c246c5089640e9d4017c02d1d0a644b9aa793c233f87825ed.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/909bac9849f9df5c246c5089640e9d4017c02d1d0a644b9aa793c233f87825ed.jpg)

![b808448b9c72eda1347e7568f39f017da5337324f8c0cb4ca1270ef4db23994f.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/b808448b9c72eda1347e7568f39f017da5337324f8c0cb4ca1270ef4db23994f.jpg)

![d3cc60d1955a6a5fadc739a47301170c5c2035aa0d1283b9204f893dc2d77b2d.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/d3cc60d1955a6a5fadc739a47301170c5c2035aa0d1283b9204f893dc2d77b2d.jpg)

![d7beb63b5d378151894b18afca38e7b9b6d9a26f621fa074e1c5aa98ca891f12.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/d7beb63b5d378151894b18afca38e7b9b6d9a26f621fa074e1c5aa98ca891f12.jpg)

![dc2008785414c09686736f378753e01b2a0c9e8f571ff539351f93bae6d7771a.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/dc2008785414c09686736f378753e01b2a0c9e8f571ff539351f93bae6d7771a.jpg)

![ea446c63894dd85b126db9e9827bc687c1780c5aace158f3732758afb3797df6.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/images/ea446c63894dd85b126db9e9827bc687c1780c5aace158f3732758afb3797df6.jpg)

### Tables

![0fddc50e8532092a691a338217967ec007b92ea2366d989fcb9a30281dc1a8ce.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/0fddc50e8532092a691a338217967ec007b92ea2366d989fcb9a30281dc1a8ce.jpg)

![26eba269152bb1f77a2478faba75d9a9bb33f5cb0a6106fe71fda7e14e92787d.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/26eba269152bb1f77a2478faba75d9a9bb33f5cb0a6106fe71fda7e14e92787d.jpg)

![3deeefb5bd99f6623bb88abd4727a50ae9f95ec1c0f1758cb81b5d978800bb41.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/3deeefb5bd99f6623bb88abd4727a50ae9f95ec1c0f1758cb81b5d978800bb41.jpg)

![579620c246b3a4dce1ab43eb8db283ee52b310bb0e31d3148cbcc2368016cbbc.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/579620c246b3a4dce1ab43eb8db283ee52b310bb0e31d3148cbcc2368016cbbc.jpg)

![58e917f70596ad181905d1c914b266accdf022f09235ae0b65d4c1ea718939a3.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/58e917f70596ad181905d1c914b266accdf022f09235ae0b65d4c1ea718939a3.jpg)

![706a1697bc360dacd3de64643797ebfc616e4609e22583482a55defc261338ee.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/706a1697bc360dacd3de64643797ebfc616e4609e22583482a55defc261338ee.jpg)

![74d36388bb8d7ff1e0f1f8d03babfa14cc8ecf47f3b9a6e799924ba78091f379.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/74d36388bb8d7ff1e0f1f8d03babfa14cc8ecf47f3b9a6e799924ba78091f379.jpg)

![761cda08a85161769654b4e5d8d94477777c9ab1b7b9e73a3fc3b1126a31ce94.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/761cda08a85161769654b4e5d8d94477777c9ab1b7b9e73a3fc3b1126a31ce94.jpg)

![828b9959e57cae846ab19932f88807781ed38d00ee56c3eae49aff3d059427fd.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/828b9959e57cae846ab19932f88807781ed38d00ee56c3eae49aff3d059427fd.jpg)

![89bacfcc13f7f6fdf4033a27a300847c3a532364649e2a2fcc126af25ad988ca.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/89bacfcc13f7f6fdf4033a27a300847c3a532364649e2a2fcc126af25ad988ca.jpg)

![936ceaa0b87d4b4a94d93df9a70d9f88aceb083b517c0de868884277866a339c.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/936ceaa0b87d4b4a94d93df9a70d9f88aceb083b517c0de868884277866a339c.jpg)

![9c0d2e8b57e6d265a35ef7f592abd8b43b0280427aa7271ed840bf4877d3735e.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/9c0d2e8b57e6d265a35ef7f592abd8b43b0280427aa7271ed840bf4877d3735e.jpg)

![ab3023a778137f21954d1e4a6c9d04ee60f313031d2036edf0bdddc913400b9c.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/ab3023a778137f21954d1e4a6c9d04ee60f313031d2036edf0bdddc913400b9c.jpg)

![b5f25f7c7ec003d2ea30e1a93124b70827cd27926b799df167aeeff2fab0941b.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/b5f25f7c7ec003d2ea30e1a93124b70827cd27926b799df167aeeff2fab0941b.jpg)

![b6a232bad4de2b308d6cec67453664149bfff312daa547e11a9696cd8b1c7fd4.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/b6a232bad4de2b308d6cec67453664149bfff312daa547e11a9696cd8b1c7fd4.jpg)

![c3e23d8b574114fdc9ba8daac8283515f49d7ef1f3c96589a94336236f2e3348.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/c3e23d8b574114fdc9ba8daac8283515f49d7ef1f3c96589a94336236f2e3348.jpg)

![d026341bd004ecb4cdfde1d3d66e950287317f547ffd0ac6b5dd29ac264986bb.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/d026341bd004ecb4cdfde1d3d66e950287317f547ffd0ac6b5dd29ac264986bb.jpg)

![d57ab9cac2bd695bbc69fbeae363fc3537de7493859ce38bc9c458bfa60c70a2.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/d57ab9cac2bd695bbc69fbeae363fc3537de7493859ce38bc9c458bfa60c70a2.jpg)

![e7300fb74473756f907fb6bcc66bca2891a0a664e67c682e00837fb0201eee9c.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/e7300fb74473756f907fb6bcc66bca2891a0a664e67c682e00837fb0201eee9c.jpg)

![f25725b982b90904c94d3a52d43c20e6820d385646c91f04f4626584d4e117d5.jpg](../icml_results/2158_Underestimated%20Privacy%20Risks%20for%20Minority%20Populations%20in%20Large%20Language%20Model%20Unlearning/tables/f25725b982b90904c94d3a52d43c20e6820d385646c91f04f4626584d4e117d5.jpg)

## Balancing Preservation and Modification: A Region and Semantic Aware Metric for Instruction-Based Image Editing


### Images

![00d0b1dcebc5fe668e038ac7a0206206589cbf074e14a16b63b48b32614be629.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/00d0b1dcebc5fe668e038ac7a0206206589cbf074e14a16b63b48b32614be629.jpg)

![10cb16c7915ff25690b55a0253a61687aa1ac33d5105a567b068f8e8fae6048a.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/10cb16c7915ff25690b55a0253a61687aa1ac33d5105a567b068f8e8fae6048a.jpg)

![12902c91ab34fd55e56e58f979a254e44840f704c22e4c7b215ee19e6e1ef1eb.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/12902c91ab34fd55e56e58f979a254e44840f704c22e4c7b215ee19e6e1ef1eb.jpg)

![1c487238415bed43d73fcb4973507345a2a9ddda9ffdb042b1e706bf62a25c28.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/1c487238415bed43d73fcb4973507345a2a9ddda9ffdb042b1e706bf62a25c28.jpg)

![3275372425af1ce55436ba1276f294f3c5e7a6913c115212df61e8dee5cc0804.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/3275372425af1ce55436ba1276f294f3c5e7a6913c115212df61e8dee5cc0804.jpg)

![389c81428360f06f9d7814b9e6e546bec22499587a47710595addce2da34a5d3.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/389c81428360f06f9d7814b9e6e546bec22499587a47710595addce2da34a5d3.jpg)

![39dd38a9e23e4bfd1df0f7e064d984b86ea5aaf588534d59684699e1d74f3e4a.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/39dd38a9e23e4bfd1df0f7e064d984b86ea5aaf588534d59684699e1d74f3e4a.jpg)

![4651d2d46689ecdce1a5ce7a29ce8b8c60742f6fb1f2b37335126ec0c1114d35.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/4651d2d46689ecdce1a5ce7a29ce8b8c60742f6fb1f2b37335126ec0c1114d35.jpg)

![46a166649b5d9b0291a15b12429ca95755e2a54a45d3505f0dbdd61f24aad255.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/46a166649b5d9b0291a15b12429ca95755e2a54a45d3505f0dbdd61f24aad255.jpg)

![4977e7b96031a29c6736f847a501effb1969ad65fac229d60ce3afed67e08285.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/4977e7b96031a29c6736f847a501effb1969ad65fac229d60ce3afed67e08285.jpg)

![5893ee52f3678ebb02e34f48a33c772da077c2ecb51f1c24ee02497a3b0e83cf.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/5893ee52f3678ebb02e34f48a33c772da077c2ecb51f1c24ee02497a3b0e83cf.jpg)

![6cc3bddd222592c546ea110cfb06a347d0635214e08cb0763979e15ac7472aba.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/6cc3bddd222592c546ea110cfb06a347d0635214e08cb0763979e15ac7472aba.jpg)

![6f1d6cbdc0ffea4a8c4fc0a08b295017d65425aa20b4326fc70fcf7b8bf27675.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/6f1d6cbdc0ffea4a8c4fc0a08b295017d65425aa20b4326fc70fcf7b8bf27675.jpg)

![7421aa199e926162ebf221623b9d0cab10d887de08f50a85594750c934303031.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/7421aa199e926162ebf221623b9d0cab10d887de08f50a85594750c934303031.jpg)

![75d31bdf4a118fcf9c3f2af1981a3eec383b3f5040ad11b08230888cc1323e3c.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/75d31bdf4a118fcf9c3f2af1981a3eec383b3f5040ad11b08230888cc1323e3c.jpg)

![b0829e62dc55b0fcb7463f8c89ffca5f918bd23d0a6bd126446295e840e3b8cd.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/b0829e62dc55b0fcb7463f8c89ffca5f918bd23d0a6bd126446295e840e3b8cd.jpg)

![c033e6bfe98ddd1191f16e32ea4fa1bc37eea599e14edbed56cdb8a71c932b0c.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/c033e6bfe98ddd1191f16e32ea4fa1bc37eea599e14edbed56cdb8a71c932b0c.jpg)

![c47e3e26eab0c2781a9937e7b6a616a289a6f2e7ba8f43721ea7f6c8c4f069e9.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/c47e3e26eab0c2781a9937e7b6a616a289a6f2e7ba8f43721ea7f6c8c4f069e9.jpg)

![c6f2f090c7ddfc0d19ca0c26f6266e403853d9a260e499bbfc7e6017e660f038.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/c6f2f090c7ddfc0d19ca0c26f6266e403853d9a260e499bbfc7e6017e660f038.jpg)

![cd079a39741755896fc14c63c8790f371996a959aeb96ccdfb24d29977d500c4.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/cd079a39741755896fc14c63c8790f371996a959aeb96ccdfb24d29977d500c4.jpg)

![e522f68ff8585490ceb2860756ee77bc974e4895fb51035caffdac4ffc657ef6.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/e522f68ff8585490ceb2860756ee77bc974e4895fb51035caffdac4ffc657ef6.jpg)

![f6ec02df9be019a31aa129db32d894a653998d67f738e38f4ebebc0efc08c804.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/images/f6ec02df9be019a31aa129db32d894a653998d67f738e38f4ebebc0efc08c804.jpg)

### Tables

![1542bf134efa0a98af264518f28f3bb791617026810b977247000ddc08a52247.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/1542bf134efa0a98af264518f28f3bb791617026810b977247000ddc08a52247.jpg)

![1c96ca599670b4fddb3a2fee4a9e56850e1e6d3794837df77491456a4e1095db.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/1c96ca599670b4fddb3a2fee4a9e56850e1e6d3794837df77491456a4e1095db.jpg)

![20e05f5524f0f8d24d634492bdff98b5667163fd09a214b81c95138fed7b0dac.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/20e05f5524f0f8d24d634492bdff98b5667163fd09a214b81c95138fed7b0dac.jpg)

![283534a1b343914e597ba165adce3f4effaf2d8cbfeccf07a02832f5989b57f3.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/283534a1b343914e597ba165adce3f4effaf2d8cbfeccf07a02832f5989b57f3.jpg)

![497eea786aaaa2b5733bf18430964dac3c2cb940741d98c38faf1a3ffdcb5d11.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/497eea786aaaa2b5733bf18430964dac3c2cb940741d98c38faf1a3ffdcb5d11.jpg)

![58ca3e2b78e49ae481540cbd08c62dac94451c4618cdedfe1aa44a88ce69288f.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/58ca3e2b78e49ae481540cbd08c62dac94451c4618cdedfe1aa44a88ce69288f.jpg)

![67a6e278ba65c3b0862c8c7f20b58f4021c25573271e028aa08bdefa8a29bf76.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/67a6e278ba65c3b0862c8c7f20b58f4021c25573271e028aa08bdefa8a29bf76.jpg)

![686d04953778cb2305ed4c123aa07c907380f9b070966b65b155935f42394b64.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/686d04953778cb2305ed4c123aa07c907380f9b070966b65b155935f42394b64.jpg)

![68b61be38e2c49a3f953237f333db403c67ed1349395233e99135f8d72b8c03b.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/68b61be38e2c49a3f953237f333db403c67ed1349395233e99135f8d72b8c03b.jpg)

![7612cc3fdbc0a1efaac8aa8d0f0f08645fc7629287a03d3958898a93af1976cc.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/7612cc3fdbc0a1efaac8aa8d0f0f08645fc7629287a03d3958898a93af1976cc.jpg)

![935f72a8ae2464b2e9c759510d3c4839b036edf9221ca115bf9c76a0e808baae.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/935f72a8ae2464b2e9c759510d3c4839b036edf9221ca115bf9c76a0e808baae.jpg)

![97b266988ec644c6108c54ccbf20c5dce2f7bcfc2f839feb62f6601b722e90ee.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/97b266988ec644c6108c54ccbf20c5dce2f7bcfc2f839feb62f6601b722e90ee.jpg)

![da88f95b1f8850d2637f49974faa71517bf6ea312b1215c1df23f60d56894e82.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/da88f95b1f8850d2637f49974faa71517bf6ea312b1215c1df23f60d56894e82.jpg)

![e96c1b226ece6d03e86e065cfe7879f9412acb8b5153b25edc9ccbc35d984a34.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/e96c1b226ece6d03e86e065cfe7879f9412acb8b5153b25edc9ccbc35d984a34.jpg)

![f3f5dbec14c98882c1440d723a7758f9edd893393927a60794ff04105ca55fca.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/f3f5dbec14c98882c1440d723a7758f9edd893393927a60794ff04105ca55fca.jpg)

![fb7e6c51922f0d5961a73790f9c0ba4b31aad433e8236b45c848a5fb8bb94541.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/fb7e6c51922f0d5961a73790f9c0ba4b31aad433e8236b45c848a5fb8bb94541.jpg)

![ff8870a13ac30acf1febae80755003ee5fbbb5e1e95bbcc89d37edec4e944cf2.jpg](../icml_results/2159_Balancing%20Preservation%20and%20Modification_%20A%20Region%20and%20Semantic%20Aware%20Metric%20for%20Instruction-Based%20Im/tables/ff8870a13ac30acf1febae80755003ee5fbbb5e1e95bbcc89d37edec4e944cf2.jpg)

## MARS: Unleashing the Power of Variance Reduction for Training Large Models


### Images

![0a845dadd110f282cd4bbc2691401b4caac2abb2e0eca752a801516e967e5aee.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/0a845dadd110f282cd4bbc2691401b4caac2abb2e0eca752a801516e967e5aee.jpg)

![0e83e60447ccb090ed398aa9ec89752db86992565d526343b4da8e670b332741.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/0e83e60447ccb090ed398aa9ec89752db86992565d526343b4da8e670b332741.jpg)

![33496057af13253e89aca4ad29fa28c7ebbf2a0e5d163296ab864358753ca800.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/33496057af13253e89aca4ad29fa28c7ebbf2a0e5d163296ab864358753ca800.jpg)

![37970a314048d55b36200ee76b28ba9bfa21c066d7e9739cdc088f10f6943617.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/37970a314048d55b36200ee76b28ba9bfa21c066d7e9739cdc088f10f6943617.jpg)

![381936bbf3bf21dd742702927cac887a8c8ba14341abb081c0de9ff1d29f6a5b.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/381936bbf3bf21dd742702927cac887a8c8ba14341abb081c0de9ff1d29f6a5b.jpg)

![38c3d290ee58375514e1578f27f0001faaa90039145a7a2cca5d06719dd93e37.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/38c3d290ee58375514e1578f27f0001faaa90039145a7a2cca5d06719dd93e37.jpg)

![3bf5a2c6c34243a752a444f23d2a1a33e6bedf1a8219928cd3076fcafd85ffd3.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/3bf5a2c6c34243a752a444f23d2a1a33e6bedf1a8219928cd3076fcafd85ffd3.jpg)

![49387fc7f6b1e5c150fd76d67027ee29cde9c9504f31ab9a1b4077558e8173e4.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/49387fc7f6b1e5c150fd76d67027ee29cde9c9504f31ab9a1b4077558e8173e4.jpg)

![62963c3dc6fe900fee1f46c65a8229f42a600fb6807c5899650f6ea536b5e8a2.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/62963c3dc6fe900fee1f46c65a8229f42a600fb6807c5899650f6ea536b5e8a2.jpg)

![6a182bb69db23c39751023d9f4e4194aadfcc3e23fde1205357f08d06ed4077c.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/6a182bb69db23c39751023d9f4e4194aadfcc3e23fde1205357f08d06ed4077c.jpg)

![6f3001738bb20313fc6e20cc81af89befea703dce50c578bae734a80c36ce3a9.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/6f3001738bb20313fc6e20cc81af89befea703dce50c578bae734a80c36ce3a9.jpg)

![837948b1dc73fe010e04b720c5bfb22ae0ed2d10daebe4e9ca5317dd4e45cd88.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/837948b1dc73fe010e04b720c5bfb22ae0ed2d10daebe4e9ca5317dd4e45cd88.jpg)

![91543e4f9c042a48d98a2d328241fd5d158b7deabaa55d8d4607b399e9d2ce91.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/91543e4f9c042a48d98a2d328241fd5d158b7deabaa55d8d4607b399e9d2ce91.jpg)

![c5c1156d14c7de570feed7a0920fd58f6da3136f3168dd0a669d8cbfb8addf5a.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/c5c1156d14c7de570feed7a0920fd58f6da3136f3168dd0a669d8cbfb8addf5a.jpg)

![c619debed864bc967ee00a540ea9a60336d46b0f802526d14f1c94245bd51a11.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/c619debed864bc967ee00a540ea9a60336d46b0f802526d14f1c94245bd51a11.jpg)

![cabd9441daf8dd2480033eb5bb6f1a83bfeb377921a6fd605ba1b05552f1300e.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/cabd9441daf8dd2480033eb5bb6f1a83bfeb377921a6fd605ba1b05552f1300e.jpg)

![ed0ac2f5f2ed19ce951710b2c210edadeef07fae718682a301d11264c72856a2.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/ed0ac2f5f2ed19ce951710b2c210edadeef07fae718682a301d11264c72856a2.jpg)

![f73836794402be94b124ca903c52261263e9bb5e3725c2e1cc8eae7c8a4cac9a.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/f73836794402be94b124ca903c52261263e9bb5e3725c2e1cc8eae7c8a4cac9a.jpg)

![f885aa5064f01197d9789225172ee9b92881aac7bb1cc436a1ba6e103333e17b.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/images/f885aa5064f01197d9789225172ee9b92881aac7bb1cc436a1ba6e103333e17b.jpg)

### Tables

![0661500f9184dc915d12b99d41d8c443c1652ac49e0859955f9c116a030cf25f.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/0661500f9184dc915d12b99d41d8c443c1652ac49e0859955f9c116a030cf25f.jpg)

![13d82c21f92d28b9fe301653ff660f57dba3db74f404620b6b12ea2c0d612bf0.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/13d82c21f92d28b9fe301653ff660f57dba3db74f404620b6b12ea2c0d612bf0.jpg)

![169c782dcc2710da6064a5e1e56ac9bd6a693c59bdf760b847d46be064c91d9e.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/169c782dcc2710da6064a5e1e56ac9bd6a693c59bdf760b847d46be064c91d9e.jpg)

![227ac39b9dd520007543224625271b00b09defe1a7261b6111fbb31defb5285c.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/227ac39b9dd520007543224625271b00b09defe1a7261b6111fbb31defb5285c.jpg)

![2f0e3a47a03de9eb52509e67e9f5ab4641448ae77b42626d6925663d953d7f9c.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/2f0e3a47a03de9eb52509e67e9f5ab4641448ae77b42626d6925663d953d7f9c.jpg)

![45f151db3e86cdf290b97453dda301f0d4708ab7f3113b7e5cff5ca272d00a5b.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/45f151db3e86cdf290b97453dda301f0d4708ab7f3113b7e5cff5ca272d00a5b.jpg)

![4b44cdaecb67f9852ea6e4e976af52b632684eaaeb3383f865ccbfd3e42764f4.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/4b44cdaecb67f9852ea6e4e976af52b632684eaaeb3383f865ccbfd3e42764f4.jpg)

![6806825f1053fec11c45cca213963fb8075cab0c45964c04431e59ca47d0c9a8.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/6806825f1053fec11c45cca213963fb8075cab0c45964c04431e59ca47d0c9a8.jpg)

![8748b0bfb214497b6286109cfaf16751035f0705bf48ecfa52c753b34d846fd1.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/8748b0bfb214497b6286109cfaf16751035f0705bf48ecfa52c753b34d846fd1.jpg)

![91c57c58cd4d0c463124190baf930c45d489e8916707b9d00c243f7aad1c21af.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/91c57c58cd4d0c463124190baf930c45d489e8916707b9d00c243f7aad1c21af.jpg)

![c227c6ef4b86d50353afba4f5dc4d4834946cf60a2b7bc48d0ddcc4caf973a22.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/c227c6ef4b86d50353afba4f5dc4d4834946cf60a2b7bc48d0ddcc4caf973a22.jpg)

![d5b7f9a4fdd2aef6e5b12a3388bc7cd3aa903d3023eafd59bbc67ba3e2e8070a.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/d5b7f9a4fdd2aef6e5b12a3388bc7cd3aa903d3023eafd59bbc67ba3e2e8070a.jpg)

![db01511516f1a9a1471990afb67850c75e8adde31ba6c4b7dccb7ae0f0ce0088.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/db01511516f1a9a1471990afb67850c75e8adde31ba6c4b7dccb7ae0f0ce0088.jpg)

![f58dced94de5478aaba695d77b7fd4bf9a05c7205715ede9af4630bf761c525d.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/f58dced94de5478aaba695d77b7fd4bf9a05c7205715ede9af4630bf761c525d.jpg)

![fbc88a1ca33858489725e62763e97a96129a39cf34c27d34544eb74e4442f03e.jpg](../icml_results/2160_MARS_%20Unleashing%20the%20Power%20of%20Variance%20Reduction%20for%20Training%20Large%20Models/tables/fbc88a1ca33858489725e62763e97a96129a39cf34c27d34544eb74e4442f03e.jpg)

## Improving Value Estimation Critically Enhances Vanilla Policy Gradient


### Images

![1d32ffc19c7f9a751e4e0374af1de2e622410507bfb02036940121e0c571c899.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/1d32ffc19c7f9a751e4e0374af1de2e622410507bfb02036940121e0c571c899.jpg)

![4ac965fb215efe4e2b64685467a29b5fc0462a31f8753d0d16768fbab3eaec84.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/4ac965fb215efe4e2b64685467a29b5fc0462a31f8753d0d16768fbab3eaec84.jpg)

![5a364864544a11c3746d40f81ca3a3b3f9dd5a1da206e56d344fcf4c70ad0fc7.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/5a364864544a11c3746d40f81ca3a3b3f9dd5a1da206e56d344fcf4c70ad0fc7.jpg)

![6543fbd1b285334b06f90a3fb8ab44be44e551d14c9a7d29204781b7b2ffc50f.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/6543fbd1b285334b06f90a3fb8ab44be44e551d14c9a7d29204781b7b2ffc50f.jpg)

![73457557cb5c5dcabf221713c82fc4466a48e7a1f96cb9bd3c4555e12e6febd5.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/73457557cb5c5dcabf221713c82fc4466a48e7a1f96cb9bd3c4555e12e6febd5.jpg)

![818f92d81409fddaeabf7e2888ee8c6bd296cadfd5624f2bc5dd28d318100769.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/818f92d81409fddaeabf7e2888ee8c6bd296cadfd5624f2bc5dd28d318100769.jpg)

![8712680216b1f80ec99bdd6a7501dff14b2550d1189ac8167717a36399fe1d25.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/8712680216b1f80ec99bdd6a7501dff14b2550d1189ac8167717a36399fe1d25.jpg)

![8b3edf1b1f6ee1a116dd967491674810a1c94e33e9d5ab14b05a65af8bef3e8a.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/8b3edf1b1f6ee1a116dd967491674810a1c94e33e9d5ab14b05a65af8bef3e8a.jpg)

![a51501e7a51b27c14e1518f89664a49ece1f025ae1af64ce79df9bd5570b8c40.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/a51501e7a51b27c14e1518f89664a49ece1f025ae1af64ce79df9bd5570b8c40.jpg)

![cdd7071d317af675d6c9e7d3d941c6c7da9872a6f34131fe868bcceeee13c6e7.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/cdd7071d317af675d6c9e7d3d941c6c7da9872a6f34131fe868bcceeee13c6e7.jpg)

![ea61c2cef40e06e30c79c3b9dd257f254e0b5da94099e0e4fbcac1eb233691e5.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/ea61c2cef40e06e30c79c3b9dd257f254e0b5da94099e0e4fbcac1eb233691e5.jpg)

![f0cd166b67aed90abd1a86985d03cf39d741119d7b8335b26e2af3900c0a5e5d.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/images/f0cd166b67aed90abd1a86985d03cf39d741119d7b8335b26e2af3900c0a5e5d.jpg)

### Tables

![0494bc92c6a276a6d8f08e875c65e3fee47612537ff302f11f3c5bca9a9bfa5c.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/tables/0494bc92c6a276a6d8f08e875c65e3fee47612537ff302f11f3c5bca9a9bfa5c.jpg)

![06aa0fb2d0ebdc1e198e13785b2f2d2504c948c8e5d0e7d4c59322f99e0c5c2b.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/tables/06aa0fb2d0ebdc1e198e13785b2f2d2504c948c8e5d0e7d4c59322f99e0c5c2b.jpg)

![349d51ec24a71370dfec7b7a69aa530702e32b977d4ea523297264841a9fa746.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/tables/349d51ec24a71370dfec7b7a69aa530702e32b977d4ea523297264841a9fa746.jpg)

![d38368e0335604a34dec37f0fb4cfcf501bdd0da5bb22f0668024fc3cd751ecd.jpg](../icml_results/2161_Improving%20Value%20Estimation%20Critically%20Enhances%20Vanilla%20Policy%20Gradient/tables/d38368e0335604a34dec37f0fb4cfcf501bdd0da5bb22f0668024fc3cd751ecd.jpg)

## Refining Adaptive Zeroth-Order Optimization at Ease


### Images

![6e7194ee3bb7c4346472d999d842f32aca58f86994f3284b2f2e6b1935d5e6f3.jpg](../icml_results/2162_Refining%20Adaptive%20Zeroth-Order%20Optimization%20at%20Ease/images/6e7194ee3bb7c4346472d999d842f32aca58f86994f3284b2f2e6b1935d5e6f3.jpg)

![bed35bce3bf60912fb3ccdb2bd035a17d2cf526ab95c7725d661125b7c50f010.jpg](../icml_results/2162_Refining%20Adaptive%20Zeroth-Order%20Optimization%20at%20Ease/images/bed35bce3bf60912fb3ccdb2bd035a17d2cf526ab95c7725d661125b7c50f010.jpg)

![da6ccb1761296ed457cb361f4e4dae7a8b76beb542b9dda01b0566e83578a02d.jpg](../icml_results/2162_Refining%20Adaptive%20Zeroth-Order%20Optimization%20at%20Ease/images/da6ccb1761296ed457cb361f4e4dae7a8b76beb542b9dda01b0566e83578a02d.jpg)

### Tables

![c85778ae1ae2ffdf54c954d10010e981b7972a952645ea3ddff1370fba8686e2.jpg](../icml_results/2162_Refining%20Adaptive%20Zeroth-Order%20Optimization%20at%20Ease/tables/c85778ae1ae2ffdf54c954d10010e981b7972a952645ea3ddff1370fba8686e2.jpg)

## Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks


### Images

![164623af2dbda59e8a62b28af89e371ceab3fa8a1d9ab13c058eb3fee644e0f9.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/164623af2dbda59e8a62b28af89e371ceab3fa8a1d9ab13c058eb3fee644e0f9.jpg)

![286672edb5ad036e416cc5b23ebdaa706bd92c551d824ad96ec26f00db67e9b2.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/286672edb5ad036e416cc5b23ebdaa706bd92c551d824ad96ec26f00db67e9b2.jpg)

![4352ce040b4a937cb75c8a687ab3ea902d5f067f657c1622cc13d51e4b483aa7.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/4352ce040b4a937cb75c8a687ab3ea902d5f067f657c1622cc13d51e4b483aa7.jpg)

![7c67cc9cb75cd3385bed28feaf08934e016f47a9f0a3fcc8f7e91bc007898a7e.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/7c67cc9cb75cd3385bed28feaf08934e016f47a9f0a3fcc8f7e91bc007898a7e.jpg)

![afb70b8602aae41dfab6c6b046baa3a3c117173048f22c2c001ed1834ab951d1.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/afb70b8602aae41dfab6c6b046baa3a3c117173048f22c2c001ed1834ab951d1.jpg)

![b5fc6d3501c3a0f6e264852e45219e7d225dbeef15afccf8939b7dd8ea497ad9.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/b5fc6d3501c3a0f6e264852e45219e7d225dbeef15afccf8939b7dd8ea497ad9.jpg)

![c6c0e35dc3cc12f7dcd60336115e364aaae78686138b6d907ed72043bb43b112.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/c6c0e35dc3cc12f7dcd60336115e364aaae78686138b6d907ed72043bb43b112.jpg)

![fc3acd89255ac85c23b81b2aaebdcb770ade6d668998f0d85f218377cf51d966.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/images/fc3acd89255ac85c23b81b2aaebdcb770ade6d668998f0d85f218377cf51d966.jpg)

### Tables

![23c806fb8751deb0f92d03bfef3e957500bc31a417fb42515c934c587e0acad4.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/23c806fb8751deb0f92d03bfef3e957500bc31a417fb42515c934c587e0acad4.jpg)

![314288a063035c36ce92dfdee914425cb5f0cf6a2f9cab646ea461afc2e197b8.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/314288a063035c36ce92dfdee914425cb5f0cf6a2f9cab646ea461afc2e197b8.jpg)

![40cb66f8856d4c55279ca0df10b899c72dc99b26a1f0ded8edcca97774bbd0d0.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/40cb66f8856d4c55279ca0df10b899c72dc99b26a1f0ded8edcca97774bbd0d0.jpg)

![471421fc436a397852e263dec296019897e3e2dfbc6b3800e2f1ce089f3eabb7.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/471421fc436a397852e263dec296019897e3e2dfbc6b3800e2f1ce089f3eabb7.jpg)

![5d319aaaba4988195ce3a375ea42b5f4c8ceeaa7fbfa33215664dbea64e43136.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/5d319aaaba4988195ce3a375ea42b5f4c8ceeaa7fbfa33215664dbea64e43136.jpg)

![6bbc18cc9c58b8c01a8f272efbe1fe5957b84d9579e59c9b6f0ed0c9cc4553a3.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/6bbc18cc9c58b8c01a8f272efbe1fe5957b84d9579e59c9b6f0ed0c9cc4553a3.jpg)

![76874f6e0b6e37951431f57daa9346eabd5b807056df1372b1837b718be234be.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/76874f6e0b6e37951431f57daa9346eabd5b807056df1372b1837b718be234be.jpg)

![794832adad4e3645aed6eb1449a859f348f81fb24594fdf0b06bd595b94216bd.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/794832adad4e3645aed6eb1449a859f348f81fb24594fdf0b06bd595b94216bd.jpg)

![7b25f789d43b113addb88b19916971e7025f31ba2ae15a9734905953159ce2de.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/7b25f789d43b113addb88b19916971e7025f31ba2ae15a9734905953159ce2de.jpg)

![88ef2423699f0e5fb0c31f74f455a826fa467e9dae2da97cd4eb751b423c5eb8.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/88ef2423699f0e5fb0c31f74f455a826fa467e9dae2da97cd4eb751b423c5eb8.jpg)

![91b92fb0358071f35e2ec04b151b2beaa584a76bed7830e1e4c4b0b4431776ab.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/91b92fb0358071f35e2ec04b151b2beaa584a76bed7830e1e4c4b0b4431776ab.jpg)

![968d12eaa91fd126dbc25769968392b455ce28a440b48b9b6932e4895a5ef1d7.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/968d12eaa91fd126dbc25769968392b455ce28a440b48b9b6932e4895a5ef1d7.jpg)

![a739e6540b58117c6abd805de80f8b30b0688313ad2c8cdfd74ae3fadfd207a7.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/a739e6540b58117c6abd805de80f8b30b0688313ad2c8cdfd74ae3fadfd207a7.jpg)

![a9049920f0093c8f0785843c8b647ca8641299be36ad9477cc1ff415b6cf4061.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/a9049920f0093c8f0785843c8b647ca8641299be36ad9477cc1ff415b6cf4061.jpg)

![a9dbe14d784572ef876b7d9f64dc4a2b58375d3885c9bb83bfa0423cebbea97b.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/a9dbe14d784572ef876b7d9f64dc4a2b58375d3885c9bb83bfa0423cebbea97b.jpg)

![b66e822c158f343131f3cdf85ebed113a5c299f6c86e098abc44242d2a47d963.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/b66e822c158f343131f3cdf85ebed113a5c299f6c86e098abc44242d2a47d963.jpg)

![bbbf01946098cabb14eb51c4aad3dd2d01c4a984293e967977448c1e37968dc2.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/bbbf01946098cabb14eb51c4aad3dd2d01c4a984293e967977448c1e37968dc2.jpg)

![c0f932ec3b5f4b356f2c8e4031ce56e7721fa389702ab75ad6b8908b547d12be.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/c0f932ec3b5f4b356f2c8e4031ce56e7721fa389702ab75ad6b8908b547d12be.jpg)

![c8809f38094f7af3a14a89e348c311c14295a17fcb013fe48a4bec63944e3fdc.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/c8809f38094f7af3a14a89e348c311c14295a17fcb013fe48a4bec63944e3fdc.jpg)

![d37338e386c6f5b43991bde36763326aabb10822f016ce598992e105d26675e2.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/d37338e386c6f5b43991bde36763326aabb10822f016ce598992e105d26675e2.jpg)

![e4da91077825b38c6d99c828d6c41969ba255d00b7486acb6d5fd4233a6ffea1.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/e4da91077825b38c6d99c828d6c41969ba255d00b7486acb6d5fd4233a6ffea1.jpg)

![eec8523f8f60322475d4d0fec5c18bd19d3b8a70fa6a24ed234587c1a5fd2520.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/eec8523f8f60322475d4d0fec5c18bd19d3b8a70fa6a24ed234587c1a5fd2520.jpg)

![f1dac26c95ba908c0a9d06687bf1e2604e325b2d1b60a36b59207e7492c9d90f.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/f1dac26c95ba908c0a9d06687bf1e2604e325b2d1b60a36b59207e7492c9d90f.jpg)

![f912d1bdcc1c86a97796535e2d65fce00097d8e11da62d1ddb6e0f9ec0e5217b.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/f912d1bdcc1c86a97796535e2d65fce00097d8e11da62d1ddb6e0f9ec0e5217b.jpg)

![fd0d6da9f70c8ac27de35bfe19fa9d0a8380a55573f2b8ef516e193707d3334d.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/fd0d6da9f70c8ac27de35bfe19fa9d0a8380a55573f2b8ef516e193707d3334d.jpg)

![fea54841ad0f49eb401d447a9e627a4ed99fd3740454787fa284828c37d38b51.jpg](../icml_results/2163_Sparse%20Spectral%20Training%20and%20Inference%20on%20Euclidean%20and%20Hyperbolic%20Neural%20Networks/tables/fea54841ad0f49eb401d447a9e627a4ed99fd3740454787fa284828c37d38b51.jpg)

## MDDM: Practical Message-Driven Generative Image Steganography Based on Diffusion Models


### Images

![0bb7df5330e3e06011dcbb485c30ca0e9c62add6377ae0cb578104065afc3548.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/0bb7df5330e3e06011dcbb485c30ca0e9c62add6377ae0cb578104065afc3548.jpg)

![4f51d936e4f3aaa111b3dde10ab58e62ca2ca58f3b5474d50c1468ce6007edbf.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/4f51d936e4f3aaa111b3dde10ab58e62ca2ca58f3b5474d50c1468ce6007edbf.jpg)

![6dae2e9e3c38cc3ab70ce77ed255bd107c9b5ee2baf6e3d5df7e1902678f7e8c.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/6dae2e9e3c38cc3ab70ce77ed255bd107c9b5ee2baf6e3d5df7e1902678f7e8c.jpg)

![72bbb70a8d19e407c424540c26d02eb7cbf31257517594906235cc1ceb64f92a.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/72bbb70a8d19e407c424540c26d02eb7cbf31257517594906235cc1ceb64f92a.jpg)

![7c737b255259953d90ea59f8a01cc59d1da44f5e2a9a70fd65af0a73b2833dcf.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/7c737b255259953d90ea59f8a01cc59d1da44f5e2a9a70fd65af0a73b2833dcf.jpg)

![9a8ae5f764ed8c540bab94f017e779d80fa8b6d03e3a71651f6779698201a9ea.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/9a8ae5f764ed8c540bab94f017e779d80fa8b6d03e3a71651f6779698201a9ea.jpg)

![bcb0ebc7d33a69a9dfb2ef0b7d8cbfd90c4800c2a654a9f83943c3710bc1ecca.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/bcb0ebc7d33a69a9dfb2ef0b7d8cbfd90c4800c2a654a9f83943c3710bc1ecca.jpg)

![c8f0e87e35bc9f1fcdb9e016f0aba48ea857022beada06e5bb8668e1895ab125.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/c8f0e87e35bc9f1fcdb9e016f0aba48ea857022beada06e5bb8668e1895ab125.jpg)

![dbf19ae82f8a397582336cc30e84366d164bc868b1d2cb2e63015e05ba4b0784.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/dbf19ae82f8a397582336cc30e84366d164bc868b1d2cb2e63015e05ba4b0784.jpg)

![e9fb906bcb3ec56583e0efca85e57c8a7c7616f4e5dc9d03dd756d8fdda2d9ee.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/images/e9fb906bcb3ec56583e0efca85e57c8a7c7616f4e5dc9d03dd756d8fdda2d9ee.jpg)

### Tables

![499f1a856359e48b644e661835f8c90b85a450b3dfe48bb3b4b9382481ed60e9.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/tables/499f1a856359e48b644e661835f8c90b85a450b3dfe48bb3b4b9382481ed60e9.jpg)

![54fee9af3d0de568618ee6e11380acf21d502f17ff0a6420ed0b17101322dda8.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/tables/54fee9af3d0de568618ee6e11380acf21d502f17ff0a6420ed0b17101322dda8.jpg)

![a6285555fe0550217d069796b2a3e1ceb7379759ff405ebe4f79542616da8f1c.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/tables/a6285555fe0550217d069796b2a3e1ceb7379759ff405ebe4f79542616da8f1c.jpg)

![a98a7b35ca4a67d78d6c74515f433dbb6f3e20d590a0fcbae25de6584076983c.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/tables/a98a7b35ca4a67d78d6c74515f433dbb6f3e20d590a0fcbae25de6584076983c.jpg)

![c279bf8b435967863376c3e8d83e8fbd4b95d69da2490f94a9d3c806c7f25fe6.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/tables/c279bf8b435967863376c3e8d83e8fbd4b95d69da2490f94a9d3c806c7f25fe6.jpg)

![dc96832f236586fff646c4623607ee3195f7156f60c176cd1b2d42bd9d19e872.jpg](../icml_results/2164_MDDM_%20Practical%20Message-Driven%20Generative%20Image%20Steganography%20Based%20on%20Diffusion%20Models/tables/dc96832f236586fff646c4623607ee3195f7156f60c176cd1b2d42bd9d19e872.jpg)

## Pixel-level Certified Explanations via Randomized Smoothing


### Images

![02a59285f3ff4e659d924209dfa08f45efaf757a51dd622e81d09b2a3b66efd3.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/02a59285f3ff4e659d924209dfa08f45efaf757a51dd622e81d09b2a3b66efd3.jpg)

![06a49d22918dcbee7aca170dcd40823d9231ae739b798a7837adbddfa9a0b3d0.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/06a49d22918dcbee7aca170dcd40823d9231ae739b798a7837adbddfa9a0b3d0.jpg)

![171fd42ae6421f953198fff8f26f430fc25c440e17a17a6cb0302bb6a5e4ac1e.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/171fd42ae6421f953198fff8f26f430fc25c440e17a17a6cb0302bb6a5e4ac1e.jpg)

![2b5c883df9378e613ffa4e142d375ffec5ef6481c7eb699c0ecb3679fdf20437.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/2b5c883df9378e613ffa4e142d375ffec5ef6481c7eb699c0ecb3679fdf20437.jpg)

![312710b65ade5ef310a7a5e1eb646cb18051dec58c66a65a89ef636a33dae19b.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/312710b65ade5ef310a7a5e1eb646cb18051dec58c66a65a89ef636a33dae19b.jpg)

![38d7cda9bfbc3c67156f0460447d924c619b43f67c892c796566f7a972344d20.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/38d7cda9bfbc3c67156f0460447d924c619b43f67c892c796566f7a972344d20.jpg)

![5c84d9ab383d345da9614e5e95907209196c5384954079aac4091390852d9185.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/5c84d9ab383d345da9614e5e95907209196c5384954079aac4091390852d9185.jpg)

![63314086487c15a2e2e8e3500c0d5169f47a763eec24fe03547373efdbf0238d.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/63314086487c15a2e2e8e3500c0d5169f47a763eec24fe03547373efdbf0238d.jpg)

![6ac4803283fc6b73a8231275e6bafe454dc8bad8d9cf489b1c1cada6fafd1786.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/6ac4803283fc6b73a8231275e6bafe454dc8bad8d9cf489b1c1cada6fafd1786.jpg)

![709ea358906e04ea32e0c897dd1fb76017a180cbc8c787f421afc3097bee55f8.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/709ea358906e04ea32e0c897dd1fb76017a180cbc8c787f421afc3097bee55f8.jpg)

![7946ca0a353b542304df7ab356e52f38d97b1da468edf35469c1482e6f60b84d.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/7946ca0a353b542304df7ab356e52f38d97b1da468edf35469c1482e6f60b84d.jpg)

![8044263ea3ad3d5a15b6b612a4e7a1dad4fbb0c95d94f595bdb71b0f3fded465.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/8044263ea3ad3d5a15b6b612a4e7a1dad4fbb0c95d94f595bdb71b0f3fded465.jpg)

![875b8ec5cf80162761e8afe474e9fea74f54c04a86e24c9249b35e5ae17daa9a.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/875b8ec5cf80162761e8afe474e9fea74f54c04a86e24c9249b35e5ae17daa9a.jpg)

![943594eee5971d06bace59b0f89ea337d8d64b655fe94ebf67eac50d6b45f628.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/943594eee5971d06bace59b0f89ea337d8d64b655fe94ebf67eac50d6b45f628.jpg)

![9e75b7e039d9d68a41ab71ea4d5e96d65c2a5e98f42b33d3d59f8eebeb5b7200.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/9e75b7e039d9d68a41ab71ea4d5e96d65c2a5e98f42b33d3d59f8eebeb5b7200.jpg)

![acc2b6064661bfb743346af66de54f94c04a4b56b644284b78bbf7ab221d9354.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/acc2b6064661bfb743346af66de54f94c04a4b56b644284b78bbf7ab221d9354.jpg)

![b1e06cdeb7396b99c246a4a5f1598ef5c2db14a11b06694a4ae3b349d51ff518.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/b1e06cdeb7396b99c246a4a5f1598ef5c2db14a11b06694a4ae3b349d51ff518.jpg)

![bdfde290ab6a82a95a026411d87a755bc428d8ca0f5ffd620d0ed4c1437d5a09.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/bdfde290ab6a82a95a026411d87a755bc428d8ca0f5ffd620d0ed4c1437d5a09.jpg)

![befbca2f39165dfc570f150e7380b220c0a75c05b5d7c55f028a234f4f771943.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/befbca2f39165dfc570f150e7380b220c0a75c05b5d7c55f028a234f4f771943.jpg)

![c46875b05c029a45555d4d6f5f16ae4ea64ebf891be0999d0a3b6b2d9311b656.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/c46875b05c029a45555d4d6f5f16ae4ea64ebf891be0999d0a3b6b2d9311b656.jpg)

![ca2bf45cfbe27365fb486da222ad9eb76ca4dfd29014a02ee860f293fbea98b6.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/ca2bf45cfbe27365fb486da222ad9eb76ca4dfd29014a02ee860f293fbea98b6.jpg)

![ef043245974a7cc00a69221a47ac17c88f618f60025d21ae8eb0faec543d4464.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/ef043245974a7cc00a69221a47ac17c88f618f60025d21ae8eb0faec543d4464.jpg)

![f08794dd3905a6bf2fcb8afb9373226d8cb8900c973858804f51658a6a937741.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/f08794dd3905a6bf2fcb8afb9373226d8cb8900c973858804f51658a6a937741.jpg)

![ff9e24f7448801b6162235f37850bb406b807879e9d6fdf7bd3d0173e22221d1.jpg](../icml_results/2165_Pixel-level%20Certified%20Explanations%20via%20Randomized%20Smoothing/images/ff9e24f7448801b6162235f37850bb406b807879e9d6fdf7bd3d0173e22221d1.jpg)

## Agent-as-a-Judge: Evaluate Agents with Agents

### Images

![0fb7b0799c31dd991661beb9435d98851adc8b28c87612ff989febcbc92941ac.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/0fb7b0799c31dd991661beb9435d98851adc8b28c87612ff989febcbc92941ac.jpg)

![15894999681de7102065d72eb5c3236aaf1dc9328922e72b42342b8f2be63ea9.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/15894999681de7102065d72eb5c3236aaf1dc9328922e72b42342b8f2be63ea9.jpg)

![1736f00def2baa9107857d8113526d0a8c5ed8f4c8881ed5e4312114ccf460d9.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/1736f00def2baa9107857d8113526d0a8c5ed8f4c8881ed5e4312114ccf460d9.jpg)

![3aeb737d37bf27102ede5a951ae73c8e16022e3ca25ffe5cffc98d97fb8b7ca3.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/3aeb737d37bf27102ede5a951ae73c8e16022e3ca25ffe5cffc98d97fb8b7ca3.jpg)

![46258a5e6a7b29ad735f495288e549e56a2df6bb5c41ccc5115058a1b8c33d87.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/46258a5e6a7b29ad735f495288e549e56a2df6bb5c41ccc5115058a1b8c33d87.jpg)

![46a268d78369a05f243622b10a0c53dca5d82672c5c56e69126a1e6044022aad.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/46a268d78369a05f243622b10a0c53dca5d82672c5c56e69126a1e6044022aad.jpg)

![507556fb5bfa860f29554b0eb894dd6a2c6a21035654a4c3a9c72df4285230d6.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/507556fb5bfa860f29554b0eb894dd6a2c6a21035654a4c3a9c72df4285230d6.jpg)

![719d34913093e5d1bbe161c16c40709fcc2124b5f947f69c525a9fda49a1b488.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/719d34913093e5d1bbe161c16c40709fcc2124b5f947f69c525a9fda49a1b488.jpg)

![8aa52ea488e16ce6d415fd074397421a0a23f127e4371cf407b5b1fda2322021.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/8aa52ea488e16ce6d415fd074397421a0a23f127e4371cf407b5b1fda2322021.jpg)

![bf9f1cc3059eeb346abe4877e1a84aec6cb2ea423ef9653e0759f09576b10357.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/bf9f1cc3059eeb346abe4877e1a84aec6cb2ea423ef9653e0759f09576b10357.jpg)

![c6080887f5940be4ee3f459ceaff551d6dfee9c29f664d777e81eca13643292e.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/c6080887f5940be4ee3f459ceaff551d6dfee9c29f664d777e81eca13643292e.jpg)

![e2d7e356ba993a42a115f89db3eb9e8eff873270621187efb3f380595387dfed.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/e2d7e356ba993a42a115f89db3eb9e8eff873270621187efb3f380595387dfed.jpg)

![f4838d8fee95af0ea08a108d3e600a2b740787cd5f77d3697053164667161b03.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/images/f4838d8fee95af0ea08a108d3e600a2b740787cd5f77d3697053164667161b03.jpg)

### Tables

![18e41b81c22bd161cdd1280dcd994d36e7e7c4460800c849735334735fe6c839.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/18e41b81c22bd161cdd1280dcd994d36e7e7c4460800c849735334735fe6c839.jpg)

![2bbed15bffd0df456b420f5e387fff825dc67506efef20e6f27d00a7442184a4.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/2bbed15bffd0df456b420f5e387fff825dc67506efef20e6f27d00a7442184a4.jpg)

![36a764727dad63568d05989bec15ade4b653975b19cfd911977514f1526a6d46.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/36a764727dad63568d05989bec15ade4b653975b19cfd911977514f1526a6d46.jpg)

![3e41563a1a15a71a3e37d69ae8a89be0663b291dcd273eed416862a751b3e541.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/3e41563a1a15a71a3e37d69ae8a89be0663b291dcd273eed416862a751b3e541.jpg)

![49d7fed16e96c97292dc044422bfc08c6b466d0a2794988696ac9c55ca4682b4.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/49d7fed16e96c97292dc044422bfc08c6b466d0a2794988696ac9c55ca4682b4.jpg)

![4dffa2646416ac1a86a3f1f8c408b8c5d34a72ce7917d6998a3a10c320c97b09.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/4dffa2646416ac1a86a3f1f8c408b8c5d34a72ce7917d6998a3a10c320c97b09.jpg)

![4ea8a68c89d0260d457efa7ceeb4e0ffa18f5f0575fdc69c6884377939393590.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/4ea8a68c89d0260d457efa7ceeb4e0ffa18f5f0575fdc69c6884377939393590.jpg)

![8969d963d4d38bd9bf4941839fa9de5fa0ff8621cfe2cad39cde8be521373328.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/8969d963d4d38bd9bf4941839fa9de5fa0ff8621cfe2cad39cde8be521373328.jpg)

![b636f0af4d2c048d7e0b9f1c6283ec597a8af8f0c0c2c701f171893bc9368c7c.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/b636f0af4d2c048d7e0b9f1c6283ec597a8af8f0c0c2c701f171893bc9368c7c.jpg)

![eb6f5e38611acac5a35d16b6464286c45e39227f726f5a36cb71b61084b0976b.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/eb6f5e38611acac5a35d16b6464286c45e39227f726f5a36cb71b61084b0976b.jpg)

![f67576f04ae76910a1ac05159f57bac10382ba85d36b5cbc16b99e1a5a1c7352.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/f67576f04ae76910a1ac05159f57bac10382ba85d36b5cbc16b99e1a5a1c7352.jpg)

![f6a480a9e7e5993e67ca789559fcb2c99b3065dd4472c5aae409984a614fd11f.jpg](../icml_results/2166_Agent-as-a-Judge_%20Evaluate%20Agents%20with%20Agents/tables/f6a480a9e7e5993e67ca789559fcb2c99b3065dd4472c5aae409984a614fd11f.jpg)
