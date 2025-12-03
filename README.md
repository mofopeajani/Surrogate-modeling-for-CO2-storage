# Accelerating CO₂ Storage Optimization with Deep Learning–Based Surrogates

**Authors:** Mofopefoluwa Ajani — Energy Science & Engineering, Stanford University; Victor Awosiji — Earth & Planetary Sciences, Stanford University  

## 1. Background and Problem Statement  

We propose building a deep learning surrogate model for CO₂ storage simulations, capable of predicting reservoir pressure evolution, CO₂ saturation, and optimal well placement and control strategies. The model will learn to approximate a high-fidelity multiphase flow simulator (GEOS) by mapping well configurations and time-dependent injection schedules to full spatiotemporal fields of pressure and CO₂ saturation.  

Accurate prediction of these quantities is essential because pressure buildup can induce CO₂ leakage risks, including groundwater contamination or surface seepage. High-fidelity reservoir simulations are computationally expensive, limiting real-time decision making and uncertainty quantification.  

A successful surrogate model would emulate the full-physics simulator at a fraction of the computational cost, enabling rapid scenario evaluation, real-time optimization, and improved safety of large-scale CO₂ storage operations.  

## 2. Challenges  

The primary challenge lies in the high-dimensional search space associated with varying well placements and injection controls. For nₚ control periods and n_w wells, the optimization domain is ℝ^{α n_w + n_w n_c}, capturing nonlinear interactions in both space and time.  

Learning an accurate surrogate within this space is difficult because:  

- The dimensionality introduces steep computational demands.  
- Only a limited number of expensive GEOS simulations (500–1000) are available.  
- The surrogate must remain accurate while generalizing across unseen well placements and injection policies.  

Thus, the architecture must be expressive enough to capture nonlinear flow physics yet computationally efficient enough to train with limited data.  

## 3. Dataset  

Our dataset will consist of **1,000 GEOS simulations** generated on Stanford’s SHERLOCK cluster. We assume a fixed geological realization (porosity, permeability, anisotropy), while well placements and injection schedules vary.  

**Simulation design:**  

- A synthetic model with **three vertical injection wells**.  
- **Three control periods**, with total injection for each period partitioned among the wells.  
- Ensembles:  
  - 250 simulations: fixed well placements, varying controls  
  - 250 simulations: fixed controls, varying well placements  
  - 500 simulations: both varied  

Outputs — stored as NumPy arrays — include 3D pressure and gas saturation fields at discrete time steps, with dimensions **[Nt, Nx, Ny, Nz]**. An 80/20 train–test split will be used.  

Automated generation, scheduling, and file management will be implemented for efficient HPC execution.  

## 4. Learning Methods  

We will evaluate two surrogate architectures specifically designed for high-dimensional spatiotemporal prediction:  

### (a) 3D Residual U-Net  

A convolutional architecture capable of capturing multi-scale spatial patterns through skip-connections and residual blocks.  

### (b) 3D Patch-Based Transformer

A Vision-Transformer–style encoder architecture that operates on 3D patches of the reservoir state. The input volume is first decomposed into non-overlapping 3D patches and linearly embedded into a sequence of tokens. These tokens are processed by a stack of multi-head self-attention and feed-forward layers, enabling the model to capture long-range spatial dependencies and interactions between wells and the surrounding geomodel. A final projection layer “unpatchifies” the token representations back to a full 3D grid of pressure and CO₂ saturation predictions. 

**Inputs** to both models include:  

- Static geomodel (porosity/permeability fields)  
- Spatial coordinates  
- Per-well masks  
- Injection-rate channels  
- Current pressure & saturation fields  

**Outputs:** Predicted next-step pressure and CO₂ saturation fields.  

Both models are trained to minimize errors relative to GEOS output, enabling accurate CO₂ plume forecasting under variable controls.  

## 5. Evaluation  

Model accuracy will be assessed using:  

- **RMSE**, **MAE**, **R²** for pressure and CO₂ saturation  
- **SSIM** for structural similarity  
- Temporal error-accumulation plots  
- Loss curves  
- Qualitative analysis via 2D/3D slices comparing GEOS vs. surrogate predictions  

An example surrogate vs. simulation comparison is shown in Fig. 1 of the proposal.
