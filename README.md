# Unsupervised 2D Deformable Image Registration & Tissue Growth Analytics Engine

A self-supervised deep learning pipeline built in PyTorch to register structural deformations between multi-temporal images without ground-truth alignment labels. This project serves as a validated 2D Proof-of-Concept (PoC) demonstrating spatial transformation physics, joint loss optimization, and local deformation mechanics, designed to directly scale into 3D/4D longitudinal medical intelligence frameworks.

## 🚀 Live Demo
👉 **[Interact with the Live Web Dashboard on Hugging Face Spaces](YOUR_HUGGINGFACE_SPACE_URL_HERE)**
*(Note: Upload your own slices or click "Load Demo Data" to instantly compute alignments and visualize tissue expansion maps in real-time.)*

---

## 🔬 Core System Architecture

Instead of relying on human-annotated labels, this network frames registration as an unsupervised optimization problem. Given a **Moving Image ($I_m$)** and a **Fixed Image ($I_f$)**, the framework computes a continuous displacement field to morph the moving anatomy into the target state.



### The Spatial Transformer Pipeline:
1. **Feature Extraction & Dense Flow Prediction:** The system concatenates $I_m$ and $I_f$ along the channel axis and passes them through a 2D U-Net backbone modified to output a 2-channel dense **Deformation Vector Field (DVF)** representing $\Delta x$ and $\Delta y$ pixel displacements.
2. **Differentiable Warping Engine:** The predicted DVF is fed into a Spatial Transformer layer utilizing a differentiable grid sampler (`torch.nn.functional.grid_sample`). This warps the original Moving Image using bilinear interpolation to generate the aligned **Warped Image ($I_w$)**.
3. **Self-Supervised Constraint Mapping:** Because no ground-truth annotations exist, the network updates its parameters backpropagating through a composite loss function:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{similarity}}(I_w, I_f) + \lambda \mathcal{L}_{\text{smoothness}}(\nabla \text{DVF})$$

---

## 📊 Longitudinal Growth Analytics

To derive clinical utility from the deformation mechanics, the pipeline features a downstream **Growth Analytics Engine** that calculates the spatial derivative of the predicted displacement fields using the **Jacobian Determinant**:

$$J(x, y) = \det \begin{bmatrix} 1 + \frac{\partial \Delta x}{\partial x} & \frac{\partial \Delta x}{\partial y} \\ \frac{\partial \Delta y}{\partial x} & 1 + \frac{\partial \Delta y}{\partial y} \end{bmatrix}$$

* **$J(x, y) > 1$:** Indicates localized tissue **expansion** (e.g., tumor growth, swelling, inflation).
* **$J(x, y) < 1$:** Indicates localized tissue **contraction** (e.g., muscle atrophy, deflation, degeneration).

These scalar attributes are mapped to a divergent color spectrum, creating a high-fidelity visual heatmap overlaying the anatomical boundaries to automatically track voxel-level volumetric developments across time.

---

## 🛠️ Dimension-Agnostic Scaling Map (2D to 4D)

This prototype was developed to explicitly lay down the mathematical foundations required for 4D longitudinal processing. The underlying mechanics translate directly across spatial dimensions:

| Component | 2D PoC Implementation | 4D Scale-Up Implementation |
| :--- | :--- | :--- |
| **Feature Extraction** | 2D Convolutions (`Conv2d`) | 3D Convolutions (`Conv3d`) / 3D Vision Transformers |
| **Target Tensor Channels** | 2-Channel Grid ($\Delta x, \Delta y$) | 3-Channel Grid ($\Delta x, \Delta y, \Delta z$) over $T$ frames |
| **Grid Differentiability** | Bilinear Interpolation | Trilinear Interpolation |
| **Analytics Mapping** | $2 \times 2$ Jacobian Matrix | $3 \times 3$ Jacobian Spatial Matrix |

---

## 📂 Repository Structure

```text
├── notebooks/          # Google Colab notebooks outlining step-by-step learning milestones
├── src/                # Modular, production-ready source scripts
│   ├── model.py        # Custom U-Net + Spatial Transformer Network modules
│   ├── losses.py       # Multi-Scale MSE and Diffusion Regularization implementations
│   └── train.py        # Unsupervised training loop and gradient configurations
├── app.py              # Gradio web frontend pipeline code
├── requirements.txt    # System dependency configuration file
└── README.md           # Project architecture documentation
