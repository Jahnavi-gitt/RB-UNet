# RB-UNet Methodology & Mathematical Formulation

## 1. Motivation & Problem Formulation
Lesion boundary segmentation in dermoscopic images is complicated by irregular geometries, fuzzy transitions between lesion and healthy tissue, and acquisition degradations (blur, low illumination, sensor noise). Standard U-Net architectures rely on pixel-level Cross-Entropy and soft Dice losses, which treat interior and boundary pixels identically, leading to over-smoothed lesion borders and sharp performance drops under clinical image corruption.

**RB-UNet (Robust Boundary-Consistent U-Net)** introduces two lightweight mechanisms:
1. **Auxiliary Boundary Supervision**: Guides the network to explicitly detect high-frequency lesion transitions.
2. **Corruption-Consistency Regularization**: Forces feature representations to be invariant to input degradations.

---

## 2. Model Architecture
RB-UNet maintains the classical 4-level U-Net encoder-decoder backbone:
- **Encoder**: 4 stages of `DoubleConv(Conv2d -> BatchNorm2d -> ReLU)` separated by $2 \times 2$ `MaxPool2d`.
- **Bottleneck**: 1024-channel dual convolution.
- **Decoder**: 4 stages of `ConvTranspose2d` with skip-connection feature concatenation.
- **Dual Heads**:
  - **Segmentation Head**: $1 \times 1$ convolution mapping 64 decoder features to binary lesion logits $\hat{P}_{\text{seg}} \in \mathbb{R}^{B \times 1 \times H \times W}$.
  - **Auxiliary Boundary Head**: Lightweight sequential block `Conv2d(64, 32, 3)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `Conv2d(32, 1, 1)` producing boundary logits $\hat{B}_{\text{pred}} \in \mathbb{R}^{B \times 1 \times H \times W}$.

---

## 3. Mathematical Objectives

### A. Morphological Boundary Extraction
Given ground-truth binary mask $M \in \{0, 1\}^{H \times W}$, the boundary target $B$ is generated using morphological operations:
$$D = \text{Dilation}(M, k), \quad E = \text{Erosion}(M, k)$$
$$B = D - E$$
where $k=3$ is a structuring element. $B$ captures the 1-2 pixel transition zone around the lesion contour.

### B. Loss Formulation
1. **Primary Segmentation Loss**:
   $$\mathcal{L}_{\text{seg}} = \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}(\hat{P}_{\text{clean}}, M) + \lambda_{\text{bce}} \mathcal{L}_{\text{bce}}(\hat{P}_{\text{clean}}, M)$$
2. **Auxiliary Boundary Loss**:
   $$\mathcal{L}_{\text{boundary}} = \mathcal{L}_{\text{bce-weighted}}(\hat{B}_{\text{pred}}, B) + \mathcal{L}_{\text{dice}}(\hat{B}_{\text{pred}}, B)$$
   with positive class weighting to handle boundary pixel sparsity.
3. **Corruption Consistency Loss**:
   For a batch of clean images $X$, corrupted variants $X_{\text{corrupt}} = \mathcal{T}_{\text{corrupt}}(X)$ are generated via random noise, blur, contrast, or brightness shifts. Passing both through the shared-weight model yields:
   $$\mathcal{L}_{\text{consistency}} = \frac{1}{HW} \sum_{i,j} \left( \sigma(\hat{P}_{\text{clean}}^{(i,j)}) - \sigma(\hat{P}_{\text{corrupt}}^{(i,j)}) \right)^2$$
4. **Total Loss**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{\text{boundary}} \mathcal{L}_{\text{boundary}} + \lambda_{\text{consistency}} \mathcal{L}_{\text{consistency}}$$
