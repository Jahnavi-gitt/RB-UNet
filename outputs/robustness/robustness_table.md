# Robustness Evaluation Benchmark

| Condition | Baseline U-Net Dice | RB-UNet Dice | Delta Dice | Baseline IoU | RB-UNet IoU | Delta IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 0.2003 | 0.9109 | **+0.7105** | 0.1159 | 0.8386 | +0.7227 |
| **Gaussian Noise** | 0.2909 | 0.8895 | **+0.5986** | 0.1828 | 0.8064 | +0.6236 |
| **Gaussian Blur** | 0.1214 | 0.8658 | **+0.7444** | 0.0673 | 0.7662 | +0.6989 |
| **Contrast Shift** | 0.0000 | 0.0000 | **+0.0000** | 0.0000 | 0.0000 | +0.0000 |
| **Brightness Shift** | 0.7865 | 0.9571 | **+0.1705** | 0.6778 | 0.9178 | +0.2400 |
