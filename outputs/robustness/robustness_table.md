# Robustness Evaluation Benchmark

| Condition | Baseline U-Net Dice | RB-UNet Dice | Delta Dice | Baseline IoU | RB-UNet IoU | Delta IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 0.5749 | 0.7200 | **+0.1451** | 0.4596 | 0.6227 | +0.1631 |
| **Gaussian Noise** | 0.5319 | 0.7219 | **+0.1900** | 0.4204 | 0.6287 | +0.2082 |
| **Gaussian Blur** | 0.5880 | 0.7199 | **+0.1319** | 0.4735 | 0.6216 | +0.1481 |
| **Contrast Shift** | 0.5540 | 0.7100 | **+0.1560** | 0.4302 | 0.6079 | +0.1776 |
| **Brightness Shift** | 0.5370 | 0.6891 | **+0.1522** | 0.4282 | 0.5845 | +0.1563 |
