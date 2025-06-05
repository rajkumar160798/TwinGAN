# TwinGAN: Generative Digital Twin for Rare Failure Simulation

## 🚀 Project Goal
Simulate rare failure sequences using GANs to augment predictive maintenance datasets and improve model robustness.

## 📦 Structure
```
TwinGAN/
├── data/           # Raw and preprocessed datasets (e.g., CMAPSS, AI4I)
├── gan_model/      # GAN model code (TimeGAN, RNN-GAN)
├── classifier/     # Predictive models trained on real + synthetic data
├── notebooks/      # Jupyter notebooks for experiments & validation
├── plots/          # All generated plots and visualizations
├── utils/          # Normalization, preprocessing, etc.
├── reports/        # Result summary PDFs, logs, etc.
```

## 🛠 Tech Stack
- PyTorch (GANs)
- NumPy, pandas, matplotlib
- t-SNE / PCA for visualization
- CMAPSS or AI4I datasets

## 💡 Use Case
Augment rare failure examples to train better predictive maintenance models in industrial settings.

## ⚙️ Setup
```bash
pip install -r requirements.txt
```
