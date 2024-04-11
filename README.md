# StABlE-Training
Official repository for the paper [Stability-Aware Training of Neural Network Interatomic Potentials with Differentiable Boltzmann Estimators](https://arxiv.org/abs/2402.13984). Code will be added shortly. Stay tuned for updates.

## Authors
- Sanjeev Raja
- Ishan Amin
- Fabian Pedregosa
- Aditi Krishnapriyan

## Abstract
Neural network interatomic potentials (NNIPs) are an attractive alternative to ab-initio methods for molecular dynamics (MD) simulations. However, they can produce unstable simulations which sample unphysical states, limiting their usefulness for modeling phenomena occurring over longer timescales. To address these challenges, we present Stability-Aware Boltzmann Estimator (StABlE) Training, a multi-modal training procedure which combines conventional supervised training from quantum-mechanical energies and forces with reference system observables, to produce stable and accurate NNIPs. StABlE Training iteratively runs MD simulations to seek out unstable regions, and corrects the instabilities via supervision with a reference observable. The training procedure is enabled by the Boltzmann Estimator, which allows efficient computation of gradients required to train neural networks to system observables, and can detect both global and local instabilities. We demonstrate our methodology across organic molecules, tetrapeptides, and condensed phase systems, along with using three modern NNIP architectures. In all three cases, StABlE-trained models achieve significant improvements in simulation stability and recovery of structural and dynamic observables. In some cases, StABlE-trained models outperform conventional models trained on datasets 50 times larger. As a general framework applicable across NNIP architectures and systems, StABlE Training is a powerful tool for training stable and accurate NNIPs, particularly in the absence of large reference datasets.

## Citation

```bibtex
@misc{raja2024stabilityaware,
  title={Stability-Aware Training of Neural Network Interatomic Potentials with Differentiable Boltzmann Estimators}, 
  author={Sanjeev Raja and Ishan Amin and Fabian Pedregosa and Aditi S. Krishnapriyan},
  year={2024},
  eprint={2402.13984},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
