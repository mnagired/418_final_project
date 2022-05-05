# 15-418 Final Project
**Manish Nagireddy (mnagired) and Ziad Khattab (zkhattab)**

Please take a look at our [Final Report](report.pdf) and our [Video Summary](https://youtu.be/Nv8kCS1lNFs)!

# Title

An Exploration of Parallelism in Neural Networks

# Summary

For our 15-418 final project, we are looking into potential axes of parallelism that exist within neural networks. We will be implementing neural
networks in `Python` (via `PyTorch` and `mpi4py`, an `MPI` package for `Python`) as well as potentially also via `MPI` in `C++` and measuring their
performance on CPUs as well as GPUs.

# Installation / Usage
You will need the following installed (preferably in a virtual environment using Python 3.6):
```bash
numpy==1.19.0
torch==1.10.2
torchvision==0.11.3
mpi4py
```

Additionally, you will need `OpenMP` and `OpenMPI` installed.

To run the `Python` code, simply navigate to the code directory and do:

```bash

python baseline.py --lr learning_rate --nepochs num_epochs --batch_size batch_size
python cnn.py --lr learning_rate --nepochs num_epochs --batch_size batch_size
mpirun -np num_procs python cnn_mpi.py --lr learning_rate --nepochs num_epochs --batch_size batch_size
mpirun -np num_procs python cnn_resnet_mpi.py --lr learning_rate --nepochs num_epochs --batch_size batch_size

```

For the `C++` code:
```cpp

make train && out/train
make test && out/test

```
