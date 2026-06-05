#EECS 247 Project — Implementing ALTO
Python implementation of the ALTO (Adaptive Linearized Storage of Sparse Tensors) format, based on Helal et al., ICS '21.
##What this implements

-ALTO encoding and decoding
-COO baseline format
Sequential MTTKRP for both COO and ALTO
Storage comparison and timing evaluation on the DARPA dataset

Requirements

Python 3
NumPy

Dataset
Download 1998darpa.tns.gz from the FROSTT repository:
http://frostt.io/tensors/darpa/
Place it in the same folder as the .py file before running.
How to run
python3 "EECS247 project2.py"
Reference
Helal et al., "ALTO: Adaptive Linearized Storage of Sparse Tensors," ICS '21.
https://doi.org/10.1145/3447818.3461703
