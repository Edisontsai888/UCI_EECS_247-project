import numpy as np
from math import ceil, log2
import gzip
import time

# COO Format (baseline)

class COOTensor:
    def __init__(self, shape):
        self.shape = shape    
        self.indices = []          
        self.values = []           

    def add_element(self, i, j, k, value):
        self.indices.append((i, j, k))
        self.values.append(value)

    def num_nonzeros(self):
        return len(self.values)

    def storage_bits(self, word_size=64):
        N = 3
        return self.num_nonzeros() * N * word_size

    def __repr__(self):
        lines = [f"COO Tensor {self.shape}, {self.num_nonzeros()} nonzeros:"]
        for idx, val in zip(self.indices, self.values):
            lines.append(f"  x[{idx[0]},{idx[1]},{idx[2]}] = {val}")
        return "\n".join(lines)


# ALTO Format

def compute_bits_needed(dim_size):
    if dim_size <= 1:
        return 0
    return ceil(log2(dim_size))


def compute_alto_masks(shape):
    N = len(shape)
    bits_per_mode = [compute_bits_needed(s) for s in shape]
    total_bits = sum(bits_per_mode)
    
    masks = [0] * N
    mode_order = sorted(range(N), key=lambda m: shape[m])
    
    bit_pos = 0
    remaining_bits = bits_per_mode[:]

    while any(r > 0 for r in remaining_bits):
        for m in mode_order:
            if remaining_bits[m] > 0:
                masks[m] |= (1 << bit_pos) 
                remaining_bits[m] -= 1
                bit_pos += 1
    
    return masks, total_bits


def alto_encode(indices, masks, total_bits):
    alto_index = 0
    for mode_idx, (idx, mask) in enumerate(zip(indices, masks)):
        idx_bit = 0
        temp_mask = mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            lsb_pos = lsb.bit_length() - 1
            if (idx >> idx_bit) & 1:
                alto_index |= lsb
            temp_mask &= temp_mask - 1    
            idx_bit += 1
    return alto_index


def alto_decode(alto_index, masks):
    indices = []
    for mask in masks:
        idx = 0
        idx_bit = 0
        temp_mask = mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            if alto_index & lsb:
                idx |= (1 << idx_bit)
            temp_mask &= temp_mask - 1
            idx_bit += 1
        indices.append(idx)
    return tuple(indices)


class ALTOTensor:
    def __init__(self, shape):
        self.shape = shape
        self.masks, self.total_bits = compute_alto_masks(shape)
        self.elements = []  
    
    def add_element(self, i, j, k, value):
        alto_index = alto_encode((i, j, k), self.masks, self.total_bits)
        self.elements.append((alto_index, value))
    
    def sort(self):
        self.elements.sort(key=lambda x: x[0])
    
    def num_nonzeros(self):
        return len(self.elements)
    
    def storage_bits(self):
        return self.num_nonzeros() * self.total_bits
    
    def compression_ratio_vs_coo(self, word_size=64):
        N = len(self.shape)
        coo_bits_per_element = N * word_size
        return coo_bits_per_element / self.total_bits
    
    def __repr__(self):
        lines = [f"ALTO Tensor {self.shape}"]
        lines.append(f"  Bits per mode: {[compute_bits_needed(s) for s in self.shape]}")
        lines.append(f"  Masks (binary): {[bin(m) for m in self.masks]}")
        lines.append(f"  Total bits per element: {self.total_bits}")
        lines.append(f"  {self.num_nonzeros()} nonzeros:")
        for alto_idx, val in self.elements:
            decoded = alto_decode(alto_idx, self.masks)
            lines.append(f"  ALTO index={alto_idx:0{self.total_bits}b} "
                         f"({alto_idx}) → x[{decoded[0]},{decoded[1]},{decoded[2]}] = {val}")
        return "\n".join(lines)


# Correctness verification
def verify_encoding(coo_tensor, alto_tensor):
    print("Correctness Verification")
    all_correct = True
    
    for (i, j, k), val in zip(coo_tensor.indices, coo_tensor.values):
        alto_idx = alto_encode((i, j, k), alto_tensor.masks, alto_tensor.total_bits)
        decoded = alto_decode(alto_idx, alto_tensor.masks)
        
        if decoded != (i, j, k):
            print(f"  FAIL: ({i},{j},{k}) → {alto_idx} → {decoded}")
            all_correct = False
        else:
            print(f"  OK:   ({i},{j},{k}) → ALTO index {alto_idx:0{alto_tensor.total_bits}b} → {decoded}")
    
    print(f"\n  Result: {'CORRECT' if all_correct else 'ERRORS'}")
    return all_correct


# Sequential MTTKRP
def mttkrp_coo(coo_tensor, B, C, R):
    I = coo_tensor.shape[0]
    A = np.zeros((I, R))
    for (i, j, k), val in zip(coo_tensor.indices, coo_tensor.values):
        A[i] += val * B[j] * C[k]   # vectorized over R
    return A


def mttkrp_alto(alto_tensor, B, C, R):
    I = alto_tensor.shape[0]
    A = np.zeros((I, R))
    for alto_idx, val in alto_tensor.elements:
        i, j, k = alto_decode(alto_idx, alto_tensor.masks)
        A[i] += val * B[j] * C[k]   # vectorized over R
    return A


# Load DARPA .tns.gz file
def load_tns_gz(filepath, max_elements=None):
    indices = []
    values = []
    
    with gzip.open(filepath, 'rt', encoding='latin-1') as f:
        for line_num, line in enumerate(f):
            if max_elements and len(indices) >= max_elements:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                i, j, k = int(parts[0])-1, int(parts[1])-1, int(parts[2])-1
                val = float(parts[3])
            except ValueError:
                continue
            indices.append((i, j, k))
            values.append(val)
    
    max_i = max(idx[0] for idx in indices) + 1
    max_j = max(idx[1] for idx in indices) + 1
    max_k = max(idx[2] for idx in indices) + 1
    shape = (max_i, max_j, max_k)
    
    print(f"  Loaded {len(values)} nonzeros, shape = {shape}")
    return indices, values, shape

if __name__ == "__main__":

    # Small example from paper
    print("Small Example ")
    shape = (4, 8, 2)
    nonzeros = [
        (1, 0, 0, 1.0),
        (3, 1, 1, 2.0),
        (0, 3, 0, 3.0),
        (2, 2, 1, 4.0),
        (3, 4, 0, 5.0),
        (1, 6, 1, 6.0),
    ]
    
    coo = COOTensor(shape)
    for i, j, k, v in nonzeros:
        coo.add_element(i, j, k, v)
    
    alto = ALTOTensor(shape)
    for i, j, k, v in nonzeros:
        alto.add_element(i, j, k, v)
    alto.sort()
    
    print(alto)
    print()
    verify_encoding(coo, alto)
    print()

    # MTTKRP correctness on small example
    R = 4
    I, J, K = shape
    np.random.seed(42)
    B = np.random.rand(J, R)
    C = np.random.rand(K, R)
    
    A_coo  = mttkrp_coo(coo, B, C, R)
    A_alto = mttkrp_alto(alto, B, C, R)
    
    if np.allclose(A_coo, A_alto):
        print("MTTKRP correctness: PASS (COO == ALTO)")
    else:
        print("MTTKRP correctness: FAIL")
        print("Max difference:", np.max(np.abs(A_coo - A_alto)))
    print()

    # DARPA Dataset 
    print("Part 2: DARPA Dataset")

    # Load only first 100,000 elements to keep runtime reasonable and remove max_elements to load the full dataset 
    MAX_ELEMENTS = 100000

    darpa_indices, darpa_values, darpa_shape = load_tns_gz(
        "1998darpa.tns.gz", max_elements=MAX_ELEMENTS
    )

    # Build COO
    darpa_coo = COOTensor(darpa_shape)
    for (i, j, k), v in zip(darpa_indices, darpa_values):
        darpa_coo.add_element(i, j, k, v)

    # Build ALTO
    print("Building ALTO tensor...")
    darpa_alto = ALTOTensor(darpa_shape)
    for (i, j, k), v in zip(darpa_indices, darpa_values):
        darpa_alto.add_element(i, j, k, v)
    darpa_alto.sort()

    # Storage comparison
    print()
    print("Storage Comparison (DARPA):")
    coo_bits  = darpa_coo.storage_bits()
    alto_bits = darpa_alto.storage_bits()
    print(f"  COO  index metadata: {coo_bits:,} bits")
    print(f"  ALTO index metadata: {alto_bits:,} bits")
    print(f"  Compression ratio:   {darpa_alto.compression_ratio_vs_coo():.2f}x")
    print()

    # MTTKRP timing comparison
    R = 16
    I, J, K = darpa_shape
    np.random.seed(0)
    B = np.random.rand(J, R)
    C = np.random.rand(K, R)

    print(f"MTTKRP Timing (R={R}, {MAX_ELEMENTS:,} nonzeros):")

    t0 = time.time()
    A_coo = mttkrp_coo(darpa_coo, B, C, R)
    t_coo = time.time() - t0
    print(f"  COO  time: {t_coo:.4f} seconds")

    t0 = time.time()
    A_alto = mttkrp_alto(darpa_alto, B, C, R)
    t_alto = time.time() - t0
    print(f"  ALTO time: {t_alto:.4f} seconds")

    if np.allclose(A_coo, A_alto):
        print("  MTTKRP results match (COO == ALTO) ✓")
    else:
        print("  MISMATCH in MTTKRP results!")

    print(f"\nDone.")