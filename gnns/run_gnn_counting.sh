#!/bin/bash

# Mutagenicity
for pattern_idx in 4-24 4-30 4-15 4-26 5-8 5-74 5-38 5-43 5-6 5-17 6-0 7-0 7-2; do
    for seed in {0..4}; do
        echo "Running for dataset: Mutagenicity, pattern-idx: $pattern_idx, seed: $seed"
        python gnn_counting.py --dataset Mutagenicity --pattern-idx $pattern_idx --seed $seed
    done
done

# MCF-7
for pattern_idx in 4-0 4-24 4-7 4-22 5-41 5-1 5-10 5-38 5-23 5-26 6-0 7-0 7-2; do
    for seed in {0..4}; do
        echo "Running for dataset: MCF-7, pattern-idx: $pattern_idx, seed: $seed"
        python gnn_counting.py --dataset MCF-7 --pattern-idx $pattern_idx --seed $seed
    done
done

# ZINC dataset
for pattern_idx in 4-21 4-29 4-11 4-22 5-1 5-27 5-17 5-19 5-2 5-32 6-0 7-0 7-2; do
    for seed in {0..4}; do
        echo "Running for dataset: ZINC, pattern-idx: $pattern_idx, seed: $seed"
        python gnn_counting.py --dataset ZINC --pattern-idx $pattern_idx --seed $seed
    done
done

# Peptides-func dataset
for pattern_idx in 4-56 4-67 4-45 4-18 5-7 5-64 5-146 5-115 5-122 5-119 6-0 7-9 7-14; do
    for seed in {0..4}; do
        echo "Running for dataset: Peptides-func, pattern-idx: $pattern_idx, seed: $seed"
        python gnn_counting.py --dataset Peptides-func --pattern-idx $pattern_idx --seed $seed
    done
done

# ogbg-molhiv dataset
for pattern_idx in 4-0 4-9 4-6 4-3 5-4 5-2 5-7 5-6 5-3 5-5 6-0 7-0 7-2; do
    for seed in {0..4}; do
        echo "Running for dataset: ogbg-molhiv, pattern-idx: $pattern_idx, seed: $seed"
        python gnn_counting.py --dataset ogbg-molhiv --pattern-idx $pattern_idx --seed $seed
    done
done

# ogbg-molpcba dataset
for pattern_idx in 4-2 4-21 4-17 4-6 5-11 5-3 5-15 5-14 5-10 5-12 6-0 7-0 7-3; do
    for seed in {0..4}; do
        echo "Running for dataset: ogbg-molpcba, pattern-idx: $pattern_idx, seed: $seed"
        python gnn_counting.py --dataset ogbg-molpcba --pattern-idx $pattern_idx --seed $seed
    done
done

# PCQM-Contact dataset
for pattern_idx in 4-8 4-4 4-41 4-34 5-11 5-41 5-39 5-81 5-44 5-55 6-0 7-1 7-0; do
    for seed in {0..4}; do
        echo "Running for dataset: PCQM-Contact, pattern-idx: $pattern_idx, seed: $seed"
        python gnn_counting.py --dataset PCQM-Contact --pattern-idx $pattern_idx --seed $seed
    done
done
