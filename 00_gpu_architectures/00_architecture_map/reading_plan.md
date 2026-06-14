# Reading Plan

How I should actually read this folder.

## Pass 1: names and targets

- [ ] read `../09_compute_capability_ptx/README.md`
- [ ] read `../09_compute_capability_ptx/target_strings.md`
- [ ] memorize only the useful current targets:
  - `sm_80`: A100
  - `sm_86`: RTX 30 / A10 / A40
  - `sm_89`: Ada
  - `sm_90`: Hopper
  - `sm_100`: Blackwell datacenter
  - `sm_103`: Blackwell Ultra
  - `sm_120`: RTX Blackwell

## Pass 2: modern architectures first

Read in this order:

1. Ampere
2. Hopper
3. Blackwell
4. Rubin
5. Ada

Reason: this repo is for learning CUDA now, so older history should not block the modern mental model.

## Pass 3: historical context

Then read:

1. Volta/Turing
2. Maxwell/Pascal
3. Tesla/Fermi/Kepler

Reason: older architectures explain why CUDA looks the way it does, but I don't need to over-optimize for them.

## Pass 4: prove things

For each architecture claim, find one of:

- [ ] NVIDIA doc line
- [ ] generated PTX
- [ ] SASS dump
- [ ] Nsight Compute metric
- [ ] CUTLASS kernel using that feature

No proof, no strong belief.

