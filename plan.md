# Checklist: Building nanoSubQ AI

## 🟦 Phase 1: Python & PyTorch Basics
* [ ] **Learn Python Class Basics:** Understand how to build a basic code blueprint (`class`) and use inheritance (`torch.nn.Module`)[cite: 4].
* [ ] **Learn Tensor Basics:** Understand what tensor shapes represent (like `[Batch, Sequence, Features]`)[cite: 4].
* [ ] **Write Multiplications from Scratch:** Write a basic matrix multiplication script using raw Python loops (no libraries allowed)[cite: 4].
* [ ] **Use PyTorch Multiplications:** Reimplement your multiplication script using PyTorch's native `@` operator[cite: 4].
* [ ] **Build a Custom Layer:** Write a custom linear layer (`Y = XW + b`) and run `.backward()` to inspect the tracking gradients[cite: 4].

## 🟩 Phase 2: Understanding Attention & The Bottleneck
* [ ] **Study Attention Projections:** Learn how inputs map into Query ($Q$), Key ($K$), and Value ($V$) vectors[cite: 4].
* [ ] **Build Masked Attention from Scratch:** Create a causal attention function using raw PyTorch operators, applying your own lower-triangular causal mask (`torch.tril`)[cite: 4].
* [ ] **Profile the Math:** Run an empirical timing script across different sequence lengths ($512$, $1024$, $2048$)[cite: 4].
* [ ] **Map the Bottleneck:** Document your profiling results in a table to visually confirm how memory and speed scale quadratically ($O(N^2)$)[cite: 4].

## 🟨 Phase 3: Sparse Attention & The Straight-Through Estimator (STE)
* [ ] **Build a Causal Top-k Selector:** Write a function that zeroes out all historical keys except the top-$k$ highest-scoring ones, while keeping future positions masked[cite: 4].
* [ ] **Write the Custom Autograd Layer:** Build `class STETopK(torch.autograd.Function)`[cite: 4]. 
    * Forward pass: Apply the hard binary mask[cite: 4].
    * Backward pass: Pass the gradients through completely unchanged[cite: 4].
* [ ] **Run a Gradient Unit Test:** Hook your `STETopK` layer up to a mock routing layer and run a test to confirm the parameter updates successfully bypass the discrete selection step[cite: 4].

## 🟧 Phase 4: Gating Networks & Dynamic Masking
* [ ] **Build the Gating Network Module:** Write an `SSARouter` class that projects queries and keys into a smaller routing space ($d_g \ll d_k$) to calculate normalized routing scores[cite: 4].
* [ ] **Code the Combined Mask Generator:** Write a function that merges causal limits, a fixed local sliding window (always active), and dynamic top-$k$ selections into one final binary mask[cite: 4].
* [ ] **Run a Causal Leakage Test:** Confirm via unit tests that tokens at index $t$ cannot affect gradients or activate tokens anywhere in the past or future[cite: 4].

## 🟥 Phase 5: Hacking and Training nanoGPT
* [ ] **Refactor `model.py`:** Open Andrej Karpathy's `nanoGPT` engine and replace the standard `CausalSelfAttention` module with your newly built sparse module[cite: 4].
* [ ] **Configure Safeguards:** Set up your training loop parameters to handle the STE adjustments[cite: 4]:
    * Set a low learning rate (`3e-4`)[cite: 4].
    * Turn off bias vectors in your routing layers to keep projections centered[cite: 4].
    * Enable gradient clipping (`grad_clip_norm = 1.0`)[cite: 4].
    * Turn on active weight decay (`1e-1`) to keep weights from saturating[cite: 4].
* [ ] **Prepare the Data:** Run the Shakespeare data prep script inside your cloned directory[cite: 4].
* [ ] **Train and Verify:** Initialize training on the Shakespeare character dataset, monitor the loss convergence, and run your profiling script to confirm that memory now scales linearly ($O(N \cdot k)$)[cite: 4].