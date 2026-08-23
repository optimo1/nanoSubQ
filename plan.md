# 🎯 The NanoSubQ "Learn This, Learn That" Checklist

## 🟦 Phase 1: Learn Basic Python & PyTorch Variables
- [done] **Learn This:** How to store different types of data in Python (Strings, Integers, Floats).
- [done] **Learn That:** How to group data together using Python Lists and Dictionaries.
- [done] **Learn This:** How to use a `for` loop to make Python repeat an action over a list of items.
- [done] **Learn That:** How to bundle your code into a reusable block using functions (`def`).
- [done] **Learn This:** Object-Oriented Programming (OOP)—specifically how to create a `class` and understand how "inheritance" works (`class MyLayer(nn.Module)`)[cite: 4].
- [done] **Learn This:** Arrays, 2D arrays, and matrices [cite: 4].
- [done] **Learn That:** Tensor shapes—how data grids are structured by `[Batch, Sequence, Features]`[cite: 4].
- [done] **Learn This:** Matrix multiplication syntax using PyTorch's native `@` operator[cite: 4].
- [done] **Learn That:** PyTorch Autograd—how calling `.backward()` automatically tracks math history and calculates gradients[cite: 4].

---

## 🟩 Phase 2: Learn How Attention Works & Why It Breaks
- [done] **Learn This:** Query ($Q$), Key ($K$), and Value ($V$) linear projections—what they mean conceptually in a Transformer.
- [done] **Learn That:** Dot-product scaling—why dividing your scores by the square root of the head dimension keeps math stable.
- [done] **Learn This:** The Softmax Filter—how scores are turned into actual percentage weights so the model knows exactly how much percentage "focus" to place on each word.
- [done] **Learn That:** The Value Aggregation Step—how the model uses those percentage weights to blend the Value ($V$) vectors together to create the final context-aware output.
- [done] **Learn This:** Causal masking—how to use `torch.triu` and `torch.masked_fill` to block a model from looking at future words.
- [done] **Learn That:** Multi-Head Attention splitting—how a single large tensor is sliced up so the model can look at multiple different relationships at the exact same time (e.g., tracking grammar style and subject-verb relationships simultaneously).
- [done] **Learn That:** RoPE
- [done] **Learn This:** The Quadratic Bottleneck—why standard attention scales at $O(N^2)$, meaning if you double the text length, the computational cost quadruples.

---

## 🟨 Phase 3: Learn the Straight-Through Estimator (STE)
- [done] **Learn This:** The `torch.topk` operator—how to make a program look across a row of numbers and extract only the highest values[cite: 4].
- [done] **Learn That:** Hard binary masking—how to convert top-$k$ positions into a matrix of hard `1.0`s and `0.0`s[cite: 4].
- [done] **Learn This:** The Non-Differentiable Problem—why picking top-$k$ tokens outputs a flat gradient of zero, which completely breaks standard backpropagation[cite: 4].
- [done] **Learn That:** Custom Autograd Functions—how to create a `torch.autograd.Function` to trick PyTorch[cite: 4].
- [done] **Learn This:** The STE pass-through trick—how to code a `backward` method that passes gradients through a hard mask completely unaltered[cite: 4].

---

## 🟧 Phase 4: Learn Routing Networks & Mixed Attention Masks
- [done] **Learn This:** Low-rank projections—how projecting big vectors into a tiny gating space ($d_g \ll d_k$) saves massive compute energy[cite: 4].
- [done] **Learn That:** Gate score normalization—how to build a router module to calculate dynamic contextual scores for historical keys[cite: 4].
- [done] **Learn This:** Sliding window attention—why keeping a small, fixed window of immediate token neighbors always active stabilizes early training[cite: 4].
- [done] **Learn That:** Mask merging—how to write a single generator function that fuses causality, local sliding windows, and dynamic top-$k$ historical selections into one final mask[cite: 4].
- [done] **Learn This:** Causal leakage validation—how to look at gradients to guarantee a token at index $t$ has absolutely zero impact on previous history[cite: 4].

## 2. Router Dynamics & Stability
- [done] **Straight-Through Estimator (STE) vs. Soft Approximations:** Compare discrete hard selection via custom `torch.autograd.Function` against soft continuous approximations (e.g., Gumbel-Softmax, Sigmoid thresholding) and analyze gradient variance trade-offs.
- [done] **Load-Balancing & Auxiliary Loss ($\mathcal{L}_{\text{aux}}$):** Implement an auxiliary load-balancing loss ($\mathcal{L}_{\text{aux}} = \alpha \cdot S \sum f_i P_i$) to prevent **Router Collapse**, where the router locks onto a small subset of tokens and starves model capacity.
- [done] **Temperature Annealing:** Implement temperature controls ($T \to 0$) on gating Softmax to smoothly transition from soft exploratory routing during early training to sharp, deterministic choices during fine-tuning and inference.

---

## 3. Edge Cases, Numerical Safety & Diagnostics
- [done] **LayerNorm Derivative Degeneracy:** Recognize why taking unweighted output sums (`output[t].sum()`) across LayerNorm layers cancels out input gradients ($\sum (v - \mu) \equiv 0$), and utilize non-uniform projection vectors (`(output[t] * proj).sum()`) for diagnostic autograd tests.
- [done] **Softmax NaN Mitigation:** Prevent division-by-zero or NaN outputs in positions where all keys are masked out (e.g., dynamic $k=0$ or edge conditions) using explicit `-inf` clamping or `torch.nan_to_num`.
- [done] **Mixed Precision (FP16/BF16) Underflow:** Ensure large additive negative values (e.g., `-1e4` vs. `-inf`) maintain correct dynamic range across GPU backends (`torch.cuda.amp`) without underflowing or causing numeric instability.

---

## 4. Hardware Realities & Scaling
- [done] **Materialized Masks vs. Kernel Fusion:** Analyze why full 2D mask tensors ($S \times S$) introduce an $\mathcal{O}(S^2)$ memory bottleneck despite sparse token selection, and explore block-sparse GPU kernel alternatives (e.g., Triton / Block-Sparse FlashAttention).
- [done] **Batched Sequence Routing ($B > 1$):** Extend 2D sequence-level mask generation to handle 3D batched inputs (`[Batch, Seq_Len, Dim]`) with variable top-$k$ selections per sequence.

---

## 5. Algorithmic Bounds & Attention Economics
- [done] **Dynamic $k$-Scaling Boundaries:** Formalize and test log-scale routing bounds ($k = \Theta(\log_2 S)$) to guarantee sub-quadratic compute scaling while keeping a constant lower bound ($k \ge 1$) for context retrieval.
- [done] **Sequence Length Invariance & Stress Testing:** Evaluate routing entropy and top-$k$ sparsity stability across drastically different context lengths (e.g., $S=64$ vs. $S=4096$) without retuning temperature hyper-parameters.
- [done] **Multi-Head Routing Strategies:** Compare **Shared Sequence-Level Routing** (single router per layer, used in `nanoSubQ`) against **Per-Head Independent Routing** ($H$ distinct gating matrices), quantifying memory vs. expressiveness trade-offs.

---

## 6. Training Dynamics & Convergence
- [done] **Routing Gradient Explosion / Vanishing:** Monitor gradient norms ($\|\nabla_{W} \mathcal{L}\|$) through the STE layer during early training steps to detect gradient saturation or spike instabilities.
- [done] **Entropy Regularization:** Experiment with router score entropy penalties ($-\sum P_i \log P_i$) to control router sharpness and prevent premature convergence to deterministic binary gates.

---

# Phase 5: Hacking, Integrating & Training nanoGPT

## 1. Model Architecture & Refactoring (`model.py`)
- [done] **Deconstruct `CausalSelfAttention`:** Study the tensor flows, projection matrices ($W_q, W_k, W_v, W_o$), and scale dot-product attention calculation in original nanoGPT.
- [done] **Inject `SubQAttention` Module:** Replace standard dense self-attention with your custom `SubQAttention` containing the `PerKVHeadRouter`.
- [done] **Hook Up Dynamic KV Token Selection:** Pass input sequence tokens through your router to compute top-$K$ indices, dynamically pruning 50–70% of non-essential Key-Value pairs before score computation.
- [done] **Pass Through Straight-Through Estimator (STE):** Maintain non-differentiable top-$K$ discrete selection on the forward pass while routing continuous gradients backward to update the router logits.
- [done] **Aggregate Auxiliary Loss:** Modify `GPT.forward()` to return both task cross-entropy loss and router loss (`total_loss = cross_entropy_loss + router_aux_loss`).
- [done] **Build Pre-allocated KV-Cache:** Implement KV-caching with sparse indexing support to speed up autoregressive decoding during `model.generate()`.

---

## 2. Dataset Pipeline & Chat Tuning
- [done] **Dataset Curation:** Select a compact, high-quality instruction dataset (e.g., SmolTalk, UltraChat 200k, or a subset of OpenHermes) to teach conversational behavior.
- [done] **Implement ChatML Templates:** Format text data into structured conversation tokens using standard delimiters:
  ```text
  <|im_start|>user
  Question here...<|im_end|>
  <|im_start|>assistant
  Answer here...<|im_end|>
  ```
- [ ] **BPE Tokenization (`tiktoken`):** Set up tokenization scripts to encode ChatML-formatted strings into raw integer numpy `.bin` arrays for fast disk-to-GPU loading.
- [ ] **Supervised Loss Masking:** Modify dataset loading in `train.py` so cross-entropy loss is computed **only on assistant response tokens**, ignoring user prompt tokens (`ignore_index = -100`).

---

## 3. STE Training Dynamics & Stabilization (`train.py`)
- [ ] **Disable Router Biases:** Set `bias=False` on router linear projections to prevent baseline logit shifts from unbalancing STE selection.
- [ ] **Hyperparameter Configuration:**
  - Set learning rate to `3e-4` with a cosine decay schedule.
  - Set AdamW weight decay to `0.1` (applying `0.0` decay to router logit clamps and temperature params).
- [ ] **Enforce Safety Guardrails:**
  - Apply **Logit Clamping** (`clamp_val=4.0`) inside the router to prevent Sigmoid derivative vanishing.
  - Enable **Gradient Clipping** (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)`) to block gradient explosions during STE temperature decay.
- [ ] **Temperature & Entropy Schedules:** Implement exponential temperature decay ($T = 2.0 \to 0.1$) alongside binary entropy tracking to monitor router convergence from soft exploration to sharp selection.

---

## 4. Verification, Benchmarking & Performance
- [ ] **Overfitting Sanity Check:** Train on a single small batch (64 tokens) for 100 steps to confirm loss drops near $0.0$ and STE gradients update router weights properly.
- [ ] **Verify Linear Complexity $O(N \cdot K)$:** Measure memory footprint (VRAM usage) and execution time across growing context lengths ($N = 512 \to 2048$), confirming scale advantages over standard $O(N^2)$ attention.
- [ ] **Chatbot Inference Test:** Run `model.generate()` using temperature (`0.7`) and top-$p$ (`0.9`) sampling with a system prompt to verify coherent, fast multi-turn responses.