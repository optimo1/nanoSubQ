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
- [ ] **Temperature Annealing:** Implement temperature controls ($T \to 0$) on gating Softmax to smoothly transition from soft exploratory routing during early training to sharp, deterministic choices during fine-tuning and inference.

---

## 3. Edge Cases, Numerical Safety & Diagnostics
- [ ] **LayerNorm Derivative Degeneracy:** Recognize why taking unweighted output sums (`output[t].sum()`) across LayerNorm layers cancels out input gradients ($\sum (v - \mu) \equiv 0$), and utilize non-uniform projection vectors (`(output[t] * proj).sum()`) for diagnostic autograd tests.
- [ ] **Softmax NaN Mitigation:** Prevent division-by-zero or NaN outputs in positions where all keys are masked out (e.g., dynamic $k=0$ or edge conditions) using explicit `-inf` clamping or `torch.nan_to_num`.
- [ ] **Mixed Precision (FP16/BF16) Underflow:** Ensure large additive negative values (e.g., `-1e4` vs. `-inf`) maintain correct dynamic range across GPU backends (`torch.cuda.amp`) without underflowing or causing numeric instability.

---

## 4. Hardware Realities & Scaling
- [ ] **Materialized Masks vs. Kernel Fusion:** Analyze why full 2D mask tensors ($S \times S$) introduce an $\mathcal{O}(S^2)$ memory bottleneck despite sparse token selection, and explore block-sparse GPU kernel alternatives (e.g., Triton / Block-Sparse FlashAttention).
- [ ] **Batched Sequence Routing ($B > 1$):** Extend 2D sequence-level mask generation to handle 3D batched inputs (`[Batch, Seq_Len, Dim]`) with variable top-$k$ selections per sequence.

---

## 5. Algorithmic Bounds & Attention Economics
- [ ] **Dynamic $k$-Scaling Boundaries:** Formalize and test log-scale routing bounds ($k = \Theta(\log_2 S)$) to guarantee sub-quadratic compute scaling while keeping a constant lower bound ($k \ge 1$) for context retrieval.
- [ ] **Sequence Length Invariance & Stress Testing:** Evaluate routing entropy and top-$k$ sparsity stability across drastically different context lengths (e.g., $S=64$ vs. $S=4096$) without retuning temperature hyper-parameters.
- [ ] **Multi-Head Routing Strategies:** Compare **Shared Sequence-Level Routing** (single router per layer, used in `nanoSubQ`) against **Per-Head Independent Routing** ($H$ distinct gating matrices), quantifying memory vs. expressiveness trade-offs.

---

## 6. Training Dynamics & Convergence
- [ ] **Routing Gradient Explosion / Vanishing:** Monitor gradient norms ($\|\nabla_{W} \mathcal{L}\|$) through the STE layer during early training steps to detect gradient saturation or spike instabilities.
- [ ] **Entropy Regularization:** Experiment with router score entropy penalties ($-\sum P_i \log P_i$) to control router sharpness and prevent premature convergence to deterministic binary gates.

---

## 🟥 Phase 5: Learn How to Hack & Control nanoGPT
- [ ] **Learn This:** The structure of Andrej Karpathy's `model.py`—locating where the text flows into the core attention loops[cite: 4].
- [ ] **Learn That:** Code refactoring—how to safely swap out the original dense `CausalSelfAttention` module and plug in your custom sparse gating module[cite: 4].
- [ ] **Learn This:** Character-level token prep—how text datasets (like Shakespeare) are converted into raw integer arrays for training[cite: 4].
- [ ] **Learn That:** STE training dynamics—why you must set a lower learning rate (`3e-4`), turn off routing layer biases, and use gradient clipping to prevent gradient explosions[cite: 4].
- [ ] **Learn This:** Linear Scaling Performance—how to monitor validation loss and track VRAM usage to confirm your model successfully achieves linear $O(N \cdot k)$ efficiency[cite: 4].