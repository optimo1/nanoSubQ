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
- [ ] **Learn That:** The Value Aggregation Step—how the model uses those percentage weights to blend the Value ($V$) vectors together to create the final context-aware output.
- [ ] **Learn This:** Causal masking—how to use `torch.tril` and `torch.masked_fill` to block a model from looking at future words.
- [ ] **Learn That:** Multi-Head Attention splitting—how a single large tensor is sliced up so the model can look at multiple different relationships at the exact same time (e.g., tracking grammar style and subject-verb relationships simultaneously).
- [ ] **Learn This:** The Quadratic Bottleneck—why standard attention scales at $O(N^2)$, meaning if you double the text length, the computational cost quadruples.

---

## 🟨 Phase 3: Learn the Straight-Through Estimator (STE)
- [ ] **Learn This:** The `torch.topk` operator—how to make a program look across a row of numbers and extract only the highest values[cite: 4].
- [ ] **Learn That:** Hard binary masking—how to convert top-$k$ positions into a matrix of hard `1.0`s and `0.0`s[cite: 4].
- [ ] **Learn This:** The Non-Differentiable Problem—why picking top-$k$ tokens outputs a flat gradient of zero, which completely breaks standard backpropagation[cite: 4].
- [ ] **Learn That:** Custom Autograd Functions—how to create a `torch.autograd.Function` to trick PyTorch[cite: 4].
- [ ] **Learn This:** The STE pass-through trick—how to code a `backward` method that passes gradients through a hard mask completely unaltered[cite: 4].

---

## 🟧 Phase 4: Learn Routing Networks & Mixed Attention Masks
- [ ] **Learn This:** Low-rank projections—how projecting big vectors into a tiny gating space ($d_g \ll d_k$) saves massive compute energy[cite: 4].
- [ ] **Learn That:** Gate score normalization—how to build a router module to calculate dynamic contextual scores for historical keys[cite: 4].
- [ ] **Learn This:** Sliding window attention—why keeping a small, fixed window of immediate token neighbors always active stabilizes early training[cite: 4].
- [ ] **Learn That:** Mask merging—how to write a single generator function that fuses causality, local sliding windows, and dynamic top-$k$ historical selections into one final mask[cite: 4].
- [ ] **Learn This:** Causal leakage validation—how to look at gradients to guarantee a token at index $t$ has absolutely zero impact on previous history[cite: 4].

---

## 🟥 Phase 5: Learn How to Hack & Control nanoGPT
- [ ] **Learn This:** The structure of Andrej Karpathy's `model.py`—locating where the text flows into the core attention loops[cite: 4].
- [ ] **Learn That:** Code refactoring—how to safely swap out the original dense `CausalSelfAttention` module and plug in your custom sparse gating module[cite: 4].
- [ ] **Learn This:** Character-level token prep—how text datasets (like Shakespeare) are converted into raw integer arrays for training[cite: 4].
- [ ] **Learn That:** STE training dynamics—why you must set a lower learning rate (`3e-4`), turn off routing layer biases, and use gradient clipping to prevent gradient explosions[cite: 4].
- [ ] **Learn This:** Linear Scaling Performance—how to monitor validation loss and track VRAM usage to confirm your model successfully achieves linear $O(N \cdot k)$ efficiency[cite: 4].