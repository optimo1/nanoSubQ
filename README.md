# nanoSubQ AI

This is my journey of developing **nanoSubQ AI** from scratch. 

The main goal of this project is to take Andrej Karpathy's legendary `nanoGPT` repository and rewrite its core attention loop. I am stripping out the standard dense self-attention mechanism and replacing it with **Subquadratic Sparse Attention (SSA)**. 

Standard Transformers scale quadratically ($O(N^2)$), meaning they get exponentially slower and memory-heavy as text gets longer. nanoSubQ uses a learned, content-dependent routing network to calculate only the most high-signal token relationships, bringing that scaling cost down to a near-linear ($O(N \cdot k)$) curve.

*Note: Since it's been a minute since I deeply used Python and PyTorch, I am building this in public while spending the first few weeks going through a serious refresher on tensor mechanics, OOP structures, and custom autograd pipelines before hacking the main architecture.*

---

## 🗺️ The Strategy

Following the core architecture blueprint, the development is broken down into progressive milestones:

1. **Foundations:** Re-mastering PyTorch syntax, manual tensor slicing, and gradient graph tracking.
2. **The Bottleneck:** Building a baseline causal attention layer from scratch to mathematically map out the $O(N^2)$ scaling issue.
3. **The STE Hack:** Implementing a custom `torch.autograd.Function` using a Straight-Through Estimator (STE) to allow gradients to flow past the non-differentiable top-$k$ selection layer.
4. **The Router:** Building a low-rank gating network (`SSARouter`) that fuses local sliding windows with dynamic historical routing.
5. **The nanoGPT Intercept:** Swapping out the attention module inside Karpathy's `model.py` and training the new sparse model on the Shakespeare dataset.

---

## 🛠️ The Architecture Blueprint

The heart of the implementation relies on a Straight-Through Estimator to optimize the routing choice alongside token updates:

$$\tilde{A}_{ij} = A_{ij} \cdot M_{STE,ij} + (1 - M_{STE,ij}) \cdot (-1\text{e}9)$$

This setup allows the model to actively learn *which* historical parts of a sequence are actually worth paying attention to instead of checking every single box by default.

---

## 📈 Follow Along

I am documenting this entire process completely in public. I will be sharing my weekly milestones, deep dives, code updates, and inevitable bug fixes over on my LinkedIn page:

👉 [www.linkedin.com/in/adilzhanturaliev](https://www.linkedin.com/in/adilzhanturaliev)

Drop by, connect, and follow along to watch nanoSubQ AI come to life! 🚀