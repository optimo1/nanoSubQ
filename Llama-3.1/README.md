# Llama-3.1 — Local scaffold for SSA integration

This folder contains the **architecture config only** (no weights) for `meta-llama/Llama-3.1-8B`.

## What's here
- `config.json` — model architecture
- `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` — tiktoken tokenizer

## No weights
This folder has architecture + tokenizer only (~9MB). The 16GB weights are loaded at runtime from HuggingFace on the cloud GPU via `LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")`.

## Key architecture details for SSA integration
| param | value |
|---|---|
| hidden_size | 4096 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA) |
| head_dim | 128 (4096/32) |
| num_hidden_layers | 32 |
| intermediate_size | 14336 |
| vocab_size | 128256 |
| max_position_embeddings | 131072 |
| rope_theta | 500000.0 |
| rope_scaling | llama3 (factor=8, linear extend to 128K) |
