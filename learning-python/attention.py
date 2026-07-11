import torch
import torch.nn as nn
import torch.nn.functional as F

'''

# NN structure with automatic layers setup and Add&Norm


class Tool(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.layer_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x
        math_output = self.layer_block(x)
        return self.norm(math_output+shortcut)

class Main(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(Tool(d_model))

    def forward(self, x):
        for layers in self.layers:
            x = layers(x)
        return x


x = torch.tensor([1.0, 4.0, 5.0, 7.0, 3.0])

kaka = Main(5, 4)
print(kaka(x))


First class is used as a reusable tool for creating complete layer with activation (ReLU), addition (shortcut + math_output), and normalization (self.norm)

Second class uses this reusable tool and sets up the needed amount of nodes(d_model) and layers(num_layer) using for loop. 

This setup is considered the gold standard for big nns.


# NN with proper attention block -------------------------------------


class Tool(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        self.Q_proj = nn.Linear(d_model, d_model)
        self.K_proj = nn.Linear(d_model, d_model)
        self.V_proj = nn.Linear(d_model, d_model)

        self.linear_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x

        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        raw_scores = torch.matmul(Q, K.T)
        scaled_scores = raw_scores / (self.d_model ** 0.5)
        attention_weights = F.softmax(scaled_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        math_output = self.linear_block(attention_output)

        combined_output = math_output + shortcut
        return self.norm(combined_output)

class Main(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()

        self.layers = nn.ModuleList()
        
        self.num_layers = num_layers
        self.d_model = d_model

        for _ in range(num_layers):
            self.layers.append(Tool(d_model))

    def forward(self, x):
        for layers in self.layers:
            x = layers(x)
        return x


input_tokens = torch.tensor([
    [1.0, 2.0, 0.5, -1.0],  # Word 1
    [0.0, 4.5, -2.1, 1.2],  # Word 2
    [2.3, -0.8, 1.1, 0.0],  # Word 3
    [3.0, 6.0, 8.0, -4.0]   # Word 4
])

# Initialize our network: width (d_model) = 4 nodes, depth = 4 layers deep
model = Main(d_model=4, num_layers=10)

# Execute the forward pass!
output_tensor = model(input_tokens)

print(output_tensor)



# NN with casual masking

class Tool(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        self.Q_proj = nn.Linear(d_model, d_model)
        self.K_proj = nn.Linear(d_model, d_model)
        self.V_proj = nn.Linear(d_model, d_model)

        self.linear_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        shortcut = x
        seq_len = x.size(0)

        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        raw_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled_scores = raw_scores / (self.d_model ** 0.5)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        masked_scores = scaled_scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        math_output = self.linear_block(attention_output)
        output = math_output + shortcut

        return self.norm(output)

class Main(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()

        self.layer = nn.ModuleList()

        self.num_layers = num_layers
        self.d_model = d_model

        for _ in range(num_layers):
            self.layer.append(Tool(d_model))
            
    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


x = torch.tensor([
    [1.0, 4.0, 6.0, 4.0],
    [5.0, 7.0, 8.0, 1.0],
    [9.0, 2.0, 3.0, 6.0]
])
Kaka = Main(num_layers=5, d_model=4)
kaka_output = Kaka(x)
print (kaka_output)


# Multi-head processing

class Tool(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "division should be even"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.linear_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x
        seq_len = x.size(0)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        raw_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled_scores = raw_scores / (self.head_dim ** 0.5)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        masked_scores = scaled_scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        stitched_output = attention_output.transpose(0, 1).contiguous().view(seq_len, self.d_model)

        blended_output = self.out_proj(stitched_output)

        math_output = self.linear_block(blended_output)

        combined_output = math_output + shortcut

        return self.norm(combined_output)

class Main(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layer = nn.ModuleList()

        for _ in range(num_layers):
            self.layer.append(Tool(d_model, num_heads))

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


# A 2D matrix of shape (4 words, 16 features each)
x = torch.tensor([
    [1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0],
    [1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0],
])

Kaka = Main(d_model = 16, num_layers = 4 , num_heads = 4)
Output = Kaka(x)
print(Output)


# Grouped-Query Attention

class Tool(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = d_model // num_q_heads
        assert d_model % num_q_heads == 0, "should sdivide evenly"
        assert num_q_heads % num_kv_heads == 0, "should sdivide evenly"

        self.num_q_per_kv_head = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)
        self.linear_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x
        seq_len = x.size(0)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(seq_len, self.num_q_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)

        K = K.repeat_interleave(self.num_q_per_kv_head, dim = 0)
        V = V.repeat_interleave(self.num_q_per_kv_head, dim = 0)

        raw_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled_scores = raw_scores / (self.head_dim ** 0.5)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        masked_scores = scaled_scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        stitched_output = attention_output.transpose(0, 1).contiguous().view(seq_len, self.d_model)
        blended_output = self.out_proj(stitched_output)

        math_output = self.linear_block(blended_output)
        combined = math_output + shortcut

        return self.norm(combined)


class Main(nn.Module):
    def __init__(self, d_model, num_layers, num_q_heads, num_kv_heads):
        super().__init__()

        self.layers = nn.ModuleList([
            Tool(d_model, num_q_heads, num_kv_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



x = torch.randn(4, 16)

Kaka = Main(16, 4, 8, 2)
output = Kaka(x)
print(output)

        
'''


# With RoPE block for order

def Rotation(x):
    num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Should be even"

    positions = torch.arange(seq_len, device = x.device).unsqueeze(1)

    inv_freq = 1.0 / (1000 ** (torch.arange(0, head_dim, 2, device = x.device).float() / head_dim))
    angles = positions * inv_freq

    cos = torch.cos(angles).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0)

    x_left = x[..., :head_dim //2]
    x_right = x[..., head_dim // 2:]

    rotates_left = x_left * cos - x_right * sin
    rotates_right = x_right * cos + x_left * sin

    return torch.cat([rotates_left, rotates_right], dim=-1)
    

class Tool(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = d_model // num_q_heads
        assert d_model % num_q_heads == 0, "should sdivide evenly"
        assert num_q_heads % num_kv_heads == 0, "should sdivide evenly"

        self.num_q_per_kv_head = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)
        self.linear_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x
        seq_len = x.size(0)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(seq_len, self.num_q_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)

        Q = Rotation(Q)
        K = Rotation(K)
        
        K = K.repeat_interleave(self.num_q_per_kv_head, dim = 0)
        V = V.repeat_interleave(self.num_q_per_kv_head, dim = 0)

        raw_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled_scores = raw_scores / (self.head_dim ** 0.5)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        masked_scores = scaled_scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        stitched_output = attention_output.transpose(0, 1).contiguous().view(seq_len, self.d_model)
        blended_output = self.out_proj(stitched_output)

        math_output = self.linear_block(blended_output)
        combined = math_output + shortcut

        return self.norm(combined)


class Main(nn.Module):
    def __init__(self, d_model, num_layers, num_q_heads, num_kv_heads):
        super().__init__()

        self.layers = nn.ModuleList([
            Tool(d_model, num_q_heads, num_kv_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


x = torch.randn(4, 16)

Kaka = Main(16, 4, 8, 2)
output = Kaka(x)
print(output)



