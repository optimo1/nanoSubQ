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
'''

# NN with proper attention block


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

        
