import torch
import torch.nn as nn

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

'''
First class is used as a reusable tool for creating complete layer with activation (ReLU), addition (shortcut + math_output), and normalization (self.norm)

Second class uses this reusable tool and sets up the needed amount of nodes(d_model) and layers(num_layer) using for loop. 

This setup is considered the gold standard for big nns.
'''