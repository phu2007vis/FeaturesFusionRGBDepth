import torch.nn as nn
from resources.utils import *


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=None, activation="ReLU"):
        super(LinearLayer, self).__init__()
        layers = [nn.Linear(in_features, out_features)]
        layers.append(get_activation(activation))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.linear_layer_main = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_layer_main(x)


class MLP(nn.Module):
    def __init__(self, feature_dims, dropout=None, activation="ReLU"):
        super(MLP, self).__init__()
        self.mlp_main = nn.ModuleList()
        dropout = duplicate(dropout, len(feature_dims)) if dropout else [None] * len(feature_dims)
        for i in range(len(feature_dims) - 1):
            self.mlp_main.append(LinearLayer(feature_dims[i], feature_dims[i + 1], dropout[i], activation))

    def forward(self, x):
        for layer in self.mlp_main:
            x = layer(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.3, activation="ReLU"):
        super(CrossAttentionLayer, self).__init__()
        self.attention_head = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm_layer1 = nn.LayerNorm(embed_dim)
        self.norm_layer2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP([embed_dim, embed_dim * 2, embed_dim], dropout=[0.1, 0.2, 0.3], activation=activation)

    def forward(self, key, value, query):
        output, _ = self.attention_head(query, key, value)
        output = self.norm_layer1(output + query)
        mlp_output = self.mlp(output)
        output = self.norm_layer2(mlp_output + output)
        return output


class PCrossAttentionModule(nn.Module):
    def __init__(self, embed_dim=512, num_layers=3, num_heads_each_layer=8, dropout=0.3, activation="ReLU"):
        super(PCrossAttentionModule, self).__init__()
        self.time_attention_module = nn.ModuleList()
        self.spatial_attention_module = nn.ModuleList()
        for _ in range(num_layers):
            self.time_attention_module.append(CrossAttentionLayer(embed_dim, num_heads_each_layer, dropout, activation))
            self.spatial_attention_module.append(
                CrossAttentionLayer(embed_dim, num_heads_each_layer, dropout, activation))
        self.activation = get_activation(activation)
        self.last_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        time_x, spatial_x = x
        assert time_x.shape == spatial_x.shape, "Shapes of time_x and spatial_x must match"
        for time_attention, spatial_attention in zip(self.time_attention_module, self.spatial_attention_module):
            time_x = time_attention(time_x, time_x, spatial_x)
            spatial_x = spatial_attention(spatial_x, spatial_x, time_x)
        output = self.last_norm(time_x + spatial_x)
        return self.activation(output)

#test
if __name__ == "__main__":
    import torch
    # x = torch.randn(3,32,64)
    # model = LinearLayer(64,56)
    # output = model(x)
    # print(output.shape)
    # output.sum().backward()
    
    # model = MLP([64,128,256],0.2)
    # output = model(x)
    # print(output.shape)
    # print(model)
    # output.sum().backward()
    # model = CrossAttentionLayer()
    # key = value = query = torch.rand(2,64,512)
    # model = PCrossAttentionModule(num_layers=2)
    # output = model((key,value))
    # sum = 0 
    # # for param  in model.parameters():
    # #     sum += param.numel()
    # print(sum)
    # print(output.sum().backward())
    pass