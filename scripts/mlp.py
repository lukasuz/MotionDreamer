import torch

class ParamNet(torch.nn.Module):
    def __init__(self, num_out, freq=6):
        super().__init__()

        self.num_out = num_out
        self.t_embedder = Embedder(input_dims=1, multires=freq)
        self.t_embed_dim = self.t_embedder.output_dim

        input_dim = self.t_embed_dim
        self.transform_net = MLP(input_dim, num_layers=6, out_dim=self.num_out, hidden_dim=256)

    def forward(self, t):
        t_embedded = self.t_embedder.embed(t)
        return self.transform_net(t_embedded)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, out_dim=3, num_layers=3, hidden_dim=256, scale=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers-1):
            if i == 0:
                layers.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
            else:
                layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(torch.nn.LayerNorm(self.hidden_dim))
            layers.append(torch.nn.SiLU())

        self.out_layer = torch.nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        weights = scale * torch.nn.init.kaiming_uniform(torch.empty_like(self.out_layer.weight.data))
        self.out_layer.weight.data = weights
        layers.append(self.out_layer)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 
    
class DeformNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.xyz_embedder = Embedder(input_dims=3, multires=3)
        self.xyz_embed_dim = self.xyz_embedder.output_dim
        self.t_embedder = Embedder(input_dims=1, multires=2)
        self.t_embed_dim = self.t_embedder.output_dim

        input_dim = self.xyz_embed_dim + self.t_embed_dim
        self.transform_net = MLP(input_dim, num_layers=6)

    def forward(self, xyz, t):
        t_embedded = self.t_embedder.embed(t)
        xyz_embedded = self.xyz_embedder.embed(xyz)
        input = torch.concat((t_embedded, xyz_embedded), dim=1)
        dx = self.transform_net(input)

        return dx

class Embedder():
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L21
    def __init__(self, input_dims, multires=5, i=0):

        if i == -1:
            self.embed = torch.nn.Identity()
            self.output_dim = input_dims
            return

        self.kwargs = {
            'include_input': True,
            'input_dims': input_dims,
            'max_freq_log2': multires-1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.output_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
if __name__ == "__main__":
    e = Embedder(1, 6)
    print(e.output_dim)
