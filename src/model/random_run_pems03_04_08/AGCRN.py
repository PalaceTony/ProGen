import torch
import torch.nn.functional as F
import torch.nn as nn


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1
        )
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            )
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum(
            "nd,dkio->nkio", node_embeddings, self.weights_pool
        )  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias  # b, N, dim_out
        return x_gconv, supports[1]


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        _, adj = self.gate(input_and_state, node_embeddings)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings)[0])
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings)[0])
        h = r * state + (1 - r) * hc
        return h, adj

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim)
            )

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state, adj = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden, adj

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.T_p
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim

        # self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_node, self.embed_dim), requires_grad=True
        )

        self.encoder = AVWDCRNN(
            args.num_nodes,
            args.input_dim,
            args.rnn_units,
            args.cheb_k,
            args.embed_dim,
            args.num_layers,
        )

        # predictor
        self.end_conv = nn.Conv2d(
            1,
            args.T_p * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        )

    def forward(self, source):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _, adj = self.encoder(
            source, init_state, self.node_embeddings
        )  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden
        adj = self.node_embeddings.detach()

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(
            -1, self.horizon, self.output_dim, self.num_node
        )
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output, adj.to(source.device)


if __name__ == "__main__":
    import torch
    from types import SimpleNamespace

    # Define arguments for the model
    args = SimpleNamespace(
        num_nodes=170,  # Number of nodes in the graph
        input_dim=1,  # Dimension of the input features
        rnn_units=64,  # Dimension of the RNN output/features
        output_dim=1,  # Dimension of the final output
        T_p=12,  # Prediction horizon
        num_layers=2,  # Number of RNN layers
        cheb_k=2,  # Order of the Chebyshev polynomial
        embed_dim=2,  # Dimension of node embeddings
    )

    # Initialize the model
    model = AGCRN(args)

    # Create dummy data
    batch_size = 2
    sequence_length = 12
    dummy_input = torch.randn(
        batch_size, sequence_length, args.num_nodes, args.input_dim
    )

    outputs, adjacency_matrix = model(dummy_input)

    print("Outputs shape:", outputs.shape)
    print("Adjacency Matrix shape:", adjacency_matrix.shape)
