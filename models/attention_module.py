import torch

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, hidden_dim):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.hidden_dim=hidden_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.hidden_dim,
                                                             self.hidden_dim))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self,graph,features):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """

        with graph.local_scope():
            n_graphs=graph.batch_size
            nodes_per_graph=graph.batch_num_nodes
            representation=torch.zeros([n_graphs,self.hidden_dim]).to(features.device)
            start=0
            stop=start+nodes_per_graph[0]

            for xx in range(n_graphs):

                embedding=features[start:stop,:]
                global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
                transformed_global = torch.tanh(global_context)
                sigmoid_scores = torch.mm(embedding, transformed_global.view(-1, 1))
                representation[xx,:] = torch.t(torch.mm(torch.t(embedding), sigmoid_scores))
                                        
                if xx < n_graphs-1:
                    start=start+nodes_per_graph[xx]
                    stop=start+nodes_per_graph[xx+1]

            return representation

