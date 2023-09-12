from lib.nn.graph_samplers.graph_sampler import GraphSampler
from lib.nn.scorer import AdjEmb

from tsl.nn.models import BaseModel


class GraphModule(BaseModel):
    def __init__(self,
                 n_nodes,
                 sampler,
                 mode,
                 k=10,
                 dummy_nodes=0,
                 tau=1.):
        super(GraphModule, self).__init__()

        self.mode = mode
        self.edge_scorer = AdjEmb(num_nodes=n_nodes + dummy_nodes)

        self.sampler = GraphSampler(mode=mode,
                                    sampler_type=sampler,
                                    tau=tau,
                                    k=k,
                                    dummy_nodes=dummy_nodes)

        self.dummy_nodes = dummy_nodes

    def forward(self, x, **kwargs):

        logits = self.edge_scorer(x)

        adj, (edge_index, edge_weight), ll = self.sampler(logits)
        _, (mean_edge_index, mean_edge_weight) = self.sampler.mode(logits)
        mean_graph = dict(edge_index=mean_edge_index,
                          edge_weight=mean_edge_weight)

        return dict(edge_index=edge_index,
                    edge_weight=edge_weight,
                    disjoint=adj.dim() > 2,
                    adj=adj,
                    ll=ll,
                    mean_graph=mean_graph,
                    logits=logits)
