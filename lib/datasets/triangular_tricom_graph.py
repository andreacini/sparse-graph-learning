import numpy as np

class Graph(object):
    """ 
    A graph class exposing some plotting utilities.
    """

    def __init__(self, edge_index=None, node_position=None, name=None):
        if edge_index is not None:
            self.edge_index = edge_index
        if node_position is not None:
            self.node_position = node_position
        if node_position is not None:
            self.name = "n.a."

        self.num_nodes = self.node_position.shape[0]
        self.num_edges = self.edge_index.shape[1]
        self.adj_ = None
        self.laplacian_ = None

    @property
    def adj(self):
        if self.adj_ is None:
            import scipy.sparse
            # print("oil")
            try:
                self.adj_ = scipy.sparse.coo_array((np.ones(self.edge_index.shape[1]), (self.edge_index[0], self.edge_index[1])))
            except AttributeError:
                self.adj_ = scipy.sparse.coo_matrix((np.ones(self.edge_index.shape[1]), (self.edge_index[0], self.edge_index[1])))
        return self.adj_

    def laplacian(self, normalized=False):
        if self.laplacian_ is None:
            deg = self.adj.sum(axis=1)
            self.laplacian_ = np.diag(deg) - self.adj
            if normalized:
                self.laplacian_ = self.laplacian_ / np.sqrt(deg.reshape(-1, 1) * deg.reshape(1, -1))
        return self.laplacian_

    def plot_static(self, signal=None, node_labels=None,
                    with_node_signal=True, with_edge_sign=True, with_node_labels=False,
                    cmap="RdBu", alpha=None,
                    savefig=None):
        """
        Plot the graph using the node positions stored in the graph (self.node_positions).
        Optionally,
            - node indices can be drawn over the nodes
            - values of a node signal (T=1, N, F=1) can be encoded as node colors
            - edges can be colored according to whether the associated node signals change or not.

        :param signal: (T=1, N, F=1).
        :param node_labels: list of indices
        :param with_node_signal: whether to color the nodes according to the provided signal or not
        :param with_edge_sign: whether to color the edges according to the sign changes
        :param with_node_labels: whether to write the node indices on top of the nodes
        :param cmap: matplotlib colormap or string
        :param alpha: transparency of the edges
        :param savefig: if a string is provided, the figure is saved to file savefig.
        :return: figure instance
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        if signal is not None:
            assert signal.ndim == 2
            assert signal.shape[1] == 1

        with_node_signal = with_node_signal and signal is not None
        with_node_labels = with_node_labels or node_labels is not None
        with_edge_sign = with_edge_sign and signal is not None

        # ff = plt.gcf()
        # ff.set_figheight(4)
        # ff.set_figwidth(10)
        # plt.box(False)

        # Colors
        if isinstance(cmap, str):
            cm_cont = plt.get_cmap(cmap)
        else:
            cm_cont = cmap
        # Sign colors
        colors = {-1: (0.7, 0.7, 0.7), 0: (1.0, 1.0, 1.0), 1: (0.0, 0.0, 0.0)}

        # Create nx graph
        G = nx.Graph(name=self.name)
        #nodes
        pos = {}
        for i in range(self.num_nodes):
            G.add_node(i, pos=self.node_position[i])
            pos[i] = self.node_position[i]
        #edges
        for e in range(self.num_edges):
            if with_edge_sign:
                xx = signal[self.edge_index[0, e]] * signal[self.edge_index[1, e]]
                sgn = np.sign(xx.sum())
            else:
                sgn = 1
            G.add_edge(self.edge_index[0, e], self.edge_index[1, e],
                       sign=sgn, color=colors[sgn])

        # Draw nodes
        if with_node_signal:
            vmax = np.abs(signal).max()
            sm = plt.cm.ScalarMappable(cmap=cm_cont, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
            plt.colorbar(sm)
            args = dict(node_color=signal, cmap=cm_cont, vmin=-vmax, vmax=vmax)
        else:
            args = dict(node_color="lightsteelblue")
        h1 = nx.draw_networkx_nodes(G, pos=pos, **args)
        if with_node_labels:
            if node_labels is None:
                lab = {i: i for i in range(self.num_nodes)}
            else:
                lab = {i: node_labels[i] for i in range(self.num_nodes)}
            h1l = nx.draw_networkx_labels(G, pos=pos, labels=lab)

        # Draw edges
        col = nx.get_edge_attributes(G, 'color').values()
        h2 = nx.draw_networkx_edges(G, pos=pos, width=4, edge_color=col, alpha=alpha)

        # Color legend
        if with_edge_sign:
            # https://stackoverflow.com/questions/19877666/add-legends-to-linecollection-plot 
            def make_proxy(clr, mappable, **kwargs):
                from matplotlib.lines import Line2D
                return Line2D([0, 1], [0, 1], color=clr, **kwargs)
            # generate proxies with the above function
            proxies = [make_proxy(c, h2, lw=5) for s, c in colors.items() if s != 0]
            # and some text for the legend -- you should use something from df.
            labels = [f"sgn = {s}" for s, _ in colors.items() if s != 0]
            plt.legend(proxies, labels)

        plt.axis("equal")
        plt.tight_layout()

        if savefig is not None:
            assert isinstance(savefig, str)
            ss = savefig.split(".")
            if len(ss)>1:
                savefig = ".".join(ss[:-1]) + "_static." + ss[-1]
            else:
                savefig = savefig + "_static"
            plt.savefig(savefig)
        return plt.gcf()

    def plot_temporal(self, signal,
                      max_aspect_ratio: int=3,
                      fig_scale: int=6,
                      savefig: [str, None]=None,
                      return_indices: bool=False,
                      cmap="RdBu",
                      **kwargs):
        """
        Heatmap of a scalar graph signal (T, N, F=1) with time on the x-axis and nodes on the y-axis.
        Nodes are sorted according to the spectral embedding with only one component to
        give similar indices to nodes that are close in graph.

        :param signal: (T, N, F=1).
        :param max_aspect_ratio: max aspect ratio of the figure, when N>>T or T>>N.
        :param fig_scale: inches of the shorter dimension of the figure.
        :param savefig: if a string is provided, the figure is saved to file savefig.
        :param return_indices: whether to the indices and eigenvector computed by the spectral embedding or not
        :param cmap: matplotlib colormap or string
        :return: (figure instance [, indices spec.emb., eigenvector spec.emb.])
        """
        from sklearn.manifold import spectral_embedding
        import matplotlib.pyplot as plt
        import seaborn as sb
        # Colors
        if isinstance(cmap, str):
            cm_cont = plt.get_cmap(cmap)
        else:
            cm_cont = cmap

        A = self.adj.todense()
        if isinstance(A, np.matrix):
            A = np.array(A)
        A += A.T
        A *= 0.5
        v = spectral_embedding(adjacency=A, n_components=1)
        ii = v.ravel().argsort()

        if signal.ndim == 2:
            signal_ = signal[..., None]
        else:
            signal_ = signal

        # (T, N, F)
        T, N, F = signal_.shape
        assert N == A.shape[0]

        if T/N < max_aspect_ratio and N/T < max_aspect_ratio:
            aspect_ratio = 1.0
            plt.figure(figsize=[fig_scale, fig_scale * F])
        elif T/N > max_aspect_ratio:
            aspect_ratio = T / (N * max_aspect_ratio)
            plt.figure(figsize=[fig_scale, fig_scale * F / max_aspect_ratio])
        else:
            aspect_ratio = (N * max_aspect_ratio) / T
            plt.figure(figsize=[fig_scale / max_aspect_ratio, fig_scale * F])

        for f in range(F):
            plt.subplot(F, 1, f+1)
            sb.heatmap(signal_[:, ii, f].T, cmap=cm_cont)
            plt.ylabel("Node")
            plt.gca().set_aspect(aspect_ratio)
        plt.xlabel("Time")
        # sb.heatmap(signal_[ii, :, 0], cmap=cm_cont)
        # plt.xlabel("Time")
        # plt.ylabel("Node")
        # plt.gca().set_aspect(aspect_ratio)

        plt.tight_layout()
        if savefig is not None:
            assert isinstance(savefig, str)
            ss = savefig.split(".")
            if len(ss)>1:
                savefig = ".".join(ss[:-1]) + "_temporal." + ss[-1]
            else:
                savefig = savefig + "_temporal"
            plt.savefig(savefig)

        if return_indices:
            return plt.gcf(), ii, v
        else:
            return plt.gcf()

    def plot(self, signal=None, cmap="RdBu", **kwargs):
        """
        Combines plot_static and plot_temporal.
        """
        temp_signal = False
        if signal is not None and signal.ndim == 2 and signal.shape[1] > 1:
            temp_signal = True

        if temp_signal:
            # plt.subplot(1, 2, 2)
            _, ii, v = self.plot_temporal(signal=signal, cmap=cmap, return_indices=True, **kwargs)
            # plt.subplot(1, 2, 1)
            import matplotlib.pyplot as plt
            plt.figure()
            self.plot_static(signal=signal[signal.shape[0]//2, :][..., None], node_labels=ii, cmap=cmap, **kwargs)
        else:
            self.plot_static(signal=signal, cmap=cmap, **kwargs)


class TriCommunityGraph(Graph):
    """
    A family of planar graphs composed of a number of communities.
    Each community takes the form of a 6-node triangle:
            2
           / \
          1 - 4
         / \ / \
        0 - 3 - 5
    All communities are arranged either as a line
        c0 - c1 - c2 -  ....
    or in a planar conformation
               c5
              /  \
            c3 - c2
           /  \ /  \
         c4 - c0 - c1 -
    """

    def __init__(self, communities: int = 1, connectivity: str = "line", **kwargs):
        """
        :param communities: number of communities
        :param connectivity: either line of triangle for linear and planar conformation respectively
        """
        self.num_communities = communities
        self.community_connectivity = connectivity
        self.edge_len = 1.0
        nodes_ = []
        edges_ = []
        pos_ = []
        for c in range(self.num_communities):
            n, p, e, ex = self.compose_community_structure(comm_id=c)
            nodes_.append(n)
            edges_.append(e)
            edges_.append(ex)
            pos_.append(p)

        self.nodes = np.concatenate(nodes_, axis=0).ravel()
        self.node_position = np.concatenate(pos_, axis=0)
        self.edge_index = np.concatenate(edges_, axis=0).T
        self.edge_index = np.concatenate([self.edge_index, self.edge_index[::-1]], axis=1)
        self.edge_index = np.unique(self.edge_index, axis=1)
        self.edge_weight = None

        super(TriCommunityGraph, self).__init__()

        self.name = f"TriCom[{self.num_communities}]"

    def compose_community_structure(self, comm_id: int = 0):
        """
        Line:
          c0 - c1 - c2 -  ....
        Triangle:
               c5
              /  \
            c3 - c2
           /  \ /  \
         c4 - c0 - c1 -
        :param pos:
        :param comm_id:
        :return:
        """
        n, p, e = self.create_community()
        n += comm_id * 6
        e += comm_id * 6

        def move_hor(pos, step):
            pos[:, 0] = pos[:, 0] + 3 * self.edge_len * step
            return pos

        def move_vert(pos, step):
            pos[:, 1] = pos[:, 1] + 3 * self.edge_len * np.sqrt(3) / 2. * step
            return pos

        def move(pos, step_h=0, step_v=0):
            pos = move_hor(pos, step=step_h)
            pos = move_vert(pos, step=step_v)
            return pos

        extra_edges = np.empty((0,2), dtype=int)
        if self.community_connectivity == "line":
            if comm_id > 0:
                p = move(p, step_h=1*comm_id)
                extra_edges = np.array([[n[0] - 1, n[0]]])
        elif self.community_connectivity == "triangle":
            if comm_id == 0:
                pass
            elif comm_id == 1:
                p = move(p, step_h=1, step_v=0)
                extra_edges = np.array([[n[0] - 1, n[0]]])
            elif comm_id == 2:
                p = move(p, step_h=0.5, step_v=1)
                extra_edges = np.array([[n[0] - 6 + 2, n[-1]],
                                        [n[0], 2]])
            elif comm_id == 3:
                p = move(p, step_h=-0.5, step_v=1)
                extra_edges = np.array([[n[0] - 6, n[-1]],
                                        [2, n[-1]]])
            elif comm_id == 4:
                p = move(p, step_h=-1, step_v=0)
                raise NotImplementedError
            elif comm_id == 5:
                p = move(p, step_h=0, step_v=2)
                raise NotImplementedError
            else:
                raise NotImplementedError
        return n, p, e, extra_edges

    def create_community(self):
        """
                  2
                 / \
                1 - 4
               / \ / \
        ... - 0 - 3 - 5 - ...
        """
        nodes = np.array(list(range(6)))
        pos = []
        for i in range(3):
            for j in range(3-i):
                # pos.append([i, 2-j])
                pos.append([i + j * 0.5, j * np.sqrt(3) / 2.])
        pos = np.array(pos) * self.edge_len
        edges = [[0, 1], [1, 2], [3, 4],  # slashes
                 [1, 3], [2, 4], [4, 5],  # backslashes
                 [0, 3], [1, 4], [3, 5]]  # horizontal
        edges = np.array(edges)
        return nodes, pos, edges
