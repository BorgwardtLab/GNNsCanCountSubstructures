import networkx as nx

class WL:

    def __init__(self):
        # setup wl hash dictionary
        self.wl_dict = {}
        self.wl_idx = 0


    def perform_k_steps(self, G, k):
        
        # get initial node and edge labels, or initialize labels if unavailable
        node_labels = nx.get_node_attributes(G, 'x')
        if not node_labels:
            node_labels = dict.fromkeys(G.nodes(), 0)
        edge_labels = nx.get_edge_attributes(G, 'edge_attr')
        if not edge_labels:
            edge_labels = dict.fromkeys(G.edges(), 0)
        
        # perform wl for k steps
        for i in range(k):

            # next iteration node labels
            new_node_labels = {}

            # relabel nodes
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                neighbors_labels = sorted([(str(node_labels.get(neighbor, None)), str(edge_labels.get((node, neighbor), None) or edge_labels.get((neighbor, node), None))) for neighbor in neighbors])
                concatenation = (str(node_labels[node]), tuple(neighbors_labels))
                if concatenation not in self.wl_dict:
                    self.wl_dict[concatenation] = self.wl_idx
                    self.wl_idx += 1
                new_label = self.wl_dict[concatenation]
                new_node_labels[node] = new_label

            # assing new labels
            node_labels = new_node_labels
            nx.set_node_attributes(G, node_labels, f'wl_{i+1}')
            G.graph[f'invariant_{i+1}'] = tuple(sorted(node_labels.values()))
        