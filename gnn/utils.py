import networkx as nx

def get_edges(graph):
    '''
    returns x,y coordinates of edges in a list
    
    Arguments:
    ---------
    graph: networkx.Graph
    
    Returns:
    -------
    edge_x: list
    edge_y: list
    '''
    edge_x = []
    edge_y = []
    
    for edge in graph.edges():
        
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    return edge_x, edge_y


def get_nodes(graph):
    
    '''
    returns x,y coordinates of node in a list
    
    Arguments:
    ---------
    graph: networkx.Graph
    
    Returns:
    -------
    node_x: list
    node_y: list
    '''
    node_x = []
    node_y = []
    
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        
    return node_x, node_y


def get_degree(graph):
    
    '''
    returns degree of nodes in a list
    
    Arguments:
    ---------
    graph: networkx.Graph
    
    Returns:
    -------
    node_degree: list
    node_text: list
    '''
    node_degree = []
    node_text = []
    
    for adjacencies in graph.adjacency():
        neighbors = adjacencies[1]
        degree = len(neighbors)
        
        node_degree.append(degree)
        node_text.append('# of connections: '+str(degree))
        
    return node_degree, node_text


def build_all_to_all(num_nodes):
    '''
    returns an all-to-all graph
    
    Arguments:
    ---------
    num_nodes: int
    
    Returns:
    -------
    graph: networkx.Graph
        all-to-all graph with num_nodes nodes
    '''
    
    # create all-to-all edge tuples
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            # no self edges
            if i == j:
                continue
                
            edges.append((i + 1, j + 1)) # networkx.Graph starts from 1 not 0
            
    G = nx.Graph()
    G.add_edges_from(edges)
    # print(edges)
    
    return G


def build_edge_trace(edge_x, edge_y):
    
    edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')
    
    return edge_trace


def build_node_trace(node_x, node_y):
    
    node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))
    
    return node_trace


def build_graph_viz(edge_trace, node_trace):
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig