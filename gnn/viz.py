import plotly.graph_objects as go
import random


def random_color():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())


def make_axis(title, tickangle):
    return {
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': tickangle,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 5,
      'showline': True,
      'showgrid': True
    }


def simplex_plot(Z_time_series):
    
    fig1 = go.Figure()
    
    for i in range (Z_time_series.shape[1]):
        simplex_map = {"Option1": Z_time_series[:,i,0], "Option2":Z_time_series[:,i,1], \
                       "Option3": Z_time_series[:,i,2]}

        marker_color = random_color()
        fig1.add_trace(go.Scatterternary({
                    'mode': 'markers',
                    'a': simplex_map['Option1'],
                    'b': simplex_map['Option2'],
                    'c': simplex_map['Option3'],
                    'marker': {
                        'symbol': 0,
                        'color': marker_color,
                        'size': 1,
                        'line': { 'width': 0.1 }
                    }
        }))
        
        fig1.add_trace(go.Scatterternary({
                    'mode': 'markers',
                    'a': [simplex_map['Option1'][-1]],
                    'b': [simplex_map['Option2'][-1]],
                    'c': [simplex_map['Option3'][-1]],
                    'marker': {
                        'symbol': 0,
                        'color': marker_color,
                        'size': 15,
                        'line': { 'width': 1 }
                    }
        }))
        
    fig1.show()
    

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