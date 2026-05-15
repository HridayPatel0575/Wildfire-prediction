import graphviz

def create_architecture_diagram():
    # Initialize graph with academic styling
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.7')
    
    # Global node and edge styles (Clean, no "fancy" colors, journal-ready)
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#F8F9FA', 
             color='#343A40', fontname='Times-Roman', fontsize='12', penwidth='1.5', margin='0.2')
    dot.attr('edge', color='#495057', penwidth='1.2', fontname='Times-Roman', fontsize='10')

    # 1. Input Node
    dot.node('Input', 'Input Weather Sequence\n(batch x 14 x 35)', fillcolor='#E9ECEF', style='filled')

    # --- Branch 1: LSTM Temporal ---
    with dot.subgraph(name='cluster_lstm') as c:
        c.attr(label='Branch 1: LSTM Temporal', style='dashed', color='gray', fontname='Times-Roman')
        c.node('LSTM1', 'LSTM Layer 1\n128 units, return_seq=True')
        c.node('Drop1', 'Dropout (30%)')
        c.node('LSTM2', 'LSTM Layer 2\n64 units, return_seq=False')
        c.node('Drop2', 'Dropout (30%)')
        c.node('Dense1', 'Dense (32 units, ReLU)')
        c.node('Drop3', 'Dropout (20%)')
        c.node('LogitLSTM', 'logit_lstm\nDense(1)')

        c.edges([('LSTM1', 'Drop1'), ('Drop1', 'LSTM2'), ('LSTM2', 'Drop2'), 
                 ('Drop2', 'Dense1'), ('Dense1', 'Drop3'), ('Drop3', 'LogitLSTM')])

    # --- Branch 2: Physics Statistics ---
    with dot.subgraph(name='cluster_physics') as c:
        c.attr(label='Branch 2: Physics Statistics', style='dashed', color='gray', fontname='Times-Roman')
        c.node('FWI_Col', 'Extract FWI Column')
        
        # Statistics nodes
        stats = ['Mean FWI', 'Max FWI', 'FWI Trend', 
                 'Fraction Elevated Days', 'FWI Volatility', 'Fraction Extreme Days']
        for stat in stats:
            c.node(stat, stat, fillcolor='#FFFFFF')
            c.edge('FWI_Col', stat)
            c.edge(stat, 'Concat')
            
        c.node('Concat', 'Concatenate 6 Statistics')
        c.node('BatchNorm', 'Batch Normalization')
        c.node('LogitPhysics', 'logit_physics\nDense(1)')
        
        c.edges([('Concat', 'BatchNorm'), ('BatchNorm', 'LogitPhysics')])

    # --- Fusion & Output ---
    dot.node('AlphaGate', 'Trainable Gate (α)', fillcolor='#FFFFFF', style='dashed,filled')
    dot.node('ScaledPhysics', 'Scaled Physics\n(α × logit_physics)')
    dot.node('ResidualFusion', 'Residual Fusion\n(final_logit = logit_lstm + Scaled Physics)', fillcolor='#DEE2E6')
    dot.node('Sigmoid', 'Sigmoid Activation')
    dot.node('Output', 'Fire Probability\n(Output)', fillcolor='#D1E7DD', style='filled,bold')

    # Connect Input to branches
    dot.edge('Input', 'LSTM1', label=' 35 features')
    dot.edge('Input', 'FWI_Col', label=' FWI extracted')

    # Connect Fusion mechanism
    dot.edge('LogitPhysics', 'ScaledPhysics')
    dot.edge('AlphaGate', 'ScaledPhysics')
    dot.edge('ScaledPhysics', 'ResidualFusion')
    dot.edge('LogitLSTM', 'ResidualFusion')
    
    dot.edge('ResidualFusion', 'Sigmoid')
    dot.edge('Sigmoid', 'Output')

    # Render to file (Generates both PDF for LaTeX and PNG for preview)
    dot.render('model_architecture_academic', view=False, format='png')
    dot.render('model_architecture_academic', view=False, format='pdf')
    print("Diagram generated successfully as 'model_architecture_academic.pdf'")

if __name__ == "__main__":
    create_architecture_diagram()