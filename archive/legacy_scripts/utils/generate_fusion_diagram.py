# Date Created: 2026-03-02 11:07:58+03:00
# Date Last Modified: 2026-05-15

from pathlib import Path

import graphviz

from src.utils.paths import docs_path

def generate_fusion_diagram():
    """
    Generates the Thesis Hybrid Fusion Diagram.
    - Font: Avenir (Mac system font)
    - Fixes typo: "Aggregation"
    - Output: docs/diagrams/fusion_diagram.png
    """

    # Keep generated diagram alongside the migrated documentation artifacts.
    output_dir = docs_path("diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Graph
    dot = graphviz.Digraph('Hybrid_Fusion', comment='Thesis Architecture')
    
    # Global Layout & Font Settings (Avenir)
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.6')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f9faff', 
             fontname='Avenir', fontsize='10', penwidth='1.0', color='#b0b0b0')
    dot.attr('edge', fontname='Avenir', fontsize='9', color='#555555')
    
    # Cluster styling
    cluster_attr = {'style': 'filled', 'color': '#eaf2f8', 'fontname': 'Avenir-Heavy'}

    # ---------------------------
    # Column 1: Lexicon-Based (NRC)
    # ---------------------------
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Lexicon-Based (NRC)\n(word → 10 emotions)', **cluster_attr)
        
        c.node('NRC_Data', label='📄 NRC Data', shape='note')
        c.node('NRC_Process', label='SUM (Σ) over review counts', shape='box')
        
        # TYPO FIXED: "Aggregation"
        c.node('NRC_Output', label=r'<<B>NRC Aggregation per perfume:</B><BR align="left"/>• Σ anger, joy, ..., negative<BR align="left"/>→ NRC_*>', 
               shape='note', align='left')

        c.edge('NRC_Data', 'NRC_Process')
        c.edge('NRC_Process', 'NRC_Output')

    # ---------------------------
    # Column 2: Lexicon-Based (VAD)
    # ---------------------------
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Lexicon-Based (VAD)\n(word → V, A, D)', **cluster_attr)
        
        c.node('VAD_Data', label='📄 VAD Data', shape='note')
        c.node('VAD_Process', label='MEAN (μ) over review scores', shape='box')
        c.node('VAD_Output', label=r'<<B>μ(V), μ(A), μ(D)</B><BR align="left"/>→ VAD_valence, VAD_arousal,<BR align="left"/>VAD_dominance>', 
               shape='note')

        c.edge('VAD_Data', 'VAD_Process')
        c.edge('VAD_Process', 'VAD_Output')

    # ---------------------------
    # Column 3: Transformer-Based
    # ---------------------------
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Transformer-Based (Contextual)\n(contextual models)', **cluster_attr)
        
        c.node('Trans_Model', label='🧠 Transformer Models', shape='component')
        c.node('Trans_Process', label='MEAN (probabilities) + MODE (top labels)', shape='box')
        c.node('Trans_Output', label=r'<<B>Transformer Aggregation per perfume:</B><BR align="left"/>• μ of each emotion score per model<BR align="left"/>• Mode of top-predicted labels → Σ_Label>', 
               shape='note')

        c.edge('Trans_Model', 'Trans_Process')
        c.edge('Trans_Process', 'Trans_Output')

    # ---------------------------
    # Fusion & Output
    # ---------------------------
    # Central Fusion Node
    dot.node('Hybrid_Fusion', 
             label=r'<<B>Hybrid Fusion (per perfume)</B><BR/>merge_emotion_profiles.py +<BR/>aggregate_and_merge_emotions.py>', 
             shape='box', style='filled,rounded', fillcolor='#ffe6cc', color='#d68910')

    # Database/Final Output Node
    dot.node('Final_CSV', 
             label=r'<<B>Unified_Emotion_Profile_Perfume.csv</B><BR/>= "Olfactory fingerprint" matrix<BR/>(one unified feature vector per perfume)>', 
             shape='cylinder', style='filled', fillcolor='#dbeaff', color='#2874a6')

    # Connect Columns to Fusion
    dot.edge('NRC_Output', 'Hybrid_Fusion')
    dot.edge('VAD_Output', 'Hybrid_Fusion')
    dot.edge('Trans_Output', 'Hybrid_Fusion')
    
    # Connect Fusion to Final CSV
    dot.edge('Hybrid_Fusion', 'Final_CSV')

    # ---------------------------
    # Render
    # ---------------------------
    try:
        # cleanup=True removes the temporary source file
        output_path = dot.render("fusion_diagram", directory=str(output_dir), format="png", cleanup=True)
        print(f"✅ Success! Diagram saved to: {output_path}")
    except Exception as e:
        print("\n❌ ERROR: Graphviz executable not found.")
        print("Please run: 'brew install graphviz' in your terminal.")
        print(f"Details: {e}")

if __name__ == "__main__":
    generate_fusion_diagram()
