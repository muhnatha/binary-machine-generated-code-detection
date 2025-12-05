import sys
import os
import tree_sitter_languages
from tree_sitter import Parser
import graphviz

# --- Import your dataset ---
sys.path.append(os.getcwd())
try:
    from src.dataset import RawCodeDataset
except ImportError:
    print("Error: Could not import RawCodeDataset. Run from project root.")
    sys.exit(1)

def add_to_graph(node, graph, source_code, parent_id=None, counter=None):
    """
    Recursively adds nodes and edges to the Graphviz object.
    """
    if counter is None:
        counter = [0]
    
    # Generate a unique ID for this node in the graph
    current_id = str(counter[0])
    counter[0] += 1
    
    # Determine the label (Type + Text for leaf nodes)
    node_label = node.type
    if len(node.children) == 0:
        node_text = source_code[node.start_byte:node.end_byte].decode("utf8", errors='replace')
        # Escape special characters for Graphviz
        node_text = node_text.replace('\\', '\\\\').replace('"', '\\"')
        if len(node_text) > 20: node_text = node_text[:20] + "..."
        node_label = f"{node.type}\n'{node_text}'"
    
    # Add node to graph
    # Shapes: 'box' for statements, 'ellipse' for expressions, 'plaintext' for leaves
    shape = 'box' if len(node.children) > 0 else 'ellipse'
    graph.node(current_id, label=node_label, shape=shape, style='filled', fillcolor='white')
    
    # Add edge from parent
    if parent_id is not None:
        graph.edge(parent_id, current_id)
    
    # Recursion
    for child in node.children:
        add_to_graph(child, graph, source_code, current_id, counter)

def main():
    # 1. Load 1 Sample
    print("Loading 1 sample...")
    dataset = RawCodeDataset(split='train', subsample=True, sample_size=6)
    item = dataset[0]
    
    code = item['code']
    lang = item['language']
    
    print(f"Visualizing {lang} code...")

    # 2. Setup Parser
    lang_map = {'Python': 'python', 'Java': 'java', 'C++': 'cpp', 'Go': 'go'}
    ts_lang = lang_map.get(lang)
    
    if not ts_lang:
        print(f"Skipping {lang} (no map)")
        return

    parser = Parser()
    language = tree_sitter_languages.get_language(ts_lang)
    parser.set_language(language)
    
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)

    # 3. Create Graph
    dot = graphviz.Digraph(comment=f'{lang} AST')
    dot.attr(rankdir='TB')  # Top to Bottom layout
    
    # Limit depth to keep image readable (optional)
    # For full tree, remove this or pass full tree
    print("Generating Graph...")
    add_to_graph(tree.root_node, dot, code_bytes)
    
    # 4. Render
    output_filename = 'ast_visualization'
    dot.render(output_filename, format='png', view=False)
    print(f"\n✅ Image saved to: {output_filename}.png")

if __name__ == "__main__":
    main()