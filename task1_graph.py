import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define positions and labels for each node
positions = {
    'Decision': (0, 10), 
    'Harvest Now': (-10, 8), 
    'Wait for Harvest': (10, 8),
    'Storm Occurs': (5, 6), 
    'No Storm': (15, 6),
    'Mold': (0, 4),
    'No Mold': (10, 4),
    'No Increase': (20, 4),
    'Typical Increase': (20, 2),
    'High Increase': (20, 0)
}

labels = {
    'Decision': 'Decision:\nHarvest Now or Wait',
    'Harvest Now': 'Harvest Now\nRevenue: $960,000',
    'Wait for Harvest': 'Wait for Harvest',
    'Storm Occurs': 'Storm Occurs (50%)',
    'No Storm': 'No Storm (50%)',
    'Mold': 'Mold (10%)\nRevenue: $2,880,000',
    'No Mold': 'No Mold (90%)\nRevenue: $480,000',
    'No Increase': 'No Increase (60%)\nRevenue: $960,000',
    'Typical Increase': 'Typical Increase (30%)\nRevenue: $1,410,000',
    'High Increase': 'High Increase (10%)\nRevenue: $1,500,000'
}

# Add nodes with labels
for node, pos in positions.items():
    G.add_node(node, pos=pos, label=labels[node])

# Define and add edges
edges = [
    ('Decision', 'Harvest Now'),
    ('Decision', 'Wait for Harvest'),
    ('Wait for Harvest', 'Storm Occurs'),
    ('Wait for Harvest', 'No Storm'),
    ('Storm Occurs', 'Mold'),
    ('Storm Occurs', 'No Mold'),
    ('No Storm', 'No Increase'),
    ('No Storm', 'Typical Increase'),
    ('No Storm', 'High Increase')
]

edge_labels = {
    ('Decision', 'Harvest Now'): 'Harvest Now',
    ('Decision', 'Wait for Harvest'): 'Wait',
    ('Wait for Harvest', 'Storm Occurs'): 'Decision: Storm?',
    ('Wait for Harvest', 'No Storm'): 'No Storm',
    ('Storm Occurs', 'Mold'): 'Mold (10%)',
    ('Storm Occurs', 'No Mold'): 'No Mold (90%)',
    ('No Storm', 'No Increase'): 'No Increase (60%)',
    ('No Storm', 'Typical Increase'): 'Typical Increase (30%)',
    ('No Storm', 'High Increase'): 'High Increase (10%)'
}

for edge in edges:
    G.add_edge(*edge)

# Set up the plot with matplotlib
plt.figure(figsize=(14, 10))
nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=6000, node_color='skyblue', font_size=9, font_weight='bold', arrowstyle='-|>', arrowsize=20)
nx.draw_networkx_edge_labels(G, pos=nx.get_node_attributes(G, 'pos'), edge_labels=edge_labels, font_color='red')
plt.title('Alejandro\'s Decision Tree', size=15)

# Save the plot to a PNG file in the current directory
plt.savefig('task1_graph.png')

# Optionally display the plot as well
plt.show()
