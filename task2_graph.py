import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define node positions
positions = {
    'Decision': (0, 10),
    'Harvest Now': (-10, 6),
    'Forecast Storm': (10, 6),
    'Forecast No Storm': (20, 6),
    'Actual Storm': (5, 2),
    'Actual No Storm': (15, 2),
    'Mold': (0, -2),
    'No Mold': (10, -2),
    'No Actual Storm': (20, 2)
}

# Define node labels with updated revenue data
labels = {
    'Decision': 'Decision:\nHarvest Now or Wait for Forecast',
    'Harvest Now': 'Harvest Now\nRevenue: $960,000',
    'Forecast Storm': 'Forecast: Storm (76.79%)',  # Updated with XGBoost accuracy
    'Forecast No Storm': 'Forecast: No Storm (23.21%)',  # Updated with 1 - XGBoost accuracy
    'Actual Storm': 'Actual Storm Occurs\n75% Accurate',  # Updated with XGBoost recall
    'Actual No Storm': 'No Storm Occurs\n25% Inaccuracy',  # Updated with 1 - XGBoost recall
    'Mold': 'Mold Develops (10%)\nRevenue: $2,880,000',
    'No Mold': 'No Mold (90%)\nRevenue: $480,000',
    'No Actual Storm': 'Correct No Storm Prediction\nRevenue: $960,000'
}

# Add nodes and labels
for node, pos in positions.items():
    G.add_node(node, pos=pos, label=labels[node])

# Define and add edges
edges = [
    ('Decision', 'Harvest Now'),
    ('Decision', 'Forecast Storm'),
    ('Decision', 'Forecast No Storm'),
    ('Forecast Storm', 'Actual Storm'),
    ('Forecast Storm', 'Actual No Storm'),
    ('Forecast No Storm', 'No Actual Storm'),
    ('Actual Storm', 'Mold'),
    ('Actual Storm', 'No Mold')
]

edge_labels = {
    ('Decision', 'Harvest Now'): 'Harvest Now',
    ('Decision', 'Forecast Storm'): 'Forecast: Storm',
    ('Decision', 'Forecast No Storm'): 'Forecast: No Storm',
    ('Forecast Storm', 'Actual Storm'): 'Recall: 75%',  # Updated with XGBoost recall
    ('Forecast Storm', 'Actual No Storm'): '1-Recall: 25%',  # Updated with 1 - XGBoost recall
    ('Forecast No Storm', 'No Actual Storm'): 'Correct No Storm Prediction',
    ('Actual Storm', 'Mold'): 'Mold Develops (10%)',
    ('Actual Storm', 'No Mold'): 'No Mold Develops (90%)'
}

for edge in edges:
    G.add_edge(*edge)

# Plot setup
plt.figure(figsize=(16, 12))
nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=5000, node_color='lightblue', font_size=10, font_weight='bold', arrowstyle='-|>', arrowsize=10)
nx.draw_networkx_edge_labels(G, pos=nx.get_node_attributes(G, 'pos'), edge_labels=edge_labels, font_color='red')

plt.title('Decision Tree with Simplified ML Model Integration and Revenues', size=15)
plt.savefig('simplified_decision_tree_with_ml_model_and_revenues.png') # Saving the graph
plt.show()