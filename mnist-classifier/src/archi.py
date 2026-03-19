import torch
from torchview import draw_graph
from model import MNISTClassifier

model = MNISTClassifier()

graph = draw_graph(
    model,
    input_size=(1, 1, 28, 28),
    expand_nested=True,
    save_graph=True,
    filename="model_architecture",
    directory="images",
    graph_dir="LR"   # ← horizontal
)

print("Architecture enregistrée dans images/model_architecture.png")