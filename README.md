# Graph-Coloring-using-Neural-Network

This project is about graph coloring using Neural Networks (NNs) applied to graph-structured data, specifically the Cora citation network dataset. The project implements a Graph Convolutional Network (GCN) model to predict node classes (colors) for graph coloring, with an emphasis on hyperparameter tuning, early stopping, and learning rate scheduling.

### Key Components of the Project:

1.  **Graph Data**:

    -   The project uses the **Cora dataset**, which is a citation network consisting of 2708 scientific papers (nodes) that are linked by citation relationships (edges).
    -   The nodes have feature vectors representing the content of the papers, and the edges represent citation links between them.
    -   Each node (paper) belongs to one of the predefined classes (or categories), and the task is to predict these classes using the graph structure and node features.
2.  **Model Architecture**:

    -   **GCN (Graph Convolutional Network)**:
        -   The core of the model is a two-layer GCN, implemented using `GCNConv` layers from PyTorch Geometric. These layers learn node representations based on both the node features and the graph structure (edges).
        -   The output layer uses **log-softmax** activation, which is appropriate for multi-class classification (coloring nodes in different categories).
    -   **GraphSAGE (Graph Sample and Aggregation)**:
        -   A second model variant, **GraphSAGE**, is implemented for comparison. GraphSAGE is a more advanced method for learning node embeddings by aggregating information from a node's neighbors.
3.  **Training Loop**:

    -   The training procedure involves optimizing the model using **cross-entropy loss** for multi-class classification and the **Adam optimizer**.
    -   The training loop includes functionality for **early stopping** to avoid overfitting. This is controlled by the `EarlyStopping` class, which monitors the validation accuracy and stops the training if the accuracy doesn't improve for a specified number of epochs.
    -   The **learning rate scheduler** (`ReduceLROnPlateau`) adjusts the learning rate based on the loss to prevent overshooting and improve convergence.
4.  **Hyperparameter Tuning**:

    -   The project uses **grid search** to experiment with different hyperparameter configurations such as learning rate, hidden dimension size, dropout rate, and the number of epochs.
    -   The grid search is done via the `ParameterGrid` from `sklearn.model_selection` to evaluate the model on different combinations of parameters and identify the best set of hyperparameters.
5.  **Evaluation**:

    -   After training, the model's performance is evaluated using accuracy on a test set (nodes that were not part of the training).
    -   The performance is also evaluated using **classification report** and **confusion matrix**, providing detailed metrics like precision, recall, and F1-score for each class (color).
6.  **Graph Visualization**:

    -   The final output is a visualization of the **Cora citation graph**, where nodes are colored according to the predicted class (color) for each paper.
    -   The graph is visualized using **NetworkX** and **matplotlib**. The nodes are positioned using a **spring layout**, which is a force-directed layout algorithm that spaces nodes based on their connectivity in the graph.
    -   The `pred` variable holds the predicted class (color) for each node, and these colors are used in the plot.
7.  **Model Saving and Loading**:

    -   The trained model is saved to a file using `torch.save()`, allowing it to be loaded later for inference or further training using `model.load_state_dict()`.

### High-Level Project Workflow:

1.  **Data Loading**: Load the Cora dataset and inspect its structure, including the number of nodes, edges, and features.
2.  **Model Definition**: Define the GCN and GraphSAGE models using PyTorch and PyTorch Geometric.
3.  **Training and Hyperparameter Tuning**: Train the model with different hyperparameter configurations using grid search, learning rate scheduling, and early stopping.
4.  **Evaluation**: After training, evaluate the model's performance using accuracy, classification metrics, and confusion matrix.
5.  **Visualization**: Visualize the graph with node colors based on the predicted classes.

### GitHub Project Explanation:

This repository likely contains the code for the above steps, and each script might be structured as follows:

-   **Data loading and exploration**: Load and inspect the Cora dataset.
-   **Model definition**: Define the GCN and GraphSAGE models and other necessary components (e.g., optimizer, loss function, etc.).
-   **Training**: Implement the training loop, validation, early stopping, and learning rate scheduling.
-   **Evaluation**: Compute and print the evaluation metrics and confusion matrix.
-   **Visualization**: Generate a visualization of the graph, showing the node classifications (colors).
-   **Hyperparameter search**: Implement grid search for hyperparameter optimization.

This project is a good example of applying deep learning models to graph-structured data and can be extended to other graph coloring problems or graph classification tasks.
