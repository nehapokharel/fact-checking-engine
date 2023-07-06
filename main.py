from data_processing import parse_rdf_to_graph, generate_node2vec_embeddings, generate_train_data
from model import train_and_evaluate_model
from output import generate_result_file

def main():
    # Parse the RDF file into a graph and a dictionary of facts
    graph, facts = parse_rdf_to_graph("fokgtrain.nt")

    # Generate node embeddings using Node2Vec
    embeddings_model = generate_node2vec_embeddings(graph)

    # Generate training data
    X_train, y_train = generate_train_data(graph, facts, embeddings_model)

    # Train and evaluate the model
    model, scaler = train_and_evaluate_model(X_train, y_train)

    # Generate a result file
    generate_result_file(facts, graph, embeddings_model, scaler, model, 'result.ttl')


if __name__ == "__main__":
    main()
