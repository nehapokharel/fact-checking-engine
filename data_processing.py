import rdflib
from node2vec import Node2Vec
import networkx as nx
import numpy as np

def parse_rdf_to_graph(file_path):
    """Parses an RDF file into a NetworkX graph and a dictionary of facts."""
    # Load the training data
    g = rdflib.Graph()
    g.parse(file_path, format="turtle")

    # Map each fact to a subject, predicate, object, and truth value
    facts = {}
    for stmt in g:
        fact_id, prop, obj = stmt
        fact_id = str(fact_id)  # Convert fact_id to string
        if fact_id not in facts:
            facts[fact_id] = {}
        if prop == rdflib.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
            continue
        elif prop == rdflib.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#subject'):
            facts[fact_id]['subject'] = str(obj)
        elif prop == rdflib.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate'):
            facts[fact_id]['predicate'] = str(obj)
        elif prop == rdflib.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#object'):
            facts[fact_id]['object'] = str(obj)
        elif prop == rdflib.URIRef('http://swc2017.aksw.org/hasTruthValue'):
            facts[fact_id]['truth_value'] = float(obj)

    # Create a NetworkX graph
    graph = nx.Graph()

    # Add edges to the graph with predicates
    for fact_id, fact in facts.items():
        subject = fact.get('subject')
        object = fact.get('object')
        predicate = fact.get('predicate')
        if subject and object and predicate:
            graph.add_edge(subject, object, predicate=predicate)
    return graph, facts

def generate_node2vec_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4):
    """
    Generate Node2Vec embeddings for the nodes in the graph.
    """
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Use the Word2Vec function
    return model

def generate_train_data(graph, facts, embeddings_model):
    """Generates training data using the node embeddings."""
    X_train = []
    y_train = []

    for fact_id, fact in facts.items():
        subject = fact.get('subject')
        object = fact.get('object')
        if subject and object and nx.has_path(graph, subject, object):
            # Use the generated embeddings as features
            vector = np.concatenate((embeddings_model.wv[subject], embeddings_model.wv[object]))
            X_train.append(vector)
            y_train.append(fact['truth_value'])

    return X_train, y_train

def fact_to_vector(subject, object, embeddings_model):
    """Converts a fact to a feature vector using the node embeddings."""
    if subject in embeddings_model.wv and object in embeddings_model.wv:
        return np.concatenate((embeddings_model.wv[subject], embeddings_model.wv[object]))
    return None

