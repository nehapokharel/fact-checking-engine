import rdflib
from model import check_fact_veracity

def write_fact_veracity_to_file(fact_id, veracity, result_file):
    """Writes the veracity of a fact to a file."""
    fact_iri = rdflib.URIRef(fact_id)
    predicate = rdflib.URIRef('http://swc2017.aksw.org/hasTruthValue')
    value = rdflib.Literal(str(veracity), datatype=rdflib.URIRef('http://www.w3.org/2001/XMLSchema#double'))

    triple = (fact_iri, predicate, value)
    result_file.write(triple[0].n3() + ' ' + triple[1].n3() + ' ' + triple[2].n3() + ' .\n')

def generate_result_file(facts, graph, embeddings_model, scaler, model, result_file_path):
    """Generates a result file containing the veracities of all facts."""
    # Open the result file for writing
    result_file = open(result_file_path, 'w')

    # Iterate over the facts in the training dataset
    for fact_id, fact in facts.items():
        fact_subject = fact['subject']
        fact_predicate = fact['predicate']
        fact_object = fact['object']

        # Perform fact-checking for each fact
        veracity = check_fact_veracity(fact_subject, fact_predicate, fact_object, graph, embeddings_model, scaler, model)

        # Write fact veracity to result file
        if veracity is not None:
            write_fact_veracity_to_file(fact_id, veracity, result_file)

    # Close the result file
    result_file.close()

