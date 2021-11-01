def create_embedding(categorical, len_categorical, embedding_dim):
    """This function returns a 3d tuple:
        1. a string with the entity embeddings layers
        2. a string with the embedding inputs
        3. a string with the embedding outputs
    
    Args:
        - categorical (list): list of categorical columns
        - len_categorical (dict): dict mapping categorical to the length of unique values in that column
        - embedding_dir (dict): dict mapping categorical to the embedding dimension
    """
    embedding = ""
    inputs = ""
    outputs = ""
    for _l in categorical:
        embedding += "input_{0} = Input(shape=(1,), name=\"{0}\")\n".format(_l)
        embedding += "output_{0} = Embedding(len_categorical[\"{0}\"], embedding_dim[\"{0}\"], name=\"{0}_embedding\", embeddings_constraint=unit_norm(axis=0))(input_{0})\n".format(_l)
        embedding += "output_{0} = Reshape(target_shape=(embedding_dim[\"{0}\"],))(output_{0})\n\n".format(_l)
        inputs += "input_{0}, \n".format(_l)
        outputs += "output_{0}, \n".format(_l)
    return embedding, inputs[:-3], outputs[:-3]
