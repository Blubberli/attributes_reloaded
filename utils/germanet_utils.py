

def get_germanet_attribute_mapping(path):
    """
    Return a mapping from germanet-specific to general attribute names
    :param path: the dictionary file that stores the mapping
    :return: the mapping from germanet-specific to general
    """
    w2attribut = {}
    for line in open(path):
        word = line.split("\t")[0]
        attribut = line.split("\t")[1].strip()
        w2attribut[word] = attribut
    return w2attribut
