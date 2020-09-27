import pandas as pd
from torch.utils.data import Dataset
from data.feature_extractor import StaticEmbeddingExtractor


def extract_all_labels(training_data, validation_data, test_data, separator, label):
    """
    This method returns a list of all unique labels that occur in a dataset
    :param training_data: the training data with a column named 'label'
    :param validation_data: the validation data with a column named 'label'
    :param test_data: the test data with a column named 'label'
    :param separator: the separator of the column based data
    :param label: the column name that stores the labels
    :return: a list with all unique labels of the dataset
    """
    training_labels = set(pd.read_csv(training_data, delimiter=separator, index_col=False)[label])
    validation_labels = set(pd.read_csv(validation_data, delimiter=separator, index_col=False)[label])
    test_labels = set(pd.read_csv(test_data, delimiter=separator, index_col=False)[label])
    all_labels = list(training_labels.union(validation_labels).union(test_labels))
    return all_labels


class RankingDataset(Dataset):

    def __init__(self, data_path, general_embedding_path, label_embedding_path, separator, mod, head, label):
        """
        This datasets can be used to train a composition model on attribute selection
        :param data_path: the path to the dataset, should have a header
        :param general_embedding_path: the path to the pretrained static word embeddings to lookup the modifier and
        head words
        :param label_embedding_path: the path to the pretrained static word embeddings to lookup the label
        represenations
        :param separator: the separator within the dataset
        :param mod: the name of the column holding the modifier words
        :param head: the name of the column holding the head words
        :param label: the name of the column holding the labels (e.g. attribute)
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._labels = list(self.data[label])
        if "status" in self.data.columns:
            self._status = list(self.data["status"])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels), "invalid input data, different lenghts"

        self._general_extractor = StaticEmbeddingExtractor(path_to_embeddings=general_embedding_path)
        self._label_extractor = StaticEmbeddingExtractor(label_embedding_path)
        self._samples = self.populate_samples()

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and attributes and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and attribute embeddings (modifier_rep,
        head_rep, attribute_rep)
        """
        mod_embeddings = self.general_extractor.get_array_embeddings(self.modifier_words)
        head_embeddings = self.general_extractor.get_array_embeddings(self.head_words)
        label_embeddings = self.label_extractor.get_array_embeddings(self.labels)
        return [
            {"modifier_rep": mod_embeddings[i], "modifier": self.modifier_words[i], "head_rep": head_embeddings[i],
             "head": self.head_words[i], "label_rep": label_embeddings[i], "label": self.labels[i]}
            for i in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def labels(self):
        return self._labels

    @property
    def status(self):
        return self._status

    @property
    def general_extractor(self):
        return self._general_extractor

    @property
    def label_extractor(self):
        return self._label_extractor

    @property
    def samples(self):
        return self._samples


class JointRankingDataset(Dataset):

    def __init__(self, data_path, general_embedding_path, label_embedding_path, separator, mod, head, label_1, label_2):
        """
        This datasets can be used to train a composition model on attribute selection
        :param data_path: the path to the dataset, should have a header
        :param general_embedding_path: the path to the pretrained static word embeddings to lookup the modifier and
        head words
        :param label_embedding_path: the path to the pretrained static word embeddings to lookup the label
        represenations
        :param separator: the separator within the dataset
        :param mod: the name of the column holding the modifier words
        :param head: the name of the column holding the head words
        :param label: the name of the column holding the labels (e.g. attribute)
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._labels_1 = list(self.data[label_1])
        self._labels_2 = list(self.data[label_2])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels_1) == len(self.labels_2), "invalid input data, different lenghts"

        self._general_extractor = StaticEmbeddingExtractor(path_to_embeddings=general_embedding_path)
        self._label_extractor = StaticEmbeddingExtractor(label_embedding_path)
        self._samples = self.populate_samples()

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and attributes and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and attribute embeddings (modifier_rep,
        head_rep, attribute_rep)
        """
        mod_embeddings = self.general_extractor.get_array_embeddings(self.modifier_words)
        head_embeddings = self.general_extractor.get_array_embeddings(self.head_words)
        attribute_embeddings = self.label_extractor.get_array_embeddings(self.labels_1)
        semclass_embeddings = self.label_extractor.get_array_embeddings(self.labels_2)
        return [
            {"modifier_rep": mod_embeddings[i], "modifier": self.modifier_words[i], "head_rep": head_embeddings[i],
             "head": self.head_words[i], "attribute_rep": attribute_embeddings[i], "attribute": self.labels_1[i],
             "semclass_rep": semclass_embeddings[i], "semclass": self.labels_2[i]}
            for i in range(len(self.labels_1))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def labels_1(self):
        return self._labels_1

    @property
    def labels_2(self):
        return self._labels_2

    @property
    def general_extractor(self):
        return self._general_extractor

    @property
    def label_extractor(self):
        return self._label_extractor

    @property
    def samples(self):
        return self._samples
