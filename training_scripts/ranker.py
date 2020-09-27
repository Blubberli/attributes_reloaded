import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score


class NearestNeigbourRanker:

    def __init__(self, path_to_predictions, embedding_extractor, data_loader, all_labels, max_rank, y_label):
        """
        This class stores the functionality to rank label representations with respect to a predicted representation and
        to compute the precision at certain ranks or use the closes predicted label as final prediction.
        :param path_to_predictions: [String] The path to numpy array stored predictions (number_of_test_instances x
        embedding_dim)
        :param path_to_ranks: [String] the path were the computed ranks will be saved to
        :param embedding_path: [String] the path to the embeddings
        :param data_loader: [Dataloader] a data loader witch batchsize 1 that holds the test data
        :param all_labels: [list(String)] a list of all unique labels that can occur in the test data
        :param max_rank: [int] the worst possible rank (even if an instance would get a lower rank it is set to this
        number)
        :param y_label: [String] the column name of the label in the test data
        this has
        to be 'label', for the StaticRankingDataset this has to be 'phrase'
        """
        # load composed predictions
        self._predicted_embeddings = np.load(path_to_predictions, allow_pickle=True)
        self._embeddings = embedding_extractor
        self._data = next(iter(data_loader))
        self._data_loder = data_loader
        # the correct labels are stored here
        self._true_labels = self.data[y_label]
        self._max_rank = max_rank

        # construct label embedding matrix, embeddings of labels are looked up in the original embeddings
        all_labels = sorted(all_labels)

        self._label2index = dict(zip(all_labels, range(len(all_labels))))
        self._index2label = dict(zip(range(len(all_labels)), all_labels))
        # normalize predictions and label embedding matrix (in case they are not normalized)
        self._label_embeddings = self._embeddings.get_array_embeddings(all_labels)
        self._label_embeddings = F.normalize(torch.from_numpy(np.array(self._label_embeddings)), p=2, dim=1)
        self._predicted_embeddings = F.normalize(torch.from_numpy(np.array(self._predicted_embeddings)), p=2,
                                                 dim=1)
        # compute the ranks, quartiles and precision
        self._ranks, self._composed_similarities, self._predicted_labels = self.get_target_based_rank()
        self._map_1 = self.precision_at_rank(1, self.ranks)
        self._map_5 = self.precision_at_rank(5, self.ranks)
        self._map_3 = self.precision_at_rank(3, self.ranks)
        self._accuracy, self._f1_macro, self._f1_weighted = self.performance_metrics()
        self._average_similarity = np.average(self.composed_similarities)

    def get_target_based_rank(self):
        """
        Computes the ranks of the composed representations, given a matrix of gold standard label embeddings.
        The ordering is relative to the composed representation, the attribute at rank 1 is the nearest neighbour of the
        composed representation.
        :return: a list with the ranks for all the composed representations in the batch, a list of similarities
        between
        correct label vector and predicted one, a list of predicted labels
        """
        all_ranks = []
        composed_similarities = []
        predicted_labels = []
        # get the index for each label in the true labels
        target_idxs = [self.label2index[label] for label in self.true_labels]

        # get a matrix, each row representing the gold representation of the corresponding label
        target_repr = np.take(self.label_embeddings, target_idxs, axis=0)

        for i in range(self._predicted_embeddings.shape[0]):
            # compute similarity between the predicted vector and all possible label vectors
            composed_attributes_similarity = np.dot(self.label_embeddings, np.transpose(self.predicted_embeddings[i]))
            predicted_attribute = self.index2label[np.argmax(composed_attributes_similarity)]
            predicted_labels.append(predicted_attribute)
            composed_target_similarity = np.dot(target_repr[i], self.predicted_embeddings[i])
            composed_similarities.append(composed_target_similarity)

            # the rank is the number of vectors with greater similarity that the one between
            # the target representation and the composed one; no sorting is required, just
            # the number of elements that are more similar
            rank = np.count_nonzero(composed_attributes_similarity > composed_target_similarity) + 1
            if rank > self.max_rank:
                rank = self.max_rank
            all_ranks.append(rank)

        return all_ranks, composed_similarities, predicted_labels

    def save_ranks(self, file_to_save):
        with open(file_to_save, "w", encoding="utf8") as f:
            for i in range(len(self._true_labels)):
                f.write(self.true_labels[i] + " " + str(self.ranks[i]) + "\n")
        print("ranks saved to file: " + file_to_save)

    @staticmethod
    def precision_at_rank(k, ranks):
        """
        Computes the number of times a rank is equal or lower to a given rank.
        :param k: the rank for which the precision is computed
        :param ranks: a list of ranks
        :return: the precision at a certain rank (float)
        """
        assert k >= 1
        correct = len([rank for rank in ranks if rank <= k])
        return correct / len(ranks)

    def performance_metrics(self):
        """
        Computes the f1 score and accuracy for the test set.
        :return: accuracy, f1 score macro and f1 score weighted
        """
        f1_macro = f1_score(y_true=self.true_labels, y_pred=self.predicted_labels, average="macro",
                            labels=list(set(self.true_labels)))
        f1_weighted = f1_score(y_true=self.true_labels, y_pred=self.predicted_labels, average="weighted")
        acc = accuracy_score(y_true=self.true_labels, y_pred=self.predicted_labels)
        return acc, f1_macro, f1_weighted

    @property
    def predicted_embeddings(self):
        return self._predicted_embeddings

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def true_labels(self):
        return self._true_labels

    @property
    def max_rank(self):
        return self._max_rank

    @property
    def label_embeddings(self):
        return self._label_embeddings

    @property
    def label2index(self):
        return self._label2index

    @property
    def ranks(self):
        return self._ranks

    @property
    def index2label(self):
        return self._index2label

    @property
    def composed_similarities(self):
        return self._composed_similarities

    @property
    def predicted_labels(self):
        return self._predicted_labels

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def f1_macro(self):
        return self._f1_macro

    @property
    def f1_weighted(self):
        return self._f1_weighted

    @property
    def average_similarity(self):
        return self._average_similarity

    @property
    def data(self):
        return self._data

    @property
    def data_loader(self):
        return self._data_loder
