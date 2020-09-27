import torch
from torch import nn
from torch.functional import F
from functions.composition_functions import transweigh, concat


class TransweighJoint(nn.Module):
    """
    This class contains a model that takes two word representations as input and combines them via
    transformations. The transformed, composed vector can be optimized compared to a gold representation.
    The composition model can thus be trained on constructing a gold representation of a two-word phrase that reveals
    some sort of attribute or relation (e.g. attribute, semantic class).
    """

    def __init__(self, input_dim, dropout_rate, transformations, normalize_embeddings):
        super().__init__()
        # --- we create the following variable that will be trainable parameters for our classifier:

        # the transformation tensor for transforming the input vectors
        self._transformation_tensor = nn.Parameter(
            torch.empty(transformations, 2 * input_dim, input_dim), requires_grad=True)
        nn.init.xavier_normal_(self.transformation_tensor)
        self._transformation_bias = nn.Parameter(torch.empty(transformations, input_dim), requires_grad=True)
        nn.init.uniform_(self.transformation_bias)

        # - the combining tensor for representation 1
        self._combining_tensor_1 = nn.Parameter(data=torch.empty(transformations, input_dim, input_dim),
                                                requires_grad=True)
        nn.init.xavier_normal_(self.combining_tensor_1)
        self._combining_bias_1 = nn.Parameter(torch.empty(input_dim), requires_grad=True)
        nn.init.uniform_(self.combining_bias_1)

        # - the combining tensor for representation 2
        self._combining_tensor_2 = nn.Parameter(data=torch.empty(transformations, input_dim, input_dim),
                                                requires_grad=True)
        nn.init.xavier_normal_(self.combining_tensor_2)
        self._combining_bias_2 = nn.Parameter(torch.empty(input_dim), requires_grad=True)
        nn.init.uniform_(self.combining_bias_2)

        # - the final layer to combine representation 1 and 2
        self._matrix_layer = nn.Linear(input_dim * 2, input_dim)
        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, batch):
        """
        Composes the input vectors into one representation.
        :param batch: batch with the representation of the first and second word (torch tensor)
        :return: the composed representation, same shape as each input representation
        """
        self._composed_phrase_1, self._composed_phrase_2 = self.compose(batch["modifier_rep"], batch["head_rep"])
        self._final_composed = self._matrix_layer(concat(self._composed_phrase_1, self._composed_phrase_2, axis=1))
        if self.normalize_embeddings:
            self._final_composed = F.normalize(self._final_composed, p=2, dim=1)

        return self._composed_phrase_1, self._composed_phrase_2, self._final_composed

    def compose(self, word1, word2):
        """
        This functions composes two input representations with the transformation weighting model. If set to True,
        the composed representation is normalized
        :param word1: the representation of the first word (torch tensor)
        :param word2: the representation of the second word (torch tensor)
        :param training: True if the model should be trained, False if the model is in inference
        :return: the composed vector representation, eventually normalized to unit norm
        """
        composed_phrase_1 = transweigh(word1=word1, word2=word2, transformation_tensor=self.transformation_tensor,
                                       transformation_bias=self.transformation_bias,
                                       combining_bias=self._combining_bias_1,
                                       combining_tensor=self._combining_tensor_1, dropout_rate=self.dropout_rate,
                                       training=self.training)
        composed_phrase_2 = transweigh(word1=word1, word2=word2, transformation_tensor=self.transformation_tensor,
                                       transformation_bias=self.transformation_bias,
                                       combining_bias=self._combining_bias_2,
                                       combining_tensor=self._combining_tensor_2, dropout_rate=self.dropout_rate,
                                       training=self.training)
        if self.normalize_embeddings:
            composed_phrase_1 = F.normalize(composed_phrase_1, p=2, dim=1)
            composed_phrase_2 = F.normalize(composed_phrase_2, p=2, dim=1)

        return composed_phrase_1, composed_phrase_2

    @property
    def combining_tensor_1(self):
        return self._combining_tensor_1

    @property
    def combining_bias_1(self):
        return self._combining_bias_1

    @property
    def combining_tensor_2(self):
        return self._combining_tensor_2

    @property
    def combining_bias_2(self):
        return self._combining_bias_2

    @property
    def transformation_tensor(self):
        return self._transformation_tensor

    @property
    def transformation_bias(self):
        return self._transformation_bias

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def normalize_embeddings(self):
        return self._normalize_embeddings

    @property
    def composed_phrase_1(self):
        return self._composed_phrase_1

    @property
    def composed_phrase_2(self):
        return self._composed_phrase_2

    @property
    def final_composed(self):
        return self._final_composed
