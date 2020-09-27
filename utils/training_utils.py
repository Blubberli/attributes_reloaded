from models.transweigh import Transweigh
from models.transweigh_joint import TransweighJoint
from models.full_additive import FullAdditive
from data.dataloader import RankingDataset, JointRankingDataset


def init_classifier(config):
    """
    This method initialized the classifier with parameter specified in the config file
    :param config: the configuration
    :return: a torch classifier
    """
    classifier = None
    if config["model"]["type"] == "tw_single":
        classifier = Transweigh(input_dim=config["model"]["input_dim"],
                                dropout_rate=config["model"]["dropout"],
                                transformations=config["model"]["transformations"],
                                normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "tw_joint":
        classifier = TransweighJoint(input_dim=config["model"]["input_dim"],
                                     dropout_rate=config["model"]["dropout"],
                                     transformations=config["model"]["transformations"],
                                     normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "full_additive":
        classifier = FullAdditive(input_dim=config["model"]["input_dim"],
                                  normalize_embeddings=config["model"]["normalize_embeddings"])

    assert classifier, "no valid classifier name specified in the configuration"
    return classifier


def get_datasets(config):
    """
    Returns the datasets with the corresponding features (defined in the config file)
    :param config: the configuration file
    :return: training, validation, test dataset
    """
    mod = config["data_loader"]["modifier"]
    head = config["data_loader"]["head"]
    if config["model"]["type"] == "tw_joint":
        label_1 = config["data_loader"]["label_1"]
        label_2 = config["data_loader"]["label_2"]
        dataset_train = JointRankingDataset(data_path=config["train_data_path"],
                                            general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                            label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                            separator=config["data_loader"]["sep"],
                                            label_1=label_1, label_2=label_2, mod=mod, head=head)
        dataset_valid = JointRankingDataset(data_path=config["validation_data_path"],
                                            general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                            label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                            separator=config["data_loader"]["sep"],
                                            label_1=label_1, label_2=label_2, mod=mod, head=head)
        dataset_test = JointRankingDataset(data_path=config["test_data_path"],
                                           general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                           label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                           separator=config["data_loader"]["sep"],
                                           label_1=label_1, label_2=label_2, mod=mod, head=head)
    else:
        label = config["data_loader"]["label"]

        dataset_train = RankingDataset(data_path=config["train_data_path"],
                                       general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                       label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                       separator=config["data_loader"]["sep"],
                                       label=label, mod=mod, head=head)
        dataset_valid = RankingDataset(data_path=config["validation_data_path"],
                                       general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                       label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                       separator=config["data_loader"]["sep"],
                                       label=label, mod=mod, head=head)
        dataset_test = RankingDataset(data_path=config["test_data_path"],
                                      general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                      label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                      separator=config["data_loader"]["sep"],
                                      label=label, mod=mod, head=head)

    return dataset_train, dataset_valid, dataset_test
