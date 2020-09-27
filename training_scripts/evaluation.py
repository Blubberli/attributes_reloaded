import numpy as np
from collections import defaultdict, Counter
import torch
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import argparse
import json
from torch.utils.data import DataLoader
from data.dataloader import RankingDataset
from data.dataloader import extract_all_labels
from training_scripts.ranker import NearestNeigbourRanker
from utils.training_utils import init_classifier
from training_scripts.train_simple_ranking import predict, save_predictions
from sklearn.metrics import classification_report
import re


def get_ranks_per_attribute(ranker, save_path):
    """
    Given a ranker and a path, compute the rank measures for each label
    :param ranker: a ranker that contains the predicted labels and the true labels
    :param save_path: a path to store the results at
    """
    attribute2ranks = defaultdict(list)
    for i in range(len(ranker.true_labels)):
        attribute2ranks[ranker.true_labels[i]].append(ranker.ranks[i])
    file = open(save_path + "_attribute_ranks.csv", "w")
    for attribute, ranks in attribute2ranks.items():
        p1 = NearestNeigbourRanker.precision_at_rank(k=1, ranks=ranks)
        p3 = NearestNeigbourRanker.precision_at_rank(k=3, ranks=ranks)
        file.write("%s\t%.2f\t%.2f" % (attribute, p1, p3))
        file.write("\n")


def performance_per_attribute(ranker, save_path):
    """
    Given a ranker for a test dataset and a path create a classification report, each attribute will have the score
    and the averaged micro and weighted F1 will also be reported
    """
    gold_labels = ranker.true_labels
    predicted_labels = ranker.predicted_labels
    labels_in_test = list(set(gold_labels))
    report = classification_report(y_true=gold_labels, y_pred=predicted_labels, labels=labels_in_test, output_dict=True,
                                   zero_division=0)
    pd.DataFrame(report).round(decimals=3).transpose().to_csv(save_path + "_classification_report.csv", sep="\t")
    get_ranks_per_attribute(ranker, save_path)


def performance_per_phrase_type(ranker, testset, save_path):
    phrase_type = testset.status
    predictions_free = [ranker.predicted_labels[i] for i in range(len(phrase_type)) if phrase_type[i] == "free"]
    predictions_colloc = [ranker.predicted_labels[i] for i in range(len(phrase_type)) if
                          phrase_type[i] == "collocation"]
    gold_labels_free = [ranker.true_labels[i] for i in range(len(phrase_type)) if phrase_type[i] == "free"]
    gold_labels_colloc = [ranker.true_labels[i] for i in range(len(phrase_type)) if phrase_type[i] == "collocation"]
    report_free = classification_report(y_true=gold_labels_free, y_pred=predictions_free,
                                        labels=list(set(gold_labels_free)),
                                        output_dict=True, zero_division=0)
    report_colloc = classification_report(y_true=gold_labels_colloc, y_pred=predictions_colloc,
                                          labels=list(set(gold_labels_colloc)),
                                          output_dict=True, zero_division=0)
    pd.DataFrame(report_free).round(decimals=3).transpose().to_csv(save_path + "freephrase_classification_report.csv",
                                                                   sep="\t")
    pd.DataFrame(report_colloc).round(decimals=3).transpose().to_csv(
        save_path + "collocations_classification_report.csv",
        sep="\t")


def get_nearest_neighbours_for_given_list(vector, vector_list, index2label):
    """
    Given a vector and a list of other vectors, each vectors index in that list corresponding to a real word that is
    defined with the dictionary "index2label", compute the similarities between the given vector and each word and
    return these in a sorted dictionary
    :param vector: a vector (e.g. a predicted representation for an adj-noun pair)
    :param vector_list: a list of vectors [vector1, vector2, vector2]
    :param index2label: a dictionary that maps the list of vectors to a word, e.g. if 0 = Haus, 1 is Liebe and 2 is
    Wahrheit, then vector1 = Haus, vector2 = Liebe, vector3 = Wahrheit
    :return: sorted List of tuples. Each tuple contains the word and the similarity that was computed between the
    given vector and that word. Sorted from most similar to lease similar words
    """
    vec2label_sim = np.dot(vector, vector_list.transpose())
    vec2label_sim = dict(zip([v for k, v in index2label.items()], vec2label_sim))
    vec2label_sim = sorted(vec2label_sim.items(), key=lambda kv: kv[1])
    vec2label_sim.reverse()
    return vec2label_sim


def nearest_neighbours_static(predicted_vectors, feature_extractor, dataset, all_labels, save_name, true_labels):
    """
    For all predicted vectors for a given dataset extract the nearest neighbours from
    a) the whole embedding space
    b) a list of labels
    :param predicted_vectors: a numpy array of predicted representations for a given dataset
    :param feature_extractor: a static feature extractor
    :param dataset: a single dataset that contains modifier, head and label
    :param all_labels:
    :return:
    """
    modifier = dataset.modifier_words
    heads = dataset.head_words
    label_embeddings = np.array(feature_extractor.get_array_embeddings(all_labels))
    index2label = dict(zip(range(len(all_labels)), all_labels))
    f = open(save_name + "_nearest_neighbours.txt", "w")
    label2closest_labels = defaultdict(list)
    for i in range(predicted_vectors.shape[0]):
        vec = predicted_vectors[i]
        # label = "unknown"
        label = true_labels[i]
        phrase = modifier[i] + " " + heads[i]
        vec2label_sim = get_nearest_neighbours_for_given_list(vec, label_embeddings, index2label)

        top_labels = [el[0] for el in vec2label_sim]
        for el in top_labels[:5]:
            # if el != label:
            label2closest_labels[label].append(el)
        general_neighbours = feature_extractor.embeds.embedding_similarity(vec)
        # s = "phrase: %s \n correct label: %s\n top predicted labels: %s \n general close words: %s\n" % (
        #    phrase, label, str(top_labels[:5]), str(general_neighbours[:5]))
        s = phrase + "\t" + str(top_labels[:5]) + "\t" + str(general_neighbours[:5]) + "\n"
        f.write(s)
    f.close()


def performance_per_adjective(ranker, save_path):
    adj2true_label = {}
    adj2predicted_label = {}
    modifier = list(ranker.data_loader.dataset._modifier_words)
    for i in range(len(ranker.true_labels)):
        adj = modifier[i]
        true_label = ranker.true_labels[i]
        predicted_label = ranker.predicted_labels[i]
        if adj not in adj2true_label:
            adj2true_label[adj] = [true_label]
            adj2predicted_label[adj] = [predicted_label]
        else:
            adj2true_label[adj].append(true_label)
            adj2predicted_label[adj].append(predicted_label)
    f = open(save_path + "_adj2accuracy.csv", "w")
    for adj, true_labels in adj2true_label.items():
        acc = accuracy_score(y_pred=adj2predicted_label[adj], y_true=true_labels)
        f.write("%s\t%.2f\n" % (adj, acc))
    f.close()


def save_results(ranker, path):
    """
    :param ranker: an object of a NearestNeighbourRanker
    :param path: where to save results
    """
    split = re.search("split\d", path).group(0)
    general_path = path.replace(split, "") + split + "result.txt"

    with open(path, 'w') as file:
        file.write("precision at rank 1:  {:0.2f} \n".format(ranker._map_1))
        file.write("precision at rank 3:  {:0.2f} \n".format(ranker._map_3))
        file.write("accuracy: {:0.2f}; f1 score macro: {:0.2f}; f1 score weighted: {:0.2f}".format(ranker.accuracy,
                                                                                                   ranker.f1_macro,
                                                                                                   ranker.f1_weighted))
    with open(general_path, 'w') as file:
        file.write("precision at rank 1:  {:0.2f} \n".format(ranker._map_1))
        file.write("precision at rank 3:  {:0.2f} \n".format(ranker._map_3))
        file.write("accuracy: {:0.2f}; f1 score macro: {:0.2f}; f1 score weighted: {:0.2f}".format(ranker.accuracy,
                                                                                                   ranker.f1_macro,
                                                                                                   ranker.f1_weighted))


def get_dataset(config):
    """
    This method creates a single Dataset with modifier, head and label (embeddings).
    :param config: a configuration that specified the format of the dataset
    :param data_path: a path to a dataset that should be used to create predictions for
    :return: a RankingDataset
    """
    mod = config["data_loader"]["modifier"]
    head = config["data_loader"]["head"]
    label = config["data_loader"]["label"]
    dataset = RankingDataset(data_path=config["test_data_path"],
                             general_embedding_path=config["feature_extractor"]["general_embeddings"],
                             label_embedding_path=config["feature_extractor"]["label_embeddings"],
                             separator=config["data_loader"]["sep"],
                             label=label, mod=mod, head=head)
    return dataset


def save_labels(labels, save_path):
    f = open(save_path, "w")
    for label in labels:
        f.write(label + "\n")
    f.close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("config", help="the path to evaluation config")
    argp.add_argument("--n", help="if nearest neighbours should be created for every phrase", action="store_true")
    argp = argp.parse_args()

    # config mit static or contextualised
    with open(argp.config, 'r') as f:
        config = json.load(f)

    save_path = config["save_path"]

    dataset = get_dataset(config)
    # load test data in batches
    test_loader = DataLoader(dataset=dataset,
                             batch_size=len(dataset),
                             shuffle=False,
                             num_workers=0)
    model = init_classifier(config)
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()

    test_predictions, test_loss, test_phrases = predict(test_loader, model, device="cpu")
    save_predictions(test_predictions, save_path + "/test_predictions.npy")

    labels = extract_all_labels(training_data=config["train_data_path"],
                                validation_data=config["validation_data_path"],
                                test_data=config["test_data_path"],
                                separator=config["data_loader"]["sep"]
                                , label=config["data_loader"]["label"])
    print("number of labels : %d" % len(labels))

    ranker_attribute = NearestNeigbourRanker(path_to_predictions=save_path + "/test_predictions.npy",
                                             embedding_extractor=dataset.label_extractor,
                                             data_loader=test_loader,
                                             all_labels=labels,
                                             y_label="label", max_rank=48)
    ranker_attribute.save_ranks(save_path + "/test_ranks.txt")
    save_labels(np.array(ranker_attribute.true_labels), save_path + "/true_labels.txt")
    save_labels(np.array(ranker_attribute.predicted_labels), save_path + "/predicted_labels.txt")

    save_results(ranker_attribute, save_path + "/test_scores.txt")
    performance_per_attribute(ranker=ranker_attribute,
                              save_path=save_path + "/_test_")
    performance_per_phrase_type(ranker=ranker_attribute, save_path=save_path + "/test_", testset=dataset)

    performance_per_adjective(ranker=ranker_attribute, save_path=save_path + "/_test_")
    if argp.n:
        nearest_neighbours_static(predicted_vectors=ranker_attribute.predicted_embeddings,
                                  feature_extractor=dataset._general_extractor, all_labels=labels, dataset=dataset,
                                  save_name=save_path + "/_neigbours_", true_labels=ranker_attribute.true_labels)
