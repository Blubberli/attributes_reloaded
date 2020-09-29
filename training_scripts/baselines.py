import numpy as np
import argparse
import json
from torch.utils.data import DataLoader
from data.dataloader import extract_all_labels
from training_scripts.ranker import NearestNeigbourRanker
from training_scripts.evaluation import get_dataset, performance_per_attribute, save_predictions


def predict(test_loader):
    data_all = next(iter(test_loader))
    averages = []
    adjectives = []
    nouns = []
    for m, h in zip(data_all["modifier_rep"], data_all["head_rep"]):
        averages.append(np.mean((np.array(m), np.array(h)), axis=0))
        adjectives.append(m)
        nouns.append(h)
    return np.array(adjectives), np.array(nouns), np.array(averages)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("config", help="the path to evaluation config")
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
    adjective_embeddings, noun_embeddings, phrase_embeddings = predict(test_loader)
    save_predictions(adjective_embeddings, save_path + "/test_adj_predictions.npy")
    save_predictions(noun_embeddings, save_path + "/test_noun_predictions.npy")
    save_predictions(phrase_embeddings, save_path + "/test_phrase_predictions.npy")

    labels = extract_all_labels(training_data=config["train_data_path"],
                                validation_data=config["validation_data_path"],
                                test_data=config["test_data_path"],
                                separator=config["data_loader"]["sep"]
                                , label=config["data_loader"]["label"])
    print("number of labels : %d" % len(labels))

    ranker_adjective = NearestNeigbourRanker(path_to_predictions=save_path + "/test_adj_predictions.npy",
                                             embedding_extractor=dataset.label_extractor,
                                             data_loader=test_loader,
                                             all_labels=labels,
                                             y_label="label", max_rank=48)
    ranker_noun = NearestNeigbourRanker(path_to_predictions=save_path + "/test_noun_predictions.npy",
                                        embedding_extractor=dataset.label_extractor,
                                        data_loader=test_loader,
                                        all_labels=labels,
                                        y_label="label", max_rank=48)
    ranker_phrase = NearestNeigbourRanker(path_to_predictions=save_path + "/test_phrase_predictions.npy",
                                          embedding_extractor=dataset.label_extractor,
                                          data_loader=test_loader,
                                          all_labels=labels,
                                          y_label="label", max_rank=48)

    performance_per_attribute(gold_labels=ranker_adjective.true_labels,
                              predicted_labels=ranker_adjective.predicted_labels, ranks=ranker_adjective.ranks,
                              target_rank=1, save_p=save_path + "/adj_attributes_rank1")
    performance_per_attribute(gold_labels=ranker_adjective.true_labels,
                              predicted_labels=ranker_adjective.predicted_labels, ranks=ranker_adjective.ranks,
                              target_rank=3, save_p=save_path + "/adj_attributes_rank3")

    performance_per_attribute(gold_labels=ranker_noun.true_labels,
                              predicted_labels=ranker_noun.predicted_labels, ranks=ranker_noun.ranks,
                              target_rank=1, save_p=save_path + "/noun_attributes_rank1")
    performance_per_attribute(gold_labels=ranker_noun.true_labels,
                              predicted_labels=ranker_noun.predicted_labels, ranks=ranker_noun.ranks,
                              target_rank=3, save_p=save_path + "/noun_attributes_rank3")

    performance_per_attribute(gold_labels=ranker_phrase.true_labels,
                              predicted_labels=ranker_phrase.predicted_labels, ranks=ranker_phrase.ranks,
                              target_rank=1, save_p=save_path + "/phrase_attributes_rank1")
    performance_per_attribute(gold_labels=ranker_phrase.true_labels,
                              predicted_labels=ranker_phrase.predicted_labels, ranks=ranker_phrase.ranks,
                              target_rank=3, save_p=save_path + "/phrase_attributes_rank3")
