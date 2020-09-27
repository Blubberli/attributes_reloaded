import argparse
import time
import json
from pathlib import Path
import logging.config
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange
from data.dataloader import extract_all_labels
from utils.training_utils import init_classifier, get_datasets
from utils.logger_config import create_config
from functions.loss_functions import get_loss_cosine_distance
from training_scripts.ranker import NearestNeigbourRanker


def train(config, train_loader, valid_loader, model_path, device, rank_loader):
    """
        method to pretrain a composition model
        :param config: config json file
        :param train_loader: dataloader torch object with training data
        :param valid_loader: dataloader torch object with validation data
        :return: the trained model
        """
    model = init_classifier(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    current_patience = 0
    tolerance = 0.005
    lowest_loss = float("inf")
    best_epoch = 1
    epoch = 1
    train_loss = 0.0
    max_f1 = 0
    early_stopping_criterion = config["validation_metric"]
    print(early_stopping_criterion)
    for epoch in range(1, config["num_epochs"] + 1):
        # training loop over all batches
        model.train()
        # these store the losses and accuracies for each batch for one epoch
        train_losses = []
        valid_losses = []
        # for word1, word2, labels in train_loader:
        pbar = trange(len(train_loader), desc='training...', leave=True)
        for batch in train_loader:
            pbar.update(1)
            batch["device"] = device
            out = model(batch).squeeze()
            out = out.to("cpu")
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["label_rep"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            # validation loop over validation batches
            model.eval()
        pbar.close()
        pbar = trange(len(train_loader), desc='validation...', leave=True)
        predictions = []
        for batch in valid_loader:
            pbar.update(1)
            batch["device"] = device
            out = model(batch).squeeze().to("cpu")
            for pred in out:
                predictions.append(pred.detach().numpy())
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["label_rep"])
            valid_losses.append(loss.item())

        predictions = np.array(predictions)
        save_predictions(predictions=predictions, path=prediction_path_dev)
        ranker_attribute = NearestNeigbourRanker(path_to_predictions=prediction_path_dev,
                                                 embedding_extractor=dataset_valid.label_extractor,
                                                 data_loader=rank_loader,
                                                 all_labels=labels,
                                                 y_label="label", max_rank=1000)
        f_macro = ranker_attribute.f1_macro
        pbar.close()

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        # stop when f1 score is the highest
        if early_stopping_criterion == "f1":
            if f_macro > max_f1 - tolerance:
                lowest_loss = valid_loss
                max_f1 = f_macro
                best_epoch = epoch
                current_patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                current_patience += 1
            if current_patience > config["patience"]:
                break
        # stop when loss is the lowest
        else:
            if lowest_loss - valid_loss > tolerance:
                lowest_loss = valid_loss
                best_epoch = epoch
                current_patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                current_patience += 1
            if current_patience > config["patience"]:
                break

        logger.info("current patience: %d , epoch %d , train loss: %.5f, validation loss: %.5f, validation f1 : %.2f" %
                    (current_patience, epoch, train_loss, valid_loss, f_macro))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f" %
        (epoch, train_loss, best_epoch, lowest_loss))


def predict(test_loader, model, device):
    """
    predicts labels on unseen data (test set)
    :param test_loader: dataloader torch object with test data
    :param model: trained model
    :param config: config: config json file
    :return: predictions for the given dataset, the loss and accuracy over the whole dataset
    """
    test_loss = []
    predictions = []
    orig_phrases = []
    model.to(device)
    pbar = trange(len(test_loader), desc='predict...', leave=True)
    for batch in test_loader:
        pbar.update(1)
        batch["device"] = device
        out = model(batch).squeeze().to("cpu")
        for pred in out:
            predictions.append(pred.detach().numpy())
        loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["label_rep"])
        test_loss.append(loss.item())
        orig_phrases.append(batch["label"])
    pbar.close()
    orig_phrases = [item for sublist in orig_phrases for item in sublist]
    predictions = np.array(predictions)
    return predictions, np.average(test_loss), orig_phrases


def save_predictions(predictions, path):
    np.save(file=path, arr=predictions, allow_pickle=True)


if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp = argp.parse_args()

    with open(argp.path_to_config, 'r') as f:  # read in arguments and save them into a configuration object
        config = json.load(f)

    ts = time.gmtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if name is specified choose specified name to save logging file, else use default name
    if config["save_name"] == "":
        save_name = format(
            "%s_%s" % (config["model"]["type"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names
    else:
        save_name = format("%s_%s" % (config["save_name"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names

    log_file = str(Path(config["logging_path"]).joinpath(save_name + "_log.txt"))  # change location
    model_path = str(Path(config["model_path"]).joinpath(save_name))
    prediction_path_dev = str(Path(config["model_path"]).joinpath(save_name + "_dev_predictions.npy"))
    prediction_path_test = str(Path(config["model_path"]).joinpath(save_name + "_test_predictions.npy"))
    rank_path_dev = str(Path(config["model_path"]).joinpath(save_name + "_dev_ranks.txt"))
    rank_path_test = str(Path(config["model_path"]).joinpath(save_name + "_test_ranks.txt"))

    logging.config.dictConfig(create_config(log_file))
    logger = logging.getLogger("train")
    logger.info("Training %s model. \n Logging to %s \n Save model to %s" % (
        config["model"]["type"], log_file, model_path))

    # set random seed
    np.random.seed(config["seed"])

    # read in data...
    dataset_train, dataset_valid, dataset_test = get_datasets(config)

    # load data with torch Data Loader
    train_loader = DataLoader(dataset_train,
                              batch_size=config["iterator"]["batch_size"],
                              shuffle=True,
                              num_workers=0)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=config["iterator"]["batch_size"],
                                               shuffle=False,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=config["iterator"]["batch_size"],
                                              shuffle=False,
                                              num_workers=0)

    model = None

    logger.info("%d training batches" % config["iterator"]["batch_size"])
    logger.info("the training data contains %d words" % len(dataset_train))
    logger.info("the validation data contains %d words" % len(dataset_valid))
    logger.info("the test data contains %d words" % len(dataset_test))
    logger.info("training with the following parameter")
    logger.info(config)

    labels = extract_all_labels(training_data=config["train_data_path"],
                                validation_data=config["validation_data_path"],
                                test_data=config["test_data_path"],
                                separator=config["data_loader"]["sep"]
                                , label=config["data_loader"]["label"])

    # train
    rank_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=len(dataset_valid), num_workers=0)

    train(config, train_loader, valid_loader, model_path, device, rank_loader)
    # test and & evaluate
    logger.info("Loading best model from %s", model_path)
    valid_model = init_classifier(config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()

    if valid_model:
        logger.info("generating predictions for validation data...")
        valid_predictions, valid_loss, valid_phrases = predict(valid_loader, valid_model, device)
        save_predictions(predictions=valid_predictions, path=prediction_path_dev)
        logger.info("saved predictions to %s" % prediction_path_dev)
        logger.info("validation loss: %.5f" % (valid_loss))
        rank_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=len(dataset_valid), num_workers=0)
        ranker_attribute = NearestNeigbourRanker(path_to_predictions=prediction_path_dev,
                                                 embedding_extractor=dataset_valid.label_extractor,
                                                 data_loader=rank_loader,
                                                 all_labels=labels,
                                                 y_label="label", max_rank=1000)
        ranker_attribute.save_ranks(rank_path_dev)

        logger.info("result for learned attribute representation")
        logger.info(
            "precision at rank 1: %.2f; precision at rank 5 %.2f" % (ranker_attribute._map_1, ranker_attribute._map_5))
        logger.info("accuracy: %.2f; f1 score macro: %.2f, f1 score weighted : %.2f" % (
        ranker_attribute.accuracy, ranker_attribute.f1_macro, ranker_attribute.f1_weighted))

        logger.info("saved ranks to %s" % rank_path_dev)
        if config["eval_on_test"]:
            logger.info("generating predictions for test data...")
            test_predictions, test_loss, test_phrases = predict(test_loader, valid_model, device)
            save_predictions(predictions=test_predictions, path=prediction_path_test)
            logger.info("saved predictions to %s" % prediction_path_test)
            logger.info("test loss: %.5f" % (test_loss))
            rank_loader = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), num_workers=0)
            ranker_attribute = NearestNeigbourRanker(path_to_predictions=prediction_path_test,
                                                     embedding_extractor=dataset_test.label_extractor,
                                                     data_loader=rank_loader,
                                                     all_labels=labels,
                                                     y_label="label", max_rank=1000)
            ranker_attribute.save_ranks(rank_path_dev)

            logger.info("result for learned attribute representation")
            logger.info(
                "precision at rank 1: %.2f; precision at rank 5 %.2f" % (
                    ranker_attribute._map_1, ranker_attribute._map_5))
            logger.info("accuracy: %.2f; f1 score macro: %.2f, f1 score weighted : %.2f" % (
                ranker_attribute.accuracy, ranker_attribute.f1_macro, ranker_attribute.f1_weighted))

            logger.info("saved ranks to %s" % rank_path_test)
    else:
        logging.error("model could not been loaded correctly")
