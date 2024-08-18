import datetime
import random
import sys
import logging
import copy
import numpy as np
import torch
from utils import factory
from utils.data_manager_T import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    global experiment_dir, weights_dir

    lr_steps_str = list(map(lambda k: str(int(k)), args["lr_steps"]))
    experiment_name = '_'.join((args["dataset"], args["arch"],
                                ''.join(args["modality"]).lower(),
                                'lr' + str(args["lr"]),
                                'lr_st' + '_'.join(lr_steps_str),
                                'dr' + str(args["dropout"]),
                                'ep' + str(args["epochs"]),
                                'segs' + str(args["num_segments"]),
                                args["experiment_suffix"]))
    experiment_dir = os.path.join(experiment_name, datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
    weights_dir = os.path.join('metricsweights', experiment_dir)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    logs_name = "metricslog/{}/".format(experiment_dir)
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "metricslog/{}/{}_{}_{}_{}_{}_{}".format(
        experiment_dir,
        args["prefix"],
        args["seed"],
        args["model_name"],
        args["dataset"],
        args["init_cls"],
        args["increment"]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    _set_random()
    _set_device(args)
    print_args(args)

    model = factory.get_model(args["model_name"], args)

    # Freeze stream weights (leaves only fusion and classification trainable)
    if args["freeze"]:
        model._network.feature_extract_network.freeze()

    image_tmpl = {}
    for m in args["modality"]:
        # Prepare dictionaries containing image name templates for each modality
        if m == 'RGB':
            image_tmpl[m] = "img_{:06d}.jpg"

    data_manager = DataManager(model, image_tmpl, args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task(weights_dir)
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            # nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            # logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1)
    random.seed(1)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
