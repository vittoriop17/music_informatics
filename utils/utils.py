import argparse
import json
import os
import numpy as np
import torch


def upload_args(file_path=os.path.join("configuration.json")):
    """
    The function loads ALL the parameters passed through COMMAND LINE and also those reported in the configuration file.
    The configuration file is a json.
    The parameters passed through command line have more importance than those reported in the json.
    In case one argument is found in both (json and command line), the json parameter is discarded
    :return:
    """
    parser = argparse.ArgumentParser(description=f'Arguments from command line and json')
    parser.add_argument("--epochs", required=False, type=int, help="Number of epochs")
    parser.add_argument("--input_size", required=False, type=int, help="Input size of a singular time sample")
    parser.add_argument("--hidden_size", required=False, type=int)
    parser.add_argument("--num_layers", required=False, type=int)
    parser.add_argument("--sequence_length", required=False, type=int)
    parser.add_argument("--lr", required=False, type=float)
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--train", required=False, type=str)
    parser.add_argument("--video", required=False, type=str, help="Video path. Video used for evaluation of results")
    parser.add_argument("--config_file", required=False, type=str, help="Configuration file for additional parameters")
    parser.add_argument("--features_dataset_path", required=False, type=str, help="Path dataset used from the svm model")
    parser.add_argument("--dataset_path", required=False, type=str, help="Path dataset")
    parser.add_argument("--test_dataset_path", required=False, type=str, help="Test dataset path")
    parser.add_argument("--checkpoint_path", required=False, type=str, help="Checkpoint path: for saving model parameters")
    parser.add_argument("--n_classes", required=False, type=int, help="Number of distinct instruments (classes)")
    parser.add_argument("--dropout", required=False, type=float, help="Dropout for fully connected layers")
    parser.add_argument("--load_model", required=False, type=bool,
                        help="Load existing model, if present. The checkpoint path is given through 'checkpoint_path' argument")
    args = parser.parse_args()
    args = upload_args_from_json(args, file_path)
    print(args)
    return args


def upload_args_from_json(args, file_path=os.path.join("configuration.json")):
    if args is None:
        parser = argparse.ArgumentParser(description=f'Arguments from json')
        args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        # do not override pre-existing arguments, if present.
        # In other terms, the arguments passed through CLI have the priority
        if hasattr(args, option) and getattr(args, option) is not None:
            continue
        if option_value == 'None':
            option_value = None
        if option_value == "True":
            option_value = True
        if option_value == "False":
            option_value = False
        setattr(args, option, option_value)
    return args


def load_existing_model(model, optimizer, checkpoint_path):
    try:
        print(f"Trying to load existing model from checkpoint @ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("...existing model loaded")
        max_test_f1_score = getattr(checkpoint, "max_test_f1_score", 0)
        epoch = getattr(checkpoint, "epoch", 0)
    except Exception as e:
        print("...loading failed")
        print(f"During loading the existing model, the following exception occured: \n{e}")
        print("The execution will continue anyway")
        max_test_f1_score = 0
        epoch = 0
    return max_test_f1_score, epoch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]