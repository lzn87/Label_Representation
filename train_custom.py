from __future__ import absolute_import, division, print_function, unicode_literals

import architecture
import argparse
import cifar
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils.trainutil import (
    train_directory_setup,
    train_log_results,
    train,
    valid_highdim,
    valid_category,
    valid_lowdim,
    test_highdim,
    test_category,
    test_lowdim,
)
from torchtoolbox.nn import LabelSmoothingLoss
from sklearn.manifold import TSNE
import logging

if __name__ == "__main__":
    # Params setup
    parser = argparse.ArgumentParser(description="CIFAR High-dimensional Model.")
    parser.add_argument(
        "--label",
        type=str,
        help="Label in [speech, uniform, shuffle, composite, random, uniform, lowdim, bert, glove]",
    )
    parser.add_argument(
        "--model", type=str, help="Image encoder in [vgg19, resnet110, resnet32]"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Manual seed.")
    parser.add_argument("--seed", type=int, help="Manual seed.", required=True)
    parser.add_argument("--level", type=int, default=100, help="Data level.")
    parser.add_argument(
        "--label_dir",
        type=str,
        help="Directory where labels are stored",
        default="./labels/label_files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where CIFAR datasets are stored",
        default="data",
    )
    parser.add_argument(
        "--base_dir", type=str, default="./outputs", help="Directory to save outputs"
    )
    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument(
        "--smoothing", type=float, default=0, help="Label smoothing level (default: 0)."
    )

    args = parser.parse_args()
    label = args.label
    data_dir = args.data_dir
    model_name = args.model
    seq_seed = args.seed
    data_level = args.level
    base_dir = args.base_dir
    label_dir = args.label_dir
    dataset = args.dataset
    smoothing = args.smoothing
    batch_size = args.batch_size

    assert dataset in ("iwildcam", "camelyon17")

    less_data = data_level < 100
    assert label in (
        "speech",
        "uniform",
        "shuffle",
        "composite",
        "random",
        "bert",
        "lowdim",
        "glove",
        "category",
    )

    if smoothing > 0:
        label = "category_smooth{}".format(smoothing)

    assert model_name in ("vgg19", "resnet110", "resnet32")
    if less_data:
        assert data_level < 90
    print(
        "Start training {}% {} {} model with manual seed {} and model {}.".format(
            data_level, dataset, label, seq_seed, model_name
        )
    )

    # Directory setup
    (
        best_model_path,
        checkpoint_path,
        log_path,
        snapshots_folder,
    ) = train_directory_setup(
        label, model_name, dataset, seq_seed, data_level, base_dir
    )

    log_file = log_path.replace('.csv', f'_{model_name}.txt')
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    logging.info(f'label {label} model_name {model_name} seq_seed {seq_seed} dataset {dataset}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 8

    # Loads train, validation, and test data

    train_set = cifar.dataset_wrapper(data_dir, dataset, 'train', label, label_dir)
    valid_set = cifar.dataset_wrapper(data_dir, dataset, 'val', label, label_dir)
    test_set = cifar.dataset_wrapper(data_dir, dataset, 'test', label, label_dir)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    num_classes = 182 if dataset == 'iwildcam' else 2


    # num_classes = int(dataset.split("cifar")[-1])
    # trainloader = cifar.get_train_loader(
    #     data_dir, label, num_classes, num_workers, 128, seq_seed, data_level, label_dir
    # )
    # validloader = cifar.get_valid_loader(
    #     data_dir, label, num_classes, num_workers, 100, seq_seed, label_dir
    # )
    # testloader = cifar.get_test_loader(
    #     data_dir, label, num_classes, num_workers, 100, label_dir
    # )

    # Model setup
    if "category" in label or label in ("lowdim", "glove"):
        if label == "glove":
            model = architecture.PytorchCategoryModel(model_name, 50)
            # model = architecture.CategoryModel(model_name, 50)
        else:
            model = architecture.PytorchCategoryModel(model_name, num_classes)
            # model = architecture.CategoryModel(model_name, num_classes)
    elif label == "bert":
        # model = architecture.BERTHighDimensionalModel(model_name, num_classes)
        raise NotImplementedError
    else:
        model = architecture.PytorchHighDimensionalModel(model_name, num_classes)
        # model = architecture.HighDimensionalModel(model_name, num_classes)

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[29, 49], gamma=0.1
    )

    epoch_stop = 50

    if "category" in label:
        if smoothing > 0:
            criterion = LabelSmoothingLoss(num_classes, smoothing=smoothing)
        else:
            criterion = nn.CrossEntropyLoss()

    else:
        criterion = nn.SmoothL1Loss()

    # Initializes training
    load_from_checkpoint = False
    if load_from_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        epoch_start = checkpoint["epoch"]
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        valid_acc = checkpoint["valid_acc"]
        model.load_state_dict(checkpoint["model_state_dict"])
        model = nn.DataParallel(model.module).to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        min_valid_loss = np.min(valid_loss)
        max_valid_acc = np.max(valid_acc)
        print(
            "Loaded checkpoint from epoch {} with min valid loss {} | max valid acc {}".format(
                epoch_start, min_valid_loss, max_valid_acc
            )
        )
    else:
        epoch_start = 0
        min_valid_loss = float("inf")
        max_valid_acc = 0.0
        train_loss = []
        valid_loss = []
        valid_acc = []

    # Trains model from epoch_start to epoch_stop
    for epoch in range(epoch_start, epoch_stop):
        new_train_loss = train(model, trainloader, optimizer, criterion, device, logging)
        if "category" in label:
            new_valid_loss, new_valid_acc = valid_category(
                model, validloader, criterion, device
            )
        elif label in ("lowdim", "glove"):
            new_valid_loss, new_valid_acc = valid_lowdim(
                model, validloader, criterion, device
            )
        else:
            new_valid_loss, new_valid_acc = valid_highdim(
                model, validloader, criterion, device
            )

        scheduler.step()
        train_loss.append(new_train_loss)
        valid_loss.append(new_valid_loss)
        valid_acc.append(new_valid_acc)
        print(
            "Epoch {} train loss {} | valid loss {} | valid acc {}".format(
                epoch + 1, new_train_loss, new_valid_loss, new_valid_acc
            )
        )
        logging.info(
            "Epoch {} train loss {} | valid loss {} | valid acc {}".format(
                epoch + 1, new_train_loss, new_valid_loss, new_valid_acc
            )
        )
        if new_valid_acc > max_valid_acc or (
            new_valid_acc == max_valid_acc and new_valid_loss < min_valid_loss
        ):
            print("Saving new best checkpoint...")
            min_valid_loss = new_valid_loss
            max_valid_acc = new_valid_acc
            torch.save(model.state_dict(), best_model_path)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            },
            checkpoint_path,
        )
        if epoch % 10 == 9:
            snapshot_file = "{}_seed{}_{}_epoch{}_model.pth".format(
                label, seq_seed, model_name, epoch + 1
            )
            snapshot_path = os.path.join(snapshots_folder, snapshot_file)
            torch.save(model.state_dict(), snapshot_path)

            # save visualized feature space
            with torch.no_grad():
                model.eval()
                valid_feat = []
                valid_targ = []
                test_feat = []
                test_targ = []

                valid_set_category = cifar.dataset_wrapper(data_dir, dataset, 'train', 'category', label_dir)
                test_set_category  = cifar.dataset_wrapper(data_dir, dataset, 'test', 'category', label_dir)
                validloader_category = torch.utils.data.DataLoader(valid_set_category, batch_size=batch_size, num_workers=num_workers)
                testloader_category = torch.utils.data.DataLoader(test_set_category, batch_size=batch_size, num_workers=num_workers)

                for _, (inputs, targets) in enumerate(validloader_category):
                    inputs, targets = inputs.to(device), targets.to(device)
                    _, outputs = model(inputs)
                    valid_feat.append(outputs)
                    valid_targ.append(targets)


                for _, (inputs, targets) in enumerate(testloader_category):
                    inputs, targets = inputs.to(device), targets.to(device)
                    _, outputs = model(inputs)
                    test_feat.append(outputs)
                    test_targ.append(targets)

                valid_feat = torch.cat(valid_feat).cpu().numpy()
                valid_targ = torch.cat(valid_targ).cpu().numpy()
                test_feat = torch.cat(test_feat).cpu().numpy()
                test_targ = torch.cat(test_targ).cpu().numpy()

                tsne_valid = TSNE()
                tsne_test = TSNE()

                valid_2d = tsne_valid.fit_transform(valid_feat)
                test_2d = tsne_test.fit_transform(test_feat)

                np.savez(os.path.join(snapshots_folder, 'valid_' + snapshot_file.replace('.pth', '.npz')), feat=valid_2d, targ=valid_targ)
                np.savez(os.path.join(snapshots_folder, 'test_' + snapshot_file.replace('.pth', '.npz')), feat=test_2d, targ=test_targ)

                print(f'snap shot saved at {snapshot_path}')
                logging.info(f'snap shot saved at {snapshot_path}')

    # Evaluates the best model
    model.load_state_dict(
        torch.load(best_model_path, map_location=torch.device(device))
    )

    # Test model
    if "category" in label:
        test_loss, test_acc = test_category(model, validloader, criterion, device)
    elif label in ("lowdim", "glove"):
        test_loss, test_acc = test_lowdim(model, validloader, criterion, device)
    else:
        test_loss, test_acc = test_highdim(model, validloader, criterion, device)
    print(
        "Label {}: seed {}, model {}, test loss {}, test acc {}".format(
            label, seq_seed, model_name, test_loss, test_acc
        )
    )

    # Logs results
    train_log_results(log_path, model_name, data_level, seq_seed, test_loss, test_acc)
