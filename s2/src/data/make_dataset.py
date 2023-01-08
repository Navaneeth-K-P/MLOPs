# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # exchange with the corrupted mnist dataset
    train0 = np.load(input_filepath + r"/train_0.npz")
    # print(train1['images'])
    train1 = np.load(input_filepath + r"/train_1.npz")
    train2 = np.load(input_filepath + r"/train_2.npz")
    train3 = np.load(input_filepath + r"/train_3.npz")
    train4 = np.load(input_filepath + r"/train_4.npz")
    train5 = np.load(input_filepath + r"/train_5.npz")
    train6 = np.load(input_filepath + r"/train_6.npz")
    train7 = np.load(input_filepath + r"/train_7.npz")

    train_labels = torch.tensor(
        np.concatenate(
            [
                train0["labels"],
                train1["labels"],
                train2["labels"],
                train3["labels"],
                train4["labels"],
                train5["labels"],
                train6["labels"],
                train7["labels"],
            ]
        )
    )

    # Normalize the data and convert it to a tensor
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0,), (1,))]
    )
    train_images = transform(
        np.concatenate(
            [
                train0["images"],
                train1["images"],
                train2["images"],
                train3["images"],
                train4["images"],
                train5["images"],
                train6["images"],
                train7["images"],
            ]
        )
    )

    test = np.load(input_filepath + r"/test.npz")
    test_images = transform(test["images"])
    test_labels = torch.tensor(test["labels"])

    # The dimensions of the extracted images need to be changed to match the required shape
    train_images = train_images.permute(1, 0, 2)
    test_images = test_images.permute(1, 0, 2)
    
    test_images = torch.flatten(test_images.to(torch.float32), start_dim=1)
    train_images = torch.flatten(train_images.to(torch.float32), start_dim=1)
    
    train_set = TensorDataset(train_images, train_labels)
    test_set = TensorDataset(test_images, test_labels)

    torch.save(train_set, output_filepath + r"/train_set.pt")
    torch.save(test_set, output_filepath + r"/test_set.pt")

    return train_set, test_set


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
