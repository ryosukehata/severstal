import os
import glob
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
import sys

from preprocess import dataframe_preprocess
from dataset import SegmentationDataset
from pytorch_lightning_module import ClassifyModel
from models import kaeru_classify_model
from configs import Configs
from utils import fix_seed

# sys.path.append(os.environ.get("TOGURO_LIB_PATH"))
# from slack import Slack
# from sheet import Sheet

start = time.time()


# config and fix seed, cuda config function


config = Configs()
fix_seed(config.SEED)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(config.input_path, "train.csv"))

    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for i, (train, test) in enumerate(kf.split(df)):
        if i == config.fold:
            train_loc, test_loc = train, test

    df_train = df.iloc[train_loc]
    df_valid = df.iloc[test_loc]

    train_dataset = ClassifyDataset(
        df_train, image_folder=os.path.join(config.input_path, "train_images"))
    valid_dataset = ClassifyDataset(df_valid, image_folder=os.path.join(
        config.input_path, "train_images"), train=False)
    model = kaeru_classify_model(output_class=config.output_class)

    ptl_model = ClassifyModel(model, train_dataset, valid_dataset, c)

    ouput_dir_name = config.output_path
    state_dict_path = os.path.join(ouput_dir_name, "state_dict")
    version = 0

    logger = TestTubeLogger(
        save_dir=os.getcwd(),
        name=ouput_dir_name,
        autosave=True,
        version=version,
        debug=config.debug,
        description=config.description,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=state_dict_path,
        save_best_only=True,
        verbose=True,
        save_weights_only=False,
        monitor='optim_metric',
        mode='max',
     )

    # slack = Slack()
    # worksheet = Sheet()

   try:
        trainer = Trainer(max_nb_epochs=config.num_epochs, 
                          gpus=[0],
                          log_save_interval=1, 
                          logger=logger,
                          fast_dev_run=config.fast_dev_run,
                          checkpoint_callback=checkpoint_callback)

        trainer.fit(ptl_model)

        ckpt_path = glob.glob(os.path.join(state_dict_path, "*.ckpt"))[0]

        print("training suceceded")

#        slack.notify_success(config.description)

    except Exception as e:
        print(e)
        print("training failed")
#        slack.notify_failed(config.description)
