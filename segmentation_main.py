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
from dataset import SegmentationDatasetOnly3, SegmentationDataset
from pytorch_lightning_module import SegmentationModel
from models import create_segmentation_models
from configs import Configs
from utils import fix_seed

# sys.path.append(os.environ.get("TOGURO_LIB_PATH"))
# from slack import Slack
# from sheet import Sheet

start = time.time()

config = Configs()
fix_seed(config.SEED)


if __name__ == "__main__":
    df = dataframe_preprocess(os.path.join(config.input_path, "train.csv"))
    df_train, df_valid = train_test_split(df,
                                          test_size=config.test_size,
                                          stratify=df["defects"],
                                          random_state=config.SEED)

    train_dataset = SegmentationDataset(
        df_train, image_folder=os.path.join(config.input_path, "train_images"))
    valid_dataset = SegmentationDataset(df_valid, image_folder=os.path.join(
                                        config.input_path, "train_images"), train=False)

    model = create_segmentation_models(config.encoder, config.arch)

    ptl_model = SegmentationModel(model, train_dataset, valid_dataset, config)

    ouput_dir_name = config.output_path
    state_dict_path = os.path.join(ouput_dir_name, "state_dict")
    version = 0

    logger = TestTubeLogger(
        save_dir=os.getcwd(),
        name=ouput_dir_name,
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

#    slack = Slack()
#    worksheet = Sheet()
    try:
        trainer = Trainer(max_nb_epochs=config.num_epochs,
                          gpus=[0],
                          log_save_interval=1,
                          logger=logger,
                          train_percent_check=config.train_percet_check,
                          fast_dev_run=config.fast_dev_run,
                          checkpoint_callback=checkpoint_callback)

        trainer.fit(ptl_model)

        ckpt_path = glob.glob(os.path.join(state_dict_path, "*.ckpt"))[0]
        output_data = torch.load(ckpt_path)
        total_time = time.time() - start

#        worksheet.post_severstal(config.encoder_name,
#                                 output_data['checkpoint_callback_best'],
#                                 config.description,
#                                 {'batch_size': config.batch_size, 'num_epochs': config.num_epochs,
#                                  'total_time': total_time, 'best_epoch': output_data['epoch']},
#                                 'hata')

#        slack.notify_success(config.description)
        print("training suceceded")

    except Exception as e:
        print(e)
        print("training failed")
#        slack.notify_failed(config.description)
