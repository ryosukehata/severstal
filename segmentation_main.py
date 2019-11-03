import os
import glob
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
import sys

from preprocess import dataframe_preprocess
from dataset import SegmentationDatasetOnly3, SegmentationDataset
from pytorch_lightning_module import SegmentationModel
from models import create_segmentation_models
from configs import Configs
from utils import fix_seed
from lib.slack import Slack
from lib.sheet import Sheet
print(SegmentationDataset)
print(SegmentationModel)


start = time.time()

config = Configs()
fix_seed(config.SEED)


if __name__ == "__main__":
    df = dataframe_preprocess(os.path.join(config.input_path, "train.csv"))
    df_train, df_valid = train_test_split(df, test_size=0.1, stratify=df["defects"], random_state=config.SEED)

    train_dataset = SegmentationDataset(df_train, input_filepath=os.path.join(config.input_path, "train_images"))
    valid_dataset = SegmentationDataset(df_valid, input_filepath=os.path.join(
                                        config.input_path, "train_images"), train=False)
    model = create_segmentation_models(config.encoder, config.arch)

    ptl_model = SegmentationModel(model, train_dataset, valid_dataset, config)

    ouput_dir_name = config.output_path
    state_dict_path = os.path.join(ouput_dir_name, "state_dict")
    version = 0

    exp = Experiment(
        name=ouput_dir_name,
        save_dir=os.getcwd(),
        autosave=True,
        version=version,
        description='test',
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=state_dict_path,
        save_best_only=True,
        verbose=True,
        save_weights_only=False,
        monitor='avg_val_loss',
        mode='min',
    )

    slack = Slack()
    worksheet = Sheet()
    try:
        trainer = Trainer(max_nb_epochs=config.num_epochs, gpus=[0],
                          log_save_interval=1, experiment=exp,
                          checkpoint_callback=checkpoint_callback)

        trainer.fit(ptl_model)

        ckpt_path = glob.glob(os.path.join(state_dict_path, "*.ckpt"))[0]
        output_data = torch.load(ckpt_path)
        total_time = time.time() - start

        worksheet.post_severstal(config.encoder_name,
                                 'only channel3 {}'.format(output_data['checkpoint_callback_best']),
                                 config.description,
                                 {'batch_size': config.batch_size, 'num_epochs': config.num_epochs,
                                  'total_time': total_time, 'best_epoch': output_data['epoch']},
                                 'hata')

        slack.notify_success(config.description)

    except:
        slack.notify_failed(config.description)
