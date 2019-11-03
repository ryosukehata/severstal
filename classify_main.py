from pytorch_lightning import Trainer
import os
import random
import glob
import time
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold

from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
import torch

from dataset import ClassifyDataset
from pytorch_lightning_train import ClassifyModel
from model import kaeru_classify_model
from lib.slack import Slack

start = time.time()


# config and fix seed, cuda config function


class Config:
    fold = 0
    input_path = "./input"
    encoder_name = "resnet18"
    classify_model = "resnet34"
    output_path = os.path.join("./output/classifer", classify_model, "fold_{}".format(fold))
    num_epochs = 1
    batch_size = 16
    output_class = 4
    lr = 0.001
    cuda_idx = 0
    SEED = 1119
    threshold = 0.5
    use_kaeru_model = True
    description = "classifer, model {}, fold {}".format(classify_model, fold)


def fix_seed(seed=1119):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def device_cuda(cuda_idx):
    if torch.cuda.is_available():
        return 'cuda:{}'.format(cuda_idx)

    else:
        return "cpu"


c = Config()
fix_seed(c.SEED)


df = pd.read_csv(os.path.join(c.input_path, "train.csv"))

kf = KFold(n_splits=5, shuffle=True, random_state=c.SEED)
for i, (train, test) in enumerate(kf.split(df)):
    if i == c.fold:
        train_loc, test_loc = train, test

df_train = df.iloc[train_loc]
df_valid = df.iloc[test_loc]

train_dataset = ClassifyDataset(df_train, input_filepath=os.path.join(c.input_path, "train_images"))
valid_dataset = ClassifyDataset(df_valid, input_filepath=os.path.join(c.input_path, "train_images"), train=False)
model = kaeru_classify_model(output_class=c.output_class)

ptl_model = ClassifyModel(model, train_dataset, valid_dataset, c)


if __name__ == "__main__":
    df = dataframe_preprocess(os.path.join(config.input_path, "train.csv"))
    df_train, df_valid = train_test_split(df, test_size=0.1, stratify=df["defects"], random_state=config.SEED)

    train_dataset = ClassifyDataset(df_train, input_filepath=os.path.join(config.input_path, "train_images"))
    valid_dataset = ClassifyDataset(df_valid, input_filepath=os.path.join(
        config.input_path, "train_images"), train=False)
    model =

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
        monitor='avg_val_acc',
        mode='max'
    )

    slack = Slack()
    worksheet = Sheet()

    try:
        trainer = Trainer(max_nb_epochs=c.num_epochs, gpus=[0],
                          log_save_interval=1, experiment=exp,
                          checkpoint_callback=checkpoint_callback)

#        trainer.fit(ptl_model)

        ckpt_path = glob.glob(os.path.join(state_dict_path, "*.ckpt"))[0]

        print(ckpt_path)
        output_data = torch.load(ckpt_path)
        print(output_data.keys())

        slack.notify_success(c.description)

    except:

        slack.notify_failed(c.description)
