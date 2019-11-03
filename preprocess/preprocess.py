import pandas as pd
import numpy as np


def dataframe_preprocess(df_filepath):
    """
    df_filepath : input fileのpathを指定する。
    """
    df = pd.read_csv(df_filepath)
    df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
    df["ClassId"] = df["ClassId"].astype(int)
    df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
    df["defects"] = df.count(axis=1)
    return df


def make_mask(row_id, df):
    """
    Data Encoder
    This function is intended to use for dataframe
    after "dataframe_prerocess" function.
    Input
    row_id : gicven a low
    df     : dataframe after above function dataframe_prerocess
    Output
    fname  : image_id
    mask   : numpy array (256, 1600, 4) indecites where decects is
    https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    """
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]  # 4 channel
    masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos: (pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order="F")
    return fname, masks


def make_mask_only3(row_id, df):
    """                                                                                                                                                                       
    Data Encoder                                                                                                                                                              
    This function is intended to use for dataframe                                                                                                                            
    after "dataframe_prerocess" function.                                                                                                                                    
    Input                                                                                                                                                                     
    row_id : gicven a low                                                                                                                                                     
    df     : dataframe after above function dataframe_prerocess                                                                                                               
    Output                                                                                                                                                                    
    fname  : image_id                                                                                                                                                         
    mask   : numpy array (256, 1600, 4) indecites where decects is                                                                                                            
    https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode                                                                                                     
    """
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]  # 4 channel
    masks = np.zeros((256, 1600, 1), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan and idx == 2:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos: (pos + le)] = 1
                masks[:, :, 0] = mask.reshape(256, 1600, order="F")
    return fname, masks
