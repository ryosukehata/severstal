import pytorch_lightning as ptl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from metrics.metric import dice_channel_torch, dice_channel_torch_with_each_channel
from loss_functions.loss_functions import FocalLoss


class ClassifyModel(ptl.LightningModule):
    '''
    definition of models
    '''

    def __init__(self, model, train_dataset, valid_dataset, c):
        super(ClassifyModel, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.criterion = nn.BCEWithLogitsLoss()
        self.c = c

    def forward(self, image):
        x = self.model(image)
        return x

    def _accuracy(self, preds, labels):
        preds = (torch.sigmoid(preds) > self.c.threshold).reshape_as(preds)
        labels = (labels > self.c.threshold).reshape_as(preds)
        return (preds == labels).sum().float() / preds.size(0)

    def training_step(self, batch, batch_nb):
        img, mask, label = batch
        pred = self.forward(img)
        label = label.float()
        return {'loss': self.criterion(pred, label)}

    def validation_step(self, batch, batch_nb):
        img, mask, label = batch
        pred = self.forward(img)
        label = label.float()
        return {'val_loss': self.criterion(pred, label),
                'val_accuracy': self._accuracy(pred, label)}

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        return {'avg_val_loss': avg_val_loss, 'avg_val_acc': avg_val_accuracy}

    # choose optimizers and scheduler
    def configure_optimizers(self):
        # optimizer=torch.optim.Adam(self.parameters(), lr=c.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.c.lr, momentum=0.9, weight_decay=0.0001)
        shceduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]

    # difine dataloader
    @ptl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.c.batch_size,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True)

    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.c.batch_size,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True)

# this function is needed, however currently I don't use them
    @ptl.data_loader
    def test_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.c.batch_size,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True)


class SegmentationModel(ptl.LightningModule):
    '''
    Model for segmentation
    '''

    def __init__(self, model, train_dataset, valid_dataset, c):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.criterion = FocalLoss(gamma=2)
        self.c = c

    def forward(self, image):
        x = self.model(image)
        return x

    def _metric(self, preds, masks):
        return dice_channel_torch_with_each(preds, masks)

    def training_step(self, batch, batch_nb):
        img, mask, label = batch
        pred = self.forward(img)
        label = label.float()
        return {'loss': self.criterion(pred, mask)}

    def validation_step(self, batch, batch_nb):
        img, mask, label = batch
        pred = self.forward(img)
        label = label.float()
        all_dice, single_channel = self._metric(pred, mask)
        return {'val_loss': self.criterion(pred, mask),
                'val_dice': all_dice, 'dice_1': single_channel[0],
                'dice_2': single_channel[1], 'dice_3': single_channel[2],
                'dice_4': single_channel[3]}

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_dice = torch.stack([x['val_dice'] for x in outputs]).mean()
        dice_1 = torch.stack([x['dice_1'] for x in outputs]).mean()
        dice_2 = torch.stack([x['dice_2'] for x in outputs]).mean()
        dice_3 = torch.stack([x['dice_3'] for x in outputs]).mean()
        dice_4 = torch.stack([x['dice_4'] for x in outputs]).mean()

        return {'avg_val_loss': avg_val_loss, 'avg_val_dice': avg_val_dice,
                'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3, 'dice_4': dice_4}

    # choose optimizers and scheduler
    def configure_optimizers(self):
        if self.c.encoder == "deeplabv3":
            train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.c.lr},
                            {'params': self.model.get_10x_lr_params(), 'lr': self.c.lr * 10}]
            optimizer = torch.optim.SGD(train_params,  momentum=0.9, weight_decay=5e-4)

        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.c.lr)
            #optimizer = torch.optim.SGD(self.parameters(), lr=self.c.lr, momentum=0.9, weight_decay=0.0001)
        #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer]  # , [scheduler]

    # difine dataloader
    @ptl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.c.batch_size,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True)

    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.c.batch_size,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True)

    # this function is needed, however currently I don't use them
    @ptl.data_loader
    def test_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.c.batch_size,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True)
