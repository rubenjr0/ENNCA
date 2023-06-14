from lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torch import nn
from depth_net import DepthNet
from pytorch_optimizer import Adan, Ranger


class DepthEstimationModule(LightningModule):
    def __init__(self, learning_rate=2e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = DepthNet()
        self.loss = nn.MSELoss()
        self.accuracy = SSIM(data_range=1.0)

    def forward(self, x):
        x = self.net(x)
        return x

    def _do_stuff(self, batch, name):
        x, y = batch
        estimated = self(x)
        acc = self.accuracy(estimated, y)
        self.log(f'{name}_acc', acc)
        loss = self.loss(estimated, y)
        self.log(f'{name}_loss', loss)
        return loss

    def training_step(self, batch, _batch_idx):
        return self._do_stuff(batch, 'train')

    def validation_step(self, batch, _batch_idx):
        loss = self._do_stuff(batch, 'validation')
        self.log('hp_metric', loss)
        return loss

    def test_step(self, batch, _batch_idx):
        return self._do_stuff(batch, 'test')

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        optimizer = Adan(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer


if __name__ == '__main__':
    mod = DepthEstimationModule()
    assert mod is not None
