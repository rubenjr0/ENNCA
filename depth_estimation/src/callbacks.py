from lightning import Callback
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


class ImageSampler(Callback):
    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range=None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def to_grid(self, images):
        return make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        logger = pl_module.logger.experiment
        images, targets = next(iter(DataLoader(
            trainer.datamodule.val, batch_size=self.num_samples)))
        # generate images
        estimated = pl_module(images.to(pl_module.device)).detach()
        # estimated_grid = to_pil_image(self.to_grid(estimated))
        logger.add_images("validation original images",
                          images, trainer.current_epoch)
        logger.add_images("validation real depth",
                          targets, trainer.current_epoch)
        logger.add_images("validation estimated depth", estimated, trainer.current_epoch)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        logger = pl_module.logger.experiment
        images, targets = next(iter(DataLoader(
            trainer.datamodule.test, batch_size=self.num_samples)))
        # generate images
        estimated = pl_module(images.to(pl_module.device)).detach()
        # estimated_grid = to_pil_image(self.to_grid(estimated))
        logger.add_images("test original images", images)
        logger.add_images("test real depth", targets)
        logger.add_images("test estimated depth", estimated)
