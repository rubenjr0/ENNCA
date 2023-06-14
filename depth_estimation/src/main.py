import lightning.pytorch as pl
from utils import visualize
from data import NyuDataModule
from depth_estimation_module import DepthEstimationModule
from callbacks import ImageSampler


def run_experiment(name, lr, apply_augs):
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='validation_loss',
        min_delta=1e-3,
        patience=4,
        mode='min',
        verbose=True
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='models/',
        filename=f'best_model_{name}',
        save_top_k=1,
        monitor='validation_loss',
        mode='min',
    )

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='tb_logs',
        version=name
    )

    data = NyuDataModule(
        root='',
        csv_train='data/nyu2_train.csv',
        csv_test='data/nyu2_test.csv',
        batch_size=32,
        apply_augs=apply_augs
    )

    model = DepthEstimationModule(learning_rate=lr)

    trainer = pl.Trainer(
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            ImageSampler(),
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.ModelSummary(),
        ],
        max_epochs=30,
        precision=32 if str(model.device) == 'cpu' else 16,
        logger=tb_logger,
        gradient_clip_val=0.5,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        limit_test_batches=0.1,
    )

    print('Starting training...')
    trainer.fit(model, datamodule=data)

    print('Starting testing...')
    trainer.test(model, datamodule=data)

    model.eval()
    model.freeze()
    vis_batch = next(iter(data.test_dataloader()))
    visualize(f'{name} estimation', vis_batch, model)


experiments = [{
    'name': 'low_lr_with_augs',
    'lr': 2e-4,
    'apply_augs': True
},
    {
    'name': 'mid_lr_with_augs',
    'lr': 2e-3,
    'apply_augs': True
},
    {
    'name': 'high_lr_with_augs',
    'lr': 8e-3,
    'apply_augs': True
},
    {
    'name': 'without_augs',
    'lr': 2e-3,
    'apply_augs': False
}]

for (i, exp) in enumerate(experiments):
    print(f'Running experiment {i+1}/{len(experiments)}:', exp)
    run_experiment(**exp)
