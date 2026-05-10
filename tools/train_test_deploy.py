import json
import os

import yaml
from addict import Dict

from wildlife_tools.fork_additions import (
    PtReIDModel,
    ClassificationTrainerWithValidation,
    BalancedImageDataset,
    cuda_available,
    test_metrics,
    test_classification,
    print_info,
    deploy_model,
    ArcFaceWithCrossEntropyLoss,
    NumpyDataset,
)


def train(config):
    model = PtReIDModel(config.model_config, pretrained=True)
    train_transforms = model.strategy.get_train_transforms(config.img_size)
    test_transforms = model.strategy.get_test_transforms(config.img_size)

    dataset = BalancedImageDataset(
        metadata=config.metadata,
        root=config.dataset_directory,
        phase="train",
        transform=train_transforms,
        max_length=2000,
        select_every=1,
        detector_checkpoint=config.detector_checkpoint,
    )
    n_training_dataset = dataset.num_classes
    training_label_map = dataset.labels_map

    val_dataset = NumpyDataset(
        phase="val",
        metadata=config.metadata,
        root=config.dataset_directory,
        transform=test_transforms,
        img_size=config.img_size,
        max_length=2000,
        select_every=10,
        return_isolation=True,
        detector_checkpoint=config.detector_checkpoint,
    )
    validation_label_map = val_dataset.labels_map

    assert (
        config.num_classes == n_training_dataset
    ), f"The 'user_configs.yaml file has num_classes set to {config.num_classes}, but the training dataset contain {n_training_dataset} distinct classes.'"
    for v_lbl in validation_label_map:
        assert (
            v_lbl in training_label_map
        ), f"The validation label {v_lbl} in not in the training labels: {training_label_map}"

    with open(os.path.join(config.save_directory, "re-identification_metadata.yaml"), "w") as f:
        yaml.dump(
            dict(input_shape=[224, 224], nb_features=config.model_config.n_output_embd, identities=training_label_map),
            f,
        )

    objective = ArcFaceWithCrossEntropyLoss(
        num_classes=dataset.num_classes, embedding_size=config.model_config.n_output_embd, margin=0.5, scale=64
    )

    epochs = config.epochs
    optimizer = model.strategy.build_optimizer(model, objective)
    scheduler = model.strategy.build_scheduler(optimizer, epochs)

    trainer = ClassificationTrainerWithValidation(
        dataset=dataset,
        val_dataset=val_dataset,
        save_dir=config.save_directory,
        checkpoint_name="precision_track_re-identificator.pth",
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=config.batch_size,
        accumulation_steps=160 // config.batch_size,
        num_workers=2,
        epochs=epochs,
        device=config.device,
    )
    trainer.train()

    print_info("Done training.")


def test(config):
    ckpt_path = os.path.abspath(os.path.join(config.save_directory, "precision_track_re-identificator.pth"))
    model = PtReIDModel(config=config.model_config, checkpoint=ckpt_path)
    config.test_transforms = model.strategy.get_test_transforms(config.img_size)

    f1_metrics = test_metrics(config, model)
    f1_classification = test_classification(config, model)

    results = dict(
        f1_metrics=f1_metrics,
        f1_classification=f1_classification,
        nb_params=model.nb_params,
    )

    out_path = os.path.abspath(os.path.join(config.save_directory, "test_results.json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_info(f"Test results saved at: {out_path}")
    print_info("Done testing.")


def deploy(config):
    ckpt_path = os.path.abspath(os.path.join(config.save_directory, "precision_track_re-identificator.pth"))
    model = PtReIDModel(config=config.model_config, checkpoint=ckpt_path)
    model.eval()

    deploy_model(config, model)

    print_info("Done deploying.")


def main():
    with open(os.path.join("..", "configs", "user_configs.yaml"), "r") as f:
        config = Dict(yaml.safe_load(f))

    if cuda_available():
        print_info("Your machine is CUDA accelerated. Therefore, the processes will take place on GPU.")
        config.device = "cuda"
    else:
        print_info("Your machine is NOT CUDA accelerated. Therefore, the processes will take place on CPU.")
        config.device = "cpu"

    os.makedirs(config.save_directory, exist_ok=True)

    config.img_size = (224, 224)

    config.model_config = Dict(
        dict(
            backbone_name=config.backbone_name,
            freeze_backbone=config.freeze_backbone,
            n_output_embd=128,
            n_layers=3,
            n_classes=config.num_classes,
            dropout=0.0,
            bias=True,
        )
    )

    if config.train:
        print_info("Training...")
        train(config)
    else:
        print_info("Skipping training...")

    if config.test:
        print_info("Testing...")
        test(config)
    else:
        print_info("Skipping testing...")

    if config.deploy:
        print_info("Deploying")
        deploy(config)
    else:
        print_info("Skipping deploying...")


if __name__ == "__main__":
    main()
