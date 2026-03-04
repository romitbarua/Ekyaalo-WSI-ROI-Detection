#!/usr/bin/env python3
# python run_training.py --config configs/training_example.yaml --metadata data/metadata/metadata_patient_split.csv --experiment_name resnet18

import argparse, logging, json, yaml, shutil
import pandas as pd, torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.models.train_model import train, TrainingConfig
from src.utils.logging_config import setup_logging
from src.utils.validate_config import validate_training_paths
from src.data.slide_dataloader import create_data_loader
from src.data.transforms import DEFAULT_TRANSFORM_CONFIG, create_transforms_from_config, save_resolved_transform_config

logger = logging.getLogger(__name__)


def create_experiment_dir(base_dir="experiments", name=None):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = base / (f"{name}_{ts}" if name else ts)
    exp.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "checkpoints", "logs", "configs", "metrics"):
        (exp / sub).mkdir(exist_ok=True)
    return exp


def prepare_data_loaders(metadata, train_config, transform_config, random_seed=42, log_dir=None):
    train_df = metadata[metadata['split'] == 'train'].copy()
    val_df = metadata[metadata['split'] == 'val'].copy()
    test_df = metadata[metadata['split'] == 'test'].copy()
    logger.info("Splits: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    tcfg = transform_config or DEFAULT_TRANSFORM_CONFIG
    train_tf, val_tf = create_transforms_from_config(tcfg, log_dir)

    bs = train_config.batch_size
    train_loader = create_data_loader(train_df, batch_size=bs, shuffle=True, transform=train_tf)
    val_loader = create_data_loader(val_df, batch_size=bs, shuffle=False, transform=val_tf)
    test_loader = create_data_loader(test_df, batch_size=bs, shuffle=False, transform=val_tf)

    no_aug_df = train_df.sample(frac=0.1, random_state=random_seed)
    extra = {"no_aug_train": create_data_loader(no_aug_df, batch_size=bs, shuffle=False, transform=val_tf)}

    return train_loader, val_loader, test_loader, extra, tcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--experiment_name', required=True)
    ap.add_argument('--experiment_dir', default='experiments')
    args = ap.parse_args()

    validate_training_paths(args.config, args.metadata)

    exp_dir = create_experiment_dir(args.experiment_dir, args.experiment_name)
    setup_logging(exp_dir)

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    train_config = TrainingConfig(**config_dict)
    shutil.copy(args.config, exp_dir / "configs" / "training_config.yaml")

    metadata = pd.read_csv(args.metadata)
    logger.info("Loaded %d samples from %s", len(metadata), args.metadata)

    transform_config = config_dict.get('transforms', DEFAULT_TRANSFORM_CONFIG)
    train_loader, val_loader, test_loader, extra, saved_tcfg = prepare_data_loaders(
        metadata, train_config, transform_config,
        random_seed=train_config.random_seed, log_dir=exp_dir / "logs",
    )
    save_resolved_transform_config(saved_tcfg, exp_dir / "configs" / "transform_config.json")

    train_config.save_dir = str(exp_dir / "models")
    train_config.checkpoint_dir = str(exp_dir / "checkpoints")

    info = dict(experiment_dir=str(exp_dir), created_at=datetime.now().isoformat(),
                metadata_path=args.metadata, device=train_config.device)
    (exp_dir / "experiment_info.json").write_text(json.dumps(info, indent=2))

    save_paths = {
        'model': exp_dir / "models" / "best_model.pth",
        'history': exp_dir / "metrics" / "training_history.json",
        'artifacts': exp_dir / "artifacts",
    }

    results = train(
        config=train_config, train_loader=train_loader, val_loader=val_loader,
        extra_eval_loaders=extra, save_paths=save_paths, save_history_every_n_epochs=5,
    )

    torch.save(results['model'].state_dict(), exp_dir / "models" / "final_model.pth")
    logger.info("Done. Best val acc: %.4f  ->  %s", results['best_val_acc'], exp_dir)


if __name__ == "__main__":
    main()
