from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cascade import classify_artwork
from .data import prepare_manifest
from .evaluate import evaluate_model
from .predict import predict_image
from .tasks import TASKS, TaskSpec
from .train import train_model


def build_parser(task: TaskSpec, *, include_cascade: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=task.key,
        description=f"{task.display_name} artwork classifier",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Classify one image")
    predict.add_argument("image", type=Path)
    predict.add_argument("--checkpoint", type=Path, default=task.checkpoint)
    predict.add_argument("--device", default="auto")

    prepare = subparsers.add_parser("prepare", help="Validate data and create fixed splits")
    prepare.add_argument("--data-dir", type=Path, default=task.data_dir)
    prepare.add_argument("--manifest", type=Path, default=task.manifest)
    prepare.add_argument("--duplicate-report", type=Path, default=task.duplicate_report)
    prepare.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help=f"Train {task.display_name}")
    train.add_argument("--manifest", type=Path, default=task.manifest)
    train.add_argument("--output", type=Path, default=task.checkpoint)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate.add_argument("--manifest", type=Path, default=task.manifest)
    evaluate.add_argument("--checkpoint", type=Path, default=task.checkpoint)
    evaluate.add_argument("--output", type=Path, default=task.evaluation_report)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-workers", type=int, default=4)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default="auto")

    if include_cascade:
        classify = subparsers.add_parser(
            "classify", help="Classify as sketch, portrait, or general painting"
        )
        classify.add_argument("image", type=Path)
        classify.add_argument(
            "--sketch-checkpoint", type=Path, default=TASKS["sketchnet"].checkpoint
        )
        classify.add_argument(
            "--portrait-checkpoint", type=Path, default=TASKS["portraitnet"].checkpoint
        )
        classify.add_argument("--device", default="auto")
    return parser


def run(task: TaskSpec, *, include_cascade: bool = False) -> None:
    args = build_parser(task, include_cascade=include_cascade).parse_args()
    if args.command == "predict":
        result = predict_image(task, args.image, args.checkpoint, args.device)
    elif args.command == "prepare":
        result = prepare_manifest(
            task, args.data_dir, args.manifest, args.duplicate_report, args.seed
        )
    elif args.command == "train":
        result = train_model(
            task,
            args.manifest,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            seed=args.seed,
            device_name=args.device,
        )
    elif args.command == "evaluate":
        result = evaluate_model(
            task,
            args.manifest,
            args.checkpoint,
            args.output,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device_name=args.device,
        )
    else:
        result = classify_artwork(
            args.image,
            args.sketch_checkpoint,
            args.portrait_checkpoint,
            args.device,
        )
    print(json.dumps(result, indent=2))


def sketchnet_main() -> None:
    run(TASKS["sketchnet"], include_cascade=True)


def portraitnet_main() -> None:
    run(TASKS["portraitnet"])
