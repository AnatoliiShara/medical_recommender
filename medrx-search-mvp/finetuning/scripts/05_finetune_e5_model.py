"""
05_finetune_e5_model.py

Fine-tuning intfloat/multilingual-e5-base для Ukrainian medical search (Stage 1)
- CPU-optimized під ноут з 12 логічними ядрами, 15 GiB RAM.
- Використовує training_pairs_stage1.jsonl:
    {"query": ..., "positive": ..., "hard_negatives": [...]}

Основні рішення:
- Використовуємо MultipleNegativesRankingLoss з парами (query, positive).
  Hard negatives слугують додатковим джерелом "важких" прикладів,
  але в Stage 1 ми покладаємося на in-batch negatives (класична схема E5).
- Обмежуємо кількість CPU-тредів, щоб ноут залишався живим.
- Підтримка resume через --resume_from <checkpoint_dir>.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
import random

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# -------------------------------------------------------
# Базові налаштування
# -------------------------------------------------------

RANDOM_SEED = 42


def setup_cpu_threads(num_threads: int = 6) -> None:
    """
    Обмежуємо кількість потоків для BLAS / PyTorch, щоб машина не задихалась.
    """
    env_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTORCH_NUM_THREADS",
    ]
    for var in env_vars:
        os.environ.setdefault(var, str(num_threads))

    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except Exception:
        pass


def get_repo_root() -> Path:
    """
    Знаходимо корінь репозиторію відносно поточного файлу.
    """
    script_path = Path(__file__).resolve()
    # finetuning/scripts/05_finetune_e5_model.py -> repo_root = parents[2]
    return script_path.parents[2]


def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"e5_stage1_{timestamp}.log"

    logger = logging.getLogger("finetune_e5_stage1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"📄 Logging to: {log_file}")
    return logger


# -------------------------------------------------------
# Завантаження даних
# -------------------------------------------------------

def load_training_data(path: Path, logger: logging.Logger) -> list[InputExample]:
    """
    Завантажуємо training_pairs_stage1.jsonl і будуємо InputExample'и
    для MultipleNegativesRankingLoss як пари (query, positive).

    hard_negatives на цьому етапі не використовуємо явно, але вони вже
    враховані при побудові training_pairs_stage1 (через sampling).
    """
    logger.info(f"📂 Loading training data from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    examples: list[InputExample] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            query = str(data.get("query", "")).strip()
            positive = str(data.get("positive", "")).strip()
            if not query or not positive:
                continue

            # Для MNRL: тільки (anchor, positive)
            examples.append(InputExample(texts=[query, positive]))

    logger.info(f"✅ Loaded {len(examples):,} training examples")
    return examples


# -------------------------------------------------------
# Основний тренувальний пайплайн
# -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune intfloat/multilingual-e5-base on medical search data (Stage 1)."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0, щоб не плодити зайві процеси)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume from (optional)",
    )
    return parser.parse_args()


def main():
    setup_cpu_threads(num_threads=6)

    repo_root = get_repo_root()
    logs_dir = repo_root / "logs" / "finetuning"
    logger = setup_logging(logs_dir)

    logger.info("=" * 80)
    logger.info("🚀 FINE-TUNING intfloat/multilingual-e5-base (Stage 1: retrieval)")
    logger.info("=" * 80)

    args = parse_args()
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Шляхи
    base_model_name = "intfloat/multilingual-e5-base"
    train_data_path = repo_root / "data" / "training" / "finetuning" / "training_pairs_stage1.jsonl"
    output_dir = repo_root / "models" / "finetuned" / "e5-medrx-stage1"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Repo root: {repo_root}")
    logger.info(f"📁 Training data: {train_data_path}")
    logger.info(f"📁 Output dir: {output_dir}")
    logger.info(f"📁 Checkpoints dir: {checkpoint_dir}")

    # Пристрій
    device = torch.device("cpu")
    logger.info(f"🖥️  Device: {device}")

    # Завантаження моделі (base або checkpoint)
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint for resume not found: {resume_path}")
        logger.info(f"📥 Resuming from checkpoint: {resume_path}")
        model = SentenceTransformer(str(resume_path), device=str(device))
    else:
        logger.info(f"📥 Loading base model: {base_model_name}")
        model = SentenceTransformer(base_model_name, device=str(device))

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Model parameters: {num_params:,}")
    logger.info(f"📊 Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Завантаження тренувальних даних
    examples = load_training_data(train_data_path, logger)

    # DataLoader
    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(100, int(0.1 * total_steps))  # ≈10% warmup

    logger.info("\n📊 Training configuration:")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"   Total steps: {total_steps:,}")
    logger.info(f"   Warmup steps (~10%): {warmup_steps}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   DataLoader workers: {args.num_workers}")
    logger.info(f"   CPU threads (PyTorch): {torch.get_num_threads()}")

    # Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Зберігаємо тренувальну конфігурацію
    config_dict = {
        "base_model": base_model_name,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "total_examples": len(examples),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "resume_from": args.resume_from,
        "training_started_at": datetime.now().isoformat(),
    }
    config_path = output_dir / "training_config_stage1.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved training config to: {config_path}")

    logger.info("\n⏰ Training started...")
    logger.info("=" * 80)

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            output_path=str(output_dir),
            optimizer_params={"lr": args.learning_rate},
            show_progress_bar=True,
            use_amp=False,  # CPU: AMP не потрібен
            checkpoint_path=str(checkpoint_dir),
            checkpoint_save_steps=1000,
            checkpoint_save_total_limit=5,
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED (Stage 1)!")
        logger.info("=" * 80)
        logger.info(f"⏰ Finished at: {datetime.now().isoformat()}")
        logger.info(f"💾 Final model saved to: {output_dir}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user (KeyboardInterrupt).")
        logger.info(f"💾 Checkpoints available in: {checkpoint_dir}")
    except Exception as e:
        logger.error(f"\n❌ Training failed with exception: {e}")
        raise


if __name__ == "__main__":
    main()
