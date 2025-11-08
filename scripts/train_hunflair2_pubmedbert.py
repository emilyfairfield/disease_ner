# Inspired by: https://flairnlp.github.io/flair/master/tutorial/tutorial-hunflair2/training-ner-models.html
import argparse, flair
from flair.datasets import NCBI_DISEASE
from flair.splitter import SegtokSentenceSplitter
from flair.splitter import SciSpacySentenceSplitter
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
import mlflow
import torch
from flair import set_seed
import random, numpy as np, torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Hugging Face model ID (e.g. microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, michiyasunaga/BioLinkBERT-base, allenai/biomed_roberta_base)"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--use_safetensors", action="store_true")
    args = parser.parse_args()

    # For full reproducibility
    seed = 42
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")

    # Select device
    flair.device = torch.device(args.device)
    print(f"Using device: {flair.device}")

    # # # Load dataset with Segtok to avoid SciSpaCy mismatch
    splitter = SegtokSentenceSplitter()
    corpus = NCBI_DISEASE(sentence_splitter=splitter)

    # # splitter = SciSpacySentenceSplitter()
    # # corpus = NCBI_DISEASE(sentence_splitter=splitter)

    # # Token-level label setup
    tag_type = "ner"
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type, add_unk=False)
    print("!TAG DICT:", tag_dictionary.get_items())
    
    # # Transformer-based embeddings (HunFlair2-style)
    embeddings_kwargs = dict(
        model=args.model,
        fine_tune=True,
        layers="-1",
        layer_mean=True,
        use_context=False,
        subtoken_pooling="first",
    )
    if args.use_safetensors:
        embeddings_kwargs["use_safetensors"] = True
    embeddings = TransformerWordEmbeddings(**embeddings_kwargs)

    # # SequenceTagger (Transformer + CRF, no RNN)
    tagger = SequenceTagger(
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
        use_rnn=False,
        reproject_embeddings=False,
        hidden_size=256,
        dropout=0.0, 
        word_dropout=0.0, 
        locked_dropout=0.1
    )

    # # Trainer setup
    trainer = ModelTrainer(tagger, corpus)

    # # MLflow logging
    mlflow.set_experiment("ncbi_disease_ner")
    run_name = f"hunflair2_{args.model.split('/')[-1]}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "device": args.device,
            "use_safetensors": args.use_safetensors,
            "tag_format": "BIOES",
            "use_crf": True,
            "use_rnn": False,
        })

        embeddings_storage_mode = "gpu" if flair.device.type == "cuda" else "cpu"

        trainer.train(
            base_path=f"runs/{run_name}",
            learning_rate=args.lr,
            mini_batch_size=args.batch_size,
            max_epochs=args.epochs,
            embeddings_storage_mode=embeddings_storage_mode,
            train_with_dev=False,
            save_final_model=True,
            main_evaluation_metric=("micro avg", "f1-score"),
            min_learning_rate=1e-6,
            patience=3,
        )

if __name__ == "__main__":
    main()
