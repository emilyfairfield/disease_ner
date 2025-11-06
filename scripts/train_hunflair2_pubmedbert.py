#Inspired by this demo: https://flairnlp.github.io/flair/master/tutorial/tutorial-hunflair2/training-ner-models.html
import argparse, os, flair
from flair.datasets import NCBI_DISEASE
from flair.splitter import SciSpacySentenceSplitter
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if flair.device.type == "cuda" else "cpu")
    args = parser.parse_args()

    flair.device = args.device

    print(f"Using device: {flair.device}")
    # Load corpus
    splitter = SciSpacySentenceSplitter()
    corpus = NCBI_DISEASE(sentence_splitter=splitter)
    tag_type = "ner"
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    # Embeddings: HunFlair2 (BioLinkBERT)
    # docs: https://flairnlp.github.io/docs/tutorial-embeddings/transformer-embeddings
    embeddings = TransformerWordEmbeddings(
        model="allenai/biomed_roberta_base",   # biomedical; safetensors on main
        fine_tune=True,
        layers="-1",
        layer_mean=True,
        use_context=True,
        subtoken_pooling="first",
        use_safetensors=True,                  # <- <-- correct place for this flag
    )
    embeddings.tokenizer.model_max_length = 512

    #sequence tagger documentation: https://flairnlp.github.io/flair/v0.13.1/api/flair.models.html#flair.models.SequenceTagger
    #train a sequence tagger demo explaining args: https://remotedesktop.google.com/access/session/11bbfe87-19df-6bea-a66e-4f667e1fb0f9
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    trainer = ModelTrainer(tagger, corpus)

    mlflow.set_experiment("ncbi_disease_ner")
    with mlflow.start_run(run_name="hunflair2_pubmedbert"):
        mlflow.log_params({"model": "allenai/biomed_roberta_base", "epochs": args.epochs, "lr": args.lr})
        trainer.train(
            base_path="runs/hunflair2_pubmedbert",
            learning_rate=args.lr,
            mini_batch_size=8,
            max_epochs=args.epochs,
            embeddings_storage_mode="gpu" if flair.device.type == "cuda" else "cpu",
            train_with_dev=False,
            checkpoint=True,
            save_final_model=True,
            main_evaluation_metric=EvaluationMetric.MICRO_F1_SCORE
        )

if __name__ == "__main__":
    main()