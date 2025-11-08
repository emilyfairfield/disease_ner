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
    parser.add_argument("--epochs", type=int, default=20)
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

    # For full reproducibility - try different seeds if current one fails
    seed = 42
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")
    
    # Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    
    # Debug: Check if we have entity labels in training data
    entity_count = 0
    total_tokens = 0
    for sentence in corpus.train[:100]:  # Check first 100 sentences
        for token in sentence:
            total_tokens += 1
            if token.get_label(tag_type).value != 'O':
                entity_count += 1
    
    print(f"Debug: Found {entity_count} entity tokens out of {total_tokens} total tokens in first 100 sentences")
    print(f"Entity percentage: {entity_count/total_tokens*100:.1f}%")
    
    if entity_count == 0:
        print("⚠️  WARNING: No entity labels found in training data!")
    
    # Check dev set too
    dev_entities = sum(1 for sent in corpus.dev[:50] for token in sent if token.get_label(tag_type).value != 'O')
    print(f"Debug: Found {dev_entities} entity tokens in first 50 dev sentences")
    
    # # Transformer-based embeddings (EXACT WORKING CONFIG)
    embeddings_kwargs = dict(
        model=args.model,
        fine_tune=True,
        layers="-1",
        layer_mean=True,
        use_context=False,  # This was the working setting
        subtoken_pooling="first",
    )
    if args.use_safetensors:
        embeddings_kwargs["use_safetensors"] = True
    embeddings = TransformerWordEmbeddings(**embeddings_kwargs)

    # # SequenceTagger (EXACT WORKING CONFIG: Transformer + CRF, no RNN)
    tagger = SequenceTagger(
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
        use_rnn=False,  # This was the working setting - NO BiLSTM
        # Remove the extra parameters that weren't in working config
    )
    
    # Debug: Test model on a few sentences before training
    print("Debug: Testing model initialization...")
    test_sentences = corpus.dev[:3]
    for i, sentence in enumerate(test_sentences):
        tagger.predict(sentence)
        predictions = [token.get_label(tag_type).value for token in sentence]
        non_o_preds = sum(1 for pred in predictions if pred != 'O')
        print(f"Sentence {i+1}: {non_o_preds}/{len(predictions)} non-O predictions")
        if non_o_preds > 0:
            print(f"  Non-O predictions: {[pred for pred in predictions if pred != 'O']}")

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
            min_learning_rate=1e-6,  # Same as working config
            patience=3,  # Same as working config
        )
        
        # Explicitly evaluate the best model from THIS training run on test set
        print("\n" + "="*80)
        print("FINAL EVALUATION: Best model from THIS training run on test set")
        print("="*80)
        
        # Load the final model saved during this training run
        final_model_path = f"runs/{run_name}/final-model.pt"
        print(f"Loading final model from: {final_model_path}")
        
        import os
        if os.path.exists(final_model_path):
            final_tagger = SequenceTagger.load(final_model_path)
            test_results = final_tagger.evaluate(corpus.test, gold_label_type=tag_type)
            
            print(f"\nTest set evaluation (final model from epochs 1-{args.epochs}):")
            print(f"Test F1 Score: {test_results.main_score:.4f}")
            
            # Log the test metrics to MLflow
            mlflow.log_metric("test_f1", test_results.main_score)
            
            if test_results.main_score > 0.5:
                print(f"🎉 SUCCESS: Achieved F1 = {test_results.main_score:.4f}")
            else:
                print(f"⚠️  Low F1: {test_results.main_score:.4f} - may need parameter tuning")
        else:
            print(f"ERROR: Final model not found at {final_model_path}")

if __name__ == "__main__":
    main()
