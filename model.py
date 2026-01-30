#!/usr/bin/env python3
from neurogc.models.lstm import (
    GCDataset,
    LSTMNetwork as GCPredictor,
    LSTMPredictor as GCPredictorWrapper,
    load_model,
    train_model,
)


def predict(metrics: dict, model_path: str = "gc_model.pth") -> float:
    wrapper = load_model(model_path)

    for _ in range(wrapper.config.sequence_length):
        wrapper.add_metrics(metrics)

    return wrapper.predict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test GC predictor model")
    parser.add_argument("--train", type=str, help="Path to CSV file for training")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--model", type=str, default="gc_model.pth", help="Path to model file")
    parser.add_argument("--test", action="store_true", help="Test the model with sample data")

    args = parser.parse_args()

    if args.train:
        print(f"Training model from {args.train}...")
        model, loss, params = train_model(args.train, args.config, args.model)
        print(f"Training complete. Final loss: {loss:.6f}")

    elif args.test:
        print("Testing model with sample metrics...")
        sample_metrics = {
            "cpu": 45.0,
            "mem": 60.0,
            "disk_read": 1000000.0,
            "disk_write": 500000.0,
            "net_sent": 100000.0,
            "net_recv": 200000.0,
            "rps": 100.0,
            "p95": 50.0,
            "p99": 100.0,
            "gc_triggered": False,
        }

        try:
            prediction = predict(sample_metrics, args.model)
            print(f"GC Urgency Prediction: {prediction:.4f}")
        except FileNotFoundError:
            print(f"Model file {args.model} not found. Train a model first.")

    else:
        parser.print_help()
