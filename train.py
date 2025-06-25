import os
import argparse
import time
import json
import warnings
from sklearn.metrics import classification_report
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np

warnings.filterwarnings("ignore")

def evaluar(nombre, y_true, y_pred):
    print(f"\n📊 Classification Report: {nombre}")
    print(classification_report(y_true, y_pred))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    print(f"📁 Usando datasets desde: {args.data_dir}")

    train_path = os.path.join(args.data_dir, "train")
    val_path = os.path.join(args.data_dir, "val")
    test_path = os.path.join(args.data_dir, "test")

    tokenized_train = load_from_disk(train_path)
    tokenized_val = load_from_disk(val_path)
    tokenized_test = load_from_disk(test_path)

    print("✅ Datasets cargados")

    tokenizer = AutoTokenizer.from_pretrained("./modelo_base")
    model = AutoModelForSequenceClassification.from_pretrained("./modelo_base", num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("✅ Modelo y tokenizer precargados")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_steps=0,
        logging_steps=25,
        report_to="none",
        logging_strategy="no",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # 🔍 Evaluación previa
    print("\n🧠 Evaluación: Previa al entrenamiento")
    pred_train = trainer.predict(tokenized_train)
    pred_val = trainer.predict(tokenized_val)
    pred_test = trainer.predict(tokenized_test)

    y_true_train = tokenized_train["label"]
    y_true_val = tokenized_val["label"]
    y_true_test = tokenized_test["label"]

    y_pred_train = np.argmax(pred_train.predictions, axis=-1)
    y_pred_val = np.argmax(pred_val.predictions, axis=-1)
    y_pred_test = np.argmax(pred_test.predictions, axis=-1)

    evaluar("Train", y_true_train, y_pred_train)
    evaluar("Val", y_true_val, y_pred_val)
    evaluar("Test", y_true_test, y_pred_test)

    # 🚀 Entrenamiento
    print("\n🚀 Entrenando modelo...")
    start = time.time()
    trainer.train()
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"🕒 Tiempo entrenamiento: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("✅ Entrenamiento finalizado")

    # 🔍 Evaluación posterior
    print("\n🧠 Evaluación: Posterior al entrenamiento")
    pred_train = trainer.predict(tokenized_train)
    pred_val = trainer.predict(tokenized_val)
    pred_test = trainer.predict(tokenized_test)

    y_pred_train = np.argmax(pred_train.predictions, axis=-1)
    y_pred_val = np.argmax(pred_val.predictions, axis=-1)
    y_pred_test = np.argmax(pred_test.predictions, axis=-1)

    evaluar("Train", y_true_train, y_pred_train)
    evaluar("Val", y_true_val, y_pred_val)
    evaluar("Test", y_true_test, y_pred_test)

    # 💾 Guardar modelo
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, "modelo_final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "modelo_final"))

    print("✅ Modelo y tokenizer guardados")

if __name__ == "__main__":
    main()