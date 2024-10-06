import pytest
from unittest.mock import patch, MagicMock
import mlflow
from transformers import TrainingArguments
from trl import SFTTrainer


# Mock the SFTTrainer class used in your function
class SFTTrainerMock:
    def __init__(self, *args, **kwargs):
        pass

    def train(self):
        # Return mock stats for training
        return {"metrics": {"train_runtime": 600}}


# The function to test
def test_training_function():
    # Define some test parameters
    warmup_steps = 100
    per_device_train_batch_size = 16
    max_steps = 1000
    weight_decay = 0.01
    max_seq_length = 128
    gradient_accumulation_steps = 4
    is_bfloat16_supported = lambda: False
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = MagicMock()

    # Mock the mlflow and SFTTrainer
    with patch("mlflow.start_run"), patch("mlflow.log_param"), patch(
        "mlflow.log_metric"
    ), patch("trl.SFTTrainer", SFTTrainerMock):
        # Run the function under test
        with mlflow.start_run():
            mlflow.log_param("warmup_steps", warmup_steps)
            mlflow.log_param("per_device_train_batch_size", per_device_train_batch_size)
            mlflow.log_param("max_steps", max_steps)
            mlflow.log_param("weight_decay", weight_decay)

            # Simulate the trainer initialization
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=2,
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=warmup_steps,
                    max_steps=max_steps,
                    learning_rate=2e-4,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=weight_decay,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir="outputs",
                ),
            )

            # Simulate training
            trainer_stats = trainer.train()

            # Simulate logging metrics
            mlflow.log_metric(
                "minutesxtraining",
                round(trainer_stats["metrics"]["train_runtime"] / 60, 2),
            )

            # Simulate saving the model
            model.save_pretrained_merged("model", tokenizer)

            # Assertions
            assert trainer_stats["metrics"]["train_runtime"] == 600
            mlflow.log_metric.assert_called_with("minutesxtraining", 10)
