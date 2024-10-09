import unittest
from unittest.mock import patch, MagicMock
import mlflow
import sys


class TestTrainMyModel(unittest.TestCase):

    @patch("trainmodel.SFTTrainer")  # Patch the SFTTrainer
    @patch("trainmodel.load_dataset")  # Patch the dataset loader
    @patch("trainmodel.mlflow.start_run")  # Patch mlflow.start_run
    @patch("trainmodel.mlflow.log_param")  # Patch mlflow.log_param
    @patch("trainmodel.mlflow.log_metric")  # Patch mlflow.log_metric
    @patch("trainmodel.mlflow.pyfunc.log_model")  # Patch the model logging
    def test_train_my_model(
        self,
        mock_log_model,
        mock_log_metric,
        mock_log_param,
        mock_start_run,
        mock_load_dataset,
        mock_sft_trainer,
    ):
        # Mock dataset returned by load_dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        # Mock the SFTTrainer behavior
        mock_trainer = MagicMock()
        mock_sft_trainer.return_value = mock_trainer

        # Mock the training process return
        mock_trainer.train.return_value = {
            "train_runtime": 300
        }  # Mocked runtime for metrics

        # Mock the entire unsloth module
        mock_unsloth = MagicMock()
        mock_FastLanguageModel = MagicMock()
        # Create mock objects for model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        # Configure the from_pretrained method to return the mock model and tokenizer
        mock_FastLanguageModel.from_pretrained.return_value = (
            mock_model,
            mock_tokenizer,
        )

        mock_unsloth.FastLanguageModel = mock_FastLanguageModel
        sys.modules["unsloth"] = mock_unsloth
        sys.modules["unsloth.FastLanguageModel"] = mock_FastLanguageModel
        sys.modules["unsloth.is_bfloat16_supported"] = MagicMock()

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            from trainmodel import train_my_model, formatting_prompts_func, Phi3

            train_my_model()
        # Verify that load_dataset was called
        mock_load_dataset.assert_called_once_with(
            "somosnlp/recetasdelaabuela_it", split="train"
        )

        # Verify that SFTTrainer was initialized correctly
        mock_sft_trainer.assert_called_once()

        # Verify that train was called
        mock_trainer.train.assert_called_once()

        # Verify that MLflow functions were called
        mock_start_run.assert_called_once()
        mock_log_param.assert_any_call("warmup_steps", 5)
        mock_log_param.assert_any_call("per_device_train_batch_size", 2)
        mock_log_param.assert_any_call("max_steps", 10)
        mock_log_param.assert_any_call("weight_decay", 0.01)
        mock_log_metric.assert_any_call("minutesxtraining", 5.0)  # 300 seconds / 60

        # Verify model logging
        mock_log_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
