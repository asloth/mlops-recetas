import unittest
from unittest.mock import patch, MagicMock
import mlflow


class TestTrainMyModel(unittest.TestCase):

    @patch("trl.SFTTrainer")  # Patch the SFTTrainer
    @patch("datasets.load_dataset")  # Patch the dataset loader
    @patch("mlflow.start_run")  # Patch mlflow.start_run
    @patch("mlflow.log_param")  # Patch mlflow.log_param
    @patch("mlflow.log_metric")  # Patch mlflow.log_metric
    @patch("mlflow.pyfunc.log_model")  # Patch model logging
    @patch("unsloth.FastLanguageModel")  # Patch the unsloth FastLanguageModel
    @patch("unsloth.is_bfloat16_supported")  # Patch the unsloth is_bfloat16_supported
    def test_train_my_model(
        self,
        mock_is_bfloat16_supported,
        mock_fast_language_model,
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

        # Mock the FastLanguageModel's behavior
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_fast_language_model.from_pretrained.return_value = (
            mock_model,
            mock_tokenizer,
        )

        # Mock the SFTTrainer behavior
        mock_trainer = MagicMock()
        mock_sft_trainer.return_value = mock_trainer

        # Mock the training process return
        mock_trainer.train.return_value = {
            "train_runtime": 300
        }  # Mocked runtime for metrics

        # Mock is_bfloat16_supported to return False for testing purposes
        mock_is_bfloat16_supported.return_value = False

        # Call the function to be tested
        from trainmodel import train_my_model

        train_my_model()

        # Verify that load_dataset was called
        mock_load_dataset.assert_called_once_with(
            "somosnlp/recetasdelaabuela_it", split="train"
        )

        # Verify that FastLanguageModel was called correctly
        mock_fast_language_model.from_pretrained.assert_called_once_with(
            model_name="unsloth/Phi-3.5-mini-instruct",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
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
