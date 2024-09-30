from unsloth import FastLanguageModel
import torch
import pickle
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import mlflow.pytorch
import transformers

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

app = FastAPI()


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


alpaca_prompt = """
<|user|>
{}<|end|>
<|assistant|>
{}<|end|>
"""


def formatting_prompts_func(examples):
    instructions = examples["question"]
    outputs = examples["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, output)
        texts.append(text)
    return {
        "text": texts,
    }


pass


model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

remote_server_uri = "http://mlflow-server:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("recetas-model")


class Phi3(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model snapshot directory.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts["snapshot"], padding_side="left"
        )

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts["snapshot"], trust_remote_code=True
        )
        # If you are running this in a system that has a sufficiently powerful GPU with available VRAM,
        # uncomment the configuration setting below to leverage triton.
        # Note that triton dramatically improves the inference speed performance

        # config.attn_config["attn_impl"] = "triton"

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts["snapshot"],
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # NB: If you do not have a CUDA-capable device or have torch installed with CUDA support
        # this setting will not function correctly. Setting device to 'cpu' is valid, but
        # the performance will be very slow.
        # self.model.to(device="cpu")
        # If running on a GPU-compatible environment, uncomment the following line:
        self.model.to(device="cuda")

        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def predict(self, context, model_input, params=None):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]

        # Retrieve or use default values for temperature and max_tokens
        temperature = params.get("temperature", 0.1) if params else 0.1
        max_tokens = params.get("max_tokens", 1000) if params else 1000

        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        # NB: Sending the tokenized inputs to the GPU here explicitly will not work if your system does not have CUDA support.
        # If attempting to run this with GPU support, change 'cpu' to 'cuda' for maximum performance
        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(
            encoded_input,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors="pt")[0])
        generated_response = self.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        )

        return {"candidates": [generated_response]}


import numpy as np
import pandas as pd

from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "prompt"),
    ]
)
output_schema = Schema([ColSpec(DataType.string, "candidates")])

parameters = ParamSchema(
    [
        ParamSpec("temperature", DataType.float, np.float32(0.1), None),
        ParamSpec("max_tokens", DataType.integer, np.int32(1000), None),
    ]
)

signature = ModelSignature(
    inputs=input_schema, outputs=output_schema, params=parameters
)


# Define input example
input_example = pd.DataFrame({"prompt": ["What is Neo4J?"]})


def train_my_model():
    dataset = load_dataset("somosnlp/recetasdelaabuela_it", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    warmup_steps = 5
    per_device_train_batch_size = 2
    max_steps = 60
    weight_decay = 0.01
    gradient_accumulation_steps = (4,)

    with mlflow.start_run():
        mlflow.log_param("warmup_steps", warmup_steps)
        mlflow.log_param("per_device_train_batch_size", per_device_train_batch_size)
        mlflow.log_param("max_steps", max_steps)
        mlflow.log_param("weight_decay", weight_decay)

        print("Loading trainer")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
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
        print("Begin training")
        trainer_stats = trainer.train()
        print("Train finished")
        mlflow.log_metric(
            "minutesxtraining", round(trainer_stats.metrics["train_runtime"] / 60, 2)
        )

        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_16bit",
        )

        mlflow.pyfunc.log_model(
            "phi3-instruct",
            python_model=Phi3(),
            # NOTE: the artifacts dictionary mapping is critical! This dict is used by the load_context() method in our PHi3() class.
            artifacts={"snapshot": "./model"},
            pip_requirements=[
                f"transformers=={transformers.__version__}",
            ],
            registered_model_name="PhiModel",
            input_example=input_example,
            signature=signature,
        )


@app.get("/train")
async def trainn():
    train_my_model()
    return {"message": "Hello World"}


@app.get("/")
async def root():
    return {"message": "Hello World"}
