import argparse
import json
import logging
import os
import sys
from pathlib import Path

#import sagemaker_containers
import torch
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_dataset(tokenizer, dataset_path: str):
    with open(dataset_path, 'r') as f:
        prompts = [json.loads(line) for line in f.readlines()]
    
    df = pd.DataFrame({
        'prompts': [p['prompt'] for p in prompts],
        'answers': [p['referenceResponse'] for p in prompts]
    })
    
    training_data = Dataset.from_pandas(df)

    def to_chat(prompts):
        texts = []
        inputs = prompts["prompts"]
        outputs = prompts["answers"]
    
        for input_, output in zip(inputs, outputs):
            text = tokenizer.apply_chat_template([
              {"role": "user", "content": input_},
              {"role": "assistant", "content": output},
            ], tokenize=False)
            texts.append(text)
    
        return { "text" : texts, }
    
    return training_data.map(to_chat, batched=True)

def peft_model_fn(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    return get_peft_model(model, peft_config)

def save_model(peft_model, tokenizer, save_dir: Path):
    peft_path = Path.cwd() / 'peft'
    peft_model.save_pretrained(peft_path)
    model = AutoPeftModelForCausalLM.from_pretrained(peft_path)
    model = model.merge_and_unload()

    model.save_pretrained(
        save_dir,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(
        save_dir,
        safe_serialization=True,
        max_shard_size="2GB"
    )

def train(args):
    use_cuda = torch.cuda.is_available
    device = torch.device("cuda" if use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Get Dataset
    data_path = str(Path(args.data_dir) / args.data_file)
    training_data = get_dataset(tokenizer, data_path)
    
    # Get Model
    peft_model = peft_model_fn(model)

    peft_model.train()
    tokenizer.padding_side = 'right'

    trainer=SFTTrainer(
        model=peft_model,
        train_dataset=training_data,
        tokenizer=tokenizer,
        args=SFTConfig(
            learning_rate=args.lr,
            output_dir="output",
            num_train_epochs=args.epochs,
            max_seq_length=1024,
            dataset_text_field='text',
        ),
    )
    trainer.train()

    save_model(peft_model, tokenizer, Path(args.model_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Id of the foundation model",
    )
    parser.add_argument("--data-file", type=str)

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
