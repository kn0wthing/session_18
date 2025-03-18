
"""
Phi-2 Fine-tuning Script with OpenAssistant Dataset

This script implements supervised fine-tuning of Microsoft's Phi-2 model using the
OpenAssistant dataset (OASST1) with advanced techniques:
- 4-bit quantization
- LoRA adaptation
- Efficient resumable training

Author: [Your Name]
License: [License]
"""

import os
import json
import logging
import shutil
import argparse
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    logging as transformers_logging,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_info()

# Constants
DEFAULT_MODEL_NAME = "microsoft/phi-2"
DEFAULT_DATASET_NAME = "OpenAssistant/oasst1"
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_SEQ_LENGTH = 2048


class ConfigManager:
    """Manages configuration for training and model setup."""
    
    @staticmethod
    def get_quantization_config() -> BitsAndBytesConfig:
        """
        Creates the 4-bit quantization configuration.
        
        Returns:
            BitsAndBytesConfig: Configuration for quantized training
        """
        compute_dtype = getattr(torch, "float16")
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    @staticmethod
    def get_lora_config() -> LoraConfig:
        """
        Creates the LoRA configuration.
        
        Returns:
            LoraConfig: Configuration for LoRA adaptation
        """
        return LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    
    @staticmethod
    def get_training_config(
        output_dir: str,
        checkpoint_dir: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    ) -> SFTConfig:
        """
        Creates the training configuration.
        
        Args:
            output_dir: Directory to save the model
            checkpoint_dir: Directory containing checkpoint to resume from
            batch_size: Batch size for training
            learning_rate: Learning rate
            max_seq_length: Maximum sequence length
            
        Returns:
            SFTConfig: Configuration for supervised fine-tuning
        """
        return SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            learning_rate=learning_rate,
            fp16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=False,
            resume_from_checkpoint=checkpoint_dir if checkpoint_dir else True,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=False
        )


class DatasetManager:
    """Handles dataset loading and processing."""
    
    @staticmethod
    def prepare_dataset(tokenizer):
        """
        Loads and processes the OpenAssistant dataset.
        
        Args:
            tokenizer: Tokenizer for processing text
            
        Returns:
            Dataset: Processed and tokenized dataset
        """
        logger.info("Loading OpenAssistant dataset...")
        try:
            # Load the dataset
            dataset = load_dataset(DEFAULT_DATASET_NAME)
            
            # Filter for only English conversations
            logger.info("Filtering for English conversations...")
            dataset = dataset.filter(lambda x: x['lang'] == 'en')
            
            # Format conversations
            logger.info("Formatting conversations...")
            train_dataset = dataset['train']
            train_dataset = train_dataset.map(
                lambda x: {
                    'text': DatasetManager._format_conversation(x)
                }
            )
            
            # Tokenize the dataset
            logger.info("Tokenizing dataset...")
            train_dataset = train_dataset.map(
                lambda x: DatasetManager._tokenize(x, tokenizer),
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            logger.info(f"Dataset prepared with {len(train_dataset)} examples")
            return train_dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    @staticmethod
    def _format_conversation(example: Dict[str, Any]) -> str:
        """Format a conversation turn."""
        if example['role'] == 'assistant':
            return f"Assistant: {example['text']}\n"
        else:
            return f"Human: {example['text']}\n"
    
    @staticmethod
    def _tokenize(examples: Dict[str, Any], tokenizer) -> Dict[str, Any]:
        """Tokenize examples with proper truncation and padding."""
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=DEFAULT_MAX_SEQ_LENGTH,
            return_tensors=None
        )


class ModelManager:
    """Handles model loading and configuration."""
    
    @staticmethod
    def setup_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
        """
        Loads and configures the model and tokenizer.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            tuple: (model, tokenizer)
        """
        logger.info(f"Loading model: {model_name}")
        try:
            # Configure quantization
            bnb_config = ConfigManager.get_quantization_config()
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            model.config.use_cache = False
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            logger.info("Model and tokenizer loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


class TrainingCallbacks:
    """Custom callbacks for training."""
    
    @staticmethod
    def get_archive_callback(archive_dir: str = "/content"):
        """Returns a callback for archiving checkpoints."""
        
        class ArchiveCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
                archive_path = f"{archive_dir}/checkpoint-{state.global_step}.zip"
                
                try:
                    # Create a zip archive of the checkpoint directory
                    shutil.make_archive(archive_path.replace('.zip', ''), 'zip', checkpoint_dir)
                    logger.info(f"Archived checkpoint-{state.global_step} to {archive_path}")
                except Exception as e:
                    logger.error(f"Failed to archive checkpoint: {str(e)}")
        
        return ArchiveCheckpointCallback()
    
    @staticmethod
    def get_logging_callback(log_file: str):
        """Returns a callback for persistent logging."""
        
        class PersistentLoggingCallback(TrainerCallback):
            def __init__(self, log_file):
                self.log_file = log_file
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    try:
                        # Append log entry as a JSON line
                        with open(self.log_file, 'a') as f:
                            f.write(json.dumps({
                                "step": state.global_step,
                                "epoch": state.epoch,
                                "loss": logs.get("loss"),
                                "learning_rate": logs.get("learning_rate"),
                                "train_runtime": logs.get("train_runtime")
                            }) + "\n")
                        logger.info(f"Logged step {state.global_step} to {self.log_file}")
                    except Exception as e:
                        logger.error(f"Failed to write to log file: {str(e)}")
        
        return PersistentLoggingCallback(log_file)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 with OpenAssistant dataset")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="phi2-assistant",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default=None,
        help="Directory containing checkpoint to resume from"
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        default="logs.txt",
        help="File to save training logs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="Model name or path"
    )
    parser.add_argument(
        "--disable_wandb", 
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Disable wandb if requested
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define log file path
    log_file = args.log_file
    
    # Check for checkpoint
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        logger.info(f"Checkpoint found at {checkpoint_dir}")
        logger.info(f"Files in checkpoint: {os.listdir(checkpoint_dir)}")
    else:
        logger.info(f"No checkpoint found at {checkpoint_dir}; starting from scratch")
        checkpoint_dir = None
    
    # Check for log file
    if os.path.exists(log_file):
        logger.info(f"Existing log file found at {log_file}; will append to it")
    else:
        logger.info(f"No log file found; creating new one at {log_file}")
    
    try:
        # Setup model and tokenizer
        model, tokenizer = ModelManager.setup_model_and_tokenizer(args.model_name)
        
        # Prepare dataset
        train_dataset = DatasetManager.prepare_dataset(tokenizer)
        
        # Setup LoRA configuration
        lora_config = ConfigManager.get_lora_config()
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Setup training configuration
        sft_config = ConfigManager.get_training_config(
            output_dir=args.output_dir,
            checkpoint_dir=checkpoint_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Initialize trainer with SFTConfig
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=sft_config,
            tokenizer=tokenizer
        )
        
        # Add callbacks
        trainer.add_callback(TrainingCallbacks.get_archive_callback())
        trainer.add_callback(TrainingCallbacks.get_logging_callback(log_file))
        
        # Start or resume training
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            logger.info(f"Resuming training from {checkpoint_dir}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            logger.info("Starting training from scratch")
            trainer.train()
        
        # Save the final model
        logger.info(f"Saving final model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        
        # Archive the final model
        final_archive_path = f"{args.output_dir}-final.zip"
        shutil.make_archive(final_archive_path.replace('.zip', ''), 'zip', args.output_dir)
        logger.info(f"Archived final model to {final_archive_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()