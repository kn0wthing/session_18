# Phi-2 Fine-tuning with OpenAssistant Dataset

This repository contains code and implementation for fine-tuning Microsoft's Phi-2 model using the OpenAssistant dataset (OASST1). The project demonstrates advanced training techniques including quantization and LoRA (Low-Rank Adaptation) to efficiently train a large language model with limited computational resources.

## 🎯 Project Overview

This project aims to:
- Fine-tune the Phi-2 model for improved conversational abilities
- Implement memory-efficient training techniques
- Create a production-ready conversational AI model
- Demonstrate best practices in modern LLM fine-tuning

## 🏗️ Project Structure

```
.
├── train.py              # Main training script
├── requirements.txt      # Project dependencies
├── logs.txt             # Training logs
└── phi2-assistant/      # Output directory
    └── checkpoints/     # Model checkpoints
```

### Key Components

#### `train.py`
The main training script implements:
1. **Model Initialization**
   - Loads Phi-2 with 4-bit quantization
   - Configures tokenizer with padding
   - Implements LoRA adaptation

2. **Dataset Processing**
   - Filters OpenAssistant dataset for English
   - Formats conversations
   - Implements efficient tokenization

3. **Training Loop**
   - Implements checkpointing
   - Manages training resumption
   - Logs training metrics

## 🛠️ Training Configuration

### Hardware Requirements
- GPU: T4 x2 (minimum 16GB VRAM)
- Training Duration: 1 epoch (multiple sessions)
- Training Steps: 2,455 steps completed

### Model Details
- Base Model: `microsoft/phi-2`
- Dataset: `OpenAssistant/oasst1` (English conversations)
- Training Type: Supervised Fine-tuning (SFT)

### Advanced Training Techniques

#### Quantization Configuration
- 4-bit quantization (BitsAndBytes)
- NF4 quantization type
- Double quantization enabled
- Float16 compute dtype
- Enables training on consumer GPUs

#### LoRA Configuration
- Rank (r): 16
- Alpha: 32
- Target Modules: 
  - q_proj (Query projection)
  - k_proj (Key projection)
  - v_proj (Value projection)
  - dense (Feed-forward layers)
- Dropout: 0.05
- Task Type: Causal Language Modeling
- Memory Efficient: Only trains a small number of parameters

### Training Hyperparameters
- Batch Size: 4 (optimized for memory constraints)
- Gradient Accumulation Steps: 4 (effective batch size = 16)
- Learning Rate: 2e-4 with cosine decay
- Optimizer: paged_adamw_32bit (memory efficient)
- LR Scheduler: Cosine with warmup
- Warmup Ratio: 0.03 (helps stabilize initial training)
- Max Sequence Length: 2048 tokens
- Gradient Checkpointing: Enabled (trades computation for memory)

## 📊 Training Monitoring and Checkpoints

### Checkpoint Management
- Saves checkpoints every 100 steps
- Maintains last 2 checkpoints for safety
- Automatic checkpoint archiving to ZIP
- Implements efficient resumption logic

### Logging System
- Detailed training metrics in `logs.txt`
- Tracks:
  - Loss progression
  - Learning rate schedule
  - Training speed
  - Step/epoch information

### Training Progress
- Initial Loss: ~1.895
- Final Loss: ~1.628
- Total Training Steps: 2,455
- Consistent loss reduction indicating successful training

## 🔧 Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Training

1. Prepare your environment:
```bash
export WANDB_DISABLED=true  # Disable W&B logging if not needed
```

2. Start training:
```bash
python train.py
```

3. Resume training from checkpoint:
```bash
python train.py --resume_from_checkpoint path/to/checkpoint
```

## 📝 Dataset Processing
The OpenAssistant dataset undergoes the following processing:
1. English conversation filtering
2. Human/Assistant conversation formatting:
   ```
   Human: [user_message]
   Assistant: [assistant_response]
   ```
3. Tokenization with truncation and padding
4. Maximum sequence length: 2048 tokens

## 🔍 Monitoring Training

You can monitor the training progress through:
1. Console output showing current loss and learning rate
2. `logs.txt` containing detailed metrics
3. Checkpoint directory for model saves

## 🤔 Common Issues and Solutions

1. **OOM Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Ensure 4-bit quantization is active

2. **Training Instability**
   - Adjust learning rate
   - Increase warmup steps
   - Check gradient clipping

## 📄 License
[Add license information]

## 🤝 Contributing
[Add contribution guidelines]

## 📫 Contact
[Add contact information]

## 🙏 Acknowledgments
- Microsoft for the Phi-2 model
- OpenAssistant for the dataset
- HuggingFace for the transformers library
