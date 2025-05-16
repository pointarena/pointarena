<p align="center">
    <h1 align="center">
        PointArena: Probing Multimodal Grounding Through Language-Guided Pointing
    </h1>
</p>

<p align="center">
  <a href="https://victorthecreator.me/">Long Cheng<sup>1∗</sup></a>, 
  <a href="https://duanjiafei.com">Jiafei Duan<sup>1,2∗</sup></a>, 
  <a href="https://helen9975.github.io">Yi Ru Wang<sup>1†</sup></a>, 
  <a href="https://hq-fang.github.io">Haoquan Fang<sup>1,2†</sup></a>, 
  <a href="#">Boyang Li<sup>1†</sup></a>, 
  <br>
  <a href="#">Yushan Huang<sup>1</sup></a>, 
  <a href="#">Elvis Wang<sup>3</sup></a>, 
  <a href="#">Ainaz Eftekhar<sup>1,2</sup></a>, 
  <a href="#">Jason Lee<sup>1,2</sup></a>, 
  <a href="#">Wentao Yuan<sup>1</sup></a>, 
  <br>
  <a href="#">Rose Hendrix<sup>2</sup></a>, 
  <a href="https://nasmith.github.io/">Noah A. Smith<sup>1,2</sup></a>, 
  <a href="https://linguistics.washington.edu/people/fei-xia">Fei Xia<sup>1</sup></a>, 
  <a href="https://homes.cs.washington.edu/~fox">Dieter Fox<sup>1</sup></a>, 
  <a href="https://ranjaykrishna.com">Ranjay Krishna<sup>1,2</sup></a>
  <br>
  <sup>1</sup>University of Washington, 
  <sup>2</sup>Allen Institute for Artificial Intelligence, 
  <sup>3</sup>Anderson Collegiate Vocational Institute
</p>

<div align="center">
  <p>
    <a href="https://pointarena.github.io/">
      <img src="https://img.shields.io/badge/Website-grey?logo=google-chrome&logoColor=white&labelColor=blue">
    </a>
    <a href="https://arxiv.org/abs/2505.09990">
      <img src="https://img.shields.io/badge/arXiv-grey?logo=arxiv&logoColor=white&labelColor=red">
    </a>
    <a href="https://huggingface.co/datasets/PointArena/pointarena-data">
      <img src="https://img.shields.io/badge/Dataset-grey?logo=huggingface&logoColor=white&labelColor=yellow">
    </a>
    <a href="https://x.com/victor_UWer">
      <img src="https://img.shields.io/badge/Post-grey?logo=x&logoColor=white&labelColor=black">
    </a>
  </p>
</div>

<br>

This project provides a comprehensive platform for evaluating and benchmarking multimodal AI vision-language models on image point recognition tasks. It combines manual annotation capabilities with automated segmentation for precise object identification, and includes extensive evaluation tools for comparing various vision-language models.

## Key Features

- **Annotation System**: Grid-based selection interface for precise point annotations
- **Segment Anything Model (SAM) Integration**: Automatic segmentation using Meta's Segment Anything Model
- **Multi-Model Evaluation**: Compare various vision-language models including:
  - OpenAI models (GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano)
  - Google models (Gemini 2.5/2.0 series, including flash and pro variants)
  - Open-source models (Molmo series, Qwen 2.5-VL, LLaVA OneVision)
  - Claude (claude-3-7-sonnet-20250219) and Grok (grok-2-vision-latest) models
- **Performance Analysis**: Visualize model performance with:
  - ELO ratings system with confidence intervals
  - Pairwise win rates and match count heatmaps
  - Success rate metrics and performance summaries
- **Dynamic Testing Mode**: Test models in real-time with user-uploaded images
- **Human Benchmark**: Compare model performance against human baselines

## Installation

### Core System

1. Clone the repository:
```bash 
git clone <repository-url>
cd pointarena
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Molmo model evaluation:
```bash
pip install -r requirements_molmo.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
XAI_API_KEY=your_xai_api_key
SAM_CHECKPOINT_PATH=./sam_vit_h_4b8939.pth
SAM_MODEL_TYPE=vit_h
SAVED_MODELS_DIR=./models
```

5. Download the SAM model checkpoint:
```bash
# Download directly from Meta AI's repository
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

### Static Evaluation Interface

1. Start the annotation interface:
```bash
python app.py
```

2. Open your browser at `http://localhost:7860`

3. Use the interface to:
   - Manually annotate images with grid selection
   - Use SAM for automatic object segmentation
   - Compare different model predictions
   - Save annotations to a structured data format


### Point-Bench

Evaluate vision-language models on point recognition tasks:

```bash
# Run evaluation for a specific model
# For example:
python model_evaluator.py --model gpt-4o --type openai
python model_evaluator.py --model gemini-2.0-flash --type gemini
python molmo_evaluator.py --model Molmo-7B-D-0924 --type molmo
```

The evaluator will:
1. Generate visualizations showing points predicted by each model
2. Save these visualizations to the `point_on_mask` directory
3. Create a JSON results file with detailed metrics   


### Point-Battle

1. Start the dynamic testing interface:
```bash
python dynamic.py
```

2. Open your browser at `http://localhost:7860`

3. Use the interface to:
   - Test models with provided test images from different categories
   - Upload your own images for testing
   - Compare model performance in head-to-head battles
   - View dynamic ELO leaderboard



### Performance Analysis

Generate performance visualizations and statistics:

```bash
# Generate ELO leaderboard with confidence intervals
python elo_leaderboard.py

# Generate pairwise win rates and match counts
python pairwise_win_rates.py

# For human benchmark comparison
python human_benchmark.py
```

## Project Structure

- `app.py`: Main annotation application with Gradio UI for static evaluation
- `dynamic.py`: Point-Battle interface for head-to-head model comparisons
- `model_evaluator.py `: Point-Bench interface for evaluating different vision-language models
- `molmo_evaluator.py `: Point-Bench interface for evaluating different vision-language models
- `elo_leaderboard.py`: Generate ELO ratings and confidence intervals for model performance
- `pairwise_win_rates.py`: Calculate and visualize pairwise model comparisons with heatmaps
- `molmo_api.py`: API client for Molmo model inference with support for local or remote execution
- `optimize_user_input.py`: Optimize user prompts for better model performance
- `human_benchmark.py`: Evaluate human performance
- `segment_utils.py`: Helper utilities for the Segment Anything Model integration

## Image Categories

The system supports five specialized task categories:
1. **Affordable**: Tool recognition tasks requiring fine-grained object identification
2. **Counting**: Object counting tasks with numerical reasoning requirements
3. **Spatial**: Spatial relationship tasks requiring positional understanding
4. **Reasoning**: Visual reasoning tasks requiring complex visual inference
5. **Steerable**: Tasks with reference points requiring contextual understanding

## Model Support

### OpenAI Models
- gpt-4o
- o3
- gpt-4.1

### Google Models
- gemini-2.5-flash-preview-04-17
- gemini-2.5-pro-preview-05-06
- gemini-2.0-flash

### Open Source Models
- Molmo-7B-D-0924
- Molmo-7B-O-0924
- Molmo-72B-0924
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-32B-Instruct
- Qwen2.5-VL-72B-Instruct
- llava-onevision-qwen2-7b-ov-hf

### Additional Models
- claude-3-7-sonnet-20250219
- grok-2-vision-latest

## Data and Evaluation

- Uses a structured annotation format with point coordinates
- Stores masked regions for precise evaluation
- Supports multiple evaluation metrics:
  - Point-in-mask accuracy
  - ELO rating system with confidence intervals
  - Pairwise win rate comparisons
  - Total success rate across categories

## Requirements

Core dependencies:
- PyTorch (2.2.0) and torchvision (0.17.0)
- Gradio (5.22.0) for interactive interfaces
- OpenAI, Google Generative AI, Anthropic, and x.ai APIs
- Segment Anything Model from Meta AI
- Transformers library for local model inference
- Pillow, NumPy, Matplotlib for image processing and visualization
- FastAPI and Uvicorn for API services
- Pandas and Seaborn for data analysis and visualization
