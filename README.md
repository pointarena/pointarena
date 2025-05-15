# Static Set Builder - AI Image Point Recognition Evaluation Tool

This application provides a comprehensive platform for evaluating and benchmarking multimodal AI vision-language models on image point recognition tasks. It combines manual annotation capabilities with automated segmentation for precise object identification, and includes extensive evaluation tools for comparing various vision-language models.

## Key Features

- **Annotation System**: Grid-based selection interface for precise point annotations
- **Segment Anything Model (SAM) Integration**: Automatic segmentation using Meta's Segment Anything Model
- **Multi-Model Evaluation**: Compare various vision-language models including:
  - OpenAI models (GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano)
  - Google models (Gemini 2.5/2.0/1.5 series)
  - Open-source models (Molmo 7B series, Qwen 2.5-VL, LLaVA OneVision)
  - Claude and Grok models
- **Performance Analysis**: Visualize model performance with:
  - ELO ratings system
  - Pairwise win rates
  - Success rate metrics
- **Dynamic Testing Mode**: Test models in real-time with user-uploaded images
- **Cloud Storage Integration**: Save results to cloud storage via R2 integration

## Installation

### Core System

1. Clone the repository:
```bash 
git clone <repository-url>
cd static_set_builder
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

# For dynamic mode with cloud storage (optional)
R2_ENDPOINT_URL=your_r2_endpoint
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=your_r2_bucket
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

### Dynamic Testing Interface

1. Start the dynamic testing interface:
```bash
python dynamic.py
```

2. Open your browser at `http://localhost:7860`

3. Use the interface to:
   - Test models with provided test images
   - Upload your own images for testing
   - Compare model performance with ELO ratings

### Model Evaluator

Evaluate vision-language models on point recognition tasks:

```bash
# Run evaluation for a specific model
python model_evaluator.py --model MODEL_NAME --type MODEL_TYPE

# For example:
python model_evaluator.py --model gpt-4o --type openai
python model_evaluator.py --model gemini-2.0-flash --type gemini
python model_evaluator.py --model Molmo-7B-D-0924 --type molmo
```

The evaluator will:
1. Generate visualizations showing points predicted by each model
2. Save these visualizations to the `point_on_mask` directory
3. Create a JSON results file with detailed metrics

### Performance Analysis

Generate performance visualizations:

```bash
# Generate ELO leaderboard
python elo_leaderboard.py

# Generate pairwise win rates
python pairwise_win_rates.py
```

## Project Structure

- `app.py`: Main annotation application with Gradio UI
- `dynamic.py`: Dynamic testing interface with real-time model comparisons
- `model_evaluator.py`: Framework for evaluating different vision-language models
- `elo_leaderboard.py`: Generate ELO ratings for model performance
- `pairwise_win_rates.py`: Calculate pairwise model comparisons
- `molmo_api.py`: API client for Molmo model inference
- `optimize_user_input.py`: Optimize user prompts for better model performance
- `run.py`: Simple utility script for running evaluations

## Image Categories

The system supports five task categories:
1. **Affordable**: Tool recognition tasks
2. **Counting**: Object counting tasks
3. **Spatial**: Spatial relationship tasks
4. **Reasoning**: Visual reasoning tasks
5. **Steerable**: Tasks with reference points

## Model Support

### OpenAI Models
- gpt-4o
- gpt-4o-mini
- gpt-4.1
- gpt-4.1-mini
- gpt-4.1-nano

### Google Models
- gemini-2.5-flash-preview
- gemini-2.0-flash
- gemini-2.0-flash-lite
- gemini-1.5-flash
- gemini-1.5-flash-8b
- gemini-1.5-pro

### Open Source Models
- Molmo-7B-D-0924
- Molmo-7B-O-0924
- Qwen2.5-VL-7B-Instruct
- llava-onevision-qwen2-7b-ov-hf

### Additional Models
- claude-3-7-sonnet-20250219
- grok-2-vision-latest

## Requirements

Core dependencies:
- PyTorch and torchvision
- Gradio
- OpenAI, Google Generative AI, Anthropic, and x.ai APIs
- Segment Anything Model
- Transformers library
- Pillow, NumPy, Matplotlib
- FastAPI and Uvicorn (for API servers)
- Boto3 (for R2 cloud storage)

## Acknowledgments

- The Segment Anything Model (SAM) is provided by Meta AI Research
- This project builds on research from various multimodal AI research teams