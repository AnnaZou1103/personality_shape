# Persona Simulation
This repository contains codes that generate prompts instructing LLMs to adopt different personalities to respond to psychological tests such as BFI, IPIP, and HEXACO tests,
as well as complete real-world simulation tasks.

## Installation

1. Clone this repository:
```
https://github.com/AnnaZou1103/personality_shape
```

2. Install the required packages:
```
pip install -r requirements.txt
```

### Usage
Run the preprocess script preprocess.py to generate the instructions, and then run main script main.py to simulate the corresponding personalities to give responses. The results will be stored in ./output directory.

### Arguments
You can customize the behavior of the main.py script by modifying the command-line arguments:
```
--instruction: Path to the input CSV file containing the prompt instructions.
--save: Path to the output CSV file where the results will be saved.
--task: The mode in which to run the script. Choose 'force_choice', 'score_scale' or 'real_world_simulation'.
--voter: The number of geneartions for each data point. If n > 1, the final code is an aggreation of mutiple generations by majority vote. Default is 1.
--api_key: Your OpenAI API key. If not provided, the script will attempt to use the OPENAI_API_KEY environment variable.
--model_type: The type of LLMs to use. Choose 'gpt' or 'claude'.
--model_name: The name of the GPT or Claude model to use. Default is 'gpt-3.5-turbo'.
--batch_size: The batch size for saving the coding progress. Default is 100 (reuslts will be saved for everyon 100 data points).
```