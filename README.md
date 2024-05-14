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
You can customize the behavior of the script by modifying the command-line arguments:
```
--instruction: Path to the input CSV file containing the prompt instructions.
--save: Path to the output CSV file where the results will be saved.
--mode: The mode in which to run the script. Choose 'personality_trait' or 'real_world_task'.
--voter: The number of geneartions for each data point. If n > 1, the final code is an aggreation of mutiple generations by majority vote. Default is 1.
--api_key: Your OpenAI API key. If not provided, the script will attempt to use the OPENAI_API_KEY environment variable.
--model: The name of the GPT model to use (e.g., 'gpt-4-0613', 'text-davinci-003'). Default is 'gpt-4-0613'.
--batch_size: The batch size for saving the coding progress. Default is 100 (reuslts will be saved for everyon 100 data points).
```