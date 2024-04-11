import os
import openai
import pandas as pd
from src.vRA import RaLLM
from utils.krippendorff_alpha import krippendorff
from utils.utils import majority_vote
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import csv
import re
import argparse
import tqdm
import time


def deductive_coding(args):
    """
    This example function demonstrates how to use the RaLLM package for deductive coding of qualitative data.

    The function reads data and a codebook from CSV files, generates the codebook prompt, and then uses the
    RaLLM package to obtain codes for each data point. The obtained codes are stored in a new column in the
    original DataFrame. Finally, the function calculates Cohen's Kappa or Krippendorff's Alpha to assess the
    inter-coder reliability.

    Returns:
    - DataFrame: A pandas DataFrame containing the original data and a new column with the obtained codes.
    """
    lm = RaLLM(args.key)
    # Read data and codebook from CSV files
    data = pd.read_csv(args.input)
    codebook = pd.read_csv(args.codebook)
    # Generate the codebook prompt from the codebook
    codebook_prompt, code_set = lm.codebook2prompt(codebook, format=args.codebook_format,
                                                   num_of_examples=args.number_of_example, language=args.language,
                                                   has_context=args.context)
    if args.na_label:
        code_set.append("NA")

    # Define the identity modifier and context description   
    if args.language == 'fr':
        meta_prompt = open('prompts/meta_prompt_fr.txt').read()
    elif args.language == 'ch':
        meta_prompt = open('prompts/meta_prompt_ch.txt').read()
    else:
        meta_prompt = open('prompts/meta_prompt_eng.txt').read()

    meta_prompt = meta_prompt.replace('{{CODE_SET}}', str(code_set))

    # Iterate through each row of the data
    results = []
    model_exp = []
    idx = 0
    for index, row in tqdm.tqdm(data.iterrows(), position=0, total=data.shape[0]):
        # Generate the final prompt
        prompt = lm.prompt_writer(str(row['data']), str(row['context']), codebook_prompt, code_set, meta_prompt,
                                  args.na_label, args.language, args.cot)
        # Obtain the code using the coder function from the RaLLM package
        if args.model == 'text-davinci-003':
            response = lm.coder(prompt, engine=args.model)
            code = response.choices[0].message.content.strip()
        else:
            response = lm.coder(prompt, engine=args.model, voter=args.voter)
            code_voters = [response.choices[i].message.content for i in range(len(response.choices))]
            code = majority_vote(code_voters).strip()
        # Add the obtained code to the dataset
        results.append(code)
        if args.cot:
            model_exp.append(code)
        idx += 1
        if idx % args.batch_size == 0:
            if args.cot:
                data['model_exp'] = pd.Series(model_exp)
            results = lm.code_clean(results, code_set)
            data['result'] = pd.Series(results)
            csv_idx = args.save.index('.csv')
            file_name = args.save[:csv_idx] + '_' + str(idx) + args.save[csv_idx:]
            data.to_csv(file_name, encoding="utf_8_sig", index=False)

    # NOTE: Please double check/post processing the results before beofre calculating the inter-rater reliability. Some codes maybe slightly different than the codebook.
    if args.cot:
        data['model_exp'] = pd.Series(model_exp)
    results = lm.code_clean(results, code_set)
    data['result'] = pd.Series(results)
    data.to_csv(args.save, encoding="utf_8_sig", index=False)
    # Calculate the Cohen's Kappa and Krippendorff's Alpha
    if args.verification:
        print(data['code'])
        print("Cohen's Kappa: %.3f" % lm.cohens_kappa_measure(data['code'].astype(str), data['result']))
        print("Krippendorff's Alpha: %.3f" % lm.krippendorff_alpha_measure(data['code'].astype(str), data['result'],
                                                                           code_set))

    return data


def personality_simulation(args):
    # Read data and codebook from CSV files
    lm = RaLLM(args.key, args.model_type)
    instructions = pd.read_csv(args.instruction, delimiter=";")

    # Iterate through each row of the data
    results = []
    idx = 0
    for index, row in instructions.iterrows():
        # Generate the prompt
        if args.type == 'force_choice':
            if 'personal_profile' in instructions.keys():
                prompt = ("For the following task, respond in a way that matches this description: \"" +" I'm " + row['personal_profile'] + (".\" Considering the statement, select "
                                                                                  "one of the following descriptions that "
                                                                                  "is the MOST like you and another one "
                                                                                  "that is the LEAST like you by indicating their index: \"") +
                          row['item'] + "\"")
            else:
                prompt = "For the following task, respond in a way that matches this description: \"" + row[
                    'persona_description'] + "\" " + row[
                             'test_instruction'] + (" select one of the following descriptions that is the MOST like you "
                                                    "and another one that is the LEAST like you by indicating their index: \"") + row['item'] + "\""
        elif args.type == 'score_scale':
            if 'personal_profile' in instructions.keys():
                prompt = ("For the following task, respond in a way that matches this description: \"I'm " + row['personal_profile'] + ".\" Considering the statement, please indicate the extent to which you agree or disagree on a scale from 1 to 5 (where 1 = \"disagree strongly\", 2 = \"disagree a little\", 3 = \"neither agree nor disagree\", 4 = \"agree a little\", and 5 = \"agree strongly\"): \"" +
                          row['item'] + ".\"")
            else:
                prompt = "For the following task, respond in a way that matches this description: \"" + row['persona_description'] + "\" Considering the statement, please indicate the extent to which you agree or disagree on a scale from 1 to 5 (where 1 = \"disagree strongly\", 2 = \"disagree a little\", 3 = \"neither agree nor disagree\", 4 = \"agree a little\", and 5 = \"agree strongly\"): \"" + row['item'] + ".\""

        result_voters = lm.coder(prompt, engine=args.model, voter=args.voter)
        result = majority_vote(result_voters).strip()
        results.append(result)

        idx += 1
        if idx == 1:
            print('Example prompt:', prompt)

        if idx % args.batch_size == 0:
            print("Load {}% prompts".format(idx / len(instructions) * 100))
            instructions['result'] = pd.Series(results)
            csv_idx = args.save.index('.csv')
            file_name = args.save[:csv_idx] + '_' + str(idx) + args.save[csv_idx:]
            instructions.to_csv(file_name, encoding="utf_8_sig", index=False)

    instructions['result'] = pd.Series(results)
    instructions.to_csv(args.save, encoding="utf_8_sig", index=False)

    return instructions

def real_world_task_simulation(args):
    # Read data and codebook from CSV files
    lm = RaLLM(args.key, args.model_type)
    instructions = pd.read_csv(args.instruction, delimiter=";")

    # Iterate through each row of the data
    results = []
    idx = 0
    for index, row in instructions.iterrows():
        # Generate the prompt
        prompt = "For the following task, respond in a way that matches this description: \"" + row[
            'persona_description'] + "I'm " + row['personal_profile'] + (
                     ". Considering the statement, generate a list of 20 different Facebook status updates as this "
                     "person. Each update must be verbose and reflect the person’s character and description. The "
                     "updates should cover, but should not be limited to, the following topics: work, "
                     "family, friends, free time, romantic life, TV / music / media consumption, "
                     "and communication with others.")
        response = lm.coder(prompt, engine=args.model, voter=1)
        updates = response.choices[0].message.content
        results.append(updates)

        idx += 1
        if idx % args.batch_size == 0:
            print("Load {}% prompts".format(idx / len(instructions)*100))
            instructions['result'] = pd.Series(results)
            csv_idx = args.save.index('.csv')
            file_name = args.save[:csv_idx] + '_' + str(idx) + args.save[csv_idx:]
            instructions.to_csv(file_name, encoding="utf_8_sig", index=False)

    instructions['result'] = pd.Series(results)
    instructions.to_csv(args.save, encoding="utf_8_sig", index=False)

    return instructions


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, default='./data/data_example.csv')
    argparser.add_argument('--codebook', type=str, default='./data/codebook_example.csv')
    argparser.add_argument('--instruction', type=str, default='./data/instruction_example.csv')
    argparser.add_argument('--type', type=str, choices = ['force_choice', 'score_scale'])
    argparser.add_argument('--save', type=str, default='results/results_example.csv')
    argparser.add_argument('--mode', type=str, default='deductive_coding')
    argparser.add_argument('--codebook_format', type=str, default='codebook')
    argparser.add_argument('--context', type=int, default=0)
    argparser.add_argument('--number_of_example', type=int, default=10)
    argparser.add_argument('--voter', type=int, default=1)
    argparser.add_argument('--language', type=str, default='en')
    argparser.add_argument('--key', type=str, default=None)
    argparser.add_argument('--model', type=str, default='gpt-4-0613')
    argparser.add_argument('--model_type', type=str, default='claude')
    argparser.add_argument('--verification', type=int, default=0)
    argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--na_label', type=int, default=0)
    argparser.add_argument('--cot', type=int, default=0)
    args = argparser.parse_args()

    if args.key:
        openai.api_key = args.key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    if args.mode == 'deductive_coding':
        deductive_coding(args)
    if args.mode == 'personality_trait':
        personality_simulation(args)
    if args.mode == 'real_world_task':
        real_world_task_simulation(args)


if __name__ == "__main__":
    main()
