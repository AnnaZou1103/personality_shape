import argparse
import pandas as pd

import tiktoken


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--profile_save_path', type=str, default='data/personal_profile_social.csv')
    argparser.add_argument('--instruction_save_path', type=str,
                           default='../output/instructions/shape_bfi60_instruction.csv')
    argparser.add_argument('--model_name', type=str, default='gpt-4o', choices=['gpt-4o', 'gpt-4'])

    args = argparser.parse_args()

    encoding = tiktoken.encoding_for_model(args.model_name)
    instruction_df = pd.read_csv(args.instruction_save_path, delimiter=";")

    token_num = 0
    for idx, row in instruction_df.iterrows():
        if 'personal_profile' in instruction_df.keys():
            description = f"I'm {row['personal_profile']}."
        else:
            description = row['persona_description']

        prompt = (f"For the following task, respond in a way that matches this description: \"{description}\" "
                  f"Considering the statement, please indicate the extent to which you agree or disagree on a "
                  f"scale from 1 to 5 (where 1 = \"disagree strongly\", 2 = \"disagree a little\", 3 = \"neither "
                  f"agree nor disagree\", 4 = \"agree a little\", and 5 = \"agree strongly\"): \"{row['item']}.\"")

        # print(prompt)
        token_num += len(encoding.encode(prompt))

    print(f"Token number: {token_num}")

