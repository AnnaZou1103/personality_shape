import random
import re
from collections import Counter
import csv
from itertools import product

import pandas as pd
from itertools import permutations
from os import listdir
from os.path import isfile, join
import numpy as np

def majority_vote(output):
    '''
    This function takes a list called output and returns the most common element
    in the list. If there is a tie between elements, the function returns one of
    the tied elements that appears first in the list.
    '''
    x = Counter(output)
    return x.most_common(1)[0][0]


def generate_trait_instructions():
    persona_file = open("prompts/persona_description_v2.txt", "r")
    # instruction_file = open("prompts/test_instruction.txt", "r")
    item_file = open("prompts/item_description.txt", "r")

    persona = persona_file.read().splitlines()
    # instruction = instruction_file.read().splitlines()
    item = item_file.read().splitlines()

    # result = list(product(persona, instruction, item))
    result = list(product(persona, item))

    # header = ["persona_description", "test_instruction", "item"]
    header = ["persona_description", "item"]
    with open('data/persona_force_instruction_v2.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)

    persona_file.close()
    # instruction_file.close()
    item_file.close()


def generate_personal_profile():
    qualifier_file = open("prompts/qualifier.txt", "r")

    adjectival_markers = pd.read_csv("prompts/adjectival_markers.csv")
    qualifiers = qualifier_file.read().splitlines()

    adjectivals = {}

    for index, row in adjectival_markers.iterrows():
        domain = row['Domain']
        if domain not in adjectivals.keys():
            adjectivals[domain] = {'low_marker': [row['Low Marker']], 'high_marker': [row['High Marker']]}
        else:
            adjectivals[domain]['low_marker'].append(row['Low Marker'])
            adjectivals[domain]['high_marker'].append(row['High Marker'])

    with open('data/personal_profile.txt', 'w') as file:
        for qualifier in qualifiers:
            if qualifier:
                connector = ', ' + qualifier + ' '
                starter = qualifier + ' '
            else:
                connector = ', '
                starter = ''

            for domain, markers in adjectivals.items():
                file.write(starter + connector.join(markers['low_marker'][:-1]) + ' and {} {}\n'.format(qualifier,
                                                                                          markers['low_marker'][-1]))
                file.write(starter + connector.join(markers['high_marker'][:-1]) + ' and {} {}\n'.format(qualifier,
                                                                                           markers['high_marker'][-1]))

        for domain, markers in adjectivals.items():
            adj_list = []
            for low_marker, high_marker in zip(markers['low_marker'][:-1], markers['high_marker'][:-1]):
                adj_list.append('neither {} nor {}, '.format(low_marker, high_marker))
            file.write(''.join(adj_list) + 'and neither {} nor {}\n'.format(markers['low_marker'][-1],
                                                                            markers['high_marker'][-1]))

    qualifier_file.close()


def generate_extreme_personal_profile():
    adjectival_markers = pd.read_csv("prompts/adjectival_markers.csv")
    qualifier = 'extremely'

    adjectivals = {}
    for index, row in adjectival_markers.iterrows():
        domain = row['Domain']
        if domain not in adjectivals.keys():
            adjectivals[domain] = {'low_marker': [row['Low Marker']], 'high_marker': [row['High Marker']]}
        else:
            adjectivals[domain]['low_marker'].append(row['Low Marker'])
            adjectivals[domain]['high_marker'].append(row['High Marker'])
    values = list(adjectivals.values())
    all_index = list(product([0, 1], repeat=5))
    with open('data/extreme_personal_profile.txt', 'w') as file:
        starter = ', '+qualifier + ' '
        for index_list in all_index:
            profile = ''
            for idx, value in enumerate(index_list):
                if value:
                    profile += qualifier + ' '+ starter.join(values[idx]['low_marker'][:-1]) + ' and {} {}, '.format(qualifier,
                                                                                      values[idx]['low_marker'][-1])
                else: profile += qualifier + ' '+ starter.join(values[idx]['high_marker'][:-1]) + ' and {} {}, '.format(qualifier,
                                                                                      values[idx]['high_marker'][-1])
            file.write(profile+'\n')


def generate_shape_instructions():
    # persona_file = open("prompts/persona_description.txt", "r")
    profile_file = open("data/personal_profile.txt", "r")
    item_file = open("prompts/bfi2_items.txt", "r")

    # persona = persona_file.read().splitlines()
    profile = profile_file.read().splitlines()
    item = item_file.read().splitlines()

    # result = list(product(persona, item))
    result = list(product(profile, item))

    # header = ["persona_description", "item"]
    header = ["personal_profile", "item"]
    with open('data/shape_BFI_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)

    # persona_file.close()
    profile_file.close()
    item_file.close()

def post_process():
    items = pd.read_csv('results/gpt4/shape_BFI.csv')
    new_result = []
    for score in items['result']:
        if len(re.findall('\d', score))>0:
            new_result.append(re.findall('\d', score)[0])
        else: new_result.append(score)

    items['result'] = pd.Series(new_result)
    items.to_csv('results/gpt4/shape_BFI_new.csv', encoding="utf_8_sig", index=False)

def convert_format():
    item_file = open("prompts/bfi2_items.txt", "r")
    items = item_file.read().splitlines()
    
    results = pd.read_csv('results/gpt4/shape_BFI_new.csv')
    final_results = {}
    
    for idx, row in results.iterrows():
        # profile = row['simulation_profile']
        profile = row['personal_profile']
        # instruction = row['test_instruction']
        key = profile
        if key not in final_results.keys():
            # final_results[profile+instruction] = {'test_instruction': instruction, row['item']: row['result']}
            final_results[row['personal_profile']] = {row['item']: row['result']}
            # final_results[row['simulation_profile']] = {row['item']: row['result']}
        else:
            # final_results[profile+instruction][row['item']] = row['result']
            final_results[row['personal_profile']][row['item']] = row['result']
            # final_results[row['simulation_profile']][row['item']] = row['result']


    with open('results/gpt4/shape_BFI_new_v2.csv','w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['']+items)
        for key, value in final_results.items():
            writer.writerow([key]+ list(value.values()))

def sort_function(question):
    return int(question.split('.')[0])

def generate_people_simulation_instructions():
    item_file = open("prompts/bfi2_items.txt", "r")
    items = item_file.read().splitlines()

    question_file = open("prompts/chatbot_question.txt", "r")
    questions = question_file.read().splitlines()

    results = pd.read_csv('results/static-v-bot-study.csv')
    answers = results.iloc[:, 99:131]
    answers.columns = questions
    questions.sort(key=sort_function)
    answers = answers[questions]

    profile_list = []
    for idx, row in answers.iterrows():
        profile = ''
        for question, answer in row.items():
            question = '. '.join(question.split('. ')[1:])
            profile += f'Q: {question}\nA: {answer}\n'
        profile_list.append(profile[:-1])

    result = list(product(profile_list, items))
    header = ["simulation_profile", "item"]
    with open('data/simulation_BFI_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)

def id_mapping():
    dir_path = "results/human_interview/transcript/"
    file_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    start_sentence = 'Ava: <p>To get us started, where are you from?</p>'
    end_sentence = 'Ava: <p>That was all the questions about you that I wanted us to chat about! If you want to retain the right to contact my server later to delete your recorded responses, please type your email address below so I have a reference for your entry.</p>'

    profile_list = {}

    for file_path in file_list:
        transcript_file = open(join(dir_path, file_path), "r")
        transcripts = transcript_file.read().splitlines()
        updated_transcripts = []
        for line in transcripts:
            updated_transcripts += line.split('] ')[1:]

        start_index = updated_transcripts.index(start_sentence)
        try:
            end_index = updated_transcripts.index(end_sentence)
            profile_list[file_path]='\n'.join(updated_transcripts[start_index:end_index])
        except:
            profile_list[file_path]='\n'.join(updated_transcripts[start_index:])

    processed_interview = pd.read_excel('results/human_interview/processed_interview.xlsx', index_col=0)
    processed_interview_id = list(processed_interview['transcript'])

    results = pd.read_csv('results/interview_simulation_BFI_new_v2.csv')

    count = 0
    for index, row in results.iterrows():
        transcript = row[0]
        for k,v in profile_list.items():
            if transcript == v and k in processed_interview_id:
                count += 1
                results.loc[index,'ID'] = k.split('.')[0]
                break
    results.to_csv('results/interview_simulation_BFI_new_id.csv', encoding="utf_8_sig", index=False)


def preprocess_transcript():
    item_file = open("prompts/bfi2_items.txt", "r")
    items = item_file.read().splitlines()

    dir_path = "results/human_interview/transcript/"
    file_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    start_sentence = 'Ava: <p>To get us started, where are you from?</p>'
    end_sentence = 'Ava: <p>That was all the questions about you that I wanted us to chat about! If you want to retain the right to contact my server later to delete your recorded responses, please type your email address below so I have a reference for your entry.</p>'

    profile_list = []
    count = 0
    for file_path in file_list:
        transcript_file = open(join(dir_path, file_path), "r")
        transcripts = transcript_file.read().splitlines()
        updated_transcripts = []
        for line in transcripts:
            updated_transcripts += line.split('] ')[1:]

        start_index = updated_transcripts.index(start_sentence)
        try:
            end_index = updated_transcripts.index(end_sentence)
            profile_list.append('\n'.join(updated_transcripts[start_index:end_index]))
        except:
            count+=1
            profile_list.append('\n'.join(updated_transcripts[start_index:]))
    print('Incomplete transcripts: ', count)
    result = list(product(profile_list, items))
    header = ["simulation_profile", "item"]
    with open('data/interview_simulation_BFI_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)


def random_persona_selection():
    persona_file = open("prompts/persona_description.txt", "r")
    personas = persona_file.read().lower().splitlines()

    personalities = pd.read_csv("prompts/personality.csv")
    new_personality_list = []
    for idx, row in personalities.iterrows():
        personality = row['Persona'].lower().strip()
        if personality not in personas:
            new_personality_list.append(personality+'\n')

    random_index_list = random.sample(range(0, len(new_personality_list)), 50)
    res_list = np.take(new_personality_list, random_index_list)

    with open('prompts/persona_description_v2.txt', 'w') as file:
        file.writelines(res_list)

if __name__ == "__main__":
    # generate_trait_instructions()
    # generate_personal_profile()
    # generate_shape_instructions()
    # generate_extreme_personal_profile()
    # generate_people_simulation_instructions()
    post_process()
    convert_format()

    # preprocess_transcript()
    # random_persona_selection()
    # id_mapping()


