import argparse
import random
import csv
from itertools import product
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np


def sort_function(question):
    return int(question.split('.')[0])

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
        starter = ', ' + qualifier + ' '
        for index_list in all_index:
            profile = ''
            for idx, value in enumerate(index_list):
                if value:
                    profile += qualifier + ' ' + starter.join(values[idx]['low_marker'][:-1]) + ' and {} {}, '.format(
                        qualifier,
                        values[idx]['low_marker'][-1])
                else:
                    profile += qualifier + ' ' + starter.join(values[idx]['high_marker'][:-1]) + ' and {} {}, '.format(
                        qualifier,
                        values[idx]['high_marker'][-1])
            file.write(profile + '\n')


def random_persona_selection(persona_save_path, sample_num=250):
    persona_file = open("data/persona_description.txt", "r")

    personas = persona_file.read().lower().splitlines()

    personalities = pd.read_csv("data/personality.csv")
    new_personality_list = []
    for idx, row in personalities.iterrows():
        personality = row['Persona'].lower().strip()
        if personality not in personas:
            new_personality_list.append(row['Persona'] + '\n')

    random_index_list = random.sample(range(0, len(new_personality_list)), sample_num)
    res_list = np.take(new_personality_list, random_index_list)

    with open(persona_save_path, 'w') as file:
        combined_list = ['\n'.join(personas), '\n'] + list(res_list)
        file.writelines(combined_list)


def generate_persona_instructions(persona_save_path, questionnaire):
    persona_file = open(persona_save_path, "r")
    item_file = open(f"data/{questionnaire}_items.txt", "r")

    persona = persona_file.read().splitlines()
    item = item_file.read().splitlines()

    result = list(product(persona, item))

    header = ["persona_description", "item"]
    with open(f'output/instructions/persona_{questionnaire}_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)

    persona_file.close()
    item_file.close()


def generate_personal_profile(profile_save_path, sample_num=None, adj_num=None):
    '''
        sample_num: the number of personal profile in each category
    '''
    qualifier_file = open("data/qualifier.txt", "r")

    adjectival_markers = pd.read_csv("data/adjectival_markers.csv")
    qualifiers = qualifier_file.read().splitlines()

    # Classify adjective words
    adjectives = {}
    for index, row in adjectival_markers.iterrows():
        domain = row['Domain']
        if domain not in adjectives.keys():
            adjectives[domain] = {'low_marker': [row['Low Marker']], 'high_marker': [row['High Marker']]}
        else:
            adjectives[domain]['low_marker'].append(row['Low Marker'])
            adjectives[domain]['high_marker'].append(row['High Marker'])

    with open(profile_save_path, 'a') as file:
        # Generate personal profiles
        for qualifier in qualifiers:
            if qualifier:
                connector = ', ' + qualifier + ' '
                starter = qualifier + ' '
            else:
                connector = ', '
                starter = ''

            for domain, markers in adjectives.items():
                if adj_num:  # Generate personal profiles with a specified number of adj words
                    save_index = []
                    while len(save_index) < sample_num:
                        adj_index_list = random.sample(range(0, len(markers['low_marker'])), adj_num)
                        low_adj_list = np.take(markers['low_marker'], adj_index_list)
                        high_adj_list = np.take(markers['high_marker'], adj_index_list)
                        if adj_index_list not in save_index:
                            save_index.append(adj_index_list)

                            file.write(starter + connector.join(low_adj_list[:-1]) + ' and {} {}\n'.format(qualifier,
                                                                                                           low_adj_list[
                                                                                                               -1]))
                            file.write(starter + connector.join(high_adj_list[:-1]) + ' and {} {}\n'.format(qualifier,
                                                                                                            high_adj_list[
                                                                                                                -1]))
                else:
                    file.write(starter + connector.join(markers['low_marker'][:-1]) + ' and {} {}\n'.format(qualifier,
                                                                                                            markers[
                                                                                                                'low_marker'][
                                                                                                                -1]))
                    file.write(starter + connector.join(markers['high_marker'][:-1]) + ' and {} {}\n'.format(qualifier,
                                                                                                             markers[
                                                                                                                 'high_marker'][
                                                                                                                 -1]))

        # Generate neutral personal profiles
        for domain, markers in adjectives.items():
            if adj_num:
                save_index = []
                while len(save_index) < sample_num / 2:
                    adj_index_list = random.sample(range(0, len(markers['low_marker'])), adj_num)
                    if adj_index_list not in save_index:
                        save_index.append(adj_index_list)
                        low_adj_list = np.take(markers['low_marker'], adj_index_list)
                        high_adj_list = np.take(markers['high_marker'], adj_index_list)

                        adj_list = []
                        for low_marker, high_marker in zip(low_adj_list[:-1], high_adj_list[:-1]):
                            adj_list.append('neither {} nor {}, '.format(low_marker, high_marker))
                        file.write(
                            ''.join(adj_list) + 'and neither {} nor {}\n'.format(low_adj_list[-1], high_adj_list[-1]))
            else:
                adj_list = []
                for low_marker, high_marker in zip(markers['low_marker'][:-1], markers['high_marker'][:-1]):
                    adj_list.append('neither {} nor {}, '.format(low_marker, high_marker))
                file.write(''.join(adj_list) + 'and neither {} nor {}\n'.format(markers['low_marker'][-1],
                                                                                markers['high_marker'][-1]))

    qualifier_file.close()


def generate_shape_instructions(profile_save_path, questionnaire):
    profile_file = open(profile_save_path, "r")
    item_file = open(f"data/{questionnaire}_items.txt", "r")

    profile = profile_file.read().splitlines()
    item = item_file.read().splitlines()

    result = list(product(profile, item))

    header = ["personal_profile", "item"]
    with open(f'output/instructions/shape_{questionnaire}_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)

    profile_file.close()
    item_file.close()


def generate_interview_simulation_instructions(questionnaire):
    item_file = open(f"data/{questionnaire}_items.txt", "r")
    items = item_file.read().splitlines()

    question_file = open("data/chatbot_question.txt", "r")
    questions = question_file.read().splitlines()

    results = pd.read_csv('data/static-v-bot-study.csv')
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
    with open(f'output/instructions/interview_simulation_{questionnaire}_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)


def generate_transcript_based_instruction(questionnaire):
    item_file = open(f"data/{questionnaire}_items.txt", "r")
    items = item_file.read().splitlines()

    dir_path = "data/human_interview/transcript/"
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
            count += 1
            profile_list.append('\n'.join(updated_transcripts[start_index:]))

    print('Incomplete transcript number: ', count)
    result = list(product(profile_list, items))
    header = ["simulation_profile", "item"]
    with open(f'output/instructions/transcript_simulation_{questionnaire}_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)


def id_mapping(questionnaire):
    dir_path = "data/human_interview/transcript/"
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
            profile_list[file_path] = '\n'.join(updated_transcripts[start_index:end_index])
        except:
            profile_list[file_path] = '\n'.join(updated_transcripts[start_index:])

    processed_interview = pd.read_excel('data/transcript/processed_interview.xlsx', index_col=0)
    processed_interview_id = list(processed_interview['transcript'])

    results = pd.read_csv(f'output/instructions/transcript_simulation_{questionnaire}_instruction.csv')

    count = 0
    for index, row in results.iterrows():
        transcript = row[0]
        for k, v in profile_list.items():
            if transcript == v and k in processed_interview_id:
                count += 1
                results.loc[index, 'ID'] = k.split('.')[0]
                break
    results.to_csv(f'output/instructions/filtered_transcript_simulation_{questionnaire}.csv', encoding="utf_8_sig",
                   index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--description', type=str,
                           choices=['persona_description', 'personal_profile', 'interview', 'transcript'])
    argparser.add_argument('--questionnaire', type=str, choices=['bfi60', 'ipip300', 'hexaco', 'force_choice'])
    argparser.add_argument('--persona_save_path', type=str,
                           default='output/description/persona_description.txt')
    argparser.add_argument('--profile_save_path', type=str, default='output/description/personal_profile.txt')

    args = argparser.parse_args()

    # Instruction Generation
    if args.description == 'persona_description':  # Persona simulation
        random_persona_selection(args.persona_save_path)
        generate_persona_instructions(args.persona_save_path, args.questionnaire)
    elif args.description == 'personal_profile':  # Shape simulation
        generate_personal_profile(args.profile_save_path)
        generate_personal_profile(args.profile_save_path, 6, 5)
        # profile_file = open("output/description/personal_profile.txt")
        #
        # personalities = profile_file.read().splitlines()
        # random_index_list = random.sample(range(0, len(personalities)), 100)
        # res_list = np.take(personalities, random_index_list)
        # with open('output/description/sampled_personal_profile_sample.txt', 'w') as file:
        #     file.writelines('\n'.join(res_list))
        generate_shape_instructions(args.profile_save_path, args.questionnaire)
    elif args.description == 'interview':
        generate_interview_simulation_instructions(args.questionnaire)
    elif args.description == 'transcript':
        generate_transcript_based_instruction(args.questionnaire)
        id_mapping(args.questionnaire)
