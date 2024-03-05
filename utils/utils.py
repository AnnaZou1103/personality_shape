import re
from collections import Counter
import csv
from itertools import product

import pandas as pd
from itertools import permutations

def majority_vote(output):
    '''
    This function takes a list called output and returns the most common element
    in the list. If there is a tie between elements, the function returns one of
    the tied elements that appears first in the list.
    '''
    x = Counter(output)
    return x.most_common(1)[0][0]


def generate_trait_instructions():
    persona_file = open("prompts/persona_description.txt", "r")
    instruction_file = open("prompts/test_instruction.txt", "r")
    item_file = open("prompts/item_description.txt", "r")

    persona = persona_file.read().splitlines()
    instruction = instruction_file.read().splitlines()
    item = item_file.read().splitlines()

    result = list(product(persona, instruction, item))

    header = ["persona_description", "test_instruction", "item"]
    with open('data/instruction_example.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)

    persona_file.close()
    instruction_file.close()
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
    persona_file = open("prompts/persona_description.txt", "r")
    # profile_file = open("data/personal_profile.txt", "r")
    item_file = open("prompts/item_description_BFI.txt", "r")

    persona = persona_file.read().splitlines()
    # profile = profile_file.read().splitlines()
    item = item_file.read().splitlines()

    result = list(product(persona, item))
    # result = list(product(profile, item))

    header = ["persona_description", "item"]
    # header = ["personal_profile", "item"]
    with open('data/persona_BFI_instruction.csv', 'w') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(result)

    persona_file.close()
    # profile_file.close()
    item_file.close()

def post_process():
    items = pd.read_csv('results/shape_BFI.csv')
    new_result = []
    for score in items['result']:
        if len(re.findall('\d', score))>0:
            new_result.append(re.findall('\d', score)[0])
        else: new_result.append(score)

    items['result'] = pd.Series(new_result)
    items.to_csv('results/shape_BFI_new.csv', encoding="utf_8_sig", index=False)


def convert_format():
    item_file = open("prompts/item_description.txt", "r")
    items = item_file.read().splitlines()
    
    results = pd.read_csv('results/persona_force_multiple_instruction.csv')
    final_results = {}
    
    for idx, row in results.iterrows():
        # profile = row['personal_profile']
        profile = row['persona_description']
        instruction = row['test_instruction']
        key = profile+instruction
        if key not in final_results.keys():
            final_results[profile+instruction] = {'test_instruction': instruction, row['item']: row['result']}
            # final_results[row['personal_profile']] = {row['item']: row['result']}
        else:
            final_results[profile+instruction][row['item']] = row['result']
            # final_results[row['personal_profile']][row['item']] = row['result']


    with open('results/persona_force_multiple_v2.csv','w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['','test_instruction']+items)
        for key, value in final_results.items():
            writer.writerow([key]+ list(value.values()))


if __name__ == "__main__":
    # post_process()
    convert_format()
    # generate_trait_instructions()
    # generate_personal_profile()
    # generate_shape_instructions()
    # generate_extreme_personal_profile()


