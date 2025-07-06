import numpy as np
import json

import os
import json


# def load_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

def fix_json_format(json_str):
    """
    Fix JSON format errors (e.g., missing commas).
    """
    # Fix }{ and ][ missing commas
    fixed_json_str = json_str.replace("}{", "},{").replace("][", "],[")
    return fixed_json_str


def load_and_fix_json(file_path):
    """
    Read the JSON file and try to fix format errors.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_str = f.read()

    try:
        # Try to parse JSON directly
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Detected JSON format error: {e}")
        # Try to fix JSON format
        fixed_json_str = fix_json_format(json_str)
        try:
            data = json.loads(fixed_json_str)
            print("JSON file fixed and loaded successfully.")
        except json.JSONDecodeError as e:
            print(f"Fix failed, unable to load JSON file: {e}")
            return None
    return data


# Process two four-layer nested arrays to generate new observation data
def process_observations(obs1, obs2):
    new_obs = []  # Store processed observation data

    # Traverse each observation group
    for group1, group2 in zip(obs1, obs2):
        new_group = []  # Store processed observation group
        # Traverse each 2D array

        # Traverse each row of the matrix
        for row1, row2 in zip(group1, group2):
            new_row = []  # Store processed row
            # Compare first 12 bits (AND operation) and 13th bit (OR operation)
            for col_index in range(len(row1)):
                if col_index < 12:
                    # First 12 bits perform AND operation
                    new_row.append(row1[col_index] and row2[col_index])
                elif col_index == 12:
                    # 13th bit perform OR operation
                    new_row.append(row1[col_index] or row2[col_index])
                else:
                    # Other bits normal comparison
                    new_row.append(row1[col_index] and row2[col_index])

            new_group.append(new_row)
        new_obs.append(new_group)

    return new_obs


def p0_data(data):
    # input_file = 'game_history_20250225_152757.json'  # Original file path
    # output_file = 'processed_observations.json'  # Output file path
    # data = load_json_file(input_file)

    # Traverse each step, process observation data
    for step_data in data:
        observations = step_data['observations']
        if len(observations) < 2:
            print(f"Step {step_data['step']}: observation groups insufficient, unable to process.")
            continue

        obs1 = observations[0]
        obs2 = observations[1]

        # Process two groups of observation data
        new_obs = process_observations(obs1, obs2)

        # Store processed observation data in new structure
        step_data['observations'] = new_obs
        step_data['actions'] = step_data['actions'][1]

    return data


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int32):
        return float(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def flatten_to_1d_with_subarrays(multi_dim_array):
    flattened = []
    for sub_array in multi_dim_array:
        if isinstance(sub_array, list) and any(isinstance(item, list) for item in sub_array):
            # If sub-array is multi-dimensional, flatten it
            flattened.extend(flatten_to_1d_with_subarrays(sub_array))
        else:
            # If sub-array is one-dimensional or minimal array, keep unchanged
            flattened.append(sub_array)
    return flattened


def dim_reduction(data):
    processed_data = []
    for item in data:
        step = item['step']
        # # Flatten multi-dimensional arrays in observation to one-dimensional arrays, but keep minimal arrays unchanged
        # flattened_observation = flatten_to_1d_with_subarrays(item['observation'])

        # # Flatten multi-dimensional arrays in action to one-dimensional arrays, but keep minimal arrays unchanged
        # flattened_action = flatten_to_1d_with_subarrays(item['action'])

        # flattened_observations = flatten_to_1d_with_subarrays(item['p1']['observations'])
        # flattened_actions1 = flatten_to_1d_with_subarrays(item['p1']['actions'])
        # flattened_actions2 = flatten_to_1d_with_subarrays(item['p2']['actions'])

        flattened_observations = flatten_to_1d_with_subarrays(item['observations'])
        flattened_actions1 = flatten_to_1d_with_subarrays(item['actions'])

        processed_item = {
            'step': step,
            'observations': flattened_observations,
            'action': flattened_actions1,
            'rewards': item['rewards']
        }
        processed_data.append(processed_item)

    return processed_data


def merge_observations_and_actions(data):
    """
    Merge p1's observations and p2's actions in JSON data one-to-one.
    If lengths are inconsistent, fill the shorter part with zeros.
    The merged data is named `state`, and the `step` field is retained.
    """
    merged_data = []

    for step_info in data:
        step = step_info['step']
        observations = step_info['observations']
        actions = step_info['action']

        # Determine the maximum length of p1 and p2 in each step
        max_length = max(len(observations), len(actions))

        # Initialize merged state
        merged_step = []

        for i in range(max_length):
            # Get p1's observation, fill with zeros if out of range
            observation = observations[i] if i < len(observations) else [0] * len(observations[0])

            # Get p2's action, fill with zeros if out of range
            action = actions[i] if i < len(actions) else [0] * len(actions[0])

            # If p1's observation is all zeros, set the corresponding p2's action to all zeros
            if all(obs == 0 for obs in observation):
                action = [0] * len(action)

            # Merge p1's observation and p2's action
            state = observation + action
            merged_step.append(state)

        # Retain step field, and name the merged data state
        merged_data.append({
            "step": step,
            "state": merged_step
        })

    return merged_data


# def count_state_arrays(data):

#     for step_info in data:
#     for step_info in data:
#         state = step_info.get("state", [])
#         state_count = len(state)  # Count the number of state arrays

#     return state_count

# def merge_ob(data):
#     """
#     Process each step's state data.
#     Replace the first 27 bits of the state according to the grouping method.
#     """
#     group_sizes = [5, 5, 3, 8, 6]  # Group sizes
#     total_size = sum(group_sizes)  # Total 27 bits

#     result = []

#     for step_info in data:
#         step = step_info["step"]
#         states = step_info["state"]
#         processed_states = []

#         for state in states:
#             new_state = state[total_size:]  # Keep the part of the state after the 27th bit
#             start = 0

#             # Process the first 27 bits, according to the grouping method
#             for group_size in group_sizes:
#                 group = state[start:start + group_size]
#                 index = group.index(1) if 1 in group else -1  # Find the index of value 1, if not present then 0
#                 new_state.insert(0, index+1)  # Insert the new value at the beginning of the new state
#                 start += group_size

#             processed_states.append(new_state)

#         result.append({"step": step, "state": processed_states})

#     return result

def process_state(state):
    # First 27 bits
    first_27 = state[:29]

    # Grouping
    groups = [
        first_27[:5],  # First 5 bits
        first_27[5:10],  # Next 5 bits
        first_27[10:13],  # Next 3 bits
        first_27[13:21],  # Next 8 bits
        first_27[21:27]  # Last 6 bits
    ]

    # Generate new 5 numbers
    new_numbers = []
    for group in groups:
        # print("Current group", group)
        # Find the index of value 1 and add 1
        indices = [i + 1 for i, val in enumerate(group) if val == 1]
        # If all zeros, use 0 as replacement
        if not indices:
            new_numbers.append(0.0)
            # print("Index 0")
            # break
        else:
            # print("Index", int(''.join(map(str, indices))))
            # Combine indices into a number
            num = int(''.join(map(str, indices)))
            new_numbers.append(float(num))
        # print("end")
    # Replace first 29 bits
    new_state = new_numbers + state[29:]
    return new_state


def merge_ob(data):
    # Process each step's state
    for step in data:
        step["state"] = [process_state(state) for state in step["state"]]

    return data


def process_data(data):
    processed_data = p0_data(data)  # Check format
    processed_data = convert_to_serializable(processed_data)  # Normalize format
    processed_data = dim_reduction(processed_data)  # Reduce dimensions
    processed_data = merge_observations_and_actions(processed_data)  # Merge observations and actions
    processed_data = merge_ob(processed_data)  # Reduce observation bits

    return processed_data

# file_path1 = 'game_history_20250225_152757.json'
# with open(file_path1, 'r') as file:
#     data = json.load(file)

# processed_data = dim_reduction(data)


# with open('a7.json', 'w') as f:
#     json.dump(processed_data, f, indent=4)

# # print("count:",count)
# print("ok了z")


# # Specify folder path
# folder_path = "/mnt/671cbd8b-55cf-4eb4-af6d-a4ab48e8c9d2/JL/PPOGDI-2"
# target_folder = "/home/hello/sth/rts/MicroRTS-Py/experiments/JL/PPOGDI"
# os.makedirs(target_folder, exist_ok=True)
# # print("hh")
# # Get all filenames in the folder
# files = os.listdir(folder_path)
#
# # Filter out JSON files
# json_files = [file for file in files if file.endswith(".json")]
#
# # Sort by filename (optional)
# json_files.sort()
# i = 0
# print(i)
# # print(json_files)
# # Sequentially read each JSON file
# for json_file in json_files:
#     i = i + 1
#     print(i)
#     file_path = os.path.join(folder_path, json_file)
#     # with open(file_path, "r", encoding="utf-8") as f:
#     #     data = json.load(f)
#
#     data = load_and_fix_json(file_path)
#     if data is None:
#         print(f"File {json_file} cannot be fixed, skipping processing.")
#         continue
#
#     processed_data = process_data(data)
#
#     # tu_data = convert_state_to_tensor(processed_data)
#
#     target_path = os.path.join(target_folder, f"{i}.json")
#
#     # print(tu_data)
#     print("ok了")
#
#     with open(target_path, 'w') as f:
#         json.dump(processed_data, f, indent=4)
#
# print("end")