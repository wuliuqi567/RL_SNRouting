import argparse
import numpy as np
import yaml

def get_configs(file_dir):
    """Get dict variable from a YAML file.
    Args:
        file_dir: the directory of the YAML file.

    Returns:
        config_dict: the keys and corresponding values in the YAML file.
    """
    with open(file_dir, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, file_dir + " error: {}".format(exc)
    return config_dict

def recursive_dict_update(basic_dict, target_dict):
    """Update the dict values.

    Args:
        basic_dict: the original dict variable that to be updated.
        target_dict: the target dict variable with new values.

    Returns:
        A dict mapping keys of basic_dict to the values of the same keys in target_dict.
        For example:

        basic_dict = {'a': 1, 'b': 2}
        target_dict = {'a': 3, 'c': 4}
        out_dict = recursive_dict_update(basic_dict, target_dict)

        output_dict = {'a': 3, 'b': 2, 'c': 4}
    """
    out_dict = deepcopy(basic_dict)
    for key, value in target_dict.items():
        if isinstance(value, dict):
            out_dict[key] = recursive_dict_update(out_dict.get(key, {}), value)
        else:
            out_dict[key] = value
    return out_dict