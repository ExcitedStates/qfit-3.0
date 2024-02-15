import os
import pickle

def load_water_rotamer_dict():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "water_data.pkl")
    with open(file_path, "rb") as f:
        water_rotamer_dict = pickle.load(f)
    return water_rotamer_dict