import os
import json

if __name__ == "__main__":
    with open("/root/projs/PesticideRecipeGen/data/distill/formulation_names.json", "r") as f:
        obj = json.load(f)
        for key in obj.keys():
            print(f"{key}:{len(obj[key])}")
