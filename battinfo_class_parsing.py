import json 


json_file = open("context.json")
batt_info_json = json.load(json_file)

classes = [key for key in batt_info_json["@context"].keys()]
skip_index = classes.index("3DPrinting")

classes=classes[200:]

with open("classes.txt", "w") as f:
    for idx, batt_info_class in enumerate(classes):
        f.write(batt_info_class+" ")


