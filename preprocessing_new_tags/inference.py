import json 


with open("preprocessing_new_tags/company_sentences_euro.json") as jsonFile:
    json_obj = json.load(jsonFile)

print(json_obj)
