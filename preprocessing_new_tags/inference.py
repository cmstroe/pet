import json 
import torch


with open("preprocessing_new_tags/company_sentences_euro.json") as jsonFile:
    json_obj = json.load(jsonFile)

sentences = [], j = 0

for i in range (100, 300):
    obj = json_obj[i]
    sentences[j] = obj['sentences']
    j+=1

model = torch.load('pytorch_model.bin', map_location='cpu')

