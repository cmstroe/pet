import json
import pandas as pd 

def group(df, tag):
    return df.groupby(df.label).get_group(tag)

def create_dataset(df):
    dataset = pd.DataFrame(columns = ['label', 'text'])
    for key, elem in df.iterrows():
        dataset = dataset.append({'label' : 1, 'text': elem.sentence.replace(',', ' ')},ignore_index=True)
    return dataset

def create_csv(df, tag):
    df.to_csv(tag+ "_full.csv", index = False)

def add_zeros(df, zero_dataset):
    count = 0 
    max_length = len(df)
    for key,elem in zero_dataset.iterrows():
        if count < max_length:
            df = df.append({'label' : 0, 'text': elem.sentence.replace(',', ' ')},ignore_index=True)
            count += 1
    return df

def create_unlabeled_dataset(file, df):
    dataset = pd.DataFrame(columns = ['label', 'text'])
    for obj in file[:40]:
        count = 0
        for sentence in obj['sentences']:
            if sentence not in df.text.astype(str).values.tolist() and count <= 15:
                dataset = dataset.append({'label' : '', 'text': sentence.replace(',', ' ')},ignore_index=True)
                count+=1
            else:
                print("patesti")
            if count == 15:
                break
    return dataset


if __name__ == "__main__":
    df = pd.read_csv("preprocessing_new_tags/annotations_news_orig.csv")

    df_ma  = create_dataset(group(df, 'M&A'))
    df_financials = create_dataset(group(df, 'financials'))
    df_partnership  = create_dataset(group(df, 'partnership'))
    df_funding  = create_dataset(group(df ,"funding"))
    df_clients = create_dataset(group(df, "client"))

    df_funding = df_funding.append(df_clients, ignore_index = True)

    unlabeled_data_1 =  group(df, "general updates")[:70]
    unlabeled_data_2 =  group(df, "none/useless")[:70]
    
    
    unlabeled_data = unlabeled_data_1.append(unlabeled_data_2, ignore_index = True).sample(frac=1)
    
    # total_data_ma = add_zeros(df_ma, unlabeled_data)
    # create_csv(total_data_ma, "ma")

    # total_data_financials = add_zeros(df_financials, unlabeled_data)
    # create_csv(total_data_financials, "financials")

    # total_data_partnership = add_zeros(df_partnership, unlabeled_data)
    # create_csv(total_data_partnership, "partenership")

    total_data_funding = add_zeros(df_funding, unlabeled_data)
    create_csv(total_data_funding, "funding")


    with open("preprocessing_new_tags/company_sentences_euro.json") as jsonFile:
        json_obj = json.load(jsonFile)

    # unlabeled_fianncials = create_csv(create_unlabeled_dataset(json_obj, df_financials), "financials")
    # unlabeled_ma = create_csv(create_unlabeled_dataset(json_obj, df_ma), "M&A")
    # unlabeled_partnership = create_csv(create_unlabeled_dataset(json_obj, df_partnership), "partnership")
    # unlabeled_funding = create_csv(create_unlabeled_dataset(json_obj, df_funding), "funding")
