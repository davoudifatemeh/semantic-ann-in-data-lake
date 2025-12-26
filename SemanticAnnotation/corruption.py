import json
import nlpaug.augmenter.char as nac
import random

def nlpaug_corruption(text):
    augmenter = random.choice([aug_swap, aug_delete, aug_keyboard])
    return augmenter.augment(text)[0]

aug_swap = nac.RandomCharAug(action="swap")
aug_delete = nac.RandomCharAug(action="delete")
aug_keyboard = nac.KeyboardAug()

# schema = json.load(open("/home/zdavoudi/deepjoin-code/helper/csv_schema_GT.json"))
app_name = "wiki"
schema = json.load(open(f"/home/zdavoudi/deepjoin-code/data/{app_name}/{app_name}_csv_schema.json"))
print(f"{len(schema)} number of tables have been loaded.")
all_column_names = [col['name'] for entry in schema for col in entry['columns']]

for entry in schema:
    num_cols = len(entry['columns'])
    if num_cols == 0:
        print(f"Table {entry['table_id']} has zero columns, skipping corruption.")
        continue
    
    unique_idxs = list(range(num_cols))
    random.shuffle(unique_idxs)

    # for i, column in enumerate(entry['columns']):
    for column, i in zip(entry['columns'], unique_idxs):
        column['corrupt_1'] = f"attribiute_{i}"
        if random.random() < 0.65:
            column['corrupt_2'] = nlpaug_corruption(column['name'])
        else:
            column['corrupt_2'] = column['name']
        column['corrupt_3'] = random.choice(all_column_names)

# with open("/home/zdavoudi/deepjoin-code/helper/ecb_csv_schema_corr.json", "w") as f:
with open(f"/home/fdavoudi/Joinable-Tables-Discovery/Schema Extraction/output/{app_name}/{app_name}_csv_schema_corr.json", "w") as f:
    json.dump(schema, f, indent=4)