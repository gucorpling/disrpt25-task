import disrptdata

dataset = disrptdata.get_combined_dataset(1, 30)
print(dataset['dev'][1])