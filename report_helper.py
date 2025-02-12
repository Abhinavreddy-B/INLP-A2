import os

all_benchmark_files = os.listdir("benchmarks")

mp = {}

for file in all_benchmark_files:
    with open("benchmarks/" + file, "r") as f:
        avg_perplexity = float(f.readline().strip())

        [model, dataset] = file.split("-")
        dataset = dataset.split(".")[0]

        no_of_eps = int(model.split("_")[-1][:-2])
        model = model.split("_")[:-1]

        corpus = model[1]
        if model[0] == "f":
            model_type = model[0] + '_' + model[2]
        else:
            model_type = model[0]

        model_tup = (corpus, model_type, no_of_eps, dataset)
        mp[model_tup] = avg_perplexity

corpus_list = ['p', 'u']
model_types = ['f_3n', 'f_5n', 'r', 'l']
no_of_eps = [5, 10, 15, 20, 25, 50]

print('| corpus\t| model\t| epochs\t| train perplexity\t| test perplexity\t|')
print('|-------|-------|-------|----------------|----------------|')
for corpus in corpus_list:
    for model_type in model_types:
        for no_of_ep in no_of_eps:
            if((corpus, model_type, no_of_ep, 'train') in mp and (corpus, model_type, no_of_ep, 'test') in mp):
                print(f"| {corpus}\t| {model_type}\t| {no_of_ep}\t| {mp[(corpus, model_type, no_of_ep, 'train')]:.2f}\t| {mp[(corpus, model_type, no_of_ep, 'test')]:.2f}\t|")