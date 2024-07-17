import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data, clean_dataset
import argparse
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="all", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--gradient_collection", type=str, default=False)
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names = ["ArXiv", "DM Mathematics",
                     "FreeLaw", "Github", "HackerNews", "NIH ExPorter",
                      "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                      "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
    #dataset_names = ["Github", "HackerNews", "NIH ExPorter","Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
    #                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
else:
    dataset_names = [args.dataset_name]

skip_calculation = False
model = GPTNeoXForCausalLM.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
).half().cuda(args.cuda).eval()
#model = model.to_bettertransformer()

tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True


f = open(f"{args.model_size}_embedding_result.txt", "w")
for dataset_name in dataset_names:
    dataset = form_dataset(dataset_name)
    device = f'cuda:{args.cuda}'
    member_embed_list = {}
    non_member_embed_list = {}
    for set_name in ["train", "test"]:
        cleaned_dataset = clean_dataset(dataset)
        for idx, batch in tqdm(enumerate(batched_data(dataset[set_name], batch_size=args.batch_size))):
            if idx * args.batch_size > args.samples:
                break
            batched_text = [item for item in batch]
            input_ids = tokenizer(batched_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            entropy = []
            for ratio in [0.2, 0.4, 0.6, 0.8]:
                generations = model.generate(input_ids[0][:int(input_ids.shape[1]*ratio)], temperature=0.0,
                                             top_k=0, top_p=0, max_length=input_ids[0][:int(input_ids.shape[1]*ratio+0.2)],
                                             min_length=input_ids[0][:int(input_ids.shape[1]*ratio+0.2)])
                logits = generations["scores"]
                pdb.set_trace()
                probability_scores = torch.nn.functional.softmax(logits[0].float(), dim=1)
                entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
                entropy.append(entropy_scores)





