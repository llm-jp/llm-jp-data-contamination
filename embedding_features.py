import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data
import argparse
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=5000)
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
model = model.to_bettertransformer()

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
for dataset_name in dataset_names:
    dataset = form_dataset(dataset_name)
    device = f'cuda:{args.cuda}'
    loss_collect = []
    mink_collect = []
    mink_plus_collect = []
    ppl_collect = []
    zlib_collect = []
    ref_loss_collect = []
    for set_name in ["train", "validation", "test"]:
        for batch in tqdm(batched_data(dataset[set_name], batch_size=args.batch_size)):
            batched_text = [item for item in batch]
            tokenized_inputs = tokenizer(batched_text,
                                         return_tensors="pt",
                                         truncation=True,
                                         padding=True,
                                         max_length=2048,
                                         )
            tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
            target_labels = tokenized_inputs["input_ids"].clone().to(device)
            target_labels[tokenized_inputs["attention_mask"] == 0] = -100
            with torch.no_grad():
                outputs = model(**tokenized_inputs, labels=target_labels, output_attentions=True,
                                output_hidden_states=True, return_dict=True)
            pdb.set_trace()
