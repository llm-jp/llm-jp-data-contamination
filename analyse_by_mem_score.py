
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import argparse
from utils import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=96)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
parser.add_argument("--reference_model", type=str, default=False)
parser.add_argument("--samples", type=int, default=5000)
parser.add_argument("--dir", type=str, default="mem_score_result")
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names = ["ArXiv", "DM Mathematics",
                  "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                 "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                 "USPTO Backgrounds", "Wikipedia (en)","WikiMIA32","WikiMIA64", "WikiMIA128","WikiMIA256",
                     "WikiMIAall"]
else:
    dataset_names = [args.dataset_name]

dataset_names = get_dataset_list(args.dataset_name)
for dataset_name in dataset_names:
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{args.model_size}-deduped",
        revision="step143000",
        cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
        torch_dtype=torch.bfloat16,
    ).cuda(args.cuda).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{args.model_size}-deduped",
        revision="step143000",
        cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    )
    if args.reference_model == "True":
        refer_model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2",
                                                           trust_remote_code=True,
                                                           torch_dtype=torch.bfloat16).cuda(args.refer_cuda).eval()
        refer_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
    else:
        refer_model = None
        refer_tokenizer = None
    tokenizer.pad_token = tokenizer.eos_token
    refer_tokenizer.pad_token = refer_tokenizer.eos_token
    mem_score_data = pd.read_csv(f"mem_score/{args.model_size}/{dataset_name}_mem_score.csv", index_col=0)
    mem_score_data["original_idx"] = mem_score_data["original_idx"].astype(int)
    train_dict = mem_score_data[mem_score_data['set_name'] == 'train'].set_index('original_idx')['mem_score'].to_dict()
    test_dict = mem_score_data[mem_score_data['set_name'] == 'test'].set_index('original_idx')['mem_score'].to_dict()
    valid_dict = mem_score_data[mem_score_data['set_name'] == 'test'].set_index('original_idx')['mem_score'].to_dict()
    dataset = obtain_dataset(dataset_name)
    loss_dict = {}
    prob_dict = {}
    ppl_dict = {}
    mink_plus_dict = {}
    zlib_dict = {}
    refer_dict = {}
    loss_dict[dataset_name] = {"train": [], "valid": [], "test": []}
    prob_dict[dataset_name] = {"train": [], "valid": [], "test": []}
    ppl_dict[dataset_name] = {"train": [], "valid": [], "test": []}
    mink_plus_dict[dataset_name] = {"train": [], "valid": [], "test": []}
    zlib_dict[dataset_name] = {"train": [], "valid": [], "test": []}
    refer_dict[dataset_name] = {"train": [], "valid": [], "test": []}
    for split in ["train", "test", "valid"]:
        if split == "train":
            mem_score_dict = train_dict
        elif split == "test":
            mem_score_dict = test_dict
        else:
            mem_score_dict = valid_dict
        filtered_results = dataset[split][list(mem_score_dict.keys())]
        loss_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
        prob_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
        ppl_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
        mink_plus_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
        zlib_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
        refer_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_refer_dict.pkl", "rb"))
        idx_list = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_idx_list.pkl", "rb"))





