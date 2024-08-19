import argparse
from utils import *
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all", "ArXiv"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
parser.add_argument("--min_len", type=int, default=100)
parser.add_argument("--local_data", type=bool)
parser.add_argument("--same_length", type=bool, default=True)
parser.add_argument("--samples", type=int, default=5000)
parser.add_argument("--gradient_collection", type=str, default=False)
parser.add_argument("--dir", type=str, default="absolute_filtered_result")
parser.add_argument("--load_dir", type=str, default="absolute_filtered_dataset")

args = parser.parse_args()

dataset_names = get_dataset_list(args.dataset_name)
if args.model_size == "12b":
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 开启8位量化
        bnb_8bit_use_double_quant=True,  # 使用双重量化技术
        bnb_8bit_compute_dtype=torch.float16  # 计算过程中使用float16
    )
    model = GPTNeoXForCausalLM.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
      torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    ).eval()#cuda(args.cuda).eval()
else:
    model = GPTNeoXForCausalLM.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
      torch_dtype=torch.bfloat16
    ).cuda(args.cuda).eval()
model = model.to_bettertransformer()
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
)

refer_model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2",
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16).cuda(args.refer_cuda).eval()
refer_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
tokenizer.pad_token = tokenizer.eos_token
refer_tokenizer.pad_token = refer_tokenizer.eos_token
for dataset_name in dataset_names:
    df = pd.DataFrame()
    dataset = obtain_dataset(dataset_name, args)
    loss_dict = {}
    prob_dict = {}
    ppl_dict = {}
    mink_plus_dict = {}
    zlib_dict = {}
    refer_dict = {}
    grad_dict = {}
    loss_dict[dataset_name] = {"member": [], "nonmember": []}
    prob_dict[dataset_name] = {"member": [], "nonmember": []}
    ppl_dict[dataset_name] = {"member": [], "nonmember": []}
    mink_plus_dict[dataset_name] = {"member": [], "nonmember": []}
    zlib_dict[dataset_name] = {"member": [], "nonmember": []}
    refer_dict[dataset_name] = {"member": [], "nonmember": []}
    grad_dict[dataset_name] = {"member": [], "nonmember": []}
    for split in ["member", "nonmember"]:
        loss_list, prob_list, ppl_list, mink_plus_list, zlib_list, refer_list, idx_list, grad_list = feature_collection(model, tokenizer, dataset[split], args,
                                                                                                             dataset_name,
                                                                                       min_len = args.min_len,
                                                                                       upper_limit=args.samples,
                                                                                       refer_model=refer_model,
                                                                                       refer_tokenizer=refer_tokenizer,
                                                                                       )
        loss_dict[dataset_name][split].extend(loss_list)
        prob_dict[dataset_name][split].extend(prob_list)
        ppl_dict[dataset_name][split].extend(ppl_list)
        mink_plus_dict[dataset_name][split].extend(mink_plus_list)
        zlib_dict[dataset_name][split].extend(zlib_list)
        refer_dict[dataset_name][split].extend(refer_list)
        grad_dict[dataset_name][split].extend(grad_list)
    os.makedirs(args.dir, exist_ok=True)
    os.makedirs(f"{args.dir}/{dataset_name}", exist_ok=True)
    pickle.dump(loss_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_loss_dict.pkl", "wb"))
    pickle.dump(prob_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_prob_dict.pkl", "wb"))
    pickle.dump(ppl_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_ppl_dict.pkl", "wb"))
    pickle.dump(mink_plus_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_mink_plus_dict.pkl", "wb"))
    pickle.dump(zlib_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_zlib_dict.pkl", "wb"))
    pickle.dump(refer_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_refer_dict.pkl", "wb"))
    pickle.dump(grad_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_grad_dict.pkl", "wb"))
    pickle.dump(idx_list, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_idx_list.pkl", "wb"))
    df = results_caculate_and_draw(dataset_name, args, df, split_set=["member", "nonmember"])
    df.to_csv(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}.csv")



