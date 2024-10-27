import argparse
from utils import *
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


def compute_recall(args):
    dataset_names, length_list = get_dataset_list(args)
    model = GPTNeoXForCausalLM.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
      torch_dtype=torch.bfloat16,
      #load_in_8bit=True,
      device_map=args.cuda
      #quantization_config=bnb_config,
    ).eval()#.to(args.cuda)
    model = model.to_bettertransformer()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    #model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    model.eval()
    #model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    for dataset_idx in list(range(args.dataset_idx, 3)):
        args.dataset_idx = dataset_idx
        for min_len in length_list:
            args.min_len = min_len
            for dataset_name in dataset_names:
                if os.path.exists(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_eda_pac_dict.pkl"):
                    print(f"{dataset_idx} {dataset_name} {args.min_len} {args.model_size} finished")
                    continue
                df = pd.DataFrame()
                dataset = obtain_dataset(dataset_name, args)
                recall_dict = {}
                idx_dict = {}
                recall_dict[dataset_name] = {"member": [], "nonmember": []}
                idx_dict[dataset_name] = {"member": [], "nonmember": []}
                nonmember_prefix = dataset["nonmember"][:12]
                member_data_prefix =dataset["member_data"][:12]
                for split in ["member", "nonmember"]:
                    eda_pac_list, idx_list = recall_collection(model, tokenizer, dataset[split],dataset_name, nonmember_prefix, args, min_len = args.min_len)
                    recall_dict[dataset_name][split].extend(eda_pac_list)
                    idx_dict[dataset_name][split].extend(idx_list)
                os.makedirs(f"{args.save_dir}_{args.dataset_idx}", exist_ok=True)
                os.makedirs(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}", exist_ok=True)
                pickle.dump(idx_dict, open(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_idx_list.pkl", "wb"))
                pickle.dump(recall_dict, open(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_recall_dict.pkl", "wb"))
                df = results_caculate_and_draw(dataset_name, args, df, method_list=["recall"])
                if args.same_length:
                    df.to_csv(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_same_length.csv")
                else:
                    df.to_csv(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_all_length.csv")



