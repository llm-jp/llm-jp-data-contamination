import argparse
from utils import *
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

def compute_gray_box_method(args):
    dataset_names, length_list = get_dataset_list(args)
    if args.model_size == "12b" or args.model_size == "6.9b":
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_8bit=True,  # 开启8位量化
    #         bnb_8bit_use_double_quant=True,  # 使用双重量化技术
    #         bnb_8bit_compute_dtype=torch.float16  # 计算过程中使用float16
    #     )
    #     device_map = {
    #     "transformer.word_embeddings": args.cuda,
    #     "transformer.word_embeddings_layernorm": args.cuda,
    #     "lm_head": "cpu",
    #     "transformer.h": args.cuda,
    #     "transformer.ln_f": args.cuda,
    # }
        model = GPTNeoXForCausalLM.from_pretrained(
          f"EleutherAI/pythia-{args.model_size}-deduped",
          revision="step143000",
          cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
          torch_dtype=torch.bfloat16,
          load_in_8bit=True,
          device_map=args.cuda
          #quantization_config=bnb_config,
        ).eval()#.to(args.cuda)
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
    for min_len in length_list:
        args.min_len = min_len
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
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}", exist_ok=True)
            pickle.dump(loss_dict, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_loss_dict.pkl", "wb"))
            pickle.dump(prob_dict, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_prob_dict.pkl", "wb"))
            pickle.dump(ppl_dict, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_ppl_dict.pkl", "wb"))
            pickle.dump(mink_plus_dict, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_mink_plus_dict.pkl", "wb"))
            pickle.dump(zlib_dict, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_zlib_dict.pkl", "wb"))
            pickle.dump(refer_dict, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_refer_dict.pkl", "wb"))
            pickle.dump(grad_dict, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_grad_dict.pkl", "wb"))
            pickle.dump(idx_list, open(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_idx_list.pkl", "wb"))
            df = results_caculate_and_draw(dataset_name, args, df, method_list=["loss", "prob", "ppl", "mink_plus", "zlib", "refer", "grad"])
            if args.same_length:
                df.to_csv(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_same_length.csv")
            else:
                df.to_csv(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_all_length.csv")



