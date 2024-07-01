python distribution_validating.py --model_size 160m --dataset_name Pile-CC --cuda 0 --skip_calculation False
python distribution_validating.py --model_size 160m --dataset_name StackExchange -- cuda 1 --skip_calculation False
python distribution_validating.py --model_size 160m --dataset_name Github --cuda 2 --skip_calculation False
python distribution_validating.py --model_size 160m --dataset_name "PubMed Abstracts" --cuda 3 --skip_calculation False
python distribution_validating.py --model_size 160m --dataset_name "USPTO Backgrounds" --cuda 4 --skip_calculation False
python distribution_validating.py --model_size 160m --dataset_name "Wikipedia (en)" --cuda 5 --skip_calculation False
