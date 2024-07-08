nohup python distribution_validating.py --model_size 160m --dataset_name Pile-CC --cuda 0 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &
nohup python distribution_validating.py --model_size 160m --dataset_name StackExchange --cuda 1 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &
nohup python distribution_validating.py --model_size 160m --dataset_name Github --cuda 2 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &
nohup python distribution_validating.py --model_size 160m --dataset_name "PubMed Abstracts" --cuda 3 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &
nohup python distribution_validating.py --model_size 160m --dataset_name "USPTO Backgrounds" --cuda 4 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &
nohup python distribution_validating.py --model_size 160m --dataset_name FreeLaw --cuda 5 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &
nohup python distribution_validating.py --model_size 160m --dataset_name EuroParl --cuda 6 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &
nohup python distribution_validating.py --model_size 160m --dataset_name HackerNews --cuda 7 --skip_calculation False  --samples 5000 --reference_model False --samples 5000 --reference_model False &


