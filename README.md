# Label_Representation

## Prerequisite
Install dependencies with pip
```
$ pip install -r requirement
```
Download wilds dataset
```
python -c "import wilds; wilds.get_dataset(dataset=\"iwildcam\", download=True, root_dir=$PATH$)"
```
## Training 
```
python train_custom.py --model resnet110 \
					--dataset iwildcam \
					--label category \
					--seed 100 \
					--data_dir /work/zli/wilds \
					--base_dir /work/zli/ \
					--batch_size 1024
```

## Evaluation
```
python eval_models.py
```
