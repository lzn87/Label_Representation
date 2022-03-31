# Label_Representation

## Prerequisite
Install dependencies with pip
```
$ pip install -r requirement
```
Download iwildcam dataset
```
python -c "import wilds; wilds.get_dataset(dataset=\"iwildcam\", download=True, root_dir=$PATH$)"
```
## Training 
Model can be ```vgg19```, ```resnet110``` or ```resnet34```.
To train a category model:
```
python train_custom.py --model $MODEL$ \
					--dataset iwildcam \
					--label $LABEL$ \
					--seed 100 \
					--data_dir $PATH$ \
					--base_dir . \
					--batch_size 1024
```

To train a speech model:
```
python train_custom.py --model $MODEL$ \
					--dataset iwildcam \
					--label $LABEL$ \
					--seed 100 \
					--data_dir $PATH$ \
					--base_dir . \
					--batch_size 1024 \
					--label_dir $LABEL_DIR$
```
## Evaluation
```
python eval_models.py
```
