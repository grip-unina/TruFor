# Setup

To set up a **conda environment**, run:
```
conda env create -f trufor_conda.yaml
```

# Train

## Pretrained weights

The pretrained Noiseprint++ and SegFormer-B2 weights are already included in the github in the `pretrained_models` folder.

To download the final TruFor trained weights (not required for training) see instructions in the inference section.

## Training datasets

Before running the training, update the datasets paths in `project_config.py`.

To download the datasets used in the paper:
- tampCOCO and compRAISE: https://github.com/mjkwon2021/CAT-Net
- FantasticReality: there is a link by CAT-Net authors here https://github.com/mjkwon2021/CAT-Net/issues/51
- CASIA 2.0 revised: https://github.com/namtpham/casia2groundtruth
- IMD: https://staff.utia.cas.cz/novozada/db/IMD2020.zip

To add your own dataset:
- create a dataloader in the `dataset` folder (you may use the existing ones as a reference)
- add it in the `data_core.py` file (both in section `mode == "train"` and `mode == "valid"`)
- to use the dataset add it to the list in the `DATASET.TRAIN` and/or `DATASET.VALID` options in the config file

## Flags and outputs

Flags:
- `-g` or `--gpu`: default is gpu '0'. Put '-1' if you want to use cpu. You can run on multiple gpus on the same device (e.g. `-g 0 1`).
- `-exp` or `--experiment`: name of the experiment. It must have the same name as the config file (without the extension).

Any other config option (to change the values without editing the .yaml file) has to be put in the end of the command in the form `NAME.OF.PARAMETER value_of_parameter`, using the parameter names included in the config file. 
For example, to perform an extra validation step before the training starts, you can add at the end of the command
`VALID.FIRST_VALID True`. 
To change the batch size, change it in the `TRAIN.BATCH_SIZE_PER_GPU` setting.


## Training using provided configs (to replicate the paper's results)
### Phase 1: training the *Noiseprint++ extractor* (optional)
This step is optional, as you can use our Noiseprint++ weights.
Code for the training of Noiseprint++ is not yet available.

### Phase 2: training the *localization network*

```
python train.py -exp trufor_ph2
```

### Phase 3: training the *detection network* and the *confidence estimator*

First of all, make sure that `TRAIN.PRETRAINING` in `lib/config/trufor_ph3.yaml` contains the path to the weights of phase 2. Then run:

```
python train.py -exp trufor_ph3
```

You can also specify it directly in the command, without editing the yaml:

```
python train.py -exp trufor_ph3 TRAIN.PRETRAINING "weights/trufor_ph2/best.pth.tar"
```


## Custom training

If you want to create your own training, duplicate `trufor_ph2.yaml` and `trufor_ph3.yaml` in the `lib/config` folder, rename and edit them according to your needs.
Then, follow the same training instructions as above, using the name of your config files in `-exp`.

**Remember to update the `TRAIN.PRETRAINING` value either in the yaml of ph3 (or in the command itself) with the path to the ph2 weights.**


# Inference

## Flags and outputs

Flags:
- `-g` or `--gpu`: default is gpu '0'. Put '-1' if you want to use cpu.
- `-in` or `--input`:  default is "images/". It can be a single file, a directory, or a glob statement
- `-out` or `--output`: output folder
- `-exp` or `--experiment`: name of the experiment. It must have the same name as the config file (without the extension).
- `--save_np`: if you want to save the Noiseprint++ aswell

Any other config option (to change the values without editing the .yaml file) has to be put in the end of the command in the form `NAME.OF.PARAMETER value_of_parameter`, using the parameter names included in the config file. 
For example, `TEST.MODEL_FILE "pretrained_models/trufor.pth.tar"`


The output is a .npz containing the following files:
- **'map'**: anomaly localization map
- **'conf'**: confidence map
- **'score'**: score in the range [0,1]
- **'np++'**: Noiseprint++ (if flag `--save_np` is specified)
- **'imgsize'**: size of the image

## Inference using our provided weights (no training required)

Download the [weights](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip) and unzip them in the "pretrained_models" folder (or wherever you prefer). 
MD5 is 7bee48f3476c75616c3c5721ab256ff8.

Then run:
```
python test.py -in path/to/image_or_folder -out path/to/output_folder -exp trufor_ph3 TEST.MODEL_FILE "pretrained_models/trufor.pth.tar" 
```

## Inference using your trained weights
The `TEST.MODEL_FILE` option is not required as it uses the name specified in `-exp`.
```
python test.py -in path/to/image_or_folder -out path/to/output_folder -exp name_of_your_yaml_ph2
```

# Metrics

In the file `metrics.py` you can find the functions we used to compute the metrics. <br/>
Localization metrics have to be computed only on fake images, and the ground truth **has to be 0 for pristine pixels and 1 for forged pixels**. <br/>
When computing F1 score, we take the maximum between the F1 using the localization map and the F1 using the inverse of the localization map.
We do not consider pixels close to the borders of the forged area in the ground truth, since in most cases they are not accurate. 


# Visualization

To visualize the output for an image, run the following:
```
python visualize.py --image image_path --output output_path [--mask mask_path]
```
Providing the mask is optional.
