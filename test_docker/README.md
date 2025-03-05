# Setup

Run the following command:
```
bash docker_build.sh
```

This command will also automatically download the [weights](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip) and unzip them in the "test_docker/weights" folder. 
MD5 is 7bee48f3476c75616c3c5721ab256ff8.

# Usage

To run all images in "images/" directory, run:
```
bash docker_run.sh
```

You can change the following parameters in docker_run.sh:
- `-gpu`: default is gpu '0'. Put '-1' if you want to use cpu.
- `-in`:  default is "images/". It can be a single file (data/tampered1.png), a directory (data/), or a glob statement (data/*.png)
- `-out`: output folder
- `--save_np`: if you want to save the Noiseprint++ aswell

The output is a .npz containing the following files:
- **'map'**: anomaly localization map
- **'conf'**: confidence map
- **'score'**: score in the range [0,1]
- **'np++'**: Noiseprint++ (if flag `--save_np` is specified)
- **'imgsize'**: size of the image

Note: if the output file already exists, it is not overwritten and it is skipped

Note: that the score values can slightly change when a different version of python, pytorch, cuda, cudnn, or other libraries changes.


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

Note that ```matplotlib``` and ```pillow``` packages are required to run this script.

