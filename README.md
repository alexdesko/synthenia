**For display purposes only**
# Synthetic Neural Implicit Attenuation
## Summary

This repository contains the implementation of a inverse renderer to recover structures from Transmission Electron Microscopes (TEMs) tomograms.
Assumptions include attenuation only rendering model and orthographic camera models. Given these, the goal is to input a collection of projections associated with their angle relative to the capturing axis (arbitrarily set), and to recover a 3D representation of the studied structures.
This project's goal was to essentially apply the Neral Radiance Field's technology to electron micropscopy.

It contains also a synthetic data generator, which produces 3D volumes of ellipsoids and associated projections, mimicking a microscope in an elementary fashion.

## File structure
```
├── configs
    ├── config_job.yaml
    ├── config_debug.yaml
├── dataset
    ├── dataset.py
├── models
    ├── fields.py
    ├── tensorf.py
├── regularization
    ├── regularization.py
├── rendering
    ├── encoder.py
    ├── pointsampler.py
    ├── raymarcher.py
    ├── renderer.py
├── experiment.py
├── README.md
├── run.py
├── synth_data_generator.py
```

## Implementation details
`data/dataset.py` contains a class to wrap the dataset and provides useful methods

`models/fields.py` contains neural implicit representations to encode the structure of the sample of interest. There are 3 different sizes possible. \
Inpired by [NeRF](https://www.matthewtancik.com/nerf). \
`models/tensorf.py` contains 3D volumes representations that are tensor-decomposed using VM and CP decomposition methods. \
Taken from [TensoRF](https://apchenstu.github.io/TensoRF/)

`regularization/regularization.py` contains multiple regularization methods for the densities (that are encoded with the fields) and the features (that are encoded with the 3D volumes). These include Total Variation and Non Local Means in different fashions.

`rendering/encoder.py` contains an simplified implementation of the positional encoder required for NeRFs. \
`rendering/pointsampler.py` contains point samplers to perform rendering. It includes a simple version, a coarse-to-fine version and one for the 3D non-local-means. \
`rendering/raymarcher.py` contains an implementation of a raymarcher. Taken from Pytorch3D's implementation. \
`rendering/renderer.py` contains a wrapper class that takes as input instances of all of what is necessary to have a full rendering engine at ready. Additionaly it performs integration.

`experiment.py` contains a wrapper to prepare, run and log the experiment.

`run.py` contains parser and sets up the experiment.

`synth_data_generator.py` generates a new synthetic dataset and stors it in `data/synthetic`


## How to use
- Install requirements
```
pip install -r requirements.txt
```
or using `pip3`

- Create an account for [WandB](https://wandb.ai) OR disable wandb by uncommenting the line
```
os.environ["WANDB_MODE"] = "disable"
```
in `run.py`

- Run the following command
```
python run.py configs/confid_debug.yaml
```
All the logs will be stored in `logs/` afterwards

## Author

This project was done by [@alexdesko](https://github.com/alexdesko) whil being PhD Student at [CVLab](https://www.epfl.ch/labs/cvlab/) at EPFL in 2022-2023.