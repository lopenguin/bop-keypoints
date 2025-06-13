# Keypoint Detection for Conformal Uncertainty Estimation
This repo is a fork of the [6D pose code base](https://github.com/yufu-wang/6D_Pose) (original documentation below). We provide brief instructions for using the keypoint estimator to generate and save detections.

## Setup and Requirements

First, clone this repo:
```shell
git clone https://github.com/lopenguin/bop-keypoints.git
```

We recommend working in a python virtual environment or Conda environment. This code was tested with **Python 3.12.3** and Cuda 12.6.

<details open>
<summary><b>Environment Setup</b></summary>

Install the requirements manually via pip:
```shell
pip install tqdm
pip install opencv-python
pip install imageio
pip install pypng
pip install pytz
pip install scipy
pip install scikit-image

# PYTORCH (https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

```
</details>

<details closed>
<summary><b>Setup via requirements.txt</b></summary>

Alternatively, install via `requirements.txt`:
```shell
# fixes version numbers. You may need to install pytorch manually for your GPU.
pip install -r requirements.txt
```
</details>

We provide sample LMO data in the `data` folder. Lastly, follow the [author instructions](#data) to download the detector weights.


Thank you to [Nathan Hughes](https://github.com/nathanhhughes) for showing me this format!



## Running Keypoint Detection
To run keypoint detection, use the following command (replace "lmo" with "ycbv" or "tudl" as needed). The `split` keyword refers to the name of the data folder.

```shell
python bbox_and_keypoints.py --dataset lmo --split test
```

This will save the JSON file `detections_{dataset_name}.json` in your working directory.

<details closed>
<summary><b>JSON Format</b></summary>

The JSON is a dictionary of dictionaries. The format is:
```python
{ image_number : {object_id : [u, v, conf]} }
```
where `[u,v]` are the pixel coordinates and `conf` is the network confidence (heatmap value).

</details>



# Author-Provided Documentation
Python implementation for the BOP benchmark section of the paper: \
**Semantic keypoint-based pose estimation from single RGB frames**  
Field Robotics \
[[Paper](https://arxiv.org/abs/2204.05864)]


## Data
You can download the pretrained models for [detection](https://drive.google.com/drive/folders/1Jzg-9sU4nEGawTREsMFblmBEZouPMOjM?usp=sharing) and [keypoint detection](https://drive.google.com/drive/folders/1i9Y5lFm3jc2t8qtxoB-qQJEDLc0urZao?usp=sharing). Please place the models as follows. We also put the test images for the LMO dataset in this repo for convenience.
```
- data
-- detect_checkpoints
-- kpts_checkpoints
```

## Demo
Our method uses additional 3D keypoint annotation on the CAD models, which is included in **kpts_3d.json**. We provide two demo. To explore the 3D annotation, please use **demo_data.ipynb**. To explore the inference pipeline, please use **demo_pipeline.ipynb**. 


## Reference
	@article{schmeckpeper2022semantic,
	  Title          = {Semantic keypoint-based pose estimation from single RGB frames},
	  Author         = {Schmeckpeper, Karl and Osteen, Philip R and Wang, Yufu and Pavlakos, Georgios and Chaney, Kenneth and Jordan, Wyatt and Zhou, Xiaowei and Derpanis, Konstantinos G and Daniilidis, Kostas},
	  Booktitle      = {Field Robotics},
	  Year           = {2022}
	}
