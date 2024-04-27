# Skeleton-based-Human-Fall-Detection
Download UR Fall Detection dataset 

Download pretrained models 

## Installation
- Python 3.9
- Pytorch: latest version
- Pip install latest version of other dependencies 

## Usage

Extract the pose information from URFD by running:
```
python3 process_urfd.py
```

Create the `pkl` data file by running:
```
python3 process_urfd_2.py
```

Train the action recognition model:
```
python3 train.py
```
