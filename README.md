# Skeleton-based-Human-Fall-Detection


<table align="center">
  <tr>
    <td>
      <img src="https://github.com/MyNameIsPHP/Skeleton-based-Human-Fall-Detection/blob/main/example_output_1.gif" width="80%" height="80%">
    </td>
    <td>
      <img src="https://github.com/MyNameIsPHP/Skeleton-based-Human-Fall-Detection/blob/main/example_output_2.gif" width="80%" height="80%">
    </td>
    <td>
      <img src="https://github.com/MyNameIsPHP/Skeleton-based-Human-Fall-Detection/blob/main/example_output_3.gif" width="80%" height="80%">
    </td>
  </tr>
</table>



## Installation
- Python 3.9
- Pytorch: latest version
- Pip install latest version of other dependencies 

## Usage

Extract the pose information from URFD or Le2i by running:
```
python3 process_urfd.py
python3 propess_le2i.py
```

Create the `pkl` data file by running:
```
python3 process_annotation.py
```

Train the action recognition model:
```
python3 train.py
```
