"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.

"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

csv_pose_file = 'Annotations/all_data.csv'
save_dir = 'DataFiles/All_2classes'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class_names = ['Not fall', 'Fall']
# class_names = ['Not fall', 'Falling', 'Fall']

main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LHip_x', 'LHip_y',
              'RHip_x', 'RHip_y']
main_idx_parts = [1, 2, 7, 8, -1]  # 1.5



# Params.
smooth_labels_step = 8
n_frames = 30
skip_frame = 1

annot = pd.read_csv(csv_pose_file)
# Columns: video,frame,Nose_x,Nose_y,Nose_s,LShoulder_x,LShoulder_y,LShoulder_s,RShoulder_x,RShoulder_y,RShoulder_s,LElbow_x,LElbow_y,LElbow_s,RElbow_x,RElbow_y,RElbow_s,LWrist_x,LWrist_y,LWrist_s,RWrist_x,RWrist_y,RWrist_s,LHip_x,LHip_y,LHip_s,RHip_x,RHip_y,RHip_s,LKnee_x,LKnee_y,LKnee_s,RKnee_x,RKnee_y,RKnee_s,LAnkle_x,LAnkle_y,LAnkle_s,RAnkle_x,RAnkle_y,RAnkle_s,label
# label columns: 0 -> Not Fall, 1 -> Falling, 2 -> Fall

# Remove NaN.
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0]
annot = annot.drop(idx)

if len(class_names) == 2:
    # remove 'Falling' class.
    annot = annot[annot['label'] != 1]

# One-Hot Labels.
label_onehot = pd.get_dummies(annot['label'])
annot = annot.drop('label', axis=1).join(label_onehot)
cols = label_onehot.columns.values


def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels


feature_set = np.empty((0, n_frames, 14, 3))
labels_set = np.empty((0, len(cols)))
vid_list = annot['video'].unique()
for vid in vid_list:
    print(f'Process on: {vid}')
    data = annot[annot['video'] == vid].reset_index(drop=True).drop(columns='video')

    # Label Smoothing.
    esp = 0.1
    data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)
    data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)

    # Separate continuous frames.
    frames = data['frame'].values
    frames_set = []
    fs = [0]
    for i in range(1, len(frames)):
        if frames[i] < frames[i-1] + 10:
            fs.append(i)
        else:
            frames_set.append(fs)
            fs = [i]
    frames_set.append(fs)

    for fs in frames_set:
        print(f'Frames: {fs[0]} - {fs[-1]}')
        xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 13, 3)
        # Scale pose normalize.
        xys[:, :, :2] = scale_pose(xys[:, :, :2])
        # Add center point.
        xys = np.concatenate((xys, np.expand_dims((xys[:, 1, :] + xys[:, 2, :]) / 2, 1)), axis=1)

        # Weighting main parts score.
        scr = xys[:, :, -1].copy()
        scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
        # Mean score.
        scr = scr.mean(1)

        # Targets.
        lb = data.iloc[fs, -len(cols):].values
        # Apply points score mean to all labels.
        lb = lb * scr[:, None]

        for i in range(xys.shape[0] - n_frames):
            feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
            labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0)


# Split into train, val, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(feature_set, labels_set, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

print("Train set shapes:", X_train.shape, y_train.shape)
print("Validation set shapes:", X_val.shape, y_val.shape)
print("Test set shapes:", X_test.shape, y_test.shape)

# Print the distribution of each set
train_dist = np.sum(y_train, axis=0)
val_dist = np.sum(y_val, axis=0)
test_dist = np.sum(y_test, axis=0)

print("Train set distribution:", train_dist)
print("Validation set distribution:", val_dist)
print("Test set distribution:", test_dist)

# Save the datasets

with open(os.path.join(save_dir, 'train.pkl'), 'wb') as f:
    pickle.dump((X_train, y_train), f)


with open(os.path.join(save_dir, 'val.pkl'), 'wb') as f:
    pickle.dump((X_val, y_val), f)


with open(os.path.join(save_dir, 'test.pkl'), 'wb') as f:
    pickle.dump((X_test, y_test), f)
