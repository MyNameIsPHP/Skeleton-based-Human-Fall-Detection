# label - describes human posture in the depth frame; 
# '-1' means person is not lying, 
#'1' means person is lying on the ground; 
#'0' is temporary pose, when person "is falling", we don't use '0' frames in classification,


import pandas as pd
import cv2
import os
import numpy as np
import torch
from fn import vis_frame_fast
import time
from fn import draw_single

from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker

image_directory = "Data/UR_FallDetection/cam0"
csv_file = "Data/UR_FallDetection/urfall-cam0-falls.csv"
save_path = 'Annotations/URFD_annotations/pose_urfd_3classes.csv'
input_size = 384
inp_h = 320
inp_w = 256
num_class = 3

detect_model = TinyYOLOv3_onecls(device='cuda')
pose_estimator = SPPE_FastPose('resnet50', inp_h, inp_w)

# with score.
columns = ['video', 'frame', 
           'Nose_x', 'Nose_y', 'Nose_s', 
           'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 
           'LElbow_x', 'LElbow_y', 'LElbow_s', 
           'RElbow_x','RElbow_y', 'RElbow_s', 
           'LWrist_x', 'LWrist_y', 'LWrist_s', 
           'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 
           'RHip_x', 'RHip_y', 'RHip_s', 
           'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 
           'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 
           'RAnkle_x', 'RAnkle_y', 'RAnkle_s', 
           'label']


# Read CSV file into a pandas DataFrame
df = pd.read_csv(csv_file, header=None)
result_df = pd.DataFrame(columns=columns)
cur_row = 0

def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
# Iterate over each row in the DataFrame
prev_directory = ""
fps_time = 0

for index, row in df.iterrows():
    # Extract directory, filename, and label from the row
    directory = row[0]
    filename = row[1]
    label = row[2]

    
    # Construct the full path to the image
    image_path = f"{image_directory}/{directory}-cam0-rgb/{directory}-cam0-rgb-{int(filename):03d}.png"
    print(image_path)
    if prev_directory != directory:
        tracker = Tracker(max_age=30)
    prev_directory = directory

    # Read the image using OpenCV   
    image = cv2.imread(image_path)
    frame = image.copy()
    # width = frame.shape[1]
    # # Get the width and height of the frame
    # height, width = frame.shape[:2]

    # # Calculate the width of the left 30% of the frame
    # left_width = int(width * 0.3)

    # # Calculate the height of the top 80% of the frame
    # top_height = int(height * 0.8)

    # # Set the top-left portion of the frame to black
    # frame[:top_height, :left_width, :] = 0
    # frame[:, :int(width*0.2), :] = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_size = (frame.shape[1], frame.shape[0])

    detected = detect_model.detect(frame, expand_bb=10)
    # keep only the highest score
    if (detected != None):
        if (len(detected) > 1):
            max_score = detected[:, 4].max()
            detected = detected[detected[:, 4] == max_score]
        # print(frame_count,len(detected))
    # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
    tracker.predict()
    # Merge two source of predicted bbox together.
    for track in tracker.tracks:
        det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
        detected = torch.cat([detected, det], dim=0) if detected is not None else det

    detections = []  # List of Detections object for tracking.
    if detected is not None:
        #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
        # Predict skeleton pose of each bboxs.
        poses = pose_estimator.predict(frame, detected[:, 0:4], detected[:, 4])

        # Create Detections object.
        detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                np.concatenate((ps['keypoints'].numpy(),
                                                ps['kp_score'].numpy()), axis=1),
                                ps['kp_score'].mean().numpy()) for ps in poses]

        # VISUALIZE.

        # for bb in detected[:, 0:5]:
        #     frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

    # Update tracks by matching each track information of current and previous frame or
    # create a new track if no matched.
    tracker.update(detections)
        
    for i, track in enumerate(tracker.tracks):
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr().astype(int)
        
        # skip bbox if it width or height is less than 100
        if bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 50:
            continue
    

        center = track.get_center().astype(int)


        # check if the current frame is within the falling period
        label += 1
        if label == 1:
            action = 'Falling'
            clr = (255, 200, 0)
        elif label == 2:
            action = 'Lying'
            clr = (255, 0, 0)
        else:
            label = 0   
            action = 'Not Fall'
            clr = (0, 255, 0)
        
        # VISUALIZE.
        if track.time_since_update == 0:
            
            frame = draw_single(frame, track.keypoints_list[-1])
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr, 1)
            frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                0.4, (255, 0, 0), 2)
            frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.4, clr, 1)
            if len(poses) > 0:
                pt_norm = normalize_points_with_size(poses[0]['keypoints'].numpy().copy(),
                                                        frame_size[0], frame_size[1])
                pt_norm = np.concatenate((pt_norm, poses[0]['kp_score']), axis=1)

                #idx = poses[0]['kp_score'] <= 0.05
                #pt_norm[idx.squeeze()] = np.nan
                new_row = [directory, int(filename), *pt_norm.flatten().tolist(), label]

            else:
                new_row = [directory, int(filename), *[np.nan] * (13 * 3), label]

            result_df.loc[cur_row] = new_row
            cur_row += 1    
                        

    # Show Frame.
    # frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
    frame = cv2.putText(frame, '%d, FPS: %f' % (int(filename), 1.0 / (time.time() - fps_time)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    frame = frame[:, :, ::-1]
    fps_time = time.time()

    cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

result_df.to_csv(save_path, mode='w', index=False)

# Count the number of each class
class_counts = result_df['label'].value_counts()
print(class_counts)
cv2.destroyAllWindows()

