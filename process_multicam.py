import cv2
import pandas as pd
import os
import numpy as np
import torch
from Detection.Utils import bbox_iou
from fn import vis_frame_fast
from fn import draw_single
import time

from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
input_size = 384
inp_h = 320
inp_w = 256


detect_model = TinyYOLOv3_onecls(device='cuda', conf_thres=0.5)
pose_estimator = SPPE_FastPose('resnet50', inp_h, inp_w)
# Tracker.
max_age = 30

data_dir = "Data/MultiCam"
grouth_truth_file = f"{data_dir}/ground_truth.csv"
delay_in_frame_file = f"{data_dir}/delay_in_frame.csv"

# read ground truth file, it has the following columns: scenario_number,start,falling_period_start,falling_period_end,lying_period_start,lying_period_end
ground_truth = pd.read_csv(grouth_truth_file)

# read delay in frame file, it has the following columns: scenario_number,cam1,cam2,cam3,cam4,cam5,cam6,cam7,cam8
delay_in_frame = pd.read_csv(delay_in_frame_file)

save_name = 'Annotations/MultiCam_annotations/pose_multicam_3classes.csv'

def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy


#make dir if not exist
save_name_dir = os.path.dirname(save_name)
if not os.path.exists(save_name_dir):
    os.makedirs(save_name_dir)

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

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

result_df = pd.DataFrame(columns=columns)

cur_row = 0

# loop through the ground truth file
for index, row in ground_truth.iterrows():
    print(f"Scenario: {row['scenario_number']}")

    video_folder = f"{data_dir}/chute{row['scenario_number']:02d}"
    print(video_folder)

    # read the all video files in video_folder
    
    # print total frames of each video file

    # loop through the video files
    for cam_idx in range(1,9):
        video_file = f"cam{cam_idx}.avi"
        print(video_folder, video_file)
        delay = delay_in_frame.loc[delay_in_frame['scenario_number'] == row['scenario_number'], f'cam{cam_idx}'].values[0]
        start = row['start'] + delay
        falling_period_start = row['falling_period_start'] + delay
        falling_period_end = row['falling_period_end'] + delay
        lying_period_start = row['lying_period_start'] + delay
        lying_period_end = row['lying_period_end'] + delay

        tracker = Tracker(max_age=max_age, n_init=3)

        # read the video file
        cap = cv2.VideoCapture(f"{video_folder}/{video_file}")
        frame_count = 0
        fps_time = 0
        # read the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("error")
                break
            if frame_count < start:
                frame_count += 1
                continue
            if frame_count > lying_period_end:
                break
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
                
                # skip bbox that width is greater than 50% of the frame width or height is greater than 80% of the frame height
                if bbox[2] - bbox[0] > frame_size[0] * 0.5 or bbox[3] - bbox[1] > frame_size[1] * 0.8:
                    continue


                center = track.get_center().astype(int)


                # check if the current frame is within the falling period
                if frame_count >= falling_period_start and frame_count <= falling_period_end:
                    label = 1
                    action = 'Falling'
                    clr = (255, 200, 0)
                elif frame_count >= lying_period_start and frame_count <= lying_period_end:
                    label = 2
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
                        new_row = [f'{video_folder}/{video_file}', frame_count, *pt_norm.flatten().tolist(), label]

                    else:
                        new_row = [f'{video_folder}/{video_file}', frame_count, *[np.nan] * (13 * 3), label]

                    result_df.loc[cur_row] = new_row
                    cur_row += 1    
                                

            # Show Frame.
            # frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (frame_count, 1.0 / (time.time() - fps_time)),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = frame[:, :, ::-1]
            fps_time = time.time()

            frame_count += 1
            cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

result_df.to_csv(f'{save_name}', mode='w', index=False)
class_counts = result_df['label'].value_counts()
print("3 class count: ", class_counts)

cap.release()
cv2.destroyAllWindows()