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


from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose

image_directory = "UR_FallDetection/cam0"
csv_file = "UR_FallDetection/urfall-cam0-falls.csv"
save_path = 'pose_urfd.csv'
input_size = 384
inp_h = 320
inp_w = 256


detect_model = TinyYOLOv3_onecls(device='cuda')
pose_estimator = SPPE_FastPose('resnet50', inp_h, inp_w)

# with score.
columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']


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

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Extract directory, filename, and label from the row
    directory = row[0]
    filename = row[1]
    label = row[2]
    
    # Skip rows with label 0
    if label == 0:
        continue
    if label == -1:
        label = 0 # not lying

    # Construct the full path to the image
    image_path = f"{image_directory}/{directory}-cam0-rgb/{directory}-cam0-rgb-{int(filename):03d}.png"
    print(image_path)
    
    # Read the image using OpenCV   
    image = cv2.imread(image_path)
    frame = image.copy()
    width = frame.shape[1]
    # Get the width and height of the frame
    height, width = frame.shape[:2]

    # Calculate the width of the left 30% of the frame
    left_width = int(width * 0.3)

    # Calculate the height of the top 80% of the frame
    top_height = int(height * 0.8)

    # Set the top-left portion of the frame to black
    frame[:top_height, :left_width, :] = 0
    frame[:, :int(width*0.2), :] = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_size = (frame.shape[1], frame.shape[0])

    detected = detect_model.detect(frame, expand_bb=10)
    if (detected != None):
        # print(detected)

        ########### Way 1
        # if detected is not None:
        #     #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
        #     # Predict skeleton pose of each bboxs.
        #     poses = pose_estimator.predict(frame, detected[:, 0:4], detected[:, 4])
        # print(poses.shape)

        ########## Way 2
        bb = detected[0, :4].numpy().astype(int)
        bb[:2] = np.maximum(0, bb[:2] - 5)
        bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]
        result = []
        if bb.any() != 0:
            result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                            torch.tensor([[1.0]]))
            # print(result.shape)
        if len(result) > 0:
            pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                    frame_size[0], frame_size[1])
            pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)

            #idx = result[0]['kp_score'] <= 0.05
            #pt_norm[idx.squeeze()] = np.nan
            row = [directory, int(filename), *pt_norm.flatten().tolist(), label]
            scr = result[0]['kp_score'].mean()
        else:
            row = [directory, int(filename), *[np.nan] * (13 * 3), label]
            scr = 0.0

        result_df.loc[cur_row] = row
        cur_row += 1

        # VISUALIZE.
        frame = vis_frame_fast(frame, result)
        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        label_text = "Lying" if label == 1 else "Not Lying"
        frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(filename, label_text, scr),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        ##########

    result_df.to_csv(save_path, mode='w', index=False)

    frame = frame[:, :, ::-1]
    cv2.imshow("Image with Label", frame)
    cv2.waitKey(1)  # Wait for any key press to continue to the next image

# Count the number of each class
class_counts = result_df['label'].value_counts()
print(class_counts)
cv2.destroyAllWindows()

