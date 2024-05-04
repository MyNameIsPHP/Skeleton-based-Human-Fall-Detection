# label - describes human posture in the depth frame; 
# '-1' -> 0 means person is not lying, 
#'1' -> 2 means person is lying on the ground; 
#'0' -> 1 is temporary pose, when person "is falling", we don't use '0' frames in classification,



# Error: coffee_room26, 52, 50, 
import pandas as pd
import cv2
import os
import numpy as np
import torch
from fn import vis_frame_fast


from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose

input_size = 384
inp_h = 320
inp_w = 256


detect_model = TinyYOLOv3_onecls(device='cuda')
pose_estimator = SPPE_FastPose('resnet50', inp_h, inp_w)
save_name = 'pose_le2i'

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

def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy

directory = "Data/Le2i"
# get all the sub-directories in the directory
sub_directories = [f.path for f in os.scandir(directory) if f.is_dir()]


# Read CSV file into a pandas DataFrame
result_df_3classes = pd.DataFrame(columns=columns)
result_df_2classes_1 = pd.DataFrame(columns=columns)
result_df_2classes_2 = pd.DataFrame(columns=columns)
result_df_2classes_3 = pd.DataFrame(columns=columns)

cur_row = 0

for sub_directory in sub_directories:
    # get all the files in the sub-directory, only get the name of the file, not extension
    files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(sub_directory, "Annotation_files"))]
    
    for target_filename in files:

        annotation_file = f"{sub_directory}/Annotation_files/{target_filename}.txt"
        video_file = f"{sub_directory}/Videos/{target_filename}.avi"
        
        # print annotation_file and video_file
        print(annotation_file, video_file)
        # Read the annotation file
        with open(annotation_file, 'r') as file:
            lines = file.readlines()
            start_frame = int(lines[0])
            end_frame = int(lines[1])
            # fall_frames = [int(line) for line in lines[2:]]

        # Read and visualize the video
        cap = cv2.VideoCapture(video_file)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("error")
                break
            
        
            if frame_count < start_frame:
                label_3classes = 0
            elif frame_count >= start_frame and frame_count <= end_frame:
                label_3classes = 1
            else:
                label_3classes = 2
            

            # Check if the current frame is within the fall event
            # if frame_count >= start_frame and frame_count <= end_frame:
            #     label = 1   
            # else:
            #     label = 0
            if frame_count >= start_frame and frame_count <= end_frame: # as notation files
                label_2classes_1 = 1   
            else:
                label_2classes_1 = 0

            if frame_count < start_frame: # urdf 2 class style
                label_2classes_2 = 0
            elif frame_count > end_frame:
                label_2classes_2 = 1

            if frame_count < start_frame: # test
                label_2classes_3 = 0
            else:
                label_2classes_3 = 1
        
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
                    row_3classes = [sub_directory, target_filename, *pt_norm.flatten().tolist(), label_3classes]
                    row_2classes_1 = [sub_directory, target_filename, *pt_norm.flatten().tolist(), label_2classes_1]
                    row_2classes_2 = [sub_directory, target_filename, *pt_norm.flatten().tolist(), label_2classes_2]
                    row_2classes_3 = [sub_directory, target_filename, *pt_norm.flatten().tolist(), label_2classes_3]

                    scr = result[0]['kp_score'].mean()
                else:
                    row_3classes = [sub_directory, target_filename, *[np.nan] * (13 * 3), label_3classes]
                    row_2classes_1 = [sub_directory, target_filename, *[np.nan] * (13 * 3), label_2classes_1]
                    row_2classes_2 = [sub_directory, target_filename, *[np.nan] * (13 * 3), label_2classes_2]
                    row_2classes_3 = [sub_directory, target_filename, *[np.nan] * (13 * 3), label_2classes_3]

                    scr = 0.0

                result_df_3classes.loc[cur_row] = row_3classes
                result_df_2classes_1.loc[cur_row] = row_2classes_1
                result_df_2classes_2.loc[cur_row] = row_2classes_2
                result_df_2classes_3.loc[cur_row] = row_2classes_3
                cur_row += 1

                # VISUALIZE.
                frame = vis_frame_fast(frame, result)
                frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                # if num_class == 2:
                #     label_text = "Fall" if label_3classes == 1 else "Not Fall"
                # elif num_class == 3:
                label_text = "Fall" if label_3classes == 2 else "Falling" if label_3classes == 1 else "Not Fall"
                frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(target_filename, label_text, scr),
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                ##########
            # print frame count and label
            print(f"Frame: {cur_row}, Label: {label_3classes}")

            result_df_3classes.to_csv(f'{save_name}_3classes.csv', mode='w', index=False)
            result_df_2classes_1.to_csv(f'{save_name}_2classes.csv', mode='w', index=False)
            result_df_2classes_2.to_csv(f'{save_name}_2classes_2.csv', mode='w', index=False)
            result_df_2classes_3.to_csv(f'{save_name}_2classes_3.csv', mode='w', index=False)

            frame = frame[:, :, ::-1]
            
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Increase the delay to slow down the video (50 milliseconds)
                break
            
            frame_count += 1

cap.release()
# Count the number of each class
class_counts = result_df_3classes['label'].value_counts()
print("3 class count: ", class_counts)

class_counts = result_df_2classes_1['label'].value_counts()
print("2 class count: ", class_counts)

class_counts = result_df_2classes_2['label'].value_counts()
print("2 class count: ", class_counts)

class_counts = result_df_2classes_3['label'].value_counts()
print("2 class count: ", class_counts)


cv2.destroyAllWindows()









    
    
    



