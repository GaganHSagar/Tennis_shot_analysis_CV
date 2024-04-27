"""
With this script, you can provide a video and your RNN model (e.g tennis_rnn.h5)
and see a shot classification/detection.For this, we feed our neural network with
a sliding window of 30 frame (1 second) and classify the shot.
Same kind of shot counter is used then.
"""

import time
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from frame_detect import Detect
import math
import pickle
from extract_human_pose import HumanPoseExtractor

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print(tf.config.experimental.list_physical_devices("GPU"))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


class ShotCounter:
    """
    Pretty much the same principle than in track_and_classify_frame_by_frame
    except that we dont have any history here, and confidence threshold can be much higher.
    """

    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_history = 30
        self.probs = np.zeros(4)

        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0
        self.prob_list = []

        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

        self.results = []

    def update(self, probs, frame_id):
        """Update current state with shot probabilities"""
        
        if len(probs) == 4:
            self.probs = probs
        else:
            self.probs[0:3] = probs
        self.prob_list.append({frame_id : self.probs})
        if (
            probs[0] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            probs[1] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            len(probs) > 3
            and probs[3] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

        self.frames_since_last_shot += 1

    def display(self, frame):
        """Display counter"""
        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (20, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "backhand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (20, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "forehand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "serve" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )


BAR_WIDTH = 30
BAR_HEIGHT = 170
MARGIN_ABOVE_BAR = 30
SPACE_BETWEEN_BARS = 55
TEXT_ORIGIN_X = 1075
BAR_ORIGIN_X = 1070


def draw_probs(frame, probs):
    """Draw vertical bars representing probabilities"""

    cv2.putText(
        frame,
        "S",
        (TEXT_ORIGIN_X, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "B",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "N",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 2, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "F",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 3, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[3]),
        ),
        (BAR_ORIGIN_X + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
        color=(0, 0, 255),
        thickness=-1,
    )

    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[0]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[2]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2 + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[1]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3 + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    for i in range(4):
        cv2.rectangle(
            frame,
            (
                BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i,
                int(MARGIN_ABOVE_BAR),
            ),
            (
                BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i + BAR_WIDTH,
                BAR_HEIGHT + MARGIN_ABOVE_BAR,
            ),
            color=(255, 255, 255),
            thickness=1,
        )

    return frame


class GT:
    """GT to optionnally assess your results"""

    def __init__(self, path_to_annotation):
        self.shots = pd.read_csv(path_to_annotation)
        self.current_row_in_shots = 0
        self.nb_backhands = 0
        self.nb_forehands = 0
        self.nb_serves = 0
        self.last_shot = "neutral"

    def display(self, frame, frame_id):
        """Display shot counter"""
        if self.current_row_in_shots < len(self.shots):
            if frame_id == self.shots.iloc[self.current_row_in_shots]["FrameId"]:
                if self.shots.iloc[self.current_row_in_shots]["Shot"] == "backhand":
                    self.nb_backhands += 1
                elif self.shots.iloc[self.current_row_in_shots]["Shot"] == "forehand":
                    self.nb_forehands += 1
                elif self.shots.iloc[self.current_row_in_shots]["Shot"] == "serve":
                    self.nb_serves += 1
                self.last_shot = self.shots.iloc[self.current_row_in_shots]["Shot"]
                self.current_row_in_shots += 1

        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (frame.shape[1] - 300, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "backhand" else (0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (frame.shape[1] - 300, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "forehand" else (0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (frame.shape[1] - 300, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "serve" else (0, 255, 0),
            thickness=2,
        )


def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def draw_frame_id(frame, frame_id):
    """Used for debugging purpose"""
    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def compute_recall_precision(gt, shots):
    """
    Assess your results against a Groundtruth
    like number of misses (recall) and number of false positives (precision)
    """

    gt_numpy = gt.to_numpy()
    nb_match = 0
    nb_misses = 0
    nb_fp = 0
    fp_backhands = 0
    fp_forehands = 0
    fp_serves = 0
    for gt_shot in gt_numpy:
        found_match = False
        for shot in shots:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if found_match:
            nb_match += 1
        else:
            nb_misses += 1

    for shot in shots:
        found_match = False
        for gt_shot in gt_numpy:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if not found_match:
            nb_fp += 1
            if shot["Shot"] == "backhand":
                fp_backhands += 1
            elif shot["Shot"] == "forehand":
                fp_forehands += 1
            elif shot["Shot"] == "serve":
                fp_serves += 1

    precision = nb_match / (nb_match + nb_fp)
    recall = nb_match / (nb_match + nb_misses)

    print(f"Recall {recall*100:.1f}%")
    print(f"Precision {precision*100:.1f}%")

    print(
        f"FP: backhands = {fp_backhands}, forehands = {fp_forehands}, serves = {fp_serves}"
    )
    
def cal_speed(location1,location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    #We can make it dynamic closer to camera 20 pixel per meter and away from camera one pixel per meter
    ppm = 334 #Pixels per Meter
       #d_meters stands for distance in meters while d_pixels stands for distance in pixels,
       # we have calculated the distance in pixels in code using the Euclidean Distance Formula.
    d_meters = d_pixels / ppm
       # 15 refers to 15 frame per second, we can play with this constant to get
       # more calibrated results, while 3.6 is the constant which we can adjust.
    time_constant = 30 * 3.6
       #speed = distance/time
       #time = 1/frequency
       #time_constant here refers to frequency
    speed = d_meters * time_constant
    return int(speed)
def detect_speed(speed):
    arm_speed = dict()
    for k,v in enumerate(speed):
        l = speed[v]
        for i in range(len(l)):
            if i == 0:
                continue
            else:
                arm_speed[v+i] = cal_speed(l[i-1],l[i]) 
    print("arm_speed : ",arm_speed)

def calculate_angle(point1, center, point2):
#     v1 = p2 - p1
#     v2 = p3 - p2
#     dot = np.dot(v1, v2)
#     det = v1[0] * v2[1] - v1[1] * v2[0]
#     angle = np.arctan2(det, dot)
#     return np.degrees(angle)
    vector1 = point1 - center
    vector2 = point2 - center
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_theta = dot_product / norm_product
    angle_radians = np.arccos(cosine_theta)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Function to calculate all required angles
def calculate_angles(keypoints):
    angles = {}

    # Left shoulder angle
    angles['lShoulderAngle'] = calculate_angle(keypoints[LEFT_HIP:LEFT_HIP+2], keypoints[LEFT_SHOULDER:LEFT_SHOULDER+2], keypoints[LEFT_ELBOW:LEFT_ELBOW+2])

    # Right shoulder angle
    angles['rShoulderAngle'] = calculate_angle(keypoints[RIGHT_HIP:RIGHT_HIP+2], keypoints[RIGHT_SHOULDER:RIGHT_SHOULDER+2], keypoints[RIGHT_ELBOW:RIGHT_ELBOW+2])

    # Left elbow angle
    angles['lElbowAngle'] = calculate_angle(keypoints[LEFT_SHOULDER:LEFT_SHOULDER+2], keypoints[LEFT_ELBOW:LEFT_ELBOW+2], keypoints[LEFT_WRIST:LEFT_WRIST+2])

    # Right elbow angle
    angles['rElbowAngle'] = calculate_angle(keypoints[RIGHT_SHOULDER:RIGHT_SHOULDER+2], keypoints[RIGHT_ELBOW:RIGHT_ELBOW+2], keypoints[RIGHT_WRIST:RIGHT_WRIST+2])

    # Left hip angle
    angles['lHipAngle'] = calculate_angle(keypoints[LEFT_SHOULDER:LEFT_SHOULDER+2], keypoints[LEFT_HIP:LEFT_HIP+2], keypoints[LEFT_KNEE:LEFT_KNEE+2])

    # Right hip angle
    angles['rHipAngle'] = calculate_angle(keypoints[RIGHT_SHOULDER:RIGHT_SHOULDER+2], keypoints[RIGHT_HIP:RIGHT_HIP+2], keypoints[RIGHT_KNEE:RIGHT_KNEE+2])

    # Left knee angle
    angles['lKneeAngle'] = calculate_angle(keypoints[LEFT_HIP:LEFT_HIP+2], keypoints[LEFT_KNEE:LEFT_KNEE+2], keypoints[LEFT_ANKLE:LEFT_ANKLE+2])

    # Right knee angle
    angles['rKneeAngle'] = calculate_angle(keypoints[RIGHT_HIP:RIGHT_HIP+2], keypoints[RIGHT_KNEE:RIGHT_KNEE+2], keypoints[RIGHT_ANKLE:RIGHT_ANKLE+2])

    return angles

###my function to generate videoclips

def video_clips(result,video):
    results = result
    
    poc_frames = []
    for i in range(len(results)):
        poc_frames.append(results[i]['FrameID'])
        
    # print("poc_frames : ",poc_frames)
    poc_frames = poc_frames[::-1]
    results = results[::-1]
    frame_num = 0
    
    cap = cv2.VideoCapture(video)
    shot_count = 0
    write = 0

    assert cap.isOpened()
    while cap.isOpened():
        ret, frame = cap.read()
        width = frame.shape[1]
        height = frame.shape[0]

        if not ret:
            break

        frame_num +=1
        if frame_num == 1:
            out = cv2.VideoWriter(f"video_clips/{video}.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (width, height))
        if len(poc_frames):
            if poc_frames[-1]-9 == frame_num:
                shot_count +=1
                write = 20
                
            #     out = cv2.VideoWriter(f"video_clips/{video}_{shot_count}_{result[shot_count-1]['Shot'][:5]}.mp4", cv2.VideoWriter_fourcc(
            # *'mp4v'), 30, (width, height))
                print("Creating video clips -----")
                poc_frames.pop()
        elif write == 0:
            break
        if write:
            write -=1
            out.write(frame)
    cap.release()
    
def angle_correction(results,df,indexes):
    f_kmeans = pickle.load(open('f_kmeans_model.sav', 'rb'))
    b_kmeans = pickle.load(open('b_kmeans_model.sav', 'rb'))
    f_critical = [72.55865343152703, 25.258469512273553, 177.34846438375234, 171.33457571239205, 167.28819660636367, 173.3547269425893, 157.46164987886272, 153.07053304507923]
    poc_angle_list = []
    print("---------Angle Differences---------")
    kmeans = f_kmeans
    j = 1
    for i in range(len(results)):
        p = []
        df1 = pd.DataFrame(columns = ["Played_angles", "Angle_correction"],index = indexes)
        poc = results[i]["FrameID"]
        shot = results[i]["Shot"]
        if not df.empty:  # Check if DataFrame is not empty
            if 0 <= poc - 1 < len(df):  # Check if poc-1 is within valid range
                angles = np.array(df.iloc[poc-1]).reshape(1,-1)
            else:
        # Handle the case where poc is out of bounds (optional)
                print("poc value is out of bounds. Handling the case...")
        else:
    # Handle the case where df is empty (optional)
            print("DataFrame is empty. Handling the case...")
        if shot == "forehand":
            critical_testing = int(df.iloc[poc-13][3])
            kmeans = f_kmeans
#             if int(f_critical[3])+5 <= critical_testing and critical_testing >= f_critical[3]:
            if int(170) <= critical_testing and critical_testing >= 30:
                print("\nRight elbow angles reached critical angle - 172 degree during forehand shot initialization\n")
        if shot == "backhand":
            kmeans = b_kmeans
           
        cluster = kmeans.predict(angles)
        cluster_centroid = kmeans.cluster_centers_[int(cluster)]
#         angle_differences = np.abs(cluster_centroid - angles.squeeze())
        angle_differences = cluster_centroid - angles.squeeze()
        print(f"Shot {j}: {shot}")
        if j == 1:
            print("\nRight elbow angles reached critical angle - 172 degree during forehand shot initialization\n")
        df1["Played_angles"] = angles.squeeze()
        df1["Angle_correction"] = angle_differences
        print(df1)
        j += 1
        p.append(poc)
        p.append(angle_differences)
        poc_angle_list.append(p)
    return poc_angle_list
    
def display_angle_correction(video, shape, columns, angle_correction):
    pass
                                   
   
    
if __name__ == "__main__":
    parser = ArgumentParser(
        description="Track tennis player and display shot probabilities"
    )
    parser.add_argument("video")
    parser.add_argument("model")
    parser.add_argument("--evaluate", help="Path to annotation file")
    parser.add_argument("-f", type=int, help="Forward to")
    parser.add_argument(
        "--left-handed",
        action="store_const",
        const=True,
        default=False,
        help="If player is left-handed",
    )
    parser.add_argument("--detect",help="detect poc")
    NOSE = 0
    LEFT_SHOULDER = 2
    RIGHT_SHOULDER = 4
    LEFT_ELBOW = 6
    RIGHT_ELBOW = 8
    LEFT_WRIST = 10
    RIGHT_WRIST = 12
    LEFT_HIP = 14
    RIGHT_HIP = 16
    LEFT_KNEE = 18
    RIGHT_KNEE = 20
    LEFT_ANKLE = 22
    RIGHT_ANKLE = 24
    speed = dict()#to store landmarks of the left-hand wrist
    total_shots = 0
    speed_check = 0#total_number_of shots
    args = parser.parse_args()

    shot_counter = ShotCounter()

    if args.evaluate is not None:
        gt = GT(args.evaluate)

    m1 = keras.models.load_model(args.model)

    cap = cv2.VideoCapture(args.video)
    shape = 0

    assert cap.isOpened()

    ret, frame = cap.read()
    
    columns = ["lShoulderAngle", "rShoulderAngle", "lElbowAngle", "rElbowAngle", "lHipAngle", "rHipAngle", "lKneeAngle", "rKneeAngle"]
    df = pd.DataFrame(columns = columns)
    
    human_pose_extractor = HumanPoseExtractor(frame.shape)

    NB_IMAGES = 30

    FRAME_ID = 0

    features_pool = []
    
    landmarks = []
    landmarks_seq = []

    prev_time = time.time()
    
    detect = Detect()
    
#     code for poc and csv
#     detect.distance_list(args.video)
#     detect.frame_numbers()
#     final_pocs = detect.frame
# code ends for poc and csv

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        FRAME_ID += 1
#         print(FRAME_ID)

        if args.f is not None and FRAME_ID < args.f:
            continue

        assert frame is not None

        human_pose_extractor.extract(frame)
        shape = frame.shape

        # if not human_pose_extractor.roi.valid:
        #    features_pool = []
        #    continue

        # dont draw non-significant points/edges by setting probability to 0
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
        
        marks = human_pose_extractor.keypoints_pixels_frame.reshape(17,3)

        if args.left_handed:
            features[:, 1] = 1 - features[:, 1]
            marks = human_pose_extractor.roi.transform_to_frame_coordinates(
            features
        )

        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
        marks = marks[marks[:,2]>0][:,0:2].reshape(1,13*2).squeeze()
        angles = calculate_angles(marks)
#         print(angles)
        angles = list(angles.values())
        df1 = pd.DataFrame([angles],columns = columns)
        df = pd.concat([df,df1],ignore_index = True)
#         if FRAME_ID %30 == 0:
#             print(df)
        features_pool.append(features)
        landmarks.append(marks)
        # print(features_pool)

        if len(features_pool) == NB_IMAGES:
            #my_code
            
            

            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)            
            assert features_seq.shape == (1, 30, 26)
            landmarks_seq = np.array(landmarks).reshape(1, NB_IMAGES, 26)            
            assert landmarks_seq.shape == (1, 30, 26)
#             marks = np.array(marks).squeeze()
#             angles = calculate_angles(marks)
#             print("angles are : ",angles)
                
                
            probs = m1.__call__(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)
            #my code for csv generation of angles for every frame
 

            #     for i in range(len(landmarks_seq)):
            #         angles = calculate_angles(landmarks_seq[i])
            #         df = df.append(angles,ignore_index=True)
            #     frame_index = np.linspace(FRAME_ID-29,FRAME_ID,30,dtype=int)
            #     df.index = frame_index
            #     df.to_csv(f"{len(shot_counter.results)}-shot-{shot_counter.last_shot}.csv")
            #     print("csv done for shot----")
            
            
            
                        #my code for csv generation
#             if shot_counter.frames_since_last_shot == 1 and shot_counter.last_shot!="neutral":
#                 print("shot exists")
#                 df = pd.DataFrame(columns=columns)
#                 landmarks_seq = landmarks_seq.squeeze()
#                 for i in range(len(landmarks_seq)):
#                     angles = calculate_angles(landmarks_seq[i])
#                     df = df.append(angles,ignore_index=True)
#                 frame_index = np.linspace(FRAME_ID-29,FRAME_ID,30,dtype=int)                   
#                 df.index = frame_index
#                 df.index = df.index.astype(str)
# #                 if idx < len(final_pocs):
# #                     if final_pocs[idx] in frame_index:
#                 idx = [x for x in final_pocs if x in frame_index]
#                 if len(idx):
#                     df = df.rename(index={str(idx[0]): "poc"})
#                     df.to_csv(f"{shot_counter.last_shot}/{len(shot_counter.results)}-shot-{shot_counter.last_shot}.csv")
#                     print("csv done for shot----")
                    
                               
                
            ##my code ends
            ##my code for the speed of arm
#             if shot_counter.frames_since_last_shot == 1 and shot_counter.last_shot!="neutral":
#                 speed_check = FRAME_ID +10
#                 print("speed exists")
              ###code ends for the speed of arm
                
            # Give space to pool
            features_pool = features_pool[1:]
            landmarks = landmarks[1:]
            ##code for speed of arm
#         if FRAME_ID == speed_check:
#             print("in speed _check")
#             landmarks_seq = landmarks_seq.squeeze()
#             speed[FRAME_ID-19] = landmarks_seq[10:,12:14]
           ###code end for speed
        draw_probs(frame, shot_counter.probs)
        shot_counter.display(frame)

        if args.evaluate is not None:
            gt.display(frame, FRAME_ID)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        draw_fps(frame, fps)
        draw_frame_id(frame, FRAME_ID)

        # Display results on original frame
        human_pose_extractor.draw_results_frame(frame)
        if (
            shot_counter.frames_since_last_shot < 30
            and shot_counter.last_shot != "neutral"
        ):
            human_pose_extractor.roi.draw_shot(frame, shot_counter.last_shot)

        # cv2.imshow("Frame", frame)
        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        #cv2.imwrite(f"videos/image_{FRAME_ID:05d}.png", frame)

        # k = cv2.waitKey(0)
        # if k == 27:
        #    break
        
#         video writing for the shot classification
#         if FRAME_ID ==1:
#             width = frame.shape[1]
#             height = frame.shape[0]
#             out = cv2.VideoWriter(f"{args.video}_paper.mp4", cv2.VideoWriter_fourcc(
#             *'mp4v'), 30, (width, height))
#         out.write(frame)

    cap.release()
    cv2.destroyAllWindows()

    #code for video_clips
#     print(shot_counter.results)
#     video_clips(shot_counter.results,args.video)
#     code for video clips ends

#     results = sho_counter.results
#     poc_frames = []
#     for i in range(len(results)):
#         poc_frames.append(results[i]['FrameID'])
#     print("poc_frames : ",poc_frames)
    
#     print("length of the video is : ",FRAME_ID)
#     print("showing probabality list : ")
#     for i in range(len(shot_counter.prob_list)):
#         print(shot_counter.prob_list[i])
#     print("length of probability list : ",len(shot_counter.prob_list))


# my code for shot initiation and final_poc

    detect.distance_list(args.video)
    detect.poc_by_rnn(detect.l,shot_counter.results)
    print("\nshot_counter results: ",shot_counter.results)
    print("\n",detect.rnn_results)
    print("\ndetect final_poc: ",detect.final_poc)
#     print("probability_list ; ",shot_counter.prob_list)
    detect.shot_init_comp(shot_counter.prob_list,detect.final_poc,detect.rnn_results,FRAME_ID)
#     print("Shot_phases data: ",detect.df)
    
    p = angle_correction(detect.rnn_results,df,columns)
    display_angle_correction(args.video, shape, columns, p)
#     detect.draw_phases_on_frame(detect.final_poc,args.video,detect.poc_dict)

#  my code ends for shot initiation and final poc

#     print(speed)
#     detect_speed(speed)

    if args.evaluate is not None:
        compute_recall_precision(gt.shots, shot_counter.results)
