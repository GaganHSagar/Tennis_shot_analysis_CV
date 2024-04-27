import os
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from ultralytics import YOLO
import math
import matplotlib.pyplot as plt
import cv2


class Detect():
    def __init__(self, weights="2500.pt"):
        self.model = YOLO(weights)
        # loading object detection model
        self.prob_list = []
        self.poc_dict = {}
        self.results = []

    def distance_list(self, video="VID20220619095153.mp4"):
        # function to calculate the distance between racket and ball
        
        cap = cv2.VideoCapture(video)
        frame_counter = 1
        ret = 1
        l = []
        # new=[]
        while (ret):

            # Capture frame-by-frame
            dist = 0
            overlap = 0
            ret, frame = cap.read()
            if ret == True:

                result = self.model.predict(frame)
                # predicting objects in the frame
                le = len(result[0].boxes)
                
                # proceeding only if we found both racket and ball
                if le >= 2:
                    b_l = []
                    r_l = []
                    # to store the coordinates of the ball and racket
                    
                    for i in range(le):
                        cls = int(result[0].boxes[i].cls.tolist()[0])
                        # obtaining the class of the object
                        if cls == 0:
                            b_l.append(i)
                        elif (cls == 1):
                            overlap = 1
                        else:
                            r_l.append(i)

                    #                 b_cls=int(result[0].boxes[0].cls.tolist()[0])
                    #                 r_cls=int(result[0].boxes[1].cls.tolist()[0])
                    # print("b_cls : ",b_cls)
                    # print("r_cls : ",r_cls)
                    # if b_cls==1 and r_cls==0:
                    
                    # calculating distance only if the racket and ball object is present
                    if len(b_l) and len(r_l):
                        dist = 9999
                        
                        # calculating distance between each ball and racket (if the model predicts multiple ball and racket)
                        for i in b_l:
                            for j in r_l:
                                ball = result[0].boxes[i].xyxy[0].tolist()
                                racket = result[0].boxes[j].xyxy[0].tolist()
                                bx = (ball[0] + ball[2]) / 2
                                by = (ball[1] + ball[3]) / 2
                                rx = (racket[0] + racket[2]) / 2
                                ry = (racket[1] + racket[3]) / 2

                                #print(len(result[0].boxes[1].xyxy))
                                #print("bx : ", bx, " by : ", by)
                                #print("rx : ", rx, " ry : ", ry)

                                # print("_Box :",ball)
                                # print("_box :",racket)
                                dist1 = math.sqrt((bx - rx) ** 2 + (by - ry) ** 2)

#                                 print("distance is :", dist1)
#                                 print("frame_counter is :", frame_counter)

                                if dist1 < dist:
                                    dist = dist1
            #                 print("result boxes : ",end="")
            #                 print(result[0].boxes[0].cls)
            #                 print(result[0].boxes[1].cls)
            #                 print(type(result[0].boxes[0].cls))
            #                     if(dist<200):
            #                         image = cv2.rectangle(frame, (int(ball[0]), int(ball[1])), (int(ball[2]), int(ball[3])), (255, 0, 0), 2)
            #                         image = cv2.rectangle(frame, (int(racket[0]), int(racket[1])), (int(racket[2]), int(racket[3])), (255, 0, 0), 2)
            #                         #new.append(image)#cv2.imshow("15",image)
            #                         #cv2.waitkey(0)
            #                         plt.imshow(image)

            frame_counter += 1
            # if overlap class is present we are simply appending -1 to the distance list to easily identify it as a poc
            if (overlap and le>2):
                l.append(-1)
            else:
                l.append(dist)
        cap.release()
        # # l=[i if i else None for i in l]
        # plt.plot(l)
        # # plt.scatter([i for i in range(len(l))],[i if i else None for i in l])
        # plt.xlabel("Frame number")
        # plt.ylabel("Distance")
        # plt.title("Distance between racket and ball")
        # plt.show()
        l=[i if i!=0 else None for i in l]
        # if the distance is 0, replacing it by None, so that we can easily ignore None distance in detecting poc
        self.l = l

    def frame_numbers(self):
        l = self.l
        l = [i if i != 0 else None for i in l]
        frame: list[int] = []
        for i in range(len(l) - 30):
         #to compare next 30 values if they are in ascending order or not
            if (l[i] is None):
                continue
            elif (l[i] == -1): #is the distance is -1, it is overlap, therefore POC
                if len(frame):
                    if frame[-1] > (i - 40):
                    #if any value in the frame which is very near to the poc framei.e nearby 40 frames
                        frame[-1] = i + 1
                    else:
                        frame.append(i + 1)
                else:
                    frame.append(i + 1) #overlap found in first shot
            elif (l[i] < 130):  #distance threshold
                miss = 1
                for j in range(i + 1, i + 31):  #to check next 30 frames if the distance is ascending or not. If it ascends then it is poc
                    if (l[j] is None):
                        continue
                    if l[j] < l[i]: #check if there is a next distance which is less than the current distance
                        miss = 0 #there is another distance which can be POC
                        break
                if (miss): #if miss ==1 then the next 30 frames distances are ascending. so it is the poc
                    if len(frame):
                        if (l[frame[-1]] == -1):
                            pass
                        elif frame[-1] > (i - 30):
                            frame[-1] = i + 1
                        else:
                            frame.append(i + 1)
                    else:
                        frame.append(i + 1)
        for i in range(len(l) - 30, len(l)): #check if there is a poc in the last leftout part
            if (l[i] is None):
                continue
            elif (l[i] == -1):
                if len(frame):
                    if frame[-1] > (i - 40):
                        frame[-1] = i + 1
                    else:
                        frame.append(i + 1)
                else:
                    frame.append(i + 1)
            elif (l[i] < 130):
                miss = 1
                for j in range(i + 1, len(l)):
                    if (l[j] is None):
                        continue
                    if l[j] < l[i]:
                        miss = 0
                        break
                if (miss):
                    if len(frame):
                        if (l[frame[-1]] == -1):
                            pass
                        elif frame[-1] > (i - 30):
                            frame[-1] = i + 1
                        else:
                            frame.append(i + 1)
                    else:
                        frame.append(i + 1)
        print("POC happens at : ", frame)
        self.frame = frame
        return frame
    def poc_by_rnn(self,distance,results):
        rnn_frames = []
        for i in range(len(results)):
            rnn_frames.append(results[i]['FrameID'])
        print("RNN POCS extracted from shot_counter results: ",rnn_frames)
        final_poc = []
        rnn_poc_dict = []
        shot_cont = 0
        for l in rnn_frames:
            for j in range(l-9,l+11):
                if distance[j] is None:
                    continue
                elif distance[j] == -1:
                    final_poc.append(j+1)
                    rnn_poc_dict.append(results[shot_cont])
                    break
                elif(distance[j] < 130):
                    dist = distance[j]
                    flag = 1
                    for k in range(j+1,j+11):
                        if distance[k] is None:
                            continue
                        elif distance[k] < dist:
                            flag = 0
                            break
                    if flag:
                        final_poc.append(j+1)
                        rnn_poc_dict.append(results[shot_cont])
                        break
            shot_cont += 1
        self.final_poc = final_poc
        self.rnn_results = rnn_poc_dict #rnn_results in the dictionary
        print("final_poc : ",final_poc)
        
        
    def draw_phases_on_frame(self,frame1,video,poc_dict):
        # code to draw POC phases on a video (shot initialization, poc, shot completion)
        
        video_name = video.split(".")[0]
        print("In the draw poc")
        end_list= []
        init_list = []
        poc_list = []
        for i in poc_dict:
            init_list.append(i)
            poc_list.append(poc_dict[i][0])
            end_list.append(poc_dict[i][1])
        end_list = end_list[::-1]
        init_list = init_list[::-1]
        poc_list = poc_list[::-1]
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("Error opening video")
            return
        fr_num = 0
        total_poc = 0
        i=0
        init = 0
        ending = 0
        poc_start = 0
        ##frame_indx -- i
        ret = True
        while(ret):
            fr_num = fr_num+1
            ret , img = cap.read()
            poc_frame = 0
            poc_end = 0
            miss =1
            if ret:
                cv2.putText(
                img,
                f"Frame Num : {fr_num}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 255),
                thickness=3,
                    )
                if any(fr_num==f for f in frame1):
                    total_poc +=1
            #         cv2.putText(
            #     img,
            #     f"POC happened",
            #     (img.shape[1] - 550, img.shape[0] - 100),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=2,
            #     color=(0, 0, 255),
            #     thickness=3,
            # )
                    cv2.putText(
                img,
                f"Total_POCS = {total_poc}",
                (img.shape[1] - 660, img.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=3,
            )
                    miss =0
                if miss:
                    cv2.putText(
                img,
                f"Total_POCS = {total_poc}",
                (img.shape[1] - 660, img.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                thickness=3,
            )
            #     if any(fr_num - j in frame1 for j in range(10)):
            #         cv2.putText(
            #     img,
            #     f"POC happened",
            #     (img.shape[1] - 550, img.shape[0] - 100),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=2,
            #     color=(0, 0, 255),
            #     thickness=3,
            # )
                if fr_num==1:
                    out = cv2.VideoWriter(f"{video_name}_init_end.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (img.shape[1], img.shape[0]))
                    print("Done. Video created")
                if fr_num in init_list:
                    init = end_list[-1]-init_list[-1]+10
                
                if init:
                    init -= 1
                    cv2.putText(
            img,
            f"Shot initiliazed : {init_list[-1]}",
            (img.shape[1] - 660, img.shape[0] - 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 0, 0),
            thickness=3,
                )
                    if init == 0:
                        init_list.pop()
                if fr_num in poc_list:
                    poc_start = end_list[-1]-poc_list[-1]+10
                if poc_start:
                    poc_start -= 1
                    cv2.putText(
                img,
                f"POC happened : {poc_list[-1]}",
                (img.shape[1] - 660, img.shape[0] - 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=3,
            )
                    if poc_start == 0:
                        poc_list.pop()
                    
                if fr_num in end_list:
                    ending = 10
                if ending:
                    ending -= 1
                    cv2.putText(
            img,
            f"Shot completed :{end_list[-1]}",
            (img.shape[1] - 660, img.shape[0] - 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 0, 0),
            thickness=3,
                )
                    if ending == 0:
                        end_list.pop()
                      
                    
                out.write(img)
        cap.release()
        
        
    def draw_poc_on_frame(self,frame1,video,poc_dict):
        # code to draw poc frame number on the video
        
        video_name = video.split(".")[0]
        print("In the draw poc")
#         end_list= []
#         init_list = []
        poc_list = []
        for i in poc_dict:
#             init_list.append(i)
            poc_list.append(poc_dict[i][0])
#             end_list.append(poc_dict[i][1])
#         end_list = end_list[::-1]
#         init_list = init_list[::-1]
        poc_list = poc_list[::-1]
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("Error opening video")
            return
        fr_num = 0
        total_poc = 0
        i=0
        init = 0
        ending = 0
        poc_start = 0
        ##frame_indx -- i
        ret = True
        while(ret):
            fr_num = fr_num+1
            ret , img = cap.read()
            poc_frame = 0
            poc_end = 0
            miss =1
            if ret:
                cv2.putText(
                img,
                f"Frame Num : {fr_num}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 255),
                thickness=3,
                    )
                if any(fr_num==f for f in frame1):
                    total_poc +=1
                    cv2.putText(
                img,
                f"POC happened",
                (img.shape[1] - 550, img.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=3,
            )
                    cv2.putText(
                img,
                f"Total_POCS = {total_poc}",
                (img.shape[1] - 660, img.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=3,
            )
                    miss =0
                if miss:
                    cv2.putText(
                img,
                f"Total_POCS = {total_poc}",
                (img.shape[1] - 660, img.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                thickness=3,
            )
            #     if any(fr_num - j in frame1 for j in range(10)):
            #         cv2.putText(
            #     img,
            #     f"POC happened",
            #     (img.shape[1] - 550, img.shape[0] - 100),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=2,
            #     color=(0, 0, 255),
            #     thickness=3,
            # )
                if fr_num==1:
                    out = cv2.VideoWriter(f"{video_name}_init_end.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (img.shape[1], img.shape[0]))
                    print("Done. Video created")
                if fr_num in init_list:
                    init = end_list[-1]-init_list[-1]+10
                
                if init:
                    init -= 1
                    cv2.putText(
            img,
            f"Shot initiliazed : {init_list[-1]}",
            (img.shape[1] - 660, img.shape[0] - 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 0, 0),
            thickness=3,
                )
                    if init == 0:
                        init_list.pop()
                if fr_num in poc_list:
                    poc_start = end_list[-1]-poc_list[-1]+10
                if poc_start:
                    poc_start -= 1
                    cv2.putText(
                img,
                f"POC happened : {poc_list[-1]}",
                (img.shape[1] - 660, img.shape[0] - 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=3,
            )
                    if poc_start == 0:
                        poc_list.pop()
                    
                if fr_num in end_list:
                    ending = 10
                if ending:
                    ending -= 1
                    cv2.putText(
            img,
            f"Shot completed :{end_list[-1]}",
            (img.shape[1] - 660, img.shape[0] - 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 0, 0),
            thickness=3,
                )
                    if ending == 0:
                        end_list.pop()
                      
                    
                out.write(img)
        cap.release()
        
    def shot_init_comp(self,prob_list,poc_list,results,last_frame):
        # code for POC phases by considering probability for each frame
        self.shot_phases_dict = dict()
        b_prep, b_follow = 11, 33 #backhand preparation and follow phase time obtained from csv analysis
        f_prep, f_follow = 12, 29 #forehand preparation and follow phase time obtained from csv analysis
        print("------ For Backhand ------")
        print(f"Preparation time: {b_prep}, Follow through time: {b_follow}")
        print()
        print("------ For Forehand ------")
        print(f"Preparation time: {f_prep}, Follow through time: {f_follow}")
        print()
        self.results = results
        poc_dict = dict()
        shot_dict = {'forehand':1,'backhand':0,'serve':3}
        columns = ["shot_type","Initialization","POC","Completion"]
        last = 0 #for indicating last frame for shot completion
        comp_frame = 0 #to store the frame of completion of shot
        if os.path.exists("Shot_phases.csv"):
            df = pd.read_csv("Shot_phases.csv")
        else:
            df = pd.DataFrame(columns = columns)
        for i in range(len(poc_list)):
            poc = poc_list[i]
            shot = results[i]["Shot"]
            shot_idx = shot_dict[shot]
            # decoding poc's frame, shot and storing into variables(poc, shot, shot_idx)
            init = 0
            poc_start = poc-25
            if poc < 55:
                poc_start = 30
            for j in range(poc_start,poc):
                # going back 25 frames back to look for shot initialization frame
                if prob_list[j-30][j][2] < .98:
                    # checking probability less than 98%, as it will be less than 98% if is not POC
                    flag = 1
                    # flag_check = prob_list[j-30][j][2]
                    for k in range(j+1,j+6):
                        # checking if the probability of next 5 frames is increasing, if it keeps on increasing then the frame can be classified as shot initialization
                        if prob_list[k-30][k][2] > prob_list[k-1-30][k-1][2]:
                            flag = 0
                            break
                    if flag:
                        poc_dict[j] = []
                        poc_dict[j].append(poc)
                        init = j 
                        break
            if init == 0:
                poc_dict[poc-20] = [poc]
                init = poc-20
            if (poc+35) > last_frame:
                last = last_frame
            else:
                last = poc+35
            miss = 0
            for l in range(poc+8,last):
                if prob_list[l-30][l][shot_idx] < .15:
                    poc_dict[init].append(l)
                    comp_frame = l
                    miss = 1
                    break
            if miss == 0:
                poc_dict[init].append(last)
                comp_frame = last
            self.shot_phases_dict[i] = [poc-init, comp_frame-poc]
            
            
#             frame_list = [shot,init,poc,comp_frame]
#             df1 = pd.DataFrame([frame_list],columns = columns)
#             df = pd.concat([df,df1],ignore_index = True)
            
            if shot == "forehand":
                self.phases_time(shot, poc-init, comp_frame-poc, f_prep, f_follow)
            if shot == "backhand":
                self.phases_time(shot, poc-init, comp_frame-poc, b_prep, b_follow)
                
#         df.to_csv("Shot_phases.csv")
#         print("shot_phases csv is created")
        self.poc_dict = poc_dict
        print("printint shot phases dict: \n",self.shot_phases_dict)
#         self.df = df
#         for i in poc_dict:
#             print("shot inittiated at frame : ",i)
#             print("shot ended at frame : ",poc_dict[i])
            
    def phases_time(self, shot, prep_time, follow_time, true_pre, true_follow):
        print("Shot played: ",shot)
        print("Preparation time in frames: ",prep_time)
        print("follow through time in frames: ",follow_time)
        if prep_time < true_pre:
            print("Took good preparation time before POC")
        else:
            print("Need to be ready or play faster before POC")
        if follow_time < true_follow:
            print("Took good follow through time after POC")
        else:
            print("Need to swing faster after POC")   
        print()
        
        
    def shot_init1_comp(self,prob_list,poc_list,results,last_frame):
        # code for POC phases by considering probability for each frame
        self.results = results
        poc_dict = dict()
        shot_dict = {'forehand':1,'backhand':0,'serve':3}
        for i in range(len(results)):
            poc = poc_list[i]
            shot = results[i]["Shot"]
            shot_idx = shot_dict[shot]
            init = 0
            for j in range(poc-35,poc):
                if prob_list[j-30][j][2] < .98:
                    flag = 1
                    # flag_check = prob_list[j-30][j][2]
                    for k in range(j+1,j+6):
                        if prob_list[k-30][k][2] > prob_list[k-1-30][k-1][2]:
                            flag = 0
                            break
                    if flag:
                        poc_dict[j] = []
                        poc_dict[j].append(poc)
                        init = j 
                        break 
            for j in range(poc,poc+35):
                if prob_list[j-30][j][shot] < .15:
                    poc_dict[init].append(j)
                    break
        self.poc_dict = poc_dict
        print(poc_dict)
                    
                    
                
                    
        
    

