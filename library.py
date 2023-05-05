
from ultralytics import YOLO
import pandas as pd
import re
import numpy as np
import cv2
import math

BATSMAN_HEIGHT=191
CORNER_POINT_A=(-140,0,0)
CORNER_POINT_B=(140,0,0)
CORNER_POINT_C=(166, -321,0)
CORNER_POINT_D=(-166, -321,0)

def open_video(video_path):
    vid=list()
    cap=cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc=0
    while True:
        ret, frame = cap.read()
        if not ret: break # break if no next frame
        
        # buf[fc]=frame # append frame
        # fc+=1
        vid.append(frame)
    # release and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    return vid

def save_video(vid,name):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    frame_width=len(vid[0][0])
    frame_height=len(vid[0])
    out = cv2.VideoWriter(name, fourcc, 25.0, (frame_width, frame_height))
    for frame in vid:
        out.write(frame)
    out.release

def suppress_duplicates(detections):
    # assuming your data frame is named `detections`
    # and has the columns `Frame`, `Class`, and `Confidence`

    # group by frame number and class name
    groups = detections.groupby(['Frame', 'Class'])

    # filter rows to keep only the detection with the highest confidence score for each group
    filtered_detections = groups.apply(lambda x: x.loc[x['Confidence'].idxmax()])

    # reset index of the filtered data frame
    filtered_detections = filtered_detections.reset_index(drop=True)
    
    return filtered_detections

def detect(vid):
    model=YOLO('v2.pt')
    results=model(vid)
    df=pd.DataFrame(columns=["Frame","Class","Confidence","Xmin","Ymin","Xmax","Ymax","Xc","Yc","W","H","A"])
    for result in results:
        for box in result.boxes:
            xyxy=box.xyxy.numpy()[0]
            xywh=box.xywh.numpy()[0]
            frame_number=int(re.search('(?<=image)\d+',result.path).group())
            cls="Bat" if box.cls==2 else "Ball" if box.cls==0 else "Batsman"
            df.loc[len(df)]=[frame_number,cls,int(float(box.conf)*100),xyxy[0],xyxy[1],xyxy[2],xyxy[3],xywh[0],xywh[1],xywh[2],xywh[3],xywh[2]*xywh[3]]
    return suppress_duplicates(df)

def calculate_distances(detections):
    bat_ball = detections.loc[detections['Class'].isin(['Bat', 'Ball'])]

    # group rows by Frame
    groups = bat_ball.groupby('Frame')

    # compute distance between Bat and Ball coordinates for each group
    distances = []
    for name, group in groups:
        bat_coords = group.loc[group['Class'] == 'Bat', ['Xc', 'Yc']].values
        ball_coords = group.loc[group['Class'] == 'Ball', ['Xc', 'Yc']].values
        if len(bat_coords) > 0 and len(ball_coords) > 0:
            # distance = np.linalg.norm(bat_coords - ball_coords)
            distance=math.dist(bat_coords[0],ball_coords[0])
            distances.append((name, distance))

    # create a new data frame with the distances
    distances_df = pd.DataFrame(distances, columns=['Frame', 'Distance'])
    return distances_df

def detect_impact_frame(detections):
    distances_df=calculate_distances(detections)
    f_impact=distances_df.loc[distances_df['Distance'].idxmin()]["Frame"]
    return int(f_impact)

def detect_bounce_frame(detections,f_impact):
    ball_detections=detections.loc[(detections["Class"]=="Ball") & (detections["Frame"]<f_impact)]#class is ball and frame is before impact frame
    f_bounce=ball_detections.loc[ball_detections['Yc'].idxmax()]["Frame"]#get point where y-axis is max
    return int(f_bounce)

def visualize_bounding_boxes(detections, vid, classes):
    # create a list to store the output frames
    output_frames = []
    
    # loop over each frame in the video
    for frame_number, frame in enumerate(vid):
        # create a copy of the frame
        output_frame = np.copy(frame)
        
        # select rows for the specified classes
        selected_rows = detections[detections['Class'].isin(classes) & (detections['Frame'] == frame_number)]
        
        # loop over the selected rows and draw bounding boxes on the output frame
        for index, row in selected_rows.iterrows():
            class_name = row['Class']
            xmin = int(row['Xmin'])
            ymin = int(row['Ymin'])
            xmax = int(row['Xmax'])
            ymax = int(row['Ymax'])
            color = (0, 255, 0)  # set the color for the bounding box
            
            # draw the bounding box on the output frame
            cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(output_frame, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # add the output frame to the list of output frames
        output_frames.append(output_frame)
    
    # return the list of output frames
    return output_frames




def unzoom(x,y,detections,frame):
    d_height=191
    f_height=int(detections.loc[(detections["Class"].isin(["Batsman"]))&(detections["Frame"]==frame)]["H"])
    zf=f_height/d_height
    x/=zf
    y/=zf
    return (int(x),int(y))


def estimate_bowling_side(detections):
    bat_ball = detections.loc[detections['Class'].isin(['Ball', 'Batsman'])]

    # group rows by Frame
    groups = bat_ball.groupby('Frame')

    # compute distance between Bat and Ball coordinates for each group
    distances = []
    for name, group in groups:
        batsman_coords = group.loc[group['Class'] == 'Batsman', ['Xc', 'Yc']].values
        ball_coords = group.loc[group['Class'] == 'Ball', ['Xc', 'Yc']].values
        if len(batsman_coords) > 0 and len(ball_coords) > 0:
            # distance = np.linalg.norm(bat_coords - ball_coords)
            x_vect=batsman_coords[0][0]-ball_coords[0][0]
            if x_vect>0:
                return "LEFT"
            else:
                return "RIGHT"

def calc_coords_of_ball_relative_to_batsman(detections,cls,f):
    Batsman=detections.loc[detections["Class"].isin(["Batsman"]) & (detections["Frame"]==f)]
    Ball=detections.loc[detections["Class"].isin([cls]) & (detections["Frame"]==f)]
    P_X_Batsman=(Batsman["Xmin"]+Batsman["Xmax"])/2
    P_Y_Batsman=Batsman["Ymax"]
    X=int(Ball['Xc'])-int(P_X_Batsman)
    # Y=(frame_height(vid)-int(Ball['Yc']))-(frame_height(vid)-int(Batsman['Yc']))
    Y=int(P_Y_Batsman)-int(Ball['Yc'])
    return (X,Y)
            
def estimate_bounce_point(detections):
    f_impact=detect_impact_frame(detections)
    f_bounce=detect_bounce_frame(detections,f_impact)
    (X,Y)=calc_coords_of_ball_relative_to_batsman(detections,"Ball",f_bounce)
    (X,Y)=unzoom(X,Y,detections,f_bounce)
    return (X,Y)

def estimate_impact_point(detections):
    f_impact=detect_impact_frame(detections)
    (X,Y)=calc_coords_of_ball_relative_to_batsman(detections,"Ball",f_impact)
    (X,Y)=unzoom(X,Y,detections,f_impact)
    return (X,Y)

def estimate_trajectory(detections):
    f_impact=detect_impact_frame(detections)
    (X,Y)=calc_coords_of_ball_relative_to_batsman(detections,"Ball",f_impact+1)
    (X,Y)=unzoom(X,Y,detections,f_impact)
    return (X,Y)

def estimate_release_point(bowlingSide):
    if "LEFT":
        return (-83,-321,BATSMAN_HEIGHT)
    else:
        return (83,-321,BATSMAN_HEIGHT)
        