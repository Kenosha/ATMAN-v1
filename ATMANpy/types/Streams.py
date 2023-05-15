# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:03:08 2023

@author: gianl
"""

from ATMANpy.types.Frames import BGRFrame, GreyFrame, HSVFrame
from ATMANpy.types.Features import StreamFeatures, StreamFeature

from ATMANpy.funcs.draw_funcs import draw

from dataclasses import dataclass
import cv2 as cv

from ATMANpy.funcs.util_funcs import get_positional_arguments, get_return_names

def array_to_frames(bgrArray, frameList):
    ### frameList: a list of any of these elements: 'cFrm', 'gFrm', 'hFrm'
    ### takes in a raw bgr array, and returns, in order of frameList, the required frames
    
    bgrFrm = BGRFrame(bgrArray)
    
    outList = []
    for name in frameList:
        if name == 'cFrm':
            outList.append(bgrFrm)
        if name == 'gFrm':
            outList.append(bgrFrm.to_grey())
        if name == 'hFrm':
            outList.append(bgrFrm.to_HSV())

    return outList

@dataclass(frozen = True)
class Stream:
    cap: cv.VideoCapture
    start_frame: int
    stop_frame: int
    features: StreamFeatures = StreamFeatures({})
        
    def __post_init__(self):
        mySF = self.features
        if isinstance(self.features, StreamFeature):
            ### transform StreamFeature into StreamFeatures
            mySF = mySF.unsqueeze()
        ### conform StreamFeatures to start and stop frame
        # mySF = mySF.conform(list(range(self.start_frame, self.stop_frame + 1))) ### capture contains last frame!
        object.__setattr__(self, 'features', mySF)
    
    def __getitem__(self, frame_num):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        return BGRFrame(frame)
        
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
            
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.start_frame)
        frmNum = self.start_frame
        
        while self.cap.isOpened() and (self.start_frame <= frmNum < self.stop_frame):
            frmNum = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
            ret, frame = self.cap.read()
            
            frame = yield frmNum, frame # take in frame, possibly altered, to show
            # frame = yield frame
            
            if frame_counter:
                cv.putText(frame, str(frmNum), org = (40, 40), 
                           fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 2,
                           color = (0, 255, 0), thickness = 3)
            
            
            if frame is not None:
                cv.imshow('frame', frame)
                if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                    cv.destroyAllWindows()  
                    yield None
        
        else:
            cv.destroyAllWindows()

    def show(self, ms_between_frames = 25, frame_counter = False):
        gen = self.generator(ms_between_frames = ms_between_frames, frame_counter = frame_counter)
        frmNum, rawArr = next(gen) ### prime the generator
        drawFunc = draw(self.features.get_features())
        
        while True:
            cFrm = BGRFrame(rawArr)
            
            try:
                featDict = self.features[frmNum]
            except KeyError:
                sendArr = rawArr
            else:
                sendFrm = drawFunc({'cFrm': cFrm}, featDict)
                sendArr = sendFrm.array  
                 
            try:
                frmNum, rawArr = gen.send(sendArr)
            except StopIteration:
                cv.destroyAllWindows()
                break 
                

    def apply(self, mySF: StreamFeatures):
        return Stream(self.cap, self.start_frame, self.stop_frame, features = mySF)

    def extract(self, featureFunc, **kwargs): ### cv_funcs.get_stream_features
        
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)

        gen = self.generator()
        frmNum, rawArr = next(gen) ### prime the generator
        
        streamDict = {}
        while True:
            
            ### get arguments to plug into feature extraction function
            args = []
            for argStr in funcArgs:
                if argStr == 'cFrm':
                    argObj = BGRFrame(rawArr)
                elif argStr == 'gFrm':
                    argObj = BGRFrame(rawArr).to_grey()
                elif argStr == 'hFrm':
                    argObj = BGRFrame(rawArr).to_HSV()
                else:
                    try:
                        argObj = self.features[frmNum][argStr]
                    except KeyError as e:
                        raise e
                args.append(argObj)
                
            assert len(args) == len(funcArgs)
            
            if len(funcVals) == 1:
                myFeature = featureFunc(*args, **kwargs)
                streamDict[frmNum] = myFeature
            else:
                myFeatures = featureFunc(*args, **kwargs)
                streamDict[frmNum] = dict(zip(funcVals, myFeatures))
            
            try:
                frmNum, rawArr = gen.send(rawArr)
            except StopIteration:
                break 
        
        if len(funcVals) == 1:
            return StreamFeature(streamDict, name = funcVals[0])
        else:
            return StreamFeatures(streamDict)
            
            
            
            
        
        
        