#!/usr/bin/env python
import roslib
roslib.load_manifest('hw4')
import sensor_msgs.msg
import rospy
import os
import sys
import cv2.cv as cv
from cv_bridge import CvBridge

def is_rect_nonzero(r):
    (_,_,w,h) = r
    return (w > 0) and (h > 0)

class CamShiftDemo:

    def __init__(self):
        
        #self.frame = cv.CreateImage( (320,200), 8, 3)
        #self.backproject = cv.CreateImage( (320,200), 8, 3)
        #self.header = None
        cv.NamedWindow( "CamShiftDemo", 1 )
        cv.NamedWindow( "HSV", 1 )
        cv.NamedWindow( "HIST", 1 )
        #cv.NamedWindow( "Histogram", 1 )
        self.br=CvBridge();
        self.pause = False

        self.selection = (60, 60, 80, 80)

        print( "Keys:\n"
            "    ESC - quit the program\n"
            "    b - switch to/from backprojection view\n"
            "    p - pause processing\n"
            "To initialize tracking, drag across the object with the mouse\n" )

    def hue_histogram_as_image(self, hist):
        """ Returns a nice representation of a hue histogram """

        histimg_hsv = cv.CreateImage( (320,200), 8, 3)

        mybins = cv.CloneMatND(hist.bins)
        cv.Log(mybins, mybins)
        (_, hi, _, _) = cv.MinMaxLoc(mybins)
        cv.ConvertScale(mybins, mybins, 255. / hi)

        w,h = cv.GetSize(histimg_hsv)
        hdims = cv.GetDims(mybins)[0]
        for x in range(w):
            xh = (180 * x) / (w - 1)  # hue sweeps from 0-180 across the image
            val = int(mybins[int(hdims * x / w)] * h / 255)
            cv.Rectangle( histimg_hsv, (x, 0), (x, h-val), (xh,255,64), -1)
            cv.Rectangle( histimg_hsv, (x, h-val), (x, h), (xh,255,255), -1)

        histimg = cv.CreateImage( (320,200), 8, 3)
        cv.CvtColor(histimg_hsv, histimg, cv.CV_HSV2BGR)
        return histimg

    def detect_and_draw(self, imgmsg):
        
        #print imgmsg.height, " ", imgmsg.width
        #img = cv.CreateImage( (imgmsg.height,imgmsg.width), 8, 3)

        img = self.br.imgmsg_to_cv(imgmsg,"bgr8")
        hsv = cv.CreateImage((imgmsg.height,imgmsg.width), 8, 3)
        cv.CvtColor(img, hsv, cv.CV_BGR2HSV)

        self.hue = cv.CreateImage((imgmsg.height,imgmsg.width), 8, 1)
        cv.Split(hsv, self.hue, None, None, None)

        # Compute back projection
        backproject = cv.CreateImage((imgmsg.height,imgmsg.width), 8, 1)

        # Run the cam-shift
        cv.CalcArrBackProject( [self.hue], backproject, self.hist )
        if is_rect_nonzero(self.selection):
            crit = ( cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
            (iters, (area, value, rect), track_box) = cv.CamShift(backproject, self.selection, crit)
            self.selection = rect

        # If mouse is pressed, highlight the current selected rectangle
        # and recompute the histogram

        
        sub = cv.GetSubRect(img, self.selection)
        save = cv.CloneMat(sub)
        cv.ConvertScale(img, img, 0.5)
        cv.Copy(save, sub)
        x,y,w,h = self.selection
        cv.Rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sel = cv.GetSubRect(self.hue, self.selection )
        cv.CalcArrHist( [sel], self.hist, 0)
        (_, max_val, _, _) = cv.GetMinMaxHistValue( self.hist)
        if max_val != 0:
            cv.ConvertScale(self.hist.bins, self.hist.bins, 255. / max_val)
    	
        cv.EllipseBox( img, track_box, cv.CV_RGB(255,0,0), 3, cv.CV_AA, 0 )

        cv.ShowImage( "CamShiftDemo", img )
        cv.ShowImage( "HSV", hsv )
        cv.ShowImage( "HIST", self.hue_histogram_as_image(self.hist) )
        #print('detect and draw')
        cv.WaitKey(6)
       

    def run(self):
        self.disp_hist = True
        self.backproject_mode = False
        self.hist = cv.CreateHist([180], cv.CV_HIST_ARRAY, [(0,180)], 1 )

        rospy.init_node('blob_tracker')

        rospy.Subscriber('/vrep/follower/visionSensor',sensor_msgs.msg.Image, self.detect_and_draw)
        
        rospy.spin()

if __name__=="__main__":
    
    demo = CamShiftDemo()
    demo.run()
    #cv.DestroyAllWindows()
