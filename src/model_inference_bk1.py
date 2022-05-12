#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import cv2, cv_bridge
from scipy.spatial.transform import Rotation as R
from openvino.inference_engine import IECore
from image_geometry import PinholeCameraModel
from realsense2_camera.msg import Extrinsics

class Img_GroundPS:

    def __init__(self, model_path_, ref_path_):
        self.get_cam_info = False
        self.imgOutput = rospy.Publisher("/ground_score/heatmap", Image, queue_size=1)
        self.scoreOutput = rospy.Publisher("/ground_score/depth", Image, queue_size=1)
        self.depthOutput = rospy.Publisher("/ground_score/score", Image, queue_size=1)
        self.infoOutput = rospy.Publisher("/ground_score/camera_info", CameraInfo, queue_size=1)
        self.image_topic_ = "/image_in"
        self.depth_topic_ = "/depth_in"
        self.depth_info_topic_ = "/depth_info_in"
        self.get_cam_info = False
        self.get_depth_info = False
        self.get_extrinsics = False

        self.homography = np.eye(3)
        self.depth_model_ = None
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)

        self.cam_info_sub = rospy.Subscriber("/camera_info_in", CameraInfo, self.info_callback, queue_size=1)
        self.extrinsics_sub = rospy.Subscriber("/depth_to_color", Extrinsics, self.extrinsics_callback, queue_size=1)
        
        # Load network to Inference Engine
        ie = IECore()
        net_ = ie.read_network(model=model_path_+'.xml', weights=model_path_+'.bin')
        self.exec_net_  = ie.load_network(network=net_, device_name="MULTI:GPU,CPU", num_requests=0)
        # Get names of input and output layers
        input_iter = iter(self.exec_net_.input_info)
        self.input_layer_1 = next(input_iter)
        self.input_layer_2 = next(input_iter)
        self.output_layer_ = next(iter(self.exec_net_.outputs))
        ref_img = (cv2.cvtColor(cv2.resize(cv2.imread(ref_path_, cv2.IMREAD_COLOR),(256,256)), cv2.COLOR_BGR2RGB)/255).astype(np.float32)
        self.ref_img = np.expand_dims(np.transpose(ref_img,(2, 0, 1)), axis=0)
        self.alpha = 0.5

    def run(self):
        image_sub = Subscriber(self.image_topic_, Image)
        depth_sub = Subscriber(self.depth_topic_, Image)
        depth_info_sub = Subscriber(self.depth_info_topic_, CameraInfo)
        ts = ApproximateTimeSynchronizer([image_sub, depth_sub, depth_info_sub], queue_size=20, slop=0.1)
        ts.registerCallback(self.img_callback)
        rospy.loginfo("Neural network online!")

    def extrinsics_callback(self,extrinsics_msg):
        if self.get_extrinsics:
            self.extrinsics_sub.unregister()
            return
        self.rotation = np.asarray(extrinsics_msg.rotation)  
        self.translation = np.asarray(extrinsics_msg.translation)
        self.rotation = self.rotation.reshape((3,3))
        self.translation = self.translation.reshape((3,1))
        self.get_extrinsics = True
        rospy.loginfo("Got camera extrinsics info!")

    def info_callback(self,info_msg):
        if self.get_cam_info:
            self.cam_info_sub.unregister()
            return
        if self.get_depth_info and self.get_extrinsics:
            cam_model_ = PinholeCameraModel()
            cam_model_.fromCameraInfo(info_msg)
            
            rgb_to_depth = np.linalg.inv(np.vstack((np.hstack((self.rotation,self.translation)),np.array([0,0,0,1])))) #rgb_to_depth
            transformation = np.vstack((np.hstack((np.array([[1,0,0],[0,0.8660254,0.5],[0, -0.5,0.8660254]]),np.array([[0],[-1],[0]]))),np.array([0,0,0,1])))
            rgb_to_depth = np.matmul(transformation,rgb_to_depth)

            P_rgb = cam_model_.projectionMatrix()
            P_depth = self.depth_model_.projectionMatrix()

            H_depth = np.matmul(P_depth, transformation)
            H_depth = np.delete(H_depth,2,1)

            H_rgb = np.matmul(P_rgb, rgb_to_depth)
            H_rgb = np.delete(H_rgb,2,1)
 
            self.homography = np.matmul(H_depth,np.linalg.inv(H_rgb))
            # [[ 1.90316610e-01  1.12064484e-02  1.55960742e+01]
            # [ 8.34631815e-04  1.90185279e-01 -7.94971261e+00]
            # [ 3.24382058e-06 -4.90432041e-07  9.89395512e-01]]

            self.get_cam_info  = True
            rospy.loginfo("Got rgb camera info!")

    def img_callback(self, img_msg, depth_msg, info_msg):
        if not self.get_depth_info:
            self.depth_model_ = PinholeCameraModel()
            self.depth_model_.fromCameraInfo(info_msg)
            self.get_depth_info = True
            rospy.loginfo("Got depth camera info!")

        bgr_img = None
        depth_img = None
        bridge = cv_bridge.CvBridge()

        try:
            bgr_img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            bgr_height = bgr_img.shape[0]
            bgr_width = bgr_img.shape[1]
            bgr_img = cv2.resize(bgr_img,(256,256))
            depth_img = bridge.imgmsg_to_cv2(depth_msg, '16UC1')# [:,39:199]
            depth_height = depth_img.shape[0]
            depth_width = depth_img.shape[1]
        except cv_bridge.CvBridgeError as e:
            rospy.logerr( 'image message to cv conversion failed :' )
            rospy.logerr( e )
            print( e )
            return
        frame = (cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)/255).astype(np.float32)
        res_net_ = self.exec_net_.infer(inputs={self.input_layer_1: np.expand_dims(np.transpose(frame,(2, 0, 1)), axis=0), self.input_layer_2: self.ref_img})
        raw_score = np.squeeze(res_net_[self.output_layer_])
        scores = (255*raw_score).astype(np.uint8)
        HM = cv2.applyColorMap(scores, cv2.COLORMAP_JET)
        HM = (self.alpha*bgr_img + (1-self.alpha)*HM).astype(np.uint8)
        HM_out_msg = bridge.cv2_to_imgmsg(HM, 'bgr8')

        timestamp = rospy.Time.now()
        HM_out_msg.header.stamp = timestamp
        self.imgOutput.publish(HM_out_msg)

        if self.get_cam_info and self.get_depth_info and self.get_extrinsics:
            # depth_scores = cv2.warpPerspective(cv2.resize(raw_score,(640,480)),self.homography,(212,120))
            depth_scores = cv2.warpAffine(cv2.resize(raw_score,(bgr_width,bgr_height)),self.homography[:2,:],(depth_width,depth_height))#(212,120)
            
            # depth_img = np.where(np.any(depth_scores != 0, axis = 2), depth_img, 0) # show small obstacles
            depth_img = np.where(depth_scores > 0.5, depth_img, 0)
            score_img = np.array([np.zeros([depth_height,depth_width]),np.zeros([depth_height,depth_width]),depth_scores]).transpose(1, 2, 0)*255 #[120,212]

            depth_out_msg = bridge.cv2_to_imgmsg(depth_img, '16UC1')
            score_out_msg = bridge.cv2_to_imgmsg(score_img.astype('uint8'), 'bgr8')
            depth_out_msg.header.stamp = timestamp
            score_out_msg.header.stamp = timestamp
            info_msg.header.stamp = timestamp
            depth_out_msg.header.frame_id = info_msg.header.frame_id
            score_out_msg.header.frame_id = info_msg.header.frame_id

            self.depthOutput.publish(depth_out_msg)
            self.scoreOutput.publish(score_out_msg)
            self.infoOutput.publish(info_msg)

if __name__ == "__main__" :
    rospy.init_node('Img_GroundPS')
    try :
        # ir_16_path = '/home/upboard/OS_TR/openvino_models/FP16/ostr_mb3_large_ver2_shrinked/ostr_mb3_large_ver2_shrinked'
        ir_16_path = '/home/upboard/OS_TR/openvino_models/FP16/ostr_mb3_large_ver4_perlin16_shrinked/ostr_mb3_large_ver4_perlin_shrinked'
        ref_path = '/home/upboard/OS_TR/test_images/ref.jpg'
        IGSP = Img_GroundPS(ir_16_path, ref_path)
        IGSP.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
