#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import cv2, cv_bridge
# from scipy.spatial.transform import Rotation as R
from openvino.inference_engine import IECore
# from image_geometry import PinholeCameraModel
# from realsense2_camera.msg import Extrinsics
from p2i_cast.msg import homoMatrix

class Img_GroundPS:

    def __init__(self, model_path_, ref_path_):
        self.imgOutput = rospy.Publisher("/ground_score/heatmap", Image, queue_size=1)
        self.scoreOutput = rospy.Publisher("/ground_score/score", Image, queue_size=1)
        self.depthOutput = rospy.Publisher("/ground_score/depth", Image, queue_size=1)
        self.infoOutput = rospy.Publisher("/ground_score/camera_info", CameraInfo, queue_size=1)
        self.image_topic_ = "/image_in"
        self.depth_topic_ = "/depth_in"
        self.depth_info_topic_ = "/depth_info_in"

        self.get_homo = False
        self.homography = np.eye(3)
        self.homo_sub = rospy.Subscriber("/homo_matrix_in", homoMatrix, self.homo_callback, queue_size=1)
        
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

    def homo_callback(self,homo_msg):
        if self.get_homo:
            self.homo_sub.unregister()
            return
        self.homography = np.asarray(homo_msg.rgb_to_depth).reshape((3,3))
        self.get_homo  = True
        rospy.loginfo("Got homography matrix!")
        print(self.homography)

    def img_callback(self, img_msg, depth_msg, info_msg):
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
        HM_out_msg = bridge.cv2_to_imgmsg(cv2.resize(HM,(bgr_width,bgr_height)), 'bgr8')

        timestamp = rospy.Time.now()
        HM_out_msg.header.stamp = timestamp
        self.imgOutput.publish(HM_out_msg)

        if self.get_homo:
            depth_scores = cv2.warpAffine(cv2.resize(raw_score,(bgr_width,bgr_height)),self.homography[:2,:],(depth_width,depth_height))#(212,120)

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
