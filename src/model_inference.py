#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import cv2, cv_bridge
from scipy.spatial.transform import Rotation as R
from openvino.inference_engine import IECore

class Img_GroundPS:

    def __init__(self, model_path_, ref_path_, image_topic_, depth_topic_, depth_info_topic_, out_img_topic_, out_depth_topic_, out_score_topic_, out_info_topic_):
        self.get_cam_info = False
        self.imgOutput = rospy.Publisher(out_img_topic_, Image, queue_size=1)
        self.scoreOutput = rospy.Publisher(out_score_topic_, Image, queue_size=1)
        self.depthOutput = rospy.Publisher(out_depth_topic_, Image, queue_size=1)
        self.infoOutput = rospy.Publisher(out_info_topic_, CameraInfo, queue_size=1)
        self.image_topic_ = image_topic_
        self.depth_topic_ = depth_topic_
        self.depth_info_topic_ = depth_info_topic_

        rotation = np.array([[0.999994158744812, -0.001701555447652936, -0.0029583023861050606], [0.0016987272538244724, 0.9999980926513672, -0.0009582429775036871], [0.0029599270783364773, 0.0009532120893709362, 0.9999951720237732]])
        translation = np.array([[-0.05916924029588699], [0.00016153734759427607], [0.000526325311511755]])
        rgb_to_depth = np.linalg.inv(np.vstack((np.hstack((rotation,translation)),np.array([0,0,0,1])))) #rgb_to_depth
        transformation = np.vstack((np.hstack((np.array([[1,0,0],[0,0.8660254,0.5],[0, -0.5,0.8660254]]),np.array([[0],[-1],[0]]))),np.array([0,0,0,1])))
        rgb_to_depth = np.matmul(transformation,rgb_to_depth)

        P_rgb = np.array([[381.8409423828125, 0.0, 315.6470031738281, 0.0], [0.0, 381.46221923828125, 243.59426879882812, 0.0], [0.0, 0.0, 1.0, 0.0]])
        P_depth = np.array([[106.16276550292969, 0.0, 105.9286117553711, 0.0], [0.0, 106.16276550292969, 59.86717987060547, 0.0], [0.0, 0.0, 1.0, 0.0]])

        H_depth = np.matmul(P_depth, transformation)
        H_depth = np.delete(H_depth,2,1)

        H_rgb = np.matmul(P_rgb, rgb_to_depth)
        H_rgb = np.delete(H_rgb,2,1)

        self.homography = np.matmul(H_depth,np.linalg.inv(H_rgb))

        # Load network to Inference Engine
        ie = IECore()
        net_ = ie.read_network(model=model_path_)
        self.exec_net_  = ie.load_network(network=net_, device_name="MULTI:GPU,CPU")
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

    def img_callback(self, img_msg, depth_msg, info_msg):
        bgr_img = None
        depth_img = None
        bridge = cv_bridge.CvBridge()

        try:
            bgr_img = cv2.resize(bridge.imgmsg_to_cv2(img_msg, 'bgr8'),(256,256))
            depth_img = bridge.imgmsg_to_cv2(depth_msg, '16UC1')# [:,39:199]
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

        # depth_scores = cv2.warpPerspective(cv2.resize(raw_score,(640,480)),self.homography,(212,120))
        depth_scores = cv2.warpAffine(cv2.resize(raw_score,(640,480)),self.homography[:2,:],(212,120))
        
        # depth_img = np.where(np.any(depth_scores != 0, axis = 2), depth_img, 0) # show small obstacles
        depth_img = np.where(depth_scores > 0.5, depth_img, 0)
        score_img = np.array([np.zeros([120,212]),np.zeros([120,212]),depth_scores]).transpose(1, 2, 0)*255

        HM_out_msg = bridge.cv2_to_imgmsg(HM, 'bgr8')
        depth_out_msg = bridge.cv2_to_imgmsg(depth_img, '16UC1')
        score_out_msg = bridge.cv2_to_imgmsg(score_img.astype('uint8'), 'bgr8')

        timestamp = rospy.Time.now()
        HM_out_msg.header.stamp = timestamp
        depth_out_msg.header.stamp = timestamp
        score_out_msg.header.stamp = timestamp
        info_msg.header.stamp = timestamp
        depth_out_msg.header.frame_id = info_msg.header.frame_id
        score_out_msg.header.frame_id = info_msg.header.frame_id

        self.imgOutput.publish(HM_out_msg)
        self.depthOutput.publish(depth_out_msg)
        self.scoreOutput.publish(score_out_msg)
        self.infoOutput.publish(info_msg)

if __name__ == "__main__" :
    rospy.init_node('Img_GroundPS')
    try :
        # ir_16_path = '/home/upboard/OS_TR/openvino_models/FP16/ostr_mb3_large_ver2_shrinked/ostr_mb3_large_ver2_shrinked.xml'
        # ir_16_path = '/home/upboard/OS_TR/openvino_models/FP16/ostr_mb3_large_ver4_perlin16_shrinked/ostr_mb3_large_ver4_perlin_shrinked.xml'
        ir_16_path = '/home/upboard/OS_TR/openvino_models/FP16/ostr_mb3_large_ver4_perlin16_shrinked/ostr_mb3_large_ver4_perlin_shrinked.xml'
        ref_path = '/home/upboard/OS_TR/test_images/ref.jpg'
        IGSP = Img_GroundPS(ir_16_path, ref_path, "/image_in", "/depth_in", "depth_info_in", "/ground_score/heatmap", "/ground_score/depth", "/ground_score/score", "/ground_score/camera_info")
        IGSP.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass