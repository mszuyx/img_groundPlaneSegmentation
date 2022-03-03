#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs import point_cloud2
import std_msgs.msg as std_msgs
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import cv2, cv_bridge
from image_geometry import PinholeCameraModel
from scipy.spatial.transform import Rotation as R
from openvino.inference_engine import IECore
from pc_gps.msg import gpParam

class Img_GroundPS:

    def __init__(self, model_path_, ref_path_, image_topic_, info_topic_, param_topic_, out_img_topic_, out_pt_topic_, parent_frame_):
        self.get_cam_info = False
        self.imgOutput = rospy.Publisher(out_img_topic_, Image, queue_size=1)
        self.pointOutput = rospy.Publisher(out_pt_topic_, PointCloud2, queue_size=1)
        self.parent_frame_ = parent_frame_
        self.image_topic_ = image_topic_
        self.info_topic_ = info_topic_
        self.param_topic_ = param_topic_
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
        rospy.Subscriber(self.info_topic_, CameraInfo, self.info_callback, queue_size=1)
        image_sub = Subscriber(self.image_topic_, Image)
        param_sub = Subscriber(self.param_topic_, gpParam)
        ts = ApproximateTimeSynchronizer([image_sub, param_sub], queue_size=20, slop=0.1)
        ts.registerCallback(self.img_callback)
        rospy.loginfo("Neural network online!")

    def info_callback(self,info_msg):
        if self.get_cam_info:
            return
        self.cam_model_ = PinholeCameraModel()
        self.cam_model_.fromCameraInfo(info_msg)
        R = np.array([[0,0,1],
                      [0,1,0],
                      [-1,0,0]])
        # R = np.array([[-0.0872,0,0.9962],
        #               [0,1,0],
        #               [-0.9962,0,-0.0872]])
        # R = np.eye(3)
        t = np.array([[-3],[0],[0]])
        P = np.dot(self.cam_model_.intrinsicMatrix(),np.hstack((R,t)))
        self.m_inv_3x3 = np.linalg.inv(P[:, :3]).astype('float32')
        self.get_cam_info  = True
        rospy.loginfo("Got camera projection matrix!")

    def to_point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx7 array of xyz positions (m) and rgba colors (0..1)
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        data = points.astype(dtype).tobytes()
        fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]
        header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())
        return PointCloud2(
                header=header,
                height=1,
                width=points.shape[0],
                is_dense=False,
                is_bigendian=False,
                fields=fields,
                point_step=(itemsize * 3),
                row_step=(itemsize * 3 * points.shape[0]),
                data=data)

    def cast_2d_points_as_3d_rays(self, sub_pixels_nx2: np.array, m_inv_3x3: np.array):
        """
        see Harley & Zisserman pg 162, section 6.2.2, figure 6.14
        see https://math.stackexchange.com/a/597489/541203
        cast rays from camera center through the sub_pixels_nx2. Return camera center and ray directions.
        """
        # projection matrix + pixel locations to ray directions
        sub_pixels_homo_3xn = np.hstack([sub_pixels_nx2, np.ones((sub_pixels_nx2.shape[0], 1))]).T
        ray_directions_3xn = np.dot(m_inv_3x3,sub_pixels_homo_3xn)
        return ray_directions_3xn

    def find_true_xyz(self, params, ray_3xn):
        alpha_1xn = params[3]/np.dot(params[:3],ray_3xn)
        xyz_3xn = np.multiply(ray_3xn,alpha_1xn)
        return xyz_3xn

    def img_callback(self, img_msg, param_msg):
        bgr_img = None
        ori_img = None
        bridge = cv_bridge.CvBridge()
        params = np.asarray(param_msg.data).astype('float32')
        params[[0,1,2]] = params[[2,0,1]]
        params[0] *= -1
        # params = np.array([-0.3,0,0.95394,-1])

        try:
            ori_img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            bgr_img = cv2.resize(ori_img,(256,256)) # what is the effect of resizing on back projection
        except cv_bridge.CvBridgeError as e:
            rospy.logerr( 'image message to cv conversion failed :' )
            rospy.logerr( e )
            print( e )
            return
        frame = (cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)/255).astype(np.float32)
        res_net_ = self.exec_net_.infer(inputs={self.input_layer_1: np.expand_dims(np.transpose(frame,(2, 0, 1)), axis=0), self.input_layer_2: self.ref_img})
        scores = (255*np.squeeze(res_net_[self.output_layer_])).astype(np.uint8)
        HM = cv2.applyColorMap(scores, cv2.COLORMAP_JET)
        HM = (self.alpha*bgr_img + (1-self.alpha)*HM).astype(np.uint8)
        self.imgOutput.publish(bridge.cv2_to_imgmsg(HM, 'bgr8'))
        # print("callback")

        if self.get_cam_info:
            drivable = np.argwhere(np.any(ori_img != 0, axis = 2)).astype('float32')
            # ray = self.cam_model_.projectPixelTo3dRay(drivable[0])
            ray_c = self.cast_2d_points_as_3d_rays(drivable,self.m_inv_3x3)
            print(ray_c.T[0,:])
            # ground_score_pts = self.find_true_xyz(params,ray_c)
            self.pointOutput.publish(self.to_point_cloud(ray_c.T,self.parent_frame_))


if __name__ == "__main__" :
    rospy.init_node('Img_GroundPS')
    try :
        ir_16_path = '/home/upboard/OS_TR/openvino_models/FP16/ostr_mb3_large_ver2_shrinked.xml'
        ref_path = '/home/upboard/OS_TR/test_images/ref_4.jpg'
        IGSP = Img_GroundPS(ir_16_path, ref_path, "/image_in", "/d455/color/camera_info", "/groundplane_param","/ground_score","/ground_score_pts","d455_link")
        IGSP.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass