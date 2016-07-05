/**
* @file single_board.cpp
* @author Hamdi Sahloul
* @date September 2014
* @version 0.1
* @brief ROS version of the example named "simple_board" in the Aruco software package.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <aruco/aruco.h>
#include <aruco/boarddetector.h>
#include <aruco/cvdrawingutils.h>

#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ar_sys/utils.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

using namespace aruco;

class ArSysSingleBoard
{
	private:
		cv::Mat inImage, resultImg;
		aruco::CameraParameters camParam;
		bool useRectifiedImages;
		bool draw_markers;
		bool draw_markers_cube;
		bool draw_markers_axis;
    bool publish_tf;
		bool only_4dof;
		MarkerDetector mDetector;
		vector<Marker> markers;
		BoardConfiguration the_board_config;
		BoardDetector the_board_detector;
		Board the_board_detected;
		ros::Subscriber cam_info_sub;
		bool cam_info_received;
		image_transport::Publisher image_pub;
		image_transport::Publisher debug_pub;
		ros::Publisher pose_pub;
		ros::Publisher transform_pub;
		ros::Publisher position_pub;
		std::string board_frame;

		double marker_size;
		std::string board_config;

		ros::NodeHandle nh;
		image_transport::ImageTransport it;
		image_transport::Subscriber image_sub;

		tf::TransformListener _tfListener;

	public:
		ArSysSingleBoard()
			: cam_info_received(false),
			nh("~"),
			it(nh)
		{
			image_pub = it.advertise("result", 1);
			debug_pub = it.advertise("debug", 1);
			pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 100);
			transform_pub = nh.advertise<geometry_msgs::TransformStamped>("transform", 100);
			position_pub = nh.advertise<geometry_msgs::Vector3Stamped>("position", 100);

			nh.param<double>("marker_size", marker_size, 0.05);
			nh.param<std::string>("board_config", board_config, "boardConfiguration.yml");
			nh.param<std::string>("board_frame", board_frame, "");
			nh.param<bool>("image_is_rectified", useRectifiedImages, true);
			nh.param<bool>("draw_markers", draw_markers, false);
			nh.param<bool>("draw_markers_cube", draw_markers_cube, false);
			nh.param<bool>("draw_markers_axis", draw_markers_axis, false);
      nh.param<bool>("publish_tf", publish_tf, false);
			nh.param<bool>("only_4dof", only_4dof, false);

			the_board_config.readFromFile(board_config.c_str());

			ROS_INFO("ArSys node started with marker size of %f m and board configuration: %s",
					 marker_size, board_config.c_str());


		  image_sub = it.subscribe("/image", 1, &ArSysSingleBoard::image_callback, this);
			bool use_camera_info;
	    nh.param<bool>("use_camera_info", use_camera_info, false);
	    if (use_camera_info)
	    {
				cam_info_sub = nh.subscribe("/camera_info", 1, &ArSysSingleBoard::cam_info_callback, this);
			}
			else
			{
				camParam = ar_sys::getCamParams(useRectifiedImages);
				cam_info_received = true;
			}
		}

		void image_callback(const sensor_msgs::ImageConstPtr& msg)
		{
      static tf::TransformBroadcaster br;

			if(!cam_info_received) return;

			cv_bridge::CvImagePtr cv_ptr;
			try
			{
				cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);

				cv::undistort(cv_ptr->image,inImage,camParam.CameraMatrix, camParam.Distorsion);
				camParam.Distorsion = cv::Mat::zeros(4,1,CV_32FC1);
				//inImage = cv_ptr->image;

				resultImg = cv_ptr->image.clone();

				//detection results will go into "markers"
				markers.clear();
				//Ok, let's detect
				mDetector.detect(inImage, markers, camParam, marker_size, false);
				//Detection of the board
				float probDetect = 0.0;
				if(only_4dof)
					probDetect=the_board_detector.detect_4dof(markers, the_board_config, the_board_detected, camParam, marker_size);
				else
					probDetect=the_board_detector.detect(markers, the_board_config, the_board_detected, camParam, marker_size);
				if (probDetect > 0.0)
				{
					tf::Transform transform = ar_sys::getTf(the_board_detected.Rvec, the_board_detected.Tvec);
					cv::Mat cov = the_board_detected.Cov;

					/*double roll, pitch, yaw;
					tf::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
					tf::Quaternion q;
			    q.setEuler(3.14,0.0,yaw);
					transform.setRotation(q);*/

					tf::StampedTransform stampedTransform(transform, msg->header.stamp, "cam_pos", board_frame);

          if (publish_tf)
              br.sendTransform(stampedTransform);

					geometry_msgs::PoseWithCovarianceStamped poseMsg;
					tf::poseTFToMsg(transform, poseMsg.pose.pose);
					poseMsg.header.frame_id = msg->header.frame_id;
					poseMsg.header.stamp = msg->header.stamp;
					for (unsigned i = 0; i < 6; ++i)
				    for (unsigned j = 0; j < 6; ++j)
				      poseMsg.pose.covariance[j + 6 * i] = cov.at<float>(i,j);
					pose_pub.publish(poseMsg);

					geometry_msgs::TransformStamped transformMsg;
					tf::transformStampedTFToMsg(stampedTransform, transformMsg);
					transform_pub.publish(transformMsg);

					geometry_msgs::Vector3Stamped positionMsg;
					positionMsg.header = transformMsg.header;
					positionMsg.vector = transformMsg.transform.translation;
					position_pub.publish(positionMsg);
				}
				else
				{
					geometry_msgs::PoseWithCovarianceStamped poseMsg;
					poseMsg.header.frame_id = msg->header.frame_id;
					poseMsg.header.stamp = msg->header.stamp;
					poseMsg.pose.pose.position.x = 0;
				  poseMsg.pose.pose.position.y = 0;
				  poseMsg.pose.pose.position.z = 0;
				  poseMsg.pose.pose.orientation.x = 0;
				  poseMsg.pose.pose.orientation.y = 0;
				  poseMsg.pose.pose.orientation.z = 0;
				  poseMsg.pose.pose.orientation.w = 0;
					for (unsigned i = 0; i < 6; ++i)
						for (unsigned j = 0; j < 6; ++j)
							poseMsg.pose.covariance[j + 6 * i] = 0.0;
					pose_pub.publish(poseMsg);
				}
				//for each marker, draw info and its boundaries in the image
				for(size_t i=0; draw_markers && i < markers.size(); ++i)
				{
					markers[i].draw(resultImg,cv::Scalar(0,0,255),2);
				}


				if(camParam.isValid() && marker_size != -1)
				{
					//draw a 3d cube in each marker if there is 3d info
					for(size_t i=0; i<markers.size(); ++i)
					{
						if (draw_markers_cube) CvDrawingUtils::draw3dCube(resultImg, markers[i], camParam);
						if (draw_markers_axis) CvDrawingUtils::draw3dAxis(resultImg, markers[i], camParam);
					}
					//draw board axis
					if (probDetect > 0.0) CvDrawingUtils::draw3dAxis(resultImg, the_board_detected, camParam);
				}

				if(image_pub.getNumSubscribers() > 0)
				{
					//show input with augmented information
					cv_bridge::CvImage out_msg;
					out_msg.header.frame_id = msg->header.frame_id;
					out_msg.header.stamp = msg->header.stamp;
					out_msg.encoding = sensor_msgs::image_encodings::RGB8;
					out_msg.image = resultImg;
					image_pub.publish(out_msg.toImageMsg());
				}

				if(debug_pub.getNumSubscribers() > 0)
				{
					//show also the internal image resulting from the threshold operation
					cv_bridge::CvImage debug_msg;
					debug_msg.header.frame_id = msg->header.frame_id;
					debug_msg.header.stamp = msg->header.stamp;
					debug_msg.encoding = sensor_msgs::image_encodings::MONO8;
					debug_msg.image = mDetector.getThresholdedImage();
					debug_pub.publish(debug_msg.toImageMsg());
				}
			}
			catch (cv_bridge::Exception& e)
			{
				ROS_ERROR("cv_bridge exception: %s", e.what());
				return;
			}
		}

		// wait for one camerainfo, then shut down that subscriber
		void cam_info_callback(const sensor_msgs::CameraInfo &msg)
		{
			camParam = ar_sys::getCamParams(msg, useRectifiedImages);
			cam_info_received = true;
			cam_info_sub.shutdown();
		}
};


int main(int argc,char **argv)
{
	ros::init(argc, argv, "ar_single_board");

	ArSysSingleBoard node;

	ros::spin();
}
