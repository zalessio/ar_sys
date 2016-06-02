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

using namespace aruco;

class ArSysSingleBoard
{
	private:
		cv::Mat inImage, resultImg;
		cv::Mat inImage2, resultImg2;
		aruco::CameraParameters camParam;
		aruco::CameraParameters camParam2;
		bool useRectifiedImages;
		bool draw_markers;
		bool draw_markers_cube;
		bool draw_markers_axis;
    bool publish_tf;
		MarkerDetector mDetector;
		MarkerDetector mDetector2;
		vector<Marker> markers;
		vector<Marker> markers2;
		BoardConfiguration the_board_config;
		BoardDetector the_board_detector;
		BoardDetector the_board_detector2;
		Board the_board_detected;
		Board the_board_detected2;
		ros::Subscriber cam_info_sub;
		ros::Subscriber cam_info_sub2;
		bool cam_info_received;
		bool cam_info_received2;
		image_transport::Publisher image_pub;
		image_transport::Publisher debug_pub;
		ros::Publisher pose_pub;
		ros::Publisher pose_pub2;
		ros::Publisher transform_pub;
		ros::Publisher transform_pub2;
		ros::Publisher position_pub;
		ros::Publisher position_pub2;
		std::string board_frame;

		double marker_size;
		std::string board_config;

		ros::NodeHandle nh;
		image_transport::ImageTransport it;
		image_transport::Subscriber image_sub;
		image_transport::Subscriber image_sub2;

		tf::TransformListener _tfListener;

	public:
		ArSysSingleBoard()
			: cam_info_received(false),
			nh("~"),
			it(nh)
		{
			image_pub = it.advertise("result", 1);
			debug_pub = it.advertise("debug", 1);
			pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 100);
			transform_pub = nh.advertise<geometry_msgs::TransformStamped>("transform", 100);
			position_pub = nh.advertise<geometry_msgs::Vector3Stamped>("position", 100);
			pose_pub2 = nh.advertise<geometry_msgs::PoseStamped>("pose2", 100);
			transform_pub2 = nh.advertise<geometry_msgs::TransformStamped>("transform2", 100);
			position_pub2 = nh.advertise<geometry_msgs::Vector3Stamped>("position2", 100);

			nh.param<double>("marker_size", marker_size, 0.05);
			nh.param<std::string>("board_config", board_config, "boardConfiguration.yml");
			nh.param<std::string>("board_frame", board_frame, "");
			nh.param<bool>("image_is_rectified", useRectifiedImages, true);
			nh.param<bool>("draw_markers", draw_markers, false);
			nh.param<bool>("draw_markers_cube", draw_markers_cube, false);
			nh.param<bool>("draw_markers_axis", draw_markers_axis, false);
            nh.param<bool>("publish_tf", publish_tf, false);

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

			image_sub2 = it.subscribe("/image2", 1, &ArSysSingleBoard::image_callback2, this);
	    if (use_camera_info)
	    {
				cam_info_sub2 = nh.subscribe("/camera_info2", 1, &ArSysSingleBoard::cam_info_callback2, this);
			}
			else
			{
				camParam2 = ar_sys::getCamParams(useRectifiedImages);
				cam_info_received2 = true;
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
				inImage = cv_ptr->image;
				resultImg = cv_ptr->image.clone();

				//detection results will go into "markers"
				markers.clear();
				//Ok, let's detect
				mDetector.detect(inImage, markers, camParam, marker_size, false);
				//Detection of the board
				float probDetect=the_board_detector.detect(markers, the_board_config, the_board_detected, camParam, marker_size);
				if (probDetect > 0.0)
				{
					tf::Transform transform = ar_sys::getTf(the_board_detected.Rvec, the_board_detected.Tvec);
					tf::StampedTransform stampedTransform(transform, msg->header.stamp, "cam_pos", board_frame);

          if (publish_tf)
              br.sendTransform(stampedTransform);

					geometry_msgs::PoseStamped poseMsg;
					tf::poseTFToMsg(transform, poseMsg.pose);
					poseMsg.header.frame_id = msg->header.frame_id;
					poseMsg.header.stamp = msg->header.stamp;
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
					geometry_msgs::PoseStamped poseMsg;
					poseMsg.header.frame_id = msg->header.frame_id;
					poseMsg.header.stamp = msg->header.stamp;
					poseMsg.pose.position.x = 0;
				  poseMsg.pose.position.y = 0;
				  poseMsg.pose.position.z = 0;
				  poseMsg.pose.orientation.x = 0;
				  poseMsg.pose.orientation.y = 0;
				  poseMsg.pose.orientation.z = 0;
				  poseMsg.pose.orientation.w = 0;
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

		void image_callback2(const sensor_msgs::ImageConstPtr& msg)
		{
      static tf::TransformBroadcaster br;

			if(!cam_info_received2) return;

			cv_bridge::CvImagePtr cv_ptr;
			try
			{
				cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
				inImage2 = cv_ptr->image;
				resultImg2 = cv_ptr->image.clone();

				//detection results will go into "markers"
				markers2.clear();
				//Ok, let's detect
				mDetector2.detect(inImage2, markers2, camParam2, marker_size, false);
				//Detection of the board
				float probDetect=the_board_detector2.detect(markers2, the_board_config, the_board_detected2, camParam2, marker_size);
				if (probDetect > 0.0)
				{
					tf::Transform transform = ar_sys::getTf(the_board_detected2.Rvec, the_board_detected2.Tvec);
					tf::StampedTransform stampedTransform(transform, msg->header.stamp, "cam_pos2", board_frame);

          if (publish_tf)
              br.sendTransform(stampedTransform);

					geometry_msgs::PoseStamped poseMsg;
					tf::poseTFToMsg(transform, poseMsg.pose);
					poseMsg.header.frame_id = msg->header.frame_id;
					poseMsg.header.stamp = msg->header.stamp;
					pose_pub2.publish(poseMsg);

					geometry_msgs::TransformStamped transformMsg;
					tf::transformStampedTFToMsg(stampedTransform, transformMsg);
					transform_pub2.publish(transformMsg);

					geometry_msgs::Vector3Stamped positionMsg;
					positionMsg.header = transformMsg.header;
					positionMsg.vector = transformMsg.transform.translation;
					position_pub2.publish(positionMsg);
				}
				else
				{
					geometry_msgs::PoseStamped poseMsg;
					poseMsg.header.frame_id = msg->header.frame_id;
					poseMsg.header.stamp = msg->header.stamp;
					poseMsg.pose.position.x = 0;
				  poseMsg.pose.position.y = 0;
				  poseMsg.pose.position.z = 0;
				  poseMsg.pose.orientation.x = 0;
				  poseMsg.pose.orientation.y = 0;
				  poseMsg.pose.orientation.z = 0;
				  poseMsg.pose.orientation.w = 0;
					pose_pub2.publish(poseMsg);
				}
				//for each marker, draw info and its boundaries in the image
				for(size_t i=0; draw_markers && i < markers2.size(); ++i)
				{
					markers2[i].draw(resultImg2,cv::Scalar(0,0,255),2);
				}


				if(camParam2.isValid() && marker_size != -1)
				{
					//draw a 3d cube in each marker if there is 3d info
					for(size_t i=0; i<markers2.size(); ++i)
					{
						if (draw_markers_cube) CvDrawingUtils::draw3dCube(resultImg2, markers2[i], camParam2);
						if (draw_markers_axis) CvDrawingUtils::draw3dAxis(resultImg2, markers2[i], camParam2);
					}
					//draw board axis
					if (probDetect > 0.0) CvDrawingUtils::draw3dAxis(resultImg2, the_board_detected2, camParam2);
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
					debug_msg.image = mDetector2.getThresholdedImage();
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
		void cam_info_callback2(const sensor_msgs::CameraInfo &msg)
		{
			camParam2 = ar_sys::getCamParams(msg, useRectifiedImages);
			cam_info_received2 = true;
			cam_info_sub2.shutdown();
		}
};


int main(int argc,char **argv)
{
	ros::init(argc, argv, "ar_single_board");

	ArSysSingleBoard node;

	ros::spin();
}
