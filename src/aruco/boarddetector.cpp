/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/
#include <aruco/boarddetector.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <fstream>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;
using namespace cv;
namespace aruco {
    /**
    */
    BoardDetector::BoardDetector ( bool  setYPerpendicular ) {
        _setYPerpendicular=setYPerpendicular;
        _areParamsSet=false;
        repj_err_thres=-1;
    }
    /**
       * Use if you plan to let this class to perform marker detection too
       */
    void BoardDetector::setParams ( const BoardConfiguration &bc,const CameraParameters &cp, float markerSizeMeters ) {
        _camParams=cp;
        _markerSize=markerSizeMeters;
        _bconf=bc;
        _areParamsSet=true;
    }
    /**
    *
    *
    */
    void BoardDetector::setParams ( const BoardConfiguration &bc ) {
        _bconf=bc;
        _areParamsSet=true;
    }

    /**
    *
    *
    */
    float  BoardDetector::detect ( const cv::Mat &im ) throw ( cv::Exception ) {
        _mdetector.detect ( im,_vmarkers );

        float res;

        if ( _camParams.isValid() )
            res=detect ( _vmarkers,_bconf,_boardDetected,_camParams.CameraMatrix,_camParams.Distorsion,_markerSize );
        else res=detect ( _vmarkers,_bconf,_boardDetected );
        return res;
    }
    /**
    *
    *
    */
    float BoardDetector::detect ( const vector<Marker> &detectedMarkers,const  BoardConfiguration &BConf, Board &Bdetected,const CameraParameters &cp, float markerSizeMeters ) throw ( cv::Exception )
    {
        return detect ( detectedMarkers, BConf,Bdetected,cp.CameraMatrix,cp.Distorsion,markerSizeMeters );
    }
    /**
    *
    *
    */
    float BoardDetector::detect ( const vector<Marker> &detectedMarkers,const  BoardConfiguration &BConf, Board &Bdetected, Mat camMatrix,Mat distCoeff,float markerSizeMeters ) throw ( cv::Exception ) {
        if ( BConf.size() ==0 )
          throw cv::Exception ( 8881,"BoardDetector::detect","Invalid BoardConfig that is empty",__FILE__,__LINE__ );
        if ( BConf[0].size() <2 )
          throw cv::Exception ( 8881,"BoardDetector::detect","Invalid BoardConfig that is empty 2",__FILE__,__LINE__ );
        //compute the size of the markers in meters, which is used for some routines(mostly drawing)
        float ssize;
        if ( BConf.mInfoType==BoardConfiguration::PIX && markerSizeMeters>0 )
          ssize=markerSizeMeters;
        else if ( BConf.mInfoType==BoardConfiguration::METERS )
        {
            ssize=cv::norm ( BConf[0][0]-BConf[0][1] );
        }

        Bdetected.clear();
        ///find among detected markers these that belong to the board configuration
        for ( unsigned int i=0; i<detectedMarkers.size(); i++ )
        {
            int idx=BConf.getIndexOfMarkerId ( detectedMarkers[i].id );
            if ( idx!=-1 )
            {
                Bdetected.push_back ( detectedMarkers[i] );
                Bdetected.back().ssize=ssize;
            }
        }
        //copy configuration
        Bdetected.conf=BConf;

        bool hasEnoughInfoForRTvecCalculation=false;
        if ( Bdetected.size() >=1 )
        {
            if ( camMatrix.rows!=0 )
            {
                if ( markerSizeMeters>0 && BConf.mInfoType==BoardConfiguration::PIX )
                  hasEnoughInfoForRTvecCalculation=true;
                else if ( BConf.mInfoType==BoardConfiguration::METERS )
                  hasEnoughInfoForRTvecCalculation=true;
            }
        }

        //calculate extrinsic if there is information for that
        if ( hasEnoughInfoForRTvecCalculation )
        {
            //calculate the size of the markers in meters if expressed in pixels
            double marker_meter_per_pix=0;
            if ( BConf.mInfoType==BoardConfiguration::PIX )
              marker_meter_per_pix=markerSizeMeters /  cv::norm ( BConf[0][0]-BConf[0][1] );
            else marker_meter_per_pix=1;//to avoind interferring the process below

            // now, create the matrices for finding the extrinsics
            vector<cv::Point3f> objPoints;
            vector<cv::Point2f> imagePoints;
            for ( size_t i=0; i<Bdetected.size(); i++ )
            {
                int idx=Bdetected.conf.getIndexOfMarkerId ( Bdetected[i].id );
                assert ( idx!=-1 );
                for ( int p=0; p<4; p++ )
                {
                    imagePoints.push_back ( Bdetected[i][p] );
                    const aruco::MarkerInfo &Minfo=Bdetected.conf.getMarkerInfo ( Bdetected[i].id );
                    objPoints.push_back ( Minfo[p]*marker_meter_per_pix );
                }
            }
            if ( distCoeff.total() ==0 )
              distCoeff = cv::Mat::zeros ( 1,4,CV_32FC1 );

            cv::Mat rvec,tvec;
            cv::solvePnP ( objPoints,imagePoints,camMatrix,distCoeff,rvec,tvec );
            rvec.convertTo ( Bdetected.Rvec,CV_32FC1 );
            tvec.convertTo ( Bdetected.Tvec,CV_32FC1 );

            double N = 2*objPoints.size();
            cv::Mat PixelError = cv::Mat::zeros(N,N,CV_64FC1);
            for (int c=0; c < N; c++)
              PixelError.at<double>(c,c)=2.0;

            cv::Mat J;
            cv::Mat JImageToTransRodr(N,6,CV_64FC1 );
            vector<cv::Point2f> reprojected;
            cv::projectPoints ( objPoints,rvec,tvec,camMatrix,distCoeff,reprojected,J);
            JImageToTransRodr = cv::Mat(J, cv::Rect(0,0,6,N));

            cv::Mat JRodrToRotMat(3,9,CV_64FC1 );
            cv::Mat R( 3,3,CV_64FC1 );
            cv::Rodrigues ( rvec, R, JRodrToRotMat );

            cv::Mat JRotMatToEuler =  cv::Mat::zeros( 9,3,CV_64FC1 );
            double phi   = std::atan2(R.at<double>(2,1),R.at<double>(2,2));
            double theta = std::atan2(R.at<double>(2,0),sqrt(R.at<double>(2,1)*R.at<double>(2,1) + R.at<double>(2,2)*R.at<double>(2,2)));
            double psi   = std::atan2(R.at<double>(1,0),R.at<double>(0,0));

            double cphi   = std::cos(phi);
            double ctheta = std::cos(theta);
            double cpsi   = std::cos(psi);
            double sphi   = std::sin(phi);
            double stheta = std::sin(theta);
            double spsi   = std::sin(psi);
            JRotMatToEuler.at<double>(0,0)= 0;
            JRotMatToEuler.at<double>(0,1)= -cpsi*stheta;
            JRotMatToEuler.at<double>(0,2)= -spsi*ctheta;

            JRotMatToEuler.at<double>(1,0)=  cpsi*stheta*cphi + spsi*sphi;
            JRotMatToEuler.at<double>(1,1)=  cpsi*ctheta*sphi;
            JRotMatToEuler.at<double>(1,2)= -spsi*stheta*sphi - cpsi*cphi;

            JRotMatToEuler.at<double>(2,0)= -cpsi*stheta*sphi + spsi*cphi;
            JRotMatToEuler.at<double>(2,1)=  cpsi*ctheta*cphi;
            JRotMatToEuler.at<double>(2,2)= -spsi*stheta*cphi + cpsi*sphi;

            JRotMatToEuler.at<double>(3,0)= 0;
            JRotMatToEuler.at<double>(3,1)= -spsi*stheta;
            JRotMatToEuler.at<double>(3,2)=  cpsi*ctheta;

            JRotMatToEuler.at<double>(4,0)=  spsi*stheta*cphi - cpsi*sphi;
            JRotMatToEuler.at<double>(4,1)=  spsi*ctheta*sphi;
            JRotMatToEuler.at<double>(4,2)=  cpsi*stheta*sphi - spsi*cphi;

            JRotMatToEuler.at<double>(5,0)= -spsi*stheta*sphi - cpsi*cphi;
            JRotMatToEuler.at<double>(5,1)=  spsi*ctheta*cphi;
            JRotMatToEuler.at<double>(5,2)=  cpsi*stheta*cphi + spsi*sphi;

            JRotMatToEuler.at<double>(6,0)= 0;
            JRotMatToEuler.at<double>(6,1)= -ctheta;
            JRotMatToEuler.at<double>(6,2)= 0;

            JRotMatToEuler.at<double>(7,0)= ctheta*cphi;
            JRotMatToEuler.at<double>(7,1)= -stheta*sphi;
            JRotMatToEuler.at<double>(7,2)= 0;

            JRotMatToEuler.at<double>(8,0)= -ctheta*sphi;
            JRotMatToEuler.at<double>(8,1)= -stheta*cphi;
            JRotMatToEuler.at<double>(8,2)= 0;

            cv::Mat JRodrToEuler = JRodrToRotMat*JRotMatToEuler;
            cv::Mat JRodrToEulerAndIdenty = cv::Mat::zeros(6,6,CV_64FC1 );
            for (int i=0; i < 3; i++)
              for (int j=0; j < 3; j++)
                JRodrToEulerAndIdenty.at<double>(i,j)=JRodrToEuler.at<double>(i,j);
            for (int i=3; i < 6; i++)
              JRodrToEulerAndIdenty.at<double>(i,i)=1.0;

            cv::Mat JImageToTransEuler(N,6,CV_64FC1 );
            JImageToTransEuler = JImageToTransRodr*JRodrToEulerAndIdenty;

            cv::Mat sigmaTransEuler = cv::Mat(JImageToTransEuler.t() *PixelError.inv()* JImageToTransEuler).inv();
            sigmaTransEuler.convertTo ( Bdetected.Cov,CV_32FC1 );

            /*cv::Mat finalSigma = cv::Mat::zeros( 6,6,CV_64FC1 );
            finalSigma.at<double>(0,0)=sigmaTransEuler.at<double>(3,3);
            finalSigma.at<double>(1,1)=sigmaTransEuler.at<double>(4,4);
            finalSigma.at<double>(2,2)=sigmaTransEuler.at<double>(5,5);
            finalSigma.at<double>(3,3)=sigmaTransEuler.at<double>(0,0);
            finalSigma.at<double>(4,4)=sigmaTransEuler.at<double>(1,1);
            finalSigma.at<double>(5,5)=sigmaTransEuler.at<double>(2,2);

            finalSigma.convertTo ( Bdetected.Cov,CV_32FC1 );*/
            //std::cout << "Covariance "<< sigmaTransEuler.diag() << std::endl;

            double errSum=0;
            //check now the reprojection error and
            for ( size_t i=0; i<reprojected.size(); i++ )
                errSum+=cv::norm ( reprojected[i]-imagePoints[i] );

            //now, do a refinement and remove points whose reprojection error is above a threshold, then repeat calculation with the rest
            if ( repj_err_thres>0 )
            {
                vector<int> pointsThatPassTest;//indices
                //check now the reprojection error and
                for ( size_t i=0; i<reprojected.size(); i++ )
                {
                    float err=cv::norm ( reprojected[i]-imagePoints[i] );
                    if ( err<repj_err_thres )
                      pointsThatPassTest.push_back ( i );
                }

                cout<<"Number of points after reprjection test "<<pointsThatPassTest.size() <<"/"<<objPoints.size() <<endl;
                //copy these data to another vectors and repeat
                vector<cv::Point3f> objPoints_filtered;
                vector<cv::Point2f> imagePoints_filtered;
                for ( size_t i=0; i<pointsThatPassTest.size(); i++ )
                {
                    objPoints_filtered.push_back ( objPoints[pointsThatPassTest[i] ] );
                    imagePoints_filtered.push_back ( imagePoints[pointsThatPassTest[i] ] );
                }

                PixelError = cv::Mat::zeros(2*objPoints_filtered.size(),2*objPoints_filtered.size(),CV_64FC1);
                for (int c=0; c < 2*objPoints_filtered.size(); c++)
                  PixelError.at<double>(c,c)=2.0;

                cv::solvePnP ( objPoints_filtered,imagePoints_filtered,camMatrix,distCoeff,rvec,tvec );
                rvec.convertTo ( Bdetected.Rvec,CV_32FC1 );
                tvec.convertTo ( Bdetected.Tvec,CV_32FC1 );
                cv::projectPoints ( objPoints_filtered,rvec,tvec,camMatrix,distCoeff,reprojected, J );
                cv::Mat sigma = cv::Mat(J.t() *PixelError.inv()* J, cv::Rect(0,0,6,6)).inv();
                sigma.convertTo ( Bdetected.Cov,CV_32FC1 );
            }

            //now, rotate 90 deg in X so that Y axis points up
            if ( _setYPerpendicular )
            {
                std::cout << "_setYPerpendicular true" << std::endl;
                rotateXAxis ( Bdetected.Rvec );
            }
        }

        float prob=float ( Bdetected.size() ) /double ( Bdetected.conf.size() );
        return prob;
    }

    void BoardDetector::rotateXAxis ( Mat &rotation )
    {
        cv::Mat R ( 3,3,CV_32FC1 );
        Rodrigues ( rotation, R );
        //create a rotation matrix for x axis
        cv::Mat RX=cv::Mat::eye ( 3,3,CV_32FC1 );
        float angleRad=-M_PI/2;
        RX.at<float> ( 1,1 ) =cos ( angleRad );
        RX.at<float> ( 1,2 ) =-sin ( angleRad );
        RX.at<float> ( 2,1 ) =sin ( angleRad );
        RX.at<float> ( 2,2 ) =cos ( angleRad );
        //now multiply
        R=R*RX;
        //finally, the the rodrigues back
        Rodrigues ( R,rotation );

    }

    /**Static version (all in one)
     */
    Board BoardDetector::detect ( const cv::Mat &Image, const BoardConfiguration &bc,const CameraParameters &cp, float markerSizeMeters ) {
        BoardDetector BD;
        BD.setParams ( bc,cp,markerSizeMeters );
        BD.detect ( Image );
        return BD.getDetectedBoard();
    }
};
