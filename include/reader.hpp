#pragma once
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

#include "camerautils.hpp"
#include "data.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>


namespace ISUE {
  namespace RelocForests {
    class Reader {
    public:
      Reader() {};
      Reader(std::string path_to_assoc_file)
      {};

      ~Reader()
      {
        delete data;
      }

      Data* GetData()
      {
        return data;
      }

      // Loads data from association file
      // returns true if error
      bool Load(std::string path_to_data)
      {
        data = new Data();
        data->path_to_assoc_file_ = path_to_data;
        std::string path_to_assoc_file_ = path_to_data;
        CameraInfo startpos;

        //std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d> > poses_from_assocfile;

        std::fstream associationFile;
        float junkstamp;
        std::string depthName;
        std::string rgbName;
        float qx, qy, qz, qw, tx, ty, tz;

        std::cout << "[Reader] Opening: " << path_to_assoc_file_ + "associate.txt" << "\n";

        associationFile.open(path_to_assoc_file_ + "associate.txt", std::ios::in);

        if (!associationFile.is_open()) {
          std::cout << "Could not open association file.\n";
          return true;
        }
        else {
          while (!associationFile.eof()) {
            std::string temp("");
            getline(associationFile, temp);

            std::stringstream stream(temp);
            stream >> junkstamp; // throwaway timestamps

            // get pose
            stream >> tx; stream >> ty; stream >> tz;
            stream >> qx; stream >> qy;
            stream >> qz; stream >> qw;

            // rgb and depth
            stream >> junkstamp;
            stream >> depthName;

            stream >> junkstamp;
            stream >> rgbName;

            data->poses_eigen_.push_back(std::pair<Eigen::Matrix3d, Eigen::Vector3d>(Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix(), Eigen::Vector3d(tx, ty, tz)));

            data->depth_names_.push_back(depthName);
            data->rgb_names_.push_back(rgbName);
          }

          std::cout << "[Reader] Read "
            << data->depth_names_.size()
            << " frames from association file: \n"
            << "\t" << path_to_assoc_file_ + "associate.txt" << ".\n";

          // intrinsics for kinect
          // todo: read from file but in general datasets are from kinect
          float fx = 525.0f;
          float fy = 525.0f;
          float cx = 319.5f;
          float cy = 239.5f;

          for (unsigned int i = 0; i < data->poses_eigen_.size(); i++) {
            data->poses_.push_back(CameraInfo::kinectPoseFromEigen(data->poses_eigen_[i], fx, fy, cx, cy));
            data->poses_.back().setExtrinsic(startpos.getExtrinsic()*data->poses_.back().getExtrinsic());
            /*
            depth_names_[i] = path_to_assoc_file_ + depth_names_[i];
            if (useColor) result->rgb_names_[i] = prefix + result->rgb_names_[i];
            */
          }
        }
        return false;
      }
    private:
      Data *data;
    };
  }
}
