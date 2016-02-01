#pragma once
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

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
        }
        return false;
      }
    private:
      Data *data;
    };
  }
}
