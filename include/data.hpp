#pragma once
#include "camerautils.hpp"

#include "opencv2/opencv.hpp"

#include <vector>
#include <map>
#include <cstdint>

namespace ISUE {
  namespace RelocForests {
    class LabeledPixel {
    public:
      LabeledPixel(uint32_t frame, cv::Point2i pos, cv::Point3f label)
        : frame_(frame), pos_(pos), label_(label)
      {}
      uint32_t frame_;
      cv::Point2i pos_;
      cv::Point3f label_;
    };

    class Data {
    public:
      Data() {};

      cv::Mat GetRGBImage(uint32_t frame)
      {
        if (rgb_images_.count(frame)) {
          return rgb_images_.at(frame);
        }
        else {
          cv::Mat img = cv::imread(path_to_assoc_file_ + rgb_names_.at(frame));
          rgb_images_.insert(std::pair<uint32_t, cv::Mat>(frame, img));
          return img;
        }
      }

      cv::Mat GetDepthImage(uint32_t frame)
      {
        if (depth_images_.count(frame)) {
          return depth_images_.at(frame);
        }
        else {
          cv::Mat img = cv::imread(path_to_assoc_file_ + depth_names_.at(frame), CV_LOAD_IMAGE_ANYDEPTH);
          depth_images_.insert(std::pair<uint32_t, cv::Mat>(frame, img));
          return img;
        }
      }

      std::string path_to_assoc_file_;

      std::vector<std::string> depth_names_;
      std::vector<std::string> rgb_names_;
      std::vector<CameraInfo> poses_;

      // loaded images
      std::map<uint32_t, cv::Mat> rgb_images_;
      std::map<uint32_t, cv::Mat> depth_images_;

    };
  }
}
