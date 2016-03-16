#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <vector>
#include <map>
#include <cstdint>

namespace ISUE {
  namespace RelocForests {

    /// Labeled data point for training
    class LabeledPixel {
    public:
      LabeledPixel(uint32_t frame, cv::Point2i pos, cv::Point3f label)
        : frame_(frame), pos_(pos), label_(label)
      {}
      uint32_t frame_;
      cv::Point2i pos_;
      cv::Point3f label_;
    };

    /// struct which holds pose for calculating world coordinates
    /// for labeled pixels
    struct Pose {
      Eigen::Matrix3d rotation;
      Eigen::Vector3d position;

      Pose(Eigen::Matrix3d rotation, Eigen::Vector3d position)
        :rotation(rotation), position(position)
      {}
    };

    class Data {
    public:
      Data()
      {
        num_frames_ = 0;
      }

      Data(std::string path_to_data) 
        : data_path_(path_to_data)
      {
        num_frames_ = 0;
      };

      cv::Mat GetRGBImage(uint32_t frame)
      {
        if (rgb_images_.count(frame)) {
          return rgb_images_.at(frame);
        }
        else {
          cv::Mat img = cv::imread(data_path_ + rgb_names_.at(frame));
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
          cv::Mat img = cv::imread(data_path_ + depth_names_.at(frame), CV_LOAD_IMAGE_ANYDEPTH);
          depth_images_.insert(std::pair<uint32_t, cv::Mat>(frame, img));
          return img;
        }
      }

      Pose GetPose(uint32_t frame)
      {
        return poses_.at(frame);
      }

      void AddFrame(cv::Mat rgb_frame, cv::Mat depth_frame, Pose pose)
      {
        rgb_images_.insert(std::pair<uint32_t, cv::Mat>(num_frames_, rgb_frame));
        depth_images_.insert(std::pair<uint32_t, cv::Mat>(num_frames_, depth_frame));
        poses_.push_back(pose);
        num_frames_++;
      }

      void AddFrame(std::string rgb_frame, std::string depth_frame, Pose pose)
      {
        poses_.push_back(pose);
        depth_names_.push_back(depth_frame);
        rgb_names_.push_back(rgb_frame);
        num_frames_++;
      }

      int GetNumFrames()
      {
        return num_frames_;
      }

      /// Write all frames to a folder. Write all poses to poses.txt.
      void Serialize(std::string data_path)
      {
        std::ofstream o(data_path + "poses.txt");
        o << GetNumFrames() << "\n";
        for (int i = 0; i < GetNumFrames(); i++) {
          cv::Mat rgb = GetRGBImage(i);
          cv::Mat depth = GetDepthImage(i);
          cv::Mat depth_w;
          depth.convertTo(depth_w, CV_16U, 5000); // scaled meter values

          cv::imwrite(data_path + std::to_string(i) + "_rgb.png", rgb);
          cv::imwrite(data_path + std::to_string(i) + "_d.png", depth_w);

          Pose p = poses_.at(i);
          Eigen::Quaterniond q(p.rotation);
          o << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " 
            << p.position.x() << " " << p.position.y() << " " << p.position.z() << "\n";
        }
      }

      void Deserialize(std::string data_path)
      {
        data_path_ = data_path;
        float qx, qy, qz, qw, tx, ty, tz;

        std::ifstream in(data_path + "poses.txt");

        std::string num("");
        getline(in, num);
        std::stringstream num_s(num);

        in >> this->num_frames_;

        for (int i = 1; i <= GetNumFrames(); i++) {
          cv::Mat rgb =  cv::imread(data_path + std::to_string(i) + "_rgb.png");
          cv::Mat depth =  cv::imread(data_path + std::to_string(i) + "_d.png");

          std::string temp("");
          getline(in, temp);

          std::stringstream stream(temp);

          stream >> qx; stream >> qy;
          stream >> qz; stream >> qw;
          stream >> tx; stream >> ty; stream >> tz;

          Pose pose(Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix(), Eigen::Vector3d(tx, ty, tz));

          AddFrame(rgb, depth, pose);
        }
      }

      private:
        std::vector<std::string> depth_names_;
        std::vector<std::string> rgb_names_;

        // loaded images
        std::map<uint32_t, cv::Mat> rgb_images_;
        std::map<uint32_t, cv::Mat> depth_images_;
        std::vector<Pose> poses_;

        std::string data_path_;

        uint32_t num_frames_;
    };
  }
}
