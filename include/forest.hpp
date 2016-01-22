#pragma once

#include "data.hpp"
#include "settings.hpp"
#include "tree.hpp"
#include "random.hpp"
#include "Kabsch.hpp"

#include <vector>
#include <random>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <istream>
#include <fstream>

#include <Eigen/Geometry>

#include "opencv2/opencv.hpp"

namespace ISUE {
  namespace RelocForests {
    class Forest {
    public:
      Forest(Data *data, Settings *settings) 
        : data_(data), settings_(settings)
      {
        for (int i = 0; i < settings->num_trees_; ++i)
          forest_.push_back(new Tree());
        random_ = new Random();
      };

      ~Forest()
      {
        delete random_;
        for (auto t : forest_)
          delete t;
        forest_.clear();
      }

      void Serialize(const std::string& path)
      {
        std::ofstream o(path, std::ios_base::binary);
        Serialize(o);
      }

      void Serialize(std::ostream& stream)
      {
        const int majorVersion = 0, minorVersion = 0;

        stream.write(binaryFileHeader_, strlen(binaryFileHeader_));
        stream.write((const char*)(&majorVersion), sizeof(majorVersion));
        stream.write((const char*)(&minorVersion), sizeof(minorVersion));

        int treeCount = settings_->num_trees_;
        stream.write((const char*)(&treeCount), sizeof(treeCount));

        for (auto tree : forest_)
          tree->Serialize(stream);
        
        if (stream.bad())
          throw std::runtime_error("Forest serialization failed");
      }

      // Train time
      /////////////

      std::vector<LabeledPixel> LabelData()
      {
          std::vector<LabeledPixel> labeled_data;

          // randomly choose frames
          for (int i = 0; i < settings_->num_frames_per_tree_; ++i) {
            int curr_frame = random_->Next(0, data_->depth_names_.size());
            // randomly sample pixels per frame
            for (int j = 0; j < settings_->num_pixels_per_frame_; ++j) {
              int row = random_->Next(0, settings_->image_height_);
              int col = random_->Next(0, settings_->image_width_);

              // calc label
              cv::Mat rgb_image = data_->GetRGBImage(curr_frame);
              cv::Mat depth_image = data_->GetDepthImage(curr_frame);

              float Z = (float)depth_image.at<ushort>(row, col) / (float)settings_->depth_factor_;
              float Y = (row - settings_->cy) * Z / settings_->fy;
              float X = (col - settings_->cx) * Z / settings_->fx;

              cv::Point3f label(X, Y, Z);

              // convert label to world coordinates
              CameraInfo pose = data_->poses_.at(curr_frame);
              cv::Mat R = pose.getRotation();
              cv::Mat T = pose.getTranslation();

              cv::Mat P(label, true);
              P.convertTo(P, CV_64FC2); // fix exception when doing operations with R, T
              cv::Mat C = R * P + T;
              label = cv::Point3f(C);

              // store labeled pixel
              LabeledPixel pixel(curr_frame, cv::Point2i(col, row), label);
              labeled_data.push_back(pixel);
            }
          }
          return labeled_data;
      }


      void Train()
      {
        // train each tree individually
        int index = 1;
        for (auto t : forest_) {

          std::cout << "[Tree " << index << "] " << "Generating Training Data.\n";
          std::vector<LabeledPixel> labeled_data = LabelData();

          // train tree with set of pixels recursively
          std::cout << "[Tree " << index << "] " << "Training.\n";

          std::clock_t start;
          double duration;
          start = std::clock();

          t->Train(data_, labeled_data, random_, settings_);
          
          std::ofstream o("swag.tree", std::ios_base::binary);
          t->Serialize(o);

          std::cout << "Wrote tree\n";

          duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
          std::cout << "[Tree " << index << "] "
                    << "Train time: " << duration << " Seconds \n";
          index++;
        }
      }

      // Test time
      ////////////

      std::vector<Eigen::Vector3d> Eval(int row, int col, cv::Mat rgb_image, cv::Mat depth_image)
      {
        std::vector<Eigen::Vector3d> modes;

        for (auto t : forest_) {
          auto m = t->Eval(row, col, rgb_image, depth_image);
          modes.push_back(m);
        }

        return modes;
      }


      uint8_t tophat_error(double val)
      {
        if (val > 0 && val < 0.1)
          return 1;
        else
          return 0;
      }

      class Hypothesis {
      public:
        Eigen::Affine3d pose;
        Eigen::Vector3d camera_space_point;
        uint32_t energy;

        bool operator < (const Hypothesis& h) const
        {
          return (energy < h.energy);
        }
      };

      std::vector<Hypothesis> CreateHypotheses(int K_init, cv::Mat rgb_frame, cv::Mat depth_frame)
      {
        std::vector<Hypothesis> hypotheses;

        // sample initial hypotheses
        for (uint16_t i = 0; i < K_init; ++i) {
          Hypothesis h;
          // 3 points
          Eigen::Matrix3Xd input(3, 3);
          Eigen::Matrix3Xd output(3, 3);

          for (uint16_t j = 0; j < 3; ++j) {
            int col = random_->Next(0, settings_->image_width_);
            int row = random_->Next(0, settings_->image_height_);
            // todo: get camera space points?
            double Z = (float)depth_frame.at<ushort>(row, col) / (float)settings_->depth_factor_;
            double Y = (row - settings_->cy) * Z / settings_->fy;
            double X = (col - settings_->cx) * Z / settings_->fx;

            Eigen::Vector3d tmp_in(X, Y, Z);
            h.camera_space_point = tmp_in;
            // add to input
            input << tmp_in;

            auto modes = Eval(row, col, rgb_frame, depth_frame);
            auto point = modes.at(random_->Next(0, modes.size() - 1));
            // add point to output
            //Eigen::Vector3d tmp_out(point.x, point.y, point.z);
            output << point;

          }
          // kabsch algorithm
          Eigen::Affine3d transform = Find3DAffineTransform(input, output);
          h.pose = transform;
          h.energy = 0;
          hypotheses.push_back(h);
        }

      }

      // Get pose hypothesis from depth and rgb frame
      Eigen::Affine3d Test(cv::Mat rgb_frame, cv::Mat depth_frame)
      {
        int K_init = 1024;
        int K = K_init;

        std::vector<Hypothesis> hypotheses = CreateHypotheses(K_init, rgb_frame, depth_frame);

        while (K > 1) {
          // sample set of B test pixels
          std::vector<cv::Point2i> test_pixels;
          int batch_size = 500; // todo put in settings

          // sample points in test image
          for (int i = 0; i < batch_size; ++i)  {
            int col = random_->Next(0, settings_->image_width_);
            int row = random_->Next(0, settings_->image_height_);
            test_pixels.push_back(cv::Point2i(col, row));
          }

          for (auto p : test_pixels) {
            // evaluate forest to get modes (union)
            auto modes = Eval(p.y, p.x, rgb_frame, depth_frame);

            for (int i = 0; i < K; ++i) {
              double e_min = DBL_MAX;

              for (auto mode : modes) {
                //Eigen::Vector3d mode(m.x, m.y, m.z);
                Eigen::Vector3d e = mode - (hypotheses.at(i).pose * hypotheses.at(i).camera_space_point);
                double e_norm = e.norm();

                if (e_norm < e_min) {
                  e_min = e_norm;
                }
              }
              // update energy
              hypotheses.at(i).energy += tophat_error(e_min); // todo make sure this is right
              if (tophat_error(e_min) == 0) { // inlier
              }
            }
          }

          // sort hypotheses 
          std::sort(hypotheses.begin(), hypotheses.end());
          K = K / 2; // discard half
          // refine hypotheses (kabsch)
        }
        // return best pose and energy        
        return hypotheses.front().pose;
      }

    private:
      Data *data_;
      Settings *settings_;
      std::vector<Tree*> forest_;
      Random *random_;
      const char* binaryFileHeader_ = "ISUE.RelocForests.Forest";
    };
  }

}
