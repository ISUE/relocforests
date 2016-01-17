#pragma once

#include "data.hpp"
#include "settings.hpp"
#include "tree.hpp"
#include "random.hpp"

#include <vector>
#include <random>
#include <ctime>
#include <cstdint>

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

      void Train()
      {
        // train each tree individually
        int index = 1;
        for (auto t : forest_) {
          std::vector<LabeledPixel> labeled_data;
          std::cout << "[Tree " << index << "] " << "Generating Training Data.\n";

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

          // train tree with set of pixels recursively
          std::cout << "[Tree " << index << "] " << "Training.\n";

          std::clock_t start;
          double duration;
          start = std::clock();

          t->Train(data_, labeled_data, random_, settings_);

          duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
          std::cout << "[Tree " << index << "] "
                    << "Train time: " << duration << " Seconds \n";
          index++;
        }
      }

      std::vector<cv::Point3d> Eval(int row, int col, cv::Mat rgb_image, cv::Mat depth_image)
      {
        std::vector<cv::Point3d> modes;

        for (auto t : forest_) {
          auto m = t->Eval(row, col, rgb_image, depth_image);
          modes.insert(modes.end(), m.begin(), m.end());
        }

        return modes;
      }



      CameraInfo Test(cv::Mat rgb_frame, cv::Mat depth_frame)
      {
        int K_init = 1024;
        int K = K_init;

        std::vector<CameraInfo> hypotheses;

        // sample initial hypotheses
        for (uint16_t i = 0; i < K_init; ++i) {
          // 3 points
          std::vector<cv::Point3d> points;
          for (uint16_t j = 0; j < 3; ++j) {
            int col = random_->Next(0, settings_->image_width_);
            int row = random_->Next(0, settings_->image_height_);
            auto modes = Eval(row, col, rgb_frame, depth_frame);
            points.push_back(modes.at(random_->Next(0, modes.size() - 1)));
          }
          // kabsch
        }

        // init energies
        std::vector<uint32_t> energies (K, 0);

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
              // update energy
              // E_k <- E_k + e_i(H_K)
            }
          }

          // sort hypotheses 

          K = K / 2;

          // refine hypotheses 

        }

        // return best pose and energy        

      }

    private:
      Data *data_;
      Settings *settings_;
      std::vector<Tree*> forest_;
      Random *random_;
    };
  }

}
