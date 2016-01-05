#pragma once

#include "data.hpp"
#include "settings.hpp"
#include "tree.hpp"
#include "random.hpp"

#include <vector>
#include <random>
#include <ctime>

#include "opencv2/opencv.hpp"

namespace ISUE {
  namespace RelocForests {
    class Forest {
    public:
      Forest(Data *data, Settings settings) 
        : data_(data), settings_(settings)
      {
        for (int i = 0; i < settings.num_trees_; ++i)
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
          for (int i = 0; i < settings_.num_frames_per_tree_; ++i) {
            int curr_frame = random_->Next(0, data_->depth_names_.size());
            // randomly sample pixels per frame
            for (int j = 0; j < settings_.num_pixels_per_frame_; ++j) {
              int row = random_->Next(0, settings_.image_height_); // row
              int col = random_->Next(0, settings_.image_width_);  // col

              // calc label
              cv::Mat rgb_image = data_->GetRGBImage(curr_frame);
              cv::Mat depth_image = data_->GetDepthImage(curr_frame);

              float Z = (float)depth_image.at<ushort>(row, col) / (float)settings_.depth_factor_;
              float Y = (row - settings_.cy) * Z / settings_.fy;
              float X = (col - settings_.cx) * Z / settings_.fx;

              cv::Point3f label(X, Y, Z);
              // todo: convert to world coordinates

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

          t->Train(data_, labeled_data, random_, &settings_);

          duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
          std::cout << "[Tree " << index << "] " << "Train time: " << duration << " Seconds \n";
          
          index++;
        }
      }

    private:
      Data *data_;
      Settings settings_;
      std::vector<Tree*> forest_;
      Random *random_;
    };
  }

}