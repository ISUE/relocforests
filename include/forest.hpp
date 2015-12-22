#pragma once

#include "data.hpp"
#include "settings.hpp"
#include "tree.hpp"
#include "random.hpp"

#include <vector>
#include <random>

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
        delete data_;
        delete settings_;
        delete random_;
        for (auto t : forest_)
          delete t;
        forest_.clear();
      }

      void Train()
      {
        // train each tree individually 
        for (auto t : forest_) {
          std::vector<LabeledPixel> labeled_data;
          // randomly choose frames
          for (int i = 0; i < settings_->num_frames_per_tree_; ++i) {
            int curr_frame = random_->Next(0, data_->depth_names_.size());
            // randomly sample pixels per frame
            for (int j = 0; j < settings_->num_pixels_per_frame_; ++j) {
              int x = random_->Next(0, settings_->image_width_);
              int y = random_->Next(0, settings_->image_height_);
              cv::Point2i();

              // calc label
              cv::Mat rgb_image = data_->GetRGBImage(curr_frame);
              cv::Mat depth_image = data_->GetDepthImage(curr_frame);

              float Z = depth_image.at<uchar>(x, y) / settings_->depth_factor_;
              float Y = (y - settings_->cy) * Z / settings_->fy;
              float X = (x - settings_->cx) * Z / settings_->fx;

              cv::Point3f label(X, Y, Z);
              // convert to world coordinates

              // store labeled pixel
              LabeledPixel pixel(curr_frame, cv::Point2i(x, y), label);
              labeled_data.push_back(pixel);
            }

          }
          // train tree with set of pixel recursively
          t->Train(labeled_data, random_, settings_->image_width_, settings_->image_height_);
        }
      }

    private:
      Data *data_;
      Settings *settings_;
      std::vector<Tree*> forest_;
      Random *random_;
    };
  }

}