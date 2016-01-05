#pragma once

#include "data.hpp"
#include "random.hpp"
#include "opencv2/opencv.hpp"

namespace ISUE {
  namespace RelocForests {

    enum OUT { LEFT, RIGHT, TRASH };

    class Feature {
    public:
      Feature(cv::Point2i offset_1, cv::Point2i offset_2)
        : offset_1_(offset_1), offset_2_(offset_2) {};
      virtual std::pair<float, bool> GetResponse(Data* data, LabeledPixel pixel) = 0;
    protected:
      cv::Point2i offset_1_;
      cv::Point2i offset_2_;
    };

    /*
     *   Depth Feature Response Function
     */
    class Depth : public Feature {
    public:
      Depth(cv::Point2i offset_1, cv::Point2i offset_2)
        : Feature(offset_1, offset_2)
      {};

      virtual std::pair<float, bool> GetResponse(Data* data, LabeledPixel pixel) override
      {
        cv::Mat depth_img = data->GetDepthImage(pixel.frame_);

        uchar depth_at_pos = depth_img.at<ushort>(pixel.pos_);
        cv::Point2i depth_inv_1(offset_1_.x / depth_at_pos, offset_2_.y / depth_at_pos);
        cv::Point2i depth_inv_2(offset_2_.x / depth_at_pos, offset_2_.y / depth_at_pos);

        float D_1 = depth_img.at<uchar>(pixel.pos_ + depth_inv_1);
        float D_2 = depth_img.at<uchar>(pixel.pos_ + depth_inv_2);

        return std::pair<float, bool>(D_1 - D_2, true);
      }
      
    };

    /*
     *   Depth Adaptive RGB Feature Response Function
     */
    class DepthAdaptiveRGB : public Feature {
    public:
      DepthAdaptiveRGB(cv::Point2i offset_1, cv::Point2i offset_2, int color_channel_1, int color_channel_2, float tau)
        : Feature(offset_1, offset_2), color_channel_1_(color_channel_1), color_channel_2_(color_channel_2), tau_(tau)
      {};

      static DepthAdaptiveRGB* CreateRandom(Random *random, int image_width, int image_height)
      {
        cv::Point2i offset_1(random->Next(-130, 130), random->Next(-130, 130)); // I believe this is correct
        cv::Point2i offset_2(random->Next(-130, 130), random->Next(-130, 130));
        int color_channel_1 = random->Next(0, 2);
        int color_channel_2 = random->Next(0, 2);
        int tau_ = random->Next(0, 12); // todo figure out a good value
        return new DepthAdaptiveRGB(offset_1, offset_2, color_channel_1, color_channel_2, tau_);
      }

      // todo settings
      // returns 
      virtual std::pair<float, bool> GetResponse(Data* data, LabeledPixel pixel) override
      {
        cv::Mat depth_img = data->GetDepthImage(pixel.frame_);
        cv::Mat rgb_img = data->GetRGBImage(pixel.frame_);

        ushort depth_at_pos = depth_img.at<ushort>(pixel.pos_);
        float depth = (float)depth_at_pos;

        if (depth <= 0) {
          depth = 6.0f;
          // throw away point
          return std::pair<float, bool>(0.0f, false);
        }
        else {
          depth /= 5000.0; // scale value
        }

        // depth invariance
        cv::Point2i depth_inv_1(offset_1_.x / depth, offset_2_.y / depth);
        cv::Point2i depth_inv_2(offset_2_.x / depth, offset_2_.y / depth);

        cv::Point2i pos1 = pixel.pos_ + depth_inv_1;
        cv::Point2i pos2 = pixel.pos_ + depth_inv_2;

        // check bounds
        if (pos1.x >= 640.0 ||
            pos1.y >= 480.0 ||
            pos1.x < 0.0 ||
            pos1.y < 0.0 ) {
          return std::pair<float, bool>(0.0f, false);
          // throw away
        }
        if (pos2.x >= 640.0 ||
          pos2.y >= 480.0 ||
          pos2.x < 0.0 ||
          pos2.y < 0.0) {
          return std::pair<float, bool>(0.0f, false);
          // throw away
        }

        float I_1 = rgb_img.at<cv::Vec3b>(pixel.pos_ + depth_inv_1)[this->color_channel_1_];
        float I_2 = rgb_img.at<cv::Vec3b>(pixel.pos_ + depth_inv_2)[this->color_channel_2_];

        return std::pair<float, bool>(I_1 - I_2, true);
      }

      float GetThreshold()
      {
        return tau_;
      }

    protected:
      int color_channel_1_, color_channel_2_;
      float tau_;
    };

  }
}