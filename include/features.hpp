#pragma once

#include "data.hpp"
#include "random.hpp"
#include "opencv2/opencv.hpp"

namespace ISUE {
  namespace RelocForests {


    class Feature {
    public:
      Feature()
      {
        offset_1_ = cv::Point2i(0,0);
        offset_2_ = cv::Point2i(0, 0);
      }

      Feature(cv::Point2i offset_1, cv::Point2i offset_2)
        : offset_1_(offset_1), offset_2_(offset_2) {};

        virtual float GetResponse(cv::Mat depth_image, cv::Mat rgb_image, cv::Point2i pos, Settings &settings, bool &valid) = 0;

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

      virtual float GetResponse(cv::Mat depth_image, cv::Mat rgb_image, cv::Point2i pos, bool &valid)
      {

        uchar depth_at_pos = depth_image.at<ushort>(pos);
        cv::Point2i depth_inv_1(offset_1_.x / depth_at_pos, offset_2_.y / depth_at_pos);
        cv::Point2i depth_inv_2(offset_2_.x / depth_at_pos, offset_2_.y / depth_at_pos);

        if (depth_at_pos == 0)
          valid = false;

        float D_1 = depth_image.at<uchar>(pos + depth_inv_1);
        float D_2 = depth_image.at<uchar>(pos + depth_inv_2);

        return D_1 - D_2;
      }
      
    };

    /*
     *   Depth Adaptive RGB Feature Response Function
     */
    class DepthAdaptiveRGB : public Feature {
    public:
      DepthAdaptiveRGB()
      {
        color_channel_1_ = 0;
        color_channel_2_ = 0;
        tau_ = 0;
      }

      DepthAdaptiveRGB(cv::Point2i offset_1, cv::Point2i offset_2, int color_channel_1, int color_channel_2, float tau)
        : Feature(offset_1, offset_2), color_channel_1_(color_channel_1), color_channel_2_(color_channel_2), tau_(tau)
      {};

      static DepthAdaptiveRGB CreateRandom(Random *random)
      {
        cv::Point2i offset_1(random->Next(-130, 130), random->Next(-130, 130)); // Value from the paper -- +/- 130 pixel meters
        cv::Point2i offset_2(random->Next(-130, 130), random->Next(-130, 130));
        int color_channel_1 = random->Next(0, 2);
        int color_channel_2 = random->Next(0, 2);
        int tau_ = random->Next(-128, 128);
        return DepthAdaptiveRGB(offset_1, offset_2, color_channel_1, color_channel_2, tau_);
      }

      virtual float GetResponse(cv::Mat depth_img, cv::Mat rgb_img, cv::Point2i pos, Settings &settings, bool &valid) override
      {

        ushort depth_at_pos = depth_img.at<ushort>(pos);
        float depth = (float)depth_at_pos;

        if (depth <= 0) {
          valid = false;
          return 0.0;
        }
        else {
          depth /= settings.depth_factor_; // scale value
        }

        // depth invariance
        cv::Point2i depth_inv_1(offset_1_.x / depth, offset_2_.y / depth);
        cv::Point2i depth_inv_2(offset_2_.x / depth, offset_2_.y / depth);

        cv::Point2i pos1 = pos + depth_inv_1;
        cv::Point2i pos2 = pos + depth_inv_2;

        int width = settings.image_width_;
        int height = settings.image_width_;
        // check bounds
        if (pos1.x >= width || pos1.y >= height ||
            pos1.x < 0.0    || pos1.y < 0.0 ) {
          valid = false;
          return 0.0f;
        }
        if (pos2.x >= width || pos2.y >= height ||
            pos2.x < 0.0    || pos2.y < 0.0) {
          valid = false;
          return 0.0f;
        }

        float I_1 = rgb_img.at<cv::Vec3b>(pos1)[this->color_channel_1_];
        float I_2 = rgb_img.at<cv::Vec3b>(pos2)[this->color_channel_2_];

        return I_1 - I_2;
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