#pragma once

#include "data.hpp"
#include "opencv2/opencv.hpp"

namespace ISUE {
  namespace RelocForests {
    class Feature {
    public:
      Feature(cv::Point2d offset_1, cv::Point2d offset_2)
        : offset_1_(offset_1), offset_2_(offset_2) {};
      virtual float GetResponse(Data* data, int frame, cv::Point2d pos) = 0;
    protected:
      cv::Point2d offset_1_;
      cv::Point2d offset_2_;
    };

    /*
     *   Depth Feature Response Function
     */
    class Depth : public Feature {
    public:
      Depth(cv::Point2d offset_1, cv::Point2d offset_2)
        : Feature(offset_1, offset_2)
      {};

      virtual float GetResponse(Data* data, int frame, cv::Point2d pos)
      {
        return 0.0f;
      }
      
    };

    /*
     *   Depth Adaptive RGB Feature Response Function
     */
    class DepthAdaptiveRGB : public Feature {
    public:
      DepthAdaptiveRGB(cv::Point2d offset_1, cv::Point2d offset_2, int color_channel_1, int color_channel_2)
        : Feature(offset_1, offset_2), color_channel_1_(color_channel_1), color_channel_2_(color_channel_2)
      {};

      virtual float GetResponse(Data* data, int frame, cv::Point2d pos)
      {
        return 0.0f;
      }
    protected:
      int color_channel_1_, color_channel_2_;
    };

  }
}