#pragma once

#include "node.hpp"
#include "random.hpp"
#include "settings.hpp"
#include "data.hpp"
#include "MeanShift.hpp"

#include <vector>
#include <cstdint>
#include <ctime>

#include <unordered_map>

namespace ISUE {
  namespace RelocForests {
    class Point3D {
    public:
      Point3D(double x, double y, double z) : x(x), y(y), z(z) {};
      double x, y, z;
    };

    struct hashFunc{
      size_t operator()(const Point3D &k) const {
        size_t h1 = std::hash<double>()(k.x);
        size_t h2 = std::hash<double>()(k.y);
        size_t h3 = std::hash<double>()(k.z);
        return (h1 ^ (h2 << 1)) ^ h3;
      }
    };

    struct equalsFunc {
      bool operator()(const Point3D &l, const Point3D &r) const{
        return (l.x == r.x) && (l.y == r.y) && (l.z == r.z);
      }
    };

    typedef std::unordered_map<Point3D, uint32_t, hashFunc, equalsFunc> Point3DMap;


    class Tree {
    public:
      Tree()
      {
        root = new Node();
      };

      ~Tree()
      {
        delete root;
      };

      void Serialize(std::ostream& stream) const
      {
        const int majorVersion = 0, minorVersion = 0;
        
        stream.write(binaryFileHeader_, strlen(binaryFileHeader_));
        stream.write((const char*)(&majorVersion), sizeof(majorVersion));
        stream.write((const char*)(&minorVersion), sizeof(minorVersion));

        stream.write((const char*)(&settings->max_tree_depth_), sizeof(settings->max_tree_depth_));

        stream.write((const char*)(this), sizeof(this));
      }


      // learner output
      enum OUT { LEFT, RIGHT, TRASH };

      //  Evaluates weak learner. Decides whether the point should go left or right.
      //  Returns OUT enum value.
      OUT eval_learner(Data *data, LabeledPixel pixel, DepthAdaptiveRGB *feature) 
      {
        auto response = feature->GetResponse(data, pixel);
        //auto response1 = feature->GetResponse(data->GetDepthImage(pixel.frame_), data->GetDepthImage(pixel.frame_), pixel);

        bool is_valid_point = response.second;
        if (!is_valid_point) // no depth or out of bounds
          return OUT::TRASH;

        return (OUT)(response.first >= feature->GetThreshold());
      }

      // V(S)
      double variance(std::vector<LabeledPixel> labeled_data)
      {
        if (labeled_data.size() == 0)
          return 0.0;
        double V = (1.0f / (double)labeled_data.size());
        double sum = 0.0f;

        // calculate mean of S
        cv::Point3f tmp;
        for (auto p : labeled_data)
          tmp += p.label_;
        uint32_t size = labeled_data.size();
        cv::Point3f mean(tmp.x / size, tmp.y / size, tmp.z / size);

        for (auto p : labeled_data) {
          cv::Point3f val = (p.label_ - mean);
          sum += val.x * val.x  + val.y * val.y + val.z * val.z;
        }

        return V * sum;
      }


      // Q(S_n, \theta)
      double objective_function(std::vector<LabeledPixel> data, std::vector<LabeledPixel> left, std::vector<LabeledPixel> right)
      {
        double var = variance(data);
        double sum;

        // left
        double left_var = variance(left);
        double left_val = ((double)left.size() / (double)data.size()) * left_var;
        // right
        double right_var = variance(right);
        double right_val = ((double)right.size() / (double)data.size()) * right_var;

        sum = left_val + right_val;
        return var - sum;
      }

      // Returns height from current node to root node.
      uint32_t traverse_to_root(Node *node) {
        if (node == nullptr)
          return 0;
        return 1 + traverse_to_root(node->parent_);
      }


      void train_recurse(Node *node, std::vector<LabeledPixel> S) 
      {
        uint16_t height = traverse_to_root(node);
        if (S.size() == 1 || height >= settings->max_tree_depth_) {

          std::vector<Eigen::Vector3d> data;

          // calc mode for leaf, sub-sample N_SS = 500
          for (uint16_t i = 0; i < (S.size() < 500 ? S.size() : 500); i++) {
            auto p = S.at(i);
            Eigen::Vector3d point { p.label_.x, p.label_.y, p.label_.z };
            data.push_back(point);
          }

          // cluster
          MeanShift *ms = new MeanShift(nullptr);
          double kernel_bandwidth = 0.01f; // gaussian
          std::vector<Eigen::Vector3d> cluster = ms->cluster(data, kernel_bandwidth);

          // find mode
          std::vector<Point3D> clustered_points;
          for (auto c : cluster)
            clustered_points.push_back(Point3D(floor(c[0] * 10000) / 10000, floor(c[1] * 10000) / 10000, floor(c[2] * 10000) / 10000));

          Point3DMap cluster_map;

          for (auto p : clustered_points)
            cluster_map[p]++;

          std::vector<cv::Point3d> modes;
          for (auto p : cluster_map)
            modes.push_back(cv::Point3d(p.first.x, p.first.y, p.first.z));

          node->modes_ = modes;
          node->is_leaf_ = true;
          return;
        }
        else if (S.size() == 0) {
          delete node;
          node = nullptr;
          return;
        }

        node->is_split_ = true;
        node->is_leaf_ = false;

        uint32_t num_candidates = 5,
                feature = 0;
        double minimum_objective = DBL_MAX;

        std::vector<DepthAdaptiveRGB*> candidate_params;
        std::vector<LabeledPixel> left_final, right_final;


        for (uint32_t i = 0; i < num_candidates; ++i) {

          // add candidate
          candidate_params.push_back(DepthAdaptiveRGB::CreateRandom(random, settings->image_width_, settings->image_height_));

          // partition data with candidate
          std::vector<LabeledPixel> left_data, right_data;

          for (uint32_t j = 0; j < S.size(); ++j) {
            // todo throw away undefined vals

            OUT val = eval_learner(data, S.at(j), candidate_params.at(i));

            switch (val) {
            case LEFT:
              left_data.push_back(S.at(j));
              break;
            case RIGHT:
              right_data.push_back(S.at(j));
              break;
            case TRASH:
              // do nothing
              break;
            }

          }

          // eval tree training objective function and take best
          // todo: ensure objective function is correct
          double objective = objective_function(S, left_data, right_data);

          if (objective < minimum_objective) {
            feature = i;
            minimum_objective= objective;
            left_final = left_data;
            right_final = right_data;
          }
        }

        // set feature
        node->feature_ = candidate_params.at(feature);
        node->left_ = new Node();
        node->right_ = new Node();
        node->left_->parent_ = node->right_->parent_ = node;

        train_recurse(node->left_, left_final);
        train_recurse(node->right_, right_final);
      }

      void Train(Data *data, std::vector<LabeledPixel> labeled_data, Random *random, Settings *settings) 
      {
        this->data = data;
        this->random = random;
        this->settings = settings;
        train_recurse(this->root, labeled_data);
      }

      std::vector<cv::Point3d> eval_recursive(Node *node, int row, int col, cv::Mat rgb_image, cv::Mat depth_image)
      {

        if (node->is_leaf_) {
          return node->modes_;
        }

        auto pair = node->feature_->GetResponse(depth_image, rgb_image, cv::Point2i(col, row));

        OUT val = (OUT)(pair.first >= node->feature_->GetThreshold());

        switch (val) {
        case LEFT:
          eval_recursive(node->left_, row, col, rgb_image, depth_image);
          break;
        case RIGHT:
          eval_recursive(node->right_, row, col, rgb_image, depth_image);
          break;
        case TRASH:
          // do nothing
          break;
        }
      }

      // Evaluate tree at a pixel
      std::vector<cv::Point3d> Eval(int row, int col, cv::Mat rgb_image, cv::Mat depth_image)
      {
        eval_recursive(root, row, col, rgb_image, depth_image);
      }


    private:
      Node *root;
      Data *data;
      Random *random;
      Settings *settings;
      const char* binaryFileHeader_ = "ISUE.RelocForests.Tree";
    };
  }
}
