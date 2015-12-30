#pragma once

#include "node.hpp"
#include "random.hpp"
#include "settings.hpp"
#include <vector>
#include <data.hpp>

namespace ISUE {
  namespace RelocForests {
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

      bool eval_learner(Data *data, LabeledPixel pixel, DepthAdaptiveRGB *feature) 
      {
        return feature->GetResponse(data, pixel) >= feature->GetThreshold();
      }

      double variance(std::vector<LabeledPixel> labeled_data)
      {
        double V = (1 / labeled_data.size());
        double sum = 0;

        // calculate mean of S
        cv::Point3f tmp;
        for (auto p : labeled_data)
          tmp += p.label_;
        int size = labeled_data.size();
        cv::Point3f mean(tmp.x / size, tmp.y / size, tmp.z / size);
        
        for (auto p : labeled_data) {
          cv::Point3f val = (p.label_ - mean);
          sum += powf(val.x, 2) + powf(val.y, 2) + powf(val.x, 2);
        }

        return V * sum;
      }



      double objective_function(std::vector<LabeledPixel> data, std::vector<LabeledPixel> left, std::vector<LabeledPixel> right)
      {
        double var = variance(data);
        double sum;

        // left
        double left_val = (left.size() / data.size()) * variance(left);
        // right
        double right_val = (right.size() / data.size()) * variance(right);

        sum = left_val + right_val;

        return var - sum;
      }

      int get_height(Node *node)
      {
        if (node == nullptr)
          return 0;
        return std::max(1 + get_height(node->left_), 1 + get_height(node->right_));
      }

      Data *data;
      Random *random;
      Settings *settings;

      void train_recurse(Node *node, std::vector<LabeledPixel> S) 
      {
        if (S.size() < 1 || get_height(root) >= settings->max_tree_depth_) {
          node->is_leaf_ = true;
          node->distribution_ = S;
          return;
        }

        int num_candidates = 5;
        std::vector<DepthAdaptiveRGB*> candidate_params;
        std::vector<LabeledPixel> left_final;
        std::vector<LabeledPixel> right_final;
        double minimum_variation = DBL_MAX;
        int feature;

        for (int i = 0; i < num_candidates; ++i) {
          candidate_params.push_back(DepthAdaptiveRGB::CreateRandom(random, settings->image_width_, settings->image_height_));
          // partition data
          for (int j = 0; j < S.size(); ++j) {
            std::vector<LabeledPixel> left_data;
            std::vector<LabeledPixel> right_data;
            if (eval_learner(data, S.at(j), candidate_params.at(i)))
              left_data.push_back(S.at(j));
            else
              right_data.push_back(S.at(j));

            // eval tree training objective function and take best
            double variance = objective_function(S, left_data, right_data);
            if (variance < minimum_variation) {
              feature = i;
              minimum_variation = variance;
              left_final = left_data;
              right_final = right_data;
            }
          }
        }
        // set feature
        node->feature_ = candidate_params.at(feature);
        node->left_ = new Node();
        node->right_ = new Node();

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

      // void train_recurse(Node *head, set<LabeledData*> S)
    private:
      Node *root;
    };
  }
}