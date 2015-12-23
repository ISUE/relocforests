#pragma once

#include "node.hpp"
#include "random.hpp"
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

      bool eval_learner(Data *data, LabeledPixel pixel, DepthAdaptiveRGB *feature) 
      {
        return feature->GetResponse(data, pixel) >= feature->GetThreshold();
      }

      float objective_function()
      {

      }


      float variance(std::vector<LabeledPixel> labeled_data)
      {
        float V = (1 / labeled_data.size());
        float sum = 0;
        for (int i = 0; i < lab)
      }


      void Train(Data *data, std::vector<LabeledPixel> labeled_data, Random *random, int image_width, int image_height) 
      {
        int num_candidates = 5;
        std::vector<DepthAdaptiveRGB*> candidate_params;

        for (int i = 0; i < num_candidates; ++i) {
          candidate_params.push_back(DepthAdaptiveRGB::CreateRandom(random, image_width, image_height));
          // partition data
          for (int j = 0; j < labeled_data.size(); ++j) {
            std::vector<LabeledPixel> left_data;
            std::vector<LabeledPixel> right_data;
            if (eval_learner(data, labeled_data.at(j), candidate_params.at(i)))
              left_data.push_back(labeled_data.at(j));
            else
              right_data.push_back(labeled_data.at(j));

            // eval tree training objective function and take best




          }

        }



        //root->feature_ = DepthAdaptiveRGB::CreateRandom(random, image_width, image_height);
      }

      // void train_recurse(Node *head, set<LabeledData*> S)
    private:
      Node *root;
    };
  }
}