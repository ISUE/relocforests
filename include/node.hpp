#pragma once

namespace ISUE {
  namespace RelocForests {
    class Node {
    public:
      Node()
      {
        left_ = nullptr;
        right_ = nullptr;
      };
    private:
      bool is_leaf_;
      bool is_split_;
      Node *left_;
      Node *right_;
    };
  }
}