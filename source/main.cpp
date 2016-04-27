#include <iostream>

#include "reader.hpp"
#include "settings.hpp"
#include "forest.hpp"

using namespace std;
using namespace ISUE::RelocForests;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: ./relocforests <path_to_association_file> (train|test) <forest_file_name>\n";
    return 1;
  }

  // get path
  string data_path(argv[1]);

  Reader *reader = new Reader();
  bool err = reader->Load(data_path);
  if (err) {
    return 1;
  }
  
  // train or test time
  string shouldTrain(argv[2]);
  bool train = (shouldTrain == "train");

  // Get data from reader
  Data *data = reader->GetData();

  // settings for forest
  Settings *settings = new Settings(640, 480, 5000, 525.0f, 525.0f, 319.5f, 239.5f);

  Forest<ushort, cv::Vec3b> *forest = nullptr;
  
  string forest_file_name;
  if (argv[3])
    forest_file_name = string(argv[3]);
  else
    cout << "Train and Test time requires an output file name.\n";

  if (shouldTrain) {
    forest = new Forest<ushort, cv::Vec3b>(data, settings);
    forest->Train();
    forest->Serialize(forest_file_name);

    cout << "Is forest valid:" << forest->IsValid() << endl;
  }
  else {

    // load forest
    forest = new Forest<ushort, cv::Vec3b>(data, settings, "forest.rf");

    cout << "Is forest valid:" << forest->IsValid() << endl;

    // eval forest at frame
    std::clock_t start;
    double duration;
    start = std::clock();

    Eigen::Affine3d pose = forest->Test(data->GetRGBImage(200), data->GetDepthImage(200));

    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    cout << "Train time: " << duration << " Seconds \n";

    // compare pose to known value 
    auto known_pose = data->GetPose(200);

    cout << "found pose:" << endl;
    cout << pose.rotation() << endl << endl;
    cout << pose.rotation().eulerAngles(0, 1, 2) * 180 / M_PI << endl;
    cout << pose.translation() << endl;

    cout << "known pose:" << endl;
    cout << known_pose.rotation << endl;
    cout << known_pose.rotation.eulerAngles(0, 1, 2) * 180 / M_PI << endl;
    cout << known_pose.position << endl;

  }

  delete forest;
  delete reader;
  delete settings;


  return 0;
}
