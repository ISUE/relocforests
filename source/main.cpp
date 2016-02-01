#include <iostream>

#include "reader.hpp"
#include "settings.hpp"
#include "forest.hpp"


#include "rendar.hpp"
#include "engine.hpp"
#include "camera.hpp"
#include "cube.hpp"
#include "pyramid.hpp"
#include "light.hpp"
#include "model.hpp"
#include "first_person_camera.hpp"


using namespace std;
using namespace ISUE::RelocForests;
using namespace RendAR;
using namespace glm;


// control callbacks
void do_movement();
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);

Forest *forest;
Data *data;

Camera* camera;
Pyramid* known_pose;
Pyramid* forest_pose;
Light* light;

GLfloat lastX = 400, lastY = 400;
bool keys[1024];
GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;
bool firstMouse = true;

int frame_num = 0;
void updateLoop()
{
  GLfloat currentFrame = glfwGetTime();
  deltaTime = currentFrame - lastFrame;
  lastFrame = currentFrame;
  do_movement();

  // show pose comparison
  if (frame_num < data->rgb_images_.size()) {
    Eigen::Affine3d pose = forest->Test(data->GetRGBImage(frame_num), data->GetDepthImage(frame_num));

    Quaternion<double> q(pose.rotation());
    forest_pose->setPosition(vec3(pose.translation().x(), pose.translation().y(), pose.translation().z()));
    forest_pose->setRotation(quat(q.x(), q.y(), q.z(), q.w()));
    auto data_pose = data->poses_eigen_.at(frame_num);
    Quaternion<double> qk(data_pose.first);
    known_pose->setPosition(vec3(data_pose.second.x(), data_pose.second.y(), data_pose.second.z()));
    known_pose->setRotation(quat(qk.x(), qk.y(), qk.z(), qk.w()));
    frame_num++;
  }
}


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
  string time(argv[2]);
  bool train = (time == "train");

  // Get data from reader
  data = reader->GetData();

  // settings for forest
  Settings *settings = new Settings();
  settings->fx = 525.0f;
  settings->fy = 525.0f;
  settings->cx = 319.5f;
  settings->cy = 239.5f;

  forest = nullptr;

  string forest_file_name;
  if (argv[3])
    forest_file_name = string(argv[3]);
  else
    cout << "Train and Test time requires an output file name.\n";

  if (train) {
    

    forest = new Forest(data, settings);
    forest->Train();
    forest->Serialize(forest_file_name);

    cout << "Is forest valid:" << forest->IsValid() << endl;
  }
  else {

    // load forest
    forest = new Forest(data, settings, forest_file_name);

    cout << "Is forest valid:" << forest->IsValid() << endl;

    // eval forest at frame
    std::clock_t start;
    double duration;
    start = std::clock();

    Eigen::Affine3d pose = forest->Test(data->GetRGBImage(200), data->GetDepthImage(200));

    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    cout << "Train time: " << duration << " Seconds \n";

    // compare pose to known value 
    auto known_pose = data->poses_eigen_.at(200);

    cout << "found pose:" << endl;
    cout << pose.rotation() << endl << endl;
    cout << pose.rotation().eulerAngles(0, 1, 2) * 180 / M_PI << endl;
    cout << pose.translation() << endl;

    cout << "known pose:" << endl;
    cout << known_pose.first << endl;
    cout << known_pose.first.eulerAngles(0, 1, 2) * 180 / M_PI << endl;
    cout << known_pose.second << endl;

    if ((known_pose.first - pose.rotation()).cwiseAbs().maxCoeff() > 1e-13 ||
      (known_pose.second - pose.translation()).cwiseAbs().maxCoeff() > 1e-13)
      cout << "Pose could not be found\n";
    else
      cout << "Pose was found!\n";
  }

  cout << "Done.\n";

  Engine::init(argc, argv);

  auto glfw_context = dynamic_cast<GLFW3Context*>(Engine::context()); 
  glfw_context->setKeyCallBack(&key_callback);
  glfw_context->setCursorPosCallback(&mouse_callback);

  glfw_context->setClearColor(vec3(0.0f, 0.0f, 0.0f));

  Scene* scene = Engine::activeScene();
  camera = new FirstPersonCamera(vec3(0.0f, 3.0f, 1.5f));
  scene->setCamera(camera);

  known_pose = new Pyramid();
  known_pose->setColor(vec3(0.0f, 0.737f, 0.831f));
  known_pose->setWireframe(true);

  forest_pose = new Pyramid();
  forest_pose->setColor(vec3(1.0f, 0.0f, 0.831f));
  forest_pose->setWireframe(true);

  Cube *floor = new Cube();
  floor->setColor(vec3(0.3f, 0.3f, 0.35f));
  floor->setScale(vec3(20.0f, .2f, 20.0f));
  floor->setPosition(vec3(0, -2.2f, 0));
  floor->setShader(Shader("Shaders/default.vert", "Shaders/default.frag"));

  //Model *model = new Model("long_hallway.stl");

  light = new Light();
  light->setPosition(vec3(0.0f, 2.0f, 0.0f));

  scene->add(light);
  //scene->add(model);
  scene->add(floor);
  scene->add(forest_pose);
  scene->add(known_pose);

  Engine::startMainLoop(&updateLoop);

  // cleanup forest
  delete forest;
  delete reader;
  delete settings;

  // cleanup rendar
  delete known_pose;
  delete forest_pose;


  return 0;
}

void do_movement()
{
  if (keys[GLFW_KEY_W])
    camera->move(FORWARD, deltaTime);
  if (keys[GLFW_KEY_S])
    camera->move(BACK, deltaTime);
  if (keys[GLFW_KEY_A])
    camera->move(LEFT, deltaTime);
  if (keys[GLFW_KEY_D])
    camera->move(RIGHT, deltaTime);
}


void
mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
  if (firstMouse) {
    lastX = xpos;
    lastY = ypos;
    firstMouse = false;
  }
  vec2 offset(xpos - lastX, lastY - ypos);
  lastX = xpos;
  lastY = ypos;
  camera->processMouse(offset);
}


void
key_callback(GLFWwindow *window, int key, int scancode, int action, int mode)
{
  //GLfloat cameraSpeed = 0.05f;
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  if (key >= 0 && key < 1024) {
    if (action == GLFW_PRESS)
      keys[key] = true;
    else if (action == GLFW_RELEASE)
      keys[key] = false;
  }
}
