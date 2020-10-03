#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <thread>
#include "RetinaFace.h"
#include "timer.h"

using namespace std;

// gflags::DEFINE_double(alpha, .35, "coefficient for equal score");
// gflags::DEFINE_string(imgout, "/dev/null", "cropped iamge outpath");
// gflags::DEFINE_string(logout, "/dev/null", "log outpath");
// gflags::DEFINE_int32(margin, 0, "cropping bound added on box");
// gflags::DEFINE_int32(numthread, 1, "number of program thread to use");

void do_detect(string model_path, vector<string> imgnames, int thread_num, int thread_size);
vector<string> file_chunk(vector<string> imgnames, int thread_num, int thread_size);

int main(int argc, char* argv[])
{
    // gflags::ParseCommandLineFlags(&argc, &argv, true);
    string allpath = argv[1];
    const int numthread = std::atoi(argv[2]);

    string path = "../model";

    //raed path from file
    vector<string> imgs;
    string fileline;
    ifstream imgtxt(allpath);
    while(getline(imgtxt, fileline)){
        imgs.push_back(fileline);
    }
    imgtxt.close();
    cout << "image load complete." << endl;

    vector<thread> thread_pool;
    for (int n_thread=0; n_thread!=numthread; ++n_thread){
        thread_pool.emplace_back(do_detect, ref(path), ref(imgs), ref(n_thread), ref(numthread));
    }
    for (auto i=0; i!=thread_pool.size(); ++i){
        thread_pool[i].join();
        cout << "starting thread " << i;
    }
    cout << endl;

    return 0;
}

void do_detect(string model_path, vector<string> imgnames, int thread_num, int thread_size){
    RetinaFace *rf = new RetinaFace(model_path, "net3");
    vector<string> chunk_imgpaths = file_chunk(imgnames, thread_num, thread_size);
    for (auto &n : chunk_imgpaths){
        cout << "Predict image: " << n << endl;
        cv::Mat mat_img = cv::imread(n);
        rf->detect(mat_img, 0.9);
    }
}

vector<string> file_chunk(vector<string> imgnames, int thread_num, int thread_size){
    vector<string> imgname_chunk;
    for(int i=thread_num; i<=imgnames.size(); i+=thread_size){
        imgname_chunk.push_back(imgnames[i]);
    }
    return imgname_chunk;
}
