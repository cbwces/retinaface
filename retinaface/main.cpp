#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <math.h>
#include "RetinaFace.h"

using namespace std;

void do_detect(string& path, vector<string>& imgnames, int thread_num, int& thread_size, float& margin, string& savepath);
vector<string> file_chunk(vector<string>& imgnames, int thread_num, int& thread_size);
void savefile(Mat& img, string& current_path, string& aim_path);

int main(int argc, char* argv[])
{
    // gflags::ParseCommandLineFlags(&argc, &argv, true);

    string allpath = argv[1];
    int numthread = 1;
    float margin = 0.0;
    string savepath = "";
    if (argc <= 3){
        numthread = std::atoi(argv[2]);
    }
    else if(argc == 4){
        savepath = argv[2];
        numthread = std::atoi(argv[3]);
        margin = 0.0;
    }
    else if(argc == 5){
        margin = std::atof(argv[2]);
        savepath = argv[3];
        numthread = std::atoi(argv[4]);
    }
    else{
        cout << "(imagepathfile, margin(optional, default=0), savepath(optional, default=\"\"), threads(optional, default=1)" << endl;
        EXIT_FAILURE;
    }

    cout << "thread:" << numthread << endl;
    cout << "margin:" << margin << endl;
    cout << "savepath:" << savepath << endl;

    string modelpath = "../model";

    //raed path from file
    vector<string> imgs;
    string fileline;
    ifstream imgtxt(allpath);
    while(getline(imgtxt, fileline)){
        imgs.push_back(fileline);
    }
    imgtxt.close();

    vector<thread> thread_pool;
    for (int n_thread=0; n_thread!=numthread; ++n_thread){
        thread_pool.emplace_back(do_detect, ref(modelpath), ref(imgs), n_thread, ref(numthread), ref(margin), ref(savepath));
    }
    for (auto i=0; i!=thread_pool.size(); ++i){
        thread_pool[i].join();
    }

    return 0;
}

void do_detect(string& path, vector<string>& imgnames, int thread_num, int& thread_size, float& margin, string& savepath){

    vector<string> chunk_imgpaths = file_chunk(imgnames, thread_num, thread_size);
    RetinaFace *rf = new RetinaFace(path, "net3");

    for (auto &n : chunk_imgpaths){
        cv::Mat mat_img = cv::imread(n);
        FaceDetectInfo face = rf->detect(mat_img, 0.9);

        cv::Mat croped_img;
        if (savepath != ""){
            if ((double)(long long)margin == margin){
                croped_img = rf->icropimg(face.rect, margin);
            }
            else 
            {
                croped_img = rf->fcropimg(face.rect, margin);
            }
        //裁剪并保存图片
            cvtColor(croped_img, croped_img, COLOR_BGR2RGB);
            savefile(croped_img, n, savepath);
        }
    }
}

vector<string> file_chunk(vector<string>& imgnames, int thread_num, int& thread_size){
    vector<string> imgname_chunk;
    for(int i=thread_num; i<imgnames.size(); i=i){
        imgname_chunk.push_back(imgnames[i]);
        i += thread_size;
    }
    return imgname_chunk;
}

void savefile(Mat& img, string& current_path, string& aim_path){
    auto last_splash_pos = current_path.find_last_of("/\\");
    string complete_save_path = aim_path + "/" +current_path.substr(last_splash_pos+1);
    cv::imwrite(complete_save_path, img);
}