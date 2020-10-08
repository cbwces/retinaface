#include <iostream>
#include <fstream>
#include <thread>
#include <math.h>
#include "RetinaFace.h"

using namespace std;

void do_detect(string& path, vector<string>& imgnames, float& threshold, int thread_num, int& thread_size, float& margin, string& savepath);
vector<string> file_chunk(vector<string>& imgnames, int thread_num, int& thread_size);
void savefile(Mat& img, string& current_path, string& aim_path);
void print_info(string& current_path, anchor_box& bbox, FacePts& pts);

int main(int argc, char* argv[])
{
    // gflags::ParseCommandLineFlags(&argc, &argv, true);

    string allpath = argv[1];
    int numthread = 1;
    float margin = 0.0;
    string savepath = "";
    float face_quality_threshold = 0.9;

    if (argc == 3){
        savepath = argv[2];
    }
    else if(argc == 4){
        savepath = argv[2];
        margin = std::atof(argv[3]);
    }
    else if(argc == 5){
        savepath = argv[2];
        margin = std::atof(argv[3]);
        numthread = std::atoi(argv[4]);
    }
    else if(argc == 6){
        savepath = argv[2];
        margin = std::atof(argv[3]);
        numthread = std::atoi(argv[4]);
        face_quality_threshold = std::atof(argv[5]);
    }
    else{
        cout << "imagepathfile, savepath, margin(default=0), threads(optional, default=1), score_threshold(optional, default=.9)" << endl;
        return 1;
    }

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
        thread_pool.emplace_back(do_detect, ref(modelpath), ref(imgs), ref(face_quality_threshold), n_thread, ref(numthread), ref(margin), ref(savepath));
    }
    for (auto i=0; i!=thread_pool.size(); ++i){
        thread_pool[i].join();
    }

    return 0;
}

void do_detect(string& path, vector<string>& imgnames, float& threshold, int thread_num, int& thread_size, float& margin, string& savepath){

    vector<string> chunk_imgpaths = file_chunk(imgnames, thread_num, thread_size);
    RetinaFace *rf = new RetinaFace(path, "net3");

    for (auto &n : chunk_imgpaths){
        cv::Mat mat_img = cv::imread(n);
        vector<FaceDetectInfo> face = rf->detect(mat_img, threshold);

        if (face.size() != 0){

            print_info(n, face[0].rect, face[0].pts);
            cv::Mat croped_img;
            if ((double)(long long)margin == margin){
                croped_img = rf->icropimg(face[0].rect, margin);
            }
            else 
            {
                croped_img = rf->fcropimg(face[0].rect, margin);
            }
            //裁剪并保存图片
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

void print_info(string& current_path, anchor_box& bbox, FacePts& pts){

    std::cout << current_path << std::endl;

    std::cout << "bbox" 
        << ": x1 " << std::to_string(bbox.x1) << " y1 " << std::to_string(bbox.y1)
        << " x2 " << std::to_string(bbox.x2) << " y2 " << std::to_string(bbox.y2) << std::endl;

    std::cout << "landmark:";
    for(size_t j = 0; j < 5; ++j) {
        std::cout << " " << std::to_string(pts.x[j]) << " " << std::to_string(pts.y[j]);
    }
    std::cout << std::endl;
}