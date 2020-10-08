#include <algorithm>
#include <cuda_runtime_api.h>
#include "RetinaFace.h"

void imageROIResize8U3C(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight);
void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream);
void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream);

//processing
anchor_win  _whctrs(anchor_box anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    vector<anchor_box> anchors;
    for(size_t i = 0; i < ratios.size(); i++) {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
    vector<anchor_box> anchors;
    for(size_t i = 0; i < scales.size(); i++) {
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = {0.5, 1, 2},
                      vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

    vector<anchor_box> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

    vector<anchor_box> anchors;
    for(size_t i = 0; i < ratio_anchors.size(); i++) {
        vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if(dense_anchor) {
        assert(stride % 2 == 0);
        vector<anchor_box> anchors2 = anchors;
        for(size_t i = 0; i < anchors2.size(); i++) {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

vector<vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, vector<anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    vector<vector<anchor_box>> anchors;
    for(size_t i = 0; i < cfg.size(); i++) {
        //stride从小到大[32 16 8]
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
        vector<float> ratios = tmp.RATIOS;
        vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

        vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors)
{
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

    vector<anchor_box> all_anchors;
    for(size_t k = 0; k < base_anchors.size(); k++) {
        for(int ih = 0; ih < height; ih++) {
            int sh = ih * stride;
            for(int iw = 0; iw < width; iw++) {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(vector<anchor_box> &boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for(size_t i = 0; i < boxes.size(); i++) {
        if(boxes[i].x1 < 0) {
            boxes[i].x1 = 0;
        }
        if(boxes[i].y1 < 0) {
            boxes[i].y1 = 0;
        }
        if(boxes[i].x2 > width - 1) {
            boxes[i].x2 = width - 1;
        }
        if(boxes[i].y2 > height - 1) {
            boxes[i].y2 = height -1;
        }
//        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
    }
}

void clip_boxes(anchor_box &box, int width, int height)
{
    //Clip boxes to image boundaries.
    if(box.x1 < 0) {
        box.x1 = 0;
    }
    if(box.y1 < 0) {
        box.y1 = 0;
    }
    if(box.x2 > width - 1) {
        box.x2 = width - 1;
    }
    if(box.y2 > height - 1) {
        box.y2 = height -1;
    }
//    boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//    boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//    boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//    boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);

}

//######################################################################
//retinaface
//######################################################################

RetinaFace::RetinaFace(string &model, string network, float nms)
    : network(network), nms_threshold(nms)
{
    //主干网络选择
    int fmc = 3;

    if (network=="ssh" || network=="vgg") {
        pixel_means[0] = 103.939;
        pixel_means[1] = 116.779;
        pixel_means[2] = 123.68;
    }
    else if(network == "net3") {
        _ratio = {1.0};
    }
    else if(network == "net3a") {
        _ratio = {1.0, 1.5};
    }
    else if(network == "net6") { //like pyramidbox or s3fd
        fmc = 6;
    }
    else if(network == "net5") { //retinaface
        fmc = 5;
    }
    else if(network == "net5a") {
        fmc = 5;
        _ratio = {1.0, 1.5};
    }

    else if(network == "net4") {
        fmc = 4;
    }
    else if(network == "net5a") {
        fmc = 4;
        _ratio = {1.0, 1.5};
    }
    else {
        std::cout << "network setting error" << network << std::endl;
    }

    //anchor配置
    if(fmc == 3) {
        _feat_stride_fpn = {32, 16, 8};
        anchor_cfg tmp;
        tmp.SCALES = {32, 16};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 32;
        cfg.push_back(tmp);

        tmp.SCALES = {8, 4};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 16;
        cfg.push_back(tmp);

        tmp.SCALES = {2, 1};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 8;
        cfg.push_back(tmp);
    }
    else {
        std::cout << "please reconfig anchor_cfg" << network << std::endl;
    }

    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    Net_.reset(new Net<float>((model + "/mnet-deconv-0517.prototxt"), TEST));
    Net_->CopyTrainedLayersFrom((model + "/mnet-deconv-0517.caffemodel"));

    bool dense_anchor = false;
    vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
    for(size_t i = 0; i < anchors_fpn.size(); i++) {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
    }
}


RetinaFace::~RetinaFace(){}

vector<anchor_box> RetinaFace::bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress)
{
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """

    vector<anchor_box> rects(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box RetinaFace::bbox_pred(anchor_box anchor, cv::Vec4f regress)
{
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    //人脸框中心点
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

vector<FacePts> RetinaFace::landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts)
{
    vector<FacePts> pts(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for(size_t j = 0; j < 5; j ++) {
            pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

FacePts RetinaFace::landmark_pred(anchor_box anchor, FacePts facePt)
{
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for(size_t j = 0; j < 5; j ++) {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool RetinaFace::CompareBBox(const FaceDetectInfo & a, const FaceDetectInfo & b)
{
    return a.score > b.score;
}

std::vector<FaceDetectInfo> RetinaFace::nms(std::vector<FaceDetectInfo>& bboxes, float threshold)
{
    std::vector<FaceDetectInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        //如果全部执行完则返回
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        anchor_box select_bbox = bboxes[select_idx].rect;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            anchor_box& bbox_i = bboxes[i].rect;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float 型不加1
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;

   
            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}

vector<FaceDetectInfo> RetinaFace::detect(Mat &img, float threshold, float scales)
{
    if(img.empty()) {
        EXIT_FAILURE;
    }

    // double pre = (double)getTickCount(); //计时

    this->resized_img = img;
    //边缘拉伸为32的整数倍
    int ws = (img.cols + 31) / 32 * 32;
    int hs = (img.rows + 31) / 32 * 32;
    cv::copyMakeBorder(img, img, 0, hs - img.rows, 0, ws - img.cols, cv::BORDER_CONSTANT,cv::Scalar(0));

    // int ws = img.cols / 32 * 32;
    // int hs = img.rows / 32 * 32;
    // if (ws == 0)
    // {
    //     ws = 32;
    // }
    // if (hs == 0)
    // {
    //     hs = 32;
    // }
    // cv::Mat dstimg(hs, ws, CV_32FC3);
    img.convertTo(img, CV_32FC3);
    // cv::resize(dstimg, dstimg, cv::Size(ws, hs));

    //rgb
    cvtColor(img, img, COLOR_BGR2RGB);

    //图片送入caffe输入层
    Blob<float>* input_layer = Net_->input_blobs()[0];

    input_layer->Reshape(1, 3, img.rows, img.cols);
    Net_->Reshape();

    vector<Mat> input_channels;
    int width = input_layer->width();
    int height = input_layer->height();
    int input_area = width * height;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += input_area;
    }

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    split(img, input_channels);

    // pre = (double)getTickCount() - pre;
    // std::cout << "pre compute time :" << pre*1000.0 / cv::getTickFrequency() << " ms \n";

    //LOG(INFO) << "Start net_->Forward()";
    // double t1 = (double)getTickCount();
    Net_->Forward();
    // t1 = (double)getTickCount() - t1;
    // std::cout << "infer compute time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";
    //LOG(INFO) << "Done net_->Forward()";

    // double post = (double)getTickCount();
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score ="face_rpn_cls_prob_reshape_";
    string name_landmark ="face_rpn_landmark_pred_";

    vector<FaceDetectInfo> faceInfo;
    for(size_t i = 0; i < _feat_stride_fpn.size(); i++) {
///////////////////////////////////////////////
        // double s1 = (double)getTickCount();
///////////////////////////////////////////////
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        int stride = _feat_stride_fpn[i];

        string str = name_score + key;
        const boost::shared_ptr<Blob<float>> score_blob = Net_->blob_by_name(str);
        const float* scoreB = score_blob->cpu_data() + score_blob->count() / 2;
        const float* scoreE = scoreB + score_blob->count() / 2;
        std::vector<float> score = std::vector<float>(scoreB, scoreE);

        str = name_bbox + key;
        const boost::shared_ptr<Blob<float>> bbox_blob = Net_->blob_by_name(str);
        const float* bboxB = bbox_blob->cpu_data();
        const float* bboxE = bboxB + bbox_blob->count();
        std::vector<float> bbox_delta = std::vector<float>(bboxB, bboxE);

        str = name_landmark + key;
        const boost::shared_ptr<Blob<float>> landmark_blob = Net_->blob_by_name(str);
        const float* landmarkB = landmark_blob->cpu_data();
        const float* landmarkE = landmarkB + landmark_blob->count();
        std::vector<float> landmark_delta = std::vector<float>(landmarkB, landmarkE);

        int width = score_blob->width();
        int height = score_blob->height();
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

///////////////////////////////////////////////
        // s1 = (double)getTickCount() - s1;
        // std::cout << "s1 compute time :" << s1*1000.0 / cv::getTickFrequency() << " ms \n";
///////////////////////////////////////////////

        //存储顺序 h * w * num_anchor
        vector<anchor_box> anchors = anchors_plane(height, width, stride, _anchors_fpn[key]);
        
        float best_score = 0.0;
        for(size_t num = 0; num < num_anchor; num++) {
            for(size_t j = 0; j < count; j++) {

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //回归人脸框
                anchor_box rect = bbox_pred(anchors[j + count * num], regress);
                //越界处理
                clip_boxes(rect, ws, hs);

                //置信度小于阈值跳过
                float conf = score[j + count * num];
                if(conf <= threshold) {
                    continue;
                }
                conf = 0.65 * conf + 0.35 * (rect.x2-rect.x1)*(rect.y2-rect.y1) / (float)input_area;
                if (conf < best_score){
                    continue;
                }
                else{
                    best_score = conf;
                }

                FacePts pts;
                for(size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                FacePts landmarks = landmark_pred(anchors[j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }
    if (faceInfo.size() == 0){
        return faceInfo;
    }
    //排序nms
    faceInfo = nms(faceInfo, nms_threshold);

    std::cout << "bbox" 
        << ": x1 " << std::to_string(faceInfo[0].rect.x1) << " y1 " << std::to_string(faceInfo[0].rect.y1)
        << " x2 " << std::to_string(faceInfo[0].rect.x2) << " y2 " << std::to_string(faceInfo[0].rect.y2) << std::endl;

    std::cout << "landmark:";
    for(size_t j = 0; j < 5; ++j) {
        cout << " " << std::to_string(faceInfo[0].pts.x[j]) << " " << std::to_string(faceInfo[0].pts.y[j]);
    }
    cout << std::endl;

    return faceInfo;
}

bool comp_min(const int &a, const int &b)
{
    return a < b;
}

bool comp_max(const int &a, const int &b)
{
    return a < b;
}

cv::Mat RetinaFace::icropimg(anchor_box &rect, int margin){
    int left_x = std::max(0, (int)(rect.x1-margin), comp_max);
    int top_y = std::max(0, (int)(rect.y1-margin), comp_max);
    int right_x = std::min(resized_img.cols, (int)(rect.x2+margin), comp_min);
    int bottom_y = std::min(resized_img.rows, (int)(rect.y2+margin), comp_min);
    cv::Rect rect_box = cv::Rect(left_x, top_y, right_x-left_x, bottom_y-top_y);
    cout << rect_box << endl;
    cv::Mat croped_img(resized_img, rect_box);
    return croped_img;
}

cv::Mat RetinaFace::fcropimg(anchor_box &rect, float margin){
    float width = rect.x2 - rect.x1;
    float height = rect.y2 - rect.y1;
    int left_x = std::max(0, (int)(rect.x1-width*margin), comp_max);
    int top_y = std::max(0, (int)(rect.y1-height*margin), comp_max);
    int right_x = std::min(resized_img.cols, (int)(rect.x2+width*margin), comp_min);
    int bottom_y = std::min(resized_img.rows, (int)(rect.y2+height*margin), comp_min);
    cv::Rect rect_box = cv::Rect(left_x, top_y, right_x-left_x, bottom_y-top_y);
    cv::Mat croped_img(resized_img, rect_box);
    return croped_img;
}