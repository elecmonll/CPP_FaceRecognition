#pragma warning(disable : 4996)
#define CVUI_IMPLEMENTATION
#include "cvui-2.7.0/cvui.h"

#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <ctime>

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/face.hpp>

#define MAIN_WINDOW_NAME "Opencvui"
#define CAMERA_WINDOW_NAME "Opencvui Camera"
#define VIDEO_WINDOW_NAME "Opencvui Video"
#define PHOTO_WINDOW_NAME "Opencvui Photo"
#define ADD_WINDOW_NAME "Opencvui Add Photo"

namespace fs = std::filesystem;
namespace fc = cv::face;

class RecognationFace {
public:
    cv::CascadeClassifier cascadeClassifier;
    cv::Ptr<fc::LBPHFaceRecognizer> model =
        fc::LBPHFaceRecognizer::create();

    cv::Mat guiframe = cv::Mat(300, 250, CV_8UC3);
    cv::Mat cameraframe = cv::Mat(400, 500, CV_8UC3);
    cv::Mat videoframe = cv::Mat(400, 500, CV_8UC3);
    //cv::Mat photoframe = cv::Mat(400, 500, CV_8UC3);

    std::vector<std::string> Id;
    std::string pathDataFace = "C:/Source/dataFace/";
    std::string pathDataVideo = "C:/Source/dataVideo/";
    std::string pathHaar = "xml/haarcascade_frontalface_default.xml";

public:
    void GUI();
    void Camera();
    void Video();
    void Photo();
    void AddPhoto();
    void TrainnerLBP();

};
void RecognationFace::GUI() {
    cv::namedWindow(MAIN_WINDOW_NAME);
    cvui::init(MAIN_WINDOW_NAME);

    while (true) {
        guiframe = cv::Scalar(49, 52, 49);
        if (cvui::button(guiframe, 10, 10, 230, 40, "Camera")) {
            Camera();
        }
        if (cvui::button(guiframe, 10, 60, 230, 40, "Video")) {
            Video();
        }
        if (cvui::button(guiframe, 10, 110, 230, 40, "Photo")) {
            //Photo();
        }
        if (cvui::button(guiframe, 10, 160, 230, 40, "Add photo")) {
            AddPhoto();
        }
        if (cvui::button(guiframe, 10, 250, 230, 40, "Exit")) {
            cv::destroyWindow(MAIN_WINDOW_NAME);
            break;
        }

        cvui::update();
        cv::imshow(MAIN_WINDOW_NAME, guiframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Exit from " << MAIN_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MAIN_WINDOW_NAME);
            break;
        }
    }
}
void RecognationFace::Camera() {
    cv::VideoCapture videoCapture(0);
    cascadeClassifier.load(pathHaar);

    while (true) {
        videoCapture.read(cameraframe);
        if (cameraframe.empty()) {
            std::cout << "Camera disconnected!" << std::endl;
            return;
        }

        cv::Mat grayframe;
        cv::cvtColor(cameraframe, grayframe, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        cascadeClassifier.detectMultiScale(grayframe, faces, 1.3, 5);

        for (int i = 0; i < faces.size(); ++i) {
            int predictedLabel = -1;
            double confidence = 0.0;
            std::string faceId = "Empty";

            //model->predict(grayframe, predictedLabel, confidence);
            //if (predictedLabel == 1) {
            //    faceId = "Kim";
            //}
            //else{
            //    faceId = "Unknown";
            //}

            rectangle(cameraframe, faces[i].tl(), faces[i].br(), 
                      cv::Scalar(50, 50, 255), 3);
            putText(cameraframe, faceId, faces[i].tl() + cv::Point(5, 15),
                cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(50, 50, 255));
        }
                
        cvui::update();
        cvui::imshow(CAMERA_WINDOW_NAME, cameraframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the " << CAMERA_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(CAMERA_WINDOW_NAME);
            break;
        }
    }
}
void RecognationFace::Video() {
    cascadeClassifier.load(pathHaar);

    for (auto const& buffPath : fs::directory_iterator(pathDataVideo)) {
        std::string filePath = buffPath.path().string();
        
        cv::namedWindow(VIDEO_WINDOW_NAME);
        cv::VideoCapture videoCapture(filePath);
        
        while (true) {
            videoCapture.read(videoframe);
            if (videoframe.empty()) {
                std::cout << "Video disconnected!" << std::endl;
                return;
            }

            std::vector<cv::Rect> faces;
            cascadeClassifier.detectMultiScale(videoframe, faces, 1.3, 5);
            for (int i = 0; i < faces.size(); ++i) {
                rectangle(videoframe, faces[i].tl(), faces[i].br(), 
                          cv::Scalar(50, 50, 255), 3);
                putText(videoframe, "Unknown", faces[i].tl() + cv::Point(5, 15), 
                        cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(50, 50, 255));
            }

            cvui::update();
            cvui::imshow(VIDEO_WINDOW_NAME, videoframe);
            if (cv::waitKey(20) == 27) {
                std::cout << "ESC. Close the " << VIDEO_WINDOW_NAME << "!" << std::endl;
                cv::destroyWindow(VIDEO_WINDOW_NAME);
                return;
            }
        }
    }
}
//void RecognationFace::Photo() {
//
//}
void RecognationFace::AddPhoto() {
    cv::VideoCapture videoCapture(0);
    cascadeClassifier.load(pathHaar);
    cv::Mat grayframe;

    int num = 0;
    char buffer[80];
    std::string faceId;

    std::time_t t = std::time(&t);
    std::tm* now = std::localtime(&t);
    
    strftime(buffer, 80, "%d%m%Y%H%M%S", now);
    faceId = buffer;
    std::cout << faceId << std::endl;

    while (true) {
        videoCapture.read(cameraframe);
        if (cameraframe.empty()) {
            std::cout << "Camera disconnected!" << std::endl;
            return;
        }

        cv::cvtColor(cameraframe, grayframe, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        cascadeClassifier.detectMultiScale(grayframe, faces, 1.3, 5);

        for (int i = 0; i < faces.size(); ++i) {
            num++;
            cv::imwrite(pathDataFace + "User." + faceId + "." + std::to_string(num) + ".jpg", grayframe);
            rectangle(cameraframe, faces[i].tl(), faces[i].br(), 
                      cv::Scalar(50, 50, 255), 3);
            cv::waitKey(100);
        }

        cvui::update();
        cvui::imshow(ADD_WINDOW_NAME, cameraframe);
        if (num == 20) {
            std::cout << "Close the " << ADD_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(ADD_WINDOW_NAME);
            break;
        }
    }
}
void RecognationFace::TrainnerLBP() {


}



//for (auto const& buffPath : fs::directory_iterator(pathDataFace)) {
//    std::string filePath = buffPath.path().string();
//    std::string faceIdBuff = filePath.substr(filePath.find("\\") + 1, filePath.size());
//    std::string faceId = faceIdBuff.erase(faceIdBuff.find(".jpg"));
//}07052023172836

int main() {
    RecognationFace recognationface;
    recognationface.GUI();
    return 0;
}