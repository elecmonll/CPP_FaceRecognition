#pragma warning(disable : 4996)
#define CVUI_IMPLEMENTATION
#include "cvui-2.7.0/cvui.h"

#include <iostream>
#include <string>
#include <sstream>
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

class RecognitionFace {
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

public:
    void GUI();
    void Camera();
    void Video();
    void Photo();
    void AddPhoto();
    void TrainnerLBP();

};
void RecognitionFace::GUI() {
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
void RecognitionFace::Camera() {
    cv::VideoCapture videoCapture(0);
    while (true) {
        videoCapture.read(cameraframe);
        if (cameraframe.empty()) {
            std::cout << "ERROR! Camera disconnected!" << std::endl;
            return;
        }

        cv::Mat grayframe;
        cv::cvtColor(cameraframe, grayframe, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        cascadeClassifier.detectMultiScale(grayframe, faces, 1.3, 5);

        int predictedLabel = -1;
        double confidence = 0.0;
        TrainnerLBP();
        for (int i = 0; i < faces.size(); ++i) {
            model->predict(grayframe, predictedLabel, confidence);
            rectangle(cameraframe, faces[i].tl(), faces[i].br(),
                cv::Scalar(50, 50, 255), 3);
            if (confidence >= 70) {
                putText(cameraframe, std::to_string(predictedLabel) + "   " + std::to_string(confidence), faces[i].tl() +
                    cv::Point(5, 15), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(50, 50, 255));
            }
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
void RecognitionFace::Video() {
    for (auto const& buffPath : fs::directory_iterator(pathDataVideo)) {
        std::string filePath = buffPath.path().string();
        
        cv::namedWindow(VIDEO_WINDOW_NAME);
        cv::VideoCapture videoCapture(filePath);
        
        while (true) {
            videoCapture.read(videoframe);
            if (videoframe.empty()) {
                std::cout << "ERROR! Video disconnected!" << std::endl;
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
void RecognitionFace::AddPhoto() {
    cv::VideoCapture videoCapture(0);
    cv::Mat grayframe, bufframe;

    int num = 0;
    char buffer[80];
    std::string faceId;

    std::time_t t = std::time(&t);
    std::tm* now = std::localtime(&t);
    
    strftime(buffer, 80, "%j%d%M%S", now);
    faceId = buffer;
    std::cout << faceId << std::endl;

    while (true) {
        videoCapture.read(cameraframe);
        if (cameraframe.empty()) {
            std::cout << "ERROR! Camera disconnected!" << std::endl;
            return;
        }

        cv::cvtColor(cameraframe, grayframe, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        cascadeClassifier.detectMultiScale(grayframe, faces);
        bufframe = grayframe(faces[0]);

        for (int i = 0; i < faces.size(); ++i) {
            num++;
            cv::imwrite(pathDataFace + "User." + faceId + "_" + std::to_string(num) + ".jpg", bufframe); //"_" + std::to_string(num)
            cv::waitKey(100);
        }

        cvui::update();
        cvui::imshow(ADD_WINDOW_NAME, cameraframe);
        if (num == 1) {
            std::cout << "Close the " << ADD_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(ADD_WINDOW_NAME);
            break;
        }
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the " << ADD_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(ADD_WINDOW_NAME);
            break;
        }
    }
}
void RecognitionFace::TrainnerLBP() {
    std::vector<int> faceIds;
    std::vector<cv::Mat> faceSamples;

    for (auto const& buffPath : fs::directory_iterator(pathDataFace)) {
        std::string filePath = buffPath.path().string();
        std::string faceIdBuff = filePath.substr(filePath.find(".") + 1, filePath.size());
        std::string faceIdBuff1 = faceIdBuff.erase(faceIdBuff.find("_"), 2);
        std::string faceId = faceIdBuff1.erase(faceIdBuff.find(".jpg"));
        int buf;
        std::istringstream(faceId) >> buf;
        std::cout << buf << std::endl;

        cv::Mat srcfaceframe, grayfaceframe;
        srcfaceframe = cv::imread(buffPath.path().string());
        cv::cvtColor(srcfaceframe, grayfaceframe, cv::COLOR_BGR2GRAY);

        faceIds.emplace_back(buf);
        faceSamples.emplace_back(grayfaceframe);
        model->train(faceSamples, faceIds);
    }
    std::cout << "Trainning complete!" << std::endl;
}

int main() {
    RecognitionFace recognitionface;
    recognitionface.cascadeClassifier.load("xml/haarcascade_frontalface_default.xml");
    recognitionface.GUI();
    return 0;
}