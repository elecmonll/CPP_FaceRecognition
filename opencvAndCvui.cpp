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
#define MENU_VIDEO_WINDOW_NAME "Opencvui Menu Video"
#define VIDEO_WINDOW_NAME "Opencvui Video"
#define MENU_PHOTO_WINDOW_NAME "Opencvui Menu Photo"
#define PHOTO_WINDOW_NAME "Opencvui Photo"
#define MENU_ADD_WINDOW_NAME "Opencvui Menu Add Photo"
#define ADD_WINDOW_NAME "Opencvui Add Photo"

namespace fs = std::filesystem;
namespace fc = cv::face;

class RecognitionFace {
public:
    cv::CascadeClassifier cascadeClassifier;
    cv::Ptr<fc::LBPHFaceRecognizer> model =
        fc::LBPHFaceRecognizer::create();

    cv::Mat guiframe = cv::Mat(310, 250, CV_8UC3);
    cv::Mat cameraframe = cv::Mat(400, 500, CV_8UC3);
    cv::Mat videoframe = cv::Mat(400, 500, CV_8UC3);
    cv::Mat menuphotoframe = cv::Mat(160, 250, CV_8UC3);
    cv::Mat menuvideoframe = cv::Mat(160, 250, CV_8UC3);
    cv::Mat menuaddphotoframe = cv::Mat(210, 250, CV_8UC3);

    std::vector<std::string> Id;
    std::string pathDataFace = "C:/Source/dataFace/";
    std::string pathDataVideo = "C:/Source/dataVideo/";
    std::string pathPhoto = "C:/Source/dataImg/";

public:
    void GUI();
    void Camera();
    void MenuVideo();
    void Video(std::string);
    void MenuPhoto();
    void Photo(std::string);
    void MenuAddPhoto();
    void AddPhoto(cv::VideoCapture);
    void TrainnerLBPH();
};
void RecognitionFace::GUI() {
    cv::namedWindow(MAIN_WINDOW_NAME);
    cvui::init(MAIN_WINDOW_NAME);

    while (true) {
        guiframe = cv::Scalar(49, 52, 49);
        if (cvui::button(guiframe, 10, 10, 230, 40, "Camera Recognizer")) {
            std::cout << "Camera Recognizer!" << std::endl;
            Camera();
            GUI();
        }
        if (cvui::button(guiframe, 10, 60, 230, 40, "Video Recognizer")) {
            std::cout << "Video Recognizer!" << std::endl;
            MenuVideo();
            GUI();
        }
        if (cvui::button(guiframe, 10, 110, 230, 40, "Photo Recognizer")) {
            std::cout << "Photo Recognizer!" << std::endl;
            MenuPhoto();
            GUI();
        }
        if (cvui::button(guiframe, 10, 160, 230, 40, "Add photo")) {
            std::cout << "Add photo to the photo database!" << std::endl;
            MenuAddPhoto();
            GUI();
        }
        if (cvui::button(guiframe, 10, 210, 230, 40, "Learn")) {
            std::cout << "Learn LBPH!" << std::endl;
            TrainnerLBPH();
            GUI();
        }
        if (cvui::button(guiframe, 10, 260, 230, 40, "Exit")) {
            std::cout << "Exit from " << MAIN_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MAIN_WINDOW_NAME);
            break;
        }

        cvui::update(MAIN_WINDOW_NAME);
        cv::imshow(MAIN_WINDOW_NAME, guiframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the " << MAIN_WINDOW_NAME << "!" << std::endl;
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
        for (int i = 0; i < faces.size(); ++i) {
            model->predict(grayframe, predictedLabel, confidence);
            rectangle(cameraframe, faces[i].tl(), faces[i].br(),
                cv::Scalar(50, 50, 255), 3);
            if (confidence >= 70) {
                putText(cameraframe, std::to_string(predictedLabel) + "   " + std::to_string(confidence), faces[i].tl() +
                    cv::Point(5, 15), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(50, 50, 255));
            }
        }
                
        cvui::update(CAMERA_WINDOW_NAME);
        cvui::imshow(CAMERA_WINDOW_NAME, cameraframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the " << CAMERA_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(CAMERA_WINDOW_NAME);
            break;
        }
    }
}
void RecognitionFace::MenuVideo() {
    cv::namedWindow(MENU_VIDEO_WINDOW_NAME);
    cvui::init(MENU_VIDEO_WINDOW_NAME);

    while (true) {
        menuvideoframe = cv::Scalar(49, 52, 49);
        if (cvui::button(menuvideoframe, 10, 10, 230, 40, "Path to the video")) {
            std::cout << "Path to the video!" << std::endl;
            cv::destroyWindow(MENU_VIDEO_WINDOW_NAME);
            std::string buf;
            std::cout << "Path to the video: ";
            std::cin >> buf;
            Video(buf);
            break;
        }
        if (cvui::button(menuvideoframe, 10, 60, 230, 40, "Use video database")) {
            std::cout << "Use video database!" << std::endl;
            cv::destroyWindow(MENU_VIDEO_WINDOW_NAME);
            for (auto const& buffPath : fs::directory_iterator(pathDataVideo)) {
                std::string buf = buffPath.path().string();
                std::cout << "Path: " << buf << std::endl;
                Video(buf);
            }
            break;
        }
        if (cvui::button(menuvideoframe, 10, 110, 230, 40, "Exit")) {
            std::cout << "Exit from " << MENU_VIDEO_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MENU_VIDEO_WINDOW_NAME);
            break;
        }

        cvui::update(MENU_VIDEO_WINDOW_NAME);
        cv::imshow(MENU_VIDEO_WINDOW_NAME, menuvideoframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the " << MENU_VIDEO_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MENU_VIDEO_WINDOW_NAME);
            break;
        }
    }
}
void RecognitionFace::Video(std::string buf) {
    cv::VideoCapture videoCapture(buf);
    while (true) {
        videoCapture.read(videoframe);
        if (videoframe.empty()) {
            std::cout << "ERROR! Video disconnected!" << std::endl;
            return;
        }

        cv::Mat grayframe;
        cv::cvtColor(videoframe, grayframe, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        cascadeClassifier.detectMultiScale(grayframe, faces, 1.3, 5);

        int predictedLabel = -1;
        double confidence = 0.0;
        for (int i = 0; i < faces.size(); ++i) {
            model->predict(grayframe, predictedLabel, confidence);
            rectangle(videoframe, faces[i].tl(), faces[i].br(),
                cv::Scalar(50, 50, 255), 3);
            if (confidence >= 70) {
                putText(videoframe, std::to_string(predictedLabel) + "   " + std::to_string(confidence), faces[i].tl() +
                    cv::Point(5, 15), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(50, 50, 255));
            }
        }

        cvui::update(VIDEO_WINDOW_NAME);
        cvui::imshow(VIDEO_WINDOW_NAME, videoframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the " << VIDEO_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(VIDEO_WINDOW_NAME);
            break;
        }
    }
}
void RecognitionFace::MenuPhoto() {
    cv::namedWindow(MENU_PHOTO_WINDOW_NAME);
    cvui::init(MENU_PHOTO_WINDOW_NAME);

    while (true) {
        menuphotoframe = cv::Scalar(49, 52, 49);
        if (cvui::button(menuphotoframe, 10, 10, 230, 40, "Path to the photo")) {
            std::cout << "Path to the photo!" << std::endl;
            cv::destroyWindow(MENU_PHOTO_WINDOW_NAME);
            std::string buf;
            std::cout << "Path to the photo: ";
            std::cin >> buf;
            Photo(buf);
            break;
        }
        if (cvui::button(menuphotoframe, 10, 60, 230, 40, "Use photo database")) {
            std::cout << "Use photo database!" << std::endl;
            cv::destroyWindow(MENU_PHOTO_WINDOW_NAME);
            for (auto const& buffPath : fs::directory_iterator(pathPhoto)) {
                std::string buf = buffPath.path().string();
                std::cout << "Path: " << buf << std::endl;
                Photo(buf);
            }
            break;
        }
        if (cvui::button(menuphotoframe, 10, 110, 230, 40, "Exit")) {
            std::cout << "Exit from " << MENU_PHOTO_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MENU_PHOTO_WINDOW_NAME);
            break;
        }

        cvui::update(MENU_PHOTO_WINDOW_NAME);
        cv::imshow(MENU_PHOTO_WINDOW_NAME, menuphotoframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the" << MENU_PHOTO_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MENU_PHOTO_WINDOW_NAME);
            break;
        }
    }
}
void RecognitionFace::Photo(std::string buf) {
    cv::Mat photoframe = cv::imread(buf);
    cv::Mat grayframe, resizeframe;

    cv::resize(photoframe, resizeframe, cv::Size(photoframe.cols / 4, photoframe.rows / 4), (0, 0), (0, 0), 3);
    cv::cvtColor(resizeframe, grayframe, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    cascadeClassifier.detectMultiScale(grayframe, faces, 1.3, 5);

    int predictedLabel = -1;
    double confidence = 0.0;
    for (int i = 0; i < faces.size(); ++i) {
        model->predict(grayframe, predictedLabel, confidence);
        rectangle(resizeframe, faces[i].tl(), faces[i].br(),
            cv::Scalar(50, 50, 255), 3);
        if (confidence >= 70) {
            putText(resizeframe, std::to_string(predictedLabel) + "   " + std::to_string(confidence), faces[i].tl() +
                    cv::Point(5, 15), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(50, 50, 255));
        }
    }
    
    cvui::update(PHOTO_WINDOW_NAME);
    cvui::imshow(PHOTO_WINDOW_NAME, resizeframe);
    cv::waitKey(1000);
    cv::destroyWindow(PHOTO_WINDOW_NAME);
    return;
}
void RecognitionFace::MenuAddPhoto() {
    cv::namedWindow(MENU_ADD_WINDOW_NAME);
    cvui::init(MENU_ADD_WINDOW_NAME);

    while (true) {
        menuaddphotoframe = cv::Scalar(49, 52, 49);
        if (cvui::button(menuaddphotoframe, 10, 10, 230, 40, "Path to the photo")) {
            std::cout << "Path to the photo!" << std::endl;
            cv::destroyWindow(MENU_ADD_WINDOW_NAME);
            std::string buf;
            std::cout << "Path to the photo: ";
            std::cin >> buf;
            cv::VideoCapture videoCapture(buf);
            AddPhoto(videoCapture);
            break;
        }
        if (cvui::button(menuaddphotoframe, 10, 60, 230, 40, "Use photo database")) {
            std::cout << "Use photo database!" << std::endl;
            cv::destroyWindow(MENU_ADD_WINDOW_NAME);
            for (auto const& buffPath : fs::directory_iterator(pathPhoto)) {
                std::string buf = buffPath.path().string();
                cv::VideoCapture videoCapture(buf);
                AddPhoto(videoCapture);
            }
            break;
        }
        if (cvui::button(menuaddphotoframe, 10, 110, 230, 40, "Use camera")) {
            std::cout << "Use camera!" << std::endl;
            cv::destroyWindow(MENU_ADD_WINDOW_NAME);
            cv::VideoCapture videoCapture(0);
            AddPhoto(videoCapture);
            break;
        }
        if (cvui::button(menuaddphotoframe, 10, 160, 230, 40, "Exit")) {
            std::cout << "Exit from " << MENU_ADD_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MENU_ADD_WINDOW_NAME);
            break;
        }

        cvui::update(MENU_ADD_WINDOW_NAME);
        cv::imshow(MENU_ADD_WINDOW_NAME, menuaddphotoframe);
        if (cv::waitKey(20) == 27) {
            std::cout << "ESC. Close the " << MENU_ADD_WINDOW_NAME << "!" << std::endl;
            cv::destroyWindow(MENU_ADD_WINDOW_NAME);
            break;
        }
    }
}
void RecognitionFace::AddPhoto(cv::VideoCapture videoCapture) {
    cv::Mat grayframe, bufframe;

    int num = 0;
    char buffer[80];
    std::string faceId;

    std::time_t t = std::time(&t);
    std::tm* now = std::localtime(&t);
    
    strftime(buffer, 80, "%j%d%M%S", now);
    faceId = buffer;
    std::cout << "ID: " << faceId << std::endl;

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

        cvui::update(ADD_WINDOW_NAME);
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
void RecognitionFace::TrainnerLBPH() {
    int buf;
    std::vector<int> faceIds;
    std::vector<cv::Mat> faceSamples;

    for (auto const& buffPath : fs::directory_iterator(pathDataFace)) {
        std::string filePath = buffPath.path().string();
        std::string faceIdBuff = filePath.substr(filePath.find(".") + 1, filePath.size());
        std::string faceIdBuff1 = faceIdBuff.erase(faceIdBuff.find("_"), 2);
        std::string faceId = faceIdBuff1.erase(faceIdBuff.find(".jpg"));
        std::istringstream(faceId) >> buf;
        std::cout << "ID: " << buf << std::endl;

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
    recognitionface.TrainnerLBPH();
    recognitionface.GUI();
    return 0;
}