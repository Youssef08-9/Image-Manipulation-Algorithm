// START OF THE ALGORITHM (main.cpp)

// Pre-built Binaries from OpenCV, iostream, stdio.h, and namespaces.
#include <iostream>
#include <stdio.h>
#include <opencv2/core/utility.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;
using namespace samples;

// Setting up variables for the Canny Detector.
Mat src, src_gray;
Mat dst, detected_edges;

int lowThreshold = 0; // Lower threshold of 0.
const int max_lowThreshold = 100; // Maximum value for the lower threshold of 100.
const int ratio = 3; // Ratio for lower:upper threshold of 3:1.
const int kernel_size = 3; // Kernel size of 3 (for the Sobel operations to be performed internally by the Canny function in the main body).
const char* window_name = "Edge Map"; // Setting up the window for the Edge Detector.

// Setting up the Canny Detector function void pointer.
static void CannyThreshold(int, void*)
{
	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * 3, kernel_size);
	dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);
	imshow(window_name, dst); // Displays the Canny Detector.
}

// Main body of the algorithm.

int main(int argc, const char** argv) { 

	// Test: To display the original image (Step 1).
	Mat CanadaMap = imread("C:/Users/AxesW/source/repos/Assignment 4/x64/Canada-Map.jpg"); // Reads the image.
	imshow("Canada", CanadaMap); // Displays the image.
	// namedWindow("Image with No Borders", WINDOW_NORMAL);
	// resizeWindow("Image with No Borders", 600, 480);
	// imshow("Image with No Borders", CanadaMap);
	waitKey(10000); // To display the image for a set duration of time.

	// Objective: Turn the image into grayscale (Step 2)
	CanadaMap = imread("C:/Users/AxesW/source/repos/Assignment 4/x64/Canada-Map.jpg", IMREAD_GRAYSCALE); // Converts the image to grayscale.
	imshow("Canada", CanadaMap); // Displays the image.
	waitKey(10000); // To display the image for a set duration of time.

	// Objective: Detect the borders between provinces, territories, lakes, etc. (Canny Detector) (Step 3)
	CommandLineParser parser(argc, argv, "{@input | Canada-Map.jpg | input image}"); // Load the source image for the detector.
	src = imread("C:/Users/AxesW/source/repos/Assignment 4/x64/Canada-Map.jpg", IMREAD_COLOR); // Load an image.
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}
	dst.create(src.size(), src.type()); // Create a matrix of the same type and size for src (to be dst).
	cvtColor(src, src_gray, COLOR_BGR2GRAY); // Converts the image to grayscale.
	namedWindow(window_name, WINDOW_AUTOSIZE); // Create a window for the detector to display the results.
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold); // Create a trackbar for the detector for the user to enter the lower thresdhold.
	CannyThreshold(0, 0); // Show the image for the Canny Detector.
	waitKey(10000); // Wait for a key stroke; the same function arranges events processing.

	// Objective: Segment the provinces of the grayscale image (Image Segmentation: Watershed Algorithm) (Step 4)
	// Objective: Show Canada as a whole (Step 5)
	Mat input_image = imread("C:/Users/AxesW/source/repos/Assignment 4/x64/Canada-Map.jpg"); // Reads the image.
	Mat grayscale_image; // Creating the grayscale matrix.
	cvtColor(input_image, grayscale_image, COLOR_BGR2GRAY); // Converts the image to grayscale.

	Mat Binary; // Create a Matrix for Binary.
	threshold(grayscale_image, Binary, 100, 255, THRESH_BINARY_INV + THRESH_OTSU);

	Mat Truncate; // Create a Matrix for Truncate.
	threshold(grayscale_image, Truncate, 150, 255, THRESH_TRUNC + THRESH_OTSU); 

	imshow("Original Image", input_image); // Displays the original image of Canada Map.
	imshow("Grayscale Image", grayscale_image); // Displays the grayscale image of Canada Map.
	imshow("Binary", Binary); // Displays the Binary (Canada as a whole) from the Canada-Map.jpg.
	imshow("Truncate", Truncate); // Displays the Truncate (Image Segmentation) from the Canada-Map.jpg for the grayscale map.

	waitKey(10000); // To display the image for a set duration of time.

}

// END OF THE ALGORITHM (main.cpp)
