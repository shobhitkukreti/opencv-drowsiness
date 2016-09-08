


// OpenCV_hello.cpp : Defines the entry point for the console application.

#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "cvaux.h"
#include <Windows.h>
#include "math.h"
using namespace std;
using namespace cv; 

#include "../stasm/stasm_dll.hpp"   // for AsmSearchDll


 int yawn_count=0;
 int dist_avg=0;
 int loop1=0;
 int avg_count=0;
CvHaarClassifierCascade *cascade,*cascade_e,*cascade_mouth;
CvMemStorage            *storage; // to be used with haarobject detect

char *face_cascade="d:\\OpenCV2.1\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
char *eye_cascade="d:\\OpenCV2.1\\data\\haarcascades\\haarcascade_mcs_eyepair_big.xml";
char *mouth_cascade="d:\\OpenCV2.1\\data\\haarcascades\\haarcascade_mcs_mouth.xml";
//char *eye_cascade="d:\\opencv\\data\\haarcascades\\haarcascade_eye.xml";// try using different classifiers
//char *eye_cascade="d:\\opencv\\data\\haarcascades\\parojosG.xml";
//char *eye_cascade="d:\\opencv\\data\\haarcascades\\frontalEyes35x16.xml";
// char *eye_cascade="d:\\opencv\\data\\haarcascade_eye.xml";

// need location data of features of face. consider proportion to set region of interest.



// Mouth detection
void detectMouth( IplImage *img,CvRect *r)
{
   CvSeq *mouth;
   //mouth detecetion - set Region of Interest,
   cvSetImageROI(img,// the source image  
                 cvRect(r->x,            // x = start from leftmost 
                        r->y+(r->height *2/3), //y = a few pixels from the top 
                        r->width,        // width = same width with the face 
                        r->height/3    // height = 1/3 of face height 
                       )
                );
   // total region of mouth detected. 
   mouth = cvHaarDetectObjects(img,// the source image, with the estimated location defined  
                                cascade_mouth,      // the mouth classifier */ 
                                storage,        // memory to store the resultant sequence
                                1.15, 4, 0,     // Detection scale set at 15%, Not using any flags 
                                cvSize(25, 15)  // minimum window size
                               );

        for( int i = 0; i < (mouth ? mouth->total : 0); i++ )
        {
      
          printf("\nDected mouth times [%d]\n",mouth->total);
			CvRect *mouth_cord = (CvRect*)cvGetSeqElem(mouth, i);
          // draw a red rectangle when mouth detected
          cvRectangle(img, cvPoint(mouth_cord->x, mouth_cord->y), cvPoint(mouth_cord->x + mouth_cord->width, mouth_cord->y + mouth_cord->height),CV_RGB(255,255, 255), 1, 8, 0 );
        }
    //end mouth detection  
}

// Detecting Eyes now. There should be reset of ROI and a new ROI is chosen.


void detectEyes( IplImage *img,CvRect *r)
{
    char *eyecascade;
    CvSeq *eyes;
    int eye_detect=0;
    int meanlx,meanly,meanrx,meanry;

   //eye detection starts
  // Region of interest for eye location.
  // define location with
    cvSetImageROI(img,                    // the source image  
          cvRect
          (  // Giving rough eye location for haardetectobjects
              r->x,                 // x = start from leftmost 
              r->y + (r->height/5.5), // y = a few pixels from the top 
              r->width,        // width = same width with the face 
              r->height/3.0    // height = 1/3 of face height 
          )
      );

    //cvHaarDetectObjects will return a rectangular region within area of ROI .we will draw a rectangle/circle on position
	// obtained from function call.


      eyes = cvHaarDetectObjects( img,            // the source image, with ROI 
                                  cascade_e,      // classifier for eye
                                  storage,        // storage location for resultant. not directly using it
                                  1.15, 3, 0,     // no flags, scale factor at 15%
                                  cvSize(25, 15)  // window size
                                );
    


      printf("\Pair eyes detected are %d",eyes->total);
    
   //   meanlx = r->x+(0.3*r->width); meanly=r->y+(0.4*r->height);
    //  meanry = r->x+(0.7*r->width), meanry=r->y+(0.4*r->height);
        /* draw a rectangle for each detected eye */
        for( int i = 0; i < (eyes ? eyes->total : 0); i++ )
          {
              eye_detect++;
              // get one eye 
              CvRect *eye = (CvRect*)cvGetSeqElem(eyes, i);
              // drawing bounding box.
               
			  // draw circle in mean position of right eye and left eye
			    cvCircle(img,cvPoint(eye->x+/*5*/+(0.3*eye->width), eye->y+(0.4*eye->height)),8,CV_RGB(255,255,0),CV_FILLED); 
				cvCircle(img,cvPoint(eye->x/*+5*/+(0.7*eye->width), eye->y+(0.4*eye->height)),8,CV_RGB(255,255,0),CV_FILLED);      
			   // Line from the mean of left eye to mean of right eye
			                    //    cvRectangle(img, 
                                  //  cvPoint(eye->x+(0.3*eye->width), eye->y+(0.4*eye->height)), 
                                   // cvPoint(eye->x+(0.7*eye->width), eye->y+(0.4*eye->height)),
               //                   cvPoint(meanlx,meanly), cvPoint(meanrx,meanry),
								//	CV_RGB(0, 0, 255), 
                                 //   5, 8, 0 
                                 //  );
           }

            
}



void stasm_apply(IplImage *img)

{

    const char * image_name="wcam.jpg";
	CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5); // Initialize Font


	char buffer[30],buffer1[30];
    int nlandmarks;
    int landmarks[500]; // space for x,y coords of up to 250 landmarks
    AsmSearchDll(&nlandmarks, landmarks,
                  image_name, img->imageData, img->width, img->height,
                 1 /* is_color */, NULL /* conf_file0 */, NULL /* conf_file1 */);

  /*  if (nlandmarks == 0) {
        printf("\nError: Cannot locate landmarks in image\n", image_name);
        return -1;
    } */

	 int *p = landmarks;

//#if 0 // print the landmarks if you want
    printf("landmarks:\n");
    for (int i = 0; i < nlandmarks; i++)
      { 
//	 printf("%3d: %4d %4d\n", i, landmarks[2 * i], landmarks[2 * i + 1]);
	itoa(i,buffer,10); // Integer to String Conversion to print landmark on face
	//if(i>40 && i<64)
 //	cvPutText(img, buffer,cvPoint(landmarks[2*i],landmarks[2*i+1]), &font, CV_RGB(255,255,255)); 
    cvCircle(img,cvPoint(landmarks[51*2],landmarks[51*2+1]),4,CV_RGB(0,255,0),1,0,0);
	cvCircle(img,cvPoint(landmarks[57*2],landmarks[57*2+1]),4,CV_RGB(0,255,0),1,0,0);	

}
//#endif


int xc = landmarks[51*2]-landmarks[57*2];
int yc = landmarks[51*2+1]-landmarks[57*2+1];
printf("U lip Co-ordinate[%d][%d]\n",landmarks[51*2],landmarks[51*2+1]);
printf("D lip Co-ordinate[%d][%d]\n",landmarks[57*2],landmarks[57*2+1]);
double distance_max = pow(xc,2.0) + pow(yc,2.0);
distance_max = pow(distance_max,0.5);
printf("Distance[%d] [%d] Between lips [%f]\n",xc,yc,distance_max);



if(loop1)
{
avg_count++;
dist_avg=dist_avg+distance_max;
if(avg_count>10)
{ loop1=0; avg_count=0;}
}


if(!loop1)
{
if((dist_avg/10)>15)
{
yawn_count++;
}
dist_avg=0;
}
itoa(yawn_count,buffer1,10);
cvPutText(img, buffer1,cvPoint(300,200), &font, CV_RGB(0,100,255));
  
    // draw the landmarks on the image

//	cvPutText(CvArr* img, const char* text, CvPoint org, const CvFont* font, CvScalar color)
          
   
         cvPolyLine(img, (CvPoint **)&p, &nlandmarks, 1, 1, CV_RGB(255,0,0));
	//cvCircle(img,cvPoint(683,511),4,CV_RGB(0,255,100),1,20,0);
	//cvCircle(img,cvPoint(684,538),4,CV_RGB(0,255,10),1,20,0);

    // show the image with the landmarks

   //   cvShowImage("ASM Applied Detection", img);
      cvSaveImage("wcam.jpg",img);

}




void detectFacialFeatures( IplImage *img,IplImage *temp_img)//,int img_no)
{
    loop1=1;
    char image[100];//,msg[100];//temp_image[100];
    float m[6];
    double factor = 1;
    int w = (img)->width;
    int h = (img)->height;
    CvSeq* faces;
    CvRect *r;

        
    CvMemStorage* storage=cvCreateMemStorage(0);
    cvClearMemStorage( storage );
    
    if( cascade )
        faces = cvHaarDetectObjects(img,cascade, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 20));
    else
        printf("\nFrontal face haar cascade not loaded\n");

    printf("\nno of faces detected are %d",faces->total); // based on no. of rectangular region received from the cvhaardetectobject function call
    

    // for each face found, draw a box
    for(int i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ )
    {        
        r = ( CvRect* )cvGetSeqElem( faces, i ); // find each face in memory and returns pointer address to it, typecast to cvrect type.
        cvRectangle( img,cvPoint( r->x, r->y ),cvPoint( r->x + r->width, r->y + r->height ),
                     CV_RGB( 255, 0, 0 ), 1, 8, 0 );    
    
        printf("\n Face Location in image x=%d y=%d width=%d height=%d\n",r->x,r->y,r->width,r->height);
        
		
        detectEyes(img,r);
        // reset roi so that next cascade can do detection
        // cvResetImageROI(img);
         //detectMouth(img,r); // Not detecting mouth with Viola Jones
        // reset roi so that next cascade can do detection
		cvResetImageROI(img);
		stasm_apply(img);
	}

	  cvShowImage("Haar Cascade Detection for Eyes",img);  
	  }




int main( int argc, char** argv )
{
    //CvCapture *capture;
    IplImage  *img,*temp_img;
    int       key;

    char image[100],temp_image[100];
    
    // load the classifier 
       
    storage = cvCreateMemStorage( 0 );
    cascade = ( CvHaarClassifierCascade* )cvLoad( face_cascade, 0, 0, 0 );
    cascade_e = ( CvHaarClassifierCascade* )cvLoad( eye_cascade, 0, 0, 0 );
    cascade_mouth = ( CvHaarClassifierCascade* )cvLoad( mouth_cascade, 0, 0, 0 );

	// check if classifier loaded.
    
    if( !(cascade || cascade_e ||cascade_mouth) )
        {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
      getch();
		return -1;
        }
    
       
		int j=1;
      CvCapture *capture = cvCaptureFromCAM(0);
	  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 352); 
      cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 288); 
//	  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640); 
  //    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480); 

    
	  while(true) {
	  if( !capture ) return 1;
		        
		 img = cvQueryFrame( capture );
		//  temp_img = img;          
        if(!img)
        {
        printf("Webcam is closed or no frame capture: %s\n",image);
          break;
		 }
        
	   detectFacialFeatures(img,temp_img);  // this function will call other functions to do detections.
			 
				
				 int c = cvWaitKey(10); // exit the program if 'c' is pressed
			  if (char(c)== 'c') return 0;

		
	 
		 }
  
	cvReleaseHaarClassifierCascade( &cascade );
    cvReleaseHaarClassifierCascade( &cascade_e );
    cvReleaseHaarClassifierCascade( &cascade_mouth );
    cvReleaseMemStorage( &storage );
     cvReleaseImage(&img);
     cvReleaseImage(&temp_img);
    

    return 0;
}