package com.example.mis.opencv;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;
    private CascadeClassifier    faceCascade;
    private CascadeClassifier    eyeCascade;
    private MatOfRect            detectedFaces;
    private MatOfRect            detectedEyes;
    private Point                noseTipPosition;
    private static Scalar red;
    private static Scalar green;
    private static Scalar blue;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();

                    red   = new Scalar(255, 0, 0);
                    green = new Scalar(0, 255, 0);
                    blue  = new Scalar(0, 0, 255);

                    //Log.e("onManagerConnected", initAssetFile("haarcascade_frontalface_default.xml"));
                    faceCascade     = new CascadeClassifier(initAssetFile("haarcascade_frontalface_default.xml"));
                    eyeCascade      = new CascadeClassifier(initAssetFile("haarcascade_eye.xml"));
                    detectedFaces   = new MatOfRect();
                    detectedEyes    = new MatOfRect();
                    noseTipPosition = new Point();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCameraIndex(1); // swapping to the frontal face camera

        mOpenCvCameraView.setCvCameraViewListener(this);

        Toast.makeText(getApplicationContext(), "Use in Landscape Mode for good results!:)", Toast.LENGTH_LONG).show();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat col  = inputFrame.rgba();
        Core.flip(col, col, 1); // flips the camera visualization in portrait mode

        faceCascade.detectMultiScale(col, detectedFaces);
        for (Rect face: detectedFaces.toList()) {
            Log.e("FaceDetection", "faceWidth = " + face.width + " faceHeight = " + face.height + " faceArea = " + face.area());
            if (face.area() > 20000) { // threshold to detect only foreground faces and delete some "noise-data"

                //draw a rectangle arounf the face (for dubugging purposes)
                Point pointStart = new Point(face.x, face.y);
                Point pointEnd = new Point(face.x + face.width, face.y + face.height);
                Imgproc.rectangle(col, pointStart, pointEnd, blue, 2);

                Mat faceFrame = col.colRange(face.x, face.x + face.width).rowRange(face.y, face.y + face.height);
                eyeCascade.detectMultiScale(faceFrame, detectedEyes);
                Rect[] eyes = filter(detectedEyes.toList(), faceFrame); // filter noise and make type conversions
                if(eyes != null) {
                    for (Rect eye : eyes) {
                        //draw a rectangle arounf the face (for dubugging purposes)
                        pointStart = new Point(eye.x, eye.y);
                        pointEnd = new Point(eye.x + eye.width, eye.y + eye.height);
                        Imgproc.rectangle(faceFrame, pointStart, pointEnd, green, 2);
                    }
                    Log.e(TAG, eyes.length + " eyes detected");
                    if (eyes.length == 2) {
                        Log.d(TAG, "Whole frontal face detected!");
                        // the average of the position of the left extreme of one eye and the right extreme
                        // of the other give us a good estimation of the position of the center of the eyes,
                        // not depending on the detecting order
                        double xMidPoint = (eyes[0].x + (eyes[1].x + eyes[1].width)) / 2;
                        // average of the y position of the average y position of each eye
                        double yMidPoint = ((eyes[0].y + eyes[0].height) / 2) + ((eyes[1].y + eyes[1].height) / 2) / 2;

                        double[] noseTipCoords = {xMidPoint, (yMidPoint - face.y + face.height) / 2};
                        noseTipPosition.set(noseTipCoords);

                        double eyesMidPointDist = Math.abs(
                                ((eyes[0].x + eyes[0].width) / 2) - ((eyes[1].x + eyes[1].width) / 2)
                        );
                        int noseRadius = (int) eyesMidPointDist / 2;
                        //draw the red nose
                        Imgproc.circle(faceFrame, noseTipPosition, noseRadius, red, -1); // thickness -1 fills the circle
                    }
                }
            }
        }

        return col;
    }

    public String initAssetFile(String filename)  {
        File file = new File(getFilesDir(), filename);
        Log.e(TAG,"processing local file: "+file.toString());
        if (!file.exists()) try {
            InputStream is = getAssets().open(filename);
            OutputStream os = new FileOutputStream(file);
            byte[] data = new byte[is.available()];
            is.read(data); os.write(data); is.close(); os.close();
        } catch (IOException e) { e.printStackTrace(); }
        Log.e(TAG,"prepared local file: "+filename);
        return file.getAbsolutePath();
    }

    private Rect[] filter(List<Rect> eyesList, Mat face){
        ArrayList<Rect> eyes = new ArrayList<>(eyesList); // necessary to remove while iterating on the eyesList
        for(Rect eye : eyesList){
            if (eye.area() < face.width() * face.height() / 25){ // the eye dimension is less then the 1/25-th of the face
                eyes.remove(eye);
            }
        }
        if (eyes.size() != 2){
            return null;
        }else{
            // inspired to https://stackoverflow.com/questions/5374311/convert-arrayliststring-to-string-array
            Rect[] actualEyes = new Rect[2];
            actualEyes = eyes.toArray(actualEyes); // type conversion
            return actualEyes;
        }
    }
}
