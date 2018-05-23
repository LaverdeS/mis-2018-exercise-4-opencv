package com.example.mis.opencv;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
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
    private static Scalar blue;
    private static Scalar red;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();

                    blue = new Scalar(0,0,255);
                    red = new Scalar(255, 0, 0);

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

        mOpenCvCameraView.setCvCameraViewListener(this);
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

        //return inputFrame.rgba();
        /*
        Mat col  = inputFrame.rgba();
        Rect foo = new Rect(new Point(100,100), new Point(200,200));
        Imgproc.rectangle(col, foo.tl(), foo.br(), new Scalar(0, 0, 255), 3);
        return col;
        */

        //Mat gray = inputFrame.gray();
        Mat col  = inputFrame.rgba();

        //Mat tmp = gray.clone();
        //Mat tmp2 = col.clone();
        //Imgproc.Canny(gray, tmp, 80, 100);
        //Imgproc.cvtColor(tmp, col, Imgproc.COLOR_GRAY2RGBA, 4);

        //Log.e("onCameraFrame", detectedFaces.toString());
        faceCascade.detectMultiScale(col, detectedFaces);
        for (Rect face: detectedFaces.toList()) {
            //Log.e("FaceDetection", "faceWidth = " + face.width + " faceHeight = " + face.height);
            if (face.area() > 1000) {
                Point pointStart = new Point(face.x, face.y);
                Point pointEnd = new Point(face.x + face.width, face.y + face.height);
                Imgproc.rectangle(col, pointStart, pointEnd, blue, 2);
                eyeCascade.detectMultiScale(col, detectedEyes);
                Rect[] eyes = detectedEyes.toArray();
                if (eyes.length == 2){
                    Log.d(TAG, "Whole frontal face detected!");
                    // the average of the position of the left extreme of one eye and the right extreme
                    // of the other give us a good estimation of the position of the center of the eyes,
                    // not depending on the detecting order
                    double xMidPoint = (eyes[0].x + (eyes[1].x + eyes[1].width) ) / 2;
                    // average of the y position of the average y position of each eye
                    double yMidPoint = ((eyes[0].y + eyes[0].height) / 2) + ((eyes[1].y + eyes[1].height) / 2) / 2;
                    double[] noseTipCoords = {xMidPoint, (yMidPoint - face.y) / 2};
                    noseTipPosition.set(noseTipCoords);

                    double eyesMidPointDist = ((eyes[0].x + eyes[0].width) / 2) - ((eyes[1].x + eyes[1].width) / 2);
                    int noseRadius = (int) eyesMidPointDist / 2;

                    Imgproc.circle(col, noseTipPosition, noseRadius, red);
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
}
