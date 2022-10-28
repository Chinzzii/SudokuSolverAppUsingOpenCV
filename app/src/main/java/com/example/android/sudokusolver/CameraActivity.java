package com.example.android.sudokusolver;

import androidx.appcompat.app.AppCompatActivity;

import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MenuItem;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import com.googlecode.tesseract.android.TessBaseAPI;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    JavaCameraView javaCameraView;
    private Mat mRGBA, mRGBAT;
    private Mat mIntermediateMat;
    private Mat mGRAY;
    Mat cropped;
    TessBaseAPI tessBaseApi;
    private static String TAG = "CameraView";
    ImageView captureButton;
    private static final String DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/SudokuSolver2/app/src/assets/";
    private static final String lang = "eng";

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(CameraActivity.this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    javaCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_camera_view);
        captureButton = (ImageView) findViewById(R.id.captureButtom);
        javaCameraView = (JavaCameraView) findViewById(R.id.cameraView);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(CameraActivity.this);

        captureButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                capture();
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRGBA = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGRAY = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRGBA.release();
        mGRAY.release();
        mIntermediateMat.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRGBA = inputFrame.rgba();
        mRGBAT = mRGBA.t();
        Core.flip(mRGBAT, mRGBAT, 1);
        Imgproc.resize(mRGBAT, mRGBAT, mRGBA.size());

        Mat grayMat= inputFrame.gray();
        Mat blurMat = new Mat();
        Imgproc.GaussianBlur(grayMat, blurMat, new Size(5,5), 0);
        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(blurMat, thresh, 255,1,1,11,2);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hier = new Mat();
        Imgproc.findContours(thresh, contours, hier, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        hier.release();

        MatOfPoint2f biggest = new MatOfPoint2f();
        double max_area = 0;
        for (MatOfPoint i : contours) {
            double area = Imgproc.contourArea(i);
            if (area > 100) {
                MatOfPoint2f m = new MatOfPoint2f(i.toArray());
                double peri = Imgproc.arcLength(m, true);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(m, approx, 0.02 * peri, true);
                if (area > max_area && approx.total() == 4) {
                    biggest = approx;
                    max_area = area;
                }
            }
        }

        // find the outer box
        Mat displayMat = inputFrame.rgba();
        Point[] points = biggest.toArray();
        cropped = new Mat();
        int t = 3;
        if (points.length >= 4) {
            // draw the outer box
            Imgproc.line(displayMat, new Point(points[0].x, points[0].y), new Point(points[1].x, points[1].y), new Scalar(255, 0, 0), 2);
            Imgproc.line(displayMat, new Point(points[1].x, points[1].y), new Point(points[2].x, points[2].y), new Scalar(255, 0, 0), 2);
            Imgproc.line(displayMat, new Point(points[2].x, points[2].y), new Point(points[3].x, points[3].y), new Scalar(255, 0, 0), 2);
            Imgproc.line(displayMat, new Point(points[3].x, points[3].y), new Point(points[0].x, points[0].y), new Scalar(255, 0, 0), 2);
            // crop the image
            Rect R = new Rect(new Point(points[0].x - t, points[0].y - t), new Point(points[2].x + t, points[2].y + t));
            cropped = new Mat(displayMat, R);
        }

        return displayMat;

    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {

    }

    @Override
    public void onBackPressed() {
        finish();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV is Connected Successfully!");
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
        else {
            Log.d(TAG,"OpenCV not Working.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
    }


    public void capture(){
        if (cropped.width() < 1 || cropped.height() < 1) {
            finish();
        }

        javaCameraView.setVisibility(View.GONE);
        ImageView iv = (ImageView) findViewById(R.id.solve_img);
        iv.setVisibility(View.VISIBLE);

        // initialize the TessBase
        tessBaseApi = new TessBaseAPI();
        tessBaseApi.init(DATA_PATH, lang);
        tessBaseApi.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_BLOCK);
        tessBaseApi.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "123456789");
        tessBaseApi.setVariable("classify_bin_numeric_mode", "1");

        Mat output = cropped.clone();

        int SUDOKU_SIZE = 9;
        int IMAGE_WIDTH = output.width();
        int IMAGE_HEIGHT = output.height();
        double PADDING = IMAGE_WIDTH/25;
        int HSIZE = IMAGE_HEIGHT/SUDOKU_SIZE;
        int WSIZE = IMAGE_WIDTH/SUDOKU_SIZE;

        int[][] sudos = new int[SUDOKU_SIZE][SUDOKU_SIZE];

        // Divide the image to 81 small grid and do the digit recognition
        for (int y = 0, iy = 0; y < IMAGE_HEIGHT - HSIZE ; y+= HSIZE,iy++) {
            for (int x = 0, ix = 0; x < IMAGE_WIDTH - WSIZE; x += WSIZE, ix++) {
                sudos[iy][ix] = 0;
                int cx = (x + WSIZE / 2);
                int cy = (y + HSIZE / 2);
                Point p1 = new Point(cx - PADDING, cy - PADDING);
                Point p2 = new Point(cx + PADDING, cy + PADDING);
                Rect R = new Rect(p1, p2);
                Mat digit_cropped = new Mat(output, R);
                Imgproc.GaussianBlur(digit_cropped,digit_cropped,new Size(5,5),0);
                Imgproc.rectangle(output, p1, p2, new Scalar(0, 0, 0));
                Bitmap digit_bitmap = Bitmap.createBitmap(digit_cropped.cols(), digit_cropped.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(digit_cropped, digit_bitmap);

                tessBaseApi.setImage(digit_bitmap);
                String recognizedText = tessBaseApi.getUTF8Text();
                if (recognizedText.length() == 1) {
                    sudos[iy][ix] = Integer.valueOf(recognizedText);
                }
                Imgproc.putText(output, recognizedText, new Point(cx, cy), 1, 3.0f, new Scalar(0));
                tessBaseApi.clear();
            }
            Log.i("testing",""+ Arrays.toString(sudos[iy]));
        }

        //Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2RGBA);

        tessBaseApi.end();

        // Copy the captured array
        int[][] test_sudo = Arrays.copyOf(sudos, sudos.length);

        // make a copy of the captured array
        int[][] temp = new int[9][9];
        for (int i = 0; i < 9; i++) {
            for (int y = 0; y < 9; y++) {
                temp[i][y] = test_sudo[i][y];
            }
        }

        // Solve the puzzle
        if(solve(test_sudo)) {
            for (int y = 0, iy = 0; y < IMAGE_HEIGHT - HSIZE; y += HSIZE, iy++) {
                for (int x = 0, ix = 0; x < IMAGE_WIDTH - WSIZE; x += WSIZE, ix++) {
                    if (temp[iy][ix] == 0) {
                        int cx = (x + WSIZE / 2);
                        int cy = (y + HSIZE / 2);
                        Point p = new Point(cx, cy);
                        Imgproc.putText(output, test_sudo[iy][ix] + "", p, 1, 3.0f, new Scalar(255));
                    }
                }
            }
        }
    }


    public static boolean solve(int board[][]) {
        int row = -1;
        int col = -1;
        boolean isEmpty = true;
        for(int i=0; i<9; i++) {
            for(int j=0; j<9; j++) {
                if(board[i][j]==0) {
                    row = i;
                    col = j;
                    isEmpty = false;
                    break;
                }
            }
            if(!isEmpty) {
                break;
            }
        }
        if(isEmpty) {
            return true;
        }

        for(int num=1; num<10; num++) {
            if(isSafe(board, row, col, num)) {
                board[row][col]=num;
                if(solve(board)) {
                    return true;
                }
                else {
                    board[row][col]=0;
                }
            }
        }

        return false;
    }

    public static boolean isSafe(int board[][], int row, int col, int num) {

        for(int i=0; i<9; i++) {
            if(board[row][i]==num) {
                return false;
            }
        }

        for(int i=0; i<9; i++) {
            if(board[i][col]==num) {
                return false;
            }
        }

        int rowStart = row - (row % 3);
        int colStart = col - (col % 3);
        for(int i = rowStart; i < rowStart+3; i++) {
            for(int j = colStart; j < colStart+3; j++) {
                if(board[i][j]==num) {
                    return false;
                }
            }
        }

        return true;
    }

}