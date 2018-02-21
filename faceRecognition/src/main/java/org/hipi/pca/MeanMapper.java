package org.hipi.pca;

import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.opencv.OpenCVMatWritable;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;

import java.io.IOException;

import org.hipi.util.util;

public class MeanMapper extends
        Mapper<HipiImageHeader, FloatImage, IntWritable, OpenCVMatWritable> {
    @Override
    public void map(HipiImageHeader header, FloatImage image, Context context)
            throws IOException, InterruptedException {
        /* Perform conversion to OpenCV */
        Mat cvImage = new Mat(image.getHeight(), image.getWidth(), opencv_core.CV_32FC1);
        if(!util.convertFloatImageToGrayscaleMat(image, cvImage)) {
            System.out.println("MeanMapper is skipping image with invalid color space.");
            return;
        }

        /* Compute mean using OpenCV */
        int N = util.patchSize;  //patch dimensions (N X N)
        Mat mean = new Mat(N, N, opencv_core.CV_32FC1, new Scalar(0.0));
        int iMax = 10, jMax = 10;  // specify number of patches to use in mean patch computation (iMax * jMax patches)

        // collect patches and add their values to mean patch mat
        for (int i = 0; i < iMax; i++) {
            int x = ((cvImage.cols() - N) * i) / iMax;
            for (int j = 0; j < jMax; j++) {
                int y = ((cvImage.rows() - N) * j) / jMax;
                Mat patch = cvImage.apply(new Rect(x, y, N, N));
                opencv_core.add(patch, mean, mean);
            }
        }

        // scale mean patch mat based on total number of patches
        mean = opencv_core.divide(mean, ((double) (iMax * jMax))).asMat();
    
        context.write(new IntWritable(0), new OpenCVMatWritable(mean));
    }
}
