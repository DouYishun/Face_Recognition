package org.hipi.util;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.opencv.OpenCVUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2GRAY;

public class util {
    public static final int patchSize = 64;  // Patch dimensions: patchSize x patchSize
    public static final int lowDimension = 30;

    // Used to convert input FloatImages into grayscale OpenCV Mats in MeanMapper and CovarianceMapper
    public static boolean convertFloatImageToGrayscaleMat(FloatImage image, opencv_core.Mat cvImage) {

        // Convert FloatImage to Mat, and convert Mat to grayscale (if necessary)
        HipiImageHeader.HipiColorSpace colorSpace = image.getColorSpace();
        switch(colorSpace) {

            //if RGB, convert to grayscale
            case RGB:
                opencv_core.Mat cvImageRGB = OpenCVUtils.convertRasterImageToMat(image);
                opencv_imgproc.cvtColor(cvImageRGB, cvImage, CV_RGB2GRAY);
                return true;

            //if LUM, already grayscale
            case LUM:
                cvImage = OpenCVUtils.convertRasterImageToMat(image);
                return true;

            //otherwise, color space is not supported for this example. Skip input image.
            default:
                System.out.println("HipiColorSpace [" + colorSpace + "] not supported in covar example. ");
                return false;
        }
    }

    public static void validateArgs(String[] args, int paraNum) throws IOException {
        if (args.length != paraNum) {
            System.out.println("Expect parameter number: " + paraNum);
            System.exit(1);
        }
    }

    public static void validatePath(String inputPathString, Configuration conf)
            throws IOException {
        Path inputPath = new Path(inputPathString);
        FileSystem fileSystem = FileSystem.get(conf);
        if (!fileSystem.exists(inputPath)) {
            System.out.println("Patch does not exist at location: " + inputPath);
            System.exit(1);
        }
    }

    public static void rmdir(String path, Configuration conf) throws IOException {
        Path outputPath = new Path(path);
        FileSystem fileSystem = FileSystem.get(conf);
        if (fileSystem.exists(outputPath)) {
            fileSystem.delete(outputPath, true);
        }
    }

    public static void mkdir(String path, Configuration conf) throws IOException {
        Path outputPath = new Path(path);
        FileSystem fileSystem = FileSystem.get(conf);
        if (!fileSystem.exists(outputPath)) {
            fileSystem.mkdirs(outputPath);
        }
    }

    public static double EuclideanDistance(double[] a, double[] b) {
        if (a.length != b.length) {
            System.out.println("Length not compatible.");
            System.exit(1);
        }
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    public static int indexOfMax(ArrayList<Double> list) {
        int index = -1;
        Double max = Double.MIN_VALUE;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) > max) {
                max = list.get(i);
                index = i;
            }
        }
        return index;
    }

    public static int getMostFrequentValue(ArrayList<Integer> list) {
        int value, freq, mValue = -1, mFreq = -1;
        for (int i = 0; i < list.size(); i++) {
            value = list.get(i);
            freq = Collections.frequency(list, value);
            if (freq > mFreq) {
                mFreq = freq;
                mValue = value;
            }
        }
        return mValue;
    }

}

















