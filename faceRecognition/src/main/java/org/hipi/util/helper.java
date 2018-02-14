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

import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2GRAY;

public class helper {

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

    public static void validateArgs(String[] args, Configuration conf, int paraNum) throws IOException {
        if (args.length != 2) {
            System.out.println("Expect parameter number: " + paraNum);
            System.exit(1);
        }
    }

    public static void validatePath(String inputPathString, Configuration conf)
            throws IOException {
        Path meanPath = new Path(inputPathString);
        FileSystem fileSystem = FileSystem.get(conf);
        if (!fileSystem.exists(meanPath)) {
            System.out.println("Patch does not exist at location: " + meanPath);
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
}
