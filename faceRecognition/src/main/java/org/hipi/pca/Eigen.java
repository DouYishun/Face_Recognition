package org.hipi.pca;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.hipi.opencv.OpenCVMatWritable;

import java.io.IOException;

import org.hipi.util.util;

public class Eigen {
    public static void main(String[] args) throws IOException {
        String baseDir = args[0];

        /* Read covar */
        System.out.println("Read covar from hdfs.");
        String covarPathStr = baseDir + "/covariance-output/part-r-00000";
        Configuration conf = new Configuration();
        Path covarPath = new Path(covarPathStr);
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream dis = fs.open(covarPath);

        OpenCVMatWritable covarWritable = new OpenCVMatWritable();
        covarWritable.readFields(dis);
        Mat covar = covarWritable.getMat();

        /* Compute eigenvectors */
        System.out.println("Running compute eigenvectors.");
        int N = util.patchSize;
        Mat eigenvalues = new Mat(N * N, N * N, opencv_core.CV_32FC1, new opencv_core.Scalar(0.0));
        Mat eigenvectors = new Mat(N * N, N * N, opencv_core.CV_32FC1, new opencv_core.Scalar(0.0));

        opencv_core.eigen(covar, eigenvalues, eigenvectors);

        Mat topEigenVectors = eigenvectors.apply(new opencv_core.Rect(0, 0, util.lowDimension, N * N));

        /* Write to hdfs */
        System.out.println("Write eigenvectors to hdfs.");
        Path eigenOutPath = new Path(baseDir + "/eigen-output/transformMatrix");
        FSDataOutputStream out = fs.create(eigenOutPath);
        OpenCVMatWritable eigenMatWritable = new OpenCVMatWritable(topEigenVectors);
        eigenMatWritable.write(out);
    }
}
