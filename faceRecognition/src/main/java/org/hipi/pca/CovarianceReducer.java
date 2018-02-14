package org.hipi.pca;

import org.hipi.opencv.OpenCVMatWritable;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;

import java.io.IOException;

public class CovarianceReducer extends
        Reducer<IntWritable, OpenCVMatWritable, NullWritable, OpenCVMatWritable> {

    @Override
    public void reduce(IntWritable key, Iterable<OpenCVMatWritable> values, Context context)
        throws IOException, InterruptedException {

        /* Compute covariance */

        int N = Covariance.patchSize;

        Mat cov = new Mat(N * N, N * N, opencv_core.CV_32FC1, new Scalar(0.0));

        // Consolidate covariance matrices
        for(OpenCVMatWritable value : values) {
            opencv_core.add(value.getMat(), cov, cov);
        }


        /* Compute eigenvectors */

        Mat eigenvalues = new Mat(N * N, N * N, opencv_core.CV_32FC1, new Scalar(0.0));
        Mat eigenvectors = new Mat(N * N, N * N, opencv_core.CV_32FC1, new Scalar(0.0));

        opencv_core.eigen(cov, eigenvalues, eigenvectors);

        Mat topEigenVectors = eigenvectors.apply(new Rect(0, 0, 30, N * N));

        context.write(NullWritable.get(), new OpenCVMatWritable(topEigenVectors));  // shape (4096 * 15)
    }
}
