package org.hipi.covar;

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

        int N = Covariance.patchSize;

        Mat cov = new Mat(N * N, N * N, opencv_core.CV_32FC1, new Scalar(0.0));

        Mat eigenValues = new Mat(N * N, N * N, opencv_core.CV_32FC1, new Scalar(0.0));
        Mat eigenVectors = new Mat(N * N, N * N, opencv_core.CV_32FC1, new Scalar(0.0));

        // Consolidate covariance matrices
        for(OpenCVMatWritable value : values) {
            opencv_core.add(value.getMat(), cov, cov);
        }

        opencv_core.eigen(cov, eigenValues, eigenVectors);

        Mat topEigenVectors = eigenVectors.apply(new Rect(0, 0, 15, N * N));

        context.write(NullWritable.get(), new OpenCVMatWritable(topEigenVectors));  // shape (4096 * 15)
    }
}
