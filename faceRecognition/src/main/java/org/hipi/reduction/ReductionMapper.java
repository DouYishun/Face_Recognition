package org.hipi.reduction;


import org.apache.hadoop.io.Text;
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.opencv.OpenCVMatWritable;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.hipi.util.helper;

import java.io.IOException;
import java.nio.FloatBuffer;

public class ReductionMapper
        extends Mapper<HipiImageHeader, FloatImage, IntWritable, Text> {
    Mat mean;
    Mat transformMatrix;

    @Override
    public void setup(Context job) {

        /* Create mean and transformMatrix mat */

        try {
            String meanPathString = job.getConfiguration().get("hipi.pca.reduction.mean.path");
            String transformMatrixPathString = job.getConfiguration().get("hipi.pca.reduction.transformMatrix.path");
            if (meanPathString == null || transformMatrixPathString == null) {
                System.err.println("Configuration path not set properly.");
                System.exit(1);
            }
            Path meanPath = new Path(meanPathString);
            FSDataInputStream dis1 = FileSystem.get(job.getConfiguration()).open(meanPath);

            OpenCVMatWritable meanWritable = new OpenCVMatWritable();
            meanWritable.readFields(dis1);
            mean = meanWritable.getMat();  // shape (64 * 64)


            Path transformMatrixPath = new Path(transformMatrixPathString);
            FSDataInputStream dis2 = FileSystem.get(job.getConfiguration()).open(transformMatrixPath);

            OpenCVMatWritable transformMatrixWritable = new OpenCVMatWritable();
            transformMatrixWritable.readFields(dis2);
            transformMatrix = transformMatrixWritable.getMat();  // shape (4096 * 30)

        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(1);
        }
    }


    public void map(HipiImageHeader header, FloatImage image, Context context)
            throws IOException, InterruptedException {

        /* Get image label */

        // e.g. filename = "123_20.png", label 123, 20th image.
        String filename = header.getMetaData("filename");
        int label = Integer.parseInt(filename.substring(0, filename.indexOf('_')));


        /* Perform conversion to OpenCV */

        Mat cvImage = new Mat(image.getHeight(), image.getWidth(), opencv_core.CV_32FC1);

        // if unable to convert input FloatImage to grayscale Mat, skip image and move on
        if(!helper.convertFloatImageToGrayscaleMat(image, cvImage)) {
            System.out.println("Skipping image with invalid color space.");
            return;
        }


        /* Get image feature */

        // patch dimensions (N X N)
        int N = Reduction.patchSize;

        Mat features = new Mat(N, N, opencv_core.CV_32FC1, new Scalar(0.0));

        // specify number of patches to use in mean patch computation (iMax * jMax patches)
        int iMax = 10, jMax = 10;

        // collect patches and add their values to mean patch mat
        for (int i = 0; i < iMax; i++) {
            int x = ((cvImage.cols() - N) * i) / iMax;
            for (int j = 0; j < jMax; j++) {
                int y = ((cvImage.rows() - N) * j) / jMax;
                Mat patch = cvImage.apply(new Rect(x, y, N, N));
                opencv_core.subtract(patch, mean, patch);
                opencv_core.add(patch, features, features);
            }
        }

        features = opencv_core.divide(mean, ((double) (iMax * jMax))).asMat();
        features.reshape(0, N * N);

        // reduction
        opencv_core.multiply(transformMatrix.t().asMat(), features, features);

        // concat label and features


        context.write(new IntWritable(0), );

    }

}
