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
import org.hipi.util.util;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class ReductionMapper
        extends Mapper<HipiImageHeader, FloatImage, IntWritable, Text> {
    private final static IntWritable zero = new IntWritable(0);
    private Text value = new Text();
    private Mat mean, transformMatrix;

    @Override
    public void setup(Context job) {
        /* Create mean and transformMatrix mat */
        try {
            String meanPathString = job.getConfiguration().get("hipi.reduction.mean.path");
            String transformMatrixPathString = job.getConfiguration().get("hipi.reduction.transformMatrix.path");
            if (meanPathString == null || transformMatrixPathString == null) {
                System.err.println("Configuration path not set properly.");
                System.exit(1);
            }
            Path meanPath = new Path(meanPathString);
            Path transformMatrixPath = new Path(transformMatrixPathString);

            FSDataInputStream dis1 = FileSystem.get(job.getConfiguration()).open(meanPath);
            FSDataInputStream dis2 = FileSystem.get(job.getConfiguration()).open(transformMatrixPath);

            OpenCVMatWritable meanWritable = new OpenCVMatWritable();
            OpenCVMatWritable transformMatrixWritable = new OpenCVMatWritable();

            meanWritable.readFields(dis1);
            transformMatrixWritable.readFields(dis2);

            mean = meanWritable.getMat();  // shape (64 * 64)
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
        String labelStr = filename.substring(0, filename.indexOf('_'));

        /* Perform conversion to OpenCV */
        Mat cvImage = new Mat(image.getHeight(), image.getWidth(), opencv_core.CV_32FC1);

        // if unable to convert input FloatImage to grayscale Mat, skip image and move on
        if(!util.convertFloatImageToGrayscaleMat(image, cvImage)) {
            System.out.println("Skipping image with invalid color space.");
            return;
        }


        /* Get image feature */
        // patch dimensions (N X N)
        int N = util.patchSize;

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

        features = opencv_core.divide(features, ((double) (iMax * jMax))).asMat();

        // reduction
        // (30*4096) * (4096*1) = (30 * 1)
        Mat newFeatures = opencv_core.multiply(transformMatrix.t().asMat(), features.reshape(0, N*N)).asMat();

        // mat features to string
        int elms = (int)(newFeatures.total() * newFeatures.channels());
        float [] floatData = new float[elms];
        ((FloatBuffer)newFeatures.createBuffer()).get(floatData);
        String featuresStr = Arrays.toString(floatData).replace(",", "").replace("[", "").replace("]", "");

        // concat label and features
        value.set(featuresStr + " " + labelStr);
        context.write(zero, value);
    }
}
