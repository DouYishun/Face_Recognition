package org.hipi.reduction;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.opencv.OpenCVUtils;
import org.hipi.pca.ComputeCovariance;
import org.hipi.pca.ComputeMean;
import org.hipi.util.helper;

import javax.rmi.CORBA.Util;
import java.io.IOException;

import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2GRAY;
import static org.hipi.util.helper.rmdir;


public class Reduction extends Configured implements Tool {

    public static final int patchSize = 64;  // Patch dimensions: patchSize x patchSize

    public int run(String[] args) throws Exception {

        // Used for initial argument validation and hdfs configuration before jobs are run
        Configuration conf = Job.getInstance().getConfiguration();

        // Validate arguments before any work is done
        helper.validateArgs(args, conf, 2);

        // Build I/O path strings
        String inputHibPath = args[0];
        String coefficientBaseDir = args[1];
        String transformMatrixPath = coefficientBaseDir + "/transform_matrix-output/part-r-00000";
        String meanPath = coefficientBaseDir + "/mean-output/part-r-00000";
        String outputDir = args[2];

        // Set up directory structure
        helper.mkdir(outputDir, conf);

        // Run compute mean
        if (runReduction.run(args, inputHibPath, transformMatrixPath, meanPath, outputDir) == 1) {
            System.out.println("Reduction job failed to complete.");
            return 1;
        }

        // Indicate success
        return 0;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new org.hipi.pca.Covariance(), args);
        System.exit(res);
    }
}
