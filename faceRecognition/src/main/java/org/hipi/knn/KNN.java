package org.hipi.knn;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.hipi.util.util;

public class KNN extends Configured implements Tool {

    public int run(String[] args) throws Exception {
        Configuration conf = Job.getInstance().getConfiguration();

        util.validateArgs(args, 3);

        // Build I/O path strings
        String dataBaseDir = args[0];
        String trainFilePath = dataBaseDir + "/train/part-r-00000";
        String testFilePath = dataBaseDir + "/test/part-r-00000";
        String outputDir = args[1];
        int k = Integer.parseInt(args[2]);

        // Set up directory structure
        util.rmdir(outputDir, conf);

        // Run KNN
        if (runKNN.run(trainFilePath, testFilePath, outputDir, k) == 1) {
            System.out.println("KNN job failed to complete.");
            return 1;
        }

        return 0;
    }

    public static void main(String[] args) throws Exception {
        /*
            args: dataBaseDir outputDir k
         */
        int res = ToolRunner.run(new KNN(), args);
        System.exit(res);
    }
}
