package org.hipi.knn;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.hipi.util.helper;


public class KNN extends Configured implements Tool {

    public int run(String[] args) throws Exception {

        // Used for initial argument validation and hdfs configuration before jobs are run
        Configuration conf = Job.getInstance().getConfiguration();

        // Validate arguments before any work is done
        helper.validateArgs(args, 3);

        // Build I/O path strings
        String dataBaseDir = args[0];
        String trainFilePath = dataBaseDir + "/train/part-r-00000";
        String testFilePath = dataBaseDir + "/test/part-r-00000";
        String outputDir = args[1];
        int k = Integer.parseInt(args[2]);

        // Set up directory structure
        helper.rmdir(outputDir, conf);

        // Run KNN
        if (runKNN.run(trainFilePath, testFilePath, outputDir, k) == 1) {
            System.out.println("KNN job failed to complete.");
            return 1;
        }

        // Indicate success
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
