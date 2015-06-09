ROSEFW-RF
====

This repository includes the MapReduce implementations used in [1].
This implementation is based on Apache Mahout 0.8 library. The Apache Mahout (http://mahout.apache.org/) project's goal is to build an environment for quickly creating scalable performant machine learning applications.

Prerequisites:
- Hadoop 2.5.
- ant

Associated paper:

- I. Triguero, S. Río, V. López, J. Bacardit, J.M. Benítez, F. Herrera. ROSEFW-RF: The winner algorithm for the ECBDL'14 Big Data Competition: An extremely imbalanced big data bioinformatics problem. Knowledge-Based Systems, in press. doi: 10.1016/j.knosys.2015.05.027 


Compile the whole project with ANT:
<pre>
$ ant
</pre>
Put the dataset folder into the HDFS system:
<pre>
hadoop fs -put datasets/
</pre>
Generate descriptor file needed by the mahout code. (Check: ...classifier.df.tools.Describe.java).
<pre>
$ hadoop jar Model.jar org.apache.mahout.classifier.df.tools.Describe -p  datasets/ECBDL14subset.data  -f  datasets/ECBDL14subset.info -d  3 N 18 C 18 N 54 C 38 N 20 C 480 N L
</pre>
==
Random Oversampling
==

<pre>
hadoop jar Model.jar  org.apache.mahout.classifier.df.mapreduce.Resampling --help

Usage:                                                                          
 [--data <path> --dataset <dataset> --time <path> --help --resampling           
<resampling> --dataPreprocessing <path> --nbpartitions <nbpartitions> --npos    
<npos> --nneg <nneg> --negclass <negclass>]                                     
Options                                                                         
  --data (-d) path                    Data path                                 
  --dataset (-ds) dataset             Dataset path                              
  --time (-tm) path                   Time path                                 
  --help (-h)                         Print out help                            
  --resampling (-rs) resampling       The resampling technique (oversampling    
                                      (overs), undersampling (unders) or SMOTE  
                                      (smote))                                  
  --dataPreprocessing (-dp) path      Data Preprocessing path                   
  --nbpartitions (-p) nbpartitions    Number of partitions                      
  --npos (-npos) npos                 Number of instances of the positive class 
  --nneg (-nneg) nneg                 Number of instances of the negative class 
  --negclass (-negclass) negclass     Name of the negative class      
</pre>
Generate the Preprocessed data example:

To compute the number of mappers, we have to check the number of bytes of the training file:
<pre>
$ ls -l datasets/
-rw-r--r--. 1 isaact users 19019170 Jun  9 14:10 ECBDL14subset.data
</pre>
If we want to have 4 maps, we should divide this number by 4 (4754792).

<pre>
$ hadoop jar Model.jar org.apache.mahout.classifier.df.mapreduce.Resampling -Dmapred.min.split.size=4754792 -Dmapred.max.split.size=4754793 -dp datasets/ECBDL14subset.data -d output-ROS -ds datasets/ECBDL14subset.info -rs overs -p 4 -tm ROS-ECBDL14-build_time
</pre>

==
Evolutionary Feature Weighting
==

<pre>
hadoop jar Model.jar org.apache.mahout.classifier.feature_weighting.mapreduce.FeatureWeightingModel --help

Usage:                                                                          
 [--data <path> --dataset <dataset> --header <header> --output <path>]          
Options                                                                         
  --data (-d) path           Data path                                          
  --dataset (-ds) dataset    The path of the file descriptor of the dataset     
  --header (-he) header      Header of the dataset in Keel format               
  --output (-o) path         Output path, will contain the set of selected      
                             features   
</pre>
Example of application of EFW on the previosly generated balanced data. (please adjust the size of the split according to the size of the input data)

<pre>
hadoop jar Model.jar org.apache.mahout.classifier.feature_weighting.mapreduce.FeatureWeightingModel -Dmapred.max.split.size=XXXX -d output-ROS/part-r-00000 -ds datasets/ECBDL14subset.info -he datasets/ECBDL14subset.header -o output-DEFW
</pre>

Create the resulting preprocessed dataset:

<pre>
hadoop jar Model.jar org.apache.mahout.classifier.feature_weighting.mapreduce.FWconstructor --help

Usage:                                                                          
 [--input <input> --info <test> --header <header> --feature_weighting <path>    
--weight threshold <path> --output <output> --help]                             
Options                                                                         
  --input (-i) input                Path to job input directory.                
  --info (-ds) test                 The path of the file descriptor of the      
                                    dataset                                     
  --header (-he) header             Header of the dataset in Keel format        
  --feature_weighting (-fw) path    Feature weights path                        
  --weight threshold (-w) path      Weight threshold to select features         
  --output (-o) output              The directory pathname for output.          
  --help (-h)                       Print out help  
</pre>

<pre>
 hadoop jar Model.jar org.apache.mahout.classifier.feature_weighting.mapreduce.FWconstructor -i output-ROS/part-r-00000 -fw output-DEFW/Pesos.txt -w 0.46 -ds datasets/ECBDL14subset.info -he datasets/ECBDL14subset.header -o output-FWconstructor
</pre>

==
RandomForest
==

First, generate the describe info file for this data:

<pre>
hadoop jar Model.jar org.apache.mahout.classifier.df.tools.Describe -p output-FWconstructor/part-r-00000.out -f   output-FWconstructor/part-r-00000.info -d 3 N 18 C 18 N 54 C 38 N 20 C 480 N L
</pre>

Build a model with the previous preprocessed data. Please adjust the split size accordingly.

<pre>
hadoop jar Model.jar  org.apache.mahout.classifier.df.mapreduce.BuildForest -Dmapred.min.split.size=XXXXX -Dmapred.max.split.size=XXXX -o output-RF/  -d output-FWconstructor/part-r-00000.out -ds output-FWconstructor/part-r-00000.info -sl 25 -p -t 200 -tm model_build_time
</pre>

Classify test data:

<pre>
hadoop jar  Model.jar org.apache.mahout.classifier.df.mapreduce.TestForest -Dmapred.min.split.size=XXXX -Dmapred.max.split.size=XXXX -i datasets/ECBDL14subset.data
-ds datasets/ECBDL14subset.info -m  output-RF/ -a -mr -o outputTEST-RF
</pre>




