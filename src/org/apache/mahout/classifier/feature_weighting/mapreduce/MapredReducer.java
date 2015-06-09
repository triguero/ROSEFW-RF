package org.apache.mahout.classifier.feature_weighting.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.feature_weighting.builder.FWgenerator;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.feature_selection.mapreduce.partial.StrataID;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.IOException;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected FWgenerator fw_algorithm;
  
  private Dataset dataset;
  protected String header;

  protected PrototypeSet join = new PrototypeSet();
  protected int strata;

  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected FWgenerator getFWgeneratorBuilder() {
    return fw_algorithm;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getFWgeneratorBuilder(conf), Builder.loadDataset(conf), Builder.getHeader(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, FWgenerator fw_algorithm, Dataset dataset, String header) {
    Preconditions.checkArgument(fw_algorithm != null, "FWgenerator not found in the Job parameters");
    this.noOutput = noOutput;
    this.fw_algorithm = fw_algorithm;
    this.dataset = dataset;
    this.header = header;
  }

  
  /**
   * Generic reducer, it only adds all the RSs.
   */
  
protected void reduce(KEYIN id, Iterable<VALUEIN> rs, Context context)
		throws IOException, InterruptedException {
	// TODO Apéndice de método generado automáticamente
	

}


}


