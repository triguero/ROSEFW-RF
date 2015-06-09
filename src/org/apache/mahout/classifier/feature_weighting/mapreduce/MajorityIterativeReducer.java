package org.apache.mahout.classifier.feature_weighting.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.feature_weighting.builder.FWgenerator;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.feature_weighting.mapreduce.partial.StrataID;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * This Mapred allows to run more than one reducers.
 * 
 */
public class MajorityIterativeReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected FWgenerator fw_algorithm;
  
  private Dataset dataset;
  protected String header;

  protected double []AggregateWeights;
  protected int mappers=0;
  protected int strata;
  private int firstId = 0;

  
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
	
		//System.out.println("Si paso por aquí: "+id);
		//strata = (StrataID) id;

		for(VALUEIN value: rs){
			MapredOutput prueba = (MapredOutput) value;
			double [] pesos= prueba.getSelectedFeatures();
			mappers++;
			
			if(AggregateWeights==null){
				AggregateWeights= new double[pesos.length];
				Arrays.fill(AggregateWeights,0);
			}
			int selecs=0;
			for(int i=0; i<pesos.length;i++){
				AggregateWeights[i]+=pesos[i];
			}
			
			//System.out.println("Seleccionadas en mapper "+id+": "+selecs);
			context.progress();
	

		}



	
	}


	 protected void cleanup(Context context) throws IOException, InterruptedException {
		 
		    System.out.println("escribo la agregación de pesos final.");
		    StrataID key = new StrataID();

		    key.set(strata, firstId + 1);
		    
		    
            FileOutputStream f = new FileOutputStream("pesosAgregados.txt");
            DataOutputStream fis = new DataOutputStream((OutputStream) f);
          //  System.out.println("Cadena="+cadena);
            
			for(int i=0; i<AggregateWeights.length;i++){
				AggregateWeights[i]/=mappers;
				// imprimir también los pesos.
				System.out.print(AggregateWeights[i]+",");
				fis.writeBytes(AggregateWeights[i]+"\n");
	            
	            	
			}
			fis.close();
			
			
			MapredOutput salida= new MapredOutput(AggregateWeights);
			context.write((KEYOUT) key, (VALUEOUT) salida);
	 }
}


