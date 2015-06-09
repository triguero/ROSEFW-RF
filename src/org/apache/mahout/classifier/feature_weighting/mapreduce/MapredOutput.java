package org.apache.mahout.classifier.feature_weighting.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import com.google.common.io.Closeables;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;


// Print a PrototypeSet! .

/**
 * Print a reduced set as a PrototypeSet.
 */
public class MapredOutput implements Writable, Cloneable {

  private double[] selectedFeatures;  // conjunto reducido.


  public MapredOutput() {
  }

  // constructor básico
  public MapredOutput(double[] selected) { //, int[] predictions
    this.selectedFeatures = selected;
  }
 

  public double[] getSelectedFeatures() {
    return selectedFeatures;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    boolean readCorrect = in.readBoolean();
    if (readCorrect) {
    	int size=in.readInt();
    	selectedFeatures=new double[size];
    	for(int i=0; i<size;i++){
    		selectedFeatures[i]= in.readDouble();

    	}
    	
    }

  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(selectedFeatures != null);
    if (selectedFeatures != null) {
      out.writeInt(selectedFeatures.length);
  	
      for(int i=0; i<selectedFeatures.length;i++){
		out.writeDouble(selectedFeatures[i]);
	  }
      
    }

  }

  @Override
  public MapredOutput clone() {
    return new MapredOutput(selectedFeatures); //, predictions
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof MapredOutput)) {
      return false;
    }

    MapredOutput mo = (MapredOutput) obj;

    return ((selectedFeatures == null && mo.getSelectedFeatures() == null) || (selectedFeatures != null && selectedFeatures.equals(mo.getSelectedFeatures()))); //&& Arrays.equals(predictions, mo.getPredictions()
  }
  
  public static double[] load(Configuration conf, Path rsPath) throws IOException {
	    FileSystem fs = rsPath.getFileSystem(conf);
	   
	    
	    /*Path[] files;
	    if (fs.getFileStatus(rsPath).isDir()) {
	      files = Utils.listOutputFiles(fs, rsPath);
	    } else {
	      files = new Path[]{rsPath};
	    }
	    */
	    FSDataInputStream dataInput = new FSDataInputStream(fs.open(rsPath));
	      
	    System.out.println("Leyendo: "+rsPath.toString());
	   // System.out.println(dataInput.readLine());
	   // boolean readCorrect = dataInput.readBoolean();
	         
	    double[] features=null;
	    
	   // if (readCorrect) {
	    	String dato = dataInput.readLine();
	    	
	    	int size=Integer.valueOf(dato);
		    System.out.println("Dato: "+dato);

		    System.out.println("Size: "+size);

		       features= new double[size];
	    	for(int i=0; i<size;i++){
	    		dato = dataInput.readLine();
	    		features[i]= Double.valueOf(dato);
	        	System.out.print(features[i]+", ");

	    	}
	    	
	    //}
	    
	    dataInput.close();
	    
        return features;
	    
	  }


}

