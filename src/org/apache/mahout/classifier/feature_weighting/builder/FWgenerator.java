package org.apache.mahout.classifier.feature_weighting.builder;

import java.util.Arrays;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.basic.data.Data;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.feature_selection.utils.PGUtils;
import org.apache.mahout.classifier.feature_selection.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDECS.IPLDECSGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.RandomGenerator;
import org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.evolutionary_algorithms.CHC.wrapper.CHCBinaryLVO;
import org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.nonevolutionary_algorithms.FOCUS.FocusIncon;
import org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.nonevolutionary_algorithms.LVF.LVFIncon;
import org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.nonevolutionary_algorithms.LVW.LVWLVO;
import org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.nonevolutionary_algorithms.RELIEF_F.Relieff;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.core.Randomize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FWgenerator  {
  
  private static final Logger log = LoggerFactory.getLogger(FWgenerator.class);	
  int nClasses, nLabels;

  public String FWmethod = "DEFW";
  
  private int iterFW=200;
  private int PopulationFW=10;
  private int Strategy=3;
  private int iterSFGSS=8;
  private int iterSFHC=20;
  private double Fl=0.1, Fu=0.9;
  private double tau[] = new double[4];
  
  protected int positiveClass=0;

  public String header;
  
  double Weights[];
  
  Context context;

  //  strata[i].print();
	  
  public FWgenerator() {
  }
  
  public FWgenerator(String alg)
  {
	  this.FWmethod = alg;
  }
  
  public void setNLabels(int nLabels) {
    this.nLabels = nLabels;
  }


  public void setHeader(String header){
	  this.header=header;
  }
  public void build(Data data, Context context) throws Exception {
    //We do here the algorithm's operations

	this.context= context;
	context.progress();
	Dataset dataset = data.getDataset();
	log.info("FW: data size = "+data.size());

	nClasses = dataset.nblabels();
	
	//Gets the number of input attributes of the data-set
	int nInputs = dataset.nbAttributes() - 1;
	
	//It returns the class labels
	String clases[] = dataset.labels();
	

	context.progress();
	
	PrototypeSet train = new PrototypeSet(data,context);
	
	// DEFW subset:
	PrototypeSet validation = new PrototypeSet();
	PrototypeSet Clases [] = new PrototypeSet[nClasses];

	int min=Integer.MAX_VALUE;
	  
	for(int i=0; i<nClasses;i++){ 
		  Clases[i] = new PrototypeSet(train.getFromClass(i).clone());
		  
		  if(Clases[i].size()<min){
			  min=Clases[i].size();
			  this.positiveClass=i;   //ESTABLISH THE POSITIVE CLASS
		  }
		  
		  int numberOfPrototypes = (int)Math.round(Clases[i].size()*0.05);// 10%
		  System.out.println(numberOfPrototypes);
		  if(numberOfPrototypes < 1){numberOfPrototypes=1;}
		  
		  for(int j=0; j< numberOfPrototypes; j++){
			  validation.add(Clases[i].getRandom());
		  }
		  
		  
		  
	}
	validation.randomize();
	
	Weights=new double[train.get(0).numberOfInputs()];
	Arrays.fill(Weights, 1.0); // initially 1.0

    tau[0]=0.1; tau[1]=0.1; tau[2]=0.03; tau[3]= 0.07;

    System.out.println("Comenzando Feature Weighting");
    Weights=FeatureWeighting(train, Weights, validation);

}



  public double[] applyFW() {
	  
	  for(int i=0;i<Weights.length;i++){
		  System.out.print(Weights[i]+",");
	  }
	  return Weights;
  }

  
  /**
   * Feature weighting optimization by SFLSDE algorithm
   * @param actual
   * @return
   */
  public double[] FeatureWeighting(PrototypeSet actual, double[] original, PrototypeSet trainingDataSet){
	  
	  	  int iterations =0;
		  int numberOfInputs = actual.get(0).numberOfInputs();
		  

	  	  iterations = this.iterFW;//*numberOfInputs;
	  	 
		   PrototypeSet nominalPopulation =actual;
	       

		  Prototype [] Population = new Prototype[PopulationFW]; // I use prototype as double[]
		  Prototype mutation[] = new Prototype[PopulationFW];
		  Prototype crossover[] = new Prototype[PopulationFW];
		  
		  // Initialize population: Randomly.
		  
		  Population[0] = new Prototype(numberOfInputs,0);
		  for(int j=0; j<numberOfInputs; j++){
				Population[0].setInput(j, original[j]);    // The first individual is 1.0 (the original)
		  }
		  
		  context.progress();
		  
		  for(int i=1; i< this.PopulationFW; i++){
			  Population[i]= new Prototype(numberOfInputs,0);
			  
			  for(int j=0; j<numberOfInputs; j++){
				Population[i].setInput(j, RandomGenerator.Rand());
			  }
			  
			 // Population[i].print();
		  }
		  
		  double ScalingFactor[] = new double[this.PopulationFW];
		  double CrossOverRate[] = new double[this.PopulationFW]; // Inside of the Optimization process.
		  double fitness[] = new double[PopulationFW];

		  double fitness_bestPopulation[] = new double[PopulationFW];
		  
		  
		   for(int i=0; i< this.PopulationFW; i++){
			   ScalingFactor[i] =  RandomGenerator.Randdouble(0, 1);
			   CrossOverRate[i] =  RandomGenerator.Randdouble(0, 1);
		   }
		   
	      for(int i=0; i< PopulationFW; i++){
	    	  context.progress();
			  fitness[i] = GeometricMean(nominalPopulation, Population[i],trainingDataSet);   // PSOfitness
			  fitness_bestPopulation[i] = fitness[i]; // Initially the same fitness.
		  }
		  
		  
		  //We select the best initial  particle
		  double bestFitness=fitness[0];
		  int bestFitnessIndex=0;
		  for(int i=1; i< PopulationFW;i++){
			  if(fitness[i]>bestFitness){
				  bestFitness = fitness[i];
				  bestFitnessIndex=i;
			  }
			  
		  }
		  
		   for(int j=0;j<PopulationFW;j++){
	         //Now, I establish the index of each prototype.
			  for(int i=0; i<Population.length; ++i)
				  Population[i].setIndex(i);
		   }
		   
  
		   double randj[] = new double[5];
		   
		   
		   for(int iter=0; iter< iterations; iter++){ // Main loop
			   context.progress();
			   System.out.print(iter+", ");
			   for(int i=0; i<PopulationFW; i++){
				   context.progress();
				   // Generate randj for j=1 to 5.
				   for(int j=0; j<5; j++){
					   randj[j] = RandomGenerator.Randdouble(0, 1);
				   }
				   

				   if(i==bestFitnessIndex && randj[4] < tau[2]){
					  // System.out.println("SFGSS applied");
					   //SFGSS
					   crossover[i] = SFGSS(Population, i, bestFitnessIndex, CrossOverRate[i],nominalPopulation,trainingDataSet);
					   
					   
				   }else if(i==bestFitnessIndex &&  tau[2] <= randj[4] && randj[4] < tau[3]){
					   //SFHC
					   //System.out.println("SFHC applied");
					   crossover[i] = SFHC(Population, i, bestFitnessIndex, ScalingFactor[i], CrossOverRate[i],nominalPopulation,trainingDataSet);
					   
				   }else {
					   
					   // Fi update
					   
					   if(randj[1] < tau[0]){
						   ScalingFactor[i] = this.Fl + this.Fu*randj[0];
					   }
					   
					   // CRi update
					   
					   if(randj[3] < tau[1]){
						   CrossOverRate[i] = randj[2];
					   }
					   				   
					  //Mutation:
						
					   mutation[i]  = mutant(Population, i, bestFitnessIndex, ScalingFactor[i]);
					   
					    // Crossver Operation.

					   crossover[i] = new Prototype(Population[i]);
					   
					   for(int j=0; j< Population[i].numberOfInputs(); j++){ // For each part of the solution
						   
						   if(RandomGenerator.Randdouble(0, 1)<CrossOverRate[i]){
							   crossover[i].setInput(j, mutation[i].getInput(j)); // Overwrite.
						   }
					   }
					   
					   
					   
					   
				   }
				   
	   
				   
				   // Fourth: Selection Operation.
			   

			      // fitness[i] =  GeometricMean(nominalPopulation,Population[i],trainingDataSet);
   			       double trialVector =  GeometricMean(nominalPopulation, crossover[i],trainingDataSet);
				
			  
				  if(trialVector > fitness[i]){
					  Population[i] = new Prototype(crossover[i]);
					  fitness[i] = trialVector;
				  }
				  
				  if(fitness[i]>bestFitness){
					  
					  bestFitness = fitness[i];
					  bestFitnessIndex=i;
					  System.out.println("FITNESSFW= "+ bestFitness);
				  }
				  
				  
			   }

			   
		   }
		  
		  
		   context.progress();
		   System.out.println("Best weightings");

		   for(int i=0; i<Population[bestFitnessIndex].numberOfInputs(); i++ ){
			   System.out.print(Population[bestFitnessIndex].getInput(i) + "  ");
		   }
		
	  
	  
	  
	  return Population[bestFitnessIndex].getInputs();
  }
  
  
  
  public static int _1nnWeighted(Prototype current, Prototype Weights, PrototypeSet dataSet)
  {
      int indexNN = 0;

      double minDist =Double.POSITIVE_INFINITY;
      double currDist;
      int _size = dataSet.size();

      for (int i=0; i<_size; i++)
      {
          Prototype pi = dataSet.get(i);

          
	          double acc = 0.0;
	          for (int j = 0; j < current.numberOfInputs(); j++)
	          {
	              acc += ((current.getInput(j) - pi.getInput(j)) * (current.getInput(j)  - pi.getInput(j)))*Weights.getInput(j);
	          }
	          
           currDist =  acc;
          
           if(currDist >0){
              if (currDist < minDist)
              {
                  minDist = currDist;
                  indexNN =i;
              }
          }
        
      }
      
     
      return (int)dataSet.get(indexNN).getOutput(0);
  }
  
  

  
  
  public double GeometricMean(PrototypeSet train, Prototype Weights, PrototypeSet test){
	  double GeometricMean;
	  
	  int [] pre = new int[test.size()];
	  
	  for(int i=0; i<test.size();i++)
      {
         pre[i]=_1nnWeighted(test.get(i), Weights, train);  
      }
	  
	  int tn=0, fp=0, tp=0,fn=0;
	  
	  for(int i=0; i<test.size();i++){
		  
		  if(test.get(i).getOutput(0)==pre[i] && pre[i]!=this.positiveClass){ // esto es un TN
			  tn++;			  
		  }else if(test.get(i).getOutput(0)!=pre[i] && pre[i]==this.positiveClass){ // es un false positve
			  fp++;
		  }else if(test.get(i).getOutput(0)==pre[i] && pre[i]==this.positiveClass){ // esto es un tp
			  tp++;			  
		  }else if(test.get(i).getOutput(0)!=pre[i] && test.get(i).getOutput(0)==this.positiveClass){ // es un false
			  fn++;
		  }
	  }
	  
	  double TNrate= ((1.*tn)/((tn+fp)*1.));
	  double TPrate=  ((1.*tp)/((tp+fn)*1.));;
	  
	  GeometricMean = TPrate*TNrate;
		  
	  return GeometricMean;
	  
  }
  
  
  

  /**
   * MUTANT FOR FEATURE WEIGHTIN
   * @param population
   * @param actual
   * @param mejor
   * @param SFi
   * @return
   */
  
public Prototype mutant(Prototype population[], int actual, int mejor, double SFi){
  	  
  	  
  	  Prototype mutant = new Prototype();
  	  Prototype r1,r2,r3,r4,r5, resta, producto, resta2, producto2, result, producto3, resta3;
  	  

  		 
  		mutant = new Prototype();
  	  
  		//We need three differents solutions of actual
  		   
  	   int lista[] = new int[population.length];
       inic_vector_sin(lista,actual);
       desordenar_vector_sin(lista);
  		      
  	  // System.out.println("Lista = "+lista[0]+","+ lista[1]+","+lista[2]);
  	  
       
  	   r1 = population[lista[0]];
  	   r2 = population[lista[1]];
  	   r3 = population[lista[2]];
  	   r4 = population[lista[3]];
  	   r5 = population[lista[4]];
  		   
  			switch(this.Strategy){
  		   	   case 1: // ViG = Xr1,G + F(Xr2,G - Xr3,G) De rand 1
  		   		 resta = r2.sub(r3);
  		   		 producto = resta.mul(SFi);
  		   		 mutant = producto.add(r1);
  		   	    break;
  			   
  		   	   case 2: // Vig = Xbest,G + F(Xr2,G - Xr3,G)  De best 1
  			   		 resta = r2.sub(r3);
  			   		 producto = resta.mul(SFi);
  			   		 mutant = population[mejor].add(producto);
  			   break;
  			   
  		   	   case 3: // Vig = ... De rand to best 1
  		   		   resta = r1.sub(r2); 
  		   		   resta2 = population[mejor].sub(population[actual]);
  		   		 			   		 
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   		
  			   	   result = population[actual].add(producto);
  			   	   mutant  = result.add(producto2);
  			   		 			   		 
  			   break;
  			   
  		   	   case 4: // DE best 2
  		   		   resta = r1.sub(r2); 
  		   		   resta2 = r3.sub(r4);
  		   		 			   		 
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   		
  			   	   result = population[mejor].add(producto);
  			   	   mutant  = result.add(producto2);
  			   break;
  			  
  		   	   case 5: //DE rand 2
  		   		   resta = r2.sub(r3); 
  		   		   resta2 = r4.sub(r5);
  		   		 			   		 
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   		
  			   	   result = r1.add(producto);
  			   	   mutant  = result.add(producto2);
  			   	   
    		       break;
    		       
  		   	   case 6: //DE rand to best 2
  		   		   resta = r1.sub(r2); 
  		   		   resta2 = r3.sub(r4);
  		   		   resta3 = population[mejor].sub(population[actual]);
  		   		   
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   	   producto3 = resta3.mul(SFi);
  			   	   
  			   	   result = population[actual].add(producto);
  			   	   result = result.add(producto2);
  			   	   mutant  = result.add(producto3);
    		       break;
    		       
  		   }   
  	   

  	  // System.out.println("********Mutante**********");
  	 // mutant.print();
  	   
  		for(int j=0; j<mutant.numberOfInputs(); j++){
  			if(mutant.getInput(j)<=0.2){                 //Suggested by Salva!
  				mutant.setInput(j, 0);
  			}else if(mutant.getInput(j)>1){
  				mutant.setInput(j, 1);
  			}
  			
  		}
  	
       
 
  	  return mutant;
 }



  
  /**
   * SFGSS local Search.  FOR  Feature Weighting
   * @param population
   * @return
   */
  public Prototype SFGSS(Prototype population[], int actual, int mejor, double CRi, PrototypeSet reduced, PrototypeSet trainingDataSet){
	  double a=0.1, b=1;

	  double fi1=0, fi2=0, fitnessFi1=0, fitnessFi2=0;
	  double phi = (1+ Math.sqrt(5))/5;
	  double scaling;
	  Prototype crossover, resta, producto, mutant;
	  
	  for (int i=0; i<this.iterSFGSS; i++){ // Computation budjet
	  
		  fi1 = b - (b-a)/phi;
		  fi2 = a + (b-a)/phi;
		  
		  fitnessFi1 = lsff(fi1, CRi, population,actual,mejor, reduced,trainingDataSet);
		  fitnessFi2 = lsff(fi2, CRi,population,actual,mejor, reduced,trainingDataSet);
		  
		  if(fitnessFi1> fitnessFi2){
			  b = fi2;
		  }else{
			  a = fi1;  
		  }
	  
	  } // End While
	  
	  
	  if(fitnessFi1> fitnessFi2){
		  scaling = fi1;
	  }else{
		  scaling = fi2;
	  }
	  
	  
	  //Mutation:
	  mutant = new Prototype();
	  mutant = mutant(population, actual, mejor, scaling);
   	  
   	  //Crossover
   	  crossover =new Prototype(population[actual]);
   	  
	   for(int j=0; j< population[actual].numberOfInputs(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.setInput(j, mutant.getInput(j)); // Overwrite.
		   }
	   }
	   
	   
	  
	return crossover;
  }
  
  
  
  /**
   * SFHC local search  for FEATURE WEITHING
   * @param xt
   * @param xr
   * @param xs
   * @param actual
   * @param SFi
   * @return
   */
  
  public  Prototype SFHC(Prototype population[], int actual, int mejor, double SFi, double CRi, PrototypeSet reduced, PrototypeSet trainingDataSet){
	  double fitnessFi1, fitnessFi2, fitnessFi3, bestFi;
	  Prototype crossover, resta, producto, mutant;
	  double h= 0.5;
	  
	  
	  for (int i=0; i<this.iterSFHC; i++){ // Computation budjet
		  		  
		  fitnessFi1 = lsff(SFi-h, CRi, population,actual,mejor, reduced,trainingDataSet);
		  fitnessFi2 = lsff(SFi, CRi,  population,actual,mejor, reduced,trainingDataSet);
		  fitnessFi3 = lsff(SFi+h, CRi,  population,actual,mejor, reduced,trainingDataSet);
		  
		  if(fitnessFi1 >= fitnessFi2 && fitnessFi1 >= fitnessFi3){
			  bestFi = SFi-h;
		  }else if(fitnessFi2 >= fitnessFi1 && fitnessFi2 >= fitnessFi3){
			  bestFi = SFi;
			  h = h/2; // H is halved.
		  }else{
			  bestFi = SFi;
		  }
		  
		  SFi = bestFi;
	  }
	  
	  
	  //Mutation:
	  mutant = new Prototype();
	  mutant = mutant(population, actual, mejor, SFi);
	 
   	  //Crossover
   	  crossover = new Prototype(population[actual]);
   	  
	   for(int j=0; j< population[actual].numberOfInputs(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.setInput(j, mutant.getInput(j)); // Overwrite.
		   }
	   }
	   
	   
	  
	return crossover;
  
  }
  
  
  /**
   * Local Search Fitness Function for feature weighting
   * @param Fi
   * @param xt
   * @param xr
   * @param xs
   * @param actual
   */
  public double lsff(double Fi, double CRi, Prototype population[], int actual, int mejor, PrototypeSet reduced, PrototypeSet trainingDataSet){
	  Prototype resta, producto, mutant;
	  Prototype crossover;
	  double FitnessFi = 0;
	  
	  
	  //Mutation:
	  mutant = new Prototype();
   	  mutant = mutant(population, actual, mejor, Fi);
   	
   	  
   	  //Crossover
   	  crossover =new Prototype(population[actual]);
   	  
	   for(int j=0; j< population[actual].numberOfInputs(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.setInput(j, mutant.getInput(j)); // Overwrite.
		   }
	   }
	   
	   
	   // Compute fitness

       FitnessFi = GeometricMean(reduced, crossover ,trainingDataSet);
	   
   	   return FitnessFi;
  }
  
  public void inic_vector(int vector[]){

  	for(int i=0; i<vector.length; i++) vector[i] = i; // Lo inicializo de 1 a n-1
  }

  public void inic_vector_sin(int vector[], int without){

  	for(int i=0; i<vector.length; i++) 
  		if(i!=without)
  			vector[i] = i; // Lo inicializo de 1 a n-1
  }

  /**
   * Cuando quitas uno, con el inic vector, el desordenar no puede coger el �ltimo..
   * necesito otro me�todo
   * @param vector
   */
  public void desordenar_vector_sin(int vector[]){
  	int tmp, pos;
  	for(int i=0; i<vector.length-1; i++){
  		pos = Randomize.Randint(0, vector.length-1);
  		tmp = vector[i];
  		vector[i] = vector[pos];
  		vector[pos] = tmp;
  	}
  }

  public void desordenar_vector(int vector[]){
  	int tmp, pos;
  	for(int i=0; i<vector.length; i++){
  		pos = Randomize.Randint(0, vector.length-1);
  		tmp = vector[i];
  		vector[i] = vector[pos];
  		vector[pos] = tmp;
  	}
  }
  
  

}
