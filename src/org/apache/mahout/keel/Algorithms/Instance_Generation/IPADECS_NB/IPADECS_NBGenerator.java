
/*
	IPADECS_NB.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.

*/

package org.apache.mahout.keel.Algorithms.Instance_Generation.IPADECS_NB;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.C45.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.HandlerSMO;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Chen.ChenGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.HYB.HYBGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.PSO.PSOGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import java.util.*;

import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.*;

import org.core.*;

import org.core.*;

import java.util.StringTokenizer;



/**
 * @param k Number of neighbors
 * @param Population Size.
 * @param ParticleSize.
 * @param Scaling Factor.
 * @param Crossover rate.
 * @param Strategy (1-5).
 * @param MaxIter
 * @author Isaac Triguero
 * @version 1.0
 */
public class IPADECS_NBGenerator extends PrototypeGenerator {

  /*Own parameters of the algorithm*/
  
  // We need the variable K to use with k-NN rule
  private int k;
 
  private int PopulationSize; 
  private int ParticleSize;
  private int MaxIter; 
  
  private double Fl, Fu;
  private int iterBasicDE;
  private double ScalingFactor;
  private double CrossOverRate;
  private int Strategy;
  private String CrossoverType; // Binomial, Exponential, Arithmetic
  private double tau[] = new double[4];
  protected int numberOfClass;
  protected int numberOfPrototypes;  // Particle size is the percentage
  /** Parameters of the initial reduction process. */
  private String[] paramsOfInitialReducction = null;

  private int iterSFGSS;
  private int iterSFHC;
  private String classifier;
  protected int positiveClass;
  
  private boolean addRand;

  
  /**
   * Build a new IPADECS_NBGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param perc Reduction percentage of the prototype set.
   */
  
  public IPADECS_NBGenerator(PrototypeSet _trainingDataSet, int neigbors,int poblacion, int perc, int iteraciones, double F, double CR, int strg)
  {
      super(_trainingDataSet);
      algorithmName="IPADECS_NB";
      
      this.k = neigbors;
      this.PopulationSize = poblacion;
      this.ParticleSize = perc;
      this.MaxIter = iteraciones;
      this.numberOfPrototypes = getSetSizeFromPercentage(perc);
      
      this.ScalingFactor = F;
      this.CrossOverRate = CR;
      this.Strategy = strg;
      
  }
  


  /**
   * Build a new IPADECS_NBGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param params Parameters of the algorithm (only % of reduced set).
   */
  public IPADECS_NBGenerator(PrototypeSet t, Parameters parameters)
  {
      super(t, parameters);
      algorithmName="IPLDE2";
      this.k =  parameters.getNextAsInt();
     //  this.iterBasicDE =  parameters.getNextAsInt();//*trainingDataSet.get(0).numberOfInputs(); //NC*1000
      
      this.PopulationSize =  parameters.getNextAsInt();
      this.MaxIter =  parameters.getNextAsInt();
      this.iterSFGSS =  parameters.getNextAsInt();
      this.iterSFHC =  parameters.getNextAsInt();
      this.Fl = parameters.getNextAsDouble();
      this.Fu = parameters.getNextAsDouble();
      this.tau[0] =  parameters.getNextAsDouble();
      this.tau[1] =  parameters.getNextAsDouble();
      this.tau[2] =  parameters.getNextAsDouble();
      this.tau[3] =  parameters.getNextAsDouble();
      this.Strategy =  parameters.getNextAsInt();
      this.classifier = parameters.getNextAsString();
      
      String aleatorio = parameters.getNextAsString();
      
      if(aleatorio.equalsIgnoreCase("true")){
    	  this.addRand = true;
      }else{
    	  this.addRand = false;
      }
      
      
      this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
      
	  if(this.numberOfClass == 2){ // si es un problema binario (.. si es para nobalanceados.)
		  double min= Double.MAX_VALUE;
		  
		  for(int i=0; i<this.numberOfClass;i++){
			  if(trainingDataSet.getFromClass(i).size()<min){
				  min=trainingDataSet.getFromClass(i).size();
				  this.positiveClass= i;
			  }
		  }
		  
		  
	  }
	  
	  
      System.out.print("\nIsaac dice:  " + k + " Swar= "+PopulationSize+ " Particle=  "+ ParticleSize + " Maxiter= "+ MaxIter+" CR=  "+this.CrossOverRate+ " CrossverType = "+ this.CrossoverType+"\n");
      //numberOfPrototypes = getSetSizeFromPercentage(parameters.getNextAsDouble());
  }
  

   
  
  
  public PrototypeSet mutant(PrototypeSet population[], int actual, int mejor, double SFi){
  	  
  	  
  	  PrototypeSet mutant = new PrototypeSet(population.length);
  	  PrototypeSet r1,r2,r3,r4,r5, resta, producto, resta2, producto2, result, producto3, resta3;
  	  
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
  		   		 resta = r2.restar(r3);
  		   		 producto = resta.mulEscalar(SFi);
  		   		 mutant = producto.sumar(r1);
  		   	    break;
  			   
  		   	   case 2: // Vig = Xbest,G + F(Xr2,G - Xr3,G)  De best 1
  			   		 resta = r2.restar(r3);
  			   		 producto = resta.mulEscalar(SFi);
  			   		 mutant = population[mejor].sumar(producto);
  			   break;
  			   
  		   	   case 3: // Vig = ... De rand to best 1
  		   		   resta = r1.restar(r2); 
  		   		   resta2 = population[mejor].restar(population[actual]);
  		   		 			   		 
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   		
  			   	   result = population[actual].sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   		 			   		 
  			   break;
  			   
  		   	   case 4: // DE best 2
  		   		   resta = r1.restar(r2); 
  		   		   resta2 = r3.restar(r4);
  		   		 			   		 
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   		
  			   	   result = population[mejor].sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   break;
  			  
  		   	   case 5: //DE rand 2
  		   		   resta = r2.restar(r3); 
  		   		   resta2 = r4.restar(r5);
  		   		 			   		 
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   		
  			   	   result = r1.sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   	   
    		       break;
    		       
  		   	   case 6: //DE rand to best 2
  		   		   resta = r1.restar(r2); 
  		   		   resta2 = r3.restar(r4);
  		   		   resta3 = population[mejor].restar(population[actual]);
  		   		   
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   	   producto3 = resta3.mulEscalar(SFi);
  			   	   
  			   	   result = population[actual].sumar(producto);
  			   	   result = result.sumar(producto2);
  			   	   mutant = result.sumar(producto3);
    		       break;
    		       
  		   	  /*// Para hacer esta estratgia, lo que hay que elegir es CrossoverType = Arithmetic
  		   	   * case 7: //DE current to rand 1
  		   		   resta = r1.restar(population[actual]); 
  		   		   resta2 = r2.restar(r3);
  		   		 		   		 
  			   	   producto = resta.mulEscalar(RandomGenerator.Randdouble(0, 1));
  			   	   producto2 = resta2.mulEscalar(this.ScalingFactor);
  			   		
  			   	   result = population[actual].sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   	   
    		       break;
    		       */
  		   }   
  	   

  	  // System.out.println("********Mutante**********");
  	 // mutant.print();
  	   
       mutant.applyThresholds();
  	
  	  return mutant;
    }


    
    /**
     * Local Search Fitness Function
     * @param Fi
     * @param xt
     * @param xr
     * @param xs
     * @param actual
     */
    public double lsff(double Fi, double CRi, PrototypeSet population[], int actual, int mejor){
  	  PrototypeSet resta, producto, mutant;
  	  PrototypeSet crossover;
  	  double FitnessFi = 0;
  	  
  	  
  	  //Mutation:
  	  mutant = new PrototypeSet(population[actual].size());
     	  mutant = mutant(population, actual, mejor, Fi);
     	
     	  
     	  //Crossover
     	  crossover =new PrototypeSet(population[actual]);
     	  
  	   for(int j=0; j< population[actual].size(); j++){ // For each part of the solution
  		   
  		   double randNumber = RandomGenerator.Randdouble(0, 1);
  			   
  		   if(randNumber< CRi){
  			   crossover.set(j, mutant.get(j)); // Overwrite.
  		   }
  	   }
  	   
  	   
  	   // Compute fitness
  	   
	   // Compute fitness
	   PrototypeSet nominalPopulation = new PrototypeSet();
       nominalPopulation.formatear(crossover);
       FitnessFi =  AUC(nominalPopulation, trainingDataSet);//accuracy(nominalPopulation,trainingDataSet);
       /*
  	   PrototypeSet nominalPopulation = new PrototypeSet();
         nominalPopulation.formatear(crossover);
         FitnessFi = accuracy(nominalPopulation,trainingDataSet);
  	   */
     	   return FitnessFi;
    }
    
    
    
    /**
     * SFGSS local Search.
     * @param population
     * @return
     */
    public PrototypeSet SFGSS(PrototypeSet population[], int actual, int mejor, double CRi){
  	  double a=0.1, b=1;
  	  double fi1=0, fi2=0, fitnessFi1=0, fitnessFi2=0;
  	  double phi = (1+ Math.sqrt(5))/5;
  	  double scaling;
  	  PrototypeSet crossover, resta, producto, mutant;
  	  
  	  for (int i=0; i<this.iterSFGSS; i++){ // Computation budjet
  	  
  		  fi1 = b - (b-a)/phi;
  		  fi2 = a + (b-a)/phi;
  		  
  		  fitnessFi1 = lsff(fi1, CRi, population,actual,mejor);
  		  fitnessFi2 = lsff(fi2, CRi,population,actual,mejor);
  		  
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
  	  mutant = new PrototypeSet(population[actual].size());
  	  mutant = mutant(population, actual, mejor, scaling);
     	  
     	  //Crossover
     	  crossover =new PrototypeSet(population[actual]);
     	  
  	   for(int j=0; j< population[actual].size(); j++){ // For each part of the solution
  		   
  		   double randNumber = RandomGenerator.Randdouble(0, 1);
  			   
  		   if(randNumber< CRi){
  			   crossover.set(j, mutant.get(j)); // Overwrite.
  		   }
  	   }
  	   
  	   
  	  
  	return crossover;
    }
    
    /**
     * SFHC local search
     * @param xt
     * @param xr
     * @param xs
     * @param actual
     * @param SFi
     * @return
     */
    
    public  PrototypeSet SFHC(PrototypeSet population[], int actual, int mejor, double SFi, double CRi){
  	  double fitnessFi1, fitnessFi2, fitnessFi3, bestFi;
  	  PrototypeSet crossover, resta, producto, mutant;
  	  double h= 0.5;
  	  
  	  
  	  for (int i=0; i<this.iterSFHC; i++){ // Computation budjet
  		  		  
  		  fitnessFi1 = lsff(SFi-h, CRi, population,actual,mejor);
  		  fitnessFi2 = lsff(SFi, CRi,  population,actual,mejor);
  		  fitnessFi3 = lsff(SFi+h, CRi,  population,actual,mejor);
  		  
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
  	  mutant = new PrototypeSet(population[actual].size());
  	  mutant = mutant(population, actual, mejor, SFi);
  	 
     	  //Crossover
     	  crossover = new PrototypeSet(population[actual]);
     	  
  	   for(int j=0; j< population[actual].size(); j++){ // For each part of the solution
  		   
  		   double randNumber = RandomGenerator.Randdouble(0, 1);
  			   
  		   if(randNumber< CRi){
  			   crossover.set(j, mutant.get(j)); // Overwrite.
  		   }
  	   }
  	   
  	   
  	  
  	return crossover;
    
    }
  
  

    public int[] classify(PrototypeSet training, PrototypeSet test)
    {
  	int predicho[] = new int[test.size()];
  	
  	
  	if(this.classifier.equals("NN")){
  	
  	  int i=0;
  		
	  if(training.size()>this.k){
	      predicho = KNN.classify2(training, test, this.k).clone();
	  }else{
	  
	      for(Prototype p : test)
	      {
	          Prototype nearestNeighbor = KNN._1nn(p, training);          
	    	      	  
	          predicho[i] = (int) nearestNeighbor.getOutput(0);
	                   
	          i++;
	      }
	  }
  		
  	}else if(this.classifier.equals("C45")){
  		C45 c45;
  		  
  		try {
  			 /// training.save("train1.dat");
  			 // test.save("test1.dat");
  			//  c45 = new C45("train1.dat", "test1.dat");
  			c45 = new C45(training.toInstanceSet(), test.toInstanceSet());
  		    predicho = c45.getPredictions().clone();   
  		    
		    c45 = null;
		    System.gc();
		    
  		} catch (Exception e) {
  			// TODO Auto-generated catch block
  			e.printStackTrace();
  		}      // C4.5 called
      
  	}else if(this.classifier.equals("SMO")){
  	    HandlerSMO SMO;
  	    
  		try {
  			
  			 // training.save("train1.dat");
  			  //test.save("test1.dat");
  			  //SMO = new HandlerSMO(this.numberOfClass, test.size(),String.valueOf(this.SEED));
  			  SMO = new HandlerSMO(training.toInstanceSet(), test.toInstanceSet(),this.numberOfClass,String.valueOf(this.SEED));
  			
  			  //SMO.generateFiles();
  			  
  		      predicho = SMO.getPredictions(0).clone();    
  		     
  		    SMO = null;
		    System.gc();
		    
  		      /*
  		      for(int i=0; i<test.size(); i++){
  		    	  System.out.print(predicho[i]+", ");
  		      }
  		      System.out.println(" ");
  		      */
  		} catch (Exception e) {
  			// TODO Auto-generated catch block
  			e.printStackTrace();
  		}      // SMO
  	      
  	
  	}
      
        return predicho;
    }
  
  

    public double TPrate(PrototypeSet test, int [] predicho){

  	  
  	  double tp=0.0,fn=0.0;
  	  
  	  for(int i=0; i<test.size();i++){
  		  
  		  if(test.get(i).getOutput(0)==predicho[i] && predicho[i]==this.positiveClass){ // esto es un tp
  			  tp++;			  
  		  }else if(test.get(i).getOutput(0)!=predicho[i] && test.get(i).getOutput(0)==this.positiveClass){ // es un false
  			  fn++;
  		  }
  	  }
  	  
  	  return ((1.*tp)/((tp+fn)*1.));
  	  
    }
    
    
    public double FPrate(PrototypeSet test, int [] predicho){

  	  
  	  double fp=0.0,tn=0.0;
  	  
  	  for(int i=0; i<test.size();i++){
  		  
  		  if(test.get(i).getOutput(0)!=predicho[i] && predicho[i]==this.positiveClass){ // esto es un fp
  			  fp++;			  
  		  }else if(test.get(i).getOutput(0)==predicho[i] && test.get(i).getOutput(0)!=this.positiveClass){ // es un tn
  			  tn++;
  		  }
  	  }
  	  
  	  return ((1.*fp)/((fp+tn)*1.));
  	  
    }
    
    
    public double AUC(PrototypeSet train, PrototypeSet test){
  	  double AUC;
  	  
  	  int []pre = classify(train,test);
  	  
  	  double tprate= this.TPrate(test, pre);
  	  double fprate= this.FPrate(test, pre);
  	  
  	  AUC = (1.0+tprate-fprate)/2.0;
  	  
  	 // System.out.println("AUC = "+(1.0+tprate-fprate)/2.0);;
  	  
  	  return AUC;
  	  
    }
  
  public PrototypeSet basicDE(PrototypeSet initial){ //, double claseOptimizar
	  //System.out.print("\nThe algorithm  SFLSDE is starting...\n Computing...\n");
	  
	  //Algorithm
	  // First, we create the population, with PopulationSize.
	  // like a prototypeSet's vector.  

	  PrototypeSet population [] = new PrototypeSet [PopulationSize];
	  PrototypeSet mutation[] = new PrototypeSet[PopulationSize];
	  PrototypeSet crossover[] = new PrototypeSet[PopulationSize];
	  
	  
	  double ScalingFactor[] = new double[this.PopulationSize];
	  double CrossOverRate[] = new double[this.PopulationSize]; // Inside of the Optimization process.
	  double fitness[] = new double[PopulationSize];

	  double fitness_bestPopulation[] = new double[PopulationSize];
	  PrototypeSet bestParticle = new PrototypeSet();
	  
	
  
	  //Each particle must have   Particle Size %

	  // First Stage, Initialization.
	  
	  PrototypeSet nominalPopulation;
	  
	  population[0]= new PrototypeSet(initial.clone()) ;

	   nominalPopulation = new PrototypeSet();
       nominalPopulation.formatear(population[0]);
       
	  fitness[0] = AUC(nominalPopulation, trainingDataSet);// accuracy(nominalPopulation,trainingDataSet);
	  
	  System.out.println("Initial Optim: Fitness "+ fitness[0]);	
	  
	 // System.out.println("Best initial fitness = "+ fitness[0]);

	  this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
	  
	  
      for(int i=1; i< PopulationSize; i++){
		  population[i] = new PrototypeSet();
		  for(int j=0; j< population[0].size(); j++){
			  Prototype aux = new Prototype(trainingDataSet.getFromClass(population[0].get(j).getOutput(0)).getRandom());
			  population[i].add(aux);
		  }
		  
		  nominalPopulation = new PrototypeSet();
	      nominalPopulation.formatear(population[i]);
	      
		  fitness[i] = AUC(nominalPopulation, trainingDataSet);// accuracy(population[i],trainingDataSet);   // PSOfitness
		  fitness_bestPopulation[i] = fitness[i]; // Initially the same fitness.
	  }
	  
	  
	  //We select the best initial  particle
	 double bestFitness=fitness[0];
	  int bestFitnessIndex=0;
	  for(int i=1; i< PopulationSize;i++){
		  if(fitness[i]>bestFitness){
			  bestFitness = fitness[i];
			  bestFitnessIndex=i;
		  }
		  
	  }
	  
	   for(int j=0;j<PopulationSize;j++){
         //Now, I establish the index of each prototype.
		  for(int i=0; i<population[j].size(); ++i)
			  population[j].get(i).setIndex(i);
	   }
	   
	   
	   // Initially the Scaling Factor and crossover for each Individual are randomly generated between 0 and 1.
	   
	   for(int i=0; i< this.PopulationSize; i++){
		   ScalingFactor[i] =  RandomGenerator.Randdouble(0, 1);
		   CrossOverRate[i] =  RandomGenerator.Randdouble(0, 1);
	   }
	   
	   
	  	   
	   double randj[] = new double[5];
	   
	   
	   for(int iter=0; iter< MaxIter; iter++){ // Main loop
		      
		   for(int i=0; i<PopulationSize; i++){

			   // Generate randj for j=1 to 5.
			   for(int j=0; j<5; j++){
				   randj[j] = RandomGenerator.Randdouble(0, 1);
			   }
			   
					   
    			   	    
			   
			   if(i==bestFitnessIndex && randj[4] < tau[2]){
				  // System.out.println("SFGSS applied");
				   //SFGSS
				   crossover[i] = SFGSS(population, i, bestFitnessIndex, CrossOverRate[i]);
				   
				   
			   }else if(i==bestFitnessIndex &&  tau[2] <= randj[4] && randj[4] < tau[3]){
				   //SFHC
				   //System.out.println("SFHC applied");
				   crossover[i] = SFHC(population, i, bestFitnessIndex, ScalingFactor[i], CrossOverRate[i]);
				   
			   }else {
				   
				   // Fi update
				   
				   if(randj[1] < tau[0]){
					   ScalingFactor[i] = this.Fl + this.Fu*randj[0];
				   }
				   
				   // CRi update
				   
				   if(randj[3] < tau[1]){
					   CrossOverRate[i] = randj[2];
				   }
				   				   
				   // Mutation Operation.
				   
				   mutation[i] = new PrototypeSet(population[i].size());
			   
				  //Mutation:
					
				   mutation[i]  = mutant(population, i, bestFitnessIndex, ScalingFactor[i]);
				   
				    // Crossver Operation.

				   crossover[i] = new PrototypeSet(population[i]);
				   
				   for(int j=0; j< population[i].size(); j++){ // For each part of the solution
					   
					   double randNumber = RandomGenerator.Randdouble(0, 1);
						   
					   if(randNumber<CrossOverRate[i]){
						   crossover[i].set(j, mutation[i].get(j)); // Overwrite.
					   }
				   }
				   
				   
				   
				   
			   }
			   
   
			   
			   // Fourth: Selection Operation.
		   
			   nominalPopulation = new PrototypeSet();
		       nominalPopulation.formatear(population[i]);
		       fitness[i] =  AUC(nominalPopulation, trainingDataSet); // accuracy(nominalPopulation,trainingDataSet);
		       
		       nominalPopulation = new PrototypeSet();
		       nominalPopulation.formatear(crossover[i]);
		       
			   double trialVector =  AUC(nominalPopulation, trainingDataSet);//accuracy(nominalPopulation,trainingDataSet);
			
		  
			  if(trialVector > fitness[i]){
				  population[i] = new PrototypeSet(crossover[i]);
				  fitness[i] = trialVector;
			  }
			  
			  if(fitness[i]>bestFitness){
				  System.out.println("BASICDE: update fitness: "+ fitness[i]);
				  bestFitness = fitness[i];
				  bestFitnessIndex=i;
			  }
			  
			  
		   }

		   
	   }

	   
		   nominalPopulation = new PrototypeSet();
           nominalPopulation.formatear(population[bestFitnessIndex]);
		  // System.err.println("\n% de acierto en training Nominal " + accuracy(nominalPopulation,trainingDataSet) );
			  
			//  nominalPopulation.print();

  
		return nominalPopulation;
  }
  

  
  
  /**
   * Generate a reduced prototype set by the IPLDE2Generator method.
   * @return Reduced set by IPLDE2Generator's method.
   */
  
  
  public PrototypeSet reduceSetNB()
  {
	  System.out.print("\nThe algorithm  IPLDE2 is starting...\n Computing...\n");
	  this.Strategy = 3;
	  
	  PrototypeGenerator.classifier = this.classifier; // IMPORTANT TO DO A GOOD FINAL CLASSIFICATION!
	  PrototypeGenerator.kNearest = this.k;
	  
	  PrototypeSet solucion = new PrototypeSet();
	  
	  PrototypeSet Clases [] = new PrototypeSet[this.numberOfClass];
	  double fitnessClass[] = new double[this.numberOfClass];
	  PrototypeSet nominalPopulation;
	  
	  
	  
	  
	 //PrototypeSet clean = new PrototypeSet(ENN(trainingDataSet)); // para evitar que los
	  // posibles outlier afecten al centroide
	  
	  // We order by size. Starting from the smaller class.
	  for(int i=0; i<this.numberOfClass; i++){
		 
		  
		  if(trainingDataSet.getFromClass(i).size() >0){ 
			  Clases[i] = new PrototypeSet(trainingDataSet.getFromClass(i).clone());
			  
			  //System.out.println("Size ->"+Clases[i].size());
			  Prototype centroid = Clases[i].avg();
			  //centroid.print();
			  solucion.add(centroid); // Centroide
			  
		  }
		 
	  }
	  
	  
	  int iteraciones = this.iterBasicDE;
	  
	  this.iterBasicDE= 100; // FIRST ITERATION IS DIFFERENT
	  solucion=basicDE(solucion); //Initial Optimization
	 // solucion.print();
	  this.iterBasicDE= iteraciones;
	  
	  double Fitness=  AUC(solucion, trainingDataSet); // accuracy(solucion,trainingDataSet);
	  System.out.println("Initial Global Fitness = "+ Fitness);
	  
	  boolean claseMarcada[] = new boolean[this.numberOfClass];
	  boolean fin[] = new boolean[this.numberOfClass];
	  Arrays.fill(claseMarcada, false);
	  Arrays.fill(fin, true);
	  
	  int iterOptimizada[] = new int [this.numberOfClass];
	  Arrays.fill(iterOptimizada, 1);
	  
	  
	  int contOptimizedPositive[] = new int[this.numberOfClass];
	  Arrays.fill(contOptimizedPositive, 0);
	  
	  PrototypeSet solRescatada = null;
	  
	  while(!Arrays.equals(claseMarcada, fin)){	  
		  
		  double minFitness= Double.MAX_VALUE;
		  int objetivo= -1;
		  
		  for(int j=0; j<this.numberOfClass; j++){
			  if(trainingDataSet.getFromClass(j).size()>1){
				  
				  
				  fitnessClass[j]=accuracy(solucion,trainingDataSet.getFromClass(j));
				   
				  //System.out.println("Fitness class["+j+"]= " +fitnessClass[j]);
				  
				  
				  if(fitnessClass[j] < minFitness && !claseMarcada[j]){
					  minFitness = fitnessClass[j];
					  objetivo = j;
				  }
				  
				  if(fitnessClass[j] == 100){
					  claseMarcada[j] = true;
				  }
			  }else{
				  claseMarcada[j] = true; // Si solo tiene un protitpo de esa clase. ya hemos acabado.
			  } 
		   }
		 
			  
		 // System.out.println("Objetivo =" + objetivo);
		  PrototypeSet tester;
		 
		  
		 
			  if(!claseMarcada[objetivo]){
				  PrototypeSet solucion2;
				  
				  if(objetivo==this.positiveClass &&  contOptimizedPositive[objetivo]> 0){ // lo que persigo es, si no he mejorado, me faltan iteraciones, solo se le permite a la clase minoritaria
					  solucion2 = new PrototypeSet(solRescatada.clone());
					  
				  }else{
					  solucion2 = new PrototypeSet(solucion.clone());
					  
					  Prototype Addition =null;
					  
					  if(this.addRand || objetivo!=this.positiveClass){
					  	  Addition= trainingDataSet.getFromClass(objetivo).getRandom();
					  
					  }else{ // Quiero coger el más lejano de trainingDataSet.getFromClass(objetivo)  a todo lo que yo tengo ahora mismo.
						  PrototypeSet delaClase = trainingDataSet.getFromClass(objetivo);
						  
						  int lejano =0;
						  double distLejano = Double.MAX_VALUE;
						  for(int z=0; z< delaClase.size();z++){
							  
							  double dist = 0;
							  for(int h=0; h<solucion2.size(); h++){
								  
								  double diz= Distance.absoluteDistance(solucion2.get(h), delaClase.get(z));
								  if(diz!=0){
									  dist+=diz;
								  }else{
									  dist+=Double.MAX_VALUE;
								  }
							  }
							  
							  if(dist<distLejano && dist !=0){
								  distLejano = dist;
								  lejano=z;
							  }
						  }
						  
						  System.out.println("Lejano "+ lejano);
						  Addition= trainingDataSet.getFromClass(objetivo).get(lejano);
					  }
					  
					  solucion2.add(new Prototype(Addition)); // Anado uno Y pruebo a optimizar.
				 // solucion2.add(trainingDataSet.farthestTo(solucion.getFromClass(objetivo).getRandom())); 
				  }
				  

				  //solucion2.print();
				  tester = basicDE(solucion2).clone();
			  		  
				   nominalPopulation = new PrototypeSet();
			       nominalPopulation.formatear(solucion);
			       Fitness= AUC(nominalPopulation, trainingDataSet);//accuracy(nominalPopulation,trainingDataSet);
			       
			       nominalPopulation = new PrototypeSet();
			       nominalPopulation.formatear(tester);
				  double trialFitness= AUC(nominalPopulation, trainingDataSet);

				  System.out.println("Trial fitnss= " + trialFitness);
				  if(trialFitness > Fitness ){ // le ponemos el >=, aquí no hay miedo por la reducción.
					    
					  iterOptimizada[objetivo]++;
					  solucion = new PrototypeSet(tester.clone());
					  Fitness = trialFitness;
					  contOptimizedPositive[objetivo]= 0; // reinicializo la cuenta.
					     System.out.println("añado de la clase ->" +objetivo);
				  }else if(trialFitness==Fitness && objetivo==this.positiveClass && iterOptimizada[objetivo]<(trainingDataSet.getFromClass(objetivo).size())){
					  
					  iterOptimizada[objetivo]++;
				      solucion = new PrototypeSet(tester.clone());
				      Fitness = trialFitness;
				  
				      System.out.println("añado de la clase AQUI ->" +objetivo);
			     }else{
			    	 if(objetivo==this.positiveClass){
			    		 
			    		 solRescatada = new PrototypeSet(tester.clone());
					  contOptimizedPositive[objetivo]++; // contar que la clase positiva no ha sido optimizada. 
					  
					  System.out.println("cont ->" +contOptimizedPositive[objetivo]);
					  if(contOptimizedPositive[objetivo]>=10){ // si ya van 5 veces que no consigo mejorarla, bueno, pues paramos.
						  claseMarcada[objetivo] = true;
					  }
			    	 }else{
			    		 claseMarcada[objetivo] = true;
			    	 }
				  }
				  
			  }
		  
			  //Fitness= accuracy(solucion,trainingDataSet);
			  System.out.println("Fitness =  "+ Fitness);

		  
		  
	  }
	  
  
	  
	  nominalPopulation = new PrototypeSet();
      nominalPopulation.formatear(solucion);
	  double trialFitness=  AUC(nominalPopulation, trainingDataSet);// accuracy(nominalPopulation,trainingDataSet);
	  
	  System.out.println("Final Fitness = "+ trialFitness);
	  System.out.println("Reduction %, result set = "+((trainingDataSet.size()-solucion.size())*100.)/trainingDataSet.size()+ "\n");
	  

	  solucion.print();
	  
     return nominalPopulation;
  }
  
  /**
   * General main for all the prototoype generators
   * @param args Arguments of the main function.
 * @throws Exception 
   */
  public static void main(String[] args) throws Exception
  {
      Parameters.setUse("IPADECS_NB", "<seed> <Number of neighbors>\n<Swarm size>\n<Particle Size>\n<MaxIter>\n<DistanceFunction>");        
      Parameters.assertBasicArgs(args);
      
      PrototypeSet training = PrototypeGenerationAlgorithm.readPrototypeSet(args[0]);
      PrototypeSet test = PrototypeGenerationAlgorithm.readPrototypeSet(args[1]);
      
      
      long seed = Parameters.assertExtendedArgAsInt(args,2,"seed",0,Long.MAX_VALUE);
      IPADECS_NBGenerator.setSeed(seed);
      
      int k = Parameters.assertExtendedArgAsInt(args,3,"number of neighbors", 1, Integer.MAX_VALUE);
      int swarm = Parameters.assertExtendedArgAsInt(args,4,"swarm size", 1, Integer.MAX_VALUE);
      int particle = Parameters.assertExtendedArgAsInt(args,5,"particle size", 1, Integer.MAX_VALUE);
      int iter = Parameters.assertExtendedArgAsInt(args,6,"max iter", 1, Integer.MAX_VALUE);

      
      //String[] parametersOfInitialReduction = Arrays.copyOfRange(args, 4, args.length);
     //System.out.print(" swarm ="+swarm+"\n");
      
      
      IPADECS_NBGenerator generator = new IPADECS_NBGenerator(training, k,swarm,particle,iter, 0.5,0.5,1);
      
  	  
      PrototypeSet resultingSet = generator.execute();
      
  	//resultingSet.save(args[1]);
      //int accuracyKNN = KNN.classficationAccuracy(resultingSet, test, k);
      int accuracy1NN = KNN.classficationAccuracy(resultingSet, test);
      generator.showResultsOfAccuracy(Parameters.getFileName(), accuracy1NN, test);
  }

}
