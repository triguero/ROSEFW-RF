
/*
	CoDE.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  6-2-2011
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.

*/

package org.apache.mahout.keel.Algorithms.Instance_Generation.CoDE;

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
public class CoDEGenerator extends PrototypeGenerator {

  /*Own parameters of the algorithm*/
  
	
 private int MAX_ITER;
 private double Beta;
  // We need the variable K to use with k-NN rule
  private int k;
 
  private int PopulationSize; 
  private int ParticleSize;
  private int MaxIter; 
  private int Strategy;

  
  private String tipoFitness;
  
  protected int numberOfClass;
  protected int numberOfPrototypes;  // Particle size is the percentage
  /** Parameters of the initial reduction process. */
  private String[] paramsOfInitialReducction = null;

  private double tau[] = new double[4];
  private double Fl, Fu;
  
  private int iterSFGSS;
  private int iterSFHC;
  
  /**
   * Build a new CoDEGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param perc Reduction percentage of the prototype set.
   */
  
  public CoDEGenerator(PrototypeSet _trainingDataSet, int neigbors,int poblacion, int perc, int iteraciones, double F, double CR, int strg)
  {
      super(_trainingDataSet);
      algorithmName="CoDE";
      
      this.k = neigbors;
      this.PopulationSize = poblacion;
      this.ParticleSize = perc;
      this.MaxIter = iteraciones;
      this.numberOfPrototypes = getSetSizeFromPercentage(perc);
      
  }
  


  /**
   * Build a new CoDEGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param params Parameters of the algorithm (only % of reduced set).
   */
  public CoDEGenerator(PrototypeSet t, Parameters parameters)
  {
      super(t, parameters);
      algorithmName="CoDE";
      
      this.k =  parameters.getNextAsInt();
      this.MAX_ITER = parameters.getNextAsInt();
      this.PopulationSize =  parameters.getNextAsInt();
      this.ParticleSize =  parameters.getNextAsInt();
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
      this.Beta =  parameters.getNextAsDouble(); // 0.5;
      this.tipoFitness = parameters.getNextAsString();
      
      this.numberOfPrototypes = getSetSizeFromPercentage(ParticleSize);
      this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
      System.out.print("\nIsaac dice:  " + k + " Swar= "+PopulationSize+ " Particle=  "+ ParticleSize + " Maxiter= "+ MaxIter+" tau4=  "+this.tau[3]+ "\n");
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
  		       
		   	  /*// Para hacer esta estrat�gia, lo que hay que elegir es CrossoverType = Arithmetic
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
  public double lsff(double Fi, double CRi, PrototypeSet population[][], int [] bestIndividual, int claseObjetivo, int actual, int mejor){
	  PrototypeSet resta, producto, mutant;
	  PrototypeSet crossover;
	  double FitnessFi = 0;
	  
	  
	  //Mutation:
	  mutant = new PrototypeSet(population[claseObjetivo][actual].size());
   	  mutant = mutant(population[claseObjetivo], actual, mejor, Fi);
   	
   	  
   	  //Crossover
   	  crossover =new PrototypeSet(population[claseObjetivo][actual]);
   	  
	   for(int j=0; j< population[claseObjetivo][actual].size(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.set(j, mutant.get(j)); // Overwrite.
		   }
	   }
	   
	   
	   // Compute fitness
	   PrototypeSet nominalPopulation = new PrototypeSet();
       nominalPopulation.formatear(crossover);
      
       
       PrototypeSet guardaPopulation  = new PrototypeSet(population[claseObjetivo][actual]);
       
       population[(int) claseObjetivo][actual] = new PrototypeSet(nominalPopulation);
       
       FitnessFi = fitnessFunction(population, bestIndividual ,claseObjetivo,actual); 
       
       
       population[(int) claseObjetivo][actual] = new PrototypeSet(guardaPopulation.clone()); //restarurar
       
       //FitnessFi = accuracy(nominalPopulation,trainingDataSet);
	   
   	   return FitnessFi;
  }
  
  
  
  /**
   * SFGSS local Search.
   * @param population
   * @return
   */
  

  
  public PrototypeSet SFGSS(PrototypeSet population[][], int [] bestIndividual, int claseObjetivo, int actual, int mejor, double CRi){
	  double a=0.1, b=1;
	  double fi1=0, fi2=0, fitnessFi1=0, fitnessFi2=0;
	  double phi = (1+ Math.sqrt(5))/5;
	  double scaling;
	  PrototypeSet crossover, resta, producto, mutant;
	  
	  for (int i=0; i<this.iterSFGSS; i++){ // Computation budjet
	  
		  fi1 = b - (b-a)/phi;
		  fi2 = a + (b-a)/phi;
		  
		  fitnessFi1 = lsff(fi1, CRi, population,bestIndividual, claseObjetivo,actual,mejor);
		  fitnessFi2 = lsff(fi2, CRi,population,bestIndividual, claseObjetivo, actual,mejor);
		  
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
	  mutant = new PrototypeSet(population[claseObjetivo][actual].size());
	  mutant = mutant(population[claseObjetivo], actual, mejor, scaling);
   	  
   	  //Crossover
   	  crossover =new PrototypeSet(population[claseObjetivo][actual]);
   	  
	   for(int j=0; j< population[claseObjetivo][actual].size(); j++){ // For each part of the solution
		   
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
  
  public  PrototypeSet SFHC(PrototypeSet population[][], int [] bestIndividual, int claseObjetivo,  int actual, int mejor, double SFi, double CRi){
	  double fitnessFi1, fitnessFi2, fitnessFi3, bestFi;
	  PrototypeSet crossover, resta, producto, mutant;
	  double h= 0.5;
	  
	  
	  for (int i=0; i<this.iterSFHC; i++){ // Computation budjet
		  		  
		  fitnessFi1 = lsff(SFi-h, CRi, population,bestIndividual, claseObjetivo,actual,mejor);
		  fitnessFi2 = lsff(SFi, CRi,  population,bestIndividual, claseObjetivo,actual,mejor);
		  fitnessFi3 = lsff(SFi+h, CRi,  population,bestIndividual, claseObjetivo,actual,mejor);
		  
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
	  mutant = new PrototypeSet(population[claseObjetivo][actual].size());
	  mutant = mutant(population[claseObjetivo], actual, mejor, SFi);
	 
   	  //Crossover
   	  crossover = new PrototypeSet(population[claseObjetivo][actual]);
   	  
	   for(int j=0; j< population[claseObjetivo][actual].size(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.set(j, mutant.get(j)); // Overwrite.
		   }
	   }
	   
	   
	  
	return crossover;
  
  }
  
  
  /** Main method */
  
  public PrototypeSet reduceSet(){
	  
	  
	  // A population per class.
	  
	  PrototypeSet population[][] = new PrototypeSet[this.numberOfClass][PopulationSize];
	  double localAcc[][] = new double[this.numberOfClass][PopulationSize];
	  double Acc[] = new double[PopulationSize];
	  
	  int bestIndividual[] = new int[this.numberOfClass];
	  double bestFitness[] = new double[this.numberOfClass];
	  
	  // First Stage, Initialization.
	  //Each class has a PopulationSize particles. 
	  
 // decide the structure of the particle.
	  	
	  PrototypeSet randomize=selecRandomSet(numberOfPrototypes,true).clone() ;
	  
	  
	  // Aseguro que al menos hay un representante de cada clase.
	  PrototypeSet clases[] = new PrototypeSet [this.numberOfClass];
	  for(int i=0; i< this.numberOfClass; i++){
		  clases[i] = new PrototypeSet(trainingDataSet.getFromClass(i));
	  }
	
	  for(int i=0; i< randomize.size(); i++){
		  for(int j=0; j< this.numberOfClass; j++){
			  if(randomize.getFromClass(j).size() ==0 && clases[j].size()!=0){
				  
				  randomize.add(clases[j].getRandom());
			  }
		  }
	  }
	  
	  // Y ahora divido en los por clases!
	  
	  for(int i=0; i< this.numberOfClass; i++){
		  
		  if(randomize.getFromClass(i).size()>0){
			  population[i][0]= new PrototypeSet(randomize.getFromClass(i).clone());
		   }else{
			   population[i][0] = null; // Sometime it can occur
			   
		   }
	  }
	  
	  
	  // El resto de particulas son iguales!!
	  
		  for(int j=1; j<this.PopulationSize; j++){
	
		  
			  for(int i=0; i< this.numberOfClass; i++){
				  
				  if(population[i][0]!=null){
					  population[i][j]= new PrototypeSet();
					
					  for(int z=0; z< population[i][0].size(); z++){
						  population[i][j].add(trainingDataSet.getFromClass(i).getRandom());
					  }
		  		  }	  
			  }
		  }
		  

	  // Calculate initial fitness
	  
	  for(int i=0; i< this.numberOfClass; i++){
		  bestIndividual[i] = 0;
		  bestFitness[i] = Double.MIN_VALUE;
		  
		  if(population[i][0]!=null){
			  for(int j=0; j<this.PopulationSize; j++){
				  
				  double fitness= initialfitnessFunction(population, i, j);
				  
				  if(fitness > bestFitness[i]){
					  bestFitness[i] = fitness;
					  bestIndividual[i] = j;
				  }
				  
				  
			  }
		  }
	  }
	  
	  
  
	  
	  // Co-evolutionary stage
	  
	  int iter =0;
	  
		while(iter<MAX_ITER){
			
			for(int i=0; i<this.numberOfClass;i++){
				 // Do generation...population[i][j]
				if(population[i][0]!=null){
					population[i]= doGeneration(population, i, bestIndividual);	
				}
			}
			
			//updateCollaborators; // who is the best ?
			
			  for(int i=0; i< this.numberOfClass; i++){
				  
				  if(population[i][0]!=null){
					  
				 
					  for(int j=0; j<this.PopulationSize; j++){  //REPASAR
						  
						  double fitness= fitnessFunction(population, bestIndividual, i, j);
						  
						  if(fitness > bestFitness[i]){
							  bestFitness[i] = fitness;
							  bestIndividual[i] = j;
						  }
						  
						  
					  }
				  
				  }
			  }
			  
			  
			
			iter++;
		}
	  
		
		// Generate Final reference set.
		
		
		  PrototypeSet Join = new PrototypeSet();
		  for(int i=0; i<this.numberOfClass;i++){
			  if(population[i][0]!=null){
				  Join.add(population[i][bestIndividual[i]]); 
			  }
		  }
		

		   PrototypeSet nominalPopulation = new PrototypeSet();
           nominalPopulation.formatear(Join);
           
    
    	 System.err.println("\n% de acierto en training Nominal " + KNN.classficationAccuracy(nominalPopulation,trainingDataSet,1)*100./trainingDataSet.size() );
    	 System.out.println("Reduction % " + (100.-(nominalPopulation.size()*100.)/trainingDataSet.size()) );
           
		
	  
	  return nominalPopulation;
  }
  
  /**
   * Initial fitness fuction
   * @param population
   * @param claseObjetivo
   * @param particle
   * @return
   */
  
  public double initialfitnessFunction(PrototypeSet population[][], double claseObjetivo, int particle){
	  double fitness =0;
	  
	  PrototypeSet Join = new PrototypeSet();
	  for(int i=0; i<this.numberOfClass;i++){
		  if(population[i][0]!=null){
			  Join.add(population[i][particle]);
		  }
	  }

	  double acc[] = new double[this.numberOfClass];
	  double global = 1;
	  
	  
	  for(int i=0; i<this.numberOfClass;i++){
		  acc[i] = accuracy(Join,trainingDataSet.getFromClass(i)); // local accss
	  }
	  
	  if(this.tipoFitness.equalsIgnoreCase("Weighted")){
		  global = accuracy(Join,trainingDataSet);
		  
	  }else if(this.tipoFitness.equalsIgnoreCase("GeometricMean")){
		 // AccLocal: Acc1
		  // AccGlobal:  Math.sqrt(Acc0*Acc1*Acc2)
		  for(int i=0; i<this.numberOfClass;i++){
			  global *= acc[i];
		  }
		  global = Math.sqrt(global);
		  
	  }
	 
	  fitness = this.Beta*global+ (1-this.Beta)*acc[(int) claseObjetivo]; 
	  
	  return fitness;
	  
  }
  
  /**
   * Fitness function
   * @param population
   * @param bestIndividual
   * @param claseObjetivo
   * @param particle
   * @return
   */
  
  public double fitnessFunction(PrototypeSet population[][], int bestIndividual[], double claseObjetivo, int particle){
	  double fitness =0;
	  
	  if(population[(int) claseObjetivo][0]!=null){
	  
		  PrototypeSet Join = new PrototypeSet(population[(int) claseObjetivo][particle]);
		  
		  for(int i=0; i<this.numberOfClass;i++){
			  if(i!= claseObjetivo && population[i][0]!=null ){
				  Join.add(population[i][bestIndividual[i]]);  
			  }
		  }
		  
		  double acc[] = new double[this.numberOfClass];
		  double global = 1;
		  
		  
		  for(int i=0; i<this.numberOfClass;i++){
			  acc[i] = accuracy(Join,trainingDataSet.getFromClass(i)); // local accss
		  }
		  
		  if(this.tipoFitness.equalsIgnoreCase("Weighted")){
			  global = accuracy(Join,trainingDataSet);
		  }else if(this.tipoFitness.equalsIgnoreCase("GeometricMean")){
			 // AccLocal: Acc1
			  // AccGlobal:  Math.sqrt(Acc0*Acc1*Acc2)
			  for(int i=0; i<this.numberOfClass;i++){
				  global *= acc[i];
			  }
			  global = Math.sqrt(global);
			  
		  }
		 
		  fitness = this.Beta*global+ (1-this.Beta)*acc[(int) claseObjetivo]; 
		  
		  
		  
	  }
	  return fitness;
	  
  }
  
  /**
   * Generate a reduced prototype set by the CoDEGenerator method.
   * @return Reduced set by CoDEGenerator's method.
   */
  
  
  public PrototypeSet[] doGeneration(PrototypeSet population[][], double claseObjetivo, int bestIndividual[])
  {
	  //Algorithm

	  PrototypeSet nominalPopulation;

	  PrototypeSet mutation[] = new PrototypeSet[PopulationSize];
	  PrototypeSet crossover[] = new PrototypeSet[PopulationSize];
	    
	  double ScalingFactor[] = new double[this.PopulationSize];
	  double CrossOverRate[] = new double[this.PopulationSize]; // Inside of the Optimization process.
	  double fitness[] = new double[PopulationSize];
	  
	  
	  // Calculate fitness function for each particle
  
	  for(int i=0; i< PopulationSize; i++){
		  fitness[i] =fitnessFunction(population, bestIndividual ,claseObjetivo,i); 
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
		   if(population[(int) claseObjetivo][0]!=null){
		   
			   for(int i=0; i<population[(int) claseObjetivo][j].size(); ++i)
				   population[(int) claseObjetivo][j].get(i).setIndex(i);
		   }
	   }
      
	   boolean cruceExp [] = new boolean[PopulationSize];
	   
	   
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
				   crossover[i] = SFGSS(population, bestIndividual, (int) claseObjetivo, i, bestFitnessIndex, CrossOverRate[i]);
				   
				   
			   }else if(i==bestFitnessIndex &&  tau[2] <= randj[4] && randj[4] < tau[3]){
				   //SFHC
				   //System.out.println("SFHC applied");
				   crossover[i] = SFHC(population, bestIndividual, (int) claseObjetivo, i, bestFitnessIndex, ScalingFactor[i], CrossOverRate[i]);
				   
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
				   
				   mutation[i] = new PrototypeSet(population[(int) claseObjetivo][i].size());
			   
				  //Mutation:
					
				   mutation[i]  = mutant(population[(int) claseObjetivo], i, bestFitnessIndex, ScalingFactor[i]);
				   
				    // Crossver Operation.

				   crossover[i] = new PrototypeSet(population[(int) claseObjetivo][i]);
				   
				   for(int j=0; j< population[(int) claseObjetivo][i].size(); j++){ // For each part of the solution
					   
					   double randNumber = RandomGenerator.Randdouble(0, 1);
						   
					   if(randNumber<CrossOverRate[i]){
						   crossover[i].set(j, mutation[i].get(j)); // Overwrite.
					   }
				   }
				   
				   
				   
				   
			   }
			   
   
			   
			   // Fourth: Selection Operation.
		   
			   nominalPopulation = new PrototypeSet();
		       nominalPopulation.formatear(population[(int) claseObjetivo][i]);
		      // fitness[i] = accuracy(nominalPopulation,trainingDataSet.getFromClass(claseObjetivo));
		       
		       PrototypeSet guardaPopulation  = new PrototypeSet(population[(int) claseObjetivo][i]);
		      
		       population[(int) claseObjetivo][i] = new PrototypeSet(nominalPopulation);
		       
		       fitness[i] = fitnessFunction(population, bestIndividual ,claseObjetivo,i); 
		       
		       nominalPopulation = new PrototypeSet();
		       nominalPopulation.formatear(crossover[i]);
		       
		       population[(int) claseObjetivo][i] = new PrototypeSet(nominalPopulation);
		       
			   double trialVector = fitnessFunction(population, bestIndividual ,claseObjetivo,i); //accuracy(nominalPopulation,trainingDataSet.getFromClass(claseObjetivo));
			
			   
			   
		  
			  if(trialVector > fitness[i]){
				  population[(int) claseObjetivo][i] = new PrototypeSet(crossover[i]);
				  fitness[i] = trialVector;
			  }else{
				  population[(int) claseObjetivo][i] = new PrototypeSet(guardaPopulation); // restitutyo
			  }
			  
			  /*
			    if(fitness[i]>bestFitness){
				  bestFitness = fitness[i];
				  bestFitnessIndex=i;
				  //System.out.println("Iter="+ iter +" Acc= "+ bestFitness);
			  }
			  */
			  
			  
		   }

		   //System.out.println("Acc= "+ bestFitness);
	   }

	   
		 //  nominalPopulation = new PrototypeSet();
          // nominalPopulation.formatear(population[bestFitnessIndex]);
		//  System.err.println("\n% de acierto en training Nominal " + KNN.classficationAccuracy(nominalPopulation,trainingDataSet,1)*100./trainingDataSet.size() );
			  
			//  nominalPopulation.print();

  
		return population[(int) claseObjetivo];
  }
  
  /**
   * General main for all the prototoype generators
   * Arguments:
   * 0: Filename with the training data set to be condensed.
   * 1: Filename which contains the test data set.
   * 3: Seed of the random number generator.            Always.
   * **************************
   * 4: .Number of neighbors
   * 5:  Swarm Size
   * 6:  Particle Size
   * 7:  Max Iter
   * 8:  C1
   * 9: c2
   * 10: vmax
   * 11: wstart
   * 12: wend
   * @param args Arguments of the main function.
 * @throws Exception 
   */
  public static void main(String[] args) throws Exception
  {
      Parameters.setUse("CoDE", "<seed> <Number of neighbors>\n<Swarm size>\n<Particle Size>\n<MaxIter>\n<DistanceFunction>");        
      Parameters.assertBasicArgs(args);
      
      PrototypeSet training = PrototypeGenerationAlgorithm.readPrototypeSet(args[0]);
      PrototypeSet test = PrototypeGenerationAlgorithm.readPrototypeSet(args[1]);
      
      
      long seed = Parameters.assertExtendedArgAsInt(args,2,"seed",0,Long.MAX_VALUE);
      CoDEGenerator.setSeed(seed);
      
      int k = Parameters.assertExtendedArgAsInt(args,3,"number of neighbors", 1, Integer.MAX_VALUE);
      int swarm = Parameters.assertExtendedArgAsInt(args,4,"swarm size", 1, Integer.MAX_VALUE);
      int particle = Parameters.assertExtendedArgAsInt(args,5,"particle size", 1, Integer.MAX_VALUE);
      int iter = Parameters.assertExtendedArgAsInt(args,6,"max iter", 1, Integer.MAX_VALUE);
      double c1 = Parameters.assertExtendedArgAsInt(args,7,"c1", 1, Double.MAX_VALUE);
      double c2 =Parameters.assertExtendedArgAsInt(args,8,"c2", 1, Double.MAX_VALUE);
      double vmax =Parameters.assertExtendedArgAsInt(args,9,"vmax", 1, Double.MAX_VALUE);
      double wstart = Parameters.assertExtendedArgAsInt(args,10,"wstart", 1, Double.MAX_VALUE);
      double wend =Parameters.assertExtendedArgAsInt(args,11,"wend", 1, Double.MAX_VALUE);
      
      //String[] parametersOfInitialReduction = Arrays.copyOfRange(args, 4, args.length);
     //System.out.print(" swarm ="+swarm+"\n");
      
      
      CoDEGenerator generator = new CoDEGenerator(training, k,swarm,particle,iter, 0.5,0.5,1);
      
  	  
      PrototypeSet resultingSet = generator.execute();
      
  	//resultingSet.save(args[1]);
      //int accuracyKNN = KNN.classficationAccuracy(resultingSet, test, k);
      int accuracy1NN = KNN.classficationAccuracy(resultingSet, test);
      generator.showResultsOfAccuracy(Parameters.getFileName(), accuracy1NN, test);
  }

}
