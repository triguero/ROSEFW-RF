/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S�nchez (luciano@uniovi.es)
    J. Alcal�-Fdez (jalcala@decsai.ugr.es)
    S. Garc�a (sglopez@ujaen.es)
    A. Fern�ndez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

/*
 * CHCBinaryLVO.java
 *
 * Created on 18 de septiembre de 2005, 13:44
 * 
 */

package org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.evolutionary_algorithms.CHC.wrapper;

import java.io.IOException;
import java.util.*;

import org.core.*;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.*;
import org.apache.mahout.keel.Algorithms.Preprocess.Feature_Selection.evolutionary_algorithms.*;
import org.apache.hadoop.mapreduce.Mapper.Context;


/** MAIN CLASS OF THE CHC FEATURE SELECTION ALGORITHM
 *
 *  @author Manuel Chica Serrano
 * 
 *  CHC Algorithm uses binary representation for feature selection (lvo wrapper). 
 *  Inputs: seed, number of nearest neighbours used in KNN Classifier, population length, 
 *          number of evaluations, divergence ratio, alfa value for fitness function
 *  Representation: Binary
 *  Elitism
 *  HUX Crossover and avoiding incest
 *  Restart using best chromosome as template
 *  Fitness: (1-alfa)*precision + alfa*features_selected
 *  Stopping criteria: number of evaluations */
public class CHCBinaryLVO {
    
    private Context context;
    
	/** Datos class with all information about datasets and feature selection methods  */
    private Datos data;
    
    
    /** needed parameters for CHC method */
    private Parametros params;
        
    
    /** population  */
    private Cromosoma poblacion[];
    
        
    /** best chromosome of the population */
    private Cromosoma mejorIndividuo;
    
    
    /** evaluation in which the best individual is obtained */
    private int nEvalMejorIndividuo;
    
    
    boolean validacionCruzada=false;

    
    /** interior class using for reading all parameters */
    private class Parametros{
    
        /** algorithm name */
        String nameAlgorithm;
        
        
        /** number of nearest neighbours for KNN Classifier */
        int paramKNN;
            
        
        /** pathname of training dataset */
        String trainFileNameInput;
    
    
        /** pathname of test dataset */
        String testFileNameInput;
    
    
        /** pathname of test dataset only with selected features */
        String testFileNameOutput;
    
    
        /** pathname of training dataset only with selected features */
        String trainFileNameOutput;
    
    
        /** pathname of an extra file with additional information about the algorithm results */
        String extraFileNameOutput;
       
                
        /** divergence ratio used in diverge operator of CHC method */
        double divergenceRatio;
        
        
        /** weight used in the fitness function (between precision & selected features)*/
        double alfa;
        
        
        /** the  length of population*/
        int tamPoblacion;
        
        
        /** seed for pseudo-random generator */
        long seed;
        
        
        /** number of evaluations */
        int numEvaluaciones;    
        
              
        Parametros(){
        	
        }
        
        /* ----------- METHODS ------------ */
        /** Parametros Class constructor 
         *  @param nombreFicheroParametros is the pathname of input parameter file */ 
        Parametros (String nombreFicheroParametros){
        
            try{
                int i;
                String fichero, linea, tok;
                StringTokenizer lineasFichero, tokens;

                /* read the parameter file using Fichero class */
                fichero = Fichero.leeFichero(nombreFicheroParametros);
                fichero += "\n";
                
                /* remove all \r characters. it is neccesary for a correst use in Windows and UNIX  */
                fichero = fichero.replace('\r', ' ');

                /* extract the differents tokens of the file */
                lineasFichero = new StringTokenizer(fichero, "\n");

                i=0;
                while(lineasFichero.hasMoreTokens()) {
                    
                    linea = lineasFichero.nextToken();    
                    i++;
                    tokens = new StringTokenizer(linea, " ,\t");
                    if(tokens.hasMoreTokens()){

                        tok = tokens.nextToken();
                        if(tok.equalsIgnoreCase("algorithm")) nameAlgorithm = getParamString(tokens);
                        else if(tok.equalsIgnoreCase("inputdata")) getInputFiles(tokens);
                        else if(tok.equalsIgnoreCase("outputdata")) getOutputFiles(tokens);
                        else if(tok.equalsIgnoreCase("paramKNN")) paramKNN = getParamInt(tokens);      
                        else if(tok.equalsIgnoreCase("divergenceRatio")) divergenceRatio = getParamDouble(tokens);
                        else if(tok.equalsIgnoreCase("seed")) seed = getParamInt(tokens);  
                        else if(tok.equalsIgnoreCase("nEval")) numEvaluaciones = getParamInt(tokens);  
                        else if(tok.equalsIgnoreCase("alfa")) alfa = getParamDouble(tokens);  
                        else if(tok.equalsIgnoreCase("popLength")) tamPoblacion = getParamInt(tokens);
                        else throw new java.io.IOException("Syntax error on line " + i + ": [" + tok + "]\n");
                    }                                                      

                }


            } catch(java.io.FileNotFoundException e){
                System.err.println(e + "Parameter file");
            }catch(java.io.IOException e){
                System.err.println(e + "Aborting program");
                System.exit(-1);
            }

            /** show the read parameter in the standard output */
            String contents = "-- Parameters echo --- \n";
            contents += "Algorithm name: " + nameAlgorithm +"\n";
            contents += "Input Train File: " + trainFileNameInput +"\n";
            contents += "Input Test File: " + testFileNameInput +"\n";
            contents += "Output Train File: " + trainFileNameOutput +"\n";
            contents += "Output Test File: " + testFileNameOutput +"\n";
            contents += "Parameter k of KNN Algorithm: " + paramKNN + "\n";
            contents += "Divergence Ratio. : " + divergenceRatio + "\n";
            contents += "Alfa: " + alfa + "\n";
            contents += "Population: " + tamPoblacion + "\n";
            contents += "Number of Evals: " + numEvaluaciones + "\n";
            contents += "Seed: " + seed + "\n";            
            System.out.println(contents);

        }
    
    
        /** obtain an integer value from the parameter file  
            @param s is the StringTokenizer  */
        private int getParamInt(StringTokenizer s){
            String val = s.nextToken();
            val = s.nextToken();
            return Integer.parseInt(val);           
        }
        
        
        /** obtain a long value from the parameter file  
            @param s is the StringTokenizer */
        private long getParamLong(StringTokenizer s){
            String val = s.nextToken();
            val = s.nextToken();
            return Long.parseLong(val);           
        }
        
    
        /** obtain a double value from the parameter file  
            @param s is the StringTokenizer */
        private double getParamDouble(StringTokenizer s){
            String val = s.nextToken();
            val = s.nextToken();
            return Double.parseDouble(val);
        }


       /** obtain a string value from the parameter file  
            @param s is the StringTokenizer */
       private String getParamString(StringTokenizer s){
            String contenido = "";
            String val = s.nextToken();
            while(s.hasMoreTokens())
                contenido += s.nextToken() + " ";

            return contenido.trim();
        }


         /**obtain the names of the input files from the parameter file  
            @param s is the StringTokenizer */
       private void getInputFiles(StringTokenizer s){
            String val = s.nextToken();

            trainFileNameInput = s.nextToken().replace('"',  ' ').trim();
            testFileNameInput = s.nextToken().replace('"',  ' ').trim();
        }



       /** obtain the names of the output files from the parameter file  
            @param s is the StringTokenizer */
       private void getOutputFiles(StringTokenizer s){
            String val = s.nextToken();

            trainFileNameOutput = s.nextToken().replace('"',  ' ').trim();
            testFileNameOutput = s.nextToken().replace('"',  ' ').trim();
            extraFileNameOutput = s.nextToken().replace('"',  ' ').trim();        

        }
    
    }
    
    /* ---- METHODS ------ */
    
    /**
     * Creates a new instance of CHCBinaryLVO 
     */
    public CHCBinaryLVO(String ficParametros) {
        
        /* load the parameter file from the Parametros class */
        params = new Parametros(ficParametros);
        
        /* set the seed parameter */
        Randomize.setSeed(params.seed);
        
        data = new Datos (params.trainFileNameInput, params.testFileNameInput, params.paramKNN);
               
        /* generate the initial population */
        poblacion = new Cromosoma[params.tamPoblacion];
        
        mejorIndividuo = new CromosomaBinario(data.returnNumFeatures());
        nEvalMejorIndividuo = -1;
        
        for(int i=0; i<params.tamPoblacion; i++)
            poblacion[i] = new CromosomaBinario(data.returnNumFeatures());
        
    }
   
    /**
     * Creates a new instance of CHCBinaryLVO 
     */
    public CHCBinaryLVO(InstanceSet train,int paramKNN, long seed, double divergenceRatio, double alfa, int numEvaluaciones, int tamPoblacion, Context context) {
        
        /* load the parameter file from the Parametros class */
        params = new Parametros();
        params.trainFileNameInput = "train1.txt";
        params.testFileNameInput = "test.txt";
        
        params.paramKNN=paramKNN;
        params.seed=seed;
        params.divergenceRatio=divergenceRatio;
        params.alfa=alfa;
        params.numEvaluaciones=numEvaluaciones;
        params.tamPoblacion=tamPoblacion;
        
        this.context=context;
        
        /* set the seed parameter */
        Randomize.setSeed(params.seed);
        
        System.out.println("Comenzando CHC");
        data = new Datos (train, params.paramKNN,context);
        
        System.out.println("CHC: datos leídos");
        /* generate the initial population */
        poblacion = new Cromosoma[params.tamPoblacion];
        
        mejorIndividuo = new CromosomaBinario(data.returnNumFeatures());
        nEvalMejorIndividuo = -1;
        
        for(int i=0; i<params.tamPoblacion; i++){
            poblacion[i] = new CromosomaBinario(data.returnNumFeatures());
            context.progress();
        }
        
        System.out.println("CHC: creados cromosomas");
    }
    
    
    /**
     * Creates a new instance of CHCBinaryLVO  con Validación cruzada con test (randomSubset)
     * @throws IOException 
     */
    public CHCBinaryLVO(InstanceSet train,InstanceSet test,int paramKNN, long seed, double divergenceRatio, double alfa, int numEvaluaciones, int tamPoblacion, Context context, String header) throws IOException {
        
    	
    	this.validacionCruzada=true;
    	
        /* load the parameter file from the Parametros class */
        params = new Parametros();
        params.trainFileNameInput = "train1.txt";
        params.testFileNameInput = "test.txt";
        
        params.paramKNN=paramKNN;
        params.seed=seed;
        params.divergenceRatio=divergenceRatio;
        params.alfa=alfa;
        params.numEvaluaciones=numEvaluaciones;
        params.tamPoblacion=tamPoblacion;
        
        this.context=context;
        
        /* set the seed parameter */
        Randomize.setSeed(params.seed);
        
        System.out.println("Comenzando CHC");
        data = new Datos (train,test, params.paramKNN,context,header);
        
        System.out.println("CHC: datos leídos");
        /* generate the initial population */
        poblacion = new Cromosoma[params.tamPoblacion];
        
        mejorIndividuo = new CromosomaBinario(data.returnNumFeatures());
        nEvalMejorIndividuo = -1;
        
        for(int i=0; i<params.tamPoblacion; i++){
            poblacion[i] = new CromosomaBinario(data.returnNumFeatures());
            context.progress();
        }
        
        System.out.println("CHC: creados cromosomas");
    }
    
    
    /** return the fitness of a chromosome
     *   @param cromosoma a chromosome
     *  @return the fitnees of a chromosome (Maximization problem) */
    private double fitness(Cromosoma cr){
        double precision;
        int numCaracSel;
        boolean fv[];
        
        context.progress();

        if(cr==null){            
            System.err.println("ERROR: Chromosome doesn't exist");
            System.exit(0); 
        }
        
        fv = cr.devolverFeaturesVector();
        
        if(this.validacionCruzada)
        	precision = data.validacionCruzada(fv);
        else
        	precision = data.LVO(fv);
        
        for(int i=numCaracSel=0; i < fv.length; i++) 
        	if(fv[i]==true) 
        		numCaracSel++;
                
        
        //System.out.println("Fitness: "+((1-params.alfa)*(1-precision) - params.alfa*((double)numCaracSel/fv.length)));
        
      //  return ((1-params.alfa)*(1-precision)-params.alfa*((double)numCaracSel/fv.length));
        return (1-precision);
        
    }
    
    
    /** replace the worst chromosomes in generation t-1 for best chromosomes in C'(t)
        @param descendientes an array with the offsprings  
        @return the number of the chromosomes replaced */
    private int reemplazar(Cromosoma descendientes[]){
        int i, mejor, peor, nuevos;
        double fitnessPeor, fitnessMejor;
        
        nuevos = 0;
        while(descendientes.length > 0) {    
        	context.progress();
            /* search the worst individual in the population */
            peor = 0;
            fitnessPeor = poblacion[0].getFitness();
           // System.out.println("Fitness población");
        	

        	
            for(i=1; i<params.tamPoblacion; i++){
            //	System.out.println(poblacion[i].getFitness());
                if(fitnessPeor > poblacion[i].getFitness()){
                    fitnessPeor = poblacion[i].getFitness();
                    peor = i;
                }
            }
            
             //System.out.println("El peor fitness es: "+fitnessPeor);
             
            /* search the best individual in the offsprings pool */
            mejor = 0;
            fitnessMejor = descendientes[0].getFitness();
            for(i=1; i<(descendientes.length - nuevos); i++){
                if(fitnessMejor < descendientes[i].getFitness()){
                    fitnessMejor = descendientes[i].getFitness();
                    mejor = i;
                }
            }
            /* compares the best individual found in the offspings pool with the worst individual in the population.
               If it isn't replaced, the algorithm finish */
            if(fitnessMejor > fitnessPeor){
                poblacion[peor] = descendientes[mejor];
                
                /* removes the individual of the offsprings pool */
                for(i=mejor; i<(descendientes.length-nuevos-1); i++)
                    descendientes[i] = descendientes[i+1];
                                    
                nuevos++;
                
            } else return nuevos;
                
        }            
        return nuevos;       
    }
    
    
    /** restart operator. It restars all population using a template. This template is the best individual found in the
        previous generation. This operator uses randomness */
    private void restart(){
    	context.progress();
        /* saves an indentical copy of the best individual */
        poblacion[0].copy(mejorIndividuo);
        
        /* the rest of population is created using the best individual as template */
        for(int i=1; i<params.tamPoblacion; i++)
            poblacion[i].initPlantilla(mejorIndividuo, params.divergenceRatio);                    
    }
    
    
    /** this methods creates the offsprings pool (variable size) to start from previous population
        @return returns an array with offsprings elements
        @param umbral is the threshold using in avoiding incest */
    private Cromosoma[] generarDescendientes(int umbral){
    	context.progress();
        Cromosoma hijo1, hijo2;
        Cromosoma a[];
        Vector v = new Vector();
        int i;
        
        i = 0;
        while(i+1 < params.tamPoblacion){
            hijo1 = new CromosomaBinario(data.returnNumFeatures());
            hijo2 = new CromosomaBinario(data.returnNumFeatures());
            
            if(poblacion[i].cruzarHUX(poblacion[i+1], hijo1, hijo2, umbral)){
                v.addElement(hijo1);
                v.addElement(hijo2);
            }
            
            i += 2;
            
        }
        
        /* we convert a vector to array for return it */
        a = new Cromosoma[v.size()];
        v.toArray(a);        
        return a;
        
    }
    
    
    /** checks if exists an empty chromosome, in other words, a chromosome that it hasn't selected any feature. 
        An empty chromosome is created randomly */
    private void comprobarCromosomasVacios(Cromosoma pobl[]){
    	context.progress();
        boolean vacio;
    
        int i = 0;
        while(i < pobl.length){ 
            vacio = true;
            for(int j=0; j<pobl[i].devolverTamCromosoma() && vacio; j++)
                if(pobl[i].devolverGen(j)!=0)
                    vacio = false;
            
            if(vacio){
                pobl[i].initRand();
                pobl[i].setFitness(-1);
            } else i++;
        }
    }
    
    
    /** main method for the CHC */
    private void CHC(){
        int nEvaluaciones, i, umbral;
        Cromosoma descendientes[];
                
        /* creates the initial population randomly */
        for(i=0; i<params.tamPoblacion-1; i++){    // ISaac: todo menos uno.
            poblacion[i].initRand();     
            context.progress();
        }
        
        for(i=0; i<poblacion[0].devolverTamCromosoma(); i++)
        	poblacion[params.tamPoblacion-1].cambiarGen(1,i);  // El último lo pongo todo a uno.
        
        		
        		
        System.out.println("CHC: Inicializado");
        
        /* check         * s if there is any empty chromosome */
        comprobarCromosomasVacios(poblacion);
            
        /* number of evaluations. Initializes to 0. incest threshold initialized to L/4 */
        nEvaluaciones = 0;
        umbral = poblacion[0].devolverTamCromosoma()/4;

        while(nEvaluaciones < params.numEvaluaciones){
        	System.out.println("CHC: eval: "+nEvaluaciones);
            context.progress();
            /* evaluates the population (only non evaluated individuals) */
            for(i=0; i<params.tamPoblacion; i++) 
                if(poblacion[i].getFitness() == -1){
                    nEvaluaciones++;
                    poblacion[i].setFitness(fitness(poblacion[i]));
                                      
                    /* checks if it is the best individual of the population. Save the number of evaluation in which 
                       the best individual is obtained */
                    if(nEvalMejorIndividuo==-1 || poblacion[i].getFitness()>mejorIndividuo.getFitness()){ 
                    	System.out.print("update mejor individuo\t");
                            mejorIndividuo.copy(poblacion[i]);
                            nEvalMejorIndividuo = nEvaluaciones;           
                    }
                } 
            context.progress();
            /* generates the next offsprings pool, C'(t) */
            descendientes = generarDescendientes(umbral);
            context.progress();
            /* checks if there is any empty chromosome */
            comprobarCromosomasVacios(descendientes);
            context.progress();
            for(i=0; i<descendientes.length; i++){
                descendientes[i].setFitness(fitness(descendientes[i]));
                if(descendientes[i].getFitness()>mejorIndividuo.getFitness()){
                	System.out.print("update mejor individuo\t"+descendientes[i].getFitness()+"\n");
                    mejorIndividuo.copy(descendientes[i]);
                    nEvalMejorIndividuo = nEvaluaciones;       
                }
                nEvaluaciones++;
            }    
            
                        
            /* replace the worst chromosomes in generation t-1 for best chromosomes in C'(t) */
            if(reemplazar(descendientes) == 0){
                umbral--;
                       
                if(umbral < 0){
                    restart();
                    umbral = (int)(params.divergenceRatio*(1-params.divergenceRatio)*poblacion[0].devolverTamCromosoma());
                }
            }  
            
        }
        
    }
        
        
    /** method inteface for CHC algorithm. */
    public void ejecutar(){
        String resultado;
        int i, numFeatures;
        Date d;
        boolean features[];
       
        d = new Date();
        resultado = "RESULTS generated at " + String.valueOf((Date)d) 
                        + " \n--------------------------------------------------\n";
        resultado += "Algorithm Name: " + params.nameAlgorithm + "\n";
       
        /* call of CHC algorithm */
        CHC();
            
        /* the results are saved to an extra file with information of the algorithm */
        resultado += "\nPARTITION Filename: "+ params.trainFileNameInput +"\n---------------\n\n";
        resultado += "Features selected: \n";
            
        features = mejorIndividuo.devolverFeaturesVector();
        for(i=numFeatures=0; i<features.length; i++) 
            if(features[i] == true){ 
                resultado += Attributes.getInputAttribute(i).getName() + " - ";
                numFeatures++;
            }   
        resultado += "\n Best individual find at " + nEvalMejorIndividuo + "evaluation. ";
        resultado += "\n\n" + String.valueOf(numFeatures) + " features of " 
                + Attributes.getInputNumAttributes() + "\n\n" ;
        
        resultado += "Error in test (using train for prediction): " 
                + String.valueOf(data.validacionCruzada(features)) + "\n";
        resultado += "Error in test (using test for prediction): " 
                + String.valueOf(data.LVOTest(features)) + "\n";
        
        resultado += "---------------\n";        
            
        System.out.println("Experiment completed successfully");
              
         /* creates the new training and test datasets only with the selected features */
        Fichero.escribeFichero(params.extraFileNameOutput, resultado);
        data.generarFicherosSalida(params.trainFileNameOutput, params.testFileNameOutput, features);
               
        
    }
    
    /** 
     * <p>
     * Return the features obtained by LVF algorithm
     * </p>
     */
    public boolean[] getFeatures(){
    	
		long tiempo = System.currentTimeMillis();
    	
    	context.progress();
        CHC();
        context.progress();
        
        boolean features[] = mejorIndividuo.devolverFeaturesVector();
        int numFeatures=0;
        for(int i=0; i<features.length; i++){ 
            if(features[i] == true){ 
                numFeatures++;
            }   
        }
        
        System.out.println("Seleccionadas "+numFeatures+", con fitness: "+mejorIndividuo.getFitness());
        
        
        
        System.out.println("Experiment completed successfully");
        
    	System.out.println("CHC time: " + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "");

        return mejorIndividuo.devolverFeaturesVector();
    }
    
}

