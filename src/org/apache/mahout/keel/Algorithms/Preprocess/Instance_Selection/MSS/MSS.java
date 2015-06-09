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

//
//  MSS.java
//
//  Salvador Garc�a L�pez
//
//  Created by Salvador Garc�a L�pez 25-11-2005.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.MSS;

import org.apache.mahout.keel.Algorithms.Preprocess.Basic.*;

import org.core.*;
import java.util.StringTokenizer;

public class MSS extends Metodo {

  public MSS (String ficheroScript) {
    super (ficheroScript);
  }

  public void ejecutar () {

    int i, j, k, l;
    int nClases;
    boolean marcas[];
    boolean disponible[];
    int nSel;
    double conjS[][];
    double conjR[][];
    int conjN[][];
    boolean conjM[][];
    int clasesS[];
    double distEnemy[];
    double distancia, minDistancia;
    int pos;

    long tiempo = System.currentTimeMillis();

    /*Inicialization of the flagged instances vector for a posterior copy*/
    marcas = new boolean[datosTrain.length];
    disponible = new boolean[datosTrain.length];
    distEnemy = new double[datosTrain.length];
    for (i=0; i<datosTrain.length; i++) {
      marcas[i] = false;
      disponible[i] = true;
    }
    nSel = 0;

    /*Getting the number of different classes*/
    nClases = 0;
    for (i=0; i<clasesTrain.length; i++)
      if (clasesTrain[i] > nClases)
        nClases = clasesTrain[i];
    nClases++;

    /*Body of the algorithm. Order the instances by the mininum distance to the nearest enemy and include in the
     MSS subset if it is a representative example of the relative neighbour of a instance*/
    for (i=0; i<datosTrain.length; i++) {
      minDistancia = Double.POSITIVE_INFINITY;
      for (j=0; j<datosTrain.length; j++) {
        if (i != j && clasesTrain[i] != clasesTrain[j]) {
          distancia = KNN.distancia(datosTrain[i], realTrain[i], nominalTrain[i], nulosTrain[i], datosTrain[j], realTrain[j], nominalTrain[j], nulosTrain[j], distanceEu);
          if (distancia < minDistancia)
            minDistancia = distancia;
        }
      }
      distEnemy[i] = minDistancia;
    }
    for (i=0; i<nClases; i++) {
      pos = 0;
      while (pos >= 0) {
        minDistancia = Double.POSITIVE_INFINITY;
        pos = -1;
        for (j = 0; j < datosTrain.length; j++) {
          if (clasesTrain[j] == i && disponible[j]) {
            if (distEnemy[j] < minDistancia) {
              minDistancia = distEnemy[j];
              pos = j; //pos is the instance with minimun distance of the nearest enemy
            }
          }
        }
        if (pos >= 0) {
          marcas[pos] = true;
          disponible[pos] = false;
          for (k = 0; k < datosTrain.length; k++) {
            if (clasesTrain[k] == i) {
              if (disponible[k] &&
                  KNN.distancia(datosTrain[pos], realTrain[pos], nominalTrain[pos], nulosTrain[pos], datosTrain[k], realTrain[k], nominalTrain[k], nulosTrain[k], distanceEu) < distEnemy[pos]) {
                disponible[k] = false;
              }
            }
          }
        }
      }
    }

    /*Building of the S set from the flags*/
    nSel = 0;
    for (i=0; i<datosTrain.length; i++)
      if (marcas[i]) nSel++;
    conjS = new double[nSel][datosTrain[0].length];
    conjR = new double[nSel][datosTrain[0].length];
    conjN = new int[nSel][datosTrain[0].length];
    conjM = new boolean[nSel][datosTrain[0].length];
    clasesS = new int[nSel];
    for (i=0, l=0; i<datosTrain.length; i++) {
      if (marcas[i]) { //the instance will be copied to the solution
        for (j=0; j<datosTrain[0].length; j++) {
          conjS[l][j] = datosTrain[i][j];
          conjR[l][j] = realTrain[i][j];
          conjN[l][j] = nominalTrain[i][j];
          conjM[l][j] = nulosTrain[i][j];
        }
        clasesS[l] = clasesTrain[i];
        l++;
      }
    }

    System.out.println("MSS "+ relation + " " + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

    OutputIS.escribeSalida(ficheroSalida[0], conjR, conjN, conjM, clasesS, entradas, salida, nEntradas, relation);
    OutputIS.escribeSalida(ficheroSalida[1], test, entradas, salida, nEntradas, relation);
  }

  public void leerConfiguracion (String ficheroScript) {

    String fichero, linea, token;
    StringTokenizer lineasFichero, tokens;
    byte line[];
    int i, j;

    ficheroSalida = new String[2];

    fichero = Fichero.leeFichero (ficheroScript);
    lineasFichero = new StringTokenizer (fichero,"\n\r");

    lineasFichero.nextToken();
    linea = lineasFichero.nextToken();

    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();

    /*Getting the names of the training and test files*/
    line = token.getBytes();
    for (i=0; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroTraining = new String (line,i,j-i);
    for (i=j+1; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroTest = new String (line,i,j-i);

    /*Getting the path and base name of the results files*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();

    /*Getting the names of output files*/
    line = token.getBytes();
    for (i=0; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroSalida[0] = new String (line,i,j-i);
    for (i=j+1; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroSalida[1] = new String (line,i,j-i);

    /*Getting the type of distance function*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    distanceEu = tokens.nextToken().substring(1).equalsIgnoreCase("Euclidean")?true:false;    
}

}

