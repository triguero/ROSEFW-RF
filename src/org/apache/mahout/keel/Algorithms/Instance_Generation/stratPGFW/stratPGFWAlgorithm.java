/**
	stratPGFW.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.stratPGFW;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * stratPGFW algorithm calling.
 * @author Isaac Triguero
 */
public class stratPGFWAlgorithm extends PrototypeGenerationAlgorithm<stratPGFWGenerator>
{
    /**
     * Builds a new IPADE algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected stratPGFWGenerator buildNewPrototypeGenerator(PrototypeSet train, PrototypeSet test, Parameters params)
    {
       return new stratPGFWGenerator(train, test,params);    
    }
    
     /**
     * Main method. Executes stratPGFW algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        stratPGFWAlgorithm isaak = new stratPGFWAlgorithm();
        isaak.execute(args); // execute2 ?  // EXECUTE THAT RETURN THE CLASSIFICATION DONE IN TRAIN AND TEST
    }

	@Override
	protected stratPGFWGenerator buildNewPrototypeGenerator(PrototypeSet train,
			Parameters params) {
		// TODO Auto-generated method stub
		return null;
	}


}
