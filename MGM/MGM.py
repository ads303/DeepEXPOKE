import os
import numpy as np
import pandas as pd

# Import module
import jpype;
# Enable Java imports
import jpype.imports;
# Pull in types
from jpype.types import *;

class MGM:
    
    
    def __init__(self):
        dir_path = os.getcwd();
        #jar path
        jar_path = os.path.join(dir_path, 'MGM'+os.path.sep+'tetradLite_likelihood_for_all.jar');
        
        #print(jpype.getDefaultJVMPath());#Test
        #try:
        #    import edu.pitt.csb.mgm;
        #    print("catch.")
        #except ImportError:
        #    print("ImportError.");
       

        
        #Add the jar to Java class path
        jpype.addClassPath("/Users/adsriram98/Documents/PARK_LAB/DAG-deepVASE-AS-final-main/MGM/tetradLite_likelihood_for_all.jar");
        print(jar_path);
        
        # Launch the JVM
        jpype.startJVM();

        
        
        
    def runMGM(self, data_folder_path, XY_file_name, lambda_continuous_continuous = 0.3, lamda_continuous_discrete = 0.3, lamda_discrete_discrete = 0.3):
        '''
        Run MGM
        Parameters:
            data_folder_path: the directory at where the input data is located.
            XY_file_name: the input data.
            lambda_continuous_continuous: the panalty value 'lamda' set for the associations whose two variables are continuous.
            lamda_continuous_discrete: the panalty value 'lamda' set for the associations whose one variable is continuous and the other is discrete.
            lamda_discrete_discrete: the panalty value 'lamda' set for the associations whose two variables are discrete.
    
        Return:
        mgm_output_file: a file that contains all the selected associations.
        '''
        
        #Import MGM classes
        jpype.JClass("edu.cmu.tetrad.data.DataSet")
        jpype.JClass("edu.cmu.tetrad.graph.Graph")
        jpype.JClass("edu.pitt.csb.mgm.ConvexProximal")
        jpype.JClass("edu.pitt.csb.mgm.IndTestMultinomialAJ")
        jpype.JClass("edu.pitt.csb.mgm.MGM")
        jpype.JClass("edu.pitt.csb.mgm.MGM$MGMParams")
        jpype.JClass("edu.pitt.csb.mgm.MixedUtils")
        jpype.JClass("edu.pitt.csb.mgm.ProximalGradient")
        jpype.JClass("edu.pitt.csb.mgm.STEPS")

        
        # Create MixedUtils object
        mixedUtils = jpype.JClass("edu.pitt.csb.mgm.MixedUtils");
        ds = mixedUtils.loadDataSet(data_folder_path,XY_file_name);

        lambda_continuous_continuous = 0.3;
        lamda_continuous_discrete = 0.3;
        lamda_discrete_discrete = 0.3;

        lamda_array = np.array([lambda_continuous_continuous,lamda_continuous_discrete,lamda_discrete_discrete]);

        # Create and initialize MGM object
        mgm = jpype.JClass("edu.pitt.csb.mgm.MGM");
        mgm = mgm(ds,lamda_array);

        # Create MGM Graph object and convert to String
        mgm_graph = mgm.search();
        
        likelihood_vals_output_name = XY_file_name.replace(".txt", "_likelihood_vals.txt");
        likelihood_vals_output_path = data_folder_path+os.path.sep+likelihood_vals_output_name;
        mgm.saveLikelihoodVals(likelihood_vals_output_path);
        
        mgm_output = mgm_graph.toString();
        py_output = str(mgm_output);
        #print(py_output);
        
        output_content = py_output.split("\nGraph Edges:")[1];
        #Create a temporary file
        text_file = open("output_content.txt", "w");
        n = text_file.write(output_content);
        text_file.close();

        #Save the associations to a "csv" file
        associations = pd.read_csv('output_content.txt',sep=" ",names=["1","Feature1","2","Feature2"]);
        associations = associations[["Feature1","Feature2"]];
        
        associations_output_name = XY_file_name.replace(".txt", "_MGM_associations.csv");
        
        associations.to_csv(data_folder_path+os.path.sep+associations_output_name);
        #Remove the temporary file.
        os.remove("output_content.txt");
        
        #Shut down the JVM
        jpype.shutdownJVM();
        
        return associations_output_name,likelihood_vals_output_path;


        
