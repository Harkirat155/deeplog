## The code is tested at:

Tensorflow version: 1.15.0

Python 2.7.18


## All source code files are in folder: ./dp_sgd


```cd dp_sgd```  


## Training:

If below code reports error "IOError: [Errno 2] No such file or directory: '../datasets/hdfs/normalBlk_allREs_10w.logSeqs'", just repeatedly run it several times and it'll find the data file.

### Train a baseline model without differential privacy:

	python train_dp.py 


###	Train a differential privacy model with noise scale sigma, for example, 1:

	python train_dp.py --dpsgd True --noise_multiplier 1


#### Training a model will automatically create a folder under "../models/hdfs" with a filename format similar to: 
	dp_clip1.0_delta1e-05_sigma1.0_clip1.0


## Test 

(Training a DP model might take hours, you can use the pre-trained models for testing, which are available at "../models/hdfs/"):

#### After a model is trained, suppose the model name is "../models/hdfs/dp_clip1.0_delta1e-05_sigma1.0_clip1.0/epoch100_validLoss0.507_validAcc88.11_eps0.9585177585005272", to test this model and save the per-log detection results to "detection_result_of_new_model":

	Because the entire test dataset is very big, you could vary "--sampleRate" to sample a small dataset for testing, for example, 0.001/0.01/0.1/1.0.
	
	python load_and_test.py --sampleRate 1 --model ../models/hdfs/dp_clip1.0_delta1e-05_sigma1.0_clip1.0/epoch100_validLoss0.507_validAcc88.11_eps0.9585177585005272 > detection_result_of_new_model


#### For the above generated per-log detection result "detection_result_of_new_model", get the topK resutls under different threshold k:

	python calcAcc.py detection_result_of_new_model




#################################################################################

## Directories:

### ./datasets:
	
	It contains parsed HDFS log dataset used for training and testing.

### ./dp_sgd:
	
	It contains all the code.



### final_results.tar.gz:
	It could be unpacked using "tar -xzvf final_results.tar.gz" to be foler "./final_results", inside this folder there are the following result files.

#### results_*******: They are the per-log prediction result files by LSTM, _sample0.1_ or _sample1_ indicates the sample rate of the total test log, _sample1_ means all test data are used for detection.

#### calcAcc.py: It reads all result files having _sample1_, process the top-k results and produce the anomaly detection metrics under different top-k criteria, stored in file "final_full_metrics"

#### final_full_metrics: It contains all tested measurements including the ones reported in the paper.	

#### drawFig.py: It reads the full metrics from final_full_metrics, and generate two experimental figures in the paper:	fpfn.pdf and fm.pdf.

#### fpfn.pdf and fm.pdf: Figures in the paper drawn using drawFig.py.


### ./models.tar.gz
	stores previously trained models; unpack it using the command "tar -xzvf models.tar.gz", and then you could use all models inside folder "./models/hdfs/" for testing. Note that the unpacked folder could be up to 2GB.


