# gopigo
GoPiGo self-driven car project 


For this project, 

1. first run: datacollection.py
  -this gives you data in the Data_sheet folder
  (Data_1, Data_2, Data_3, Data)
  
  The folders mentioned above are the sample rounds and the "Data_sheet" folder is the one we used to train the model.

2. We used Data_sheet as input to be fed into "model.py". And this complete the training model. 

3. Then we do test round and take pictures. We save the pictures to the result folder through inputCollect.py.

4. Then we feed the result from inputCollect.py into the Main_Test_Car.py, which contains the model from model.py. Then, we make the prediction. We save the output as result.csv.

5. Main_Test_Car.py reads in the result from the model and drives the car.

6. We complete our model analysis in Jupyter Notebook (graphs) and save the result in gopigo.ipynb_2.json.

7. easyCarMotorControl.py lets you drive GoPiGo around.


