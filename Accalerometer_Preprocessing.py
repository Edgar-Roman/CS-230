import os
import pandas as pd

class Accalerometer_Preprocessing:
    def __init__(self, dir_path, mode=0):
      
      # Mode 0 means we are doing aggregations across different time steps
      if mode == 0:
        self.makeAggregations(dir_path)

      # Mode 1 means we are doing preprocessing for training LSTM

    def makeAggregations(self, dir_path):
      counter = 0
      for subdir, dirs, files in os.walk(dir_path):
        # Goin through all the dirctories in the main directory
        for file in files:

          currentUser = pd.read_parquet(os.path.join(subdir, file), engine='fastparquet')
          currentUser = self.dropColumns(currentUser)
          currentUser = self.dropRows(currentUser)

          currentUser = pd.DataFrame(currentUser.mean(axis=0)).T
          currentUser['id'] = os.path.join(subdir, file)[51:59]   # Extracting the id from the name of the folder

          if counter == 0:
            self.aggregations = currentUser

          else:
            self.aggregations = pd.concat([self.aggregations, currentUser], axis=0)
            
          if counter % 10 == 0:
            print(f"Iteration: {counter}")
              
          counter += 1
      
    
    def dropColumns(self, currentUser):
      # Drop nonnumerical columns
      currentUser = currentUser.drop(columns=['step', 'battery_voltage', 'time_of_day', 'weekday', 'quarter', 'relative_date_PCIAT'], errors='ignore')
      return currentUser

    def dropRows(self, currentUser):
      # Drop the rows in which the user was not wearing the watch
      currentUser = currentUser[currentUser['non-wear_flag'] != 1]
      currentUser = currentUser.drop(columns=['non-wear_flag'], errors='ignore')
      return currentUser