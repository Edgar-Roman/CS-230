import os
import pandas as pd


class Accalerometer_Preprocessing:
    def __init__(self, acc_dir_path, tab_dir_path=None, mode=0):

        # Mode 0 means we are doing aggregations across different time steps
        if mode == 0:
            self.makeAggregations(acc_dir_path)

        # Mode 1 means we are doing preprocessing for training LSTM
        else:
            self.constructAccelerometerData(acc_dir_path, tab_dir_path)

    def constructAccelerometerData(self, acc_dir_path, tab_dir_path):
        tabular = pd.read_csv(tab_dir_path)  # Datafram holds the tabular data
        counter = 0
        for subdir, dirs, files in os.walk(acc_dir_path):
            # Goin through all the dirctories in the main directory
            for file in files:

                currentUser = pd.read_parquet(os.path.join(subdir, file), engine='fastparquet')
                currentUser = self.dropColumns(currentUser)
                currentUser = self.dropRows(currentUser)
                if len(currentUser) < 10000:  # Taking only the samples with more than 10000 steps
                    continue

                if counter == 0:
                    self.acc_df = currentUser.head(10000)
                    self.tab_df = tabular[tabular['id'] == os.path.join(subdir, file)[84:92]]  # Extracting the id from the name of the folder

                else:
                    self.acc_df = pd.concat([self.acc_df, currentUser.head(10000)], axis=0)
                    self.tab_df = pd.concat(
                        [self.tab_df, tabular[tabular['id'] == os.path.join(subdir, file)[84:92]]], axis=0) # Extracting the id from the name of the folder

                if counter % 10 == 0:
                    print(f"Iteration: {counter}")

                counter += 1

    def makeAggregations(self, dir_path):
        counter = 0
        for subdir, dirs, files in os.walk(dir_path):
            # Goin through all the dirctories in the main directory
            for file in files:

                currentUser = pd.read_parquet(os.path.join(subdir, file), engine='fastparquet')
                currentUser = self.dropColumns(currentUser)
                currentUser = self.dropRows(currentUser)

                currentUser = pd.DataFrame(currentUser.mean(axis=0)).T
                currentUser['id'] = os.path.join(subdir, file)[51:59]  # Extracting the id from the name of the folder

                if counter == 0:
                    self.aggregations = currentUser

                else:
                    self.aggregations = pd.concat([self.aggregations, currentUser], axis=0)

                if counter % 10 == 0:
                    print(f"Iteration: {counter}")

                counter += 1

    def dropColumns(self, currentUser):
        # Drop nonnumerical columns
        currentUser = currentUser.drop(
            columns=['step', 'battery_voltage', 'time_of_day', 'weekday', 'quarter', 'relative_date_PCIAT'],
            errors='ignore')
        return currentUser

    def dropRows(self, currentUser):
        # Drop the rows in which the user was not wearing the watch
        currentUser = currentUser[currentUser['non-wear_flag'] != 1]
        currentUser = currentUser.drop(columns=['non-wear_flag'], errors='ignore')
        return currentUser
