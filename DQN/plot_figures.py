import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_plot():

    df = pd.read_csv('training_log.csv')
    plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
    plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
             label='AverageScore')
    plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
             label='Solved Requirement')
    plt.legend()
    plt.savefig('training_log.png')
def test_plot():

    df = pd.read_csv('testing_log.csv')
    plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
    plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
             label='AverageScore')
    plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
             label='Solved Requirement')
    plt.legend()
    plt.savefig('testing_log.png')

