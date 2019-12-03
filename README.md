# ECEN 649 Course Project - Viola-Jones Algorithm

## Dependencies:
- Python 3.7
- glob3 0.0.1
  - Reading directories and loading data
- numpy 1.17.4
- matplotlib 3.1.2
- PIL 1.1.6
- tqdm 4.39.0
  - Visualizing training progress

## Steps to run the model

1. Clone the repository :
    ```
    git clone https://github.com/ChiehFu/649_viola_jone
    ```
2. Run and test the model

    Arguments setting:
    ```
    parser.add_argument('-T', help='# of rounds', type=int, default=10)
    parser.add_argument('-criterion', help='Criterion for model optimization', type=str, default='err', choices=['err', 'fpr', 'fnr'])
    parser.add_argument('-load_feat', help='Load features file', type=str, default='')
    parser.add_argument('-width', help='Maximal width of feature', type=int, default=8)
    parser.add_argument('-height', elp='Maximal height of feature',type=int, default=8)
    ```
    Commands to run:
    ```
    # Train the model with all default setting 
    python main.py 

    # Train the model with false positive rate as criterion for weak classifier selection for 5 rounds
    python main.py -T 5 -criterion 'fpr'

    # Train the model with false negative rate as criterion for weak classifier selection for 5 rounds
    python main.py -T 5 -criterion 'fnr'
    ```
3. Visualization 
   
   To better understand the classifiers selected by the model check out the jupiter notebook `model_viz.ipynb`, which load the saved model includes the features selected for each rounds and the details.