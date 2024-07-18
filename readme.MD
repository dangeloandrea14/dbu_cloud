### DBU-CLOUD implementation


main.py and dbu_cloud.py contain the necessary code to launch a dummy experiment on the provided files. The documents must be in that format for the experiments to work.
These snippets of code allow for a sample execution of DbU_Cloud.  

To run them, execute on the terminal:

```
python main.py --k 1 --transformer stateoftheart --dataset CISIstop --no-sentence_wide --file ./transformed_datasets/CISIstop/document_wide/stateoftheart/transformed29_56.csv --searching
```

You can test it with different values of k.

The folder in results/CISIstop/document_wide/stateoftheart/k=1 will be populated with a folder for each query, presenting the DbUScore of each document between 29 and 56 for that query. 


