
# Run the code

### Dependencies
* Python 3.11
* Run pip install -r requirements.txt

### Instructions
1. Datasets:
     Question based context data for instruction tuning: [here](https://drive.google.com/drive/folders/1BoZdJDphNPq6Ft-JPx5AoSiWogg0gvnx?usp=sharing)
   
     Raw ECTSum Data: [here](https://github.com/rajdeep345/ECTSum)
3. For Extractive Stage:

   Get the datsets required for extractive phase from [here](https://drive.google.com/drive/folders/1M3ks2kjkkeyhl1OICb9OciD2JaouJCaI?usp=drive_link)
   
    To generate the questions list:
    ```
    python questions_generation_and_sorting.py
    ```
    To apply topic modelling on generated questions:
    ```
    python topic_modelling.py
    ```
    To generate the extractive summary for train set:
    ```
    python train_extractive_creation.py
    ```
    To generate the extractive summary for test set:
    ```
    python test_extractive_generation.py
    ```   

4. For Abstrative Phase.
   To train the model please run the following code. It will produce predicted BPS for test files.
    ```
    python ectsum_finetune_flant5_question_based_context.py
    ```  


