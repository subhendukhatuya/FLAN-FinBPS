
# Run the code

### Dependencies
* Python 3.11
* Run pip install -r requirements.txt

### Instructions
1. Datasets:
     Question based context data for instruction tuning: [here](https://drive.google.com/drive/folders/1BoZdJDphNPq6Ft-JPx5AoSiWogg0gvnx?usp=sharing)
     Raw ECTSum Data: [here](https://github.com/rajdeep345/ECTSum)
2. For extractive stage:


    To generate the questions list:
    ```
    !git clone https://github.com/patil-suraj/question_generation.git
    %cd question_generation
    python questions_generation_and_sorting.py
    ```
    To apply topic modelling on generated questions:
    ```
    python topic_modelling.py
    ```
    To generate the extractive summary for train set:
    ```
    python train_extractive_creaion.py
    ```
    To generate the extractive summary for test set:
    ```
    python test_extractive_generation.py
    ```   

4. **python ectsum_finetune_flant5_question_based_context.py** to train the model. It will produce predicted BPS for test files.

### Note



