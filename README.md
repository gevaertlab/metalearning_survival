# metalearning_survival


Project name: A meta-learning approach for genomic survival analysis  

Project home page: https://github.com/gevaertlab/metalearning_survival.  
Operating system(s): Platform independent  
Programming language: Python    
Other requirements: Python 3.6.6 or higher, Pytorch 0.4.1  
License: BSD 3-Clause License   

***Example usage***

**Direct learning**

python direct_learning_subsettarget.py --config 'direct_learning_train_config.json'   
python direct_learning_eval.py --config 'direct_learning_eval_config.json'   


**Combined learning**

python combined_learning.py  --config 'combined_learning_train_config.json'   
python neuralnet_eval.py --config 'combined_eval_config.json'   


**Regular pre-train fine-tune**

python pretrain_coxnet.py --config  'pretrain_coxnet_config.json'   
python finetune.py --config 'fintune_config.json'   
python neuralnet_eval.py --config 'finetune_eval_config.json'   


**few-shot meta-learning**
python fewshot_metatrain.py --config 'fewshot_meta_config.json'   
python fewshot_finaltrain_eval.py --config 'fewshot_finaltrain_config.json'   

