## TOSEM'22 CBS

### Preparation
- For Windows, run the following cmd to add python to sys.path and check
```cmd
set PYTHONPATH=%cd%
echo %PYTHONPATH%
```
- While for Linux, run cmd below 
```bash
export PYTHONPATH=$(pwd)
```
- Dependencies
```cmd
jdk >= 1.8 
nltk == 3.6.5
scikit-learn == 0.24.2
lightgbm == 3.3.2
nlg-eval == 2.3.0
tqdm == 4.62.3 
```
### Quick Start
- Output CCI samples' category prediction labels with the original data.
```cmd
cd source
python HebCup.py ../Dataset/train_clean.jsonl 1 true
python HebCup.py ../Dataset/valid_clean.jsonl 1 true
python HebCup.py ../Dataset/test_clean.jsonl 1 true
```
*This step will output the corresponding three dataset with labels (e.g., train_clean_1.jsonl) for subsequent steps.
- Discuss base classifiers on 10-fold cross-validation. Select one of the below choices to run. You can also add other classifiers by yourself to participate the comparison.
```cmd
cd Ours
python eval_clfs.py lightGBM | naive_bayes | mlp | lstm | cnn
```
- Train and evaluation for category classifiers
```cmd
cd Ours
python classification.py lightGBM <suffix>
```
*This process will output two files (HCUP_test\<suffix\>.jsonl and CUP_test\<suffix\>.jsonl) that are used for the following two steps.
- Evaluate the performance of HebCup Side  
```cmd
cd source
python HebCup.py ../Dataset/splitted_test_sets/lightGBM/HCUP_test<suffix>.jsonl 1 false
```  
Please refer to the original repo of [HebCUP](https://github.com/Ringbo/HebCup) for detailed doc of HebCUP.
- Evaluate the performance of CUP side  
To do this, you can send the CUP_test\<suffix\>.jsonl to the repo of CUP to evaluate the performance of CUP side with the metrics in this repo. Please refer to the original repo of [CUP](https://github.com/Tbabm/CUP) for detailed doc of CUP.

### Welcome to Cite!  
- If you find this paper useful and would like to cite it, the following would be appropriate:
```
@article{yang2022significance,
  title={On the Significance of Category Prediction for Code-Comment Synchronization},
  author={Yang, Zhen and Keung, Jacky Wai and Yu, Xiao and Xiao, Yan and Jin, Zhi and Zhang, Jingyu},
  journal={ACM Transactions on Software Engineering and Methodology},
  year={2022},
  publisher={ACM New York, NY}
}
