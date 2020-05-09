### Use simulated anneal to solve TSP problem
### Require
 * python>=3.5
 * numpy
 * matplotlib
 ###Useage
  To run the code, simply excute:  
  `python main.py`  
  The best path will be saved in city.txt. you can use test.exe to verify the solution.  
  For more information about the configurations, run the scripts with `--help` flag  
  `python main.py --help`  
  
  There are five experiments to study how parameters to influence the final performance.  
```
python test_initT.py
python test_annealMode.py
python test_gamma.py
python test_L.py
python test_mode.py
python test_seed.py
```
The experiment results will be saved in the directory *experiment*. 

###Report
Because this is a task, there is also a detailed report. I will upload it soon.